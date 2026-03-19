import json
import logging
import numpy as np
from pathlib import Path

# Imports locaux
from .custom_ap import CustomAffinityPropagation
from .custom_kmedoids import CustomKMedoids
from .structures import Segment, TrajectoryPoint

logger = logging.getLogger(__name__)


def load_data(folder_path, limit=None, max_files=None, min_length=5.0):
    """Charge les segments. max_files limite le nombre de fichiers JSON lus.

    Args:
        folder_path: Chemin du dossier contenant les JSON compressés.
        limit: Nombre maximum de segments à charger (None = pas de limite).
        max_files: Nombre maximum de fichiers JSON à lire (None = tous).
        min_length: Longueur minimale d'un segment pour être inclus dans le
                    clustering. Les segments plus courts (micro-mouvements,
                    arrêts) sont ignorés.
    """
    folder = Path(folder_path)

    # On recupere tous les fichiers et on les trie
    files = sorted(list(folder.glob("*.json")))

    logger.info("Recherche dans : %s", folder)
    logger.info("   -> %d fichiers disponibles au total.", len(files))

    if max_files is not None and max_files < len(files):
        files = files[:max_files]
        logger.info("   Restriction : On ne charge que les %d premiers fichiers.", max_files)

    all_segments = []
    metadata = []

    if len(files) == 0:
        return [], []

    for file_path in files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

            match_id = str(data.get("match_id", "unknown"))
            if "players" not in data:
                continue

            for player in data["players"]:
                p_id = player["player_id"]
                for idx, s in enumerate(player["segments"]):
                    unique_seg_id = f"P{p_id}_{idx}"
                    try:
                        p1 = TrajectoryPoint(
                            s["start"]["x"], s["start"]["y"], s["start"]["tick"]
                        )
                        p2 = TrajectoryPoint(
                            s["end"]["x"], s["end"]["y"], s["end"]["tick"]
                        )
                        seg = Segment(p1, p2)

                        if seg.length() > min_length:
                            all_segments.append(seg)
                            metadata.append(
                                {"match_id": match_id, "seg_id": unique_seg_id}
                            )
                    except KeyError:
                        continue

            # Limite de segments pour eviter l'explosion memoire
            if limit is not None and len(all_segments) > limit:
                logger.info(
                    "Limite de segments (%d) atteinte. On arrete le chargement.", limit
                )
                return all_segments[:limit], metadata[:limit]

    return all_segments, metadata


def compute_traclus_similarity(segments, w_perp=1.0, w_angle=1.0, w_par=1.0):
    """Calcule la matrice de similarité TRACLUS entre segments.

    Combine trois composantes de distance :
    - Perpendiculaire (position relative des segments)
    - Angulaire (différence d'orientation, pondérée par les longueurs)
    - Parallèle (décalage le long de la direction)

    Args:
        segments: Liste de Segment.
        w_perp: Poids de la composante perpendiculaire.
        w_angle: Poids de la composante angulaire.
        w_par: Poids de la composante parallèle.

    Returns:
        Matrice de similarité N×N (valeurs négatives de distance).
    """
    starts_flt = np.array(
        [(s.start.x, s.start.y) for s in segments], dtype=np.float32
    )
    ends_flt = np.array(
        [(s.end.x, s.end.y) for s in segments], dtype=np.float32
    )
    vectors = ends_flt - starts_flt
    lengths = np.linalg.norm(vectors, axis=1)
    lengths = np.clip(lengths, 1e-9, None)
    directions = vectors / lengths[:, np.newaxis]

    # Composante angulaire
    cos_theta = np.clip(np.dot(directions, directions.T), -1.0, 1.0)
    d_angle = (1.0 - cos_theta) * (lengths[:, np.newaxis] + lengths[np.newaxis, :])

    # Composante perpendiculaire
    vx = directions[:, 0:1]
    vy = directions[:, 1:2]
    vec_sx = starts_flt[np.newaxis, :, 0] - starts_flt[:, np.newaxis, 0]
    vec_sy = starts_flt[np.newaxis, :, 1] - starts_flt[:, np.newaxis, 1]
    vec_ex = ends_flt[np.newaxis, :, 0] - starts_flt[:, np.newaxis, 0]
    vec_ey = ends_flt[np.newaxis, :, 1] - starts_flt[:, np.newaxis, 1]

    cross_s = np.abs(vx * vec_sy - vy * vec_sx)
    cross_e = np.abs(vx * vec_ey - vy * vec_ex)
    sum_cross = cross_s + cross_e
    d_perp = np.zeros_like(sum_cross)
    mask = sum_cross > 0
    d_perp[mask] = (cross_s[mask] ** 2 + cross_e[mask] ** 2) / sum_cross[mask]

    # Composante parallèle
    proj_s = vec_sx * vx + vec_sy * vy
    proj_e = vec_ex * vx + vec_ey * vy
    base_l = lengths[:, np.newaxis]
    d_par = (
        np.minimum(np.abs(proj_s), np.abs(proj_s - base_l))
        + np.minimum(np.abs(proj_e), np.abs(proj_e - base_l))
    )

    # Combinaison pondérée
    D_asym = w_perp * d_perp + w_angle * d_angle + w_par * d_par
    len_mask = lengths[:, np.newaxis] > lengths[np.newaxis, :]
    similarity_matrix = -np.where(len_mask, D_asym, D_asym.T)

    return similarity_matrix


def run_clustering(
    target_folder,
    max_files=None,
    algo="affinity",
    min_length=5.0,
    n_clusters=50,
    damping=0.9,
    max_iter=400,
    w_perp=1.0,
    w_angle=1.0,
    w_par=1.0,
):
    """Lance le clustering sur les segments compressés.

    Args:
        target_folder: Dossier contenant les JSON compressés.
        max_files: Limiter le nombre de fichiers chargés.
        algo: Algorithme ('affinity', 'kmeans', 'kmedoids').
        min_length: Longueur minimale des segments à inclure.
        n_clusters: Nombre de clusters pour KMeans/KMedoids.
        damping: Facteur d'amortissement pour Affinity Propagation.
        max_iter: Nombre max d'itérations pour AP/KMedoids.
        w_perp: Poids TRACLUS perpendiculaire (AP/kmedoids).
        w_angle: Poids TRACLUS angulaire (AP/kmedoids).
        w_par: Poids TRACLUS parallèle (AP/kmedoids).
    """
    # On passe le parametre min_length a load_data
    segments, meta = load_data(target_folder, max_files=max_files, min_length=min_length)
    n = len(segments)

    if n == 0:
        logger.warning("Aucun segment charge.")
        return

    logger.info("Analyse de %d segments...", n)

    target_path = Path(target_folder)
    output_dir = target_path.parent.parent / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # BRANCHE K-MEANS : Ne calcule JAMAIS la matrice de similarité
    # Travaille directement sur les features des segments — compatible 169k+
    # ========================================================
    if algo == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler

        logger.info("Extraction des features pour K-Means...")
        features = []
        for s in segments:
            mid_x = (s.start.x + s.end.x) / 2.0
            mid_y = (s.start.y + s.end.y) / 2.0
            dx = s.end.x - s.start.x
            dy = s.end.y - s.start.y
            length = np.sqrt(dx**2 + dy**2)
            features.append([mid_x, mid_y, dx, dy, length])

        X = np.array(features, dtype=np.float32)

        # Normalisation pour équilibrer les features (coordonnées vs dx/dy)
        logger.info("Normalisation des features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # MiniBatchKMeans : traitement par lots, compatible avec n très grand
        logger.info("Clustering en %d clusters (MiniBatchKMeans)...", n_clusters)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=4096,
            n_init=10,
            verbose=1,
        )
        labels = kmeans.fit_predict(X_scaled)
        n_clusters_found = n_clusters
        logger.info("TERMINE ! %d clusters trouves.", n_clusters_found)

    # ========================================================
    # BRANCHE AFFINITY PROPAGATION : Calcul de la matrice N×N
    # Requiert --max_files pour ne pas exploser la RAM
    # ========================================================
    elif algo == "affinity":
        MAX_SEGMENTS_AFFINITY = 5000
        if n > MAX_SEGMENTS_AFFINITY:
            logger.error("Affinity Propagation est limite a %d segments.", MAX_SEGMENTS_AFFINITY)
            logger.error("   %d segments detectes. Reduisez avec --max_files.", n)
            logger.error("   Alternative sans limite : --algo kmeans")
            return

        cache_dir = target_path.parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"sim_matrix_{target_path.name}_n{n}.npy"

        if cache_file.exists():
            print(f"Chargement de la matrice depuis le cache : {cache_file}")
            similarity_matrix = np.load(cache_file)
        else:
            print("Calcul matriciel des similarites TRACLUS...")
            similarity_matrix = compute_traclus_similarity(
                segments, w_perp=w_perp, w_angle=w_angle, w_par=w_par
            )
            np.save(cache_file, similarity_matrix)
            logger.info("Matrice sauvegardee dans le cache : %s", cache_file)

        # Remplissage diagonale (mediane)
        med = np.median(similarity_matrix)
        np.fill_diagonal(similarity_matrix, med)

        logger.info("Lancement Affinity Propagation (Custom)...")
        af = CustomAffinityPropagation(damping=damping, max_iter=max_iter, verbose=True)
        af.fit(similarity_matrix)
        labels = af.labels_
        n_clusters_found = (
            len(af.cluster_centers_indices_)
            if af.cluster_centers_indices_ is not None
            else 0
        )
        logger.info("TERMINE ! %d clusters trouves.", n_clusters_found)

    # ========================================================
    # BRANCHE K-MÉDOÏDES : Réutilise la même matrice TRACLUS que
    # l'Affinity Propagation, mais choisit des représentants réels.
    # Limité à ~5000 segments (matrice N×N en RAM).
    # ========================================================
    elif algo == "kmedoids":
        MAX_SEGMENTS_KMEDOIDS = 5000
        if n > MAX_SEGMENTS_KMEDOIDS:
            logger.error("K-Médoïdes est limité à %d segments.", MAX_SEGMENTS_KMEDOIDS)
            logger.error("   %d segments détectés. Réduisez avec --max_files.", n)
            logger.error("   Alternative sans limite : --algo kmeans")
            return

        cache_dir = target_path.parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"sim_matrix_{target_path.name}_n{n}.npy"

        if cache_file.exists():
            logger.info("Chargement de la matrice depuis le cache : %s", cache_file)
            similarity_matrix = np.load(cache_file)
        else:
            logger.info("Calcul matriciel des similarites TRACLUS...")
            similarity_matrix = compute_traclus_similarity(
                segments, w_perp=w_perp, w_angle=w_angle, w_par=w_par
            )
            np.save(cache_file, similarity_matrix)
            logger.info("Matrice sauvegardee dans le cache : %s", cache_file)

        # Conversion similarité → distance (similarity_matrix = -D_asym, donc D = -similarity_matrix)
        distance_matrix = -similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)

        logger.info("Lancement K-Médoïdes (Custom PAM) avec %d clusters...", n_clusters)
        km = CustomKMedoids(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        km.fit(distance_matrix)
        labels = km.labels_
        n_clusters_found = len(np.unique(labels))
        logger.info("TERMINE ! %d clusters trouvés.", n_clusters_found)
        logger.info("Médoïdes (indices globaux) : %s", km.medoid_indices_)

    else:
        logger.error("Algorithme inconnu : '%s'. Choisir 'affinity', 'kmeans' ou 'kmedoids'.", algo)
        return

    # Sauvegarde
    results = {}
    for idx, label in enumerate(labels):
        m_id = meta[idx]["match_id"]
        s_id = meta[idx]["seg_id"]

        if m_id not in results:
            results[m_id] = {}
        results[m_id][s_id] = int(label)

    folder_name = target_path.name
    filename = f"clusters_result_{folder_name}.json"
    full_output_path = output_dir / filename

    with open(full_output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Resultats sauvegardes dans : %s", full_output_path)
