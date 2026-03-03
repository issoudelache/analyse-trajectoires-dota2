import json
import numpy as np
from pathlib import Path

# Imports locaux
from .custom_ap import CustomAffinityPropagation
from .structures import Segment, TrajectoryPoint


def load_data(folder_path, limit=None, max_files=None):
    """Charge les segments. max_files limite le nombre de fichiers JSON lus."""
    folder = Path(folder_path)

    # On recupere tous les fichiers et on les trie
    files = sorted(list(folder.glob("*.json")))

    print(f"Recherche dans : {folder}")
    print(f"   -> {len(files)} fichiers disponibles au total.")

    if max_files is not None and max_files < len(files):
        files = files[:max_files]
        print(f"   Restriction : On ne charge que les {max_files} premiers fichiers.")

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

                        if seg.length() > 5.0:
                            all_segments.append(seg)
                            metadata.append(
                                {"match_id": match_id, "seg_id": unique_seg_id}
                            )
                    except KeyError:
                        continue

            # Limite de segments pour eviter l'explosion memoire
            if limit is not None and len(all_segments) > limit:
                print(
                    f"Limite de segments ({limit}) atteinte. On arrete le chargement."
                )
                return all_segments[:limit], metadata[:limit]

    return all_segments, metadata


def run_clustering(target_folder, max_files=None, algo="affinity"):
    # On passe le parametre max_files a load_data
    segments, meta = load_data(target_folder, max_files=max_files)
    n = len(segments)

    if n == 0:
        print("Aucun segment charge.")
        return

    print(f"Analyse de {n} segments...")

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

        print("Extraction des features pour K-Means...")
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
        print("Normalisation des features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # MiniBatchKMeans : traitement par lots, compatible avec n très grand
        n_clusters = 50
        print(f"Clustering en {n_clusters} clusters (MiniBatchKMeans)...")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=4096,
            n_init=10,
            verbose=1,
        )
        labels = kmeans.fit_predict(X_scaled)
        n_clusters_found = n_clusters
        print(f"\nTERMINE ! {n_clusters_found} clusters trouves.")

    # ========================================================
    # BRANCHE AFFINITY PROPAGATION : Calcul de la matrice N×N
    # Requiert --max_files pour ne pas exploser la RAM
    # ========================================================
    elif algo == "affinity":
        MAX_SEGMENTS_AFFINITY = 5000
        if n > MAX_SEGMENTS_AFFINITY:
            print(f"⚠️  ERREUR : Affinity Propagation est limite a {MAX_SEGMENTS_AFFINITY} segments.")
            print(f"   {n} segments detectes. Reduisez avec --max_files.")
            print(f"   Alternative sans limite : --algo kmeans")
            return

        cache_dir = target_path.parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"sim_matrix_{target_path.name}_n{n}.npy"

        if cache_file.exists():
            print(f"Chargement de la matrice depuis le cache : {cache_file}")
            similarity_matrix = np.load(cache_file)
        else:
            print("Pre-calcul vectoriel des proprietes...")
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

            print("Calcul matriciel des similarites en cours (Ultra-Optimise)...")
            cos_theta = np.clip(np.dot(directions, directions.T), -1.0, 1.0)
            d_angle = (1.0 - cos_theta) * (lengths[:, np.newaxis] + lengths[np.newaxis, :])

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

            proj_s = vec_sx * vx + vec_sy * vy
            proj_e = vec_ex * vx + vec_ey * vy
            base_l = lengths[:, np.newaxis]
            d_par = (
                np.minimum(np.abs(proj_s), np.abs(proj_s - base_l))
                + np.minimum(np.abs(proj_e), np.abs(proj_e - base_l))
            )

            D_asym = d_perp + d_angle + d_par
            len_mask = lengths[:, np.newaxis] > lengths[np.newaxis, :]
            similarity_matrix = -np.where(len_mask, D_asym, D_asym.T)

            np.save(cache_file, similarity_matrix)
            print(f"Matrice sauvegardee dans le cache : {cache_file}")

        # Remplissage diagonale (mediane)
        med = np.median(similarity_matrix)
        np.fill_diagonal(similarity_matrix, med)

        print("Lancement Affinity Propagation (Custom)...")
        af = CustomAffinityPropagation(damping=0.9, max_iter=400, verbose=True)
        af.fit(similarity_matrix)
        labels = af.labels_
        n_clusters_found = (
            len(af.cluster_centers_indices_)
            if af.cluster_centers_indices_ is not None
            else 0
        )
        print(f"\nTERMINE ! {n_clusters_found} clusters trouves.")

    else:
        print(f"Algorithme inconnu : '{algo}'. Choisir 'affinity' ou 'kmeans'.")
        return

    # 4. Sauvegarde
    results = {}
    for idx, label in enumerate(labels):
        m_id = meta[idx]["match_id"]
        s_id = meta[idx]["seg_id"]

        if m_id not in results:
            results[m_id] = {}
        results[m_id][s_id] = int(label)

    # Fix: Eviter le hardcoding et se baser sur target_folder
    output_dir = target_path.parent.parent / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_name = target_path.name
    filename = f"clusters_result_{folder_name}.json"
    full_output_path = output_dir / filename

    with open(full_output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Resultats sauvegardes dans : {full_output_path}")
