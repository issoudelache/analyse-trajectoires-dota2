import json
import numpy as np
from pathlib import Path

# Imports locaux
from .custom_ap import CustomAffinityPropagation
from .structures import Segment, TrajectoryPoint
from .geometry import GeometryUtils


def compute_total_distance(
    s1: Segment, s2: Segment, w_perp=1.0, w_angle=1.0, w_par=1.0
) -> float:
    """
    Calcule la distance composite (TRACLUS) directement ici.
    Combine : Angulaire + Parallèle + Perpendiculaire
    """
    geo = GeometryUtils()

    # 1. Distance Angulaire
    v1 = s1.vector()
    v2 = s2.vector()
    # Pondération par la longueur pour favoriser les longs segments similaires
    d_angle = geo.angular_distance(v1, v2) * (s1.length() + s2.length())

    # 2. Distance Parallèle (Maintenant dans geometry.py)
    d_par = geo.parallel_distance(
        (s1.start.x, s1.start.y),
        (s1.end.x, s1.end.y),
        (s2.start.x, s2.start.y),
        (s2.end.x, s2.end.y),
    )

    # 3. Distance Perpendiculaire (TRACLUS Standard)
    # On projette le plus court sur le plus long
    if s1.length() > s2.length():
        base, other = s1, s2
    else:
        base, other = s2, s1

    d_perp_1 = geo.perpendicular_distance(
        (other.start.x, other.start.y),
        (base.start.x, base.start.y),
        (base.end.x, base.end.y),
    )
    d_perp_2 = geo.perpendicular_distance(
        (other.end.x, other.end.y),
        (base.start.x, base.start.y),
        (base.end.x, base.end.y),
    )

    if d_perp_1 + d_perp_2 == 0:
        d_perp = 0.0
    else:
        # Moyenne quadratique pour pénaliser les gros écarts
        d_perp = (d_perp_1**2 + d_perp_2**2) / (d_perp_1 + d_perp_2)

    return w_perp * d_perp + w_angle * d_angle + w_par * d_par


def load_data(folder_path, limit=3000, max_files=None):
    """Charge les segments. max_files limite le nombre de fichiers JSON lus."""
    folder = Path(folder_path)

    # On récupère tous les fichiers et on les trie pour que ce soit toujours les mêmes
    files = sorted(list(folder.glob("*.json")))

    print(f"📂 Recherche dans : {folder}")
    print(f"   -> {len(files)} fichiers disponibles au total.")

    # --- NOUVELLE LOGIQUE ICI ---
    if max_files is not None and max_files < len(files):
        files = files[:max_files]
        print(
            f"   ⚠️  Restriction : On ne charge que les {max_files} premiers fichiers."
        )
    # ----------------------------

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

            # On garde aussi la limite de segments pour éviter l'explosion mémoire
            if len(all_segments) > limit:
                print(
                    f"⚠️ Limite de segments ({limit}) atteinte. On arrête le chargement."
                )
                return all_segments[:limit], metadata[:limit]

    return all_segments, metadata


def run_clustering(target_folder, max_files=None):
    # On passe le paramètre max_files à load_data
    segments, meta = load_data(target_folder, max_files=max_files)
    n = len(segments)

    if n == 0:
        print("❌ Aucun segment chargé.")
        return

    print(f"📊 Analyse de {n} segments...")

    # 2. Matrice de Distance
    similarity_matrix = np.zeros((n, n))

    print("🔄 Calcul des distances en cours...")
    total_calculations = (n * (n - 1)) // 2
    count = 0
    milestone = max(1, total_calculations // 10)

    for i in range(n):
        for j in range(i + 1, n):
            # APPEL DE LA FONCTION LOCALE
            dist = compute_total_distance(segments[i], segments[j])

            sim = -dist
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

            count += 1
            if count % milestone == 0:
                print(f"   Progression : {int(count / total_calculations * 100)}%")

    # Remplissage diagonale (médiane)
    med = np.median(similarity_matrix)
    np.fill_diagonal(similarity_matrix, med)

    # 3. Clustering (Custom AP)
    print("🧠 Lancement Affinity Propagation (Custom)...")

    # Instanciation de notre algo maison
    af = CustomAffinityPropagation(damping=0.9, max_iter=400, verbose=True)
    af.fit(similarity_matrix)

    labels = af.labels_
    n_clusters = (
        len(af.cluster_centers_indices_)
        if af.cluster_centers_indices_ is not None
        else 0
    )

    print(f"\n✅ TERMINÉ ! {n_clusters} clusters trouvés.")

    # 4. Sauvegarde
    results = {}
    for idx, label in enumerate(labels):
        m_id = meta[idx]["match_id"]
        s_id = meta[idx]["seg_id"]

        if m_id not in results:
            results[m_id] = {}
        results[m_id][s_id] = int(label)

    OUTPUT_DIR = Path("output/clusters")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    folder_name = Path(target_folder).name
    filename = f"clusters_result_{folder_name}.json"
    full_output_path = OUTPUT_DIR / filename

    with open(full_output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Résultats sauvegardés dans : {full_output_path}")
