import json
import numpy as np
import os
from pathlib import Path

# Imports locaux
from .custom_ap import CustomAffinityPropagation
from .structures import Segment, TrajectoryPoint
from .geometry import GeometryUtils


def load_data(folder_path, limit=3000, max_files=None):
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
            if len(all_segments) > limit:
                print(f"Limite de segments ({limit}) atteinte. On arrete le chargement.")
                return all_segments[:limit], metadata[:limit]

    return all_segments, metadata


def run_clustering(target_folder, max_files=None):
    # On passe le parametre max_files a load_data
    segments, meta = load_data(target_folder, max_files=max_files)
    n = len(segments)

    if n == 0:
        print("Aucun segment charge.")
        return

    print(f"Analyse de {n} segments...")

    target_path = Path(target_folder)
    
    # 1. Gestion du Cache pour la Matrice de Distance
    # On la place au meme niveau que output/clusters par exemple ou via config
    cache_dir = target_path.parent.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"sim_matrix_{target_path.name}_n{n}.npy"

    if cache_file.exists():
        print(f"Chargement de la matrice depuis le cache : {cache_file}")
        similarity_matrix = np.load(cache_file)
    else:
        # 2. Pre-calcul vectoriel des proprietes
        print("Pre-calcul des vecteurs et longueurs...")
        starts = np.array([(s.start.x, s.start.y) for s in segments], dtype=np.float64)
        ends = np.array([(s.end.x, s.end.y) for s in segments], dtype=np.float64)
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=1)

        geo = GeometryUtils()
        similarity_matrix = np.zeros((n, n), dtype=np.float32)

        print("Calcul des distances en cours...")
        total_calculations = (n * (n - 1)) // 2
        count = 0
        milestone = max(1, total_calculations // 10)

        # Calcul matriciel
        for i in range(n):
            for j in range(i + 1, n):
                # Distance Angulaire
                d_angle = geo.angular_distance(vectors[i], vectors[j]) * (lengths[i] + lengths[j])
                
                # Distance Parallele
                d_par = geo.parallel_distance(starts[i], ends[i], starts[j], ends[j])
                
                # Distance Perpendiculaire (Projete le plus court sur le plus long)
                if lengths[i] > lengths[j]:
                    base_s, base_e = starts[i], ends[i]
                    other_s, other_e = starts[j], ends[j]
                else:
                    base_s, base_e = starts[j], ends[j]
                    other_s, other_e = starts[i], ends[i]
                    
                d_perp_1 = geo.perpendicular_distance(other_s, base_s, base_e)
                d_perp_2 = geo.perpendicular_distance(other_e, base_s, base_e)
                
                if d_perp_1 + d_perp_2 == 0:
                    d_perp = 0.0
                else:
                    d_perp = (d_perp_1**2 + d_perp_2**2) / (d_perp_1 + d_perp_2)
                    
                # Distance totale
                dist = d_perp + d_angle + d_par

                sim = -dist
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

                count += 1
                if count % milestone == 0:
                    print(f"   Progression : {int(count / total_calculations * 100)}%")

        # Sauvegarde de la matrice calculee dans le cache
        np.save(cache_file, similarity_matrix)
        print(f"Matrice sauvegardee ! ({cache_file})")

    # Remplissage diagonale (mediane)
    med = np.median(similarity_matrix)
    np.fill_diagonal(similarity_matrix, med)

    # 3. Clustering (Custom AP)
    print("Lancement Affinity Propagation (Custom)...")

    # Instanciation de l'algo
    af = CustomAffinityPropagation(damping=0.9, max_iter=400, verbose=True)
    af.fit(similarity_matrix)

    labels = af.labels_
    n_clusters = (
        len(af.cluster_centers_indices_)
        if af.cluster_centers_indices_ is not None
        else 0
    )

    print(f"\nTERMINE ! {n_clusters} clusters trouves.")

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
