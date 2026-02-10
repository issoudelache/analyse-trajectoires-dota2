import json
import glob
import os
import numpy as np
from sklearn.cluster import AffinityPropagation

# Import de tes classes (assure-toi que geometry.py et structures.py sont bien là)
from structures import Segment, TrajectoryPoint
from clustering_utils import SegmentDistance

def load_data(folder_path, limit=5000):
    """
    Charge les segments depuis le nouveau format JSON (Match -> Players -> Segments).
    """
    # Recherche des fichiers JSON
    search_path = os.path.join(folder_path, "*.json")
    files = glob.glob(search_path)
    
    all_segments = []
    metadata = [] 
    
    print(f"📂 Recherche dans : {folder_path}")
    print(f"   -> {len(files)} fichiers trouvés.")
    
    if len(files) == 0:
        print("⚠️  Aucun fichier trouvé ! Vérifiez le chemin (ex: compressed_data/w_error_12).")
        return [], []
    
    for file in files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Erreur de lecture : {file}")
                continue

            match_id = str(data.get('match_id', 'unknown'))
            
            # --- NOUVEAU PARSING ---
            # On parcourt chaque joueur
            if 'players' not in data:
                continue

            for player in data['players']:
                p_id = player['player_id']
                
                # On parcourt les segments du joueur
                for idx, s in enumerate(player['segments']):
                    
                    # On crée un ID unique artificiel : P<id_joueur>_<index_segment>
                    # Ex: P0_12 (Joueur 0, segment 12)
                    unique_seg_id = f"P{p_id}_{idx}"

                    # Création de l'objet Segment
                    try:
                        p1 = TrajectoryPoint(s['start']['x'], s['start']['y'], s['start']['tick'])
                        p2 = TrajectoryPoint(s['end']['x'], s['end']['y'], s['end']['tick'])
                        seg = Segment(p1, p2)
                        
                        # Filtre bruit (optionnel, à ajuster selon tes besoins)
                        if seg.length() > 5.0:
                            all_segments.append(seg)
                            # On stocke l'ID unique qu'on vient de créer
                            metadata.append({
                                'match_id': match_id, 
                                'seg_id': unique_seg_id 
                            })
                            
                    except KeyError as e:
                        print(f"⚠️ Segment malformé dans {file}: {e}")
                        continue

            # Sécurité mémoire (on arrête si on a trop de segments pour le test)
            if len(all_segments) > limit:
                print(f"⚠️  Limite atteinte ({limit} segments). Arrêt du chargement.")
                return all_segments[:limit], metadata[:limit]
    
    return all_segments, metadata

def run_clustering(target_folder):
    # 1. Chargement
    segments, meta = load_data(target_folder)
    
    n = len(segments)
    if n == 0:
        print("❌ Aucun segment chargé. Vérifiez vos dossiers.")
        return

    print(f"📊 Analyse de {n} segments...")

    # 2. Matrice de Distance
    calculator = SegmentDistance()
    similarity_matrix = np.zeros((n, n))

    print("🔄 Calcul des distances en cours...")
    total_calculations = (n * (n - 1)) // 2
    count = 0
    milestone = max(1, total_calculations // 10)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculator.compute_total_distance(segments[i], segments[j])
            sim = -dist
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            
            count += 1
            if count % milestone == 0:
                print(f"   Progression : {int(count/total_calculations*100)}%")

    med = np.median(similarity_matrix)
    np.fill_diagonal(similarity_matrix, med)

    # 3. Clustering
    print("🧠 Lancement Affinity Propagation...")
    af = AffinityPropagation(affinity='precomputed', damping=0.9, random_state=42)
    af.fit(similarity_matrix)

    labels = af.labels_
    n_clusters = len(af.cluster_centers_indices_)
    
    print(f"\n✅ TERMINÉ ! {n_clusters} clusters trouvés.")

    # 4. Sauvegarde
    results = {}
    for idx, label in enumerate(labels):
        m_id = meta[idx]['match_id']
        s_id = meta[idx]['seg_id'] # C'est notre ID "P0_12"
        
        if m_id not in results:
            results[m_id] = {}
        
        results[m_id][s_id] = int(label)

    OUTPUT_DIR = "clusters"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folder_name = os.path.basename(os.path.normpath(target_folder))
    filename = f"clusters_result_{folder_name}.json"
    full_output_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(full_output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Résultats sauvegardés dans : {full_output_path}")

if __name__ == "__main__":
    # Mise à jour avec ton nouveau dossier
    DOSSIER_A_ANALYSER = "compressed_data/w_error_12"
    
    # Vérification que le dossier existe
    if not os.path.exists(DOSSIER_A_ANALYSER):
        print(f"❌ Le dossier '{DOSSIER_A_ANALYSER}' n'existe pas !")
        print("   -> Vérifie que tu es bien à la racine du projet.")
    else:
        run_clustering(DOSSIER_A_ANALYSER)