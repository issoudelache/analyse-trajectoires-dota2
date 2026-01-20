import json
import glob
import os
import numpy as np
from sklearn.cluster import AffinityPropagation

# Import de tes classes
from structures import Segment, TrajectoryPoint
from clustering_utils import SegmentDistance

def load_data(folder_path, limit=2000):
    """
    Charge les segments depuis un dossier spécifique (ex: exported_data/w_error_1.1).
    """
    # On construit le chemin de recherche : dossier + *.json
    search_path = os.path.join(folder_path, "*.json")
    files = glob.glob(search_path)
    
    all_segments = []
    metadata = [] 
    
    print(f"📂 Recherche dans : {folder_path}")
    print(f"   -> {len(files)} fichiers trouvés.")
    
    if len(files) == 0:
        print("⚠️  Aucun fichier trouvé ! Vérifiez le chemin du dossier.")
        return [], []
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            # On convertit en string pour éviter des soucis si l'ID est un entier
            match_id = str(data.get('match_id', 'unknown')) 
            
            for s in data['segments']:
                # On recrée les objets Segment
                p1 = TrajectoryPoint(s['start']['x'], s['start']['y'], s['start']['tick'])
                p2 = TrajectoryPoint(s['end']['x'], s['end']['y'], s['end']['tick'])
                seg = Segment(p1, p2)
                
                # Filtre : on ignore les segments trop petits (bruit)
                if seg.length() > 5.0:
                    all_segments.append(seg)
                    metadata.append({'match_id': match_id, 'seg_id': s['id']})

    # Sécurité pour éviter de surcharger la mémoire lors du test
    if len(all_segments) > limit:
        print(f"⚠️  Trop de segments ({len(all_segments)}). On garde les {limit} premiers pour le test.")
        return all_segments[:limit], metadata[:limit]
    
    return all_segments, metadata

def run_clustering(target_folder):
    # 1. Chargement des données du dossier cible
    segments, meta = load_data(target_folder)
    
    n = len(segments)
    if n == 0:
        return

    print(f"📊 Analyse de {n} segments...")

    # 2. Calcul de la Matrice de Distance
    calculator = SegmentDistance()
    similarity_matrix = np.zeros((n, n))

    print("🔄 Calcul des distances en cours (cela peut prendre du temps)...")
    # Astuce d'optimisation : Affichage de progression
    total_calculations = (n * (n - 1)) // 2
    count = 0
    milestone = total_calculations // 10  # Barre de progression tous les 10%
    
    for i in range(n):
        for j in range(i + 1, n):
            # Distance combinée (Parallèle + Angulaire + Perpendiculaire)
            dist = calculator.compute_total_distance(segments[i], segments[j])
            
            # Affinity Propagation demande une similarité (négatif de la distance)
            sim = -dist
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            
            count += 1
            if milestone > 0 and count % milestone == 0:
                print(f"   Progression : {int(count/total_calculations*100)}%")

    # Remplir la diagonale avec la médiane (standard pour Affinity Propagation)
    med = np.median(similarity_matrix)
    np.fill_diagonal(similarity_matrix, med)

    # 3. Clustering (Affinity Propagation)
    print("🧠 Lancement de l'IA (Affinity Propagation)...")
    af = AffinityPropagation(affinity='precomputed', damping=0.9, random_state=42)
    af.fit(similarity_matrix)

    labels = af.labels_
    n_clusters = len(af.cluster_centers_indices_)
    
    print(f"\n✅ TERMINÉ ! {n_clusters} clusters trouvés.")

    # 4. Sauvegarde des résultats
    results = {}
    for idx, label in enumerate(labels):
        m_id = meta[idx]['match_id']
        s_id = str(meta[idx]['seg_id']) # Conversion en str pour JSON
        
        if m_id not in results:
            results[m_id] = {}
        
        results[m_id][s_id] = int(label)

    #  On définit le nom du dossier de sortie
    OUTPUT_DIR = "clusters"
    
    #  On crée le dossier s'il n'existe pas (le 'exist_ok=True' évite les erreurs si le dossier est déjà là)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #  On construit le nom du fichier
    folder_name = os.path.basename(os.path.normpath(target_folder))
    filename = f"clusters_result_{folder_name}.json"
    
    #  On combine dossier + fichier (ex: clusters/clusters_result_w_error_100.55.json)
    full_output_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(full_output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Résultats sauvegardés dans : {full_output_path}")

if __name__ == "__main__":
    # C'est ici que tu choisis quel dossier analyser !
    # Modifie cette ligne pour changer de test (ex: w_error_2.11, etc.)
    DOSSIER_A_ANALYSER = "exported_data/w_error_15.17"
    
    run_clustering(DOSSIER_A_ANALYSER)