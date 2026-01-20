import json
import glob
import os
import matplotlib.pyplot as plt

def visualize_cluster(cluster_id_target, results_file, original_data_folder):
    """Affiche tous les segments appartenant à un cluster donné."""
    
    # 1. Charger les résultats du clustering
    with open(results_file, 'r') as f:
        cluster_mapping = json.load(f)

    segments_to_plot = []
    
    print(f"🔍 Recherche des segments du Cluster {cluster_id_target}...")
    
    # 2. Parcourir les matchs
    for match_id, segments_dict in cluster_mapping.items():
        # Trouver le fichier original correspondant
        # On cherche *MATCH_ID*.json dans le dossier
        pattern = os.path.join(original_data_folder, f"*{match_id}*.json")
        files = glob.glob(pattern)
        
        if not files:
            continue
            
        with open(files[0], 'r') as f:
            original_data = json.load(f)
            
        # Créer un dictionnaire rapide {id_segment: données_segment}
        # Les IDs dans le JSON original sont des entiers, dans le résultat des strings
        orig_segs = {str(s['id']): s for s in original_data['segments']}
        
        # 3. Récupérer les coordonnées si le segment est dans notre cluster cible
        for seg_id_str, cluster_label in segments_dict.items():
            if cluster_label == cluster_id_target:
                if seg_id_str in orig_segs:
                    segments_to_plot.append(orig_segs[seg_id_str])

    # 4. Affichage
    if not segments_to_plot:
        print("Aucun segment trouvé pour ce cluster (vérifiez les chemins).")
        return

    print(f"✨ Affichage de {len(segments_to_plot)} segments pour le Cluster {cluster_id_target}")
    
    plt.figure(figsize=(10, 10))
    plt.title(f"Visualisation du Mouvement Type n°{cluster_id_target}")
    
    # Fond de carte approximatif (0-128 ou 0-200 selon tes données)
    # Tu peux ajuster xlim/ylim selon tes coordonnées min/max
    
    for s in segments_to_plot:
        p1 = s['start']
        p2 = s['end']
        # On trace une ligne bleu transparent pour voir la densité
        plt.plot([p1['x'], p2['x']], [p1['y'], p2['y']], 'b-', alpha=0.1)
        
        # Un point rouge au début pour voir le sens du mouvement
        plt.plot(p1['x'], p1['y'], 'r.', markersize=2, alpha=0.1)

    plt.grid(True)
    plt.axis('equal')
    plt.show()

# --- CONFIGURATION À MODIFIER ---
FILE_CLUSTERS = "clusters/clusters_result_w_error_15.17.json" # Ton fichier résultat
DOSSIER_DONNEES = "exported_data/w_error_15.17"         # Ton dossier de données brutes
CLUSTER_A_VOIR = 38                                   # Change ce numéro pour explorer !

if __name__ == "__main__":
    visualize_cluster(CLUSTER_A_VOIR, FILE_CLUSTERS, DOSSIER_DONNEES)