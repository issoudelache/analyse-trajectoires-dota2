import json
import glob
import os
import matplotlib.pyplot as plt

def visualize_cluster(cluster_id_target, results_file, original_data_folder):
    """Affiche tous les segments appartenant à un cluster donné (Compatible nouveau format)."""
    
    # 1. Charger les résultats du clustering
    if not os.path.exists(results_file):
        print(f"❌ Fichier résultats introuvable : {results_file}")
        return

    with open(results_file, 'r') as f:
        cluster_mapping = json.load(f)

    segments_to_plot = []
    found_count = 0
    
    print(f"🔍 Recherche des segments du Cluster {cluster_id_target}...")
    
    # 2. Parcourir les matchs présents dans le fichier de résultats
    for match_id, segments_dict in cluster_mapping.items():
        
        # Recherche du fichier source
        pattern = os.path.join(original_data_folder, f"*{match_id}*.json")
        files = glob.glob(pattern)
        
        if not files:
            continue
            
        with open(files[0], 'r') as f:
            original_data = json.load(f)
            
        # --- NOUVEAU LOGIQUE DE RECUPERATION ---
        # On doit recréer le mapping {ID_UNIQUE : Données} pour ce match
        orig_segs = {}
        
        if 'players' in original_data:
            for player in original_data['players']:
                p_id = player['player_id']
                for idx, s in enumerate(player['segments']):
                    # On génère le même ID que dans main_clustering.py
                    uid = f"P{p_id}_{idx}"
                    orig_segs[uid] = s
        
        # 3. Récupérer les segments du cluster
        for seg_id_str, cluster_label in segments_dict.items():
            if int(cluster_label) == cluster_id_target:
                if seg_id_str in orig_segs:
                    segments_to_plot.append(orig_segs[seg_id_str])
                    found_count += 1

    # 4. Affichage
    if not segments_to_plot:
        print(f"❌ Aucun segment trouvé pour le cluster {cluster_id_target}.")
        return

    print(f"✨ Affichage de {len(segments_to_plot)} segments.")
    
    plt.figure(figsize=(10, 10))
    plt.title(f"Cluster {cluster_id_target} ({len(segments_to_plot)} segments) - w_error 12")
    
    for s in segments_to_plot:
        p1 = s['start']
        p2 = s['end']
        
        # Flèches pour la direction
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        
        plt.arrow(p1['x'], p1['y'], dx, dy, 
                  color='blue', alpha=0.15, 
                  head_width=2, length_includes_head=True)
        
        # Point de départ
        plt.plot(p1['x'], p1['y'], 'r.', markersize=2, alpha=0.3)

    plt.grid(True)
    plt.xlim(0, 200) # Ajuste selon la taille de ta map
    plt.ylim(0, 200)
    # Inversion de l'axe Y si nécessaire (DotA a parfois l'origine en bas à gauche, parfois en haut)
    # plt.gca().invert_yaxis() 
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION MISE À JOUR ---
    
    # Le fichier que main_clustering.py va générer
    FILE_CLUSTERS = "clusters/clusters_result_w_error_12.json" 
    
    # Ton dossier de données brutes
    DOSSIER_DONNEES = "compressed_data/w_error_12"         
    
    # Choisis un cluster à voir (après avoir lancé le clustering)
    CLUSTER_A_VOIR = 26                                  

    visualize_cluster(CLUSTER_A_VOIR, FILE_CLUSTERS, DOSSIER_DONNEES)