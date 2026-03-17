"""
Module de visualisation interactive des trajectoires Dota 2.
Contient les fonctions pour afficher les trajectoires sur la carte avec des contrôles interactifs.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider
import networkx as nx
import numpy as np

# Couleurs pour les 10 joueurs
PLAYER_COLORS = [
    "#3498db",
    "#2ecc71",
    "#9b59b6",
    "#f39c12",
    "#e74c3c",
    "#16a085",
    "#27ae60",
    "#8e44ad",
    "#d35400",
    "#c0392b",
]


def get_available_w_errors(data_dir):
    """Récupère la liste des valeurs w_error disponibles."""
    w_errors = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.startswith("w_error_"):
            w_error_str = d.name.replace("w_error_", "")
            try:
                w_error = float(w_error_str)
                w_errors.append(w_error)
            except ValueError:
                pass
    return sorted(w_errors)


def get_available_games(data_dir, w_error):
    """Récupère la liste des game IDs disponibles pour un w_error donné."""
    if w_error == int(w_error):
        w_error_str = str(int(w_error))
    else:
        w_error_str = str(w_error)

    w_error_dir = data_dir / f"w_error_{w_error_str}"
    if not w_error_dir.exists():
        return []

    games = []
    for f in sorted(w_error_dir.glob("*_compressed.json")):
        game_id = f.stem.replace("_compressed", "")
        games.append(game_id)
    return games


def load_compressed_data(data_dir, w_error, game_id):
    """Charge les données compressées."""
    if w_error == int(w_error):
        w_error_str = str(int(w_error))
    else:
        w_error_str = str(w_error)

    json_path = data_dir / f"w_error_{w_error_str}" / f"{game_id}_compressed.json"

    if not json_path.exists():
        w_error_str = str(float(w_error))
        json_path = data_dir / f"w_error_{w_error_str}" / f"{game_id}_compressed.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


class InteractiveOverlay:
    """Visualisation interactive avec slider temporel."""

    def __init__(self, canvas_path, data_dir, w_error, game_id):
        self.w_error = w_error
        self.game_id = game_id

        # Charger les données
        canvas_full = mpimg.imread(canvas_path)
        self.data = load_compressed_data(data_dir, w_error, game_id)

        # Recadrer canvas pour obtenir la partie carrée (933x933 au centre)
        height, width = canvas_full.shape[:2]
        if width > height:
            left = (width - height) // 2
            self.canvas = canvas_full[:, left : left + height]
        else:
            self.canvas = canvas_full

        # Extraire les segments et trouver les bornes temporelles
        self.player_segments = {}
        self.min_tick = float("inf")
        self.max_tick = 0

        for player in self.data["players"]:
            player_id = player["player_id"]
            segments = []

            for seg in player["segments"]:
                segments.append({"start": seg["start"], "end": seg["end"]})
                self.min_tick = min(
                    self.min_tick, seg["start"]["tick"], seg["end"]["tick"]
                )
                self.max_tick = max(
                    self.max_tick, seg["start"]["tick"], seg["end"]["tick"]
                )

            self.player_segments[player_id] = segments

        # Initialiser la figure
        self.setup_figure()

    def setup_figure(self):
        """Configure la figure et les contrôles."""
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        plt.subplots_adjust(bottom=0.15)

        # --- CORRECTION SENS CARTE ---
        # origin='upper' (défaut) met le pixel [0,0] en haut.
        # extent=[0,256,0,256] mappe le bas de l'image à y=0 et le haut à y=256.
        # Cela remet la base Radiant (Bas de l'image) en Bas du graphe.
        self.ax.imshow(self.canvas, extent=[0, 256, 0, 256], origin="upper")

        self.ax.set_xlim(0, 256)
        self.ax.set_ylim(0, 256)
        self.ax.set_aspect("equal")

        # Titre
        self.ax.set_title(
            f"Match {self.game_id} - w_error={self.w_error}\n"
            f"Utiliser le slider pour avancer dans le temps",
            fontsize=14,
            fontweight="bold",
        )

        # Slider temporel
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        self.slider = Slider(
            ax_slider,
            "Temps",
            self.min_tick,
            self.max_tick,
            valinit=self.min_tick,
            valstep=30,
        )
        self.slider.on_changed(self.update)

        # Variables d'état pour zoom/pan
        self.zoom_level = 1.0
        self.pan_x = 128
        self.pan_y = 128

        # Événements
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.dragging = False
        self.drag_start = None

        # Dessiner l'état initial
        self.update(self.min_tick)

    def update(self, current_tick):
        """Met à jour l'affichage pour le tick actuel."""
        # Effacer les segments et points précédents
        for artist in self.ax.lines + self.ax.collections:
            artist.remove()

        current_positions = {}

        # Dessiner les segments jusqu'au tick actuel
        for player_id, segments in self.player_segments.items():
            color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]

            for seg in segments:
                start_tick = seg["start"]["tick"]
                end_tick = seg["end"]["tick"]

                # Segment complètement dans le passé
                if end_tick <= current_tick:
                    self.ax.plot(
                        [seg["start"]["x"], seg["end"]["x"]],
                        [seg["start"]["y"], seg["end"]["y"]],
                        color=color,
                        linewidth=2,
                        alpha=0.7,
                    )
                    current_positions[player_id] = (seg["end"]["x"], seg["end"]["y"])

                # Segment en cours
                elif start_tick <= current_tick < end_tick:
                    # Interpolation linéaire
                    ratio = (current_tick - start_tick) / (end_tick - start_tick)
                    x_current = seg["start"]["x"] + ratio * (
                        seg["end"]["x"] - seg["start"]["x"]
                    )
                    y_current = seg["start"]["y"] + ratio * (
                        seg["end"]["y"] - seg["start"]["y"]
                    )

                    # Dessiner la partie complétée
                    self.ax.plot(
                        [seg["start"]["x"], x_current],
                        [seg["start"]["y"], y_current],
                        color=color,
                        linewidth=2,
                        alpha=0.7,
                    )
                    current_positions[player_id] = (x_current, y_current)

        # Afficher les positions actuelles
        for player_id, (x, y) in current_positions.items():
            color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]
            self.ax.plot(
                x,
                y,
                "o",
                color=color,
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=10,
            )

        # Afficher le temps
        minutes = int((current_tick / 30) // 60)
        seconds = int((current_tick / 30) % 60)
        self.ax.text(
            0.02,
            0.98,
            f"Temps: {minutes:02d}:{seconds:02d}",
            transform=self.ax.transAxes,
            fontsize=14,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        """Gère le zoom avec la molette."""
        if event.inaxes != self.ax:
            return

        zoom_factor = 1.2 if event.button == "up" else 1 / 1.2
        self.zoom_level *= zoom_factor

        # Limites de zoom
        self.zoom_level = max(1.0, min(self.zoom_level, 10.0))

        # Calculer la taille de la vue
        view_size = 256 / self.zoom_level
        half_size = view_size / 2

        self.ax.set_xlim(self.pan_x - half_size, self.pan_x + half_size)
        self.ax.set_ylim(self.pan_y - half_size, self.pan_y + half_size)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        """Début du drag."""
        if event.inaxes == self.ax and event.button == 1:
            self.dragging = True
            self.drag_start = (event.xdata, event.ydata)

    def on_release(self, event):
        """Fin du drag."""
        self.dragging = False
        self.drag_start = None

    def on_motion(self, event):
        """Gère le déplacement (pan)."""
        if self.dragging and event.inaxes == self.ax and self.drag_start:
            dx = self.drag_start[0] - event.xdata
            dy = self.drag_start[1] - event.ydata

            self.pan_x += dx
            self.pan_y += dy

            # Limites de pan
            self.pan_x = max(0, min(256, self.pan_x))
            self.pan_y = max(0, min(256, self.pan_y))

            view_size = 256 / self.zoom_level
            half_size = view_size / 2

            self.ax.set_xlim(self.pan_x - half_size, self.pan_x + half_size)
            self.ax.set_ylim(self.pan_y - half_size, self.pan_y + half_size)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Gère les raccourcis clavier."""
        if event.key == "r":
            # Reset
            self.zoom_level = 1.0
            self.pan_x = 128
            self.pan_y = 128
            self.ax.set_xlim(0, 256)
            self.ax.set_ylim(0, 256)
            self.fig.canvas.draw_idle()
        elif event.key == "s":
            # Sauvegarder
            output_path = (
                Path("output/overlays")
                / f"{self.game_id}_w{self.w_error}_interactive.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✅ Sauvegardé: {output_path}")

    def show(self):
        """Affiche la visualisation."""
        plt.show()


def generate_static_overlay(canvas_path, data_dir, w_error, game_id, output_path):
    """Génère un overlay statique (sans interactivité)."""
    # Charger données
    canvas_full = mpimg.imread(canvas_path)
    data = load_compressed_data(data_dir, w_error, game_id)

    # Recadrer canvas
    height, width = canvas_full.shape[:2]
    if width > height:
        left = (width - height) // 2
        canvas = canvas_full[:, left : left + height]
    else:
        canvas = canvas_full

    # Créer figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # --- CORRECTION SENS CARTE ---
    # origin='upper' (défaut) met le haut de l'image (Dire) en haut.
    ax.imshow(canvas, extent=[0, 256, 0, 256], origin="upper", aspect="equal")

    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_title(
        f"Match {game_id} - Compression MDL (w_error={w_error})",
        fontsize=14,
        fontweight="bold",
    )

    # Dessiner tous les segments
    for player in data["players"]:
        player_id = player["player_id"]
        color = PLAYER_COLORS[player_id % len(PLAYER_COLORS)]

        for seg in player["segments"]:
            x1, y1 = seg["start"]["x"], seg["start"]["y"]
            x2, y2 = seg["end"]["x"], seg["end"]["y"]

            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

            # Flèche directionnelle
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="->",
                mutation_scale=15,
                color=color,
                alpha=0.5,
                linewidth=1,
            )
            ax.add_patch(arrow)

        # Point de départ
        if player["segments"]:
            first_seg = player["segments"][0]
            ax.plot(
                first_seg["start"]["x"],
                first_seg["start"]["y"],
                "o",
                color=color,
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=2,
                label=f"Joueur {player_id}",
            )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    # Sauvegarder
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_cluster_on_map(canvas_path, clusters_file, compressed_dir_path, cluster_id):
    """
    Visualise tous les segments appartenant au cluster K sur la carte du jeu (Mode Fenêtre).
    """
    # 1. Chargement de la carte
    if not canvas_path.exists():
        print(f"❌ Carte introuvable : {canvas_path}")
        return

    canvas_full = mpimg.imread(canvas_path)
    height, width = canvas_full.shape[:2]
    if width > height:
        left = (width - height) // 2
        canvas = canvas_full[:, left : left + height]
    else:
        canvas = canvas_full

    # 2. Chargement du fichier de résultats des clusters
    print(f"📖 Lecture des clusters : {clusters_file}")
    with open(clusters_file, "r") as f:
        cluster_data = json.load(f)

    # 3. Récupération des segments du Cluster K
    # Structure : { match_id: { "P0_12": cluster_label, ... } }
    segments_locations = {}  # { match_id: [list_of_segment_ids] }

    for match_id, segments_dict in cluster_data.items():
        for seg_id, label in segments_dict.items():
            if label == cluster_id:
                if match_id not in segments_locations:
                    segments_locations[match_id] = []
                segments_locations[match_id].append(seg_id)

    if not segments_locations:
        print(f"⚠️  Aucun segment trouvé pour le Cluster {cluster_id} !")
        return

    print(f"🔍 Cluster {cluster_id} : Trouvé dans {len(segments_locations)} matchs.")

    # 4. Chargement des coordonnées depuis les fichiers compressés
    count_segments = 0

    plt.figure(figsize=(10, 10))

    # --- CORRECTION SENS CARTE ---
    # On utilise origin='upper' (défaut) pour que le haut de l'image soit en haut du graphe (y=256)
    # Si on utilisait 'lower', l'image serait inversée verticalement.
    plt.imshow(canvas, extent=[0, 256, 0, 256], origin="upper", aspect="equal")

    # On parcourt chaque match concerné
    for match_id, target_seg_ids in segments_locations.items():
        json_path_int = compressed_dir_path / f"{match_id}_compressed.json"

        if not json_path_int.exists():
            continue

        with open(json_path_int, "r") as f:
            match_data = json.load(f)

        for player in match_data["players"]:
            p_id = player["player_id"]
            for idx, seg in enumerate(player["segments"]):
                current_seg_id = f"P{p_id}_{idx}"

                if current_seg_id in target_seg_ids:
                    p1 = seg["start"]
                    p2 = seg["end"]

                    # --- AFFICHAGE CLAIR ET SUPERPOSÉ ---
                    # 1. La ligne rouge (zorder=2 pour être visible mais sous la flèche)
                    plt.plot(
                        [p1["x"], p2["x"]],
                        [p1["y"], p2["y"]],
                        color="#FF0000",
                        linewidth=2.5,
                        alpha=0.6,
                        zorder=2,
                    )

                    # 2. La flèche jaune (zorder=10 pour être SÛREMENT au-dessus)
                    arrow = FancyArrowPatch(
                        (p1["x"], p1["y"]),
                        (p2["x"], p2["y"]),
                        arrowstyle="-|>",
                        mutation_scale=20,
                        color="#FFFF00",
                        linewidth=0,
                        alpha=1.0,
                        zorder=10,  # <-- PRIORITÉ MAXIMALE
                    )
                    plt.gca().add_patch(arrow)

                    count_segments += 1

    # 5. Finalisation du graphique
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.title(
        f"Visualisation Cluster #{cluster_id} ({count_segments} segments)",
        fontsize=16,
        fontweight="bold",
    )
    plt.axis("off")  # On cache les axes

    print(f"✅ Fenêtre ouverte avec {count_segments} segments.")
    plt.show()


def plot_markov_network(patterns_dict: dict, min_len: int = 2, output_path: str = None):
    """
    Génère un graphe de flux (transitions entre clusters) à partir des motifs PrefixSpan.
    
    Args:
        patterns_dict: Dictionnaire {tuple_de_clusters: support}
        min_len: Taille minimale de la séquence pour être tracée (défaut: 2)
        output_path: Si renseigné, sauvegarde l'image au lieu de l'afficher
    """
    G = nx.DiGraph()

    # 1. Construction du Graphe (Noeuds et Arêtes)
    for pattern, support in patterns_dict.items():
        if len(pattern) < min_len:
            continue
            
        # Création des liens pour chaque transition A -> B
        for i in range(len(pattern) - 1):
            source = pattern[i]
            target = pattern[i + 1]
            
            # Addition des poids si la transition existe déjà via un autre motif
            if G.has_edge(source, target):
                G[source][target]['weight'] += support
            else:
                G.add_edge(source, target, weight=support)

    if len(G.nodes) == 0:
        print("⚠️ Aucun motif multi-étapes à tracer.")
        return

    # 2. Esthétique : Tailles basées sur la popularité
    # La taille du nœud dépend de son utilisation globale comme point de passage
    node_sizes = [min(3000, 300 + 50 * G.in_degree(n, weight='weight') + 50 * G.out_degree(n, weight='weight')) for n in G.nodes()]
    
    # L'épaisseur de l'arête dépend du support
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w / max_weight) * 5 for w in edge_weights]
    
    # Définition des couleurs des noeuds selon leur centralité (poids total)
    node_colors = [G.degree(n, weight='weight') for n in G.nodes()]

    # 3. Rendu
    plt.figure(figsize=(16, 12)) # Agrandissement de la figure
    
    # Ajustement du layout pour mieux espacer les noeuds
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Dessin des noeuds avec palette de couleur (ex: YlOrRd)
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        cmap=plt.cm.YlOrRd, 
        edgecolors='black', 
        linewidths=1.5,
        alpha=0.95
    )
    
    # Dessin des étiquettes (IDs des clusters) avec une police plus adaptée
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="black")
    
    # Dessin des arêtes (flèches de transition) avec transparence
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_widths, 
        edge_color=edge_weights, 
        edge_cmap=plt.cm.Blues, 
        arrowsize=25, 
        alpha=0.7,   # Transparence ajoutée pour adoucir les croisements
        connectionstyle='arc3,rad=0.15' # Légère hausse de la courbure
    )

    plt.title("Graphe des Transitions Macroscopiques (Motifs Dota 2)", fontsize=18, fontweight="bold")
    plt.axis("off")

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✅ Graphe sauvegardé : {output_path}")
        plt.close()
    else:
        plt.show()


def generate_comparison_image(csv_path, json_path, w_error, output_dir):
    """Génère une image de comparaison original vs compressé.

    Args:
        csv_path: Chemin du CSV original.
        json_path: Chemin du JSON compressé.
        w_error: Valeur de w_error utilisée.
        output_dir: Dossier de sortie pour l'image.

    Returns:
        Tuple (success, match_id, size_kb) ou (success, match_id, 0, error_msg).
    """
    import pandas as pd

    match_id = csv_path.stem.replace("coord_", "")

    try:
        # Charger données
        df = pd.read_csv(csv_path)
        with open(json_path, "r") as f:
            compressed_data = json.load(f)

        # Créer figure
        colors = plt.cm.tab10(np.arange(10))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(
            f"Match {match_id} - Compression MDL (w_error={w_error})",
            fontsize=20,
            fontweight="bold",
        )

        total_orig = 0
        total_segments = 0

        # Pour chaque joueur
        for player_id in range(10):
            x_col, y_col = f"x{player_id}", f"y{player_id}"

            if x_col not in df.columns:
                continue

            # Original
            mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
            x_orig = df[x_col][mask].values
            y_orig = df[y_col][mask].values

            if len(x_orig) == 0:
                continue

            total_orig += len(x_orig)
            color = colors[player_id]

            # GRAPHE 1: Original
            step = max(1, len(x_orig) // 500)
            ax1.plot(
                x_orig[::step],
                y_orig[::step],
                color=color,
                linewidth=1.0,
                alpha=0.45,
                label=f"Joueur {player_id}",
            )
            ax1.scatter(
                x_orig[::step],
                y_orig[::step],
                c=[color] * len(x_orig[::step]),
                s=8,
                alpha=0.3,
            )

            # GRAPHE 2: Compressé
            player_data = next(
                (p for p in compressed_data["players"] if p["player_id"] == player_id),
                None,
            )

            if player_data:
                segments = player_data["segments"]
                total_segments += len(segments)

                for seg in segments:
                    ax2.plot(
                        [seg["start"]["x"], seg["end"]["x"]],
                        [seg["start"]["y"], seg["end"]["y"]],
                        color=color,
                        linewidth=1.0,
                        alpha=0.45,
                    )

                if segments:
                    xs = [seg["start"]["x"] for seg in segments] + [
                        segments[-1]["end"]["x"]
                    ]
                    ys = [seg["start"]["y"] for seg in segments] + [
                        segments[-1]["end"]["y"]
                    ]
                    ax2.scatter(
                        xs,
                        ys,
                        c=[color] * len(xs),
                        s=8,
                        zorder=10,
                        edgecolors="black",
                        linewidth=0.5,
                        alpha=0.3,
                    )

        # Configuration graphes
        reduction = (1 - total_segments / total_orig) * 100 if total_orig > 0 else 0

        ax1.set_title(f"Original: {total_orig} points", fontsize=14, fontweight="bold")
        ax1.set_xlabel("X (coordonnées carte)", fontsize=12)
        ax1.set_ylabel("Y (coordonnées carte)", fontsize=12)
        ax1.grid(True, alpha=0.2, linestyle="--")
        ax1.set_aspect("equal")
        ax1.set_facecolor("#f8f8f8")

        # Sauvegarder
        output_path = output_dir / f"{match_id}_w{w_error}_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        return True, match_id, output_path.stat().st_size // 1024

    except Exception as e:
        return False, match_id, 0, str(e)
