#!/usr/bin/env python3
"""
Visualisation interactive pour ajuster la superposition des trajectoires sur la carte.
Utilise des sliders pour ajuster position, échelle, transparence, etc.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider
import numpy as np

# Configuration des chemins
BASE_DIR = Path(__file__).parent
CANVAS_PATH = BASE_DIR / "canvas.png"
EXPORTED_DATA_DIR = BASE_DIR / "exported_data_mvc"

# Couleurs pour les 10 joueurs
PLAYER_COLORS = [
    '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c',
    '#16a085', '#27ae60', '#8e44ad', '#d35400', '#c0392b'
]


def load_compressed_data(w_error, game_id):
    """Charge les données compressées."""
    # Essayer d'abord avec le format entier (w_error_12)
    if w_error == int(w_error):
        w_error_str = str(int(w_error))
    else:
        w_error_str = str(w_error)
    
    json_path = EXPORTED_DATA_DIR / f"w_error_{w_error_str}" / f"{game_id}_compressed.json"
    
    # Si le fichier n'existe pas, essayer avec le format décimal (w_error_1.0)
    if not json_path.exists():
        w_error_str = str(float(w_error))
        json_path = EXPORTED_DATA_DIR / f"w_error_{w_error_str}" / f"{game_id}_compressed.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


class InteractiveOverlay:
    def __init__(self, w_error, game_id):
        self.w_error = w_error
        self.game_id = game_id
        
        # Charger les données
        canvas_full = mpimg.imread(CANVAS_PATH)
        self.data = load_compressed_data(w_error, game_id)
        
        # Recadrer canvas pour obtenir la partie carrée (933x933 au centre)
        height, width = canvas_full.shape[:2]
        # Prendre un carré centré de taille height×height
        start_x = (width - height) // 2
        canvas_crop = canvas_full[:, start_x:start_x+height]
        
        # Faire un flip vertical pour correspondre au système de coordonnées Dota (origine en bas à gauche)
        self.canvas = np.flipud(canvas_crop)
        
        # Trouver le tick min et max de la partie
        self.tick_min = float('inf')
        self.tick_max = 0
        for player in self.data['players']:
            for segment in player['segments']:
                self.tick_min = min(self.tick_min, segment['start']['tick'])
                self.tick_max = max(self.tick_max, segment['end']['tick'])
        
        # Tick actuel (initialiser à la fin)
        self.current_tick = self.tick_max
        
        # Paramètres initiaux
        self.params = {
            'canvas_x_min': 0,
            'canvas_x_max': 256,
            'canvas_y_min': 0,
            'canvas_y_max': 256,
            'traj_scale': 1.0,
            'traj_offset_x': 0,
            'traj_offset_y': 0,
            'linewidth': 0.8,
            'alpha': 0.7,
            'canvas_alpha': 1.0
        }
        
        # Créer la figure
        self.setup_figure()
        
    def setup_figure(self):
        """Configure la figure avec les sliders."""
        # La map est carrée (256x256), donc figure carrée
        self.fig = plt.figure(figsize=(16, 16))
        
        # Axes principal pour la visualisation
        self.ax = plt.axes([0.05, 0.15, 0.9, 0.8])
        
        # Slider pour le temps
        ax_time = plt.axes([0.15, 0.05, 0.7, 0.03])
        self.slider_time = Slider(
            ax_time, 
            'Temps (tick)', 
            self.tick_min, 
            self.tick_max, 
            valinit=self.tick_max,
            valstep=(self.tick_max - self.tick_min) / 1000
        )
        self.slider_time.on_changed(self.on_time_change)
        
        # Variables pour le zoom et le pan
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.press = None
        
        # Connecter les événements
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Dessiner initial
        self.draw_visualization()
    
    def on_time_change(self, val):
        """Mise à jour quand le slider de temps change."""
        self.current_tick = val
        self.draw_visualization()
    
    def on_scroll(self, event):
        """Zoom avec la molette de la souris."""
        if event.inaxes != self.ax:
            return
        
        # Facteur de zoom
        if event.button == 'up':
            scale_factor = 1.2
        elif event.button == 'down':
            scale_factor = 0.8
        else:
            return
        
        self.zoom_level *= scale_factor
        self.zoom_level = max(0.5, min(10.0, self.zoom_level))  # Limiter le zoom
        self.update_view()
    
    def on_press(self, event):
        """Début du déplacement."""
        if event.inaxes != self.ax or event.button != 1:
            return
        self.press = (event.xdata, event.ydata)
    
    def on_release(self, event):
        """Fin du déplacement."""
        self.press = None
    
    def on_motion(self, event):
        """Déplacement de la vue."""
        if self.press is None or event.inaxes != self.ax:
            return
        
        dx = event.xdata - self.press[0]
        dy = event.ydata - self.press[1]
        
        self.pan_x -= dx
        self.pan_y -= dy
        self.press = (event.xdata, event.ydata)
        self.update_view()
    
    def on_key(self, event):
        """Raccourcis clavier."""
        if event.key == 'r':  # Reset
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.update_view()
        elif event.key == 's':  # Save
            self.save()
    
    def update_view(self):
        """Met à jour la vue avec le zoom et le pan."""
        half_range = 128 / self.zoom_level
        x_center = 128 + self.pan_x
        y_center = 128 + self.pan_y
        
        self.ax.set_xlim(x_center - half_range, x_center + half_range)
        self.ax.set_ylim(y_center - half_range, y_center + half_range)
        self.fig.canvas.draw_idle()
        
    def draw_visualization(self):
        """Dessine la visualisation."""
        self.ax.clear()
        
        # Afficher l'image de fond
        self.ax.imshow(self.canvas, 
                      extent=[0, 256, 0, 256],
                      origin='lower', zorder=0, aspect='equal', alpha=0.8)
        
        # Dessiner les trajectoires jusqu'au tick actuel
        for idx, player in enumerate(self.data['players']):
            color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
            player_id = player['player_id']
            
            # Position actuelle du joueur
            current_x, current_y = None, None
            
            # Dessiner chaque segment jusqu'au tick actuel
            for segment in player['segments']:
                # Si le segment commence après le tick actuel, on arrête
                if segment['start']['tick'] > self.current_tick:
                    break
                
                x1 = segment['start']['x']
                y1 = segment['start']['y']
                
                # Si le segment se termine après le tick actuel, on interpole
                if segment['end']['tick'] > self.current_tick:
                    # Interpolation linéaire
                    t = (self.current_tick - segment['start']['tick']) / (segment['end']['tick'] - segment['start']['tick'])
                    x2 = x1 + t * (segment['end']['x'] - x1)
                    y2 = y1 + t * (segment['end']['y'] - y1)
                    current_x, current_y = x2, y2
                else:
                    x2 = segment['end']['x']
                    y2 = segment['end']['y']
                    current_x, current_y = x2, y2
                
                # Dessiner le segment
                arrow = FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    arrowstyle='->', 
                    color=color,
                    linewidth=0.8,
                    alpha=0.7,
                    mutation_scale=8,
                    zorder=2
                )
                self.ax.add_patch(arrow)
                
                # Si on a atteint le tick actuel, on arrête
                if segment['end']['tick'] > self.current_tick:
                    break
            
            # Point de départ
            if player['segments']:
                first_segment = player['segments'][0]
                if first_segment['start']['tick'] <= self.current_tick:
                    self.ax.plot(first_segment['start']['x'], 
                           first_segment['start']['y'],
                           'o', color=color, markersize=5, 
                           markeredgecolor='white', markeredgewidth=1.0,
                           zorder=3)
            
            # Position actuelle (plus grosse)
            if current_x is not None:
                self.ax.plot(current_x, current_y,
                       'o', color=color, markersize=8, 
                       markeredgecolor='white', markeredgewidth=2.0,
                       label=f"J{player_id}",
                       zorder=4)
        
        # Configuration
        self.ax.set_xlim(0, 256)
        self.ax.set_ylim(0, 256)
        self.ax.set_aspect('equal')
        
        # Convertir tick en temps (environ 30 ticks/seconde)
        time_sec = (self.current_tick - self.tick_min) / 30
        time_min = int(time_sec // 60)
        time_sec_remainder = int(time_sec % 60)
        
        self.ax.set_title(
            f"Match {self.game_id} - w_error={self.w_error} | Temps: {time_min}:{time_sec_remainder:02d}\n"
            f"Molette: Zoom | Clic-glisser: Déplacer | R: Reset | S: Save", 
            fontsize=12, fontweight='bold')
        self.ax.legend(loc='upper right', framealpha=0.9, fontsize=8, ncol=2)
        self.ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        self.fig.canvas.draw_idle()
    
    def save(self):
        """Sauvegarde l'image."""
        output_dir = BASE_DIR / "overlays"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{self.game_id}_w{self.w_error}_overlay.png"
        
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Image sauvegardée: {output_path}")
        
    def show(self):
        """Affiche la fenêtre interactive."""
        plt.show()


def main():
    """Point d'entrée."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python overlay_interactive.py <w_error> <game_id>")
        print("Exemple: python overlay_interactive.py 12 3841740022")
        sys.exit(1)
    
    w_error = float(sys.argv[1])
    game_id = sys.argv[2]
    
    print("=" * 60)
    print("🎨 VISUALISATION INTERACTIVE")
    print("=" * 60)
    print(f"w_error: {w_error}")
    print(f"game_id: {game_id}")
    print()
    print("📝 Contrôles:")
    print("   - Slider en bas : Avancer/reculer dans le temps de la partie")
    print("   - Molette : Zoom in/out")
    print("   - Clic gauche + glisser : Déplacer la vue")
    print("   - Touche R : Réinitialiser la vue")
    print("   - Touche S : Sauvegarder l'image")
    print("=" * 60)
    
    try:
        overlay = InteractiveOverlay(w_error, game_id)
        overlay.show()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
