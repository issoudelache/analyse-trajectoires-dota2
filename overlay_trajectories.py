#!/usr/bin/env python3
"""
Script interactif pour superposer les trajectoires compressées sur la carte Dota 2.
Permet de sélectionner un niveau de compression (w_error) et un game ID.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Configuration des chemins
BASE_DIR = Path(__file__).parent
CANVAS_PATH = BASE_DIR / "canvas.png"
EXPORTED_DATA_DIR = BASE_DIR / "exported_data_mvc"

# Couleurs pour les 10 joueurs (5 Radiant + 5 Dire)
PLAYER_COLORS = [
    '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c',  # Radiant (bleu)
    '#16a085', '#27ae60', '#8e44ad', '#d35400', '#c0392b'   # Dire (vert/rouge)
]


def get_available_w_errors():
    """Récupère la liste des valeurs w_error disponibles."""
    w_errors = []
    for d in sorted(EXPORTED_DATA_DIR.iterdir()):
        if d.is_dir() and d.name.startswith('w_error_'):
            w_error_str = d.name.replace('w_error_', '')
            try:
                w_error = float(w_error_str)
                w_errors.append(w_error)
            except ValueError:
                pass
    return sorted(w_errors)


def get_available_games(w_error):
    """Récupère la liste des game IDs disponibles pour un w_error donné."""
    # Normaliser le nom du dossier (enlever .0 pour les entiers)
    if w_error == int(w_error):
        w_error_str = str(int(w_error))
    else:
        w_error_str = str(w_error)
    
    w_error_dir = EXPORTED_DATA_DIR / f"w_error_{w_error_str}"
    if not w_error_dir.exists():
        return []
    
    games = []
    for f in sorted(w_error_dir.glob("*_compressed.json")):
        game_id = f.stem.replace('_compressed', '')
        games.append(game_id)
    return games


def load_compressed_data(w_error, game_id):
    """Charge les données compressées pour un game_id et w_error donnés."""
    # Normaliser le nom du dossier (enlever .0 pour les entiers)
    if w_error == int(w_error):
        w_error_str = str(int(w_error))
    else:
        w_error_str = str(w_error)
    
    json_path = EXPORTED_DATA_DIR / f"w_error_{w_error_str}" / f"{game_id}_compressed.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_trajectory_overlay(w_error, game_id, show_arrows=True, alpha=0.7):
    """
    Crée une visualisation avec les trajectoires compressées superposées sur la carte.
    
    Args:
        w_error: Niveau de compression
        game_id: ID du match
        show_arrows: Afficher les flèches de direction
        alpha: Transparence des trajectoires (0-1)
    """
    # Charger la carte de fond
    if not CANVAS_PATH.exists():
        raise FileNotFoundError(f"Carte non trouvée: {CANVAS_PATH}")
    
    canvas = mpimg.imread(CANVAS_PATH)
    
    # Charger les données compressées
    data = load_compressed_data(w_error, game_id)
    
    # Créer la figure avec le bon aspect ratio (canvas est 1896x933, ratio ~2:1)
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Afficher la carte de fond avec les bonnes proportions
    ax.imshow(canvas, extent=[0, 256, 0, 256], origin='lower', zorder=0, aspect='equal')
    
    # Dessiner les trajectoires pour chaque joueur
    for idx, player in enumerate(data['players']):
        color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
        player_id = player['player_id']
        
        # Dessiner chaque segment
        for segment in player['segments']:
            x1, y1 = segment['start']['x'], segment['start']['y']
            x2, y2 = segment['end']['x'], segment['end']['y']
            
            if show_arrows:
                # Flèche directionnelle plus fine
                arrow = FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    arrowstyle='->', 
                    color=color,
                    linewidth=0.8,
                    alpha=alpha,
                    mutation_scale=10,
                    zorder=2
                )
                ax.add_patch(arrow)
            else:
                # Simple ligne plus fine
                ax.plot([x1, x2], [y1, y2], 
                       color=color, 
                       linewidth=0.8, 
                       alpha=alpha,
                       zorder=2)
        
        # Ajouter le point de départ (cercle plus petit)
        first_segment = player['segments'][0]
        ax.plot(first_segment['start']['x'], 
               first_segment['start']['y'],
               'o', color=color, markersize=6, 
               markeredgecolor='white', markeredgewidth=1.0,
               label=f"Joueur {player_id}",
               zorder=3)
    
    # Configuration de l'affichage
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect('equal')
    ax.set_title(
        f"Match {game_id} - Compression w_error={w_error}\n"
        f"Algorithme: {data['compression_info']['algorithm']}",
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    
    # Légende
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # Grille légère
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    return fig, ax


def interactive_selection():
    """Interface interactive pour sélectionner w_error et game_id."""
    print("=" * 60)
    print("SUPERPOSITION DE TRAJECTOIRES COMPRESSÉES SUR LA CARTE DOTA 2")
    print("=" * 60)
    print()
    
    # Sélection du w_error
    w_errors = get_available_w_errors()
    if not w_errors:
        print("❌ Aucune donnée compressée trouvée!")
        return None, None
    
    print(f"📊 Niveaux de compression disponibles ({len(w_errors)}):")
    print()
    
    # Afficher par groupes
    for i in range(0, len(w_errors), 10):
        group = w_errors[i:i+10]
        print("  " + ", ".join(f"{w:.1f}" for w in group))
    
    print()
    while True:
        try:
            w_error_input = input("🎯 Choisissez un w_error (ou 'q' pour quitter): ").strip()
            if w_error_input.lower() == 'q':
                return None, None
            
            w_error = float(w_error_input)
            if w_error in w_errors:
                break
            else:
                print(f"⚠️  w_error={w_error} n'existe pas. Choisissez parmi la liste.")
        except ValueError:
            print("⚠️  Entrez un nombre valide.")
    
    print()
    
    # Sélection du game_id
    games = get_available_games(w_error)
    if not games:
        print(f"❌ Aucune partie trouvée pour w_error={w_error}")
        return None, None
    
    print(f"🎮 Parties disponibles pour w_error={w_error} ({len(games)}):")
    print()
    
    # Afficher les 20 premiers
    display_games = games[:20]
    for i, game_id in enumerate(display_games, 1):
        print(f"  {i:2d}. {game_id}")
    
    if len(games) > 20:
        print(f"  ... et {len(games) - 20} autres")
    
    print()
    print("💡 Vous pouvez entrer:")
    print("   - Un numéro (ex: 1)")
    print("   - Un game_id complet (ex: 3841740022)")
    print()
    
    while True:
        game_input = input("🎯 Choisissez une partie (ou 'q' pour quitter): ").strip()
        if game_input.lower() == 'q':
            return None, None
        
        # Essayer en tant que numéro
        try:
            idx = int(game_input) - 1
            if 0 <= idx < len(games):
                game_id = games[idx]
                break
        except ValueError:
            pass
        
        # Essayer en tant que game_id direct
        if game_input in games:
            game_id = game_input
            break
        
        print("⚠️  Choix invalide. Réessayez.")
    
    return w_error, game_id


def main():
    """Point d'entrée principal."""
    # Sélection interactive
    w_error, game_id = interactive_selection()
    
    if w_error is None or game_id is None:
        print("\n👋 Au revoir!")
        return
    
    print()
    print("=" * 60)
    print(f"🎨 Génération de la visualisation...")
    print(f"   w_error: {w_error}")
    print(f"   game_id: {game_id}")
    print("=" * 60)
    print()
    
    try:
        # Générer la visualisation
        fig, ax = plot_trajectory_overlay(w_error, game_id, show_arrows=True, alpha=0.7)
        
        # Sauvegarder
        output_dir = BASE_DIR / "overlays"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{game_id}_w{w_error}_overlay.png"
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Image sauvegardée: {output_path}")
        
        # Afficher
        plt.show()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
