#!/usr/bin/env python3
"""
Script pour compresser tous les fichiers CSV de data-dota/ avec un taux de compression spécifique.
Usage: python compress_single.py <w_error>
Exemple: python compress_single.py 15
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import pandas as pd

from dota_analytics.controllers import process_full_match
from dota_analytics.views import JSONExporter

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data-dota"
OUTPUT_BASE_DIR = BASE_DIR / "compressed_data"


def compress_single_file(args):
    """
    Compresse un fichier CSV avec le w_error donné.
    
    Args:
        args: Tuple (csv_path, w_error, output_dir, index, total)
    
    Returns:
        tuple: (success: bool, match_id: str, stats: dict)
    """
    csv_path, w_error, output_dir, index, total = args
    
    try:
        match_id = csv_path.stem.replace('coord_', '')
        
        # Charger le fichier CSV
        df = pd.read_csv(csv_path)
        
        # Calculer le nombre de points originaux par joueur
        original_points = {}
        for player_id in range(10):
            x_col, y_col = f'x{player_id}', f'y{player_id}'
            if x_col in df.columns:
                mask = (df[x_col] != 0.0) | (df[y_col] != 0.0)
                original_points[player_id] = int(mask.sum())
        
        # Compresser avec MDL
        results = process_full_match(df, match_id, w_error=w_error)
        
        # Sauvegarder avec JSONExporter
        output_file = output_dir / f"{match_id}_compressed.json"
        JSONExporter.export_match(
            results=results,
            match_id=match_id,
            output_path=output_file,
            w_error=w_error,
            original_points=original_points
        )
        
        # Lire le fichier pour extraire les stats
        with open(output_file, 'r') as f:
            result = json.load(f)
        
        # Calculer les statistiques
        total_original = sum(p['num_original_points'] for p in result['players'])
        total_segments = sum(p['num_segments'] for p in result['players'])
        avg_compression = sum(p['compression_rate'] for p in result['players']) / len(result['players'])
        
        stats = {
            'original_points': total_original,
            'segments': total_segments,
            'compression_rate': avg_compression
        }
        
        return (True, match_id, stats)
        
    except Exception as e:
        return (False, csv_path.name, str(e))


def main():
    """Point d'entrée principal."""
    # Vérifier les arguments
    if len(sys.argv) != 2:
        print("Usage: python compress_single.py <w_error>")
        print("Exemple: python compress_single.py 15")
        sys.exit(1)
    
    try:
        w_error = float(sys.argv[1])
    except ValueError:
        print("❌ Erreur: w_error doit être un nombre")
        sys.exit(1)
    
    # Vérifier que le dossier data-dota existe
    if not DATA_DIR.exists():
        print(f"❌ Erreur: Le dossier {DATA_DIR} n'existe pas")
        sys.exit(1)
    
    # Lister tous les fichiers CSV
    csv_files = sorted(DATA_DIR.glob("coord_*.csv"))
    
    if not csv_files:
        print(f"❌ Erreur: Aucun fichier CSV trouvé dans {DATA_DIR}")
        sys.exit(1)
    
    # Déterminer le nombre de workers
    num_workers = cpu_count()
    
    print("=" * 70)
    print(f"COMPRESSION MDL - w_error = {w_error}")
    print("=" * 70)
    print(f"📁 Dossier source: {DATA_DIR}")
    print(f"📊 Nombre de fichiers: {len(csv_files)}")
    print(f"⚙️  Nombre de workers: {num_workers}")
    print()
    
    # Créer le dossier de sortie
    if w_error == int(w_error):
        output_dir = OUTPUT_BASE_DIR / f"w_error_{int(w_error)}"
    else:
        output_dir = OUTPUT_BASE_DIR / f"w_error_{w_error}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Dossier de sortie: {output_dir}")
    print()
    
    # Préparer les arguments pour le pool
    total_files = len(csv_files)
    tasks = [(csv_path, w_error, output_dir, i+1, total_files) 
             for i, csv_path in enumerate(csv_files)]
    
    # Compresser en parallèle
    start_time = datetime.now()
    success_count = 0
    fail_count = 0
    
    print("🔄 Compression en cours (parallèle)...")
    print("-" * 70)
    
    with Pool(processes=num_workers) as pool:
        for success, identifier, data in pool.imap_unordered(compress_single_file, tasks):
            if success:
                success_count += 1
                match_id = identifier
                stats = data
                print(f"✓ [{success_count + fail_count}/{total_files}] {match_id}: "
                      f"{stats['original_points']} pts → {stats['segments']} seg "
                      f"(compression: {stats['compression_rate']:.2f}%)")
            else:
                fail_count += 1
                filename = identifier
                error = data
                print(f"✗ [{success_count + fail_count}/{total_files}] {filename}: Erreur - {error}")
    
    # Résumé
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print()
    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print(f"✅ Succès: {success_count}/{len(csv_files)}")
    print(f"❌ Échecs: {fail_count}/{len(csv_files)}")
    print(f"⏱️  Durée: {duration:.2f}s")
    print(f"📁 Sortie: {output_dir}")
    print("=" * 70)
    
    # Afficher les fichiers créés
    json_files = sorted(output_dir.glob("*_compressed.json"))
    total_size = sum(f.stat().st_size for f in json_files)
    print(f"📦 {len(json_files)} fichiers JSON créés ({total_size / 1024 / 1024:.2f} MB)")
    print()


if __name__ == "__main__":
    main()
