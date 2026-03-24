#!/usr/bin/env python3
"""
Runner principal — Lance les 4 expériences du benchmark séquentiellement.

Usage :
  python benchmark/run_all.py                    # tout lancer
  python benchmark/run_all.py --exp 0 1          # seulement Exp 0 et 1
  python benchmark/run_all.py --exp 2 --k 25     # Exp 2 avec paramètres
  python benchmark/run_all.py --exp 3 --k 25     # Exp 3 (k* requis)
  python benchmark/run_all.py --plots             # figures seulement

Pré-requis pour Exp 1 et 2 :
  Les données compressées doivent exister dans output/compressed/w_error_12.0/
  Sinon, lancez d'abord : python run.py compress --w_error 12.0 --max_files 30

Pré-requis pour Exp 3 :
  --k doit être fourni (résultat de Exp 1 + Exp 2)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = BASE_DIR / "benchmark"


def run_script(name, args_list):
    """Lance un script Python avec des arguments."""
    script = BENCHMARK_DIR / name
    cmd = [sys.executable, str(script)] + args_list
    print(f"\n{'='*70}")
    print(f"  LANCEMENT : {name} {' '.join(args_list)}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  ERREUR : {name} a retourné le code {result.returncode}")
        print(f"  Vous pouvez relancer ce script pour reprendre (écriture incrémentale).")
        return False

    print(f"\n  ✓ {name} terminé en {elapsed:.0f}s")
    return True


def check_compressed_data():
    """Vérifie que les données compressées w_error=12.0 existent."""
    compressed_dir = BASE_DIR / "output" / "compressed" / "w_error_12.0"
    if not compressed_dir.exists() or not list(compressed_dir.glob("*.json")):
        print("ERREUR : Données compressées introuvables pour w_error=12.0")
        print(f"  Chemin attendu : {compressed_dir}")
        print(f"  Lancez d'abord :")
        print(f"    python run.py compress --w_error 12.0 --max_files 30")
        return False
    n_files = len(list(compressed_dir.glob("*.json")))
    print(f"✓ Données compressées présentes : {n_files} fichiers JSON")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Runner principal du benchmark (4 expériences + figures)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--exp", nargs="*", type=int, default=None,
                        help="Expériences à lancer (0 1 2 3). Défaut: 0 1 2")
    parser.add_argument("--k", type=int, default=None,
                        help="k* pour Exp 3 (requis si Exp 3 est inclus)")
    parser.add_argument("--max_files", type=int, default=30)
    parser.add_argument("--plots", action="store_true",
                        help="Générer uniquement les figures (pas de benchmark)")
    parser.add_argument("--min_support", type=int, default=15,
                        help="min_support pour PrefixSpan (Exp 2 et 3)")
    parser.add_argument("--max_length", type=int, default=5,
                        help="max_length pour PrefixSpan (Exp 2 et 3)")
    args = parser.parse_args()

    t_start = time.perf_counter()

    # Figures seulement
    if args.plots:
        plot_exps = args.exp if args.exp else None
        plot_args = []
        if plot_exps:
            plot_args += ["--exp"] + [str(e) for e in plot_exps]
        run_script("plot_all.py", plot_args)
        return

    exps = args.exp if args.exp is not None else [0, 1, 2]

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║           BENCHMARK DOTA 2 — PLAN DE 4 EXPÉRIENCES         ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Expériences : {exps}                                      ║")
    print(f"║  max_files   : {args.max_files}                                        ║")
    if args.k:
        print(f"║  k*          : {args.k}                                        ║")
    print(f"╚══════════════════════════════════════════════════════════════╝\n")

    # ── Exp 0 ──
    if 0 in exps:
        ok = run_script("exp0_werror.py", [
            "--max_files", str(args.max_files),
        ])
        if not ok:
            print("Arrêt après erreur Exp 0. Relancez pour reprendre.")
            sys.exit(1)

    # ── Exp 1 ──
    if 1 in exps:
        if not check_compressed_data():
            sys.exit(1)
        ok = run_script("exp1_optimal_k.py", [
            "--max_files", str(args.max_files),
        ])
        if not ok:
            print("Arrêt après erreur Exp 1. Relancez pour reprendre.")
            sys.exit(1)

    # ── Exp 2 ──
    if 2 in exps:
        if not check_compressed_data():
            sys.exit(1)
        ok = run_script("exp2_prefixspan.py", [
            "--max_files", str(args.max_files),
            "--min_support", str(args.min_support),
            "--max_length", str(args.max_length),
        ])
        if not ok:
            print("Arrêt après erreur Exp 2. Relancez pour reprendre.")
            sys.exit(1)

    # ── Exp 3 ──
    if 3 in exps:
        if args.k is None:
            print("ERREUR : --k requis pour Exp 3")
            print("  Analysez d'abord les résultats de Exp 1 et Exp 2 pour trouver k*")
            sys.exit(1)
        if not check_compressed_data():
            sys.exit(1)
        ok = run_script("exp3_final_pipeline.py", [
            "--k", str(args.k),
            "--max_files", str(args.max_files),
            "--min_support", str(args.min_support),
            "--max_length", str(args.max_length),
        ])
        if not ok:
            print("Arrêt après erreur Exp 3. Relancez pour reprendre.")
            sys.exit(1)

    # ── Figures ──
    plot_exps = [e for e in exps if e in [0, 1, 2]]
    if plot_exps:
        print(f"\n{'='*70}")
        print(f"  GÉNÉRATION DES FIGURES")
        print(f"{'='*70}")
        run_script("plot_all.py", ["--exp"] + [str(e) for e in plot_exps])

    # ── Résumé ──
    t_total = time.perf_counter() - t_start
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║              BENCHMARK TERMINÉ                              ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Durée totale : {t_total:>8.0f}s ({t_total/60:>6.1f} min)                   ║")
    print(f"║  Résultats :                                                ║")
    print(f"║    output/benchmark_exp0/exp0_results.csv                   ║")
    print(f"║    output/benchmark_exp1/exp1_results.csv                   ║")
    print(f"║    output/benchmark_exp2/exp2_results.csv                   ║")
    print(f"║    output/benchmark_exp3/  (si Exp 3 lancée)                ║")
    print(f"║    output/benchmark_figures/  (toutes les figures)           ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
