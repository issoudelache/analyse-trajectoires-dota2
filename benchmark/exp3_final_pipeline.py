#!/usr/bin/env python3
"""
EXP. 3 — Pipeline final + Analyse qualitative.

Question : Quelles régularités dans les trajectoires sont
           caractéristiques de comportements stratégiques des joueurs ?

Fixé  : w_error=12.0, algo=Affinity Propagation (preference=min),
        max_files=30, min_length=5.0, N=3000, min_support=15, max_length=5,
        seed=42

Pipeline :
  1. Charger segments, tirer 3000
  2. Matrice de similarité TRACLUS
  3. AffinityPropagation(preference=min) → labels + exemplaires
  4. Recoder → séquences SPMF
  5. PrefixSpan → motifs
  6. Graphe de Markov → visualisation

Livrables :
  - fig3_1_clusters_map.png : Carte Dota 2 avec les k* clusters colorés
  - fig3_2_markov_graph.png : Graphe de Markov des transitions fréquentes
  - fig3_3_top_motifs.csv   : Top-10 motifs (motif, support)
  - exp3_summary.json       : Résumé complet (métriques, timings, médoïdes)
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.config import (
    COMPRESSED_DIR, CANVAS_PATH, OUTPUT_EXP3 as OUTPUT_DIR,
    MAX_FILES, MIN_LENGTH, N_SUBSAMPLE_DEFAULT as N_SUBSAMPLE,
    MIN_SUPPORT, MAX_LENGTH as MAX_LENGTH_PS, SEED,
    AP_PREFERENCE, AP_DAMPING, AP_MAX_ITER,
)
from dota_analytics.clustering import load_data, compute_traclus_similarity
from dota_analytics.mining import PrefixSpan
from dota_analytics.recoding import reconstruct_sequences, save_sequences_to_spmf

from sklearn.cluster import AffinityPropagation
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# ═════════════════════════════════════════════════════════════════════════════
# FEATURES (pour métriques sklearn)
# ═════════════════════════════════════════════════════════════════════════════

def segments_to_features(segments):
    feats = np.empty((len(segments), 5), dtype=np.float32)
    for i, s in enumerate(segments):
        feats[i, 0] = (s.start.x + s.end.x) * 0.5
        feats[i, 1] = (s.start.y + s.end.y) * 0.5
        feats[i, 2] = s.end.x - s.start.x
        feats[i, 3] = s.end.y - s.start.y
        feats[i, 4] = s.length()
    return feats


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_clusters_on_map(segments, labels, medoid_indices, k, output_path, canvas_path):
    """Dessine les segments colorés par cluster sur la carte Dota 2."""
    import matplotlib
    import matplotlib.pyplot as plt

    # Charger le canvas et recadrer en carré (comme dans l'appli MVC)
    if canvas_path.exists():
        from PIL import Image
        img = Image.open(str(canvas_path))
        w, h = img.size
        if w > h:
            left = (w - h) // 2
            img = img.crop((left, 0, left + h, h))
        canvas = np.asarray(img)
    else:
        print(f"  AVERTISSEMENT : canvas introuvable ({canvas_path}), fond blanc")
        canvas = np.ones((256, 256, 3))

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(k)

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(canvas, extent=[0, 256, 0, 256], origin="upper", aspect="equal")

    # Dessiner les segments non-médoïdes (fins, très transparents)
    for i, (seg, label) in enumerate(zip(segments, labels)):
        if i in medoid_indices:
            continue
        color = cmap(label % k)
        ax.plot(
            [seg.start.x, seg.end.x], [seg.start.y, seg.end.y],
            color=color, linewidth=0.3, alpha=0.15, zorder=2, solid_capstyle="round",
        )

    # Dessiner les médoïdes par-dessus (épais, opaques)
    for idx in medoid_indices:
        seg = segments[idx]
        label = labels[idx]
        color = cmap(label % k)
        ax.plot(
            [seg.start.x, seg.end.x], [seg.start.y, seg.end.y],
            color=color, linewidth=4, alpha=1.0, zorder=4, solid_capstyle="round",
        )

    # Annoter les médoïdes avec un label clair
    for idx in medoid_indices:
        seg = segments[idx]
        mx = (seg.start.x + seg.end.x) / 2
        my = (seg.start.y + seg.end.y) / 2
        label = labels[idx]
        ax.plot(mx, my, "o", color=cmap(label % k), markersize=14,
                markeredgecolor="white", markeredgewidth=2.5, zorder=5)
        ax.annotate(f"C{label}", (mx, my), fontsize=9, fontweight="bold",
                    color="white", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.85,
                              edgecolor=cmap(label % k), linewidth=2),
                    zorder=6)

    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.axis("off")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  → Figure sauvegardée : {output_path}")


def plot_markov_graph(patterns, min_support, output_path):
    """Génère le graphe de Markov depuis les motifs PrefixSpan."""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    for pattern, support in patterns.items():
        if len(pattern) < 2:
            continue
        for i in range(len(pattern) - 1):
            src, tgt = pattern[i], pattern[i + 1]
            if src == tgt:
                continue
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += support
            else:
                G.add_edge(src, tgt, weight=support)

    if len(G.nodes) == 0:
        print("  Aucun motif multi-étapes pour le graphe de Markov.")
        return

    # Filtrage adaptatif : ne garder que les arêtes au-dessus du percentile 25
    all_weights = [d["weight"] for _, _, d in G.edges(data=True)]
    threshold = float(np.percentile(all_weights, 25))
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < threshold]
    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    node_sizes = [
        min(3000, 300 + 50 * G.in_degree(n, weight="weight")
            + 50 * G.out_degree(n, weight="weight"))
        for n in G.nodes()
    ]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w / max_w) * 5 for w in edge_weights]
    node_colors = [G.degree(n, weight="weight") for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, cmap=plt.cm.YlOrRd,
                           edgecolors="black", linewidths=1.5, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_weights, edge_cmap=plt.cm.Blues,
                           arrowsize=25, alpha=0.7,
                           connectionstyle="arc3,rad=0.15")

    # Légende des poids
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=7)

    ax.set_title("Graphe de Markov — Transitions fréquentes entre clusters",
                 fontsize=18, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figure sauvegardée : {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp. 3 — Pipeline final (AP)")
    parser.add_argument("--k", type=int, default=None,
                        help="Ignoré (AP détermine k automatiquement)")
    parser.add_argument("--max_files", type=int, default=MAX_FILES)
    parser.add_argument("--n_subsample", type=int, default=N_SUBSAMPLE)
    parser.add_argument("--min_support", type=int, default=MIN_SUPPORT)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH_PS)
    parser.add_argument("--compressed_dir", type=str, default=str(COMPRESSED_DIR))
    parser.add_argument("--check", action="store_true",
                        help="Afficher les motifs PrefixSpan depuis le dernier run (sans relancer le pipeline)")
    args = parser.parse_args()

    # ── Mode --check : affiche les motifs à partir des résultats existants ──
    if args.check:
        summary_path = OUTPUT_DIR / "exp3_summary.json"
        spmf_path = OUTPUT_DIR / "sequences_final.spmf"
        if not summary_path.exists():
            print(f"ERREUR : {summary_path} introuvable — lancez d'abord sans --check")
            sys.exit(1)
        d = json.load(open(summary_path))
        nb_seq = d.get("nb_sequences", "?")
        print(f"=== TOP 10 motifs PrefixSpan (AP, N={d.get('nb_segments_sampled','?')}) ===")
        for i, m in enumerate(d["top10_motifs"]):
            motif, sup = m["motif"], m["support"]
            pct = sup / nb_seq * 100 if isinstance(nb_seq, int) else 0
            print(f"  #{i+1}: [{' -> '.join(str(x) for x in motif)}]  support={sup} ({pct:.1f}%)  len={len(motif)}")
        if spmf_path.exists():
            miner = PrefixSpan(min_support=args.min_support, max_length=args.max_length)
            db = miner.load_spmf(str(spmf_path))
            patterns = miner.mine(db, parallel=False)
            multi = [(p, s) for p, s in patterns.items() if len(p) >= 2]
            multi.sort(key=lambda x: -x[1])
            print(f"\n=== TOP 20 motifs de longueur >= 2 ===")
            for i, (p, s) in enumerate(multi[:20]):
                pct = s / nb_seq * 100 if isinstance(nb_seq, int) else 0
                print(f"  #{i+1}: [{' -> '.join(str(x) for x in p)}]  support={s} ({pct:.1f}%)  len={len(p)}")
            print(f"\nTotal motifs: {len(patterns)}, len>=2: {len(multi)}, len>=3: {len([x for x in multi if len(x[0])>=3])}")
        return

    compressed_dir = Path(args.compressed_dir)
    if not compressed_dir.exists():
        print(f"ERREUR : dossier compressé introuvable : {compressed_dir}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"algo": "AffinityPropagation", "seed": SEED, "timings": {}, "metrics": {}}
    t_global = time.perf_counter()

    # ── 1. Charger segments ───────────────────────────────────────────────
    print("Exp. 3 — Pipeline final avec Affinity Propagation")
    print(f"  Chargement des segments depuis {compressed_dir}…")
    segments_all, metadata_all = load_data(
        str(compressed_dir), max_files=args.max_files, min_length=MIN_LENGTH,
    )
    n_total = len(segments_all)
    print(f"  → {n_total} segments chargés")
    summary["nb_segments_total"] = n_total

    # Sous-échantillonnage
    rng = np.random.default_rng(SEED)
    if n_total > args.n_subsample:
        idx = rng.choice(n_total, args.n_subsample, replace=False)
        idx.sort()
        segments = [segments_all[i] for i in idx]
        metadata = [metadata_all[i] for i in idx]
    else:
        segments = segments_all
        metadata = metadata_all
    n_seg = len(segments)
    print(f"  → {n_seg} segments après sous-échantillonnage")
    summary["nb_segments_sampled"] = n_seg

    # ── 2. Matrice de similarité TRACLUS ──────────────────────────────────
    print("  Calcul de la matrice TRACLUS…")
    t0 = time.perf_counter()
    similarity_matrix = compute_traclus_similarity(segments)
    t_traclus = time.perf_counter() - t0
    print(f"  → Matrice {n_seg}×{n_seg} calculée en {t_traclus:.1f}s")
    summary["timings"]["traclus_matrix_s"] = t_traclus

    # ── 3. Affinity Propagation ───────────────────────────────────────────
    print("  Affinity Propagation (preference=-5000)…")
    t0 = time.perf_counter()
    ap = AffinityPropagation(
        affinity="precomputed",
        preference=-5000.0,
        random_state=SEED,
        max_iter=500,
        damping=0.7,
    )
    labels = ap.fit_predict(similarity_matrix)
    medoid_indices = ap.cluster_centers_indices_
    t_ap = time.perf_counter() - t0
    k = len(np.unique(labels))
    n_clusters_eff = k
    print(f"  → {n_clusters_eff} clusters, {len(medoid_indices)} exemplaires, {t_ap:.1f}s")
    summary["timings"]["affinity_propagation_s"] = t_ap
    summary["k"] = k
    summary["n_clusters_effective"] = n_clusters_eff
    summary["medoid_indices"] = [int(m) for m in medoid_indices]

    # Métriques géométriques (sur features normalisées)
    X = segments_to_features(segments)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if n_clusters_eff >= 2:
        summary["metrics"]["silhouette"] = float(silhouette_score(
            X_scaled, labels, sample_size=min(5000, n_seg), random_state=SEED))
        summary["metrics"]["davies_bouldin"] = float(davies_bouldin_score(X_scaled, labels))
        summary["metrics"]["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))

    # ── 4. Recodage → SPMF ───────────────────────────────────────────────
    # ── 4. Recodage → SPMF ───────────────────────────────────────────────
    print("  Recodage en séquences SPMF…")
    t0 = time.perf_counter()
    match_clusters = {}
    for idx, label in enumerate(labels):
        m_id = metadata[idx]["match_id"]
        s_id = metadata[idx]["seg_id"]
        if m_id not in match_clusters:
            match_clusters[m_id] = {}
        match_clusters[m_id][s_id] = int(label)

    sequences = reconstruct_sequences(match_clusters)
    spmf_path = OUTPUT_DIR / "sequences_final.spmf"
    save_sequences_to_spmf(sequences, str(spmf_path))
    t_recode = time.perf_counter() - t0
    nb_sequences = len(sequences)
    print(f"  → {nb_sequences} séquences, {t_recode:.1f}s")
    summary["nb_sequences"] = nb_sequences
    summary["timings"]["recoding_s"] = t_recode

    # ── 5. PrefixSpan ────────────────────────────────────────────────────
    print(f"  PrefixSpan (min_support={args.min_support}, max_length={args.max_length})…")
    t0 = time.perf_counter()
    miner = PrefixSpan(min_support=args.min_support, max_length=args.max_length)
    db = miner.load_spmf(str(spmf_path))
    patterns = miner.mine(db, parallel=True)
    t_ps = time.perf_counter() - t0
    print(f"  → {len(patterns)} motifs, {t_ps:.1f}s")
    summary["timings"]["prefixspan_s"] = t_ps
    summary["nb_motifs"] = len(patterns)

    # Sauvegarder motifs
    patterns_path = OUTPUT_DIR / "patterns_final.spmf"
    miner.save_results_to_spmf(str(patterns_path))

    # ── 6. Top-10 motifs ─────────────────────────────────────────────────
    sorted_patterns = sorted(patterns.items(), key=lambda x: (-x[1], len(x[0])))
    top10 = sorted_patterns[:10]

    top10_path = OUTPUT_DIR / "fig3_3_top_motifs.csv"
    with open(top10_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "motif", "length", "support"])
        for rank, (pattern, support) in enumerate(top10, 1):
            motif_str = " → ".join(str(c) for c in pattern)
            writer.writerow([rank, motif_str, len(pattern), support])
    print(f"  → Top-10 motifs sauvegardé : {top10_path}")

    # Afficher top-10
    print("\n  ╔═══════════════════════════════════════════════════╗")
    print("  ║             TOP-10 MOTIFS FRÉQUENTS              ║")
    print("  ╠════╦═══════════════════════════╦═════╦═══════════╣")
    print("  ║ #  ║ Motif                     ║ Len ║ Support   ║")
    print("  ╠════╬═══════════════════════════╬═════╬═══════════╣")
    for rank, (pattern, support) in enumerate(top10, 1):
        motif_str = " → ".join(str(c) for c in pattern)
        print(f"  ║ {rank:>2} ║ {motif_str:<25} ║  {len(pattern)}  ║ {support:>7}   ║")
    print("  ╚════╩═══════════════════════════╩═════╩═══════════╝\n")

    # ── 7. Figures ────────────────────────────────────────────────────────
    print("  Génération des figures…")

    # Fig 3.1 : Clusters sur carte
    fig31_path = OUTPUT_DIR / "fig3_1_clusters_map.png"
    plot_clusters_on_map(segments, labels, medoid_indices, k, fig31_path, CANVAS_PATH)

    # Fig 3.2 : Graphe de Markov
    fig32_path = OUTPUT_DIR / "fig3_2_markov_graph.png"
    plot_markov_graph(patterns, args.min_support, fig32_path)

    # ── 8. Résumé ─────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_global
    summary["timings"]["total_s"] = t_total
    summary["top10_motifs"] = [
        {"motif": [int(x) for x in p], "support": int(s)} for p, s in top10
    ]

    summary_path = OUTPUT_DIR / "exp3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → Résumé sauvegardé : {summary_path}")

    print(f"\n✓ Exp. 3 terminée en {t_total:.0f}s — résultats dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
