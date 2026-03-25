"""
MenuPage — Dashboard d'accueil avec métriques et accès rapide aux modules.
"""

import json

import customtkinter as ctk

from mvc.config import CLUSTERS_DIR, COMPRESSED_DIR, DATA_DIR, OUTPUT_DIR
from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, ACCENT2, BG_CARD, TEXT_DIM, TEXT_LIGHT


class MenuPage(BasePage):
    """Dashboard d'accueil avec métriques clés et navigation."""

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._metric_labels = {}
        self._build()

    def on_show(self):
        self._refresh_metrics()

    def _build(self):
        # ── Scrollable pour petits écrans ──
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=0, pady=0)

        title = ctk.CTkLabel(
            scroll,
            text="Dota 2 Trajectory Analyzer",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=ACCENT,
        )
        title.pack(pady=(30, 4))

        ctk.CTkLabel(
            scroll,
            text="Dashboard — Vue d'ensemble du pipeline d'analyse",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_DIM,
        ).pack(pady=(0, 20))

        # ── Barre de métriques ───────────────────────────────────────────
        metrics_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=30, pady=(0, 15))

        metric_defs = [
            ("matches_csv", "Matchs CSV", "0"),
            ("matches_compressed", "Matchs compressés", "0"),
            ("clusters_k", "Clusters (k)", "—"),
            ("patterns", "Motifs fréquents", "—"),
        ]

        for i, (key, label, default) in enumerate(metric_defs):
            card = ctk.CTkFrame(
                metrics_frame, fg_color=BG_CARD, corner_radius=12, height=80
            )
            card.pack(side="left", fill="x", expand=True, padx=6)
            card.pack_propagate(False)

            val_label = ctk.CTkLabel(
                card,
                text=default,
                font=ctk.CTkFont(size=28, weight="bold"),
                text_color=ACCENT,
            )
            val_label.pack(pady=(14, 0))
            ctk.CTkLabel(
                card,
                text=label,
                font=ctk.CTkFont(size=11),
                text_color=TEXT_DIM,
            ).pack(pady=(0, 8))
            self._metric_labels[key] = val_label

        # ── Cartes de navigation ─────────────────────────────────────────
        cards_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        cards_frame.pack(fill="x", padx=30, pady=(5, 10))

        modes = [
            (
                "Overlay Carte",
                "Visualiser les trajectoires\ncompressées sur la carte Dota 2",
                "overlay",
            ),
            (
                "Compression",
                "Lancer la compression MDL\nsur un ou tous les matchs",
                "compress",
            ),
            (
                "Clusters",
                "Visualiser les clusters\nde segments sur la carte",
                "cluster",
            ),
            (
                "Comparaison",
                "Brut vs Compressé\ncôte à côte avec animation",
                "comparison",
            ),
            (
                "PrefixSpan",
                "Fouille de motifs séquentiels\net visualisation des patterns",
                "mining",
            ),
        ]

        for i, (t, desc, pg) in enumerate(modes):
            row = i // 3
            col = i % 3
            card = ctk.CTkFrame(
                cards_frame, fg_color=BG_CARD, corner_radius=16, width=240, height=180
            )
            card.grid(row=row, column=col, padx=12, pady=8)
            card.grid_propagate(False)

            ctk.CTkLabel(
                card,
                text=t,
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=TEXT_LIGHT,
            ).pack(pady=(25, 8))
            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(size=12),
                text_color=TEXT_DIM,
                justify="center",
            ).pack(pady=(0, 10))
            ctk.CTkButton(
                card,
                text="Ouvrir →",
                fg_color=ACCENT,
                hover_color="#c33750",
                command=lambda p=pg: self.switch_page(p),
            ).pack(pady=(3, 15))

    # ── Refresh métriques ────────────────────────────────────────────────

    def _refresh_metrics(self):
        # Matchs CSV
        csv_count = len(list(DATA_DIR.glob("coord_*.csv")))
        self._metric_labels["matches_csv"].configure(text=str(csv_count))

        # Matchs compressés
        comp_count = 0
        if COMPRESSED_DIR.exists():
            for w_dir in COMPRESSED_DIR.iterdir():
                if w_dir.is_dir() and w_dir.name.startswith("w_error_"):
                    comp_count = max(
                        comp_count, len(list(w_dir.glob("*_compressed.json")))
                    )
        self._metric_labels["matches_compressed"].configure(text=str(comp_count))

        # Clusters k
        k_val = "—"
        if CLUSTERS_DIR.exists():
            cluster_files = list(CLUSTERS_DIR.glob("clusters_result_*.json"))
            if cluster_files:
                try:
                    with open(cluster_files[-1]) as f:
                        data = json.load(f)
                    labels = set()
                    for segs in data.values():
                        for lbl in segs.values():
                            labels.add(int(lbl))
                    k_val = str(len(labels))
                except Exception:
                    pass
        self._metric_labels["clusters_k"].configure(text=k_val)

        # Motifs PrefixSpan
        patterns_val = "—"
        patterns_file = OUTPUT_DIR / "patterns.spmf"
        if patterns_file.exists():
            try:
                line_count = sum(1 for _ in open(patterns_file, encoding="utf-8"))
                patterns_val = str(line_count)
            except Exception:
                pass
        self._metric_labels["patterns"].configure(text=patterns_val)
