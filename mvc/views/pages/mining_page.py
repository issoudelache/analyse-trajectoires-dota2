"""
MiningPage — Page de recodage et PrefixSpan (fouille de motifs sequentiels).
"""

import io
from typing import List, Tuple

import customtkinter as ctk
from PIL import Image, ImageTk

from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, BG_CARD, TEXT_DIM


class MiningPage(BasePage):
    """Page de recodage des clusters et fouille PrefixSpan."""

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._all_patterns: List[Tuple[Tuple[int, ...], int]] = []
        self._graph_image = None
        self._build()

    def _build(self):
        # Scrollable container
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            self.scroll,
            text="Recodage & PrefixSpan",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=ACCENT,
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            self.scroll,
            text="Recodage des clusters en sequences puis fouille de motifs frequents",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM,
        ).pack(pady=(0, 15))

        # === Section Recodage ===
        recode_frame = ctk.CTkFrame(self.scroll, fg_color=BG_CARD, corner_radius=12)
        recode_frame.pack(padx=30, pady=8, fill="x")

        ctk.CTkLabel(
            recode_frame,
            text="1. Recodage des clusters",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(12, 8), padx=20, anchor="w")

        row_w = ctk.CTkFrame(recode_frame, fg_color="transparent")
        row_w.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_w, text="w_error:", width=100, anchor="w").pack(side="left")
        self.w_combo = ctk.CTkComboBox(row_w, width=120, values=["12"])
        self.w_combo.pack(side="left", padx=10)

        self.recode_btn = ctk.CTkButton(
            recode_frame,
            text="Generer Sequences",
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._on_recode,
            width=180,
        )
        self.recode_btn.pack(pady=12)

        self.recode_status = ctk.CTkLabel(
            recode_frame, text="", font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.recode_status.pack(pady=(0, 12))

        # === Section PrefixSpan ===
        mining_frame = ctk.CTkFrame(self.scroll, fg_color=BG_CARD, corner_radius=12)
        mining_frame.pack(padx=30, pady=8, fill="x")

        ctk.CTkLabel(
            mining_frame,
            text="2. PrefixSpan (Parametres)",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(12, 8), padx=20, anchor="w")

        params_grid = ctk.CTkFrame(mining_frame, fg_color="transparent")
        params_grid.pack(fill="x", padx=20, pady=5)

        # Row 1: min_support, max_length
        row1 = ctk.CTkFrame(params_grid, fg_color="transparent")
        row1.pack(fill="x", pady=3)

        ctk.CTkLabel(row1, text="min_support:", width=100, anchor="w").pack(side="left")
        self.support_entry = ctk.CTkEntry(row1, width=80)
        self.support_entry.pack(side="left", padx=(5, 20))
        self.support_entry.insert(0, "10")

        ctk.CTkLabel(row1, text="max_length:", width=100, anchor="w").pack(side="left")
        self.maxlen_entry = ctk.CTkEntry(row1, width=80)
        self.maxlen_entry.pack(side="left", padx=5)
        self.maxlen_entry.insert(0, "8")

        self.mine_btn = ctk.CTkButton(
            mining_frame,
            text="Lancer PrefixSpan",
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._on_mine,
            width=180,
        )
        self.mine_btn.pack(pady=12)

        self.mine_status = ctk.CTkLabel(
            mining_frame, text="", font=ctk.CTkFont(size=11), text_color=TEXT_DIM
        )
        self.mine_status.pack(pady=(0, 12))

        # === Section Resultats avec Tabview ===
        results_frame = ctk.CTkFrame(self.scroll, fg_color=BG_CARD, corner_radius=12)
        results_frame.pack(padx=30, pady=8, fill="both", expand=True)

        header_row = ctk.CTkFrame(results_frame, fg_color="transparent")
        header_row.pack(fill="x", padx=20, pady=(12, 5))

        ctk.CTkLabel(
            header_row,
            text="3. Resultats",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left")

        # Filtres
        filter_frame = ctk.CTkFrame(header_row, fg_color="transparent")
        filter_frame.pack(side="right")

        ctk.CTkLabel(filter_frame, text="Afficher top:", width=80).pack(side="left")
        self.top_n_combo = ctk.CTkComboBox(
            filter_frame,
            width=80,
            values=["10", "20", "50", "100", "Tous"],
            command=self._on_filter_change,
        )
        self.top_n_combo.pack(side="left", padx=5)
        self.top_n_combo.set("20")

        ctk.CTkLabel(filter_frame, text="Longueur min:", width=90).pack(side="left", padx=(15, 0))
        self.min_len_combo = ctk.CTkComboBox(
            filter_frame,
            width=60,
            values=["1", "2", "3", "4", "5"],
            command=self._on_filter_change,
        )
        self.min_len_combo.pack(side="left", padx=5)
        self.min_len_combo.set("1")

        # Tabview pour les differentes vues
        self.tabview = ctk.CTkTabview(results_frame, fg_color="#1e1e2e", corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self.tabview.add("Tableau")
        self.tabview.add("Frequences")
        self.tabview.add("Graphe Transitions")

        # Tab 1: Tableau des resultats
        self.results_text = ctk.CTkTextbox(
            self.tabview.tab("Tableau"),
            fg_color="#1a1a2e",
            corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 2: Graphique frequences
        self.freq_canvas_frame = ctk.CTkFrame(
            self.tabview.tab("Frequences"), fg_color="#1a1a2e", corner_radius=8
        )
        self.freq_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.freq_label = ctk.CTkLabel(
            self.freq_canvas_frame,
            text="Lancez PrefixSpan pour voir le graphique des frequences",
            text_color=TEXT_DIM,
        )
        self.freq_label.pack(expand=True)

        # Tab 3: Graphe de transitions
        self.graph_frame = ctk.CTkFrame(
            self.tabview.tab("Graphe Transitions"), fg_color="#1a1a2e", corner_radius=8
        )
        self.graph_frame.pack(fill="both", expand=True, padx=5, pady=5)

        graph_controls = ctk.CTkFrame(self.graph_frame, fg_color="transparent")
        graph_controls.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(graph_controls, text="Longueur min motifs:").pack(side="left")
        self.graph_min_len = ctk.CTkComboBox(
            graph_controls, width=60, values=["2", "3", "4", "5"]
        )
        self.graph_min_len.pack(side="left", padx=5)
        self.graph_min_len.set("2")

        self.graph_btn = ctk.CTkButton(
            graph_controls,
            text="Generer Graphe",
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._on_generate_graph,
            width=140,
        )
        self.graph_btn.pack(side="left", padx=15)

        self.graph_canvas_label = ctk.CTkLabel(
            self.graph_frame,
            text="Lancez PrefixSpan puis generez le graphe de transitions",
            text_color=TEXT_DIM,
        )
        self.graph_canvas_label.pack(expand=True, fill="both", padx=10, pady=10)

    def on_show(self):
        w_errors = self.controller.get_available_w_errors()
        if w_errors:
            self.w_combo.configure(values=[str(w) for w in w_errors])
            self.w_combo.set(str(w_errors[0]))
        else:
            self.w_combo.configure(values=["12"])
            self.w_combo.set("12")

    def _on_recode(self):
        try:
            w = float(self.w_combo.get())
        except ValueError:
            self.recode_status.configure(text="Valeur w_error invalide")
            return

        self.recode_btn.configure(state="disabled", text="En cours...")
        self.recode_status.configure(text="")
        self.controller.start_recoding(w)

    def _on_mine(self):
        try:
            min_sup = int(self.support_entry.get())
            max_len = int(self.maxlen_entry.get())
        except ValueError:
            self.mine_status.configure(text="Parametres invalides")
            return

        self.mine_btn.configure(state="disabled", text="Calcul...")
        self.mine_status.configure(text="Calcul en cours (mode parallele)...")
        self.results_text.delete("1.0", "end")
        self._all_patterns = []
        self.controller.start_mining(min_sup, max_len)

    def _on_filter_change(self, _=None):
        """Met a jour l'affichage selon les filtres."""
        self._update_table_display()
        self._update_freq_chart()

    def _on_generate_graph(self):
        """Genere le graphe de transitions."""
        if not self._all_patterns:
            return

        try:
            min_len = int(self.graph_min_len.get())
        except ValueError:
            min_len = 2

        self.graph_btn.configure(state="disabled", text="Generation...")
        self.controller.generate_transition_graph(self._all_patterns, min_len)

    def _get_filtered_patterns(self) -> List[Tuple[Tuple[int, ...], int]]:
        """Retourne les patterns filtres selon les criteres."""
        if not self._all_patterns:
            return []

        try:
            min_len = int(self.min_len_combo.get())
        except ValueError:
            min_len = 1

        filtered = [p for p in self._all_patterns if len(p[0]) >= min_len]

        top_n_str = self.top_n_combo.get()
        if top_n_str != "Tous":
            try:
                top_n = int(top_n_str)
                filtered = filtered[:top_n]
            except ValueError:
                pass

        return filtered

    def _update_table_display(self):
        """Met a jour le tableau avec les patterns filtres."""
        self.results_text.delete("1.0", "end")

        if not self._all_patterns:
            return

        filtered = self._get_filtered_patterns()
        total = len(self._all_patterns)

        self.results_text.insert("end", f"Total: {total} motifs | Affiches: {len(filtered)}\n")
        self.results_text.insert("end", "=" * 60 + "\n\n")

        # Header
        self.results_text.insert("end", f"{'Rang':<6} {'Support':<10} {'Longueur':<10} {'Motif'}\n")
        self.results_text.insert("end", "-" * 60 + "\n")

        for i, (pattern, support) in enumerate(filtered, 1):
            pattern_str = " -> ".join(str(x) for x in pattern)
            self.results_text.insert(
                "end", f"{i:<6} {support:<10} {len(pattern):<10} [{pattern_str}]\n"
            )

    def _update_freq_chart(self):
        """Met a jour le graphique des frequences."""
        # Clear previous
        for widget in self.freq_canvas_frame.winfo_children():
            widget.destroy()

        if not self._all_patterns:
            self.freq_label = ctk.CTkLabel(
                self.freq_canvas_frame,
                text="Lancez PrefixSpan pour voir le graphique",
                text_color=TEXT_DIM,
            )
            self.freq_label.pack(expand=True)
            return

        filtered = self._get_filtered_patterns()[:30]  # Max 30 pour lisibilité

        if not filtered:
            ctk.CTkLabel(
                self.freq_canvas_frame,
                text="Aucun motif ne correspond aux filtres",
                text_color=TEXT_DIM,
            ).pack(expand=True)
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a1a2e")
            ax.set_facecolor("#1a1a2e")

            labels = [" -> ".join(str(x) for x in p[0]) for p in filtered]
            supports = [p[1] for p in filtered]

            # Truncate labels if too long
            labels = [l[:25] + "..." if len(l) > 25 else l for l in labels]

            bars = ax.barh(range(len(labels)), supports, color="#e63950", alpha=0.85)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9, color="white")
            ax.set_xlabel("Support", color="white", fontsize=11)
            ax.set_title("Distribution des Supports", color="white", fontsize=13, fontweight="bold")
            ax.tick_params(colors="white")
            ax.invert_yaxis()

            for spine in ax.spines.values():
                spine.set_color("#444")

            ax.grid(axis="x", alpha=0.3, color="#666")

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, self.freq_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            plt.close(fig)

        except ImportError:
            ctk.CTkLabel(
                self.freq_canvas_frame,
                text="matplotlib requis pour les graphiques",
                text_color=TEXT_DIM,
            ).pack(expand=True)

    # === Callbacks from Controller ===

    def on_recode_done(self, success: bool, num_sequences: int, error_msg: str):
        self.recode_btn.configure(state="normal", text="Generer Sequences")
        if success:
            self.recode_status.configure(
                text=f"OK: {num_sequences} sequences generees (sequences.spmf)"
            )
        else:
            self.recode_status.configure(text=f"Erreur: {error_msg}")

    def on_mining_done(
        self,
        success: bool,
        num_patterns: int,
        all_patterns: List[Tuple[Tuple[int, ...], int]],
        error_msg: str,
    ):
        self.mine_btn.configure(state="normal", text="Lancer PrefixSpan")
        if success:
            self.mine_status.configure(text=f"OK: {num_patterns} motifs trouves")
            self._all_patterns = all_patterns
            self._update_table_display()
            self._update_freq_chart()
        else:
            self.mine_status.configure(text=f"Erreur: {error_msg}")

    def on_graph_generated(self, success: bool, image_bytes: bytes, error_msg: str):
        self.graph_btn.configure(state="normal", text="Generer Graphe")

        # Clear previous
        for widget in self.graph_frame.winfo_children():
            if widget != self.graph_frame.winfo_children()[0]:  # Keep controls
                widget.destroy()

        if success and image_bytes:
            try:
                img = Image.open(io.BytesIO(image_bytes))
                # Resize to fit
                max_w, max_h = 900, 500
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self._graph_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)

                img_label = ctk.CTkLabel(self.graph_frame, image=self._graph_image, text="")
                img_label.pack(expand=True, pady=10)
            except Exception as e:
                ctk.CTkLabel(
                    self.graph_frame,
                    text=f"Erreur affichage: {e}",
                    text_color=TEXT_DIM,
                ).pack(expand=True)
        else:
            ctk.CTkLabel(
                self.graph_frame,
                text=f"Erreur: {error_msg}" if error_msg else "Aucun motif a afficher",
                text_color=TEXT_DIM,
            ).pack(expand=True)
