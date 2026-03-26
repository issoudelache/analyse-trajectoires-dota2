"""
PipelinePage — page dédiée au pipeline complet :
Compression (parallèle) → Clustering → Recodage → PrefixSpan → Graphes.
"""

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, ACCENT2, BG_CARD, BG_DARK, TEXT_DIM, TEXT_LIGHT


class PipelinePage(BasePage):
    """Page Pipeline avec configuration, progression et résultats scrollables."""

    _STEP_LABELS = [
        "Compression",
        "Clustering",
        "Recodage",
        "PrefixSpan",
        "Graphes",
    ]

    def __init__(self, master, controller, switch_page):
        super().__init__(master, controller)
        self.switch_page = switch_page
        self._step_widgets = []
        self._graph_images = {}  # name -> (PhotoImage, bytes)
        self._running = False
        self._patterns = []  # patterns trouvés par PrefixSpan
        self._result_images = {}  # title -> bytes (pour sauvegarde)
        self._build_ui()

    # ── construction de l'interface ────────────────────────────────────

    def _build_ui(self):
        # ─── Header ──────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        header.pack(fill="x", padx=20, pady=(15, 8))
        ctk.CTkLabel(
            header,
            text="Pipeline Complet",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left", padx=20, pady=12)

        # Boutons Sauvegarder / Charger
        self._save_btn = ctk.CTkButton(
            header, text="💾 Sauvegarder", width=130, height=32,
            fg_color=ACCENT2, hover_color="#1a4a8a",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._save_results, state="disabled",
        )
        self._save_btn.pack(side="right", padx=(4, 20), pady=12)

        ctk.CTkButton(
            header, text="📂 Charger", width=120, height=32,
            fg_color=ACCENT2, hover_color="#1a4a8a",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._load_results,
        ).pack(side="right", padx=4, pady=12)

        # ─── Contenu : Config à gauche / Résultats à droite ─────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        # -- Panneau gauche (config + étapes) --------------------------
        left = ctk.CTkFrame(body, width=310, fg_color=BG_CARD, corner_radius=12)
        left.pack(side="left", fill="y", padx=(0, 10), pady=0)
        left.pack_propagate(False)

        self._build_config_panel(left)
        self._build_steps_panel(left)

        # -- Panneau droit (résultats scrollables) ---------------------
        right = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=12)
        right.pack(side="left", fill="both", expand=True)

        # Barre stratégies en haut du panneau droit
        self._strat_bar = ctk.CTkFrame(right, fg_color=BG_DARK, corner_radius=8)
        self._strat_bar.pack(fill="x", padx=8, pady=(8, 0))

        ctk.CTkLabel(
            self._strat_bar, text="Stratégies trouvées",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_LIGHT,
        ).pack(side="left", padx=(10, 6), pady=6)

        self._strat_var = ctk.StringVar(value="— lancer le pipeline —")
        self._strat_menu = ctk.CTkComboBox(
            self._strat_bar, variable=self._strat_var,
            values=["— lancer le pipeline —"],
            width=400, height=28, state="readonly",
            command=self._on_strategy_selected,
        )
        self._strat_menu.pack(side="left", padx=4, pady=6, fill="x", expand=True)

        self._strat_detail = ctk.CTkLabel(
            self._strat_bar, text="", text_color=ACCENT,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        self._strat_detail.pack(side="right", padx=10, pady=6)

        self._results_scroll = ctk.CTkScrollableFrame(
            right, fg_color="transparent", corner_radius=0
        )
        self._results_scroll.pack(fill="both", expand=True, padx=8, pady=8)

        self._placeholder_lbl = ctk.CTkLabel(
            self._results_scroll,
            text="Lancez le pipeline pour voir les résultats ici.",
            text_color=TEXT_DIM,
            font=ctk.CTkFont(size=13),
        )
        self._placeholder_lbl.pack(pady=40)

    # ── panneau de configuration ─────────────────────────────────────

    def _build_config_panel(self, parent):
        cfg = ctk.CTkFrame(parent, fg_color="transparent")
        cfg.pack(fill="x", padx=14, pady=(14, 6))

        ctk.CTkLabel(
            cfg, text="Configuration", font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(anchor="w")

        # Nombre de matchs
        ctk.CTkLabel(cfg, text="Nombre de matchs (0 = tous)", text_color=TEXT_DIM, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(8, 0))
        self._max_matches_var = ctk.StringVar(value="0")
        ctk.CTkEntry(cfg, textvariable=self._max_matches_var, width=260, height=30).pack(anchor="w", pady=2)

        # w_error
        ctk.CTkLabel(cfg, text="w_error", text_color=TEXT_DIM, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(8, 0))
        self._w_error_var = ctk.StringVar(value="12.0")
        ctk.CTkEntry(cfg, textvariable=self._w_error_var, width=260, height=30).pack(anchor="w", pady=2)

        # Algorithme
        ctk.CTkLabel(cfg, text="Algorithme de clustering", text_color=TEXT_DIM, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(10, 0))
        self._algo_var = ctk.StringVar(value="affinity")
        self._algo_menu = ctk.CTkComboBox(
            cfg, values=["affinity", "kmeans", "kmedoids"],
            variable=self._algo_var, width=260, height=30,
            command=self._on_algo_changed,
        )
        self._algo_menu.pack(anchor="w", pady=2)

        # Frame paramètres dynamiques
        self._params_frame = ctk.CTkFrame(cfg, fg_color="transparent")
        self._params_frame.pack(fill="x", pady=(6, 0))
        self._param_entries = {}
        self._on_algo_changed(self._algo_var.get())

        # Séparateur
        ctk.CTkFrame(cfg, height=1, fg_color=TEXT_DIM).pack(fill="x", pady=10)

        # PrefixSpan params
        ctk.CTkLabel(cfg, text="PrefixSpan", font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_LIGHT).pack(anchor="w")

        row_ps = ctk.CTkFrame(cfg, fg_color="transparent")
        row_ps.pack(fill="x", pady=4)

        ctk.CTkLabel(row_ps, text="min_support", text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left")
        self._min_support_var = ctk.StringVar(value="10")
        ctk.CTkEntry(row_ps, textvariable=self._min_support_var, width=60, height=28).pack(side="left", padx=(6, 12))

        ctk.CTkLabel(row_ps, text="max_len", text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left")
        self._max_length_var = ctk.StringVar(value="8")
        ctk.CTkEntry(row_ps, textvariable=self._max_length_var, width=60, height=28).pack(side="left", padx=6)

        # Bouton lancer
        self._launch_btn = ctk.CTkButton(
            cfg, text="▶  Lancer le Pipeline", fg_color=ACCENT, hover_color="#c73650",
            height=40, corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_launch,
        )
        self._launch_btn.pack(fill="x", pady=(14, 4))

    def _on_algo_changed(self, algo: str):
        for w in self._params_frame.winfo_children():
            w.destroy()
        self._param_entries.clear()

        specs = self._algo_params(algo)
        for name, default in specs:
            row = ctk.CTkFrame(self._params_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=name, text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left")
            var = ctk.StringVar(value=str(default))
            ctk.CTkEntry(row, textvariable=var, width=80, height=28).pack(side="right")
            self._param_entries[name] = var

    @staticmethod
    def _algo_params(algo: str):
        if algo == "kmeans":
            return [("n_clusters", 50), ("max_files", "")]
        elif algo == "affinity":
            return [("damping", 0.9), ("max_iter", 400), ("preference", ""), ("max_files", 5), ("min_length", 5.0)]
        elif algo == "kmedoids":
            return [("n_clusters", 50), ("max_iter", 400), ("max_files", 5), ("min_length", 5.0)]
        return []

    # ── panneau étapes ───────────────────────────────────────────────

    def _build_steps_panel(self, parent):
        steps_frame = ctk.CTkFrame(parent, fg_color="transparent")
        steps_frame.pack(fill="x", padx=14, pady=(6, 14))
        ctk.CTkLabel(
            steps_frame, text="Progression", font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(anchor="w", pady=(0, 6))

        self._step_widgets = []
        for label in self._STEP_LABELS:
            row = ctk.CTkFrame(steps_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            icon_lbl = ctk.CTkLabel(row, text="○", width=22, text_color=TEXT_DIM, font=ctk.CTkFont(size=14))
            icon_lbl.pack(side="left")
            name_lbl = ctk.CTkLabel(row, text=label, text_color=TEXT_DIM, font=ctk.CTkFont(size=12))
            name_lbl.pack(side="left", padx=4)
            info_lbl = ctk.CTkLabel(row, text="", text_color=TEXT_DIM, font=ctk.CTkFont(size=11))
            info_lbl.pack(side="right", padx=4)
            self._step_widgets.append((icon_lbl, name_lbl, info_lbl))

        self._global_progress = ctk.CTkProgressBar(steps_frame, height=6, fg_color=BG_DARK, progress_color=ACCENT)
        self._global_progress.pack(fill="x", pady=(8, 0))
        self._global_progress.set(0)

    # ── lancement ────────────────────────────────────────────────────

    def _on_launch(self):
        if self._running:
            return
        self._running = True
        self._launch_btn.configure(state="disabled", text="⏳ Pipeline en cours…")

        # Reset étapes
        for icon_lbl, name_lbl, info_lbl in self._step_widgets:
            icon_lbl.configure(text="○", text_color=TEXT_DIM)
            name_lbl.configure(text_color=TEXT_DIM)
            info_lbl.configure(text="")
        self._global_progress.set(0)

        # Nettoyer résultats
        for w in self._results_scroll.winfo_children():
            w.destroy()
        self._graph_images.clear()
        self._patterns.clear()
        self._result_images.clear()
        self._save_btn.configure(state="disabled")
        self._strat_menu.configure(values=["⏳ en cours…"])
        self._strat_var.set("⏳ en cours…")
        self._strat_detail.configure(text="")

        # Collecter params
        try:
            w_error = float(self._w_error_var.get())
        except ValueError:
            w_error = 12.0

        algo = self._algo_var.get()
        cluster_kwargs = {}
        for name, var in self._param_entries.items():
            val = var.get().strip()
            if val == "":
                continue
            try:
                cluster_kwargs[name] = int(val) if "." not in val else float(val)
            except ValueError:
                pass

        try:
            min_support = int(self._min_support_var.get())
        except ValueError:
            min_support = 10
        try:
            max_length = int(self._max_length_var.get())
        except ValueError:
            max_length = 8

        self.controller.start_pipeline_page(
            w_error=w_error,
            algo=algo,
            cluster_kwargs=cluster_kwargs,
            min_support=min_support,
            max_length=max_length,
            max_matches=int(self._max_matches_var.get() or 0),
        )

    # ── callbacks du controller ──────────────────────────────────────

    def on_pipeline_progress(self, step: int, total: int, label: str):
        """Appelé quand une étape démarre."""
        idx = step - 1
        if 0 <= idx < len(self._step_widgets):
            icon, name, _ = self._step_widgets[idx]
            icon.configure(text="⏳", text_color=ACCENT)
            name.configure(text_color=TEXT_LIGHT)
        self._global_progress.set(max(0, (step - 1)) / total)

    def on_pipeline_step_result(self, step: int, message: str, kwargs: dict):
        """Appelé quand une étape est terminée."""
        idx = step - 1
        if 0 <= idx < len(self._step_widgets):
            icon, name, info = self._step_widgets[idx]
            icon.configure(text="✓", text_color="#50fa7b")
            name.configure(text_color="#50fa7b")
            info.configure(text=message, text_color=TEXT_LIGHT)

        total = len(self._STEP_LABELS)
        self._global_progress.set(step / total)

        # Afficher les résultats intermédiaires
        if "patterns" in kwargs and kwargs["patterns"]:
            self._patterns = kwargs["patterns"]
            self._populate_strategy_dropdown(kwargs["patterns"])
            self._show_patterns_table(kwargs["patterns"])
        if "graph_bytes" in kwargs and kwargs["graph_bytes"]:
            self._result_images["Graphe des Transitions"] = kwargs["graph_bytes"]
            self._add_graph("Graphe des Transitions", kwargs["graph_bytes"])
        if "map_bytes" in kwargs and kwargs["map_bytes"]:
            self._result_images["Stratégies sur la Carte"] = kwargs["map_bytes"]
            self._add_graph("Stratégies sur la Carte", kwargs["map_bytes"])
        if "freq_bytes" in kwargs and kwargs["freq_bytes"]:
            self._result_images["Fréquence des Motifs (Top 20)"] = kwargs["freq_bytes"]
            self._add_graph("Fréquence des Motifs (Top 20)", kwargs["freq_bytes"])

    def on_pipeline_done(self, success: bool, error_msg: str):
        self._running = False
        self._launch_btn.configure(state="normal", text="▶  Lancer le Pipeline")
        if success:
            self._save_btn.configure(state="normal")
        if not success:
            err_lbl = ctk.CTkLabel(
                self._results_scroll, text=f"❌ Erreur : {error_msg}",
                text_color=ACCENT, font=ctk.CTkFont(size=13),
                wraplength=500,
            )
            err_lbl.pack(pady=10)

    # ── dropdown stratégies ──────────────────────────────────────────

    def _populate_strategy_dropdown(self, patterns):
        """Remplit la liste déroulante avec les stratégies trouvées."""
        items = []
        for i, (pat, sup) in enumerate(patterns[:50]):
            motif = " → ".join(str(c) for c in pat)
            items.append(f"#{i+1}  {motif}  (sup={sup})")
        if not items:
            items = ["Aucune stratégie trouvée"]
        self._strat_menu.configure(values=items)
        self._strat_var.set(items[0])
        self._strat_detail.configure(text=f"{len(patterns)} stratégies")

    def _on_strategy_selected(self, choice: str):
        """Affiche le détail de la stratégie sélectionnée."""
        if not self._patterns or choice.startswith("—") or choice.startswith("Aucune"):
            return
        try:
            idx = int(choice.split("#")[1].split()[0]) - 1
        except (IndexError, ValueError):
            return
        if 0 <= idx < len(self._patterns):
            pat, sup = self._patterns[idx]
            motif = " → ".join(str(c) for c in pat)
            self._strat_detail.configure(
                text=f"Stratégie #{idx+1} — longueur {len(pat)}, support {sup}"
            )

    # ── affichage résultats ──────────────────────────────────────────

    def _show_patterns_table(self, patterns):
        """Affiche les top patterns dans un tableau scrollable."""
        card = ctk.CTkFrame(self._results_scroll, fg_color=BG_DARK, corner_radius=10)
        card.pack(fill="x", pady=(8, 4), padx=4)

        ctk.CTkLabel(
            card, text="Top Motifs (PrefixSpan)",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=ACCENT,
        ).pack(anchor="w", padx=12, pady=(10, 4))

        # En-tête
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.pack(fill="x", padx=12)
        ctk.CTkLabel(hdr, text="#", width=30, text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left")
        ctk.CTkLabel(hdr, text="Motif", text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left", padx=8)
        ctk.CTkLabel(hdr, text="Support", width=60, text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="right", padx=8)

        top = patterns[:20]
        for i, (pattern, support) in enumerate(top):
            row = ctk.CTkFrame(card, fg_color=BG_CARD if i % 2 == 0 else "transparent", corner_radius=4)
            row.pack(fill="x", padx=12, pady=1)
            ctk.CTkLabel(row, text=str(i + 1), width=30, text_color=TEXT_DIM, font=ctk.CTkFont(size=11)).pack(side="left")
            motif_str = " → ".join(str(c) for c in pattern)
            ctk.CTkLabel(row, text=motif_str, text_color=TEXT_LIGHT, font=ctk.CTkFont(size=11)).pack(side="left", padx=8)
            ctk.CTkLabel(row, text=str(support), width=60, text_color=ACCENT, font=ctk.CTkFont(size=11, weight="bold")).pack(side="right", padx=8)

        # Padding bas
        ctk.CTkFrame(card, height=8, fg_color="transparent").pack()

    def _add_graph(self, title: str, image_bytes: bytes):
        """Ajoute un graphe (image bytes) dans la zone de résultats avec export JPG."""
        card = ctk.CTkFrame(self._results_scroll, fg_color=BG_DARK, corner_radius=10)
        card.pack(fill="x", pady=(8, 4), padx=4)

        top_bar = ctk.CTkFrame(card, fg_color="transparent")
        top_bar.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            top_bar, text=title,
            font=ctk.CTkFont(size=14, weight="bold"), text_color=ACCENT,
        ).pack(side="left")

        ctk.CTkButton(
            top_bar, text="💾 Exporter JPG", width=120, height=28,
            fg_color=ACCENT2, hover_color="#1a4a8a",
            font=ctk.CTkFont(size=11),
            command=lambda b=image_bytes, t=title: self._export_graph(b, t),
        ).pack(side="right")

        # Afficher l'image
        pil_img = Image.open(io.BytesIO(image_bytes))
        # Redimensionner pour tenir dans la zone (max ~700px large)
        max_w = 700
        if pil_img.width > max_w:
            ratio = max_w / pil_img.width
            pil_img = pil_img.resize(
                (max_w, int(pil_img.height * ratio)), Image.LANCZOS
            )

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                size=(pil_img.width, pil_img.height))
        img_lbl = ctk.CTkLabel(card, image=ctk_img, text="")
        img_lbl.pack(padx=12, pady=(4, 10))

        # Garder une référence pour éviter le garbage collection
        self._graph_images[title] = (ctk_img, image_bytes)

    def _export_graph(self, image_bytes: bytes, title: str):
        safe_name = title.replace(" ", "_").replace("(", "").replace(")", "")
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"{safe_name}.jpg",
        )
        if not path:
            return
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img.save(path, quality=95)

    # ── sauvegarde / chargement des résultats ────────────────────────────

    def _save_results(self):
        """Sauvegarde patterns + images dans un fichier JSON."""
        if not self._patterns and not self._result_images:
            return

        # Collecter la config actuelle
        config = {
            "w_error": self._w_error_var.get(),
            "algo": self._algo_var.get(),
            "max_matches": self._max_matches_var.get(),
            "min_support": self._min_support_var.get(),
            "max_length": self._max_length_var.get(),
        }
        for name, var in self._param_entries.items():
            config[name] = var.get()

        data = {
            "saved_at": datetime.now().isoformat(),
            "config": config,
            "patterns": [[pat, sup] for pat, sup in self._patterns],
            "images": {
                title: base64.b64encode(img_bytes).decode("ascii")
                for title, img_bytes in self._result_images.items()
            },
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"pipeline_{ts}.json",
            initialdir=str(Path("output")),
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"✓ Résultats sauvegardés : {path}")

    def _load_results(self):
        """Charge des résultats pipeline depuis un fichier JSON."""
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")],
            initialdir=str(Path("output")),
        )
        if not path:
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Restaurer la config
        config = data.get("config", {})
        if "w_error" in config:
            self._w_error_var.set(config["w_error"])
        if "algo" in config:
            self._algo_var.set(config["algo"])
            self._on_algo_changed(config["algo"])
        if "max_matches" in config:
            self._max_matches_var.set(config["max_matches"])
        if "min_support" in config:
            self._min_support_var.set(config["min_support"])
        if "max_length" in config:
            self._max_length_var.set(config["max_length"])
        # Restaurer les params dynamiques de l'algo
        for name, var in self._param_entries.items():
            if name in config:
                var.set(config[name])

        # Nettoyer l'affichage
        for w in self._results_scroll.winfo_children():
            w.destroy()
        self._graph_images.clear()
        self._patterns.clear()
        self._result_images.clear()

        # Marquer toutes les étapes comme terminées
        for icon_lbl, name_lbl, info_lbl in self._step_widgets:
            icon_lbl.configure(text="✓", text_color="#50fa7b")
            name_lbl.configure(text_color="#50fa7b")
            info_lbl.configure(text="(chargé)")
        self._global_progress.set(1.0)

        # Restaurer les patterns
        patterns = [(pat, sup) for pat, sup in data.get("patterns", [])]
        if patterns:
            self._patterns = patterns
            self._populate_strategy_dropdown(patterns)
            self._show_patterns_table(patterns)

        # Restaurer les images
        images = data.get("images", {})
        for title, b64 in images.items():
            img_bytes = base64.b64decode(b64)
            self._result_images[title] = img_bytes
            self._add_graph(title, img_bytes)

        self._save_btn.configure(state="normal")

        saved_at = data.get("saved_at", "?")
        print(f"✓ Résultats chargés depuis : {path} (sauvé le {saved_at})")
