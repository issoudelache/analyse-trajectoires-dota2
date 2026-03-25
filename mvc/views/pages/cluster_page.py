"""
ClusterPage — Visualisation d'un cluster sur la carte.
"""

import customtkinter as ctk

from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, ACCENT2, BG_CARD, TEXT_DIM, TEXT_LIGHT
from mvc.views.widgets.map_canvas import DotaMapCanvas


class ClusterPage(BasePage):
    """Visualisation d'un cluster sur la carte."""

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._build()

    def _build(self):
        top = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10, height=60)
        top.pack(fill="x", padx=10, pady=(10, 5))
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="w_error:", font=ctk.CTkFont(size=13)).pack(
            side="left", padx=(15, 5)
        )
        self.w_error_var = ctk.StringVar(value="12.0")
        self.w_error_combo = ctk.CTkComboBox(
            top,
            variable=self.w_error_var,
            values=["12.0"],
            width=100,
            command=self._on_w_error_change,
        )
        self.w_error_combo.pack(side="left", padx=5)

        ctk.CTkLabel(top, text="Cluster:", font=ctk.CTkFont(size=13)).pack(
            side="left", padx=(20, 5)
        )
        self.cluster_var = ctk.StringVar(value="0")
        self.cluster_combo = ctk.CTkComboBox(
            top,
            variable=self.cluster_var,
            values=["0"],
            width=100,
        )
        self.cluster_combo.pack(side="left", padx=5)

        self.load_btn = ctk.CTkButton(
            top,
            text="Afficher",
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._on_load,
        )
        self.load_btn.pack(side="left", padx=15)

        self.new_cluster_btn = ctk.CTkButton(
            top,
            text="Nouveau Clustering",
            fg_color=ACCENT2,
            hover_color="#1a4a80",
            command=self._show_cluster_form,
        )
        self.new_cluster_btn.pack(side="left", padx=5)

        self.export_btn = ctk.CTkButton(
            top,
            text="Exporter JPG",
            fg_color=ACCENT2,
            hover_color="#1a4a80",
            command=self._on_export,
            width=110,
        )
        self.export_btn.pack(side="left", padx=5)

        self.all_clusters_btn = ctk.CTkButton(
            top,
            text="Tous les clusters",
            fg_color=ACCENT2,
            hover_color="#1a4a80",
            command=self._on_show_all_clusters,
            width=130,
        )
        self.all_clusters_btn.pack(side="left", padx=5)

        self.info_label = ctk.CTkLabel(
            top,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM,
        )
        self.info_label.pack(side="right", padx=15)

        # Panneau "pas de clustering"
        self.no_cluster_frame = ctk.CTkFrame(self, fg_color="transparent")

        ctk.CTkLabel(
            self.no_cluster_frame,
            text="Aucun résultat de clustering trouvé",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(pady=(80, 10))
        ctk.CTkLabel(
            self.no_cluster_frame,
            text="Lancez d'abord le clustering sur les données compressées.",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_DIM,
        ).pack(pady=(0, 25))

        cluster_form = ctk.CTkFrame(
            self.no_cluster_frame, fg_color=BG_CARD, corner_radius=12
        )
        cluster_form.pack(padx=60, fill="x")

        row_algo = ctk.CTkFrame(cluster_form, fg_color="transparent")
        row_algo.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(row_algo, text="Algorithme:", width=120, anchor="w").pack(
            side="left"
        )
        self.algo_var = ctk.StringVar(value="kmeans")
        ctk.CTkOptionMenu(
            row_algo,
            variable=self.algo_var,
            values=["kmeans", "affinity", "kmedoids"],
            width=150,
        ).pack(side="left", padx=10)

        row_mf = ctk.CTkFrame(cluster_form, fg_color="transparent")
        row_mf.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_mf, text="Max fichiers:", width=120, anchor="w").pack(
            side="left"
        )
        self.maxfiles_entry = ctk.CTkEntry(row_mf, width=100, placeholder_text="10")
        self.maxfiles_entry.pack(side="left", padx=10)
        self.maxfiles_entry.insert(0, "10")

        row_nc = ctk.CTkFrame(cluster_form, fg_color="transparent")
        row_nc.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_nc, text="Nb clusters:", width=120, anchor="w").pack(
            side="left"
        )
        self.nclusters_entry = ctk.CTkEntry(row_nc, width=100, placeholder_text="50")
        self.nclusters_entry.pack(side="left", padx=10)
        self.nclusters_entry.insert(0, "50")

        self.run_cluster_btn = ctk.CTkButton(
            cluster_form,
            text="Lancer Clustering",
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._on_run_clustering,
        )
        self.run_cluster_btn.pack(pady=20)

        self.cluster_log = ctk.CTkLabel(
            self.no_cluster_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM,
            wraplength=500,
        )
        self.cluster_log.pack(pady=10)

        self.cluster_progress = ctk.CTkProgressBar(
            self.no_cluster_frame, width=400, mode="indeterminate"
        )

        self.map_canvas = DotaMapCanvas(self, show_dots=False)

    def on_show(self):
        w_errors = self.controller.get_available_w_errors()
        if w_errors:
            vals = [str(w) for w in w_errors]
            self.w_error_combo.configure(values=vals)
            self.w_error_var.set(vals[0])
            self._on_w_error_change(vals[0])

    def _on_w_error_change(self, val):
        try:
            w = float(val)
        except ValueError:
            return
        clusters = self.controller.get_available_clusters(w)
        if clusters:
            vals = [str(c) for c in clusters]
            self.cluster_combo.configure(values=vals)
            self.cluster_var.set(vals[0])
            self.no_cluster_frame.pack_forget()
            self.map_canvas.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        else:
            self.cluster_combo.configure(values=["—"])
            self.cluster_var.set("—")
            self.map_canvas.pack_forget()
            self.no_cluster_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def _show_cluster_form(self):
        self.map_canvas.pack_forget()
        self.no_cluster_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def _on_run_clustering(self):
        try:
            w = float(self.w_error_var.get())
        except ValueError:
            return
        max_f = (
            int(self.maxfiles_entry.get())
            if self.maxfiles_entry.get().strip()
            else None
        )
        n_c = (
            int(self.nclusters_entry.get())
            if self.nclusters_entry.get().strip()
            else 50
        )
        algo = self.algo_var.get()
        self.run_cluster_btn.configure(state="disabled", text="En cours…")
        self.cluster_log.configure(text="Clustering en cours, veuillez patienter…")
        self.cluster_progress.pack(pady=(0, 10))
        self.cluster_progress.start()
        self.controller.start_clustering(w, algo=algo, max_files=max_f, n_clusters=n_c)

    def on_clustering_done(self, success, error_msg):
        self.run_cluster_btn.configure(state="normal", text="Lancer Clustering")
        self.cluster_progress.stop()
        self.cluster_progress.pack_forget()
        if success:
            self.cluster_log.configure(text="Clustering terminé ! Rechargement…")
            self._on_w_error_change(self.w_error_var.get())
        else:
            self.cluster_log.configure(text=f"Erreur : {error_msg}")

    def _on_load(self):
        try:
            w = float(self.w_error_var.get())
            cid = int(self.cluster_var.get())
        except ValueError:
            return
        self.load_btn.configure(state="disabled", text="Chargement…")
        self.controller.load_cluster_visu(w, cid)

    def on_cluster_loaded(self, data):
        self.load_btn.configure(state="normal", text="Afficher")
        if data is None:
            self.info_label.configure(text="Données introuvables")
            return
        self.map_canvas.set_background(data.canvas_image)
        self.map_canvas.draw_raw_segments(data.segments)
        self.info_label.configure(
            text=f"Cluster #{data.cluster_id} — {data.total_in_cluster} segments"
        )

    def on_all_clusters_loaded(self, data):
        """Callback pour la vue tous-clusters."""
        self.load_btn.configure(state="normal", text="Afficher")
        if data is None:
            self.info_label.configure(text="Erreur chargement tous clusters")
            return
        self.map_canvas.set_background(data.canvas_image)
        self.map_canvas.draw_raw_segments(data.segments)
        self.info_label.configure(
            text=f"Tous les clusters — {data.total_in_cluster} segments"
        )

    def _on_export(self):
        from mvc.config import OUTPUT_DIR

        out_dir = OUTPUT_DIR / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        cid = self.cluster_var.get()
        path = out_dir / f"cluster_{cid}.jpg"
        self.map_canvas.export_to_jpg(str(path))
        self.export_btn.configure(text="Exporté !", fg_color="#27ae60")
        self.after(1500, lambda: self.export_btn.configure(text="Exporter JPG", fg_color=ACCENT2))

    def _on_show_all_clusters(self):
        """Affiche tous les clusters sur la carte avec couleurs distinctes."""
        try:
            w = float(self.w_error_var.get())
        except ValueError:
            return
        clusters = self.controller.get_available_clusters(w)
        if not clusters:
            return
        self.load_btn.configure(state="disabled", text="Chargement…")
        self.info_label.configure(text="Chargement de tous les clusters…")
        self.controller.load_all_clusters_visu(w, clusters)
