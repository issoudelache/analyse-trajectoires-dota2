"""
Controller principal — orchestre Model et View.

Gère les événements utilisateur, appelle le Model en arrière-plan (thread)
et met à jour la View via des callbacks thread-safe.
"""

import threading
from typing import Optional

from mvc.models.app_model import AppModel, CompressResult, OverlayData, ClusterVisuData, ComparisonData


class AppController:
    """Contrôleur de l'application GUI."""

    def __init__(self):
        self.model = AppModel()
        self.view = None  # sera attaché par la View lors de l'initialisation

    def attach_view(self, view):
        self.view = view

    # ── données disponibles ──────────────────────────────────────────────

    def get_available_w_errors(self):
        return self.model.list_available_w_errors()

    def get_available_matches(self, w_error: float):
        return self.model.list_available_matches(w_error)

    def get_csv_matches(self):
        return self.model.list_csv_matches()

    def get_available_clusters(self, w_error: float):
        return self.model.list_available_clusters(w_error)

    # ── compression (en thread séparé) ───────────────────────────────────

    def start_compression(self, w_error: float, match_id: Optional[str] = None):
        """Lance la compression dans un thread séparé."""
        def _worker():
            try:
                def _progress(current, total, result: CompressResult):
                    if self.view:
                        self.view.after(0, self.view.on_compress_progress, current, total, result)

                results = self.model.compress(w_error, match_id, callback=_progress)

                if self.view:
                    self.view.after(0, self.view.on_compress_done, results)
            except Exception as e:
                if self.view:
                    self.view.after(0, self.view.on_compress_done, [])

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── overlay ──────────────────────────────────────────────────────────

    def load_overlay(self, w_error: float, match_id: str):
        """Charge les données d'overlay (thread)."""
        def _worker():
            try:
                data = self.model.load_overlay_data(w_error, match_id)
            except Exception:
                data = None
            if self.view:
                self.view.after(0, self.view.on_overlay_loaded, data)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── cluster visu ─────────────────────────────────────────────────────

    def load_cluster_visu(self, w_error: float, cluster_id: int):
        """Charge les données de visualisation cluster (thread)."""
        def _worker():
            try:
                data = self.model.load_cluster_visu_data(w_error, cluster_id)
            except Exception:
                data = None
            if self.view:
                self.view.after(0, self.view.on_cluster_loaded, data)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── clustering ───────────────────────────────────────────────────────

    def load_comparison(self, w_error: float, match_id: str):
        """Charge les données de comparaison brut vs compressé (thread)."""
        def _worker():
            try:
                data = self.model.load_comparison_data(w_error, match_id)
            except Exception:
                data = None
            if self.view:
                self.view.after(0, self.view.on_comparison_loaded, data)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── clustering (exécution) ───────────────────────────────────────────

    def start_clustering(self, w_error: float, **kwargs):
        """Lance le clustering dans un thread séparé."""
        def _worker():
            try:
                self.model.run_clustering(w_error, **kwargs)
                if self.view:
                    self.view.after(0, self.view.on_clustering_done, True, "")
            except Exception as e:
                if self.view:
                    self.view.after(0, self.view.on_clustering_done, False, str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
