"""
Controller principal — orchestre Model et View.

Gère les événements utilisateur, appelle le Model en arrière-plan (thread)
et met à jour la View via des callbacks thread-safe.
"""

import threading
from typing import Optional

from mvc.models.app_model import (
    AppModel,
    CompressResult,
)


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
                        self.view.after(
                            0, self.view.on_compress_progress, current, total, result
                        )

                results = self.model.compress(w_error, match_id, callback=_progress)

                if self.view:
                    self.view.after(0, self.view.on_compress_done, results)
            except Exception:
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

    def load_all_clusters_visu(self, w_error: float, cluster_ids: list):
        """Charge tous les clusters avec couleurs distinctes (thread)."""

        def _worker():
            try:
                data = self.model.load_all_clusters_visu_data(w_error, cluster_ids)
            except Exception:
                data = None
            if self.view:
                self.view.after(0, self.view.on_all_clusters_loaded, data)

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

    # ── recodage ──────────────────────────────────────────────────────────

    def start_recoding(self, w_error: float):
        """Lance le recodage dans un thread séparé."""

        def _worker():
            try:
                result = self.model.run_recoding(w_error)
                if self.view:
                    self.view.after(
                        0,
                        self.view.on_recode_done,
                        result.success,
                        result.num_sequences,
                        result.error,
                    )
            except Exception as e:
                if self.view:
                    self.view.after(0, self.view.on_recode_done, False, 0, str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── PrefixSpan ────────────────────────────────────────────────────────

    def start_mining(self, min_support: int = 10, max_length: int = 8):
        """Lance PrefixSpan dans un thread séparé (mode parallèle, sans callback)."""

        def _worker():
            try:
                # Pas de callback = mode parallèle activé = plus rapide
                result = self.model.run_mining(min_support, max_length)
                if self.view:
                    self.view.after(
                        0,
                        self.view.on_mining_done,
                        result.success,
                        result.num_patterns,
                        result.top_patterns,
                        result.error,
                    )
            except Exception as e:
                if self.view:
                    self.view.after(0, self.view.on_mining_done, False, 0, [], str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── Graphe de transitions ─────────────────────────────────────────────

    def generate_transition_graph(self, patterns: list, min_len: int = 2):
        """Génère le graphe de transitions dans un thread séparé."""

        def _worker():
            try:
                success, image_bytes, error = self.model.generate_transition_graph(
                    patterns, min_len
                )
                if self.view:
                    self.view.after(
                        0, self.view.on_graph_generated, success, image_bytes, error
                    )
            except Exception as e:
                if self.view:
                    self.view.after(0, self.view.on_graph_generated, False, b"", str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ── Pipeline page (dédié) ────────────────────────────────────────────

    def start_pipeline_page(
        self,
        w_error: float,
        algo: str = "affinity",
        cluster_kwargs: dict = None,
        min_support: int = 10,
        max_length: int = 8,
    ):
        """Pipeline complet avec compression parallèle et paramètres clustering.

        Les résultats intermédiaires sont envoyés à la PipelinePage.
        """
        if cluster_kwargs is None:
            cluster_kwargs = {}

        def _worker():
            steps = [
                "Compression",
                "Clustering",
                "Recodage",
                "PrefixSpan",
                "Graphes",
            ]
            try:
                # Étape 1 — compression parallèle
                self._notify_pp(1, len(steps), steps[0])
                results = self.model.compress_parallel(w_error)
                n_ok = sum(1 for r in results if r.success)
                self._notify_pp_result(1, f"{n_ok}/{len(results)} matchs compressés")

                # Étape 2 — clustering
                self._notify_pp(2, len(steps), steps[1])
                self.model.run_clustering(w_error, algo=algo, **cluster_kwargs)
                clusters = self.model.list_available_clusters(w_error)
                if not clusters:
                    raise RuntimeError(
                        "Clustering échoué (0 clusters). "
                        "Pour affinity/kmedoids, réduisez max_files (<5000 segments)."
                    )
                self._notify_pp_result(2, f"{len(clusters)} clusters ({algo})")

                # Étape 3 — recodage
                self._notify_pp(3, len(steps), steps[2])
                recode_res = self.model.run_recoding(w_error)
                if not recode_res.success:
                    raise RuntimeError(f"Recodage : {recode_res.error}")
                self._notify_pp_result(3, f"{recode_res.num_sequences} séquences")

                # Étape 4 — PrefixSpan
                self._notify_pp(4, len(steps), steps[3])
                mining_res = self.model.run_mining(min_support, max_length)
                if not mining_res.success:
                    raise RuntimeError(f"PrefixSpan : {mining_res.error}")
                self._notify_pp_result(
                    4,
                    f"{mining_res.num_patterns} motifs",
                    patterns=mining_res.top_patterns,
                )

                # Étape 5 — génération des graphes
                self._notify_pp(5, len(steps), steps[4])

                ok_g, graph_bytes, err_g = self.model.generate_transition_graph(
                    mining_res.top_patterns, min_len=2
                )
                ok_f, freq_bytes, err_f = self.model.generate_frequency_chart(
                    mining_res.top_patterns
                )

                self._notify_pp_result(
                    5,
                    "Graphes générés",
                    graph_bytes=graph_bytes if ok_g else b"",
                    freq_bytes=freq_bytes if ok_f else b"",
                )

                if self.view:
                    self.view.after(0, self.view.on_pp_done, True, "")

            except Exception as e:
                if self.view:
                    self.view.after(0, self.view.on_pp_done, False, str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _notify_pp(self, step: int, total: int, label: str):
        if self.view:
            self.view.after(0, self.view.on_pp_progress, step, total, label)

    def _notify_pp_result(self, step: int, message: str, **kwargs):
        if self.view:
            self.view.after(0, self.view.on_pp_step_result, step, message, kwargs)
