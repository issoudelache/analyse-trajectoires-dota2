"""
View principale — fenêtre racine avec sidebar et transitions animées.

Délègue tout le contenu des pages aux modules mvc.views.pages.*
et les widgets à mvc.views.widgets.*.
"""

from typing import Dict, Optional

import customtkinter as ctk

from mvc.views.pages import (
    BasePage,
    ClusterPage,
    ComparisonPage,
    CompressPage,
    MenuPage,
    MiningPage,
    OverlayPage,
)
from mvc.views.theme import ACCENT, ACCENT2, BG_CARD, BG_DARK, TEXT_DIM, TEXT_LIGHT


class MainWindow(ctk.CTk):
    """Fenêtre racine de l'application."""

    ANIM_DURATION_MS = 300
    ANIM_STEPS = 15

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        controller.attach_view(self)

        self.title("Dota 2 — Trajectory Analyzer")
        self.geometry("1200x800")
        self.minsize(900, 600)
        self.configure(fg_color=BG_DARK)

        self._build_layout()
        self._pages: Dict[str, BasePage] = {}
        self._current_page_name: Optional[str] = None
        self._animating = False

        self._pages["menu"] = MenuPage(self.content_frame, controller, self.switch_page)
        self._pages["overlay"] = OverlayPage(
            self.content_frame, controller, self.switch_page
        )
        self._pages["compress"] = CompressPage(
            self.content_frame, controller, self.switch_page
        )
        self._pages["cluster"] = ClusterPage(
            self.content_frame, controller, self.switch_page
        )
        self._pages["comparison"] = ComparisonPage(
            self.content_frame, controller, self.switch_page
        )
        self._pages["mining"] = MiningPage(
            self.content_frame, controller, self.switch_page
        )

        self.switch_page("menu", animate=False)

    def _build_layout(self):
        self.sidebar = ctk.CTkFrame(self, width=200, fg_color=BG_CARD, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar,
            text="DOTA 2",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=ACCENT,
        ).pack(pady=(25, 2))
        ctk.CTkLabel(
            self.sidebar,
            text="Trajectories",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_DIM,
        ).pack(pady=(0, 30))

        nav_items = [
            ("Accueil", "menu"),
            ("Overlay Carte", "overlay"),
            ("Compression", "compress"),
            ("Clusters", "cluster"),
            ("Comparaison", "comparison"),
            ("PrefixSpan", "mining"),
        ]
        self._nav_buttons = {}
        for label, page_name in nav_items:
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                fg_color="transparent",
                text_color=TEXT_LIGHT,
                hover_color=ACCENT2,
                anchor="w",
                height=40,
                corner_radius=8,
                command=lambda p=page_name: self.switch_page(p),
            )
            btn.pack(fill="x", padx=12, pady=3)
            self._nav_buttons[page_name] = btn

        ctk.CTkFrame(self.sidebar, height=1, fg_color=TEXT_DIM).pack(
            fill="x", padx=20, pady=20
        )

        ctk.CTkLabel(
            self.sidebar,
            text="v1.1.0",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM,
        ).pack(side="bottom", pady=15)

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(side="right", fill="both", expand=True)

    # ── Navigation animée ────────────────────────────────────────────────

    def switch_page(self, page_name: str, animate: bool = True):
        if page_name == self._current_page_name or self._animating:
            return
        new_page = self._pages.get(page_name)
        if new_page is None:
            return

        old_page = self._pages.get(self._current_page_name)

        for name, btn in self._nav_buttons.items():
            btn.configure(fg_color=ACCENT2 if name == page_name else "transparent")

        if not animate or old_page is None:
            if old_page:
                old_page.place_forget()
            new_page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._current_page_name = page_name
            new_page.on_show()
            return

        self._animating = True
        step_delay = self.ANIM_DURATION_MS // self.ANIM_STEPS

        new_page.place(relx=1.0, rely=0, relwidth=1, relheight=1)
        new_page.on_show()

        self._anim_step = 0

        def _tick():
            self._anim_step += 1
            t = self._anim_step / self.ANIM_STEPS
            t_ease = 1 - (1 - t) ** 3

            old_page.place(relx=-t_ease * 0.3, rely=0, relwidth=1, relheight=1)
            new_page.place(relx=1.0 - t_ease, rely=0, relwidth=1, relheight=1)

            if self._anim_step < self.ANIM_STEPS:
                self.after(step_delay, _tick)
            else:
                old_page.place_forget()
                new_page.place(relx=0, rely=0, relwidth=1, relheight=1)
                self._current_page_name = page_name
                self._animating = False

        self.after(step_delay, _tick)

    # ── Callbacks Controller → View ──────────────────────────────────────

    def on_compress_progress(self, current, total, result):
        page = self._pages.get("compress")
        if page:
            page.on_compress_progress(current, total, result)

    def on_compress_done(self, results):
        page = self._pages.get("compress")
        if page:
            page.on_compress_done(results)

    def on_overlay_loaded(self, data):
        page = self._pages.get("overlay")
        if page:
            page.on_overlay_loaded(data)

    def on_cluster_loaded(self, data):
        page = self._pages.get("cluster")
        if page:
            page.on_cluster_loaded(data)

    def on_clustering_done(self, success, error_msg):
        page = self._pages.get("cluster")
        if page:
            page.on_clustering_done(success, error_msg)

    def on_comparison_loaded(self, data):
        page = self._pages.get("comparison")
        if page:
            page.on_comparison_loaded(data)

    def on_recode_done(self, success, num_sequences, error_msg):
        page = self._pages.get("mining")
        if page:
            page.on_recode_done(success, num_sequences, error_msg)

    def on_mining_done(self, success, num_patterns, top_patterns, error_msg):
        page = self._pages.get("mining")
        if page:
            page.on_mining_done(success, num_patterns, top_patterns, error_msg)

    def on_graph_generated(self, success, image_bytes, error_msg):
        page = self._pages.get("mining")
        if page:
            page.on_graph_generated(success, image_bytes, error_msg)
