"""
MenuPage — Page d'accueil avec choix du mode.
"""

import customtkinter as ctk

from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, BG_CARD, TEXT_DIM, TEXT_LIGHT


class MenuPage(BasePage):
    """Page d'accueil — choix du mode."""

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._build()

    def _build(self):
        title = ctk.CTkLabel(
            self, text="Dota 2 Trajectory Analyzer",
            font=ctk.CTkFont(size=32, weight="bold"), text_color=ACCENT,
        )
        title.pack(pady=(60, 10))

        ctk.CTkLabel(
            self, text="Analyse de trajectoires — Compression, Clustering, Visualisation",
            font=ctk.CTkFont(size=14), text_color=TEXT_DIM,
        ).pack(pady=(0, 50))

        cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        cards_frame.pack(expand=True)

        modes = [
            ("Overlay Carte", "Visualiser les trajectoires\ncompressées sur la carte Dota 2", "overlay"),
            ("Compression", "Lancer la compression MDL\nsur un ou tous les matchs", "compress"),
            ("Clusters", "Visualiser les clusters\nde segments sur la carte", "cluster"),
            ("Comparaison", "Brut vs Compressé\ncôte à côte avec animation", "comparison"),
        ]

        for i, (t, desc, pg) in enumerate(modes):
            card = ctk.CTkFrame(cards_frame, fg_color=BG_CARD, corner_radius=16, width=240, height=200)
            card.grid(row=0, column=i, padx=20, pady=10)
            card.grid_propagate(False)

            ctk.CTkLabel(card, text=t, font=ctk.CTkFont(size=18, weight="bold"),
                          text_color=TEXT_LIGHT).pack(pady=(30, 10))
            ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(size=12),
                          text_color=TEXT_DIM, justify="center").pack(pady=(0, 15))
            ctk.CTkButton(card, text="Ouvrir →", fg_color=ACCENT, hover_color="#c33750",
                           command=lambda p=pg: self.switch_page(p)).pack(pady=(5, 20))
