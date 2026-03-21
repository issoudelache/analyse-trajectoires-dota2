"""
PlayerLegend — Légende compacte Radiant / Dire avec couleurs.
"""

import customtkinter as ctk

from dota_analytics.plotting import PLAYER_COLORS
from mvc.views.theme import BG_CARD, TEXT_LIGHT


class PlayerLegend(ctk.CTkFrame):
    """Légende compacte Radiant / Dire avec couleurs."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=10, **kwargs)
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self, text="Joueurs",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_LIGHT,
        ).pack(pady=(8, 2), padx=8)

        # ─ Radiant ─
        ctk.CTkLabel(
            self, text="RADIANT", font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#3498db",
        ).pack(anchor="w", padx=10, pady=(4, 0))
        for i in range(5):
            self._row(i, f"Joueur {i+1}", PLAYER_COLORS[i])

        # ─ Dire ─
        ctk.CTkLabel(
            self, text="DIRE", font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#e74c3c",
        ).pack(anchor="w", padx=10, pady=(6, 0))
        for i in range(5, 10):
            self._row(i, f"Joueur {i - 4}", PLAYER_COLORS[i])

    def _row(self, pid, name, color):
        f = ctk.CTkFrame(self, fg_color="transparent", height=16)
        f.pack(fill="x", padx=10, pady=0)
        ctk.CTkLabel(f, text="●", font=ctk.CTkFont(size=9), text_color=color, width=14).pack(side="left")
        ctk.CTkLabel(f, text=name, font=ctk.CTkFont(size=9), text_color=color).pack(side="left")
