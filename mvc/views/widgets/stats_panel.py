"""
StatsPanel — Panneau compact de statistiques en temps réel.
"""

from typing import Dict

import customtkinter as ctk

from mvc.views.theme import BG_CARD, TEXT_DIM, TEXT_LIGHT


class StatsPanel(ctk.CTkFrame):
    """Panneau compact de statistiques en temps réel."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=10, **kwargs)
        self._labels: Dict[str, ctk.CTkLabel] = {}
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self, text="Statistiques",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_LIGHT,
        ).pack(pady=(8, 2), padx=8)

        metrics = [
            ("segments", "Segments"),
            ("players", "Joueurs actifs"),
            ("time", "Temps de jeu"),
        ]
        for key, label in metrics:
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(
                row, text=f"{label}:", font=ctk.CTkFont(size=9),
                text_color=TEXT_DIM, width=85, anchor="w",
            ).pack(side="left")
            v = ctk.CTkLabel(
                row, text="—", font=ctk.CTkFont(size=9, weight="bold"),
                text_color=TEXT_LIGHT, anchor="w",
            )
            v.pack(side="left")
            self._labels[key] = v

    def update_from(self, stats: dict):
        if "visible_segments" in stats:
            self._labels["segments"].configure(
                text=f"{stats['visible_segments']} / {stats['total_segments']}")
        if "active_players" in stats:
            self._labels["players"].configure(text=str(stats["active_players"]))
        if "elapsed_sec" in stats:
            m, s = divmod(int(stats["elapsed_sec"]), 60)
            self._labels["time"].configure(text=f"{m:02d}:{s:02d}")
