"""
PlayerLegend — Légende interactive Radiant / Dire avec toggle par joueur.
"""

import customtkinter as ctk

from dota_analytics.plotting import PLAYER_COLORS
from mvc.views.theme import BG_CARD, TEXT_DIM, TEXT_LIGHT


class PlayerLegend(ctk.CTkFrame):
    """Légende interactive Radiant / Dire — cliquer sur un joueur le masque/affiche."""

    def __init__(self, master, on_toggle_callback=None, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=10, **kwargs)
        self._on_toggle = on_toggle_callback
        self._player_visible = {i: True for i in range(10)}
        self._row_labels = {}
        self._row_dots = {}
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self,
            text="Joueurs",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LIGHT,
        ).pack(pady=(8, 2), padx=8)

        # ─ Radiant ─
        ctk.CTkLabel(
            self,
            text="RADIANT",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#3498db",
        ).pack(anchor="w", padx=10, pady=(4, 0))
        for i in range(5):
            self._row(i, f"Joueur {i + 1}", PLAYER_COLORS[i])

        # ─ Dire ─
        ctk.CTkLabel(
            self,
            text="DIRE",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#e74c3c",
        ).pack(anchor="w", padx=10, pady=(6, 0))
        for i in range(5, 10):
            self._row(i, f"Joueur {i - 4}", PLAYER_COLORS[i])

    def _row(self, pid, name, color):
        f = ctk.CTkFrame(self, fg_color="transparent", height=16, cursor="hand2")
        f.pack(fill="x", padx=10, pady=0)
        dot = ctk.CTkLabel(
            f, text="●", font=ctk.CTkFont(size=9), text_color=color, width=14
        )
        dot.pack(side="left")
        lbl = ctk.CTkLabel(f, text=name, font=ctk.CTkFont(size=9), text_color=color)
        lbl.pack(side="left")
        self._row_labels[pid] = lbl
        self._row_dots[pid] = dot

        for widget in (f, dot, lbl):
            widget.bind("<Button-1>", lambda e, p=pid: self._toggle(p))

    def _toggle(self, pid):
        self._player_visible[pid] = not self._player_visible[pid]
        color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
        if self._player_visible[pid]:
            self._row_dots[pid].configure(text="●", text_color=color)
            self._row_labels[pid].configure(text_color=color)
        else:
            self._row_dots[pid].configure(text="○", text_color=TEXT_DIM)
            self._row_labels[pid].configure(text_color=TEXT_DIM)

        if self._on_toggle:
            self._on_toggle(pid, self._player_visible[pid])

    def get_visible_players(self):
        """Retourne le set des player IDs visibles."""
        return {pid for pid, v in self._player_visible.items() if v}
