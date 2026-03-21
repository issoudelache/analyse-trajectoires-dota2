"""
Thème global — couleurs, polices et constantes visuelles partagées.
"""

import customtkinter as ctk

# ── Apparence ────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Palette de couleurs ──────────────────────────────────────────────────
BG_DARK = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
TEXT_LIGHT = "#eaeaea"
TEXT_DIM = "#8899aa"

# ── Noms des joueurs ────────────────────────────────────────────────────
PLAYER_NAMES = [f"Radiant {i+1}" for i in range(5)] + [f"Dire {i+1}" for i in range(5)]
