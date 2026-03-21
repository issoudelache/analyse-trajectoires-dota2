"""
View principale — interface graphique CustomTkinter.

Structure:
    MainWindow
      ├─ sidebar (menu permanent à gauche)
      └─ content_frame (zone centrale qui change de page)
            ├─ MenuPage        (accueil avec choix du mode)
            ├─ OverlayPage     (visualisation compression sur la carte)
            ├─ CompressPage    (lancement de compression)
            ├─ ClusterPage     (visualisation des clusters)
            └─ ComparisonPage  (brut vs compressé côte à côte)

Optimisations :
    - Cache d'image de fond (resize PIL uniquement sur changement de taille)
    - Mise à jour incrémentale des segments (O(delta) par frame)
    - Throttle des redraws via after_idle
    - Dots de position animés, légende, statistiques, tooltips
"""

import bisect
import math
import tkinter as tk
from typing import Dict, List, Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from dota_analytics.plotting import PLAYER_COLORS

# ── Thème global ─────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_DARK = "#1a1a2e"
BG_CARD = "#16213e"
ACCENT = "#e94560"
ACCENT2 = "#0f3460"
TEXT_LIGHT = "#eaeaea"
TEXT_DIM = "#8899aa"

# Noms des joueurs
PLAYER_NAMES = [f"Radiant {i+1}" for i in range(5)] + [f"Dire {i+1}" for i in range(5)]


# ═══════════════════════════════════════════════════════════════════════════
# DotaMapCanvas — Canvas optimisé avec dots, tooltips, stats
# ═══════════════════════════════════════════════════════════════════════════


class DotaMapCanvas(ctk.CTkFrame):
    """Canvas haute performance pour la carte Dota 2.

    Optimisations clés :
    - Image de fond redimensionnée uniquement sur resize (cache PIL)
    - Lignes pré-créées comme items cachés, visibilité basculée par tick
    - Mise à jour O(delta) entre frames consécutives
    - Throttle via after_idle pour éviter les redraws multiples
    """

    _DOT_RADIUS = 5
    _TOOLTIP_DELAY_MS = 60

    def __init__(self, master, show_dots=True, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, bg="#0d0d0d", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._show_dots = show_dots

        # Image de fond
        self._canvas_img: Optional[Image.Image] = None
        self._bg_photo = None
        self._bg_cache_size = 0
        self._bg_item = None

        # Layout caché
        self._ox = 0
        self._oy = 0
        self._map_size = 0

        # Données unifiées
        self._draw_mode = "none"
        self._segments_data: Dict[int, list] = {}
        self._raw_points_data: Dict[int, list] = {}

        # Cache de lignes pré-créées
        self._lines: Dict[int, list] = {}       # pid -> [(x1,y1,x2,y2,s_tick,e_tick), ...]
        self._line_ids: Dict[int, list] = {}    # pid -> [canvas_id, ...]
        self._end_ticks: Dict[int, list] = {}   # pid -> [end_tick, ...] trié
        self._vis_count: Dict[int, int] = {}    # pid -> nb lignes visibles
        self._needs_rebuild = False

        # Head lines + dots
        self._head_ids: Dict[int, int] = {}
        self._dot_ids: Dict[int, int] = {}

        # Tooltip
        self._tooltip_after_id = None
        self._item_info: Dict[int, dict] = {}

        # Tick
        self._current_tick = 0
        self._min_tick = 0
        self._max_tick = 0

        # Throttle
        self._tick_pending = False

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._hide_tooltip)

    # ── Coordonnées ──────────────────────────────────────────────────────

    def _gx(self, x):
        return self._ox + x / 256.0 * self._map_size

    def _gy(self, y):
        return self._oy + (256 - y) / 256.0 * self._map_size

    def _compute_layout(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        s = min(cw, ch)
        self._ox = (cw - s) // 2
        self._oy = (ch - s) // 2
        self._map_size = s

    # ── Fond de carte (caché) ────────────────────────────────────────────

    def set_background(self, pil_image: Image.Image):
        self._canvas_img = pil_image.copy()
        self._bg_cache_size = 0

    def _render_bg(self):
        s = self._map_size
        if s < 10 or not self._canvas_img:
            return
        if s == self._bg_cache_size and self._bg_photo:
            if self._bg_item:
                self.canvas.coords(self._bg_item, self._ox, self._oy)
            return
        resized = self._canvas_img.resize((s, s), Image.LANCZOS)
        self._bg_photo = ImageTk.PhotoImage(resized)
        self._bg_cache_size = s
        if self._bg_item:
            self.canvas.itemconfigure(self._bg_item, image=self._bg_photo)
            self.canvas.coords(self._bg_item, self._ox, self._oy)
        else:
            self._bg_item = self.canvas.create_image(
                self._ox, self._oy, anchor="nw", image=self._bg_photo, tags="bg",
            )
        self.canvas.tag_lower("bg")

    # ── Setters de données ───────────────────────────────────────────────

    def set_segments(self, player_segments, min_tick, max_tick):
        self._segments_data = player_segments
        self._raw_points_data = {}
        self._draw_mode = "segments"
        self._min_tick = min_tick
        self._max_tick = max_tick
        self._current_tick = max_tick
        self._rebuild_cache()

    def set_raw_points(self, raw_points, min_tick, max_tick):
        self._raw_points_data = raw_points
        self._segments_data = {}
        self._draw_mode = "raw"
        self._min_tick = min_tick
        self._max_tick = max_tick
        self._current_tick = max_tick
        self._rebuild_cache()

    def set_tick(self, tick: int):
        self._current_tick = tick
        if not self._tick_pending:
            self._tick_pending = True
            self.after_idle(self._do_tick_update)

    def _do_tick_update(self):
        self._tick_pending = False
        self._update_visibility()
        self._update_heads_and_dots()

    # ── Construction du cache de lignes ──────────────────────────────────

    def _rebuild_cache(self):
        self._compute_layout()
        if self._map_size < 10:
            self._needs_rebuild = True
            return
        self._needs_rebuild = False

        # Nettoyer les anciens items de données
        self.canvas.delete("trail", "head", "dot", "tooltip")
        self._line_ids.clear()
        self._end_ticks.clear()
        self._lines.clear()
        self._vis_count.clear()
        self._head_ids.clear()
        self._dot_ids.clear()
        self._item_info.clear()

        self._render_bg()

        # Construire la liste unifiée de lignes par joueur
        source: Dict[int, list] = {}
        if self._draw_mode == "segments":
            for pid, segs in self._segments_data.items():
                lines = []
                for seg in segs:
                    lines.append((
                        seg["start"]["x"], seg["start"]["y"],
                        seg["end"]["x"], seg["end"]["y"],
                        seg["start"]["tick"], seg["end"]["tick"],
                    ))
                lines.sort(key=lambda l: l[5])
                source[pid] = lines
        elif self._draw_mode == "raw":
            for pid, pts in self._raw_points_data.items():
                lines = []
                for i in range(len(pts) - 1):
                    p1, p2 = pts[i], pts[i + 1]
                    lines.append((
                        p1["x"], p1["y"], p2["x"], p2["y"],
                        p1["tick"], p2["tick"],
                    ))
                source[pid] = lines

        # Pré-créer tous les items canvas comme cachés
        for pid, lines in source.items():
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
            self._lines[pid] = lines
            ids = []
            ticks = []
            for x1, y1, x2, y2, s_tick, e_tick in lines:
                cid = self.canvas.create_line(
                    self._gx(x1), self._gy(y1),
                    self._gx(x2), self._gy(y2),
                    fill=color, width=2, state="hidden", tags="trail",
                )
                ids.append(cid)
                ticks.append(e_tick)
                self._item_info[cid] = {
                    "pid": pid, "name": PLAYER_NAMES[pid % 10],
                    "s_tick": s_tick, "e_tick": e_tick,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                }
            self._line_ids[pid] = ids
            self._end_ticks[pid] = ticks
            self._vis_count[pid] = 0

            # Ligne d'interpolation (tête)
            self._head_ids[pid] = self.canvas.create_line(
                0, 0, 0, 0, fill=color, width=2, state="hidden", tags="head",
            )
            # Dot de position
            if self._show_dots:
                self._dot_ids[pid] = self.canvas.create_oval(
                    0, 0, 0, 0, fill=color, outline="white", width=1,
                    state="hidden", tags="dot",
                )

        # Afficher jusqu'au tick courant
        self._update_visibility()
        self._update_heads_and_dots()

    # ── Visibilité incrémentale O(delta) ─────────────────────────────────

    def _update_visibility(self):
        tick = self._current_tick
        for pid in self._end_ticks:
            et = self._end_ticks[pid]
            ids = self._line_ids[pid]
            new_n = bisect.bisect_right(et, tick)
            old_n = self._vis_count[pid]
            if new_n > old_n:
                for i in range(old_n, new_n):
                    self.canvas.itemconfigure(ids[i], state="normal")
            elif new_n < old_n:
                for i in range(new_n, old_n):
                    self.canvas.itemconfigure(ids[i], state="hidden")
            self._vis_count[pid] = new_n

    # ── Tête interpolée + dots ───────────────────────────────────────────

    def _update_heads_and_dots(self):
        tick = self._current_tick
        r = self._DOT_RADIUS
        for pid, lines in self._lines.items():
            n = self._vis_count.get(pid, 0)
            head = self._head_ids.get(pid)
            dot = self._dot_ids.get(pid)
            px, py = None, None

            if n < len(lines):
                x1, y1, x2, y2, st, et = lines[n]
                if st <= tick < et:
                    ratio = (tick - st) / max(et - st, 1)
                    ix = x1 + ratio * (x2 - x1)
                    iy = y1 + ratio * (y2 - y1)
                    if head:
                        self.canvas.coords(
                            head, self._gx(x1), self._gy(y1),
                            self._gx(ix), self._gy(iy),
                        )
                        self.canvas.itemconfigure(head, state="normal")
                    px, py = ix, iy
                else:
                    if head:
                        self.canvas.itemconfigure(head, state="hidden")
                    if tick >= st:
                        px, py = x1, y1
            else:
                if head:
                    self.canvas.itemconfigure(head, state="hidden")
                if lines:
                    px, py = lines[-1][2], lines[-1][3]

            if dot is not None and px is not None:
                cx, cy = self._gx(px), self._gy(py)
                self.canvas.coords(dot, cx - r, cy - r, cx + r, cy + r)
                self.canvas.itemconfigure(dot, state="normal")
                self.canvas.tag_raise("dot")
            elif dot is not None:
                self.canvas.itemconfigure(dot, state="hidden")

    # ── Resize ───────────────────────────────────────────────────────────

    def _on_resize(self, event=None):
        old = self._map_size
        self._compute_layout()
        if self._needs_rebuild:
            self._rebuild_cache()
            return
        if self._map_size == old and old > 0:
            return
        self._render_bg()
        # Repositionner toutes les lignes
        for pid, lines in self._lines.items():
            ids = self._line_ids[pid]
            for i, (x1, y1, x2, y2, _st, _et) in enumerate(lines):
                self.canvas.coords(
                    ids[i],
                    self._gx(x1), self._gy(y1),
                    self._gx(x2), self._gy(y2),
                )
        self._update_heads_and_dots()

    # ── Tooltip ──────────────────────────────────────────────────────────

    def _on_mouse_move(self, event):
        if self._tooltip_after_id:
            self.after_cancel(self._tooltip_after_id)
        self._tooltip_after_id = self.after(
            self._TOOLTIP_DELAY_MS,
            lambda e=event: self._show_tooltip(e.x, e.y),
        )

    def _show_tooltip(self, x, y):
        self._tooltip_after_id = None
        items = self.canvas.find_closest(x, y)
        if not items:
            self._hide_tooltip()
            return
        info = self._item_info.get(items[0])
        if info is None:
            self._hide_tooltip()
            return
        # Vérifier la proximité
        bbox = self.canvas.bbox(items[0])
        if bbox:
            cx = max(bbox[0], min(x, bbox[2]))
            cy = max(bbox[1], min(y, bbox[3]))
            if math.hypot(x - cx, y - cy) > 12:
                self._hide_tooltip()
                return
        team = "Radiant" if info["pid"] < 5 else "Dire"
        txt = (
            f"{info['name']} ({team})\n"
            f"Tick: {info['s_tick']} → {info['e_tick']}\n"
            f"({info['x1']:.0f},{info['y1']:.0f}) → ({info['x2']:.0f},{info['y2']:.0f})"
        )
        self._draw_tooltip(x + 15, y - 10, txt)

    def _draw_tooltip(self, x, y, text):
        self.canvas.delete("tooltip")
        tid = self.canvas.create_text(
            x + 6, y + 4, text=text, anchor="nw",
            fill=TEXT_LIGHT, font=("Consolas", 9), tags="tooltip",
        )
        bb = self.canvas.bbox(tid)
        if bb:
            p = 5
            self.canvas.create_rectangle(
                bb[0] - p, bb[1] - p, bb[2] + p, bb[3] + p,
                fill="#1a1a2eee", outline=TEXT_DIM, tags="tooltip",
            )
            self.canvas.tag_raise(tid)

    def _hide_tooltip(self, event=None):
        if self._tooltip_after_id:
            self.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        self.canvas.delete("tooltip")

    # ── Legacy (cluster page) ────────────────────────────────────────────

    def draw_raw_segments(self, segments_list):
        """Dessine une liste brute de (x1,y1,x2,y2,color) pour la page Cluster."""
        self.canvas.delete("trail", "head", "dot", "tooltip")
        self._lines.clear()
        self._line_ids.clear()
        self._end_ticks.clear()
        self._vis_count.clear()
        self._item_info.clear()
        self._draw_mode = "none"
        self._compute_layout()
        self._render_bg()
        for x1, y1, x2, y2, color in segments_list:
            self.canvas.create_line(
                self._gx(x1), self._gy(y1), self._gx(x2), self._gy(y2),
                fill=color, width=2, tags="trail",
            )

    # ── Stats helper ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = sum(len(ln) for ln in self._lines.values())
        visible = sum(self._vis_count.get(p, 0) for p in self._lines)
        active = sum(1 for p in self._lines if self._vis_count.get(p, 0) > 0)
        elapsed = (self._current_tick - self._min_tick) / 30.0 if self._min_tick else 0
        return {
            "total_segments": total,
            "visible_segments": visible,
            "active_players": active,
            "elapsed_sec": elapsed,
            "current_tick": self._current_tick,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Widgets utilitaires : légende et statistiques
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════════════════


class BasePage(ctk.CTkFrame):
    """Page de base avec méthodes utilitaires."""

    def __init__(self, master, controller, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller

    def on_show(self):
        """Appelé juste avant que la page soit affichée."""
        pass


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


# ═════════════════════════════════════════════════════════════════════
# OverlayPage — avec légende + stats latérales
# ═════════════════════════════════════════════════════════════════════


class OverlayPage(BasePage):
    """Visualisation des trajectoires compressées sur la carte."""

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._overlay_data = None
        self._build()

    def _build(self):
        # ── Barre d'options ──────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10, height=60)
        top.pack(fill="x", padx=10, pady=(10, 5))
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="w_error:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(15, 5))
        self.w_error_var = ctk.StringVar(value="12.0")
        self.w_error_combo = ctk.CTkComboBox(
            top, variable=self.w_error_var, values=["12.0"], width=100,
            command=self._on_w_error_change,
        )
        self.w_error_combo.pack(side="left", padx=5)

        ctk.CTkLabel(top, text="Match:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(20, 5))
        self.match_var = ctk.StringVar(value="—")
        self.match_combo = ctk.CTkComboBox(top, variable=self.match_var, values=["—"], width=180)
        self.match_combo.pack(side="left", padx=5)

        self.load_btn = ctk.CTkButton(
            top, text="Charger", fg_color=ACCENT, hover_color="#c33750", command=self._on_load,
        )
        self.load_btn.pack(side="left", padx=15)

        # ── Zone centrale : carte + panneau latéral ──────────────────────
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.pack(fill="both", expand=True, padx=10, pady=5)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=0)
        mid.rowconfigure(0, weight=1)

        self.map_canvas = DotaMapCanvas(mid, show_dots=True)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")

        # Panneau droit
        right = ctk.CTkFrame(mid, fg_color="transparent", width=140)
        right.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        right.grid_propagate(False)

        self.legend = PlayerLegend(right)
        self.legend.pack(fill="x", pady=(0, 6))

        self.stats_panel = StatsPanel(right)
        self.stats_panel.pack(fill="x")

        # ── Slider temporel ──────────────────────────────────────────────
        slider_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10, height=50)
        slider_frame.pack(fill="x", padx=10, pady=(5, 10))
        slider_frame.pack_propagate(False)

        ctk.CTkLabel(slider_frame, text="Temps:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(15, 5))
        self.tick_slider = ctk.CTkSlider(slider_frame, from_=0, to=1, command=self._on_slider)
        self.tick_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.tick_label = ctk.CTkLabel(
            slider_frame, text="tick: 0", font=ctk.CTkFont(size=11), width=100,
        )
        self.tick_label.pack(side="right", padx=10)

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
        matches = self.controller.get_available_matches(w)
        if matches:
            self.match_combo.configure(values=matches)
            self.match_var.set(matches[0])
        else:
            self.match_combo.configure(values=["—"])
            self.match_var.set("—")

    def _on_load(self):
        try:
            w = float(self.w_error_var.get())
        except ValueError:
            return
        mid = self.match_var.get()
        if mid == "—":
            return
        self.load_btn.configure(state="disabled", text="Chargement…")
        self.controller.load_overlay(w, mid)

    def on_overlay_loaded(self, data):
        self.load_btn.configure(state="normal", text="Charger")
        if data is None:
            return
        self._overlay_data = data
        self.map_canvas.set_background(data.canvas_image)
        self.map_canvas.set_segments(data.player_segments, data.min_tick, data.max_tick)
        self.tick_slider.configure(from_=data.min_tick, to=data.max_tick)
        self.tick_slider.set(data.max_tick)
        self.tick_label.configure(text=f"tick: {data.max_tick}")
        self._refresh_stats()

    def _on_slider(self, val):
        tick = int(float(val))
        self.tick_label.configure(text=f"tick: {tick}")
        self.map_canvas.set_tick(tick)
        self._refresh_stats()

    def _refresh_stats(self):
        self.stats_panel.update_from(self.map_canvas.get_stats())


# ═════════════════════════════════════════════════════════════════════
# CompressPage — inchangée
# ═════════════════════════════════════════════════════════════════════


class CompressPage(BasePage):
    """Page de lancement de compression."""

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self, text="Compression MDL",
            font=ctk.CTkFont(size=24, weight="bold"), text_color=ACCENT,
        ).pack(pady=(30, 20))

        form = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        form.pack(padx=40, pady=10, fill="x")

        row1 = ctk.CTkFrame(form, fg_color="transparent")
        row1.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(row1, text="w_error:", width=100, anchor="w").pack(side="left")
        self.w_entry = ctk.CTkEntry(row1, width=120, placeholder_text="12.0")
        self.w_entry.pack(side="left", padx=10)
        self.w_entry.insert(0, "12.0")

        row2 = ctk.CTkFrame(form, fg_color="transparent")
        row2.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row2, text="Match ID:", width=100, anchor="w").pack(side="left")
        self.mid_entry = ctk.CTkEntry(row2, width=200, placeholder_text="(vide = tous)")
        self.mid_entry.pack(side="left", padx=10)

        self.run_btn = ctk.CTkButton(
            form, text="Lancer Compression", fg_color=ACCENT,
            hover_color="#c33750", command=self._on_run,
        )
        self.run_btn.pack(pady=20)

        self.progress = ctk.CTkProgressBar(self, width=400)
        self.progress.pack(pady=10)
        self.progress.set(0)

        self.log_text = ctk.CTkTextbox(self, height=250, fg_color=BG_CARD, corner_radius=10)
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(5, 15))

    def _on_run(self):
        try:
            w = float(self.w_entry.get())
        except ValueError:
            return
        mid = self.mid_entry.get().strip() or None
        self.run_btn.configure(state="disabled", text="En cours…")
        self.log_text.delete("1.0", "end")
        self.progress.set(0)
        self.controller.start_compression(w, mid)

    def on_compress_progress(self, current, total, result):
        self.progress.set(current / max(total, 1))
        status = "OK" if result.success else f"ERREUR: {result.error}"
        line = f"[{current}/{total}] {result.match_id}: {status}"
        if result.success:
            line += f"  — {result.reduction_pct:.1f}% compression, {result.size_kb} KB"
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")

    def on_compress_done(self, results):
        self.run_btn.configure(state="normal", text="Lancer Compression")
        ok = sum(1 for r in results if r.success)
        self.log_text.insert("end", f"\n{'='*50}\nTerminé: {ok}/{len(results)} matchs compressés.\n")
        self.log_text.see("end")


# ═════════════════════════════════════════════════════════════════════
# ClusterPage — inchangée
# ═════════════════════════════════════════════════════════════════════


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

        ctk.CTkLabel(top, text="w_error:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(15, 5))
        self.w_error_var = ctk.StringVar(value="12.0")
        self.w_error_combo = ctk.CTkComboBox(
            top, variable=self.w_error_var, values=["12.0"], width=100,
            command=self._on_w_error_change,
        )
        self.w_error_combo.pack(side="left", padx=5)

        ctk.CTkLabel(top, text="Cluster:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(20, 5))
        self.cluster_var = ctk.StringVar(value="0")
        self.cluster_combo = ctk.CTkComboBox(
            top, variable=self.cluster_var, values=["0"], width=100,
        )
        self.cluster_combo.pack(side="left", padx=5)

        self.load_btn = ctk.CTkButton(
            top, text="Afficher", fg_color=ACCENT, hover_color="#c33750",
            command=self._on_load,
        )
        self.load_btn.pack(side="left", padx=15)

        self.new_cluster_btn = ctk.CTkButton(
            top, text="Nouveau Clustering", fg_color=ACCENT2, hover_color="#1a4a80",
            command=self._show_cluster_form,
        )
        self.new_cluster_btn.pack(side="left", padx=5)

        self.info_label = ctk.CTkLabel(
            top, text="", font=ctk.CTkFont(size=12), text_color=TEXT_DIM,
        )
        self.info_label.pack(side="right", padx=15)

        # Panneau "pas de clustering"
        self.no_cluster_frame = ctk.CTkFrame(self, fg_color="transparent")

        ctk.CTkLabel(
            self.no_cluster_frame, text="Aucun résultat de clustering trouvé",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT_LIGHT,
        ).pack(pady=(80, 10))
        ctk.CTkLabel(
            self.no_cluster_frame,
            text="Lancez d'abord le clustering sur les données compressées.",
            font=ctk.CTkFont(size=13), text_color=TEXT_DIM,
        ).pack(pady=(0, 25))

        cluster_form = ctk.CTkFrame(self.no_cluster_frame, fg_color=BG_CARD, corner_radius=12)
        cluster_form.pack(padx=60, fill="x")

        row_algo = ctk.CTkFrame(cluster_form, fg_color="transparent")
        row_algo.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(row_algo, text="Algorithme:", width=120, anchor="w").pack(side="left")
        self.algo_var = ctk.StringVar(value="kmeans")
        ctk.CTkOptionMenu(
            row_algo, variable=self.algo_var,
            values=["kmeans", "affinity", "kmedoids"], width=150,
        ).pack(side="left", padx=10)

        row_mf = ctk.CTkFrame(cluster_form, fg_color="transparent")
        row_mf.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_mf, text="Max fichiers:", width=120, anchor="w").pack(side="left")
        self.maxfiles_entry = ctk.CTkEntry(row_mf, width=100, placeholder_text="10")
        self.maxfiles_entry.pack(side="left", padx=10)
        self.maxfiles_entry.insert(0, "10")

        row_nc = ctk.CTkFrame(cluster_form, fg_color="transparent")
        row_nc.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(row_nc, text="Nb clusters:", width=120, anchor="w").pack(side="left")
        self.nclusters_entry = ctk.CTkEntry(row_nc, width=100, placeholder_text="50")
        self.nclusters_entry.pack(side="left", padx=10)
        self.nclusters_entry.insert(0, "50")

        self.run_cluster_btn = ctk.CTkButton(
            cluster_form, text="Lancer Clustering", fg_color=ACCENT, hover_color="#c33750",
            command=self._on_run_clustering,
        )
        self.run_cluster_btn.pack(pady=20)

        self.cluster_log = ctk.CTkLabel(
            self.no_cluster_frame, text="",
            font=ctk.CTkFont(size=12), text_color=TEXT_DIM, wraplength=500,
        )
        self.cluster_log.pack(pady=10)

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
        max_f = int(self.maxfiles_entry.get()) if self.maxfiles_entry.get().strip() else None
        n_c = int(self.nclusters_entry.get()) if self.nclusters_entry.get().strip() else 50
        algo = self.algo_var.get()
        self.run_cluster_btn.configure(state="disabled", text="En cours…")
        self.cluster_log.configure(text="Clustering en cours, veuillez patienter…")
        self.controller.start_clustering(w, algo=algo, max_files=max_f, n_clusters=n_c)

    def on_clustering_done(self, success, error_msg):
        self.run_cluster_btn.configure(state="normal", text="Lancer Clustering")
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
        self.info_label.configure(text=f"Cluster #{data.cluster_id} — {data.total_in_cluster} segments")


# ═════════════════════════════════════════════════════════════════════
# ComparisonPage — avec vitesse réglable, légende, stats
# ═════════════════════════════════════════════════════════════════════


class ComparisonPage(BasePage):
    """Comparaison côte à côte : trajectoires brutes vs compressées."""

    _SPEED_MAP = {"×0.5": 0.5, "×1": 1.0, "×2": 2.0, "×4": 4.0}
    _BASE_TICK_STEP = 300
    PLAY_INTERVAL_MS = 50

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._playing = False
        self._comparison_data = None
        self._speed = 1.0
        self._build()

    def _build(self):
        # ── Barre d'options + vitesse ────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10, height=60)
        top.pack(fill="x", padx=10, pady=(10, 5))
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="w_error:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(15, 5))
        self.w_error_var = ctk.StringVar(value="12.0")
        self.w_error_combo = ctk.CTkComboBox(
            top, variable=self.w_error_var, values=["12.0"], width=100,
            command=self._on_w_error_change,
        )
        self.w_error_combo.pack(side="left", padx=5)

        ctk.CTkLabel(top, text="Match:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(20, 5))
        self.match_var = ctk.StringVar(value="—")
        self.match_combo = ctk.CTkComboBox(top, variable=self.match_var, values=["—"], width=180)
        self.match_combo.pack(side="left", padx=5)

        self.load_btn = ctk.CTkButton(
            top, text="Charger", fg_color=ACCENT, hover_color="#c33750", command=self._on_load,
        )
        self.load_btn.pack(side="left", padx=15)

        # Sélecteur de vitesse
        ctk.CTkLabel(top, text="Vitesse:", font=ctk.CTkFont(size=13)).pack(side="left", padx=(15, 5))
        self.speed_var = ctk.StringVar(value="×1")
        speed_menu = ctk.CTkOptionMenu(
            top, variable=self.speed_var,
            values=list(self._SPEED_MAP.keys()), width=70,
            command=self._on_speed_change,
        )
        speed_menu.pack(side="left", padx=5)

        # ── Légende horizontale compacte ─────────────────────────────────
        legend_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8, height=28)
        legend_bar.pack(fill="x", padx=10, pady=(2, 2))
        legend_bar.pack_propagate(False)

        ctk.CTkLabel(
            legend_bar, text="Radiant:", font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#3498db",
        ).pack(side="left", padx=(10, 4))
        for i in range(5):
            ctk.CTkLabel(
                legend_bar, text=f"● J{i+1}",
                font=ctk.CTkFont(size=9), text_color=PLAYER_COLORS[i],
            ).pack(side="left", padx=2)

        ctk.CTkLabel(
            legend_bar, text="  Dire:", font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#e74c3c",
        ).pack(side="left", padx=(12, 4))
        for i in range(5, 10):
            ctk.CTkLabel(
                legend_bar, text=f"● J{i-4}",
                font=ctk.CTkFont(size=9), text_color=PLAYER_COLORS[i],
            ).pack(side="left", padx=2)

        # ── Zone des deux cartes ─────────────────────────────────────────
        maps_frame = ctk.CTkFrame(self, fg_color="transparent")
        maps_frame.pack(fill="both", expand=True, padx=10, pady=2)
        maps_frame.columnconfigure(0, weight=1)
        maps_frame.columnconfigure(1, weight=1)
        maps_frame.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            maps_frame, text="Brut (CSV)", font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_LIGHT,
        ).grid(row=0, column=0, pady=(0, 1))
        ctk.CTkLabel(
            maps_frame, text="Compressé", font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_LIGHT,
        ).grid(row=0, column=1, pady=(0, 1))

        self.map_raw = DotaMapCanvas(maps_frame, show_dots=True)
        self.map_raw.grid(row=1, column=0, sticky="nsew", padx=(0, 3))

        self.map_compressed = DotaMapCanvas(maps_frame, show_dots=True)
        self.map_compressed.grid(row=1, column=1, sticky="nsew", padx=(3, 0))

        # ── Contrôles ────────────────────────────────────────────────────
        controls = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10, height=50)
        controls.pack(fill="x", padx=10, pady=(2, 2))
        controls.pack_propagate(False)

        self.play_btn = ctk.CTkButton(
            controls, text="▶  Play", width=100,
            fg_color=ACCENT, hover_color="#c33750", command=self._toggle_play,
        )
        self.play_btn.pack(side="left", padx=(15, 10))

        self.tick_slider = ctk.CTkSlider(controls, from_=0, to=1, command=self._on_slider)
        self.tick_slider.pack(side="left", fill="x", expand=True, padx=10)

        self.tick_label = ctk.CTkLabel(
            controls, text="tick: 0", font=ctk.CTkFont(size=11), width=110,
        )
        self.tick_label.pack(side="right", padx=10)

        # ── Stats en bas ─────────────────────────────────────────────────
        stats_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8, height=30)
        stats_bar.pack(fill="x", padx=10, pady=(2, 8))
        stats_bar.pack_propagate(False)

        self._stat_raw = ctk.CTkLabel(
            stats_bar, text="Brut: — segments", font=ctk.CTkFont(size=10), text_color=TEXT_DIM,
        )
        self._stat_raw.pack(side="left", padx=15)

        self._stat_comp = ctk.CTkLabel(
            stats_bar, text="Compressé: — segments", font=ctk.CTkFont(size=10), text_color=TEXT_DIM,
        )
        self._stat_comp.pack(side="left", padx=15)

        self._stat_time = ctk.CTkLabel(
            stats_bar, text="Temps: 00:00", font=ctk.CTkFont(size=10), text_color=TEXT_DIM,
        )
        self._stat_time.pack(side="right", padx=15)

        self._stat_players = ctk.CTkLabel(
            stats_bar, text="Joueurs actifs: 0", font=ctk.CTkFont(size=10), text_color=TEXT_DIM,
        )
        self._stat_players.pack(side="right", padx=15)

    # ── Combos ───────────────────────────────────────────────────────────

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
        matches = self.controller.get_available_matches(w)
        if matches:
            self.match_combo.configure(values=matches)
            self.match_var.set(matches[0])
        else:
            self.match_combo.configure(values=["—"])
            self.match_var.set("—")

    def _on_speed_change(self, val):
        self._speed = self._SPEED_MAP.get(val, 1.0)

    # ── Chargement ───────────────────────────────────────────────────────

    def _on_load(self):
        try:
            w = float(self.w_error_var.get())
        except ValueError:
            return
        mid = self.match_var.get()
        if mid == "—":
            return
        self._stop_play()
        self.load_btn.configure(state="disabled", text="Chargement…")
        self.controller.load_comparison(w, mid)

    def on_comparison_loaded(self, data):
        self.load_btn.configure(state="normal", text="Charger")
        if data is None:
            return
        self._comparison_data = data

        self.map_raw.set_background(data.canvas_image)
        self.map_raw.set_raw_points(data.raw_points, data.min_tick, data.max_tick)

        self.map_compressed.set_background(data.canvas_image)
        self.map_compressed.set_segments(data.compressed_segments, data.min_tick, data.max_tick)

        self.tick_slider.configure(from_=data.min_tick, to=data.max_tick)
        self.tick_slider.set(data.min_tick)
        self.tick_label.configure(text=f"tick: {data.min_tick}")

        self.map_raw.set_tick(data.min_tick)
        self.map_compressed.set_tick(data.min_tick)
        self._refresh_stats()

    # ── Slider ───────────────────────────────────────────────────────────

    def _on_slider(self, val):
        tick = int(float(val))
        self.tick_label.configure(text=f"tick: {tick}")
        self.map_raw.set_tick(tick)
        self.map_compressed.set_tick(tick)
        self._refresh_stats()

    # ── Play / Stop ──────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if self._comparison_data is None:
            return
        self._playing = True
        self.play_btn.configure(text="⏹  Stop", fg_color="#c33750")
        self._play_tick()

    def _stop_play(self):
        self._playing = False
        self.play_btn.configure(text="▶  Play", fg_color=ACCENT)

    def _play_tick(self):
        if not self._playing or self._comparison_data is None:
            return
        current = int(float(self.tick_slider.get()))
        step = int(self._BASE_TICK_STEP * self._speed)
        new_tick = current + step
        if new_tick >= self._comparison_data.max_tick:
            new_tick = self._comparison_data.max_tick
            self._stop_play()

        self.tick_slider.set(new_tick)
        self._on_slider(new_tick)

        if self._playing:
            self.after(self.PLAY_INTERVAL_MS, self._play_tick)

    # ── Stats ────────────────────────────────────────────────────────────

    def _refresh_stats(self):
        sr = self.map_raw.get_stats()
        sc = self.map_compressed.get_stats()
        self._stat_raw.configure(
            text=f"Brut: {sr['visible_segments']}/{sr['total_segments']} segments")
        self._stat_comp.configure(
            text=f"Compressé: {sc['visible_segments']}/{sc['total_segments']} segments")
        self._stat_players.configure(text=f"Joueurs actifs: {sr['active_players']}")
        m, s = divmod(int(sr["elapsed_sec"]), 60)
        self._stat_time.configure(text=f"Temps: {m:02d}:{s:02d}")


# ═══════════════════════════════════════════════════════════════════════════
# Fenêtre principale avec sidebar et transitions animées
# ═══════════════════════════════════════════════════════════════════════════


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
        self._pages["overlay"] = OverlayPage(self.content_frame, controller, self.switch_page)
        self._pages["compress"] = CompressPage(self.content_frame, controller, self.switch_page)
        self._pages["cluster"] = ClusterPage(self.content_frame, controller, self.switch_page)
        self._pages["comparison"] = ComparisonPage(self.content_frame, controller, self.switch_page)

        self.switch_page("menu", animate=False)

    def _build_layout(self):
        self.sidebar = ctk.CTkFrame(self, width=200, fg_color=BG_CARD, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar, text="DOTA 2",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=ACCENT,
        ).pack(pady=(25, 2))
        ctk.CTkLabel(
            self.sidebar, text="Trajectories",
            font=ctk.CTkFont(size=13), text_color=TEXT_DIM,
        ).pack(pady=(0, 30))

        nav_items = [
            ("Accueil", "menu"),
            ("Overlay Carte", "overlay"),
            ("Compression", "compress"),
            ("Clusters", "cluster"),
            ("Comparaison", "comparison"),
        ]
        self._nav_buttons = {}
        for label, page_name in nav_items:
            btn = ctk.CTkButton(
                self.sidebar, text=label,
                fg_color="transparent", text_color=TEXT_LIGHT,
                hover_color=ACCENT2, anchor="w",
                height=40, corner_radius=8,
                command=lambda p=page_name: self.switch_page(p),
            )
            btn.pack(fill="x", padx=12, pady=3)
            self._nav_buttons[page_name] = btn

        ctk.CTkFrame(self.sidebar, height=1, fg_color=TEXT_DIM).pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            self.sidebar, text="v1.1.0",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM,
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
