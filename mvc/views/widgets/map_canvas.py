"""
DotaMapCanvas — Canvas haute performance pour la carte Dota 2.

Optimisations clés :
- Image de fond redimensionnée uniquement sur resize (cache PIL)
- Lignes pré-créées comme items cachés, visibilité basculée par tick
- Mise à jour O(delta) entre frames consécutives
- Throttle via after_idle pour éviter les redraws multiples
- Dots de position animés + tooltips
"""

import bisect
import math
import tkinter as tk
from typing import Dict, Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from dota_analytics.plotting import PLAYER_COLORS
from mvc.views.theme import PLAYER_NAMES, TEXT_DIM, TEXT_LIGHT


class DotaMapCanvas(ctk.CTkFrame):
    """Canvas haute performance pour la carte Dota 2."""

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
        self._lines: Dict[int, list] = {}  # pid -> [(x1,y1,x2,y2,s_tick,e_tick), ...]
        self._line_ids: Dict[int, list] = {}  # pid -> [canvas_id, ...]
        self._end_ticks: Dict[int, list] = {}  # pid -> [end_tick, ...] trié
        self._vis_count: Dict[int, int] = {}  # pid -> nb lignes visibles
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
                self._ox,
                self._oy,
                anchor="nw",
                image=self._bg_photo,
                tags="bg",
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
                    lines.append(
                        (
                            seg["start"]["x"],
                            seg["start"]["y"],
                            seg["end"]["x"],
                            seg["end"]["y"],
                            seg["start"]["tick"],
                            seg["end"]["tick"],
                        )
                    )
                lines.sort(key=lambda l: l[5])
                source[pid] = lines
        elif self._draw_mode == "raw":
            for pid, pts in self._raw_points_data.items():
                lines = []
                for i in range(len(pts) - 1):
                    p1, p2 = pts[i], pts[i + 1]
                    lines.append(
                        (
                            p1["x"],
                            p1["y"],
                            p2["x"],
                            p2["y"],
                            p1["tick"],
                            p2["tick"],
                        )
                    )
                source[pid] = lines

        # Pré-créer tous les items canvas comme cachés
        for pid, lines in source.items():
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
            self._lines[pid] = lines
            ids = []
            ticks = []
            for x1, y1, x2, y2, s_tick, e_tick in lines:
                cid = self.canvas.create_line(
                    self._gx(x1),
                    self._gy(y1),
                    self._gx(x2),
                    self._gy(y2),
                    fill=color,
                    width=2,
                    state="hidden",
                    tags="trail",
                )
                ids.append(cid)
                ticks.append(e_tick)
                self._item_info[cid] = {
                    "pid": pid,
                    "name": PLAYER_NAMES[pid % 10],
                    "s_tick": s_tick,
                    "e_tick": e_tick,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            self._line_ids[pid] = ids
            self._end_ticks[pid] = ticks
            self._vis_count[pid] = 0

            # Ligne d'interpolation (tête)
            self._head_ids[pid] = self.canvas.create_line(
                0,
                0,
                0,
                0,
                fill=color,
                width=2,
                state="hidden",
                tags="head",
            )
            # Dot de position
            if self._show_dots:
                self._dot_ids[pid] = self.canvas.create_oval(
                    0,
                    0,
                    0,
                    0,
                    fill=color,
                    outline="white",
                    width=1,
                    state="hidden",
                    tags="dot",
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
                            head,
                            self._gx(x1),
                            self._gy(y1),
                            self._gx(ix),
                            self._gy(iy),
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
                    self._gx(x1),
                    self._gy(y1),
                    self._gx(x2),
                    self._gy(y2),
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
            x + 6,
            y + 4,
            text=text,
            anchor="nw",
            fill=TEXT_LIGHT,
            font=("Consolas", 9),
            tags="tooltip",
        )
        bb = self.canvas.bbox(tid)
        if bb:
            p = 5
            self.canvas.create_rectangle(
                bb[0] - p,
                bb[1] - p,
                bb[2] + p,
                bb[3] + p,
                fill="#1a1a2eee",
                outline=TEXT_DIM,
                tags="tooltip",
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
                self._gx(x1),
                self._gy(y1),
                self._gx(x2),
                self._gy(y2),
                fill=color,
                width=2,
                tags="trail",
            )

    # ── Player visibility toggle ────────────────────────────────────────

    def set_player_visibility(self, pid: int, visible: bool):
        """Affiche ou masque tous les éléments d'un joueur."""
        state = "normal" if visible else "hidden"
        if pid in self._line_ids:
            vis_n = self._vis_count.get(pid, 0)
            for i in range(vis_n):
                self.canvas.itemconfigure(self._line_ids[pid][i], state=state)
        if pid in self._head_ids:
            self.canvas.itemconfigure(self._head_ids[pid], state="hidden" if not visible else self.canvas.itemcget(self._head_ids[pid], "state"))
        if pid in self._dot_ids:
            self.canvas.itemconfigure(self._dot_ids[pid], state="hidden" if not visible else self.canvas.itemcget(self._dot_ids[pid], "state"))

    # ── Export JPG ────────────────────────────────────────────────────────

    def export_to_jpg(self, filepath: str):
        """Exporte le contenu visible du canvas en fichier JPG."""
        import io
        from PIL import Image as PILImage

        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Recréer l'image à partir du fond + segments visibles
        img = PILImage.new("RGB", (cw, ch), "#0d0d0d")

        # Dessiner le fond
        if self._canvas_img and self._map_size > 10:
            bg = self._canvas_img.resize(
                (self._map_size, self._map_size), PILImage.LANCZOS
            )
            img.paste(bg, (self._ox, self._oy))

        # Dessiner les lignes visibles
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        for pid, ids in self._line_ids.items():
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
            vis_n = self._vis_count.get(pid, 0)
            lines = self._lines.get(pid, [])
            for i in range(vis_n):
                x1, y1, x2, y2 = lines[i][0], lines[i][1], lines[i][2], lines[i][3]
                draw.line(
                    [
                        (self._gx(x1), self._gy(y1)),
                        (self._gx(x2), self._gy(y2)),
                    ],
                    fill=color,
                    width=2,
                )

        # Fallback pour draw_raw_segments (cluster mode)
        if self._draw_mode == "none":
            items = self.canvas.find_withtag("trail")
            for item_id in items:
                coords = self.canvas.coords(item_id)
                if len(coords) >= 4:
                    fill = self.canvas.itemcget(item_id, "fill")
                    draw.line(
                        [(coords[0], coords[1]), (coords[2], coords[3])],
                        fill=fill or "white",
                        width=2,
                    )

        img.save(filepath, "JPEG", quality=95)

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
