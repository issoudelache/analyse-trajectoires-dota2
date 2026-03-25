"""
OverlayPage — Visualisation des trajectoires compressées sur la carte.
"""

import customtkinter as ctk

from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, BG_CARD
from mvc.views.widgets.map_canvas import DotaMapCanvas
from mvc.views.widgets.player_legend import PlayerLegend
from mvc.views.widgets.stats_panel import StatsPanel


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

        ctk.CTkLabel(top, text="w_error:", font=ctk.CTkFont(size=13)).pack(
            side="left", padx=(15, 5)
        )
        self.w_error_var = ctk.StringVar(value="12.0")
        self.w_error_combo = ctk.CTkComboBox(
            top,
            variable=self.w_error_var,
            values=["12.0"],
            width=100,
            command=self._on_w_error_change,
        )
        self.w_error_combo.pack(side="left", padx=5)

        ctk.CTkLabel(top, text="Match:", font=ctk.CTkFont(size=13)).pack(
            side="left", padx=(20, 5)
        )
        self.match_var = ctk.StringVar(value="—")
        self.match_combo = ctk.CTkComboBox(
            top, variable=self.match_var, values=["—"], width=180
        )
        self.match_combo.pack(side="left", padx=5)

        self.load_btn = ctk.CTkButton(
            top,
            text="Charger",
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._on_load,
        )
        self.load_btn.pack(side="left", padx=15)

        self.export_btn = ctk.CTkButton(
            top,
            text="Exporter JPG",
            fg_color="#0f3460",
            hover_color="#1a4a80",
            command=self._on_export,
            width=110,
        )
        self.export_btn.pack(side="left", padx=5)

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

        self.legend = PlayerLegend(right, on_toggle_callback=self._on_player_toggle)
        self.legend.pack(fill="x", pady=(0, 6))

        self.stats_panel = StatsPanel(right)
        self.stats_panel.pack(fill="x")

        # ── Slider temporel ──────────────────────────────────────────────
        slider_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10, height=50)
        slider_frame.pack(fill="x", padx=10, pady=(5, 10))
        slider_frame.pack_propagate(False)

        ctk.CTkLabel(slider_frame, text="Temps:", font=ctk.CTkFont(size=12)).pack(
            side="left", padx=(15, 5)
        )
        self.tick_slider = ctk.CTkSlider(
            slider_frame, from_=0, to=1, command=self._on_slider
        )
        self.tick_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.tick_label = ctk.CTkLabel(
            slider_frame,
            text="tick: 0",
            font=ctk.CTkFont(size=11),
            width=100,
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

    def _on_player_toggle(self, pid, visible):
        self.map_canvas.set_player_visibility(pid, visible)

    def _on_export(self):
        if self._overlay_data is None:
            return
        from mvc.config import OUTPUT_DIR

        out_dir = OUTPUT_DIR / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        tick = int(float(self.tick_slider.get()))
        path = out_dir / f"overlay_{self._overlay_data.match_id}_t{tick}.jpg"
        self.map_canvas.export_to_jpg(str(path))
        self.export_btn.configure(text="Exporté !", fg_color="#27ae60")
        self.after(1500, lambda: self.export_btn.configure(text="Exporter JPG", fg_color="#0f3460"))

    def _refresh_stats(self):
        self.stats_panel.update_from(self.map_canvas.get_stats())
