"""
ComparisonPage — Comparaison côte à côte brut vs compressé.
"""

import customtkinter as ctk

from dota_analytics.plotting import PLAYER_COLORS
from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, BG_CARD, TEXT_DIM, TEXT_LIGHT
from mvc.views.widgets.map_canvas import DotaMapCanvas


class ComparisonPage(BasePage):
    """Comparaison côte à côte : trajectoires brutes vs compressées."""

    _SPEED_MAP = {"×0.5": 0.5, "×1": 1.0, "×2": 2.0, "×4": 4.0}
    _BASE_TICK_STEP = 300
    PLAY_INTERVAL_MS = 50
    _TICK_JUMP = 1000  # ticks par flèche clavier

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._playing = False
        self._paused = False
        self._comparison_data = None
        self._speed = 1.0
        self._build()
        self._bind_keys()

    def _build(self):
        # ── Barre d'options + vitesse ────────────────────────────────────
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

        # Sélecteur de vitesse
        ctk.CTkLabel(top, text="Vitesse:", font=ctk.CTkFont(size=13)).pack(
            side="left", padx=(15, 5)
        )
        self.speed_var = ctk.StringVar(value="×1")
        speed_menu = ctk.CTkOptionMenu(
            top,
            variable=self.speed_var,
            values=list(self._SPEED_MAP.keys()),
            width=70,
            command=self._on_speed_change,
        )
        speed_menu.pack(side="left", padx=5)

        self.export_btn = ctk.CTkButton(
            top,
            text="Exporter JPG",
            fg_color="#0f3460",
            hover_color="#1a4a80",
            command=self._on_export,
            width=110,
        )
        self.export_btn.pack(side="left", padx=(15, 5))

        # ── Légende horizontale compacte ─────────────────────────────────
        legend_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8, height=28)
        legend_bar.pack(fill="x", padx=10, pady=(2, 2))
        legend_bar.pack_propagate(False)

        ctk.CTkLabel(
            legend_bar,
            text="Radiant:",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#3498db",
        ).pack(side="left", padx=(10, 4))
        for i in range(5):
            ctk.CTkLabel(
                legend_bar,
                text=f"● J{i + 1}",
                font=ctk.CTkFont(size=9),
                text_color=PLAYER_COLORS[i],
            ).pack(side="left", padx=2)

        ctk.CTkLabel(
            legend_bar,
            text="  Dire:",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#e74c3c",
        ).pack(side="left", padx=(12, 4))
        for i in range(5, 10):
            ctk.CTkLabel(
                legend_bar,
                text=f"● J{i - 4}",
                font=ctk.CTkFont(size=9),
                text_color=PLAYER_COLORS[i],
            ).pack(side="left", padx=2)

        # ── Zone des deux cartes ─────────────────────────────────────────
        maps_frame = ctk.CTkFrame(self, fg_color="transparent")
        maps_frame.pack(fill="both", expand=True, padx=10, pady=2)
        maps_frame.columnconfigure(0, weight=1)
        maps_frame.columnconfigure(1, weight=1)
        maps_frame.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            maps_frame,
            text="Brut (CSV)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_LIGHT,
        ).grid(row=0, column=0, pady=(0, 1))
        ctk.CTkLabel(
            maps_frame,
            text="Compressé",
            font=ctk.CTkFont(size=14, weight="bold"),
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
            controls,
            text="▶  Play",
            width=100,
            fg_color=ACCENT,
            hover_color="#c33750",
            command=self._toggle_play,
        )
        self.play_btn.pack(side="left", padx=(15, 10))

        self.tick_slider = ctk.CTkSlider(
            controls, from_=0, to=1, command=self._on_slider
        )
        self.tick_slider.pack(side="left", fill="x", expand=True, padx=10)

        self.tick_label = ctk.CTkLabel(
            controls,
            text="tick: 0",
            font=ctk.CTkFont(size=11),
            width=110,
        )
        self.tick_label.pack(side="right", padx=10)

        # ── Stats en bas ─────────────────────────────────────────────────
        stats_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8, height=30)
        stats_bar.pack(fill="x", padx=10, pady=(2, 8))
        stats_bar.pack_propagate(False)

        self._stat_raw = ctk.CTkLabel(
            stats_bar,
            text="Brut: — segments",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM,
        )
        self._stat_raw.pack(side="left", padx=15)

        self._stat_comp = ctk.CTkLabel(
            stats_bar,
            text="Compressé: — segments",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM,
        )
        self._stat_comp.pack(side="left", padx=15)

        self._stat_time = ctk.CTkLabel(
            stats_bar,
            text="Temps: 00:00",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM,
        )
        self._stat_time.pack(side="right", padx=15)

        self._stat_players = ctk.CTkLabel(
            stats_bar,
            text="Joueurs actifs: 0",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM,
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
        self.map_compressed.set_segments(
            data.compressed_segments, data.min_tick, data.max_tick
        )

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

    # ── Play / Pause / Stop ────────────────────────────────────────────

    def _toggle_play(self):
        if self._playing and not self._paused:
            self._pause_play()
        elif self._paused:
            self._resume_play()
        else:
            self._start_play()

    def _start_play(self):
        if self._comparison_data is None:
            return
        self._playing = True
        self._paused = False
        self.play_btn.configure(text="⏸  Pause", fg_color="#f39c12")
        self._play_tick()

    def _pause_play(self):
        self._paused = True
        self.play_btn.configure(text="▶  Reprendre", fg_color="#27ae60")

    def _resume_play(self):
        self._paused = False
        self.play_btn.configure(text="⏸  Pause", fg_color="#f39c12")
        self._play_tick()

    def _stop_play(self):
        self._playing = False
        self._paused = False
        self.play_btn.configure(text="▶  Play", fg_color=ACCENT)

    def _play_tick(self):
        if not self._playing or self._paused or self._comparison_data is None:
            return
        current = int(float(self.tick_slider.get()))
        step = int(self._BASE_TICK_STEP * self._speed)
        new_tick = current + step
        if new_tick >= self._comparison_data.max_tick:
            new_tick = self._comparison_data.max_tick
            self._stop_play()

        self.tick_slider.set(new_tick)
        self._on_slider(new_tick)

        if self._playing and not self._paused:
            self.after(self.PLAY_INTERVAL_MS, self._play_tick)

    # ── Keyboard shortcuts ───────────────────────────────────────────────

    def _bind_keys(self):
        root = self.winfo_toplevel()
        root.bind("<space>", self._on_space)
        root.bind("<Left>", self._on_left)
        root.bind("<Right>", self._on_right)

    def _on_space(self, event=None):
        self._toggle_play()

    def _on_left(self, event=None):
        if self._comparison_data is None:
            return
        current = int(float(self.tick_slider.get()))
        new_tick = max(self._comparison_data.min_tick, current - self._TICK_JUMP)
        self.tick_slider.set(new_tick)
        self._on_slider(new_tick)

    def _on_right(self, event=None):
        if self._comparison_data is None:
            return
        current = int(float(self.tick_slider.get()))
        new_tick = min(self._comparison_data.max_tick, current + self._TICK_JUMP)
        self.tick_slider.set(new_tick)
        self._on_slider(new_tick)

    # ── Export JPG ───────────────────────────────────────────────────────

    def _on_export(self):
        if self._comparison_data is None:
            return
        from mvc.config import OUTPUT_DIR

        out_dir = OUTPUT_DIR / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        tick = int(float(self.tick_slider.get()))
        mid = self._comparison_data.match_id

        path_raw = out_dir / f"comparison_raw_{mid}_t{tick}.jpg"
        path_comp = out_dir / f"comparison_compressed_{mid}_t{tick}.jpg"
        self.map_raw.export_to_jpg(str(path_raw))
        self.map_compressed.export_to_jpg(str(path_comp))
        self.export_btn.configure(text="Exporté !", fg_color="#27ae60")
        self.after(1500, lambda: self.export_btn.configure(text="Exporter JPG", fg_color="#0f3460"))

    # ── Stats ────────────────────────────────────────────────────────────

    def _refresh_stats(self):
        sr = self.map_raw.get_stats()
        sc = self.map_compressed.get_stats()
        self._stat_raw.configure(
            text=f"Brut: {sr['visible_segments']}/{sr['total_segments']} segments"
        )
        self._stat_comp.configure(
            text=f"Compressé: {sc['visible_segments']}/{sc['total_segments']} segments"
        )
        self._stat_players.configure(text=f"Joueurs actifs: {sr['active_players']}")
        m, s = divmod(int(sr["elapsed_sec"]), 60)
        self._stat_time.configure(text=f"Temps: {m:02d}:{s:02d}")
