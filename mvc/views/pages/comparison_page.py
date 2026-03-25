"""
ComparisonPage — Comparaison côte à côte brut vs compressé.
"""

import json
from pathlib import Path

import customtkinter as ctk

from dota_analytics.plotting import PLAYER_COLORS
from mvc.views.pages.base_page import BasePage
from mvc.views.theme import ACCENT, ACCENT2, BG_CARD, TEXT_DIM, TEXT_LIGHT
from mvc.views.widgets.map_canvas import DotaMapCanvas


class ComparisonPage(BasePage):
    """Comparaison côte à côte : trajectoires brutes vs compressées."""

    _SPEED_MAP = {"×0.5": 0.5, "×1": 1.0, "×2": 2.0, "×4": 4.0}
    _BASE_TICK_STEP = 300
    PLAY_INTERVAL_MS = 50
    _TICK_JUMP = 1000  # ticks par flèche clavier
    _REC_STEP = 300  # ticks entre chaque frame enregistrée

    def __init__(self, master, controller, switch_page_cb):
        super().__init__(master, controller)
        self.switch_page = switch_page_cb
        self._playing = False
        self._paused = False
        self._comparison_data = None
        self._speed = 1.0
        # Recording state
        self._recording = False
        self._rec_idx = 0
        self._rec_ticks: list = []
        self._rec_dir: Path | None = None
        # Playback-from-recording state
        self._playback_mode = False  # True = lecture d'enregistrement
        self._pb_frames_raw: list = []  # PIL images
        self._pb_frames_comp: list = []
        self._pb_ticks: list = []
        self._pb_idx = 0
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

        self.rec_btn = ctk.CTkButton(
            top,
            text="Enregistrer",
            fg_color=ACCENT2,
            hover_color="#1a4a80",
            command=self._on_record,
            width=110,
        )
        self.rec_btn.pack(side="left", padx=5)

        self.load_rec_btn = ctk.CTkButton(
            top,
            text="Charger Enreg.",
            fg_color=ACCENT2,
            hover_color="#1a4a80",
            command=self._on_load_recording,
            width=120,
        )
        self.load_rec_btn.pack(side="left", padx=5)

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

        # ── Barre de progression enregistrement ──────────────────────────
        self.rec_bar_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8, height=30)
        # pas pack() — on l'affiche seulement pendant l'enregistrement
        self.rec_bar_frame.pack_propagate(False)
        self.rec_progress = ctk.CTkProgressBar(self.rec_bar_frame, width=500, mode="determinate")
        self.rec_progress.set(0)
        self.rec_progress.pack(side="left", padx=15, pady=5)
        self.rec_label = ctk.CTkLabel(
            self.rec_bar_frame, text="", font=ctk.CTkFont(size=10), text_color=TEXT_DIM
        )
        self.rec_label.pack(side="left", padx=10)

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
        # Quitter le mode enregistrement si actif
        self._playback_mode = False
        self._pb_frames_raw.clear()
        self._pb_frames_comp.clear()
        self._pb_ticks.clear()
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
        if self._playback_mode:
            self._show_pb_frame_at_tick(tick)
        else:
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
        if self._comparison_data is None and not self._playback_mode:
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
        if not self._playing or self._paused:
            return
        if self._comparison_data is None and not self._playback_mode:
            return

        current = int(float(self.tick_slider.get()))
        step = int(self._BASE_TICK_STEP * self._speed)
        new_tick = current + step

        max_tick = self._pb_ticks[-1] if self._playback_mode else self._comparison_data.max_tick
        if new_tick >= max_tick:
            new_tick = max_tick
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
        if self._comparison_data is None and not self._playback_mode:
            return
        current = int(float(self.tick_slider.get()))
        min_tick = self._pb_ticks[0] if self._playback_mode else self._comparison_data.min_tick
        new_tick = max(min_tick, current - self._TICK_JUMP)
        self.tick_slider.set(new_tick)
        self._on_slider(new_tick)

    def _on_right(self, event=None):
        if self._comparison_data is None and not self._playback_mode:
            return
        current = int(float(self.tick_slider.get()))
        max_tick = self._pb_ticks[-1] if self._playback_mode else self._comparison_data.max_tick
        new_tick = min(max_tick, current + self._TICK_JUMP)
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

    # ══════════════════════════════════════════════════════════════════════
    # Enregistrement (pré-rendu de toutes les frames)
    # ══════════════════════════════════════════════════════════════════════

    def _on_record(self):
        """Lance l'enregistrement de toutes les frames."""
        if self._comparison_data is None:
            return
        if self._recording:
            return

        from mvc.config import OUTPUT_DIR

        mid = self._comparison_data.match_id
        w = self._comparison_data.w_error
        rec_dir = OUTPUT_DIR / "recordings" / f"{mid}_w{w}"
        rec_dir.mkdir(parents=True, exist_ok=True)

        min_t = self._comparison_data.min_tick
        max_t = self._comparison_data.max_tick
        ticks = list(range(min_t, max_t + 1, self._REC_STEP))
        if not ticks or ticks[-1] < max_t:
            ticks.append(max_t)

        self._recording = True
        self._rec_idx = 0
        self._rec_ticks = ticks
        self._rec_dir = rec_dir

        self._stop_play()
        self.rec_btn.configure(state="disabled", text="Enregistrement…")
        self.rec_bar_frame.pack(fill="x", padx=10, pady=(2, 2), before=self._stat_raw.master)
        self.rec_progress.set(0)
        self.rec_label.configure(text=f"0 / {len(ticks)} frames")

        # Lancer le rendu frame-par-frame via after()
        self.after(1, self._record_next_frame)

    def _record_next_frame(self):
        """Rend et sauvegarde une frame, puis planifie la suivante."""
        if not self._recording:
            return
        total = len(self._rec_ticks)
        if self._rec_idx >= total:
            self._finish_recording()
            return

        tick = self._rec_ticks[self._rec_idx]

        # Mettre à jour les canvas
        self.map_raw.set_tick(tick)
        self.map_compressed.set_tick(tick)
        self.map_raw.canvas.update_idletasks()
        self.map_compressed.canvas.update_idletasks()

        # Capturer les frames
        raw_img = self.map_raw.capture_frame()
        comp_img = self.map_compressed.capture_frame()

        # Sauvegarder
        raw_img.save(str(self._rec_dir / f"raw_{self._rec_idx:05d}.jpg"), "JPEG", quality=85)
        comp_img.save(str(self._rec_dir / f"comp_{self._rec_idx:05d}.jpg"), "JPEG", quality=85)

        self._rec_idx += 1
        self.rec_progress.set(self._rec_idx / total)
        self.rec_label.configure(text=f"{self._rec_idx} / {total} frames")

        # Planifier la frame suivante (after(1) pour garder le GUI réactif)
        self.after(1, self._record_next_frame)

    def _finish_recording(self):
        """Finalise l'enregistrement : sauvegarde les métadonnées."""
        meta = {
            "match_id": self._comparison_data.match_id,
            "w_error": self._comparison_data.w_error,
            "min_tick": self._comparison_data.min_tick,
            "max_tick": self._comparison_data.max_tick,
            "step": self._REC_STEP,
            "num_frames": len(self._rec_ticks),
            "ticks": self._rec_ticks,
        }
        with open(self._rec_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        self._recording = False
        self.rec_btn.configure(state="normal", text="Enregistrer")
        self.rec_btn.configure(text="Enregistré !", fg_color="#27ae60")
        self.rec_label.configure(text=f"Terminé — {len(self._rec_ticks)} frames sauvegardées")
        self.after(2000, self._reset_rec_btn)

    def _reset_rec_btn(self):
        self.rec_btn.configure(text="Enregistrer", fg_color=ACCENT2)
        self.rec_bar_frame.pack_forget()

    # ══════════════════════════════════════════════════════════════════════
    # Chargement d'un enregistrement
    # ══════════════════════════════════════════════════════════════════════

    def _on_load_recording(self):
        """Charge un enregistrement depuis le disque."""
        from mvc.config import OUTPUT_DIR
        from PIL import Image as PILImage

        rec_base = OUTPUT_DIR / "recordings"
        if not rec_base.exists():
            return

        # Chercher automatiquement le dernier enregistrement du match courant
        if self._comparison_data:
            mid = self._comparison_data.match_id
            w = self._comparison_data.w_error
            rec_dir = rec_base / f"{mid}_w{w}"
        else:
            # Prendre le premier dossier disponible
            dirs = sorted(rec_base.iterdir())
            if not dirs:
                return
            rec_dir = dirs[0]

        meta_path = rec_dir / "meta.json"
        if not meta_path.exists():
            return

        with open(meta_path) as f:
            meta = json.load(f)

        num_frames = meta["num_frames"]
        ticks = meta["ticks"]

        # Charger les frames en mémoire
        self._stop_play()
        self.load_rec_btn.configure(state="disabled", text="Chargement…")
        self.rec_bar_frame.pack(fill="x", padx=10, pady=(2, 2), before=self._stat_raw.master)
        self.rec_progress.set(0)
        self.rec_label.configure(text=f"Chargement 0 / {num_frames}")

        self._pb_frames_raw = [None] * num_frames
        self._pb_frames_comp = [None] * num_frames
        self._pb_ticks = ticks
        self._pb_load_idx = 0
        self._pb_load_dir = rec_dir
        self._pb_load_total = num_frames

        # Charger frame par frame via after() pour garder le GUI réactif
        self.after(1, self._load_next_frame)

    def _load_next_frame(self):
        from PIL import Image as PILImage

        idx = self._pb_load_idx
        total = self._pb_load_total
        if idx >= total:
            self._finish_loading()
            return

        raw_path = self._pb_load_dir / f"raw_{idx:05d}.jpg"
        comp_path = self._pb_load_dir / f"comp_{idx:05d}.jpg"

        if raw_path.exists() and comp_path.exists():
            self._pb_frames_raw[idx] = PILImage.open(str(raw_path)).copy()
            self._pb_frames_comp[idx] = PILImage.open(str(comp_path)).copy()

        self._pb_load_idx += 1
        self.rec_progress.set(self._pb_load_idx / total)
        self.rec_label.configure(text=f"Chargement {self._pb_load_idx} / {total}")
        self.after(1, self._load_next_frame)

    def _finish_loading(self):
        """Active le mode lecture d'enregistrement."""
        self._playback_mode = True
        self._pb_idx = 0

        self.load_rec_btn.configure(state="normal", text="Charger Enreg.")
        self.rec_bar_frame.pack_forget()
        self.rec_label.configure(text="")

        # Configurer le slider pour le range de l'enregistrement
        if self._pb_ticks:
            self.tick_slider.configure(from_=self._pb_ticks[0], to=self._pb_ticks[-1])
            self.tick_slider.set(self._pb_ticks[0])
            self.tick_label.configure(text=f"tick: {self._pb_ticks[0]}  [ENREG.]")

        # Afficher la première frame
        self._show_pb_frame(0)

        self._stat_raw.configure(text=f"Mode enregistrement — {len(self._pb_ticks)} frames")
        self._stat_comp.configure(text="Lecture fluide sans lag")

    def _show_pb_frame(self, idx: int):
        """Affiche l'image pré-rendue n°idx."""
        if idx < 0 or idx >= len(self._pb_frames_raw):
            return
        raw_img = self._pb_frames_raw[idx]
        comp_img = self._pb_frames_comp[idx]
        if raw_img and comp_img:
            self.map_raw.display_frame(raw_img)
            self.map_compressed.display_frame(comp_img)
        self._pb_idx = idx

    def _show_pb_frame_at_tick(self, tick: int):
        """Trouve et affiche la frame la plus proche du tick demandé."""
        import bisect
        idx = bisect.bisect_right(self._pb_ticks, tick)
        idx = max(0, min(idx, len(self._pb_ticks) - 1))
        self._show_pb_frame(idx)
        self.tick_label.configure(text=f"tick: {self._pb_ticks[idx]}  [ENREG.]")
