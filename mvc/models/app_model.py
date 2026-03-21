"""
Model principal — encapsule toute la logique métier d'analyse de trajectoires.

Expose les opérations (compression, clustering, overlay, etc.) sans aucune dépendance UI.
Le Controller appelle ces méthodes et transmet les résultats à la View.
"""

import json
import logging
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from config import (
    BASE_DIR,
    CANVAS_PATH,
    CLUSTERS_DIR,
    COMPRESSED_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    OVERLAYS_DIR,
    VISUALIZATIONS_DIR,
)
from dota_analytics.clustering import load_data, run_clustering
from dota_analytics.compression import MDLCompressor, process_full_match
from dota_analytics.plotting import (
    PLAYER_COLORS,
    get_available_games,
    get_available_w_errors,
    load_compressed_data,
)
from dota_analytics.structures import JSONExporter, Trajectory, TrajectoryPoint

logger = logging.getLogger(__name__)

# ── Dossiers de données secondaires (compat) ────────────────────────────
EXPORTED_DATA_MVC = BASE_DIR / "exported_data_mvc"
COMPRESSED_SOURCES = (
    [COMPRESSED_DIR, EXPORTED_DATA_MVC] if EXPORTED_DATA_MVC.exists() else [COMPRESSED_DIR]
)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes pour communiquer entre Model → Controller → View
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CompressResult:
    success: bool
    match_id: str
    original_points: int = 0
    total_segments: int = 0
    reduction_pct: float = 0.0
    size_kb: int = 0
    error: str = ""


@dataclass
class OverlayData:
    """Données prêtes à dessiner sur le canvas."""
    canvas_image: Any  # PIL Image
    player_segments: Dict[int, list]
    min_tick: int = 0
    max_tick: int = 0
    match_id: str = ""
    w_error: float = 0.0


@dataclass
class ClusterVisuData:
    """Données pour visualiser un cluster sur la carte."""
    canvas_image: Any
    segments: list  # Liste de (x1, y1, x2, y2, color)
    cluster_id: int = 0
    total_in_cluster: int = 0


@dataclass
class ComparisonData:
    """Données pour la comparaison côte à côte brut vs compressé."""
    canvas_image: Any  # PIL Image
    raw_points: Dict[int, list]       # {player_id: [{x, y, tick}, ...]}
    compressed_segments: Dict[int, list]  # {player_id: [{start, end}, ...]}
    min_tick: int = 0
    max_tick: int = 0
    match_id: str = ""
    w_error: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════


class AppModel:
    """Modèle central de l'application.

    Toutes les opérations métier sont ici.  Aucun import tkinter / customtkinter.
    """

    # ── helpers chemins ──────────────────────────────────────────────────

    @staticmethod
    def _find_compressed_file(w_error: float, match_id: str) -> Optional[Path]:
        w_error_str = str(int(w_error)) if w_error == int(w_error) else str(w_error)
        for source_dir in COMPRESSED_SOURCES:
            for name in [f"w_error_{w_error_str}", f"w_error_{float(w_error)}"]:
                p = source_dir / name / f"{match_id}_compressed.json"
                if p.exists():
                    return p
        return None

    @staticmethod
    def _resolve_w_error_folder(w_error: float) -> Optional[Path]:
        for name in [f"w_error_{int(w_error)}", f"w_error_{float(w_error)}"]:
            p = COMPRESSED_DIR / name
            if p.exists():
                return p
        return None

    # ── lister les données disponibles ───────────────────────────────────

    def list_available_w_errors(self) -> List[float]:
        all_w = []
        for src in COMPRESSED_SOURCES:
            if src.exists():
                all_w.extend(get_available_w_errors(src))
        return sorted(set(all_w))

    def list_available_matches(self, w_error: float) -> List[str]:
        all_games = []
        for src in COMPRESSED_SOURCES:
            if not src.exists():
                continue
            # Essayer les deux formats de nom de dossier (w_error_12 et w_error_12.0)
            for name in [f"w_error_{int(w_error)}", f"w_error_{float(w_error)}"]:
                w_dir = src / name
                if w_dir.exists():
                    for f in sorted(w_dir.glob("*_compressed.json")):
                        all_games.append(f.stem.replace("_compressed", ""))
        return sorted(set(all_games))

    def list_csv_matches(self) -> List[str]:
        """Liste les match IDs bruts dans data-dota/."""
        # Chercher dans data-dota/ et data-dota/data-dota/
        csv_dir = DATA_DIR
        if not list(csv_dir.glob("coord_*.csv")):
            csv_dir = DATA_DIR / "data-dota"
        ids = []
        for f in sorted(csv_dir.glob("coord_*.csv")):
            ids.append(f.stem.replace("coord_", ""))
        return ids

    def _csv_dir(self) -> Path:
        """Retourne le dossier contenant les CSV (compat data-dota/data-dota)."""
        if list(DATA_DIR.glob("coord_*.csv")):
            return DATA_DIR
        sub = DATA_DIR / "data-dota"
        if sub.exists() and list(sub.glob("coord_*.csv")):
            return sub
        return DATA_DIR

    # ── compression ──────────────────────────────────────────────────────

    @staticmethod
    def _compress_one(csv_path: Path, w_error: float, output_base: Path) -> CompressResult:
        match_id = csv_path.stem.replace("coord_", "")
        try:
            df = pd.read_csv(csv_path)
            results = process_full_match(df, match_id, w_error=w_error)
            total_orig = sum(len(df[f"x{i}"]) for i in range(10) if f"x{i}" in df.columns)
            total_segments = sum(len(segs) for segs in results.values())

            out_dir = output_base / f"w_error_{w_error}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{match_id}_compressed.json"

            exporter = JSONExporter()
            out_path = exporter.export_match(results, match_id, out_path, w_error)

            size_kb = out_path.stat().st_size // 1024
            reduction = (1 - total_segments / total_orig) * 100 if total_orig > 0 else 0
            return CompressResult(True, match_id, total_orig, total_segments, reduction, size_kb)
        except Exception as e:
            return CompressResult(False, match_id, error=str(e))

    def compress(
        self, w_error: float, match_id: Optional[str] = None, callback=None
    ) -> List[CompressResult]:
        csv_dir = self._csv_dir()
        if match_id:
            csv_files = [csv_dir / f"coord_{match_id}.csv"]
        else:
            csv_files = sorted(csv_dir.glob("coord_*.csv"))

        results: List[CompressResult] = []
        total = len(csv_files)
        for i, csv_path in enumerate(csv_files):
            r = self._compress_one(csv_path, w_error, COMPRESSED_DIR)
            results.append(r)
            if callback:
                callback(i + 1, total, r)
        return results

    # ── overlay (données pour dessin) ────────────────────────────────────

    def load_overlay_data(self, w_error: float, match_id: str) -> Optional[OverlayData]:
        json_path = self._find_compressed_file(w_error, match_id)
        if json_path is None or not CANVAS_PATH.exists():
            return None

        data_dir = json_path.parent.parent
        data = load_compressed_data(data_dir, w_error, match_id)

        canvas_img = Image.open(str(CANVAS_PATH))
        # Recadrer en carré
        w, h = canvas_img.size
        if w > h:
            left = (w - h) // 2
            canvas_img = canvas_img.crop((left, 0, left + h, h))

        player_segments: Dict[int, list] = {}
        min_tick = float("inf")
        max_tick = 0

        for player in data["players"]:
            pid = player["player_id"]
            segs = []
            for seg in player["segments"]:
                segs.append({"start": seg["start"], "end": seg["end"]})
                min_tick = min(min_tick, seg["start"]["tick"], seg["end"]["tick"])
                max_tick = max(max_tick, seg["start"]["tick"], seg["end"]["tick"])
            player_segments[pid] = segs

        return OverlayData(
            canvas_image=canvas_img,
            player_segments=player_segments,
            min_tick=int(min_tick),
            max_tick=int(max_tick),
            match_id=match_id,
            w_error=w_error,
        )

    # ── comparison brut vs compressé ─────────────────────────────────────

    def load_comparison_data(self, w_error: float, match_id: str) -> Optional[ComparisonData]:
        """Charge les données brutes (CSV) et compressées (JSON) pour comparaison."""
        json_path = self._find_compressed_file(w_error, match_id)
        if json_path is None or not CANVAS_PATH.exists():
            return None

        csv_dir = self._csv_dir()
        csv_path = csv_dir / f"coord_{match_id}.csv"
        if not csv_path.exists():
            return None

        # --- Canvas ---
        canvas_img = Image.open(str(CANVAS_PATH))
        w, h = canvas_img.size
        if w > h:
            left = (w - h) // 2
            canvas_img = canvas_img.crop((left, 0, left + h, h))

        # --- Données brutes (CSV) ---
        df = pd.read_csv(csv_path)
        raw_points: Dict[int, list] = {}
        for pid in range(10):
            xcol, ycol = f"x{pid}", f"y{pid}"
            if xcol not in df.columns:
                continue
            pts = []
            for _, row in df.iterrows():
                x, y = float(row[xcol]), float(row[ycol])
                if x != 0.0 or y != 0.0:
                    pts.append({"x": x, "y": y, "tick": int(row["tick"])})
            if pts:
                raw_points[pid] = pts

        # --- Données compressées (JSON) ---
        data_dir = json_path.parent.parent
        data = load_compressed_data(data_dir, w_error, match_id)

        compressed_segments: Dict[int, list] = {}
        min_tick = float("inf")
        max_tick = 0

        for player in data["players"]:
            pid = player["player_id"]
            segs = []
            for seg in player["segments"]:
                segs.append({"start": seg["start"], "end": seg["end"]})
                min_tick = min(min_tick, seg["start"]["tick"], seg["end"]["tick"])
                max_tick = max(max_tick, seg["start"]["tick"], seg["end"]["tick"])
            compressed_segments[pid] = segs

        # Étendre min/max avec les ticks CSV
        for pts in raw_points.values():
            if pts:
                min_tick = min(min_tick, pts[0]["tick"])
                max_tick = max(max_tick, pts[-1]["tick"])

        return ComparisonData(
            canvas_image=canvas_img,
            raw_points=raw_points,
            compressed_segments=compressed_segments,
            min_tick=int(min_tick),
            max_tick=int(max_tick),
            match_id=match_id,
            w_error=w_error,
        )

    # ── cluster visu data ────────────────────────────────────────────────

    def load_cluster_visu_data(self, w_error: float, cluster_id: int) -> Optional[ClusterVisuData]:
        folder = self._resolve_w_error_folder(w_error)
        if folder is None:
            return None

        clusters_file = self._find_clusters_file(w_error)
        if clusters_file is None:
            return None

        with open(clusters_file) as f:
            match_clusters = json.load(f)

        # Collecter les segments du cluster demandé
        target_segs = []
        for mid, segs_dict in match_clusters.items():
            for seg_id, label in segs_dict.items():
                if int(label) == cluster_id:
                    target_segs.append((mid, seg_id))

        # Charger les coordonnées correspondantes
        drawn = []
        for mid, seg_id in target_segs:
            # seg_id = "P{pid}_{idx}"
            json_path = folder / f"{mid}_compressed.json"
            if not json_path.exists():
                continue
            with open(json_path) as f:
                data = json.load(f)
            parts = seg_id.split("_")
            pid = int(parts[0][1:])
            idx = int(parts[1])
            for player in data["players"]:
                if player["player_id"] == pid and idx < len(player["segments"]):
                    s = player["segments"][idx]
                    color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
                    drawn.append((s["start"]["x"], s["start"]["y"],
                                  s["end"]["x"], s["end"]["y"], color))
                    break

        canvas_img = Image.open(str(CANVAS_PATH))
        w, h = canvas_img.size
        if w > h:
            left = (w - h) // 2
            canvas_img = canvas_img.crop((left, 0, left + h, h))

        return ClusterVisuData(
            canvas_image=canvas_img,
            segments=drawn,
            cluster_id=cluster_id,
            total_in_cluster=len(drawn),
        )

    # ── clustering (délègue à dota_analytics) ────────────────────────────

    def run_clustering(self, w_error: float, **kwargs):
        folder = self._resolve_w_error_folder(w_error)
        if folder is None:
            raise FileNotFoundError(f"Dossier w_error={w_error} introuvable")
        run_clustering(folder, **kwargs)

    # ── lister clusters disponibles ──────────────────────────────────────

    def _find_clusters_file(self, w_error: float) -> Optional[Path]:
        """Trouve le fichier de résultats de clusters pour un w_error donné."""
        for name in [f"w_error_{int(w_error)}", f"w_error_{float(w_error)}"]:
            p = CLUSTERS_DIR / f"clusters_result_{name}.json"
            if p.exists():
                return p
        return None

    def list_available_clusters(self, w_error: float) -> List[int]:
        clusters_file = self._find_clusters_file(w_error)
        if clusters_file is None:
            return []
        with open(clusters_file) as f:
            match_clusters = json.load(f)
        labels = set()
        for segs_dict in match_clusters.values():
            for label in segs_dict.values():
                labels.add(int(label))
        return sorted(labels)
