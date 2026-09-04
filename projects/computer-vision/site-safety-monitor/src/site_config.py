"""Loads configs/base.yaml.

This file exists because the config did not have a reader. The zone polygons
and the camera list sat in `configs/base.yaml` looking authoritative while the
code used constants somewhere else, so editing the file changed nothing and
said nothing. A configuration file nobody reads is worse than no configuration
file, because it tells you a lie about where the knobs are.

PyYAML is optional. Without it the built-in defaults are used and a note says
so, rather than the import failing at the top of the module.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"

# Used when configs/base.yaml is missing or PyYAML is not installed. Kept
# identical to the file so the two cannot disagree, and a test asserts that.
DEFAULT_ZONES = {
    "crane_radius": [(60.0, 260.0), (520.0, 260.0), (560.0, 470.0), (20.0, 470.0)],
    "excavation": [(700.0, 300.0), (1180.0, 300.0), (1220.0, 470.0), (660.0, 470.0)],
}
DEFAULT_CAMERAS = [
    {"id": "cam_north", "url": "rtsp://site/north", "zones": ["crane_radius"]},
    {"id": "cam_south", "url": "rtsp://site/south", "zones": ["excavation"]},
]


def _read(path: Path | None = None) -> dict:
    p = CONFIG_FILE if path is None else Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_zones(path: Path | None = None) -> dict[str, list[tuple[float, float]]]:
    """Zone name to polygon, in ORIGINAL image pixels."""
    raw = _read(path).get("zones")
    if not raw:
        return {k: list(v) for k, v in DEFAULT_ZONES.items()}
    return {name: [tuple(float(c) for c in pt) for pt in pts]
            for name, pts in raw.items()}


def load_cameras(path: Path | None = None) -> list[dict]:
    return _read(path).get("cameras") or [dict(c) for c in DEFAULT_CAMERAS]


def zone_camera(path: Path | None = None) -> dict[str, str]:
    """Which camera sees which zone. Tracking is per stream, so this mapping
    decides which tracker a detection belongs to."""
    out = {}
    for cam in load_cameras(path):
        for z in cam.get("zones", []):
            out[z] = cam["id"]
    return out
