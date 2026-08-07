"""Extract a real circuit outline (from GPS telemetry of that circuit's most
recent race's fastest lap) for every circuit that's appeared in the pulled
data. One telemetry pull per CIRCUIT, not per race -- ~30 sessions total,
not ~200, so this stays well within FastF1's rate limit in a single pass.

Output: data/track_shapes.json, keyed by circuit_key, each an SVG path
string normalized to a shared viewBox, ready to render + animate a dot
along via SVG <animateMotion>.
"""
import json
import sys
import time
from pathlib import Path

import fastf1
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_lookup import event_to_circuit_key

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / ".fastf1_cache"
OUT_PATH = REPO_ROOT / "data" / "track_shapes.json"

VIEWBOX_W = 400
VIEWBOX_H = 300
PADDING = 20
MAX_POINTS = 180

fastf1.Cache.enable_cache(str(CACHE_DIR))


def candidate_races_per_circuit(max_candidates: int = 3) -> dict[str, list[tuple[int, int, str]]]:
    """Up to max_candidates most recent races per circuit, most recent first
    -- a fallback list in case the very latest race has a telemetry data
    quality issue (seen once: 2026 Monaco)."""
    frames = [pd.read_parquet(f) for f in sorted(RAW_DIR.glob("results_*.parquet"))]
    raw = pd.concat(frames, ignore_index=True)
    raw["circuit_key"] = raw.apply(
        lambda r: event_to_circuit_key(r["event_name"], r["season"]), axis=1
    )
    races = raw[["season", "round", "event_name", "circuit_key"]].drop_duplicates()
    races = races.sort_values(["season", "round"], ascending=False)

    candidates: dict[str, list[tuple[int, int, str]]] = {}
    for row in races.itertuples():
        bucket = candidates.setdefault(row.circuit_key, [])
        if len(bucket) < max_candidates:
            bucket.append((row.season, row.round, row.event_name))
    return candidates


def downsample(points: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [points[int(i * step)] for i in range(max_points)]


def normalize_to_path(xs: list[float], ys: list[float]) -> str:
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1
    span_y = max_y - min_y or 1
    draw_w = VIEWBOX_W - 2 * PADDING
    draw_h = VIEWBOX_H - 2 * PADDING
    scale = min(draw_w / span_x, draw_h / span_y)
    offset_x = PADDING + (draw_w - span_x * scale) / 2
    offset_y = PADDING + (draw_h - span_y * scale) / 2

    norm = [
        (offset_x + (x - min_x) * scale, offset_y + (y - min_y) * scale)
        for x, y in zip(xs, ys)
    ]
    norm = downsample(norm, MAX_POINTS)

    d = f"M {norm[0][0]:.1f},{norm[0][1]:.1f} " + " ".join(
        f"L {x:.1f},{y:.1f}" for x, y in norm[1:]
    )
    return d + " Z"


def extract_shape(season: int, round_num: int) -> str | None:
    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
    except Exception as exc:
        print(f"  failed to get telemetry for {season} R{round_num}: {exc}")
        return None

    if tel is None or tel.empty or "X" not in tel.columns:
        return None
    return normalize_to_path(tel["X"].tolist(), tel["Y"].tolist())


def main():
    existing = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text())

    candidates = candidate_races_per_circuit()
    shapes = dict(existing)

    for circuit_key, races in candidates.items():
        if circuit_key in shapes:
            print(f"{circuit_key}: already have a shape, skipping")
            continue

        for season, round_num, event_name in races:
            print(f"{circuit_key}: extracting from {season} R{round_num} ({event_name})...")
            path_d = extract_shape(season, round_num)
            if path_d is not None:
                shapes[circuit_key] = {
                    "path_d": path_d,
                    "viewBox": f"0 0 {VIEWBOX_W} {VIEWBOX_H}",
                    "source_season": season,
                    "source_round": int(round_num),
                }
                break
            print(f"  {circuit_key}: no telemetry for {season} R{round_num}, trying an earlier race...")
            time.sleep(0.5)
        else:
            print(f"  {circuit_key}: no usable telemetry found in any candidate race, skipping")

        time.sleep(0.5)

    OUT_PATH.write_text(json.dumps(shapes, indent=2))
    print(f"Wrote {len(shapes)} circuit shapes to {OUT_PATH}")


if __name__ == "__main__":
    main()
