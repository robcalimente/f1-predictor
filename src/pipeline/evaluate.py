"""Compute interpretable evaluation metrics from walk-forward predictions.

All metrics are on rows already restricted to classified (non-DNF) results
for finish_position/points, and to drivers who set a quali time. DNFs are
never scored -- see README / methodology page.

Output: data/processed/eval_metrics.json, with an all-time headline metric
per target and a per-race-weekend breakdown for the dashboard's history view.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# F1's current points table, top 10 only. Used to sanity-check the
# points-podium accuracy is comparing like with like.
POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def podium_accuracy(df: pd.DataFrame) -> float:
    """Of all race weekends, what fraction had the model's predicted top-3
    finishers (by predicted finish_position) exactly matching the actual
    podium (as a set, order-agnostic)?"""
    hits, total = 0, 0
    for (season, rnd), race in df.groupby(["season", "round"]):
        actual_podium = set(race.nsmallest(3, "actual")["driver"])
        predicted_podium = set(race.nsmallest(3, "prediction")["driver"])
        if len(actual_podium) < 3:
            continue
        total += 1
        if actual_podium == predicted_podium:
            hits += 1
    return hits / total if total else float("nan")


def per_race_breakdown(df: pd.DataFrame, value_col: str) -> list[dict]:
    rows = []
    for (season, rnd), race in df.groupby(["season", "round"]):
        rows.append(
            {
                "season": int(season),
                "round": int(rnd),
                "mae": mae(race["actual"], race["prediction"]),
                "n_drivers": int(len(race)),
            }
        )
    return sorted(rows, key=lambda r: (r["season"], r["round"]))


def main():
    wf = pd.read_parquet(PROCESSED_DIR / "walk_forward_predictions.parquet")

    metrics = {}
    for target in wf["target"].unique():
        sub = wf[wf["target"] == target]
        metrics[target] = {
            "headline_mae": mae(sub["actual"], sub["prediction"]),
            "n_predictions": int(len(sub)),
            "per_race": per_race_breakdown(sub, "actual"),
        }
        if target == "finish_position":
            metrics[target]["podium_accuracy"] = podium_accuracy(sub)

    out_path = PROCESSED_DIR / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    for target, m in metrics.items():
        extra = f", podium accuracy {m['podium_accuracy']:.1%}" if "podium_accuracy" in m else ""
        print(f"{target}: MAE {m['headline_mae']:.3f} over {m['n_predictions']} predictions{extra}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
