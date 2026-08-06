"""Compute interpretable evaluation metrics from walk-forward predictions.

All metrics are on rows already restricted to classified (non-DNF) results
for finish_position/points, and to drivers who set a quali time. DNFs are
never scored -- see README / methodology page.

Beyond plain MAE, this adds metrics that speak to whether the model gets
the *shape* of a race right, not just the average error:
  - winner / pole accuracy: did it call the top spot correctly?
  - top-10 accuracy: did it call the points-scoring group correctly?
  - Spearman rank correlation: does it get the relative order right even
    when it misses the exact position?
  - grid-order baseline: is it actually better than "assume no overtakes"?

Output: data/processed/eval_metrics.json, with an all-time headline metric
per target and a per-race-weekend breakdown for the dashboard's history view.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def r_squared(actual: pd.Series, predicted: pd.Series) -> float:
    residual = np.sum((actual - predicted) ** 2)
    total = np.sum((actual - actual.mean()) ** 2)
    return float(1 - residual / total) if total else float("nan")


def top_n_set_accuracy(df: pd.DataFrame, n: int) -> float:
    """Fraction of races where the model's predicted top-n (by prediction)
    exactly matches the actual top-n, as a set (order-agnostic)."""
    hits, total = 0, 0
    for (season, rnd), race in df.groupby(["season", "round"]):
        if len(race) < n:
            continue
        actual_set = set(race.nsmallest(n, "actual")["driver"])
        predicted_set = set(race.nsmallest(n, "prediction")["driver"])
        total += 1
        if actual_set == predicted_set:
            hits += 1
    return hits / total if total else float("nan")


def top_n_overlap(df: pd.DataFrame, n: int) -> float:
    """Average fraction of the actual top-n that the model's predicted
    top-n also contains -- softer than exact-set accuracy, rewards close
    misses (correct group, one name off) instead of an all-or-nothing score."""
    fractions = []
    for (season, rnd), race in df.groupby(["season", "round"]):
        if len(race) < n:
            continue
        actual_set = set(race.nsmallest(n, "actual")["driver"])
        predicted_set = set(race.nsmallest(n, "prediction")["driver"])
        fractions.append(len(actual_set & predicted_set) / n)
    return float(np.mean(fractions)) if fractions else float("nan")


def top_1_accuracy(df: pd.DataFrame) -> float:
    return top_n_set_accuracy(df, 1)


def mean_spearman(df: pd.DataFrame) -> float:
    """Average per-race Spearman rank correlation between predicted and
    actual order -- credits the model for getting relative order right
    (who beat whom) even on races where exact positions are off."""
    correlations = []
    for (season, rnd), race in df.groupby(["season", "round"]):
        if len(race) < 3:
            continue
        corr, _ = spearmanr(race["prediction"], race["actual"])
        if np.isfinite(corr):
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else float("nan")


def grid_order_baseline_mae(wf_finish: pd.DataFrame, features: pd.DataFrame) -> float:
    """MAE of a naive 'nobody overtakes, everyone finishes where they
    qualified' baseline, on the exact same race/driver set the model was
    scored on -- the bar the model actually needs to clear."""
    merged = wf_finish.merge(
        features[["season", "round", "driver", "grid_position"]],
        on=["season", "round", "driver"], how="left",
    )
    merged = merged.dropna(subset=["grid_position"])
    return mae(merged["actual"], merged["grid_position"])


def per_race_breakdown(df: pd.DataFrame) -> list[dict]:
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
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    metrics = {}
    for target in wf["target"].unique():
        sub = wf[wf["target"] == target]
        m = {
            "headline_mae": mae(sub["actual"], sub["prediction"]),
            "r_squared": r_squared(sub["actual"], sub["prediction"]),
            "spearman_rank_correlation": mean_spearman(sub),
            "n_predictions": int(len(sub)),
            "per_race": per_race_breakdown(sub),
        }

        if target == "finish_position":
            m["podium_accuracy"] = top_n_set_accuracy(sub, 3)
            m["winner_accuracy"] = top_1_accuracy(sub)
            m["top10_accuracy"] = top_n_set_accuracy(sub, 10)
            m["top10_overlap"] = top_n_overlap(sub, 10)
            baseline = grid_order_baseline_mae(sub, features)
            m["grid_order_baseline_mae"] = baseline
            m["improvement_over_baseline_pct"] = (
                (baseline - m["headline_mae"]) / baseline if baseline else float("nan")
            )
        elif target == "quali_pace":
            m["pole_accuracy"] = top_1_accuracy(sub)

        metrics[target] = m

    out_path = PROCESSED_DIR / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    for target, m in metrics.items():
        extras = ", ".join(
            f"{k} {v:.1%}" if "accuracy" in k or "overlap" in k or "pct" in k else f"{k} {v:.3f}"
            for k, v in m.items()
            if k not in ("per_race", "n_predictions", "headline_mae")
        )
        print(f"{target}: MAE {m['headline_mae']:.3f} over {m['n_predictions']} predictions -- {extras}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
