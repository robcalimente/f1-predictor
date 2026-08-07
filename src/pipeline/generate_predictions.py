"""Generate predictions for the next unraced event on the calendar, using the
final models (trained on all available data) and the current as-of-now
feature snapshot (data/processed/feature_state.pkl from build_features.py).

Current grid (driver -> team) is taken from the most recent race in the
dataset -- a reasonable proxy; mid-season driver swaps are rare and would
just look like a one-off cold start for that seat, same as any other
team-switch case.

Output: data/processed/next_race_predictions.parquet
"""
import pickle
from pathlib import Path

import fastf1
import pandas as pd

from build_features import driver_archetype_blend, team_form_blend, circuit_conditions_blend, era_for_season
from circuit_lookup import event_to_circuit_key
from train_model import FEATURE_COLS

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
ARCHETYPE_CSV = REPO_ROOT / "data" / "circuit_archetypes.csv"
CACHE_DIR = REPO_ROOT / ".fastf1_cache"

CURRENT_SEASON = 2026


def find_next_race(features_df: pd.DataFrame) -> dict:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    schedule = fastf1.get_event_schedule(CURRENT_SEASON, include_testing=False)

    raced_rounds = set(features_df.loc[features_df["season"] == CURRENT_SEASON, "round"])
    upcoming = schedule[~schedule["RoundNumber"].isin(raced_rounds)].sort_values("RoundNumber")
    if upcoming.empty:
        raise RuntimeError(f"No unraced rounds left in {CURRENT_SEASON} schedule")

    next_event = upcoming.iloc[0]
    return {
        "season": CURRENT_SEASON,
        "round": int(next_event["RoundNumber"]),
        "event_name": next_event["EventName"],
        "circuit_key": event_to_circuit_key(next_event["EventName"], CURRENT_SEASON),
        "is_sprint_weekend": "Sprint" in str(next_event.get("EventFormat", "")),
    }


def current_grid(features_df: pd.DataFrame) -> pd.DataFrame:
    """Driver -> team as of the most recent race in the dataset."""
    latest_race_order = features_df["race_order"].max()
    latest = features_df[features_df["race_order"] == latest_race_order]
    return latest[["driver", "driver_full_name", "team"]].drop_duplicates("driver")


def build_snapshot_rows(
    grid: pd.DataFrame,
    archetype: str,
    era: str,
    circuit_key: str,
    is_sprint_weekend: bool,
    feature_state: dict,
) -> pd.DataFrame:
    driver_history = feature_state["driver_history"]
    debut_prior = feature_state["debut_prior"]
    team_history = feature_state["team_history"]
    circuit_history = feature_state["circuit_history"]
    circuit_global_prior = feature_state["circuit_global_prior"]

    wet_prob, sc_prob, _ = circuit_conditions_blend(
        circuit_history.get(circuit_key, []), circuit_global_prior
    )

    rows = []
    for _, r in grid.iterrows():
        d_key = (r["driver"], archetype)
        avg_finish, avg_gap, avg_points, d_n = driver_archetype_blend(
            driver_history.get(d_key, []), debut_prior
        )

        t_key = (r["team"], era)
        t_stats = team_form_blend(team_history.get(t_key, []))

        rows.append(
            {
                "driver": r["driver"],
                "driver_full_name": r["driver_full_name"],
                "team": r["team"],
                "archetype": archetype,
                "is_sprint_weekend": int(is_sprint_weekend),
                "driver_archetype_avg_finish": avg_finish,
                "driver_archetype_avg_quali_gap": avg_gap,
                "driver_archetype_avg_points": avg_points,
                "driver_archetype_race_count": d_n,
                "team_form_avg_finish": t_stats["avg_finish"],
                "team_form_avg_points": t_stats["avg_points"],
                "team_form_avg_quali_gap": t_stats["avg_quali_gap"],
                "team_form_trend_slope": t_stats["trend_finish"],
                "team_form_quali_trend_slope": t_stats["trend_quali"],
                "team_form_race_count": t_stats["n"],
                "circuit_wet_probability": wet_prob,
                "circuit_safety_car_probability": sc_prob,
            }
        )
    return pd.DataFrame(rows)


def main():
    features_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    with open(PROCESSED_DIR / "feature_state.pkl", "rb") as f:
        feature_state = pickle.load(f)

    next_race = find_next_race(features_df)
    archetypes = pd.read_csv(ARCHETYPE_CSV)
    archetype_row = archetypes[archetypes["circuit_key"] == next_race["circuit_key"]]
    if archetype_row.empty:
        raise ValueError(f"No archetype mapping for circuit_key {next_race['circuit_key']!r}")
    archetype = archetype_row.iloc[0]["archetype"]
    era = era_for_season(next_race["season"])

    grid = current_grid(features_df)
    snapshot = build_snapshot_rows(
        grid, archetype, era, next_race["circuit_key"], next_race["is_sprint_weekend"], feature_state
    )

    snapshot["archetype"] = snapshot["archetype"].astype("category")

    for target in ["quali_pace", "finish_position", "points"]:
        with open(MODELS_DIR / f"{target}.pkl", "rb") as f:
            model = pickle.load(f)
        predictions = model.predict(snapshot[FEATURE_COLS])
        if target == "points":
            # points can't be negative; an unconstrained regressor can drift
            # slightly below zero for a clearly-backmarker prediction
            predictions = predictions.clip(min=0)
        snapshot[f"predicted_{target}"] = predictions

    # rank-derive a clean predicted finishing order from the raw regressor
    # output so two drivers can't literally tie for the same position
    snapshot["predicted_finish_position_rank"] = (
        snapshot["predicted_finish_position"].rank(method="first").astype(int)
    )
    snapshot["predicted_quali_rank"] = (
        snapshot["predicted_quali_pace"].rank(method="first").astype(int)
    )

    team_points = (
        snapshot.groupby("team")["predicted_points"].sum().rename("predicted_team_points")
    )
    snapshot = snapshot.join(team_points, on="team")

    snapshot["season"] = next_race["season"]
    snapshot["round"] = next_race["round"]
    snapshot["event_name"] = next_race["event_name"]
    snapshot["circuit_key"] = next_race["circuit_key"]

    out_path = PROCESSED_DIR / "next_race_predictions.parquet"
    snapshot.to_parquet(out_path, index=False)
    print(f"Next race: {next_race['event_name']} ({archetype})")
    print(
        snapshot[
            ["driver", "team", "predicted_quali_rank", "predicted_finish_position_rank", "predicted_points"]
        ]
        .sort_values("predicted_finish_position_rank")
        .to_string(index=False)
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
