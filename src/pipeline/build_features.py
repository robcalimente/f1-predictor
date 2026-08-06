"""Build the driver-race feature table from raw FastF1 pulls.

Two-speed feature design:
  - driver_archetype_* : slow signal, expanding mean over ALL prior seasons
    (2018-2026) of a driver's performance at this track archetype. Shrunk
    toward a "debut driver" prior when a driver has few/no prior races at
    that archetype (rookie / archetype debut cold start).
  - team_form_*        : fast signal, rolling mean over the team's last up
    to 5 races plus a trend slope, reset at each regulation-era boundary
    (2018-2021, 2022-2025, 2026+) and by team name (a team rename, e.g.
    Racing Point -> Aston Martin, is treated as a fresh entity -- a known
    limitation, noted on the methodology page).

Output: data/processed/features.parquet, one row per driver per race.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from circuit_lookup import event_to_circuit_key

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARCHETYPE_CSV = REPO_ROOT / "data" / "circuit_archetypes.csv"

ROOKIE_SHRINKAGE_K = 3  # pseudo-count weight given to the debut prior
TEAM_FORM_WINDOW = 5

# Status strings FastF1 reports for a car that finished / was classified,
# including being lapped ("+1 Lap", "+2 Laps", ... or the literal "Lapped").
# Anything else (Retired, Accident, DNS, DSQ, mechanical failures, etc.) is
# treated as a DNF and excluded from position/points training targets.
CLASSIFIED_EXACT = ("Finished", "Lapped")


def is_classified(status: str) -> bool:
    if not isinstance(status, str):
        return False
    return status in CLASSIFIED_EXACT or status.startswith("+")


def era_for_season(season: int) -> str:
    if season <= 2021:
        return "2018-2021"
    if season <= 2025:
        return "2022-2025"
    return "2026+"


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("results_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw parquet files found in {RAW_DIR}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build_race_level(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape session-level rows into one row per driver per race."""
    race = raw[raw["session"] == "race"].copy()
    quali = raw[raw["session"] == "quali"].copy()

    # pole time + this driver's gap-to-pole as a percentage
    pole_time = quali.groupby(["season", "round"])["q_time_seconds"].min()
    quali = quali.join(pole_time.rename("pole_time_seconds"), on=["season", "round"])
    quali["quali_pct_gap_to_pole"] = (
        (quali["q_time_seconds"] - quali["pole_time_seconds"]) / quali["pole_time_seconds"] * 100
    )
    # A handful of "Q1" times are from crash/red-flag-affected or incomplete
    # laps rather than a genuine timed lap, giving physically implausible
    # gaps (seen: up to ~97%). No real modern F1 quali gap exceeds ~20%;
    # treat anything past that as missing signal, not a real pace reading.
    MAX_PLAUSIBLE_GAP_PCT = 20
    quali.loc[quali["quali_pct_gap_to_pole"] > MAX_PLAUSIBLE_GAP_PCT, "quali_pct_gap_to_pole"] = (
        np.nan
    )

    quali_slim = quali[["season", "round", "driver", "quali_pct_gap_to_pole"]]

    df = race.merge(quali_slim, on=["season", "round", "driver"], how="left")

    df["circuit_key"] = df.apply(
        lambda r: event_to_circuit_key(r["event_name"], r["season"]), axis=1
    )
    archetypes = pd.read_csv(ARCHETYPE_CSV)
    df = df.merge(archetypes[["circuit_key", "archetype"]], on="circuit_key", how="left")
    if df["archetype"].isna().any():
        missing = df.loc[df["archetype"].isna(), "circuit_key"].unique()
        raise ValueError(f"circuit_key(s) with no archetype mapping: {missing}")

    df["is_classified"] = df["classified_status"].apply(is_classified)
    df["era"] = df["season"].apply(era_for_season)

    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["race_order"] = df["season"] * 100 + df["round"]
    return df


def driver_archetype_blend(past: list[dict], debut_prior: dict) -> tuple[float, float, float, int]:
    """Shrinkage-blended (avg_finish, avg_gap, avg_points, n) for a driver's
    history at one archetype. Shared by build_features (historical rows) and
    generate_predictions (next-race snapshot) so the blend math lives in
    exactly one place.
    """
    n = len(past)
    if n == 0:
        return (
            debut_prior["finish_position"],
            debut_prior["quali_pct_gap_to_pole"],
            debut_prior["points"],
            0,
        )
    driver_mean_finish = np.nanmean([p["finish_position"] for p in past])
    driver_mean_gap = np.nanmean([p["quali_pct_gap_to_pole"] for p in past])
    driver_mean_points = np.nanmean([p["points"] for p in past])
    w = n / (n + ROOKIE_SHRINKAGE_K)
    avg_finish = w * driver_mean_finish + (1 - w) * debut_prior["finish_position"]
    avg_gap = w * driver_mean_gap + (1 - w) * debut_prior["quali_pct_gap_to_pole"]
    avg_points = w * driver_mean_points + (1 - w) * debut_prior["points"]
    return avg_finish, avg_gap, avg_points, n


def team_form_blend(past: list[dict]) -> tuple[float, float, float, int]:
    """(avg_finish, avg_points, trend_slope, n) for a team's last
    TEAM_FORM_WINDOW races within one era. Shared the same way as
    driver_archetype_blend above.
    """
    window = past[-TEAM_FORM_WINDOW:]
    n = len(window)
    if n == 0:
        return np.nan, np.nan, 0.0, 0
    finishes = np.array([p["finish_position"] for p in window], dtype=float)
    points_ = np.array([p["points"] for p in window], dtype=float)
    avg_finish = np.nanmean(finishes)
    avg_points = np.nanmean(points_)
    valid = np.isfinite(finishes)
    if valid.sum() >= 2:
        x = np.arange(n)[valid]
        trend = np.polyfit(x, finishes[valid], 1)[0]
    else:
        trend = 0.0
    return avg_finish, avg_points, trend, n


def add_driver_archetype_skill(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    df = df.sort_values(["race_order", "driver"]).reset_index(drop=True)

    # debut prior: average finish/pace/points across all drivers' first
    # ROOKIE_SHRINKAGE_K races ever (any archetype), computed once, used as
    # the fallback for a driver with no history at a given archetype.
    df["_driver_race_seq"] = df.groupby("driver").cumcount()
    debut_rows = df[df["_driver_race_seq"] < ROOKIE_SHRINKAGE_K]
    debut_prior = {
        "finish_position": debut_rows["finish_position"].mean(),
        "quali_pct_gap_to_pole": debut_rows["quali_pct_gap_to_pole"].mean(),
        "points": debut_rows["points"].mean(),
    }

    history: dict[tuple[str, str], list[dict]] = {}
    out_cols = {
        "driver_archetype_avg_finish": [],
        "driver_archetype_avg_quali_gap": [],
        "driver_archetype_avg_points": [],
        "driver_archetype_race_count": [],
    }

    for _, row in df.iterrows():
        key = (row["driver"], row["archetype"])
        past = history.get(key, [])
        avg_finish, avg_gap, avg_points, n = driver_archetype_blend(past, debut_prior)

        out_cols["driver_archetype_avg_finish"].append(avg_finish)
        out_cols["driver_archetype_avg_quali_gap"].append(avg_gap)
        out_cols["driver_archetype_avg_points"].append(avg_points)
        out_cols["driver_archetype_race_count"].append(n)

        past.append(
            {
                "finish_position": row["finish_position"],
                "quali_pct_gap_to_pole": row["quali_pct_gap_to_pole"],
                "points": row["points"],
            }
        )
        history[key] = past

    for col, values in out_cols.items():
        df[col] = values

    df = df.drop(columns=["_driver_race_seq"])
    return df, history, debut_prior


def add_team_form(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Team form is computed per RACE (one history entry = one race, the mean
    of that team's classified drivers' results that race), not per driver row.
    Feature values for a given race must only use STRICTLY PRIOR races -- the
    two teammates in the same race must never see each other's same-race
    result, so features are snapshotted for a whole race's rows before the
    history is updated with that race's outcome.
    """
    df = df.sort_values(["race_order", "team"]).reset_index(drop=True)

    history: dict[tuple[str, str], list[dict]] = {}
    feature_by_row_index: dict[int, dict] = {}

    for race_order in sorted(df["race_order"].unique()):
        race_df = df[df["race_order"] == race_order]
        for team, team_rows in race_df.groupby("team"):
            era = team_rows["era"].iloc[0]
            key = (team, era)
            past = history.get(key, [])
            avg_finish, avg_points, trend, n = team_form_blend(past)

            for idx in team_rows.index:
                feature_by_row_index[idx] = {
                    "team_form_avg_finish": avg_finish,
                    "team_form_avg_points": avg_points,
                    "team_form_trend_slope": trend,
                    "team_form_race_count": n,
                }

            # one history entry for the whole race: this team's mean result
            past.append(
                {
                    "finish_position": team_rows["finish_position"].mean(),
                    "points": team_rows["points"].mean(),
                }
            )
            history[key] = past

    feat_df = pd.DataFrame.from_dict(feature_by_row_index, orient="index")
    df = df.join(feat_df)
    return df, history


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    race_level = build_race_level(raw)
    with_driver_skill, driver_history, debut_prior = add_driver_archetype_skill(race_level)
    with_team_form, team_history = add_team_form(with_driver_skill)

    feature_state = {
        "driver_history": driver_history,
        "debut_prior": debut_prior,
        "team_history": team_history,
    }
    with open(PROCESSED_DIR / "feature_state.pkl", "wb") as f:
        pickle.dump(feature_state, f)

    out = with_team_form.sort_values(["race_order", "team", "driver"]).reset_index(drop=True)
    out_path = PROCESSED_DIR / "features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
    print(out[["season", "round", "driver", "team", "archetype",
               "driver_archetype_avg_finish", "team_form_avg_finish",
               "team_form_trend_slope"]].tail(10).to_string())


if __name__ == "__main__":
    main()
