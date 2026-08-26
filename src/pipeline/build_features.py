"""Build the driver-race feature table from raw FastF1 pulls.

Two-speed feature design:
  - driver_archetype_* : slow signal, a driver's performance at this track
    archetype measured RELATIVE TO THEIR TEAMMATE and recency-weighted.
    Absolute results at an archetype mostly measure the car the driver
    happened to be in (Russell averaged near-zero points at high-speed
    circuits in a 2019 Williams; that is a fact about the Williams).
    Differencing against the teammate holds the car constant, and an
    exponential decay by season stops 2018 outvoting this year. Shrunk
    toward a "debut driver" prior when a driver has few/no prior races at
    that archetype (rookie / archetype debut cold start).
  - team_form_*        : fast signal, rolling mean over the team's last up
    to 5 races plus a trend slope -- results AND raw car pace (speed trap,
    best lap), each normalized within its race so circuits compare. Reset
    at each regulation-era boundary
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

# Half-life, in seasons, for weighting a driver's past results at an
# archetype. At 2.5, a result from two and a half seasons ago counts half as
# much as this season's, so the signal tracks the current driver rather than
# their whole career.
DRIVER_RECENCY_HALFLIFE_SEASONS = 2.5

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


def add_race_relative_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw pace within each race so it compares across circuits.

    speed_trap_pct_of_best: this driver's median speed-trap as a percentage
    of the fastest car's that race. best_lap_pct_off_best: how far off the
    session's fastest lap, in percent. Absolute km/h is meaningless across
    Monza and Monaco; position relative to the field that day is not.

    Seasons before WEATHER_MIN_SEASON never load laps, so these are null
    there and LightGBM handles them as missing.
    """
    for col in ("speed_trap_median", "best_lap_seconds"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    by_race = df.groupby("race_order")
    trap_best = by_race["speed_trap_median"].transform("max")
    df["speed_trap_pct_of_best"] = df["speed_trap_median"] / trap_best * 100

    lap_best = by_race["best_lap_seconds"].transform("min")
    df["best_lap_pct_off_best"] = (df["best_lap_seconds"] - lap_best) / lap_best * 100
    return df


def add_teammate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """This driver's result minus their teammate's, same race, same car.

    The car is held constant by construction, so the residual is much closer
    to driver contribution than an absolute result is. Null when a team ran
    only one car that race (nothing to difference against).
    """
    pairs = [
        ("finish_position", "teammate_finish_delta"),
        ("quali_pct_gap_to_pole", "teammate_quali_delta"),
        ("points", "teammate_points_delta"),
    ]
    grouped = df.groupby(["race_order", "team"])
    for col, out_col in pairs:
        own = pd.to_numeric(df[col], errors="coerce")
        team_sum = grouped[col].transform("sum")
        team_n = grouped[col].transform("count")
        others_mean = np.where(team_n > 1, (team_sum - own) / (team_n - 1), np.nan)
        df[out_col] = own - others_mean
    return df


def _weighted_nanmean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w)
    if not ok.any() or w[ok].sum() == 0:
        return np.nan
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


DELTA_FIELDS = (
    ("finish_delta", "teammate_finish_delta"),
    ("quali_delta", "teammate_quali_delta"),
    ("points_delta", "teammate_points_delta"),
)


def driver_archetype_blend(past: list[dict], debut_prior: dict, current_season: int) -> dict:
    """Recency-weighted, teammate-relative skill at one archetype.

    Returns the shrinkage-blended teammate deltas plus the raw race count.
    Shared by build_features (historical rows) and generate_predictions
    (next-race snapshot) so the blend math lives in exactly one place.

    Shrinkage uses the EFFECTIVE sample size (the decayed weights summed),
    not the raw count -- a driver with twenty stale races should be pulled
    toward the debut prior more than one with five recent ones.
    """
    n = len(past)
    if n == 0:
        out = {key: debut_prior[key] for _, key in DELTA_FIELDS}
        return {"finish_delta": out["teammate_finish_delta"],
                "quali_delta": out["teammate_quali_delta"],
                "points_delta": out["teammate_points_delta"],
                "n": 0}

    ages = np.array([current_season - p["season"] for p in past], dtype=float)
    weights = 0.5 ** (np.clip(ages, 0, None) / DRIVER_RECENCY_HALFLIFE_SEASONS)
    effective_n = float(weights.sum())
    w = effective_n / (effective_n + ROOKIE_SHRINKAGE_K)

    blended = {"n": n}
    for out_key, hist_key in DELTA_FIELDS:
        mean = _weighted_nanmean([p.get(hist_key) for p in past], weights)
        prior = debut_prior[hist_key]
        blended[out_key] = prior if not np.isfinite(mean) else w * mean + (1 - w) * prior
    return blended


def _trend_slope(values: np.ndarray) -> float:
    n = len(values)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return 0.0
    x = np.arange(n)[valid]
    return float(np.polyfit(x, values[valid], 1)[0])


def team_form_blend(past: list[dict]) -> dict:
    """Rolling stats for a team's last TEAM_FORM_WINDOW races within one
    era: avg finish/points/quali-gap and race-relative car pace (speed trap,
    best lap), plus a trend slope for finish and for
    quali gap separately (a team's single-lap pace and its race-day
    execution can improve at different rates within a season). Shared by
    build_features (historical rows) and generate_predictions (next-race
    snapshot).
    """
    window = past[-TEAM_FORM_WINDOW:]
    n = len(window)
    if n == 0:
        return {
            "avg_finish": np.nan, "avg_points": np.nan, "avg_quali_gap": np.nan,
            "trend_finish": 0.0, "trend_quali": 0.0,
            "avg_speed_trap_pct": np.nan, "avg_best_lap_pct_off": np.nan,
            "trend_speed_trap": 0.0, "n": 0,
        }
    finishes = np.array([p["finish_position"] for p in window], dtype=float)
    points_ = np.array([p["points"] for p in window], dtype=float)
    quali_gaps = np.array([p.get("quali_gap", np.nan) for p in window], dtype=float)
    speed_trap = np.array([p.get("speed_trap_pct", np.nan) for p in window], dtype=float)
    best_lap = np.array([p.get("best_lap_pct_off", np.nan) for p in window], dtype=float)

    def _mean_or_nan(a):
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan

    return {
        "avg_finish": float(np.nanmean(finishes)),
        "avg_points": float(np.nanmean(points_)),
        "avg_quali_gap": _mean_or_nan(quali_gaps),
        "trend_finish": _trend_slope(finishes),
        "trend_quali": _trend_slope(quali_gaps),
        "avg_speed_trap_pct": _mean_or_nan(speed_trap),
        "avg_best_lap_pct_off": _mean_or_nan(best_lap),
        "trend_speed_trap": _trend_slope(speed_trap),
        "n": n,
    }


def add_driver_archetype_skill(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    df = df.sort_values(["race_order", "driver"]).reset_index(drop=True)

    # debut prior: average finish/pace/points across all drivers' first
    # ROOKIE_SHRINKAGE_K races ever (any archetype), computed once, used as
    # the fallback for a driver with no history at a given archetype.
    df["_driver_race_seq"] = df.groupby("driver").cumcount()
    debut_rows = df[df["_driver_race_seq"] < ROOKIE_SHRINKAGE_K]

    def _debut_mean(col: str) -> float:
        value = debut_rows[col].mean()
        # A rookie is, on average, a little behind an established teammate;
        # if that is somehow unmeasurable, "no different" is the safe prior.
        return float(value) if np.isfinite(value) else 0.0

    debut_prior = {key: _debut_mean(key) for _, key in DELTA_FIELDS}

    history: dict[tuple[str, str], list[dict]] = {}
    out_cols = {
        "driver_archetype_teammate_finish_delta": [],
        "driver_archetype_teammate_quali_delta": [],
        "driver_archetype_teammate_points_delta": [],
        "driver_archetype_race_count": [],
    }

    for _, row in df.iterrows():
        key = (row["driver"], row["archetype"])
        past = history.get(key, [])
        blended = driver_archetype_blend(past, debut_prior, int(row["season"]))

        out_cols["driver_archetype_teammate_finish_delta"].append(blended["finish_delta"])
        out_cols["driver_archetype_teammate_quali_delta"].append(blended["quali_delta"])
        out_cols["driver_archetype_teammate_points_delta"].append(blended["points_delta"])
        out_cols["driver_archetype_race_count"].append(blended["n"])

        entry = {"season": int(row["season"])}
        for _, hist_key in DELTA_FIELDS:
            entry[hist_key] = row[hist_key]
        past.append(entry)
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
            stats = team_form_blend(past)

            for idx in team_rows.index:
                feature_by_row_index[idx] = {
                    "team_form_avg_finish": stats["avg_finish"],
                    "team_form_avg_points": stats["avg_points"],
                    "team_form_avg_quali_gap": stats["avg_quali_gap"],
                    "team_form_trend_slope": stats["trend_finish"],
                    "team_form_quali_trend_slope": stats["trend_quali"],
                    "team_form_avg_speed_trap_pct": stats["avg_speed_trap_pct"],
                    "team_form_avg_best_lap_pct_off": stats["avg_best_lap_pct_off"],
                    "team_form_speed_trap_trend": stats["trend_speed_trap"],
                    "team_form_race_count": stats["n"],
                }

            # one history entry for the whole race: this team's mean result
            past.append(
                {
                    "finish_position": team_rows["finish_position"].mean(),
                    "points": team_rows["points"].mean(),
                    "quali_gap": team_rows["quali_pct_gap_to_pole"].mean(),
                    "speed_trap_pct": team_rows["speed_trap_pct_of_best"].mean(),
                    "best_lap_pct_off": team_rows["best_lap_pct_off_best"].mean(),
                }
            )
            history[key] = past

    feat_df = pd.DataFrame.from_dict(feature_by_row_index, orient="index")
    df = df.join(feat_df)
    return df, history


CIRCUIT_SHRINKAGE_K = 3  # pseudo-count weight given to the global rain/SC rate


def circuit_conditions_blend(past: list[dict], global_prior: dict) -> tuple[float, float, int]:
    """(wet_probability, safety_car_probability, n) for one circuit, shrunk
    toward the global rate for a circuit with little or no history yet
    (new additions to the calendar like Vegas/Miami/Madrid)."""
    n = len(past)
    if n == 0:
        return global_prior["rain_prob"], global_prior["sc_prob"], 0
    rain_mean = np.mean([p["rained"] for p in past])
    sc_mean = np.mean([p["safety_car"] for p in past])
    w = n / (n + CIRCUIT_SHRINKAGE_K)
    wet_prob = w * rain_mean + (1 - w) * global_prior["rain_prob"]
    sc_prob = w * sc_mean + (1 - w) * global_prior["sc_prob"]
    return wet_prob, sc_prob, n


def add_circuit_conditions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Circuit-level historical priors for how often a race there is wet or
    safety-car-affected -- known in advance (it's about the circuit, not
    next week's forecast), unlike actual race-day weather. Expanding,
    chronological, one history entry per race (not per driver row)."""
    df = df.sort_values(["race_order", "circuit_key"]).reset_index(drop=True)

    race_level = df.drop_duplicates("race_order")
    global_prior = {
        "rain_prob": float(race_level["rained"].mean()),
        "sc_prob": float(race_level["safety_car"].mean()),
    }

    history: dict[str, list[dict]] = {}
    feature_by_row_index: dict[int, dict] = {}

    for race_order in sorted(df["race_order"].unique()):
        race_df = df[df["race_order"] == race_order]
        circuit_key = race_df["circuit_key"].iloc[0]
        past = history.get(circuit_key, [])
        wet_prob, sc_prob, n = circuit_conditions_blend(past, global_prior)

        for idx in race_df.index:
            feature_by_row_index[idx] = {
                "circuit_wet_probability": wet_prob,
                "circuit_safety_car_probability": sc_prob,
            }

        past.append({"rained": bool(race_df["rained"].iloc[0]), "safety_car": bool(race_df["safety_car"].iloc[0])})
        history[circuit_key] = past

    feat_df = pd.DataFrame.from_dict(feature_by_row_index, orient="index")
    df = df.join(feat_df)
    return df, history, global_prior


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    race_level = build_race_level(raw)
    race_level = add_race_relative_speed(race_level)
    race_level = add_teammate_deltas(race_level)
    with_driver_skill, driver_history, debut_prior = add_driver_archetype_skill(race_level)
    with_team_form, team_history = add_team_form(with_driver_skill)
    with_conditions, circuit_history, circuit_global_prior = add_circuit_conditions(with_team_form)

    feature_state = {
        "driver_history": driver_history,
        "debut_prior": debut_prior,
        "team_history": team_history,
        "circuit_history": circuit_history,
        "circuit_global_prior": circuit_global_prior,
    }
    with open(PROCESSED_DIR / "feature_state.pkl", "wb") as f:
        pickle.dump(feature_state, f)

    out = with_conditions.sort_values(["race_order", "team", "driver"]).reset_index(drop=True)
    out_path = PROCESSED_DIR / "features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
    print(out[["season", "round", "driver", "team", "archetype",
               "driver_archetype_teammate_finish_delta", "team_form_avg_finish",
               "team_form_avg_speed_trap_pct"]].tail(10).to_string())


if __name__ == "__main__":
    main()
