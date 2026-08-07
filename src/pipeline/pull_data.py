"""Pull race + qualifying results for all seasons in SEASON_RANGE via FastF1.

Writes one row per driver per race to data/raw/results_<year>.parquet.
Safe to re-run: skips a PAST season's file if it already exists (those are
final and won't change). The current calendar year is always re-pulled in
full, since new races keep completing throughout its season -- otherwise a
weekly automation run would never pick up that week's new result once the
file exists once. Set FORCE=1 to re-pull every season regardless.
"""
import datetime
import os
import time
from pathlib import Path

import fastf1
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".fastf1_cache"
RAW_DIR = REPO_ROOT / "data" / "raw"
SEASON_RANGE = range(2018, 2027)  # 2018-2026 inclusive
CURRENT_SEASON = datetime.date.today().year
FORCE = os.environ.get("FORCE") == "1"

CACHE_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Track-status codes from FastF1's race control feed. Safety car / VSC
# periods are what makes a race's finishing order diverge from grid order --
# a circuit-level history of how often that happens is a genuine predictive
# signal, not just trivia.
SAFETY_CAR_STATUS_CODES = {"4", "5", "6"}  # Safety Car, Red Flag, VSC deployed

# A single stray "Rainfall: True" sample out of a whole session is sensor
# noise, not a wet race (verified: 2018 French GP shows 1/102 samples True
# despite being a dry race). Requiring rain across a meaningful fraction of
# the session filters that out while still catching real wet races (2023
# Dutch GP: 0.23, 2021 Belgian GP washout: 0.84).
RAIN_FRACTION_THRESHOLD = 0.05

# Weather/safety-car data needs a much heavier per-session API load (laps +
# messages, not just results) than the rest of the pull, and FastF1's public
# API caps at 500 calls/hour. Backfilling all 9 seasons in one pass reliably
# blows through that. Circuit-level rain/SC tendency doesn't need a full
# decade of history to be a meaningful prior, so it's scoped to recent
# seasons only -- 2018-2020 rows simply have these fields as null and fall
# back to the calendar-wide average in feature engineering.
WEATHER_MIN_SEASON = 2021


def race_conditions(session) -> dict:
    """Weather + safety-car summary for one race session. Requires the
    session to have been loaded with laps=True (track_status needs it) and
    weather=True."""
    rained = False
    avg_track_temp = None
    if session.weather_data is not None and not session.weather_data.empty:
        rained = bool(session.weather_data["Rainfall"].mean() > RAIN_FRACTION_THRESHOLD)
        avg_track_temp = float(session.weather_data["TrackTemp"].mean())

    safety_car = False
    try:
        if session.track_status is not None and not session.track_status.empty:
            safety_car = bool(session.track_status["Status"].isin(SAFETY_CAR_STATUS_CODES).any())
    except Exception:
        pass  # track_status not available for some older/incomplete sessions

    return {"rained": rained, "avg_track_temp": avg_track_temp, "safety_car": safety_car}


def pull_season(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows = []
    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]
        is_sprint = "Sprint" in str(event.get("EventFormat", ""))

        pull_weather = year >= WEATHER_MIN_SEASON

        for session_code, session_label in [("Q", "quali"), ("R", "race")]:
            try:
                session = fastf1.get_session(year, round_num, session_code)
                if session_label == "race" and pull_weather:
                    # laps=True is needed for track_status (safety car);
                    # telemetry stays off, that's the expensive part we
                    # don't need for this.
                    session.load(laps=True, telemetry=False, weather=True, messages=True)
                else:
                    session.load(laps=False, telemetry=False, weather=False, messages=False)
            except Exception as exc:
                print(f"  skip {year} R{round_num} {event_name} {session_label}: {exc}")
                continue

            results = session.results
            if results is None or results.empty:
                continue

            conditions = (
                race_conditions(session)
                if session_label == "race" and pull_weather
                else {"rained": None, "avg_track_temp": None, "safety_car": None}
            )

            for _, r in results.iterrows():
                rows.append(
                    {
                        "season": year,
                        "round": round_num,
                        "event_name": event_name,
                        "circuit": event.get("Location", event_name),
                        "country": event.get("Country", ""),
                        "is_sprint_weekend": is_sprint,
                        "session": session_label,
                        "driver": r.get("Abbreviation", ""),
                        "driver_full_name": r.get("FullName", ""),
                        "team": r.get("TeamName", ""),
                        "grid_position": r.get("GridPosition", None),
                        "finish_position": r.get("Position", None),
                        "classified_status": r.get("Status", ""),
                        "points": r.get("Points", None),
                        "q_time_seconds": (
                            r.get("Q3").total_seconds()
                            if session_label == "quali" and pd.notna(r.get("Q3", None))
                            else (
                                r.get("Q2").total_seconds()
                                if session_label == "quali" and pd.notna(r.get("Q2", None))
                                else (
                                    r.get("Q1").total_seconds()
                                    if session_label == "quali" and pd.notna(r.get("Q1", None))
                                    else None
                                )
                            )
                        ),
                        "rained": conditions["rained"],
                        "avg_track_temp": conditions["avg_track_temp"],
                        "safety_car": conditions["safety_car"],
                    }
                )
            time.sleep(0.5)  # be polite to the API

    return pd.DataFrame(rows)


def main():
    for year in SEASON_RANGE:
        out_path = RAW_DIR / f"results_{year}.parquet"
        is_current_season = year == CURRENT_SEASON
        if out_path.exists() and not FORCE and not is_current_season:
            print(f"{year}: already pulled, skipping (set FORCE=1 to refresh)")
            continue

        print(f"{year}: pulling...")
        try:
            df = pull_season(year)
        except Exception as exc:
            print(f"{year}: failed entirely: {exc}")
            continue

        if df.empty:
            print(f"{year}: no data returned (season may not have started yet)")
            continue

        if out_path.exists():
            existing_rows = len(pd.read_parquet(out_path))
            # A rate-limited or otherwise partial re-pull of the current
            # season must never clobber a more-complete previous pull --
            # this happened once already (a CI run got rate-limited and
            # silently overwrote a full season with a handful of rows).
            if len(df) < existing_rows:
                print(
                    f"{year}: new pull has fewer rows ({len(df)}) than the existing "
                    f"file ({existing_rows}) -- keeping the existing file, not overwriting"
                )
                continue

        df.to_parquet(out_path, index=False)
        print(f"{year}: wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
