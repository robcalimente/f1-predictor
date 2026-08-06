"""Pull race + qualifying results for all seasons in SEASON_RANGE via FastF1.

Writes one row per driver per race to data/raw/results_<year>.parquet.
Safe to re-run: skips a season file that already exists unless FORCE=1 env var is set.
"""
import os
import time
from pathlib import Path

import fastf1
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".fastf1_cache"
RAW_DIR = REPO_ROOT / "data" / "raw"
SEASON_RANGE = range(2018, 2027)  # 2018-2026 inclusive
FORCE = os.environ.get("FORCE") == "1"

CACHE_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def pull_season(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows = []
    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]
        is_sprint = "Sprint" in str(event.get("EventFormat", ""))

        for session_code, session_label in [("Q", "quali"), ("R", "race")]:
            try:
                session = fastf1.get_session(year, round_num, session_code)
                session.load(laps=False, telemetry=False, weather=False, messages=False)
            except Exception as exc:
                print(f"  skip {year} R{round_num} {event_name} {session_label}: {exc}")
                continue

            results = session.results
            if results is None or results.empty:
                continue

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
                    }
                )
            time.sleep(0.5)  # be polite to the API

    return pd.DataFrame(rows)


def main():
    for year in SEASON_RANGE:
        out_path = RAW_DIR / f"results_{year}.parquet"
        if out_path.exists() and not FORCE:
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

        df.to_parquet(out_path, index=False)
        print(f"{year}: wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
