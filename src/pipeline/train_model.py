"""Train the three LightGBM regressors (quali pace, finish position, points)
with chronological walk-forward validation -- train on seasons strictly
before the test season, never randomly shuffled, to avoid leaking future
race outcomes into training.

Produces:
  - data/processed/walk_forward_predictions.parquet: out-of-sample predictions
    for every season from FIRST_TEST_SEASON onward, used by evaluate.py.
  - models/<target>.pkl: final model trained on ALL available data, used for
    live next-race predictions.
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

FIRST_TEST_SEASON = 2020  # need a few prior seasons of history before validating

FEATURE_COLS = [
    "archetype",
    "is_sprint_weekend",
    "driver_archetype_avg_finish",
    "driver_archetype_avg_quali_gap",
    "driver_archetype_avg_points",
    "driver_archetype_race_count",
    "team_form_avg_finish",
    "team_form_avg_points",
    "team_form_avg_quali_gap",
    "team_form_trend_slope",
    "team_form_quali_trend_slope",
    "team_form_race_count",
    "circuit_wet_probability",
    "circuit_safety_car_probability",
]
CATEGORICAL_COLS = ["archetype"]

TARGETS = {
    "quali_pace": {
        "target_col": "quali_pct_gap_to_pole",
        "classified_only": False,  # everyone who sets a time counts
        "require_target_notna": True,
    },
    "finish_position": {
        "target_col": "finish_position",
        "classified_only": True,  # exclude DNFs -- "if the race goes clean"
        "require_target_notna": True,
    },
    "points": {
        "target_col": "points",
        "classified_only": True,
        "require_target_notna": True,
    },
}

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "learning_rate": 0.05,
    "n_estimators": 300,
}


def prep_frame(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = df.copy()
    if spec["classified_only"]:
        out = out[out["is_classified"]]
    if spec["require_target_notna"]:
        out = out[out[spec["target_col"]].notna()]
    out["archetype"] = out["archetype"].astype("category")
    out["is_sprint_weekend"] = out["is_sprint_weekend"].astype(int)
    return out


def make_dataset(df: pd.DataFrame, target_col: str) -> lgb.Dataset:
    return lgb.Dataset(
        df[FEATURE_COLS],
        label=df[target_col],
        categorical_feature=CATEGORICAL_COLS,
        free_raw_data=False,
    )


def walk_forward(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    frame = prep_frame(df, spec)
    test_seasons = sorted(s for s in frame["season"].unique() if s >= FIRST_TEST_SEASON)
    preds = []

    for test_season in test_seasons:
        train = frame[frame["season"] < test_season]
        test = frame[frame["season"] == test_season]
        if train.empty or test.empty:
            continue

        model = lgb.train(LGB_PARAMS, make_dataset(train, spec["target_col"]))
        test = test.copy()
        test["prediction"] = model.predict(test[FEATURE_COLS])
        preds.append(test)

    if not preds:
        return pd.DataFrame()
    return pd.concat(preds, ignore_index=True)


def train_final(df: pd.DataFrame, spec: dict) -> lgb.Booster:
    frame = prep_frame(df, spec)
    return lgb.train(LGB_PARAMS, make_dataset(frame, spec["target_col"]))


def compute_shap_importance(model: lgb.Booster, frame: pd.DataFrame) -> dict:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(frame[FEATURE_COLS])
    mean_abs = np.abs(shap_values).mean(axis=0)
    return dict(sorted(zip(FEATURE_COLS, mean_abs.tolist()), key=lambda kv: -kv[1]))


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    all_wf = []
    shap_importance = {}
    for name, spec in TARGETS.items():
        print(f"--- {name} ---")
        wf = walk_forward(df, spec)
        if wf.empty:
            print(f"  no walk-forward rows for {name}, skipping")
            continue
        wf = wf[
            ["season", "round", "driver", "team", "archetype", spec["target_col"], "prediction"]
        ].rename(columns={spec["target_col"]: "actual"})
        wf["target"] = name
        all_wf.append(wf)

        final_model = train_final(df, spec)
        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(final_model, f)

        frame = prep_frame(df, spec)
        shap_importance[name] = compute_shap_importance(final_model, frame)
        print(f"  saved models/{name}.pkl, {len(wf)} walk-forward rows")

    if all_wf:
        combined = pd.concat(all_wf, ignore_index=True)
        combined.to_parquet(PROCESSED_DIR / "walk_forward_predictions.parquet", index=False)
        print(f"Wrote {len(combined)} walk-forward prediction rows")

    with open(PROCESSED_DIR / "shap_importance.json", "w") as f:
        json.dump(shap_importance, f, indent=2)
    print("Wrote shap_importance.json")


if __name__ == "__main__":
    main()
