# f1-predictor

Predicts Formula 1 driver and team performance by track archetype: qualifying pace (% gap to pole),
finishing position, and points, split out separately for drivers and constructors.

Live dashboard: (added once deployed)

## What this is

A gradient-boosted model trained on 2018-2026 F1 data (via [FastF1](https://github.com/theOehrly/Fast-F1))
that separates two signals that move at very different speeds:

- **Driver skill by track archetype** — a slow-moving signal computed across the full 2018-2026 window,
  since a driver's ability at street circuits vs. high-speed circuits vs. technical tracks persists
  across different cars and regulation eras.
- **Team/car form** — a fast-moving signal computed from each team's last 3-5 races plus an in-season
  trend term, reset at known regulation-change boundaries (2022, 2026). This is what captures a team
  going from back-of-the-grid to front-running (or vice versa) within a season as they bring upgrades.

Full methodology write-up, including model choice, validation approach, and known limitations, lives on
the "How this works" page of the dashboard.

## Structure

- `src/pipeline/` — data pull (FastF1), feature engineering, model training, evaluation, dashboard export
- `data/` — raw pulled results (not committed) and processed feature tables
- `data/circuit_archetypes.csv` — manual circuit-to-archetype taxonomy
- `docs/` — static dashboard output, served via GitHub Pages
- `.github/workflows/` — scheduled Thursday (pre-race predictions) and Monday (post-race scoring) pipeline runs

## Running locally

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/pipeline/pull_data.py       # pulls 2018-2026 race/quali results via FastF1
python src/pipeline/build_features.py  # builds driver-skill and team-form features
python src/pipeline/train_model.py     # trains the three LightGBM models, walk-forward validated
python src/pipeline/evaluate.py        # computes MAE/podium-accuracy metrics
python src/pipeline/build_dashboard.py # renders docs/ static site
```

## Status

Under active development — see the parent portfolio site's `PROGRESS.md` for build order and context.
