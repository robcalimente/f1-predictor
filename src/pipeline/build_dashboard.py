"""Render the static dashboard: docs/index.html (next-race predictions +
history browser) and docs/methodology.html (how-this-works page). No
backend -- everything is baked into the HTML at build time, safe for
GitHub Pages.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"

ARCHETYPE_LABELS = {
    "street": "Street circuit",
    "high_speed_power": "High-speed / power",
    "technical_low_speed": "Technical / low-speed",
    "elevation_change": "Elevation change",
    "medium_hybrid": "Medium / hybrid",
}

PAGE_STYLE = """
<style>
  :root { color-scheme: dark light; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0e1116; color: #e6e6e6; margin: 0; padding: 0 1.5rem 3rem; }
  a { color: #e10600; }
  h1 { font-size: 1.9rem; margin-top: 2rem; }
  h2 { font-size: 1.3rem; margin-top: 2.5rem; border-bottom: 1px solid #2a2f3a; padding-bottom: .4rem; }
  .subtitle { color: #9aa0ab; margin-top: -.5rem; }
  nav { padding: 1rem 0; border-bottom: 1px solid #2a2f3a; }
  nav a { margin-right: 1.5rem; text-decoration: none; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { text-align: left; padding: .4rem .8rem; border-bottom: 1px solid #2a2f3a; }
  th { color: #9aa0ab; font-weight: 600; font-size: .85rem; text-transform: uppercase; }
  .headline-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1.5rem 0; }
  .headline-card { background: #171b22; border: 1px solid #2a2f3a; border-radius: 10px;
                    padding: 1rem 1.4rem; min-width: 180px; }
  .headline-card .value { font-size: 1.8rem; font-weight: 700; }
  .headline-card .label { color: #9aa0ab; font-size: .85rem; }
  select { background: #171b22; color: #e6e6e6; border: 1px solid #2a2f3a; padding: .4rem;
           border-radius: 6px; margin-bottom: 1rem; }
  .note { color: #9aa0ab; font-size: .9rem; }
  footer { margin-top: 3rem; color: #6b7280; font-size: .85rem; }
</style>
"""

NAV_HTML = """
<nav>
  <a href="index.html">Predictions</a>
  <a href="methodology.html">How this works</a>
  <a href="https://github.com/robcalimente/f1-predictor">GitHub</a>
</nav>
"""


def load_data():
    next_race = pd.read_parquet(PROCESSED_DIR / "next_race_predictions.parquet")
    wf = pd.read_parquet(PROCESSED_DIR / "walk_forward_predictions.parquet")
    with open(PROCESSED_DIR / "eval_metrics.json") as f:
        metrics = json.load(f)
    return next_race, wf, metrics


def next_race_table_html(next_race: pd.DataFrame) -> str:
    ordered = next_race.sort_values("predicted_finish_position_rank")
    rows = "".join(
        f"<tr><td>{int(r.predicted_finish_position_rank)}</td>"
        f"<td>{r.driver_full_name}</td><td>{r.team}</td>"
        f"<td>{int(r.predicted_quali_rank)}</td>"
        f"<td>{r.predicted_points:.1f}</td>"
        f"<td>{r.predicted_team_points:.1f}</td></tr>"
        for r in ordered.itertuples()
    )
    return f"""
    <table>
      <tr><th>Pred. finish</th><th>Driver</th><th>Team</th><th>Pred. quali</th>
          <th>Pred. points</th><th>Pred. team points</th></tr>
      {rows}
    </table>
    """


def history_figure(wf: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Qualifying pace (% gap to pole)", "Finishing position", "Points"),
    )
    target_to_col = {"quali_pace": 1, "finish_position": 2, "points": 3}
    for target, col in target_to_col.items():
        sub = wf[wf["target"] == target]
        fig.add_trace(
            go.Scatter(
                x=sub["actual"], y=sub["prediction"], mode="markers",
                marker=dict(size=5, opacity=0.5, color="#e10600"),
                name=target, showlegend=False,
                text=sub["driver"] + " (" + sub["season"].astype(str) + " R" + sub["round"].astype(str) + ")",
                hoverinfo="text+x+y",
            ),
            row=1, col=col,
        )
        lo, hi = sub["actual"].min(), sub["actual"].max()
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       line=dict(color="#6b7280", dash="dash"), showlegend=False),
            row=1, col=col,
        )
    fig.update_layout(
        template="plotly_dark", height=420, margin=dict(t=60, b=40, l=40, r=20),
        paper_bgcolor="#0e1116", plot_bgcolor="#171b22",
    )
    return fig


def headline_cards_html(metrics: dict) -> str:
    labels = {
        "quali_pace": ("Qualifying pace MAE", "% gap to pole"),
        "finish_position": ("Finish position MAE", "positions"),
        "points": ("Points MAE", "points"),
    }
    cards = []
    for target, (label, unit) in labels.items():
        m = metrics.get(target, {})
        val = m.get("headline_mae")
        if val is None:
            continue
        cards.append(
            f'<div class="headline-card"><div class="value">{val:.2f}</div>'
            f'<div class="label">{label} ({unit})</div></div>'
        )
    podium = metrics.get("finish_position", {}).get("podium_accuracy")
    if podium is not None:
        cards.append(
            f'<div class="headline-card"><div class="value">{podium:.0%}</div>'
            f'<div class="label">Exact podium match rate</div></div>'
        )
    return f'<div class="headline-row">{"".join(cards)}</div>'


def build_index_html(next_race: pd.DataFrame, wf: pd.DataFrame, metrics: dict) -> str:
    event_name = next_race["event_name"].iloc[0]
    archetype = ARCHETYPE_LABELS.get(next_race["archetype"].iloc[0], next_race["archetype"].iloc[0])
    history_fig_html = history_figure(wf).to_html(
        full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>F1 Predictor</title>
{PAGE_STYLE}
</head><body>
{NAV_HTML}
<h1>F1 Predictor</h1>
<p class="subtitle">Predicting driver and team performance by track archetype, from 2018-2026 F1 data.</p>

<h2>Next race: {event_name}</h2>
<p class="note">Track archetype: {archetype}. Predictions assume a clean race for every driver (no DNFs modeled).</p>
{next_race_table_html(next_race)}

<h2>Model track record</h2>
<p class="note">Out-of-sample accuracy, walk-forward validated: every prediction below was made without the model
ever seeing that race's actual result during training.</p>
{headline_cards_html(metrics)}
{history_fig_html}

<footer>Built from FastF1 data. See <a href="methodology.html">how this works</a> for the full methodology.</footer>
</body></html>"""


def build_methodology_html(metrics: dict, shap_importance: dict | None) -> str:
    shap_html = ""
    if shap_importance:
        sections = []
        for target, importances in shap_importance.items():
            top = list(importances.items())[:5]
            items = "".join(f"<li>{k}: {v:.2f}</li>" for k, v in top)
            sections.append(f"<h3>{target}</h3><ul>{items}</ul>")
        shap_html = "<h2>What drives each prediction</h2>" + "".join(sections)

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>F1 Predictor — How this works</title>
{PAGE_STYLE}
</head><body>
{NAV_HTML}
<h1>How this works</h1>

<h2>What it predicts</h2>
<p>Three things, per driver per race: qualifying pace (as a percentage gap to pole), finishing position,
and points scored. Constructor/team points are just the sum of both drivers' predicted points, not a
separate model.</p>

<h2>The two-speed feature design</h2>
<p>A driver's skill at a given type of track (street circuit, high-speed circuit, technical/low-speed
circuit, elevation-change circuit) is a <strong>slow-moving signal</strong> — it holds up across
different cars and regulation eras. A team's car competitiveness is a <strong>fast-moving signal</strong> —
it can flip within a single season as teams bring upgrades, or across a regulation change like the ones in
2022 and 2026. Blending both into one lookback window would wash out both signals, so this model computes
them separately:</p>
<ul>
  <li><strong>Driver-archetype skill</strong>: an expanding average of the driver's own results at that
      track archetype across the full 2018-2026 window, shrunk toward a "debut driver" league-average prior
      when the driver has little or no history there (handles rookies and archetype debuts).</li>
  <li><strong>Team form</strong>: a rolling average of the team's last five races, plus a trend/slope term
      that captures a team getting faster or slower within a season (in-season upgrades), reset at each
      regulation-era boundary (2022, 2026).</li>
</ul>

<h2>Model</h2>
<p>Three independent LightGBM gradient-boosted regressors, one per target, trained on the features above
plus the track archetype and whether it's a sprint weekend.</p>

<h2>Validation</h2>
<p>Chronological walk-forward validation: for each season, the model is trained only on seasons strictly
before it, then evaluated on that season's actual results. Predictions are never made using data from the
future relative to the race being predicted — this is the same discipline a real forecasting system needs,
not a random train/test split, which would leak future information into training.</p>

{shap_html}

<h2>Known limitations</h2>
<ul>
  <li><strong>DNFs are not modeled.</strong> Finishing position and points assume a clean race; a crash or
      mechanical failure is treated as an unpredictable event, excluded from both training and evaluation
      rather than silently dragging down accuracy numbers.</li>
  <li><strong>Sprint sessions are not modeled.</strong> Only the main race and main qualifying session are
      predicted for sprint weekends.</li>
  <li><strong>Team identity resets on a name change.</strong> If a team is rebranded (e.g. a title sponsor
      change), it's treated as a fresh entity for the team-form feature rather than carrying history
      forward — a simplification, not a bug.</li>
  <li><strong>Current-grid driver/team pairings</strong> for next-race predictions come from the most
      recent completed race; a same-week driver swap wouldn't be reflected until the next data refresh.</li>
</ul>

<footer><a href="index.html">Back to predictions</a></footer>
</body></html>"""


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    next_race, wf, metrics = load_data()

    shap_importance = None
    shap_path = PROCESSED_DIR / "shap_importance.json"
    if shap_path.exists():
        with open(shap_path) as f:
            shap_importance = json.load(f)

    (DOCS_DIR / "index.html").write_text(build_index_html(next_race, wf, metrics))
    (DOCS_DIR / "methodology.html").write_text(build_methodology_html(metrics, shap_importance))
    print(f"Wrote {DOCS_DIR / 'index.html'} and {DOCS_DIR / 'methodology.html'}")


if __name__ == "__main__":
    main()
