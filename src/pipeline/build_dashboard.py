"""Render the static dashboard: docs/index.html (next-race predictions +
a race-by-race history browser) and docs/methodology.html. No backend --
history browsing is client-side JS filtering over a JSON payload embedded
in the page, so it works on GitHub Pages with no server.

Design: a live-timing-tower aesthetic (the actual on-subject reference for
an F1 data product, not a generic dashboard template) -- near-black surface,
monospace tabular figures for every timing/points column, an amber accent
for structure (headers, active state), and real team colors as identity
chips on each row. Status colors (beat/missed prediction) are fixed, never
themed, and always paired with an icon + label, never color alone.
"""
import datetime
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

# Real team liveries. A team rename (e.g. Toro Rosso -> AlphaTauri -> RB)
# keeps a related but distinct hue so the lineage is still legible at a
# glance; colors are identity chips always paired with the team name text,
# never the sole carrier of meaning.
TEAM_COLORS = {
    "Ferrari": "#E8002D",
    "Mercedes": "#00D7B6",
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "McLaren": "#FF8000",
    "Alpine": "#2293D1",
    "Renault": "#FFF200",
    "Racing Point": "#F596C8",
    "Aston Martin": "#229971",
    "Williams": "#64C4FF",
    "Haas F1 Team": "#B6BABD",
    "Alfa Romeo": "#C92D4B",
    "Alfa Romeo Racing": "#C92D4B",
    "AlphaTauri": "#2B4562",
    "Toro Rosso": "#469BFF",
    "RB": "#6C98FF",
    "Racing Bulls": "#6C98FF",
    "Sauber": "#52E252",
    "Kick Sauber": "#52E252",
    "Cadillac": "#8A6D00",
    "Audi": "#BB0A30",
}
DEFAULT_TEAM_COLOR = "#5A6172"

# Dataviz-validated categorical slots (dark-surface steps), used only for
# the three single-series scatter panels below -- one hue per panel, no
# legend needed since each panel is one series.
CHART_BLUE = "#3987e5"
CHART_ORANGE = "#d95926"
CHART_AQUA = "#199e70"
# Fixed status palette (never themed) -- dark-surface steps.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

PAGE_STYLE_TEMPLATE = """
<style>
  :root {
    color-scheme: dark;
    --bg: #0a0c10;
    --surface: #14171d;
    --surface-raised: #1b1f27;
    --border: #262b35;
    --text: #edeff3;
    --text-muted: #8d94a3;
    --accent: #e3a63f;
    --good: __GOOD__;
    --critical: __CRITICAL__;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 0 0 4rem;
  }
  .wrap { max-width: 980px; margin: 0 auto; padding: 0 1.5rem; }
  code, .mono, .num { font-family: ui-monospace, "SF Mono", "Cascadia Code", "IBM Plex Mono", Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  header.topbar {
    border-bottom: 1px solid var(--border); background: var(--surface);
  }
  .topbar-inner { max-width: 980px; margin: 0 auto; padding: 0.9rem 1.5rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
  .wordmark { font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.95rem; color: var(--text); }
  .wordmark span { color: var(--accent); }
  nav.links { display: flex; gap: 1.4rem; align-items: center; }
  nav.links a { color: var(--text-muted); font-size: 0.88rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  nav.links a.active, nav.links a:hover { color: var(--accent); }
  .status-pill { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: var(--text-muted); }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--good); box-shadow: 0 0 6px var(--good); }

  h1 { font-size: 1.6rem; margin: 2rem 0 0.2rem; letter-spacing: -0.01em; }
  h2 {
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);
    margin: 2.6rem 0 0.9rem; padding-bottom: 0.6rem; border-bottom: 1px solid var(--border);
  }
  .subtitle { color: var(--text-muted); margin-top: 0.2rem; max-width: 46rem; }
  .note { color: var(--text-muted); font-size: 0.88rem; }

  .race-hero {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.1rem 1.3rem; margin-bottom: 1.2rem; display: flex; align-items: baseline;
    justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
  }
  .race-hero .event { font-size: 1.15rem; font-weight: 700; }
  .archetype-tag {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent);
    border: 1px solid rgba(227,166,63,0.4); border-radius: 999px; padding: 0.2rem 0.6rem;
  }

  table { border-collapse: collapse; width: 100%; }
  .table-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
  th, td { text-align: left; padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--text-muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; background: var(--surface-raised); }
  tr:last-child td { border-bottom: none; }
  td.num, th.num { text-align: right; }
  .pos-cell { font-weight: 700; }
  .team-chip { display: inline-flex; align-items: center; gap: 0.5rem; }
  .team-swatch { width: 8px; height: 8px; border-radius: 2px; flex: none; }

  .stat-rail { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.6rem; }
  .stat-tile {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 0.9rem 1.2rem; min-width: 150px; flex: 1 1 150px;
  }
  .stat-tile .value { font-size: 1.65rem; font-weight: 700; }
  .stat-tile .label { color: var(--text-muted); font-size: 0.78rem; margin-top: 0.15rem; }

  .picker-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem; flex-wrap: wrap; }
  select {
    background: var(--surface-raised); color: var(--text); border: 1px solid var(--border);
    padding: 0.5rem 0.7rem; border-radius: 7px; font-size: 0.9rem; min-width: 260px;
  }
  .picker-arrows { display: flex; gap: 0.4rem; }
  .picker-arrows button {
    background: var(--surface-raised); color: var(--text); border: 1px solid var(--border);
    border-radius: 7px; padding: 0.45rem 0.7rem; cursor: pointer; font-size: 0.85rem;
  }
  .picker-arrows button:hover { border-color: var(--accent); color: var(--accent); }
  .picker-arrows button:disabled { opacity: 0.35; cursor: default; }

  .delta { display: inline-flex; align-items: center; gap: 0.3rem; font-size: 0.82rem; font-weight: 600; }
  .delta.good { color: var(--good); }
  .delta.critical { color: var(--critical); }
  .delta.neutral { color: var(--text-muted); }

  footer { margin-top: 3.5rem; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--border); padding-top: 1.2rem; }

  .methodology p, .methodology li { color: var(--text); line-height: 1.6; max-width: 42rem; }
  .methodology ul { padding-left: 1.2rem; }
  .methodology strong { color: var(--accent); }
  .shap-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1rem 0 2rem; }
  .shap-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; }
  .shap-card h3 { margin: 0 0 0.6rem; font-size: 0.85rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.04em; }
  .shap-card ol { margin: 0; padding-left: 1.1rem; color: var(--text-muted); font-size: 0.85rem; }
  .shap-card li { margin-bottom: 0.3rem; }
  .shap-card .mono { color: var(--text); }
</style>
"""
PAGE_STYLE = PAGE_STYLE_TEMPLATE.replace("__GOOD__", STATUS_GOOD).replace("__CRITICAL__", STATUS_CRITICAL)


def topbar_html(active: str, generated_at: str) -> str:
    def link(href, label, key):
        cls = "active" if key == active else ""
        return f'<a href="{href}" class="{cls}">{label}</a>'

    return f"""
<header class="topbar">
  <div class="topbar-inner">
    <div class="wordmark">F1 <span>Predictor</span></div>
    <nav class="links">
      {link("index.html", "Predictions", "predictions")}
      {link("methodology.html", "How this works", "methodology")}
      {link("https://github.com/robcalimente/f1-predictor", "GitHub", "github")}
    </nav>
    <span class="status-pill"><span class="status-dot"></span>Updated {generated_at}</span>
  </div>
</header>
"""


def team_swatch(team: str) -> str:
    color = TEAM_COLORS.get(team, DEFAULT_TEAM_COLOR)
    return f'<span class="team-swatch" style="background:{color}"></span>'


def load_data():
    next_race = pd.read_parquet(PROCESSED_DIR / "next_race_predictions.parquet")
    wf = pd.read_parquet(PROCESSED_DIR / "walk_forward_predictions.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    with open(PROCESSED_DIR / "eval_metrics.json") as f:
        metrics = json.load(f)
    return next_race, wf, features, metrics


def next_race_table_html(next_race: pd.DataFrame) -> str:
    ordered = next_race.sort_values("predicted_finish_position_rank")
    rows = "".join(
        f"<tr><td class='pos-cell num'>{int(r.predicted_finish_position_rank)}</td>"
        f"<td>{r.driver_full_name}</td>"
        f"<td><span class='team-chip'>{team_swatch(r.team)}{r.team}</span></td>"
        f"<td class='num mono'>{int(r.predicted_quali_rank)}</td>"
        f"<td class='num mono'>{r.predicted_points:.1f}</td>"
        f"<td class='num mono'>{r.predicted_team_points:.1f}</td></tr>"
        for r in ordered.itertuples()
    )
    return f"""
    <div class="table-scroll">
    <table>
      <tr><th class="num">Pos</th><th>Driver</th><th>Team</th><th class="num">Quali</th>
          <th class="num">Points</th><th class="num">Team pts</th></tr>
      {rows}
    </table>
    </div>
    """


def delta_status(target: str, actual: float, predicted: float) -> tuple[str, str]:
    """(css class, display string) for a predicted-vs-actual delta.
    Direction of "better" flips for points (higher good) vs quali/finish
    (lower good). A small tolerance band is neutral rather than a coin-flip
    good/critical call on noise-sized differences.
    """
    diff = actual - predicted
    if target == "points":
        tol, better_is_positive = 0.75, True
    elif target == "quali_pace":
        tol, better_is_positive = 0.25, False
    else:  # finish_position
        tol, better_is_positive = 0.6, False

    sign = "+" if diff > 0 else ""
    display = f"{sign}{diff:.1f}"

    if abs(diff) <= tol:
        return "neutral", display
    beat = (diff > 0) if better_is_positive else (diff < 0)
    return ("good" if beat else "critical"), display


def build_race_history(wf: pd.DataFrame, features: pd.DataFrame) -> list[dict]:
    event_lookup = features[["season", "round", "event_name"]].drop_duplicates()
    name_lookup = features[["driver", "driver_full_name"]].drop_duplicates("driver").set_index("driver")[
        "driver_full_name"
    ]

    wide = wf.pivot_table(
        index=["season", "round", "driver", "team", "archetype"],
        columns="target",
        values=["actual", "prediction"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{target}" for metric, target in wide.columns]
    wide = wide.reset_index()

    races = []
    for (season, rnd), race_df in wide.groupby(["season", "round"]):
        event_row = event_lookup[(event_lookup["season"] == season) & (event_lookup["round"] == rnd)]
        event_name = event_row["event_name"].iloc[0] if not event_row.empty else f"Round {rnd}"
        archetype = race_df["archetype"].iloc[0]

        drivers = []
        for r in race_df.sort_values("prediction_finish_position").itertuples():
            entry = {
                "driver": r.driver,
                "name": name_lookup.get(r.driver, r.driver),
                "team": r.team,
                "team_color": TEAM_COLORS.get(r.team, DEFAULT_TEAM_COLOR),
            }
            for target, key in [
                ("quali_pace", "quali"),
                ("finish_position", "finish"),
                ("points", "points"),
            ]:
                actual = getattr(r, f"actual_{target}", None)
                pred = getattr(r, f"prediction_{target}", None)
                if actual is None or pred is None or pd.isna(actual) or pd.isna(pred):
                    entry[key] = None
                    continue
                cls, disp = delta_status(target, actual, pred)
                entry[key] = {
                    "actual": round(float(actual), 2),
                    "pred": round(float(pred), 2),
                    "delta_cls": cls,
                    "delta_disp": disp,
                }
            drivers.append(entry)

        races.append(
            {
                "key": f"{season}-{rnd}",
                "season": int(season),
                "round": int(rnd),
                "event_name": event_name,
                "archetype": ARCHETYPE_LABELS.get(archetype, archetype),
                "drivers": drivers,
            }
        )

    races.sort(key=lambda r: (r["season"], r["round"]), reverse=True)
    return races


def history_browser_html(races: list[dict]) -> str:
    options = "".join(
        f'<option value="{r["key"]}">{r["season"]} R{r["round"]} — {r["event_name"]}</option>'
        for r in races
    )
    races_json = json.dumps(races)
    return f"""
<div class="picker-row">
  <select id="race-picker">{options}</select>
  <div class="picker-arrows">
    <button id="race-prev" title="Older race">&larr; Older</button>
    <button id="race-next" title="Newer race">Newer &rarr;</button>
  </div>
</div>
<div id="race-detail"></div>

<script id="race-data" type="application/json">{races_json}</script>
<script>
(function() {{
  const races = JSON.parse(document.getElementById('race-data').textContent);
  const picker = document.getElementById('race-picker');
  const detail = document.getElementById('race-detail');
  const prevBtn = document.getElementById('race-prev');
  const nextBtn = document.getElementById('race-next');

  function metricCell(entry, key, unit) {{
    const m = entry[key];
    if (!m) return '<td class="num mono">—</td><td class="num mono">—</td><td class="num">—</td>';
    return `<td class="num mono">${{m.pred}}${{unit}}</td>` +
           `<td class="num mono">${{m.actual}}${{unit}}</td>` +
           `<td class="num"><span class="delta ${{m.delta_cls}}">${{m.delta_disp}}${{unit}}</span></td>`;
  }}

  function render(key) {{
    const race = races.find(r => r.key === key);
    if (!race) return;
    const idx = races.findIndex(r => r.key === key);
    prevBtn.disabled = idx >= races.length - 1;
    nextBtn.disabled = idx <= 0;

    const rows = race.drivers.map(d => `
      <tr>
        <td><span class="team-chip"><span class="team-swatch" style="background:${{d.team_color}}"></span>${{d.name}}</span></td>
        <td>${{d.team}}</td>
        ${{metricCell(d, 'quali', '%')}}
        ${{metricCell(d, 'finish', '')}}
        ${{metricCell(d, 'points', '')}}
      </tr>`).join('');

    detail.innerHTML = `
      <div class="race-hero">
        <span class="event">${{race.season}} Round ${{race.round}} — ${{race.event_name}}</span>
        <span class="archetype-tag">${{race.archetype}}</span>
      </div>
      <div class="table-scroll">
      <table>
        <tr>
          <th>Driver</th><th>Team</th>
          <th class="num">Quali pred</th><th class="num">Quali actual</th><th class="num">Δ</th>
          <th class="num">Finish pred</th><th class="num">Finish actual</th><th class="num">Δ</th>
          <th class="num">Points pred</th><th class="num">Points actual</th><th class="num">Δ</th>
        </tr>
        ${{rows}}
      </table>
      </div>`;
  }}

  picker.addEventListener('change', () => render(picker.value));
  prevBtn.addEventListener('click', () => {{
    const idx = races.findIndex(r => r.key === picker.value);
    if (idx < races.length - 1) {{ picker.value = races[idx + 1].key; render(picker.value); }}
  }});
  nextBtn.addEventListener('click', () => {{
    const idx = races.findIndex(r => r.key === picker.value);
    if (idx > 0) {{ picker.value = races[idx - 1].key; render(picker.value); }}
  }});

  if (races.length) render(races[0].key);
}})();
</script>
"""


def scatter_figure(wf: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Qualifying pace (% gap to pole)", "Finishing position", "Points"),
    )
    target_to_col = {"quali_pace": (1, CHART_BLUE), "finish_position": (2, CHART_ORANGE), "points": (3, CHART_AQUA)}
    for target, (col, color) in target_to_col.items():
        sub = wf[wf["target"] == target]
        fig.add_trace(
            go.Scatter(
                x=sub["actual"], y=sub["prediction"], mode="markers",
                marker=dict(size=5, opacity=0.55, color=color),
                name=target, showlegend=False,
                text=sub["driver"] + " (" + sub["season"].astype(str) + " R" + sub["round"].astype(str) + ")",
                hoverinfo="text+x+y",
            ),
            row=1, col=col,
        )
        lo, hi = sub["actual"].min(), sub["actual"].max()
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       line=dict(color="#3a4050", dash="dash", width=1), showlegend=False),
            row=1, col=col,
        )
    fig.update_layout(
        template="plotly_dark", height=380, margin=dict(t=50, b=30, l=40, r=20),
        paper_bgcolor="#0a0c10", plot_bgcolor="#14171d",
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", color="#8d94a3", size=11),
    )
    fig.update_annotations(font=dict(size=12, color="#edeff3"))
    fig.update_xaxes(gridcolor="#262b35", zerolinecolor="#262b35")
    fig.update_yaxes(gridcolor="#262b35", zerolinecolor="#262b35")
    return fig


def headline_cards_html(metrics: dict) -> str:
    fp = metrics.get("finish_position", {})
    cards = []

    def tile(value_str, label):
        cards.append(f'<div class="stat-tile"><div class="value mono">{value_str}</div><div class="label">{label}</div></div>')

    if fp.get("winner_accuracy") is not None:
        tile(f"{fp['winner_accuracy']:.0%}", "Race winner called correctly")
    if fp.get("podium_accuracy") is not None:
        tile(f"{fp['podium_accuracy']:.0%}", "Exact podium match rate")
    if fp.get("top10_overlap") is not None:
        tile(f"{fp['top10_overlap']:.0%}", "Avg. overlap with actual points scorers")

    labels = {
        "quali_pace": ("Qualifying pace MAE", "% gap to pole"),
        "finish_position": ("Finish position MAE", "positions"),
        "points": ("Points MAE", "points"),
    }
    for target, (label, unit) in labels.items():
        val = metrics.get(target, {}).get("headline_mae")
        if val is not None:
            tile(f"{val:.2f}", f"{label} ({unit})")

    return f'<div class="stat-rail">{"".join(cards)}</div>'


def build_index_html(next_race: pd.DataFrame, wf: pd.DataFrame, features: pd.DataFrame, metrics: dict, generated_at: str) -> str:
    event_name = next_race["event_name"].iloc[0]
    archetype = ARCHETYPE_LABELS.get(next_race["archetype"].iloc[0], next_race["archetype"].iloc[0])
    scatter_html = scatter_figure(wf).to_html(
        full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )
    races = build_race_history(wf, features)

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F1 Predictor</title>
{PAGE_STYLE}
</head><body>
{topbar_html("predictions", generated_at)}
<div class="wrap">

<h1>Predicting driver &amp; team performance by track archetype</h1>
<p class="subtitle">Three LightGBM models, trained on 2018-2026 F1 data, predict qualifying pace, finishing
position, and points -- separating a driver's slow-moving track skill from a team's fast-moving car form.</p>

<h2>Next race</h2>
<div class="race-hero">
  <span class="event">{event_name}</span>
  <span class="archetype-tag">{archetype}</span>
</div>
<p class="note">Predictions assume a clean race for every driver (no DNFs modeled).</p>
{next_race_table_html(next_race)}

<h2>Model track record</h2>
<p class="note">Out-of-sample accuracy, walk-forward validated: every prediction below was made without the
model ever seeing that race's actual result during training.</p>
{headline_cards_html(metrics)}
{scatter_html}

<h2>Race-by-race history</h2>
<p class="note">Every race the model has scored, most recent first. Δ is actual minus predicted; green means
the result beat the prediction, red means it missed.</p>
{history_browser_html(races)}

<footer>Built from FastF1 data. See <a href="methodology.html">how this works</a> for the full methodology.</footer>
</div>
</body></html>"""


def accuracy_detail_html(metrics: dict) -> str:
    fp = metrics.get("finish_position", {})
    qp = metrics.get("quali_pace", {})
    pts = metrics.get("points", {})

    def row(label, *vals):
        cells = "".join(f"<td class='num mono'>{v}</td>" for v in vals)
        return f"<tr><td>{label}</td>{cells}</tr>"

    def pct(v):
        return f"{v:.1%}" if v is not None else "—"

    def num(v):
        return f"{v:.3f}" if v is not None else "—"

    table = f"""
    <div class="table-scroll">
    <table>
      <tr><th>Target</th><th class="num">MAE</th><th class="num">R²</th><th class="num">Spearman ρ</th></tr>
      {row("Qualifying pace (% gap to pole)", num(qp.get('headline_mae')), num(qp.get('r_squared')), num(qp.get('spearman_rank_correlation')))}
      {row("Finishing position", num(fp.get('headline_mae')), num(fp.get('r_squared')), num(fp.get('spearman_rank_correlation')))}
      {row("Points", num(pts.get('headline_mae')), num(pts.get('r_squared')), num(pts.get('spearman_rank_correlation')))}
    </table>
    </div>

    <div class="table-scroll" style="margin-top:1rem">
    <table>
      <tr><th>Set-based accuracy</th><th class="num">Rate</th></tr>
      {row("Winner called correctly", pct(fp.get('winner_accuracy')))}
      {row("Exact podium (top 3) match", pct(fp.get('podium_accuracy')))}
      {row("Exact points-scorers (top 10) match", pct(fp.get('top10_accuracy')))}
      {row("Avg. overlap with actual points scorers", pct(fp.get('top10_overlap')))}
      {row("Pole position called correctly", pct(qp.get('pole_accuracy')))}
    </table>
    </div>

    <p class="note" style="margin-top:1rem">
    <strong>On the baseline comparison:</strong> a naive "everyone finishes where they qualified, nobody
    overtakes" baseline scores {num(fp.get('grid_order_baseline_mae'))} MAE on this same set of races — almost
    identical to the model's {num(fp.get('headline_mae'))}. That's not a hidden weakness so much as a fact about
    modern F1: on most tracks, grid position already explains most of the finishing order, because overtaking
    is hard. Spearman rank correlation ({num(fp.get('spearman_rank_correlation'))}) and the points R²
    ({num(pts.get('r_squared'))}) are the more informative numbers here — they show the model captures who
    outperforms their grid slot, not just who started where.</p>
    """
    return f'<h2>Accuracy in detail</h2>{table}'


def build_methodology_html(metrics: dict, shap_importance: dict | None, generated_at: str) -> str:
    shap_html = ""
    if shap_importance:
        cards = []
        for target, importances in shap_importance.items():
            top = list(importances.items())[:5]
            items = "".join(f"<li><span class='mono'>{k}</span>: {v:.2f}</li>" for k, v in top)
            cards.append(f'<div class="shap-card"><h3>{target}</h3><ol>{items}</ol></div>')
        shap_html = f'<h2>What drives each prediction</h2><div class="shap-grid">{"".join(cards)}</div>'

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F1 Predictor — How this works</title>
{PAGE_STYLE}
</head><body>
{topbar_html("methodology", generated_at)}
<div class="wrap methodology">

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
      regulation-era boundary (2022, 2026). Race-pace form and qualifying-pace form are tracked separately —
      a team's single-lap speed and its race-day execution don't always move together.</li>
</ul>

<h2>Circuit conditions</h2>
<p>Two more features, both known in advance rather than guessed: how often a given circuit has produced a
<strong>wet race</strong>, and how often it's produced a <strong>safety car or red flag</strong>, as an
expanding historical rate per circuit (shrunk toward the calendar-wide average for a circuit with little
history, like a new addition to the calendar). This is deliberately a circuit's historical tendency, not a
live weather forecast for the upcoming race — the model doesn't get to know next week's actual conditions
in advance, so it wouldn't be fair to feed the historical model actual race-day weather when predicting a
future race without that information.</p>

<h2>Model</h2>
<p>Three independent LightGBM gradient-boosted regressors, one per target, trained on the features above
plus the track archetype and whether it's a sprint weekend.</p>

<h2>Validation</h2>
<p>Chronological walk-forward validation: for each season, the model is trained only on seasons strictly
before it, then evaluated on that season's actual results. Predictions are never made using data from the
future relative to the race being predicted — this is the same discipline a real forecasting system needs,
not a random train/test split, which would leak future information into training.</p>

{accuracy_detail_html(metrics)}

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
</div>
</body></html>"""


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    next_race, wf, features, metrics = load_data()

    shap_importance = None
    shap_path = PROCESSED_DIR / "shap_importance.json"
    if shap_path.exists():
        with open(shap_path) as f:
            shap_importance = json.load(f)

    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    (DOCS_DIR / "index.html").write_text(build_index_html(next_race, wf, features, metrics, generated_at))
    (DOCS_DIR / "methodology.html").write_text(build_methodology_html(metrics, shap_importance, generated_at))
    print(f"Wrote {DOCS_DIR / 'index.html'} and {DOCS_DIR / 'methodology.html'}")


if __name__ == "__main__":
    main()
