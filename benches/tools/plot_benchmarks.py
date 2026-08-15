#!/usr/bin/env python3
"""Aggregate repeated benchmark runs and render the performance figures.

Reads the per-run CSVs written by ``benches/tools/run_repeated.sh`` from
``output/runs/``, computes mean and standard deviation across runs, writes the
aggregated tables to ``output/``, and renders two 2x1 plotly figures into
``doc/plots/`` as both interactive HTML and static PNG.

Metrics follow the pose graph optimization literature (SE-Sync, Rosen et al.
IJRR 2019; Carlone et al. ICRA 2015): solution *cost* and *runtime*. For bundle
adjustment the analogue is reprojection RMSE and runtime.

Usage:
    uv run --with plotly --with kaleido --with pandas \\
        benches/tools/plot_benchmarks.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "output" / "runs"
OUT_DIR = REPO_ROOT / "output"
PLOT_DIR = REPO_ROOT / "doc" / "plots"

# Stable solver order and colours so both figures read the same way.
SOLVER_ORDER = ["apex-solver", "factrs", "tiny-solver", "Ceres", "GTSAM", "g2o"]
SOLVER_COLOR = {
    "apex-solver": "#4C78A8",
    "factrs": "#72B7B2",
    "tiny-solver": "#54A24B",
    "Ceres": "#EECA3B",
    "GTSAM": "#E45756",
    "g2o": "#B279A2",
}

DATASETS_2D = ["M3500", "mit", "city10000", "ring"]
DATASETS_3D = ["sphere2500", "parking-garage", "torus3D", "cubicle"]
DATASETS_BA = ["Ladybug", "Trafalgar", "Dubrovnik", "Venice"]


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a column to float, turning '-' and error strings into NaN."""
    return pd.to_numeric(series, errors="coerce")


def load_runs(bench: str) -> pd.DataFrame:
    """Load the aggregate CSV of each run.

    Matches ``<bench>_run<N>.csv`` exactly, so the per-solver C++ CSVs archived
    alongside them (``<bench>_run<N>_<exe>_results.csv``) are not picked up.
    """
    rx = re.compile(rf"^{re.escape(bench)}_run(\d+)\.csv$")
    matches = sorted(
        ((int(m.group(1)), p) for p in RUN_DIR.iterdir() if (m := rx.match(p.name))),
    )
    if not matches:
        sys.exit(f"no run CSVs matching {bench}_run<N>.csv in {RUN_DIR}")
    frames = []
    for run_idx, path in matches:
        df = pd.read_csv(path)
        df["run"] = run_idx
        frames.append(df)
    print(f"  loaded {len(matches)} runs: {[p.name for _, p in matches]}")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Mean/std/n per (dataset, solver) over runs, ignoring failed rows."""
    for col in value_cols:
        df[col] = _numeric(df[col])
    grouped = df.groupby(["dataset", "solver"], as_index=False).agg(
        **{
            f"{col}_{stat}": (col, stat)
            for col in value_cols
            for stat in ("mean", "std")
        },
        n=("run", "count"),
    )
    # A single successful run yields NaN std; report 0 so error bars render.
    for col in value_cols:
        grouped[f"{col}_std"] = grouped[f"{col}_std"].fillna(0.0)
    return grouped


def _bar_traces(agg, datasets, ycol, ecol, showlegend):
    """One bar trace per solver across the given datasets."""
    traces = []
    for solver in SOLVER_ORDER:
        sub = agg[(agg["solver"] == solver) & (agg["dataset"].isin(datasets))]
        if sub.empty:
            continue
        sub = sub.set_index("dataset").reindex(datasets)
        traces.append(
            go.Bar(
                name=solver,
                x=datasets,
                y=sub[ycol],
                error_y={"type": "data", "array": sub[ecol], "visible": True},
                marker_color=SOLVER_COLOR.get(solver),
                legendgroup=solver,
                showlegend=showlegend,
            )
        )
    return traces


def save(fig: go.Figure, stem: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    html, png = PLOT_DIR / f"{stem}.html", PLOT_DIR / f"{stem}.png"
    fig.write_html(html, include_plotlyjs="cdn")
    fig.write_image(png, width=1400, height=900, scale=2)
    print(f"  wrote {html.relative_to(REPO_ROOT)} and {png.relative_to(REPO_ROOT)}")


def build_odometry() -> None:
    print("odometry:")
    df = load_runs("odometry_pose_benchmark")
    for col in ("final_cost", "final_chi2", "edges", "vertices"):
        df[col] = _numeric(df[col])

    # Cost per degree of freedom, so datasets of very different size share one axis.
    #
    # We normalize the *unweighted* cost (0.5*sum||r||^2) rather than chi2, because every
    # solver here is configured with unit/identity information (g2o setInformation(I),
    # GTSAM noiseModel::Unit, Ceres identity, apex BetweenFactor without information).
    # That unweighted objective is what they all actually minimize; chi2 re-weights by the
    # file's information matrix, whose scale varies by orders of magnitude between datasets
    # (the chi2/cost ratio ranges from ~2 on parking-garage to ~7e5 on cubicle) and would
    # grade solvers on an objective none of them was given.
    dof = df["edges"] - df["vertices"]
    df["norm_cost"] = df["final_cost"].where(dof > 0) / dof.where(dof > 0)

    agg = aggregate(df, ["norm_cost", "final_cost", "final_chi2", "elapsed_ms"])
    agg.to_csv(OUT_DIR / "odometry_aggregated.csv", index=False)
    print(f"  wrote output/odometry_aggregated.csv ({len(agg)} rows)")

    datasets = [d for d in DATASETS_2D + DATASETS_3D if d in set(agg["dataset"])]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Solution cost — final objective per degree of freedom (lower is better)",
            "Runtime (lower is better)",
        ),
    )
    for tr in _bar_traces(agg, datasets, "norm_cost_mean", "norm_cost_std", True):
        fig.add_trace(tr, row=1, col=1)
    for tr in _bar_traces(agg, datasets, "elapsed_ms_mean", "elapsed_ms_std", False):
        fig.add_trace(tr, row=2, col=1)

    fig.update_yaxes(title_text="cost / (m − n)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="time (ms)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="dataset  (2D: M3500…ring   |   3D: sphere2500…cubicle)",
                     row=2, col=1)
    fig.update_layout(
        title="Pose Graph Optimization — cost and runtime (mean ± std over 5 runs)"
        "<br><sup>missing bars = solver failed on that dataset</sup>",
        barmode="group",
        height=900,
        width=1400,
        legend_title_text="solver",
        template="plotly_white",
    )
    save(fig, "odometry_benchmark")


def build_ba() -> None:
    print("bundle adjustment:")
    df = load_runs("bundle_adjustment_benchmark")
    df = df[df["dataset"].isin(DATASETS_BA)]
    df["solver"] = df["solver"].replace({"Apex-Solver": "apex-solver", "Gtsam": "GTSAM"})

    agg = aggregate(df, ["final_rmse", "time_seconds"])
    agg.to_csv(OUT_DIR / "ba_aggregated.csv", index=False)
    print(f"  wrote output/ba_aggregated.csv ({len(agg)} rows)")

    datasets = [d for d in DATASETS_BA if d in set(agg["dataset"])]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Final reprojection RMSE (lower is better)",
            "Runtime (lower is better)",
        ),
    )
    for tr in _bar_traces(agg, datasets, "final_rmse_mean", "final_rmse_std", True):
        fig.add_trace(tr, row=1, col=1)
    for tr in _bar_traces(agg, datasets, "time_seconds_mean", "time_seconds_std", False):
        fig.add_trace(tr, row=2, col=1)

    # Log scale: g2o lands at 8-13 px while the others are sub-pixel, so a linear axis
    # flattens the differences that actually matter.
    fig.update_yaxes(title_text="RMSE (pixels)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="time (s)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="BAL dataset", row=2, col=1)
    fig.update_layout(
        title="Bundle Adjustment — reprojection RMSE and runtime (mean ± std over 5 runs)"
        "<br><sup>missing bars = solver exceeded the 10-minute timeout</sup>",
        barmode="group",
        height=900,
        width=1400,
        legend_title_text="solver",
        template="plotly_white",
    )
    save(fig, "ba_benchmark")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "odometry"):
        build_odometry()
    if which in ("all", "ba"):
        build_ba()
