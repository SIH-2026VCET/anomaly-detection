"""
Static visualizations saved to reports/figures/ for the README / report.
The Streamlit dashboard (dashboard/app.py) additionally builds its own
interactive Plotly charts on the fly.
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualization")
sns.set_theme(style="whitegrid")


def plot_population_distribution(scored: pd.DataFrame, long_df: pd.DataFrame, lot_id: str, parameter: str):
    sub = scored[(scored["lot_id"] == lot_id) & (scored["parameter"] == parameter)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    normal = sub[sub["population_flag"] == 0]["parameter_value_168h"]
    anomalous = sub[sub["population_flag"] == 1]["parameter_value_168h"]
    sns.histplot(normal, color="#4C72B0", label="Normal (population)", kde=True, ax=ax, alpha=0.6)
    if len(anomalous):
        for v in anomalous:
            ax.axvline(v, color="#C44E52", linestyle="--", linewidth=1.5)
        ax.axvline(anomalous.iloc[0], color="#C44E52", linestyle="--", linewidth=1.5, label="Population anomaly")
    ax.axvline(sub["datasheet_max"].iloc[0], color="black", linestyle=":", linewidth=2, label="Datasheet limit")
    ax.set_title(f"Population distribution @168h - {lot_id} / {parameter}")
    ax.set_xlabel(parameter)
    ax.legend()
    fig.tight_layout()
    path = config.FIGURES_DIR / "population_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_timeseries_with_anomalies(long_df: pd.DataFrame, scored: pd.DataFrame, lot_id: str, parameter: str, n_highlight=3):
    sub_ids = scored[(scored["lot_id"] == lot_id) & (scored["parameter"] == parameter)]
    anomalous_ids = sub_ids[sub_ids["risk_level"] != "NORMAL"]["component_id"].head(n_highlight).tolist()
    normal_ids = sub_ids[sub_ids["risk_level"] == "NORMAL"]["component_id"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ts = long_df[(long_df["lot_id"] == lot_id) & (long_df["parameter"] == parameter)]
    for cid in normal_ids:
        g = ts[ts["component_id"] == cid].sort_values("time_h")
        ax.plot(g["time_h"], g["parameter_value"], color="grey", alpha=0.35, linewidth=1)
    for cid in anomalous_ids:
        g = ts[ts["component_id"] == cid].sort_values("time_h")
        ax.plot(g["time_h"], g["parameter_value"], linewidth=2.5, marker="o", label=f"{cid} (anomalous)")
    dmax = ts["datasheet_max"].iloc[0]
    ax.axhline(dmax, color="black", linestyle=":", label="Datasheet limit")
    ax.set_xlabel("Burn-in time (h)")
    ax.set_ylabel(parameter)
    ax.set_title(f"Burn-in trajectories - {lot_id} / {parameter}\n(grey = normal, colored = AI-flagged anomalies)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = config.FIGURES_DIR / "timeseries_anomalies.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_anomaly_score_distribution(scored: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(data=scored, x="final_anomaly_score", hue="risk_level", multiple="stack", ax=ax,
                 palette={"NORMAL": "#55A868", "WARNING": "#DD8452", "SUSPICIOUS": "#C44E52", "CRITICAL": "#8C0800"})
    ax.set_title("Final combined anomaly score distribution, by risk level")
    ax.set_xlabel("Final Anomaly Score (0-1)")
    fig.tight_layout()
    path = config.FIGURES_DIR / "anomaly_score_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_static_vs_dynamic(scored: pd.DataFrame):
    counts = pd.DataFrame({
        "Static Screening": [
            (scored["spec_violation"] == 1).sum(),
            (scored["spec_violation"] == 0).sum(),
        ],
        "Dynamic AI Screening": [
            (scored["risk_level"] != "NORMAL").sum(),
            (scored["risk_level"] == "NORMAL").sum(),
        ],
    }, index=["Flagged", "Passed"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts.T.plot(kind="bar", stacked=True, ax=ax, color=["#C44E52", "#55A868"])
    ax.set_title("Static vs Dynamic AI Screening - components flagged")
    ax.set_ylabel("Number of components")
    fig.tight_layout()
    path = config.FIGURES_DIR / "static_vs_dynamic.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_component_health_trend(scored: pd.DataFrame, long_df: pd.DataFrame, component_id: str):
    row = scored[scored["component_id"] == component_id].iloc[0]
    ts = long_df[long_df["component_id"] == component_id].sort_values("time_h")
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ts["time_h"], ts["parameter_value"], marker="o", color="#4C72B0", label=row["parameter"])
    ax1.axhline(row["datasheet_max"], color="black", linestyle=":", label="Datasheet limit")
    ax1.set_xlabel("Burn-in time (h)")
    ax1.set_ylabel(row["parameter"], color="#4C72B0")
    ax1.set_title(f"Component health trend - {component_id} ({row['risk_level']})")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = config.FIGURES_DIR / "component_health_trend.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run(scored: pd.DataFrame, long_df: pd.DataFrame):
    # pick the largest lot/parameter combo for clean illustrative plots
    lot_id, parameter = (
        scored.groupby(["lot_id", "parameter"]).size().idxmax()
    )
    plot_population_distribution(scored, long_df, lot_id, parameter)
    plot_timeseries_with_anomalies(long_df, scored, lot_id, parameter)
    plot_anomaly_score_distribution(scored)
    plot_static_vs_dynamic(scored)

    # demo component: PASS on datasheet but flagged anomalous by AI, worst score first
    demo_candidates = scored[
        (scored["datasheet_status"] == "PASS") & (scored["ai_status"] == "ANOMALOUS")
    ].sort_values("final_anomaly_score", ascending=False)
    if len(demo_candidates):
        plot_component_health_trend(scored, long_df, demo_candidates.iloc[0]["component_id"])

    logger.info("Saved figures to %s", config.FIGURES_DIR)


if __name__ == "__main__":
    scored = pd.read_csv(config.PROCESSED_WIDE_CSV)
    long_df = pd.read_csv(config.DATA_PROCESSED_DIR / "long_clean.csv")
    run(scored, long_df)
