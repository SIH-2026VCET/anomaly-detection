"""
Streamlit dashboard for the AI-Driven Dynamic Anomaly Detection prototype.

Run with:
    streamlit run dashboard/app.py

If data/processed/components_scored_wide.csv does not exist yet, run the
pipeline first:
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, explainability

st.set_page_config(page_title="Dynamic Anomaly Detection - Burn-In Screening", layout="wide")

RISK_COLORS = {
    "NORMAL": "#55A868", "WARNING": "#DD8452",
    "SUSPICIOUS": "#C44E52", "CRITICAL": "#8C0800",
}


@st.cache_data
def load_data():
    if not config.PROCESSED_WIDE_CSV.exists():
        return None, None
    scored = pd.read_csv(config.PROCESSED_WIDE_CSV)
    long_df = pd.read_csv(config.PROCESSED_LONG_CSV)
    return scored, long_df


scored, long_df = load_data()

st.title("🛰️ AI-Driven Dynamic Anomaly Detection — Component Burn-In & Screening")
st.caption(
    "Prototype for population-aware + temporal anomaly detection in ESS/burn-in testing. "
    "Built on a **synthetic**, clearly-labeled dataset (not real industrial data)."
)

if scored is None:
    st.error(
        "No processed data found. Run the pipeline first from the project root:\n\n"
        "`python run.py`"
    )
    st.stop()

tab_overview, tab_search, tab_lot, tab_timeseries, tab_eval = st.tabs(
    ["📊 Overview", "🔎 Component Search", "🏭 Lot Analysis", "📈 Time-Series View", "✅ Static vs Dynamic"]
)

# ---------------------------------------------------------------------------
# TAB 1 - Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total components", len(scored))
    c2.metric("Normal", int((scored["risk_level"] == "NORMAL").sum()))
    c3.metric("Warning", int((scored["risk_level"] == "WARNING").sum()))
    c4.metric("Suspicious", int((scored["risk_level"] == "SUSPICIOUS").sum()))
    c5.metric("Critical", int((scored["risk_level"] == "CRITICAL").sum()))
    c6.metric("Spec failures", int((scored["spec_violation"] == 1).sum()))

    n_latent = int(((scored["datasheet_status"] == "PASS") & (scored["ai_status"] == "ANOMALOUS")).sum())
    st.info(
        f"**{n_latent} components PASS the fixed datasheet limit but are flagged ANOMALOUS "
        "by the dynamic AI system** — these are exactly the latent risks static screening misses."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            scored, x="final_anomaly_score", color="risk_level",
            color_discrete_map=RISK_COLORS, nbins=30,
            title="Final Anomaly Score distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        case_counts = scored["case_classification"].value_counts().reset_index()
        case_counts.columns = ["case", "count"]
        fig = px.bar(case_counts, x="case", y="count", title="Case classification breakdown",
                     color="case")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("All components")
    st.dataframe(
        scored[[
            "component_id", "lot_id", "component_type", "parameter",
            "datasheet_status", "ai_status", "risk_level", "case_classification",
            "final_anomaly_score", "qa_recommendation", "failure_mode",
        ]].sort_values("final_anomaly_score", ascending=False),
        use_container_width=True, height=350,
    )

# ---------------------------------------------------------------------------
# TAB 2 - Component Search / Explanation
# ---------------------------------------------------------------------------
with tab_search:
    comp_id = st.selectbox("Component ID", sorted(scored["component_id"].unique()))
    row = scored[scored["component_id"] == comp_id].iloc[0]
    exp = explainability.explain_component(scored, comp_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasheet Status", exp["datasheet_status"])
    c2.metric("AI Status", exp["ai_status"])
    c3.metric("Risk Level", exp["risk_level"])
    c4.metric("Final Anomaly Score", f"{exp['final_anomaly_score']:.2f}")

    if exp["datasheet_status"] == "PASS" and exp["ai_status"] == "ANOMALOUS":
        st.warning("⚠️ This component **passes** the fixed datasheet limit, "
                   "but is flagged **ANOMALOUS** by dynamic AI screening.")

    st.markdown(f"**Case classification:** `{exp['case_classification']}`  \n"
                f"**QA recommendation:** `{exp['qa_recommendation']}`")

    st.subheader("Why was this component flagged?")
    for r in exp["reasons"]:
        st.markdown(f"- {r}")

    st.subheader("Burn-in trajectory")
    ts = long_df[long_df["component_id"] == comp_id].sort_values("time_h")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["time_h"], y=ts["parameter_value"], mode="lines+markers",
                              name=row["parameter"], line=dict(width=3)))
    fig.add_hline(y=row["datasheet_max"], line_dash="dot", line_color="black",
                  annotation_text="Datasheet max")
    if row["datasheet_min"] not in (0, None):
        fig.add_hline(y=row["datasheet_min"], line_dash="dot", line_color="grey",
                      annotation_text="Datasheet min")
    fig.update_layout(title=f"{comp_id} — {row['parameter']}", xaxis_title="Burn-in time (h)",
                       yaxis_title=row["parameter"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Population comparison (168h, same lot & parameter)")
    peers = scored[(scored["lot_id"] == row["lot_id"]) & (scored["parameter"] == row["parameter"])]
    fig = px.box(peers, y="parameter_value_168h", points="all", title="Lot distribution @168h")
    fig.add_hline(y=row["parameter_value_168h"], line_color="red",
                  annotation_text=f"{comp_id}", line_width=3)
    fig.add_hline(y=row["datasheet_max"], line_dash="dot", line_color="black",
                  annotation_text="Datasheet limit")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 - Lot Analysis
# ---------------------------------------------------------------------------
with tab_lot:
    lot_id = st.selectbox("Lot ID", sorted(scored["lot_id"].unique()))
    lot_df = scored[scored["lot_id"] == lot_id]

    c1, c2, c3 = st.columns(3)
    c1.metric("Components in lot", len(lot_df))
    c2.metric("Flagged anomalous", int((lot_df["risk_level"] != "NORMAL").sum()))
    c3.metric("Spec failures", int((lot_df["spec_violation"] == 1).sum()))

    st.dataframe(
        lot_df[[
            "component_id", "component_type", "parameter", "parameter_value_168h",
            "robust_z_final", "population_anomaly_score", "temporal_anomaly_score",
            "final_anomaly_score", "risk_level",
        ]].sort_values("final_anomaly_score", ascending=False),
        use_container_width=True, height=350,
    )

    fig = px.scatter(
        lot_df, x="population_anomaly_score", y="temporal_anomaly_score",
        color="risk_level", color_discrete_map=RISK_COLORS, hover_name="component_id",
        size="final_anomaly_score", title=f"{lot_id}: Population vs Temporal anomaly score",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 - Time-Series View
# ---------------------------------------------------------------------------
with tab_timeseries:
    col1, col2 = st.columns(2)
    lot_sel = col1.selectbox("Lot", sorted(scored["lot_id"].unique()), key="ts_lot")
    param_sel = col2.selectbox("Parameter", sorted(scored["parameter"].unique()), key="ts_param")

    sub_ids = scored[(scored["lot_id"] == lot_sel) & (scored["parameter"] == param_sel)]
    ts_sub = long_df[(long_df["lot_id"] == lot_sel) & (long_df["parameter"] == param_sel)]

    fig = go.Figure()
    for _, r in sub_ids.iterrows():
        g = ts_sub[ts_sub["component_id"] == r["component_id"]].sort_values("time_h")
        color = RISK_COLORS.get(r["risk_level"], "grey")
        fig.add_trace(go.Scatter(
            x=g["time_h"], y=g["parameter_value"], mode="lines+markers",
            name=r["component_id"], line=dict(color=color, width=1.5 if r["risk_level"] == "NORMAL" else 3),
            opacity=0.35 if r["risk_level"] == "NORMAL" else 1.0,
        ))
    if len(sub_ids):
        fig.add_hline(y=sub_ids["datasheet_max"].iloc[0], line_dash="dot", line_color="black",
                      annotation_text="Datasheet limit")
    fig.update_layout(title=f"{lot_sel} / {param_sel} — all component trajectories",
                       xaxis_title="Burn-in time (h)", yaxis_title=param_sel, showlegend=False,
                       height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bold colored lines = AI-flagged anomalies (colored by risk level). "
               "Thin grey/faded lines = normal population.")

# ---------------------------------------------------------------------------
# TAB 5 - Static vs Dynamic evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    import json
    if config.EVAL_REPORT_PATH.exists():
        report = json.load(open(config.EVAL_REPORT_PATH))
        full = report["full_dataset"]

        st.subheader("Static Datasheet Screening vs Dynamic AI Screening")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("### 📏 Static Screening")
            st.json(full["static_screening"])
        with colB:
            st.markdown("### 🤖 Dynamic AI Screening")
            st.json(full["dynamic_ai_screening"])

        st.subheader("Latent defect detection (still within datasheet limit)")
        la = full["latent_defect_analysis"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latent defects in dataset", la["n_latent_defects_in_split"])
        c2.metric("Caught by static screening", la["n_caught_by_static_screening"])
        c3.metric("Caught by dynamic AI screening", la["n_caught_by_dynamic_ai_screening"])
        st.caption(
            "By construction, static screening can NEVER catch a latent defect "
            "(it stays within the datasheet limit). This is the central "
            "demonstration of the project."
        )

        st.subheader("Detection by injected failure mode")
        bfm = pd.DataFrame(full["by_failure_mode"])
        st.dataframe(bfm, use_container_width=True)

        fig = px.bar(bfm, x="failure_mode", y=["static_detected", "dynamic_detected"],
                     barmode="group", title="Components detected per failure mode")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No evaluation report found. Run `python run.py` first.")

st.divider()
st.caption(
    "⚠️ Hackathon-quality prototype using a synthetic labeled dataset. "
    "Weights, thresholds and contamination parameters are configurable in `src/config.py` "
    "and are NOT scientifically calibrated. Not production-ready."
)
