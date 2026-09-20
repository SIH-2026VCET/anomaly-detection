"""
FastAPI backend for the Dynamic Anomaly Detection prototype.

Serves the precomputed results from data/processed/components_scored_wide.csv
(produced by `python run.py`). This is a read/serve layer for the prototype,
not a live-retraining service — this keeps the hackathon prototype simple and
reliable, per the project brief's "prioritize a fully working Streamlit
application over unnecessary architectural complexity" guidance.

Run with:
    uvicorn api.main:app --reload --port 8000

Docs: http://localhost:8000/docs
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config, explainability

app = FastAPI(
    title="Dynamic Anomaly Detection API",
    description="AI-driven population + temporal anomaly detection for component burn-in/ESS screening (prototype, synthetic data).",
    version="0.1.0",
)

_scored: Optional[pd.DataFrame] = None
_long: Optional[pd.DataFrame] = None


def _get_data():
    global _scored, _long
    if _scored is None:
        if not config.PROCESSED_WIDE_CSV.exists():
            raise HTTPException(status_code=503, detail="Pipeline not yet run. Run `python run.py` first.")
        _scored = pd.read_csv(config.PROCESSED_WIDE_CSV)
        _long = pd.read_csv(config.PROCESSED_LONG_CSV)
    return _scored, _long


class PredictRequest(BaseModel):
    component_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/components")
def list_components(risk_level: Optional[str] = None, lot_id: Optional[str] = None):
    scored, _ = _get_data()
    df = scored
    if risk_level:
        df = df[df["risk_level"] == risk_level.upper()]
    if lot_id:
        df = df[df["lot_id"] == lot_id]
    cols = ["component_id", "lot_id", "component_type", "parameter", "datasheet_status",
            "ai_status", "risk_level", "case_classification", "final_anomaly_score"]
    return df[cols].to_dict(orient="records")


@app.get("/components/{component_id}")
def get_component(component_id: str):
    scored, long_df = _get_data()
    if component_id not in scored["component_id"].values:
        raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
    explanation = explainability.explain_component(scored, component_id)
    row = scored[scored["component_id"] == component_id].iloc[0].to_dict()
    ts = long_df[long_df["component_id"] == component_id][["time_h", "parameter_value"]].to_dict(orient="records")
    return {"details": row, "explanation": explanation, "timeseries": ts}


@app.get("/lots/{lot_id}")
def get_lot(lot_id: str):
    scored, _ = _get_data()
    df = scored[scored["lot_id"] == lot_id]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Lot {lot_id} not found")
    return {
        "lot_id": lot_id,
        "n_components": int(len(df)),
        "n_flagged": int((df["risk_level"] != "NORMAL").sum()),
        "n_spec_failures": int((df["spec_violation"] == 1).sum()),
        "components": df["component_id"].tolist(),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Return the anomaly assessment for a single already-scored component
    (prototype: looks up the precomputed result rather than scoring live)."""
    scored, _ = _get_data()
    if req.component_id not in scored["component_id"].values:
        raise HTTPException(status_code=404, detail=f"Component {req.component_id} not found")
    return explainability.explain_component(scored, req.component_id)


@app.get("/anomalies")
def get_anomalies(min_risk: str = "WARNING"):
    scored, _ = _get_data()
    order = ["NORMAL", "WARNING", "SUSPICIOUS", "CRITICAL"]
    min_idx = order.index(min_risk.upper()) if min_risk.upper() in order else 1
    df = scored[scored["risk_level"].apply(lambda r: order.index(r) >= min_idx)]
    cols = ["component_id", "lot_id", "risk_level", "case_classification", "final_anomaly_score"]
    return df[cols].sort_values("final_anomaly_score", ascending=False).to_dict(orient="records")


@app.get("/statistics/{lot_id}")
def lot_statistics(lot_id: str):
    scored, _ = _get_data()
    df = scored[scored["lot_id"] == lot_id]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Lot {lot_id} not found")
    stats = []
    for parameter, g in df.groupby("parameter"):
        v = g["parameter_value_168h"]
        stats.append({
            "parameter": parameter,
            "mean": round(float(v.mean()), 4),
            "median": round(float(v.median()), 4),
            "std": round(float(v.std(ddof=0)), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
            "p5": round(float(v.quantile(0.05)), 4),
            "p95": round(float(v.quantile(0.95)), 4),
        })
    return {"lot_id": lot_id, "statistics": stats}
