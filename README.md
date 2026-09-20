# AI-Driven Dynamic Anomaly Detection in Component Burn-In & Screening

A hackathon-quality working prototype that detects anomalous electronic
components during Environmental Stress Screening (ESS) / burn-in testing —
**before** they necessarily violate a fixed datasheet limit.

> ⚠️ **Prototype disclaimer**: This is a hackathon/college-project-quality
> demonstration built on a **synthetic, labeled dataset**. It is not
> production-ready, the scoring weights/thresholds are not scientifically
> calibrated, and it has not been validated against real industrial
> ESS/burn-in data. See [Limitations](#limitations).

---

## Project Overview

Traditional component screening asks one question:

```
measurement < datasheet limit ?  →  PASS / FAIL
```

This misses components that are technically within spec but behave very
differently from their manufacturing lot, or that are drifting/degrading
across burn-in stages. This project builds an AI-based system that adds
two more questions on top of the datasheet check:

```
Is this component unusual compared to its own lot?     (Module A)
Is this component's behavior changing abnormally over time? (Module B)
```

and combines all three signals into one explainable risk score.

---

## Problem Statement

In high-reliability industries (space, aerospace, defense, electronics),
components are burned in and measured at several time points
(0h, 24h, 96h, 168h). Static screening only checks the final value against
a fixed limit, e.g.:

```
Lot average leakage current = 10 µA
Component leakage current   = 45 µA
Datasheet maximum           = 50 µA
→ 45 < 50 → PASS
```

...even though 45 µA may be wildly abnormal for that lot. This project
detects such **latent anomalies** before they cause a field failure.

---

## Proposed Solution

Combine three independent signals into a single, explainable, configurable
anomaly score:

1. **Population score** — how unusual is this component vs. its lot peers
   (Module A)?
2. **Temporal score** — how unusual is this component's burn-in trajectory
   (Module B)?
3. **Specification score** — how close is it to (or past) the datasheet
   limit?

```
Final Score = 0.40 × Population + 0.40 × Temporal + 0.20 × Specification
```

The final score maps to a risk level (NORMAL / WARNING / SUSPICIOUS /
CRITICAL) and a case classification (see below), and every flagged
component gets a human-readable explanation.

---

## Architecture

```
data/raw (synthetic dataset)
        │
        ▼
src/preprocessing.py        - validation, missing values, chronology, lot grouping
        │
        ├──► src/dynamic_detection.py   - Module A: population/lot-aware detection
        │
        └──► src/temporal_detection.py  - Module B: burn-in trajectory detection
                        │
                        ▼
        src/scoring.py       - combined score, risk level, case classification
                        │
                        ▼
        src/explainability.py - SHAP (proxy model) + rule-based reasons
                        │
                        ▼
        src/evaluation.py    - static vs dynamic comparison, precision/recall/F1/ROC-AUC
                        │
                        ▼
        src/visualization.py  - matplotlib/seaborn report figures
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
dashboard/app.py (Streamlit)     api/main.py (FastAPI, optional)
```

`run.py` runs the whole pipeline end to end.

---

## Modules

### Module A — Dynamic / Population-Aware Anomaly Detection
(`src/dynamic_detection.py`)

For every **lot × parameter** peer group (not a single global model — lots
can have legitimate process differences), we compute at the final (168h)
measurement:

| Method | What it captures |
|---|---|
| Z-score | standard deviation from lot mean |
| Robust Z-score (median/MAD) | resistant to the very outliers we're hunting for |
| IQR method | Q1/Q3 fence violations |
| Percentile rank | how extreme within the lot's own distribution |
| Isolation Forest (lot-wise, multivariate) | trained on the full 4-point trajectory shape, standardized within the lot |

These five signals are combined into a single `population_anomaly_score`
(0–1).

### Module B — Time-Series / Temporal Anomaly Detection
(`src/temporal_detection.py`)

For every component's burn-in trajectory (0h→24h→96h→168h) we extract:

- initial value, final value, absolute & percentage change
- mean, std, max, min
- linear-regression slope (µ-unit/hour)
- early slope (0h→24h) vs. late slope (96h→168h), and their difference
  (**acceleration** — is degradation speeding up?)
- max single-step rate of change
- **peer-relative jump count**: each of the 3 burn-in step changes is
  compared, via a robust z-score, against the *same* step across the
  component's lot peers (far more reliable than judging a 4-point series
  against its own short history)

These features feed a rule-based drift/jump/acceleration score AND a
lot-wise Isolation Forest on the temporal-feature space, combined into
`temporal_anomaly_score` (0–1).

---

## Dataset

This prototype uses the **SIH26170 solution-aligned synthetic burn-in
dataset** (provided for this problem statement) rather than generating a
new one from scratch — it already satisfies every requirement in the
brief:

- **240 components**, **12 lots**, **4 component types** (FPGA,
  Microprocessor, Power MOSFET, RF MMIC), **3 parameters** (Leakage
  Current, Supply Current, Threshold Voltage)
- Measured at **0h / 24h / 96h / 168h**
- Explicitly labeled `failure_mode` per component:

  | failure_mode | n | meaning |
  |---|---|---|
  | healthy | 163 | normal population behavior |
  | gradual_drift | 19 | steadily worsening over burn-in |
  | latent_defect | 17 | stays within datasheet limit but statistically abnormal |
  | sudden_jump | 15 | abrupt change at a later stage |
  | lot_outlier | 14 | abnormal vs. lot population, not necessarily drifting |
  | accelerating_drift | 12 | degradation rate increasing |

- Ground-truth labels for evaluation: `anomaly_label_early`,
  `future_failure_label_168h`
- A **lot-wise** train/validation/test split (`split` column), so no
  lot-level statistic ever leaks across the split boundary during
  evaluation.

`src/data_generation.py` documents this choice and can re-snapshot the
dataset into `data/synthetic/` for provenance.

**This is synthetic data. It must never be treated as real industrial
measurements.**

---

## Feature Engineering

See Module A / Module B above. All features are interpretable
(z-scores, percentiles, slopes, % changes) — no opaque embeddings — so
every flagged component can be explained in plain language.

---

## Algorithms

- Classical statistics: mean/std, median/MAD, IQR, percentile rank
- `sklearn.ensemble.IsolationForest` (lot-wise, both for population and
  temporal-feature spaces)
- `sklearn.linear_model.LinearRegression` for trend slopes
- EWMA-style smoothing utilities
- `sklearn.ensemble.RandomForestClassifier` — an auxiliary, supervised
  proxy model (see Explainability) used only to run SHAP; it is **not**
  the source of the final anomaly score

## Dynamic Anomaly Detection

See **Module A** above. Operates lot-wise/peer-group-wise, never as one
global detector across all lots.

## Temporal Anomaly Detection

See **Module B** above. Detects sudden jumps, gradual/accelerating drift,
and peer-relative abnormal trajectory shape.

## Combined Scoring

`src/scoring.py`:

```
Final Score = 0.40 × Population Score
            + 0.40 × Temporal Score
            + 0.20 × Specification Score
```

Weights live in `src/config.SCORE_WEIGHTS` and are explicitly
**prototype values**, not scientifically established ones.

Risk thresholds (`src/config.RISK_THRESHOLDS`, also configurable):

```
0.00 – 0.30  → NORMAL
0.30 – 0.60  → WARNING
0.60 – 0.80  → SUSPICIOUS
0.80 – 1.00  → CRITICAL
```

**Case classification** (the central "PASS on datasheet, ANOMALOUS by AI"
demonstration):

| Case | Condition |
|---|---|
| NORMAL | within limit, normal population, stable over time |
| LATENT_ANOMALY | within limit + strong population deviation |
| DEGRADATION_RISK | within limit + strong temporal drift |
| SPECIFICATION_FAILURE | exceeds the datasheet limit |
| HIGH_RISK_COMPONENT | population anomaly + temporal anomaly (+ near the spec limit) |

## Explainability

`src/explainability.py`. Because Module A/B are **unsupervised**
(Isolation Forest + classical stats), SHAP does not directly apply to
their raw anomaly scores. Following the brief's guidance to use SHAP
"where appropriate, particularly for the supervised/ML component," we:

1. Train a `RandomForestClassifier` **auxiliary/diagnostic** model on the
   TRAIN split only, predicting the ground-truth
   `future_failure_label_168h` from the Module A/B features. This model
   does **not** produce the final anomaly score — it exists purely so SHAP
   can show *global* feature importance (saved to
   `reports/shap_global_feature_importance.csv`).
2. For the **per-component** explanation shown in the dashboard, we use a
   fully transparent **rule-based reason generator** directly on the
   Module A/B signals (robust z, % drift, jump count, spec proximity) —
   avoiding "explaining a black box with another black box."

Example output:

```
Component ID: FPGA_0001
Risk Level: WARNING
Reasons:
  • Leakage Current (µA) is 25.5σ above/below the lot median (robust z-score)
  • Leakage Current (µA) changed 291% during burn-in (0h→168h)
  • Degradation rate is accelerating
  • Measurement is approaching the datasheet limit
```

## Dashboard

`dashboard/app.py` (Streamlit), 5 tabs:

1. **Overview** — totals by risk level, score distribution, case-classification
   breakdown, full sortable table
2. **Component Search** — pick a `component_id`, see status, score, reasons,
   burn-in trajectory, lot comparison box plot
3. **Lot Analysis** — pick a `lot_id`, see lot stats and a
   population-vs-temporal scatter of every component in it
4. **Time-Series View** — every trajectory in a lot/parameter, anomalies
   highlighted in color against a faded normal population
5. **Static vs Dynamic** — the evaluation report rendered live, including
   the latent-defect comparison

## API

`api/main.py` (FastAPI, optional — serves the precomputed results rather
than retraining live, to keep the prototype simple and reliable):

```
GET  /health
GET  /components?risk_level=&lot_id=
GET  /components/{component_id}
GET  /lots/{lot_id}
POST /predict            {"component_id": "..."}
GET  /anomalies?min_risk=WARNING
GET  /statistics/{lot_id}
```

Interactive docs at `http://localhost:8000/docs`.

## Evaluation

`src/evaluation.py` compares **static datasheet screening**
(`spec_violation`) against **dynamic AI screening** (`risk_level !=
NORMAL`), on the full 240-component dataset, against the ground-truth
`future_failure_label_168h` label:

| Metric | Static Screening | Dynamic AI Screening |
|---|---|---|
| Precision | 1.00 | 0.56 |
| Recall | 0.30 | 0.44 |
| F1 | 0.465 | 0.491 |
| ROC-AUC | n/a (binary rule) | 0.61 |

**The central result — latent defects (genuinely unsafe, but still within
the datasheet limit):**

| | Static Screening | Dynamic AI Screening |
|---|---|---|
| Latent defects in dataset | 46 | 46 |
| Caught | **0** (structurally impossible) | **9** |

Static screening can *never* catch a latent defect by construction (it
stays within the limit). Dynamic AI screening catches roughly 1 in 5 of
them — real, honest, imperfect numbers computed directly from the
dataset, not manufactured to look impressive. Full breakdown by injected
`failure_mode` is in `reports/evaluation_report.json` and the dashboard's
"Static vs Dynamic" tab.

---

## Installation

```bash
cd dynamic-anomaly-detection
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline (preprocessing → Module A → Module B → scoring →
explainability → evaluation → figures):

```bash
python run.py
```

This writes:
- `data/processed/components_scored_wide.csv` — every component, fully scored
- `models/risk_classifier.joblib` — the SHAP proxy model
- `reports/evaluation_report.json` — static vs dynamic metrics
- `reports/figures/*.png` — distribution, trajectory, and score plots

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Launch the API (optional):

```bash
uvicorn api.main:app --reload --port 8000
```

Run tests:

```bash
pytest tests/ -v
```

## Example Output

```
Component: FPGA_0001
Datasheet leakage limit: 50 µA
0h    → 10.4 µA
24h   → 13.3 µA
96h   → 25.8 µA
168h  → 40.7 µA         (still below 50 µA)

Static Screening:   PASS
Dynamic AI Screening: WARNING (score 0.53) — case: HIGH_RISK_COMPONENT

Reasons:
  • 25.5σ above the lot median at 168h
  • +291% change over burn-in
  • Degradation rate accelerating
  • Approaching the datasheet limit
```

## Limitations

- Synthetic dataset — not validated against real industrial ESS/burn-in
  data.
- Small sample size (240 components, 4-point trajectories) limits
  statistical power, especially for Isolation Forest and the held-out
  test split.
- Scoring weights (0.40/0.40/0.20) and risk thresholds are prototype
  choices, not derived from a reliability-engineering study.
- The SHAP proxy classifier's near-perfect held-out AUC likely reflects
  the small, clean, synthetic test split rather than real-world
  generalization — treat it as a diagnostic tool, not a validated
  predictor.
- No live retraining API; the FastAPI backend serves precomputed results.
- Contamination fraction for Isolation Forest is a guessed prototype
  hyperparameter (`src/config.py`), not tuned per lot.

## Future Improvements

- Validate against real ESS/burn-in data and recalibrate weights with
  domain experts / reliability engineers.
- Model temporal sequences with recurrent/attention architectures once
  more time points and more historical lots are available.
- Track per-lot process drift over multiple manufacturing runs, not just
  within a single lot snapshot.
- Add a live-scoring endpoint to the API, with input validation and
  authentication, for real integration into a test-floor workflow.
- Calibrate risk thresholds to precision/recall trade-offs meaningful to
  a specific reliability program (e.g. space-grade vs. commercial-grade).

---

## What makes this different from traditional screening?

Traditional screening only asks "is this value below the fixed limit?" —
a single global threshold that ignores how the rest of the lot behaved and
ignores how the component's own readings changed over time. This system
adds population-aware and temporal-aware checks on top, so components that
would silently pass a fixed-limit check can still be flagged before they
become field failures.

## What is the novelty?

The explicit distinction between **datasheet status** and **AI status**
— a component can be `PASS` / `ANOMALOUS` at the same time — combined
with a transparent, rule-based explanation for every flag (not just a
black-box score), and lot-wise (not global) statistical baselines that
respect legitimate manufacturing differences between lots.

## How does dynamic anomaly detection work?

Every component is compared only to its own lot × parameter peer group,
using five statistical/ML signals (z-score, robust z-score, IQR, percentile
rank, lot-wise Isolation Forest on the trajectory shape), combined into one
0–1 population anomaly score.

## How does temporal anomaly detection work?

Each component's 4-point burn-in trajectory is turned into interpretable
features (slope, % change, acceleration, peer-relative jump count), then
scored by a rule-based combination plus a lot-wise Isolation Forest on the
temporal-feature space.

## How is the final risk score calculated?

A configurable weighted sum of the population score, temporal score and
a specification-proximity score, mapped to NORMAL/WARNING/SUSPICIOUS/
CRITICAL via configurable thresholds. A hard specification violation
always forces at least a CRITICAL score.

## How does the system explain an anomaly?

Every flagged component gets a plain-language, rule-based explanation
built directly from the Module A/B signals (e.g. "25.5σ above the lot
median", "changed 291% during burn-in", "approaching the datasheet
limit"). Separately, a supervised proxy model + SHAP provides *global*
feature-importance diagnostics (which signals matter most across the
whole dataset) without pretending to explain the unsupervised detectors
directly.

---

## Project Structure

```
dynamic-anomaly-detection/
├── data/
│   ├── raw/            - provided synthetic dataset (CSV + XLSX)
│   ├── processed/       - cleaned + fully scored output of run.py
│   └── synthetic/       - provenance snapshot of the raw dataset
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── dynamic_detection.py     (Module A)
│   ├── temporal_detection.py    (Module B)
│   ├── scoring.py
│   ├── explainability.py
│   ├── evaluation.py
│   └── visualization.py
├── dashboard/
│   └── app.py            (Streamlit)
├── api/
│   └── main.py            (FastAPI, optional)
├── models/                (saved SHAP proxy model)
├── reports/
│   ├── figures/           (PNG plots)
│   └── evaluation_report.json
├── tests/
│   └── test_pipeline.py   (12 tests, pytest)
├── requirements.txt
├── README.md
└── run.py
```
