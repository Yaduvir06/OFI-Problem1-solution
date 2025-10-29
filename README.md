# OFI-Problem1-solution
Predictive Delivery Optimizer:  a tool that predicts delivery delays before they happen and suggests corrective actions.
# 🚚 NEXGEN — Delivery Delay Risk Dashboard

> End-to-end Streamlit dashboard for predicting and mitigating delivery delays.
> Demonstrates a production-aware ML workflow for small logistics datasets (~150 orders): data merge → cleaning → feature engineering → K-Fold CV with SMOTE → ensemble modeling (RandomForest + XGBoost) → hybrid rule fallbacks → actionable suggestions and exports.

---

## 🔎 Highlights (TL;DR)

* **Purpose:** Flag shipments at risk of delayed delivery and provide actionable corrective suggestions (reroute, carrier swap, hold for weather, escalate handling).
* **Modeling:** Stratified K-Fold CV with per-fold SMOTE; trains RF / GB / XGB; final ensemble uses RF + XGB with CV-weighted ensemble weights.
* **Decisioning:** Thresholds tuned on a 20% holdout using precision–recall. Sidebar offers threshold strategies (Default quantile / Max-F1 / Max-Accuracy / Custom).
* **UX:** Interactive filters, single-order lookup, Plotly visualizations, downloadable scored CSVs, saveable model artifacts.
* **Outcome (sample):** Demo data produced ~30% high-risk flags (≈45/150), **recall = 1.0** (no missed delays) and a practical savings proxy (illustrative).

---

## ✅ Features

* **Data pipeline**

  * Merge 4 CSVs (`orders.csv`, `delivery_performance.csv`, `routes_distance.csv`, `customer_feedback.csv`).
  * Median/mode imputation; preserves readable `_raw` columns for UI; flags suspicious rows (toggleable removal).

* **Feature engineering**

  * Creates derived features such as `carrier_risk`, `route_risk`, `priority_carrier_risk`, `traffic_severe`, `weather_severe`, `compound_risk`, `time_pressure`, `cost_per_km`, `traffic_distance_risk`, `carrier_toll_risk`, `priority_factor`, etc.

* **ML pipeline**

  * K-Fold CV (3 / 5 / 7) with SMOTE applied inside each training fold.
  * Models: RandomForest, GradientBoosting, XGBoost.
  * Ensemble: RF + XGB weighted by CV F1.
  * Holdout (20%) used for threshold tuning (precision–recall) and overfitting checks.

* **Hybrid decisioning**

  * Rule fallbacks (e.g., severe weather + high-risk carrier → immediate flag).
  * Per-order corrective actions: reroute, hold for weather, swap carrier, escalate handling, adjust promise.

* **UI & Viz**

  * Filters by Origin / Carrier / Priority; single-order lookup with suggested actions.
  * Plotly visuals: histograms, feature importances, boxplots, scatter, pie charts, confusion matrix.
  * Tables: Top high-risk orders, origin summaries.
  * CSV downloads and model artifact saves.

---

## 📦 Prerequisites

* Python 3.10+ (tested on 3.12)
* ~150 MB RAM for training (small dataset; training ≈ 30–90s first run on CPU depending on machine)
* CSV files: `orders.csv`, `delivery_performance.csv`, `routes_distance.csv`, `customer_feedback.csv` (place in project root)

Suggested `requirements.txt`:

```text
streamlit>=1.20
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
xgboost>=1.7
imbalanced-learn>=0.10.1
plotly>=5.8
joblib>=1.2
matplotlib>=3.6
seaborn>=0.12
```

---

## 🚀 Quick start

1. Clone / copy repository files and data into a folder, e.g.:

```bash
# create venv (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run delivery_dashboard_nexgen_ensemble.py
```

3. The app opens in your browser at `http://localhost:8501`.
   First run will merge data and train models (spinner). Artifacts are cached for faster reloads.

---

## 🗂️ Data schema (expected CSV columns)

* **orders.csv**

  * `Order_ID`, `Origin`, `Destination`, `Priority`, `Promised_Delivery_Days`, `Order_Value_INR`, `Customer_Segment`, `Product_Category`, `Special_Handling`

* **delivery_performance.csv**

  * `Order_ID`, `Actual_Delivery_Days`, `Weather_Impact`, `Delivery_Cost_INR`

* **routes_distance.csv**

  * `Order_ID`, `Distance_KM`, `Traffic_Delay_Minutes`, `Toll_Charges_INR`, `Fuel_Consumption_L`, `Route`, `Carrier`

* **customer_feedback.csv** (optional)

  * `Order_ID`, `feedback_score`, `comment`

Merged result is saved as `merged_delivery_data.csv`.

---

## ⚙️ UI quick guide

* **Sidebar**

  * Pipeline options: toggle suspicious-row removal, select CV folds (3/5/7), random seed, retrain button.
  * Threshold selection: Default quantile / Max F1 / Max Accuracy / Custom slider.
  * Single Order Lookup: enter `Order_ID` → see ensemble probability, prediction, reason (rule or ensemble), and suggested actions.

* **Main**

  * Top metrics: shown orders, predicted high-risk count, actual delays (labels), savings proxy.
  * CV summary: per-model F1/recall/precision/accuracy (mean ± std).
  * Holdout metrics: unbiased holdout evaluation using selected threshold.
  * Visuals: histogram (with threshold line), importances, boxplots, scatter, pie chart.
  * Tables: Top high-risk orders (with suggested corrective actions) and Origin risk summary.
  * Downloads: filtered CSV, full scored CSV, save artifacts.

---

## 📈 Interpreting metrics (short)

* **Accuracy** can be misleading on imbalanced data. Example: recall-focused thresholds raise recall at the expense of precision/accuracy.
* **Recall** = fraction of actual delayed orders caught. In logistics, missing delays is often costlier than extra manual checks → many deployments prefer higher recall.
* **Precision** = fraction of flagged orders that truly delay. High precision reduces wasted effort.
* Use the **threshold selector** to tune the operating point according to your business costs.

---

## ⚠️ Troubleshooting & tips

* **Missing files** — verify CSV filenames and place in project root (case sensitive).
* **SMOTE errors** — if minority class is extremely small, SMOTE may fail. Pipeline falls back to class weights or skips SMOTE per fold. Increase folds or add more minority samples.
* **Low accuracy
