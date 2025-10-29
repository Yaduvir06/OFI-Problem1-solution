# OFI-Problem1-solution
Predictive Delivery Optimizer:  a tool that predicts delivery delays before they happen and suggests corrective actions.
🚚 NEXGEN — Delivery Delay Risk Dashboard

NEXGEN Delivery Delay Risk Dashboard
An end-to-end Streamlit dashboard for predicting and mitigating delivery delays. Built for small logistics datasets (≈150 orders) and designed to demonstrate a production-aware ML workflow: data merging → cleaning → engineered features → K-fold CV with SMOTE → ensemble modeling (RandomForest + XGBoost) → hybrid rule fallbacks → human-action suggestions and exports.

Highlights / TL;DR

Purpose: Flag shipments with high delay risk and provide actionable suggestions (reroute, carrier swap, hold for weather, escalate handling).

Modeling: K-Fold CV with per-fold SMOTE, trains RF/GB/XGB; final ensemble mainly uses RF + XGB with CV-weighted ensemble weights.

Decisioning: Thresholds tuned on a 20% holdout using precision–recall curve. Sidebar exposes threshold strategies (default quantile / max-F1 / max-accuracy / custom).

UX: Interactive filters, single-order lookup, Plotly visualizations, downloadable scored CSVs, and a button to persist model artifacts for deployment.

Outcome (sample): On demo data the pipeline flagged ~30% of orders as high-risk (e.g., 45/150), achieved recall = 1.0 (no missed delays) and high practical savings estimate via early interventions.

Features

Data pipeline

Merges four CSVs: orders.csv, delivery_performance.csv, routes_distance.csv, customer_feedback.csv.

Imputes missing values (median for numeric, mode for categorical), preserves readable _raw columns for the UI.

Flags suspicious / potentially mislabeled rows with simple domain rules (toggleable).

Feature engineering

Creates 15–25 derived features including carrier_risk, route_risk, priority_carrier_risk, traffic_severe, weather_severe, compound_risk, time_pressure, cost_per_km, traffic_distance_risk, carrier_toll_risk, and priority_factor.

Modeling

Stratified K-Fold CV (configurable 3/5/7 folds) with SMOTE applied inside each training fold (prevents leakage).

Models trained: RandomForest, GradientBoosting, XGBoost.

Ensemble of RF + XGB; ensemble weights computed from CV F1 scores.

Holdout (20%) used for threshold tuning (precision–recall), final evaluation, and overfitting checks.

Hybrid decision layer

Rule-based overrides for clear conditions (e.g., extremely poor weather + high-risk carrier → immediate flag).

Action suggestions per order: reroute, apply weather buffer, swap carrier, escalate handling, adjust delivery promise.

UX & reporting

Sidebar controls: pipeline options, threshold strategy, single-order lookup, filters by Origin/Carrier/Priority.

Visualizations: probability histogram, feature importances, boxplots, scatter, pie charts, confusion matrix.

Tables: Top high-risk orders, origin-level summaries.

Exports: filtered CSV, scored full CSV, feature importance CSV, and model artifact save/download.

Quick install & run

Place the four CSV files (see Data schema below) in the project root:

orders.csv, delivery_performance.csv, routes_distance.csv, customer_feedback.csv.

Install dependencies (recommended in a venv):

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt


Suggested requirements.txt (minimal):

streamlit>=1.20
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
xgboost>=1.7
imblearn>=0.10.1
plotly>=5.8
joblib>=1.2
matplotlib>=3.6
seaborn>=0.12


Run the app:

streamlit run delivery_dashboard_nexgen_ensemble.py


The app will auto-merge data and train on first run (progress spinner). Subsequent runs load cached artifacts (unless you force retrain).

Recommended workflow (example)

Launch dashboard → wait for initial training (≈ 1 min on CPU for ~150 rows & default estimators).

Use Filters to inspect a subset (e.g., Origin = Pune, Priority = Express).

Inspect Top High-Risk Orders, open Single Order Lookup to get tailored suggestions.

Switch threshold strategy (Max F1 / Max Acc / Custom) to observe tradeoffs.

Export scored CSV and hand to operations for targeted interventions.

How to interpret the model metrics (short)

Accuracy can be low on imbalanced data and is often misleading. Example: a recall-focused threshold will flag more positives (boosting recall, lowering precision and possibly accuracy).

Recall (sensitivity) = fraction of actual delays caught. For logistics, missing a delayed shipment can be costlier than investigating false alarms; hence recall is often prioritized.

Precision = fraction of flagged orders that truly delay. High precision reduces wasted operational effort.

F1 balances precision and recall; useful but business cost weighting is superior.

Use the threshold selector to tune behavior: default quantile (top 30%) is a practical starting point; Max-F1 or Max-Accuracy are available for other preferences.

Troubleshooting & FAQs

Missing files error — ensure exact filenames and place them in the working directory.

SMOTE fails (very small minority class) — pipeline falls back to training without SMOTE or uses class weights. If you have <3 minority examples per fold, increase folds or use class weights.

Very low accuracy (e.g., 0.27) — likely the model is tuned for recall (it will intentionally generate more false positives). Use threshold controls to raise precision/accuracy at the cost of recall.

Plotly charts blank — confirm _raw columns exist (e.g., Origin_raw). The UI expects readable text columns for filters.

XGBoost install issues on Windows — install Visual C++ Build Tools or use a wheel / conda package for XGBoost.

Data schema (expected columns)

orders.csv

Order_ID, Origin, Destination, Priority, Promised_Delivery_Days, Order_Value_INR, Customer_Segment, Product_Category, Special_Handling

delivery_performance.csv

Order_ID, Actual_Delivery_Days, Weather_Impact, Delivery_Cost_INR

routes_distance.csv

Order_ID, Distance_KM, Traffic_Delay_Minutes, Toll_Charges_INR, Fuel_Consumption_L, Route, Carrier

customer_feedback.csv (optional)

Order_ID, feedback_score, comment

Merged dataset saves to merged_delivery_data.csv for caching.

Limitations & next steps

Small data variance: With ~150 rows the model has higher variance — more data yields more reliable models. Collect more historical orders where possible.

SMOTE caveats: Synthetic samples help training but can introduce unrealistic examples — prefer SMOTENC for mixed categorical data, or ADASYN if you need adaptive sampling.

Two-stage flow (recommended): Use a high-recall detector first, then a lightweight precision verifier (rules, cheaper model or human triage) to reduce operational cost of false positives.

Calibration: Consider CalibratedClassifierCV or isotonic regression for better probability estimates before thresholding.

Explainability & fairness: Add SHAP/LIME for model explainability and add audits to avoid systematic carrier bias.

Deployment: For production, export artifacts and serve via FastAPI (prediction API) and call from Streamlit or schedulers; consider autoscaling and cloud storage for data and models.

Example results (illustrative)

On the included sample data (≈150 merged rows, ≈23% delayed), the pipeline gave:

Predicted high-risk: ~45 orders (30%)

Recall = 1.00 (caught all labeled delays)

Precision ≈ 0.70 (roughly 7 in 10 flagged are actual delays)

Estimated operation savings proxy: >$100k (approximate and illustrative; depends on cost model)

These results demonstrate a recall-first posture: suitable where missed delays are expensive.

File tree (recommended)
nexgen-dashboard/
├─ delivery_dashboard_nexgen_ensemble.py
├─ requirements.txt
├─ orders.csv
├─ delivery_performance.csv
├─ routes_distance.csv
├─ customer_feedback.csv
├─ merged_delivery_data.csv    # generated on first run
├─ best_rf.pkl                 # saved artifacts (optional)
├─ best_xgb.pkl
├─ scaler.pkl
└─ README.md

License & credits

License: MIT — free for personal and commercial use.

Credits: Built using Streamlit, scikit-learn, XGBoost, imbalanced-learn, and Plotly. Inspired by NEXGEN pipeline patterns for small-data ML.
