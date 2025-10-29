# delivery_dashboard_nexgen_ensemble.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (precision_recall_curve, accuracy_score, f1_score,
                             precision_score, recall_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NEXGEN Delivery Delay Dashboard (Ensemble)", layout="wide", page_icon="🚚")

# -----------------------
# Utility & caching
# -----------------------
@st.cache_data
def load_csv_safe(path):
    return pd.read_csv(path)

@st.cache_data
def load_and_merge_data():
    """Load files and preserve raw categorical text columns for UI."""
    required_files = {
        'orders.csv': None,
        'delivery_performance.csv': None,
        'routes_distance.csv': None,
        'customer_feedback.csv': None
    }
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(f"Missing files: {', '.join(missing)} — place them in the working directory.")
        st.stop()

    orders = load_csv_safe('orders.csv')
    delivery = load_csv_safe('delivery_performance.csv')
    routes = load_csv_safe('routes_distance.csv')
    feedback = load_csv_safe('customer_feedback.csv')

    merged = orders.merge(delivery, on='Order_ID', how='inner')
    merged = merged.merge(routes, on='Order_ID', how='left')
    merged = merged.merge(feedback, on='Order_ID', how='left')
    merged.drop_duplicates(inplace=True)

    # Keep readable backups
    cat_cols = ['Origin', 'Destination', 'Customer_Segment', 'Priority', 'Product_Category',
                'Special_Handling', 'Carrier', 'Weather_Impact', 'Route']
    for c in cat_cols:
        if c in merged.columns:
            merged[f"{c}_raw"] = merged[c].astype(str).fillna('Unknown')
        else:
            merged[f"{c}_raw"] = 'Unknown'

    # Save cached merged data (for faster reloads)
    merged.to_csv('merged_delivery_data.csv', index=False)
    return merged

def flag_suspicious_rows(df):
    """Simple rules to flag noisy/mislabeled rows (tunable)."""
    df = df.copy()
    df['Traffic_Delay_Minutes'] = df.get('Traffic_Delay_Minutes', 0).fillna(0)
    df['Distance_KM'] = df.get('Distance_KM', 0).fillna(0)
    df['Weather_Impact_raw'] = df.get('Weather_Impact_raw', 'None').fillna('None')

    def is_suspicious_delayed(r):
        if r['is_delayed'] == 1:
            if (r['Traffic_Delay_Minutes'] < 30 and r['Distance_KM'] < 500 and r['Weather_Impact_raw'] in ['None', 'Clear', 'No']):
                return True
        return False

    def is_suspicious_ontime(r):
        if r['is_delayed'] == 0:
            if (r['Traffic_Delay_Minutes'] > 90 or r['Weather_Impact_raw'] in ['Heavy_Rain', 'Fog'] or r['Distance_KM'] > 2000):
                return True
        return False

    df['suspicious_delayed'] = df.apply(is_suspicious_delayed, axis=1)
    df['suspicious_ontime'] = df.apply(is_suspicious_ontime, axis=1)
    df['suspicious'] = df['suspicious_delayed'] | df['suspicious_ontime']
    return df

def suggest_actions(row, importances_df, thresh, raw_probs, route_risk_thresh=0.5, no_impact=['None', 'Clear', 'No', 'Unknown']):
    """Generate detailed corrective actions based on risk, top features, and domain rules."""
    prob = row['predicted_delay_prob']
    if prob < thresh:
        return "Low risk: Standard operations."
    rank = "High" if prob > np.quantile(raw_probs, 0.9) else "Medium"
    actions = [f"{rank} risk ({prob:.2f} prob): "]
    top_features = importances_df['feature'].head(5).tolist()
    
    if rank == "High":
        actions.append("Urgent intervention.")
        if row.get('Traffic_Delay_Minutes', 0) > 30:
            actions.append("Reroute immediately to QuickShip.")
        if row.get('weather_severe', 0) == 1 or row['Weather_Impact_raw'] not in no_impact:
            actions.append("Hold for weather: Rain/Fog buffer.")
        if row.get('Carrier_raw', 'Unknown') in ['GlobalTransit', 'ReliableExpress']:  # High-risk carriers from hybrid rules
            actions.append("Swap carrier to reliable option (e.g., QuickShip).")
        if row.get('Special_Handling_raw', 'None') != 'None':
            actions.append("Escalate handling for Fragile/Hazmat.")
    else:
        if row.get('Traffic_Delay_Minutes', 0) > 30:
            actions.append("Prepare reroute for traffic.")
        if row.get('weather_severe', 0) == 1 or row['Weather_Impact_raw'] not in no_impact:
            actions.append("Weather prep: Protective packaging.")
        if row.get('Carrier_raw', 'Unknown') in ['GlobalTransit', 'ReliableExpress']:
            actions.append("Monitor and consider carrier swap.")
        if row.get('Promised_Delivery_Days', 0) > 5:
            actions.append("Adjust promise timeline.")
    
    if 'Weather_Impact_raw' in top_features and row['Weather_Impact_raw'] not in no_impact:
        actions.append("Weather focus: Route avoidance.")
    if 'route_risk' in top_features and row['route_risk'] > route_risk_thresh:
        actions.append("Route high-risk: Alternative path.")
    if 'carrier_risk' in top_features and row['carrier_risk'] > 0.3:
        actions.append("Carrier optimization.")
    if 'delay_days' in top_features and abs(row['delay_days']) > 3:
        actions.append("Escalate for severe delay potential.")
    
    if len(actions) == 1:
        actions.append("Track closely and notify customer.")
    return ' | '.join(actions)

@st.cache_resource
def build_and_train_pipeline(df, remove_suspicious=True, cv_folds=5, random_state=42):
    """Core: clean, fe, CV with SMOTE per fold, model training, final ensemble, threshold tuning."""
    df = df.copy()

    # Basic imputation for numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_raw_cols = [c for c in df.columns if c.endswith('_raw')]

    # Ensure key numeric fields exist
    for expected in ['Promised_Delivery_Days', 'Actual_Delivery_Days', 'Distance_KM', 'Traffic_Delay_Minutes', 'Toll_Charges_INR', 'Delivery_Cost_INR', 'Order_Value_INR']:
        if expected not in df.columns:
            df[expected] = 0

    # Target definition
    df['delay_days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
    df['is_delayed'] = (df['delay_days'] > 1).astype(int)

    # Optional suspicious-outlier removal
    df = flag_suspicious_rows(df)
    if remove_suspicious:
        df_clean = df[~df['suspicious']].copy()
    else:
        df_clean = df.copy()

    # Advanced feature engineering (domain knowledge)
    # 1) carrier_risk, route_risk computed on df_clean
    if 'Carrier_raw' in df_clean.columns:
        carrier_delay_rate = df_clean.groupby('Carrier_raw')['is_delayed'].mean()
    else:
        carrier_delay_rate = pd.Series(dtype=float)
    df['carrier_risk'] = df['Carrier_raw'].map(carrier_delay_rate).fillna(df['is_delayed'].mean())

    if 'Route_raw' in df_clean.columns:
        route_delay_rate = df_clean.groupby('Route_raw')['is_delayed'].mean()
    else:
        route_delay_rate = pd.Series(dtype=float)
    df['route_risk'] = df['Route_raw'].map(route_delay_rate).fillna(df['is_delayed'].mean())

    # Priority-carrier interaction
    if 'Priority_raw' in df_clean.columns and 'Carrier_raw' in df_clean.columns:
        pc_delay = df_clean.groupby(['Priority_raw', 'Carrier_raw'])['is_delayed'].mean()
        df['priority_carrier_risk'] = df.apply(lambda r: pc_delay.get((r['Priority_raw'], r['Carrier_raw']), df['is_delayed'].mean()), axis=1)
    else:
        df['priority_carrier_risk'] = df['is_delayed'].mean()

    # severity flags
    df['traffic_severe'] = (df['Traffic_Delay_Minutes'] > 60).astype(int)
    df['weather_severe'] = df['Weather_Impact_raw'].isin(['Heavy_Rain', 'Fog']).astype(int)

    # long distance, time_pressure, cost_per_km
    dist_median = df['Distance_KM'].median() if 'Distance_KM' in df else 0
    df['long_distance'] = (df['Distance_KM'] > (dist_median * 1.5)).astype(int)
    df['compound_risk'] = df['traffic_severe']*2 + df['weather_severe']*2 + df['long_distance'] + (df['carrier_risk'] > df['carrier_risk'].median()).astype(int)*2
    df['time_pressure'] = df['Distance_KM'] / (df['Promised_Delivery_Days'] + 1)
    df['cost_per_km'] = df['Delivery_Cost_INR'] / (df['Distance_KM'] + 1)

    # Priority factor mapping
    priority_map = {'Express': 1.5, 'Standard': 1.0, 'Economy': 0.5}
    df['priority_factor'] = df['Priority_raw'].map(priority_map).fillna(1.0)

    # Build feature set
    feature_cols = [
        'Order_Value_INR', 'Promised_Delivery_Days', 'Distance_KM', 'Fuel_Consumption_L',
        'Toll_Charges_INR', 'Traffic_Delay_Minutes',
        'carrier_risk', 'route_risk', 'priority_carrier_risk',
        'traffic_severe', 'weather_severe', 'long_distance',
        'compound_risk', 'time_pressure', 'cost_per_km', 'priority_factor'
    ]

    # Add encoded categorical features (Origin, Destination, Priority, Carrier, Product_Category, Special_Handling, Weather_Impact)
    encode_cols = ['Origin_raw', 'Destination_raw', 'Priority_raw', 'Carrier_raw', 'Product_Category_raw', 'Special_Handling_raw', 'Weather_Impact_raw']
    label_encoders = {}
    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col.replace('_raw','')}_enc"] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
            label_encoders[col] = le
            feature_cols.append(f"{col.replace('_raw','')}_enc")

    # Prepare X, y
    X = df[feature_cols].copy()
    y = df['is_delayed'].copy()

    # Fill NA numeric
    X = X.fillna(0)

    # Scale numeric
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # ---------- K-Fold CV with SMOTE in each fold ----------
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_split=3,
                                               class_weight='balanced', random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=random_state),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                 scale_pos_weight=1, eval_metric='logloss', use_label_encoder=False, random_state=random_state)
    }

    cv_results = {name: {'f1': [], 'precision': [], 'recall': [], 'accuracy': []} for name in models}
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # SMOTE on training fold
        sm = SMOTE(random_state=random_state, k_neighbors=3)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

        for name, model in models.items():
            try:
                model.fit(X_tr_res, y_tr_res)
                y_pred = model.predict(X_te)
            except Exception:
                # fallback: skip model if fails
                continue
            cv_results[name]['f1'].append(f1_score(y_te, y_pred, zero_division=0))
            cv_results[name]['precision'].append(precision_score(y_te, y_pred, zero_division=0))
            cv_results[name]['recall'].append(recall_score(y_te, y_pred, zero_division=0))
            cv_results[name]['accuracy'].append((y_pred == y_te).mean())

    # Summarize CV
    cv_summary = {}
    for name in cv_results:
        cv_summary[name] = {metric: (np.mean(vals), np.std(vals)) for metric, vals in cv_results[name].items()}

    # Choose best by mean F1
    best_name = max(cv_summary, key=lambda n: cv_summary[n]['f1'][0])
    best_model = models[best_name]

    # Train final models (both RF and XGB for ensemble)
    # Apply SMOTE on full training to create balanced set
    sm = SMOTE(random_state=random_state, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X, y)

    final_models = {}
    # Retrain RF
    rf = RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_split=3,
                                class_weight='balanced', random_state=random_state)
    rf.fit(X_res, y_res)
    final_models['RandomForest'] = rf

    # Retrain GradientBoosting
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=random_state)
    gb.fit(X_res, y_res)
    final_models['GradientBoosting'] = gb

    # Retrain XGBoost (balance with scale_pos_weight)
    pos_ratio = max(1, (y_res==0).sum() / max(1, (y_res==1).sum()))
    xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=pos_ratio,
                        eval_metric='logloss', use_label_encoder=False, random_state=random_state)
    xgb.fit(X_res, y_res)
    final_models['XGBoost'] = xgb

    # Ensemble weights based on CV F1 (normalized)
    f1_scores = np.array([cv_summary[m]['f1'][0] for m in ['RandomForest', 'XGBoost']])
    # If any NaN, fallback to equal weighting
    if np.isnan(f1_scores).any() or f1_scores.sum() == 0:
        weights = {'RandomForest': 0.5, 'XGBoost': 0.5}
    else:
        norm = f1_scores.sum()
        weights = {'RandomForest': float(f1_scores[0]/norm), 'XGBoost': float(f1_scores[1]/norm)}

    # Generate full predictions (probabilities) from ensemble on ALL data
    # Ensure predict_proba availability
    rf_probs = final_models['RandomForest'].predict_proba(X)[:,1] if hasattr(final_models['RandomForest'], 'predict_proba') else np.zeros(len(X))
    xgb_probs = final_models['XGBoost'].predict_proba(X)[:,1] if hasattr(final_models['XGBoost'], 'predict_proba') else np.zeros(len(X))
    ensemble_prob = weights['RandomForest'] * rf_probs + weights['XGBoost'] * xgb_probs

    df['predicted_delay_prob'] = ensemble_prob
    # Default threshold: top 30% as high risk (can tune later)
    default_quantile = 0.70
    default_threshold = float(np.quantile(ensemble_prob, default_quantile))
    df['predicted_delay'] = (df['predicted_delay_prob'] > default_threshold).astype(int)

    # Importances for actions (from XGB)
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_models['XGBoost'].feature_importances_
    }).sort_values('importance', ascending=False)

    # Pre-compute route_risk threshold for actions (fixes quantile error)
    route_risk_thresh = df['route_risk'].quantile(0.75) if 'route_risk' in df.columns and not df['route_risk'].isna().all() else 0.5

    # Compute corrective_actions for all rows
    df['corrective_actions'] = df.apply(lambda row: suggest_actions(row, importances, default_threshold, ensemble_prob, route_risk_thresh=route_risk_thresh), axis=1)

    # Threshold tuning on a hold-out test split (to present options)
    X_train_hold, X_test_hold, y_train_hold, y_test_hold = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
    # get ensemble probs for test hold
    rf_test_probs = final_models['RandomForest'].predict_proba(X_test_hold)[:,1]
    xgb_test_probs = final_models['XGBoost'].predict_proba(X_test_hold)[:,1]
    ensemble_test_prob = weights['RandomForest'] * rf_test_probs + weights['XGBoost'] * xgb_test_probs

    # Precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test_hold, ensemble_test_prob)
    # Evaluate candidate thresholds: maximize F1 and maximize accuracy
    candidate_thresholds = np.unique(np.clip(pr_thresholds, 0.0, 1.0))
    best_f1 = -1; best_f1_thresh = default_threshold
    best_acc = -1; best_acc_thresh = default_threshold
    for t in candidate_thresholds:
        preds = (ensemble_test_prob >= t).astype(int)
        f1v = f1_score(y_test_hold, preds, zero_division=0)
        accv = accuracy_score(y_test_hold, preds)
        if f1v > best_f1:
            best_f1 = f1v; best_f1_thresh = float(t)
        if accv > best_acc:
            best_acc = accv; best_acc_thresh = float(t)

    # collect model artifacts to return
    artifacts = {
        'df': df,
        'final_models': final_models,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'weights': weights,
        'cv_summary': cv_summary,
        'default_threshold': default_threshold,
        'best_f1_thresh': best_f1_thresh,
        'best_acc_thresh': best_acc_thresh,
        'holdout_y_test': y_test_hold,
        'holdout_probs': ensemble_test_prob,
        'rf_probs_all': rf_probs,
        'xgb_probs_all': xgb_probs,
        'importances': importances  # For actions
    }

    # Save artifacts to disk for reuse
    joblib.dump(final_models['RandomForest'], 'best_rf.pkl')
    joblib.dump(final_models['XGBoost'], 'best_xgb.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    return artifacts

# -----------------------
# Hybrid rule-based
# -----------------------
def hybrid_predict_single(row, artifacts):
    """Return prediction (0/1), probability, and reason using hybrid rules + ensemble."""
    # Hard-coded business rules (example)
    carrier = row.get('Carrier_raw', row.get('Carrier', 'Unknown'))
    traffic = row.get('Traffic_Delay_Minutes', 0)
    weather = row.get('Weather_Impact_raw', 'None')
    dist = row.get('Distance_KM', 0)

    # Rule 1: extremely bad conditions + known high-risk carrier => flag
    if carrier in ['GlobalTransit', 'ReliableExpress'] and (traffic > 120 or weather in ['Heavy_Rain', 'Fog']):
        return 1, 0.98, "Rule: High-risk carrier + severe conditions"

    # Rule 2: very safe shipment => on-time
    if carrier == 'QuickShip' and traffic < 15 and weather in ['None', 'Clear'] and dist < 200:
        return 0, 0.05, "Rule: Low-risk carrier + low traffic"

    # Otherwise use ensemble
    # Build feature vector consistent with artifacts['feature_cols']
    feat_cols = artifacts['feature_cols']
    row_vals = []
    for c in feat_cols:
        if c.endswith('_enc'):
            base = c.replace('_enc','')
            raw_col = base + '_raw'
            # If label encoder exists, use it; else map unknown -> 0
            le_key = raw_col
            if le_key in artifacts['label_encoders']:
                le = artifacts['label_encoders'][le_key]
                val = row.get(raw_col, 'Unknown')
                try:
                    idx = int(le.transform([str(val)])[0])
                except Exception:
                    # unseen label -> assign code 0
                    idx = 0
                row_vals.append(idx)
            else:
                row_vals.append(0)
        else:
            row_vals.append(row.get(c, 0))

    X_row = np.array(row_vals).reshape(1, -1)
    # scale numeric using scaler (note: artifacts['scaler'] fit on full X where numeric columns are many)
    try:
        X_row_scaled = artifacts['scaler'].transform(X_row)
    except Exception:
        X_row_scaled = X_row

    rf = artifacts['final_models']['RandomForest']
    xgb = artifacts['final_models']['XGBoost']
    rf_p = rf.predict_proba(X_row_scaled)[0][1] if hasattr(rf, 'predict_proba') else 0.0
    xgb_p = xgb.predict_proba(X_row_scaled)[0][1] if hasattr(xgb, 'predict_proba') else 0.0
    prob = artifacts['weights']['RandomForest'] * rf_p + artifacts['weights']['XGBoost'] * xgb_p
    # default threshold
    thresh = artifacts['default_threshold']
    pred = int(prob > thresh)
    return pred, float(prob), "Ensemble"

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.title("🚚 NEXGEN Delivery Delay Risk Dashboard — Ensemble (RF + XGB)")
    st.markdown("Advanced pipeline: K-Fold CV with SMOTE, engineered features, ensemble, hybrid rules, and action suggestions.")

    # Load or create merged data
    if os.path.exists('merged_delivery_data.csv'):
        df = pd.read_csv('merged_delivery_data.csv')
        # ensure raw columns exist
        raw_cols = ['Origin_raw','Destination_raw','Priority_raw','Carrier_raw','Product_Category_raw','Special_Handling_raw','Weather_Impact_raw','Route_raw']
        for c in raw_cols:
            if c not in df.columns:
                df[c] = df.get(c.replace('_raw',''), 'Unknown').astype(str)
        st.info("Loaded cached merged data.")
    else:
        with st.spinner("Merging source CSVs..."):
            df = load_and_merge_data()

    # Sidebar controls
    st.sidebar.header("Pipeline options")
    remove_susp = st.sidebar.checkbox("Remove suspicious/noisy samples (recommended)", value=True)
    cv_folds = st.sidebar.selectbox("K-Folds for CV", options=[3,5,7], index=1)
    random_state = st.sidebar.number_input("Random seed", value=42, step=1)
    run_training = st.sidebar.button("(Re)Train models with NEXGEN pipeline")

    # Auto-train if artifacts missing or button pressed
    # Try to load artifacts from disk to speed up
    artifacts_exist = os.path.exists('best_rf.pkl') and os.path.exists('best_xgb.pkl') and os.path.exists('scaler.pkl')
    artifacts = None
    if artifacts_exist and not run_training:
        with st.spinner("Loading models from disk..."):
            try:
                # We still need df for predicted probs; load artifacts via build_and_train_pipeline which caches
                artifacts = build_and_train_pipeline(df, remove_suspicious=remove_susp, cv_folds=cv_folds, random_state=int(random_state))
                st.success("Loaded pipeline artifacts (cached).")
            except Exception as e:
                st.warning(f"Failed to load cached artifacts: {e}")
                artifacts = None

    if artifacts is None:
        with st.spinner("Running NEXGEN training & CV (this may take a minute)..."):
            artifacts = build_and_train_pipeline(df, remove_suspicious=remove_susp, cv_folds=cv_folds, random_state=int(random_state))
            st.success("Training complete and artifacts generated.")

    # Unpack
    model_df = artifacts['df']
    final_models = artifacts['final_models']
    default_threshold = artifacts['default_threshold']
    best_f1_thresh = artifacts['best_f1_thresh']
    best_acc_thresh = artifacts['best_acc_thresh']
    cv_summary = artifacts['cv_summary']
    weights = artifacts['weights']
    importances = artifacts['importances']  # For actions

    # Sidebar: threshold strategy
    st.sidebar.header("Threshold & Scoring")
    thresh_option = st.sidebar.radio("Threshold strategy", options=[
        f"Default quantile (top 30%) - {default_threshold:.3f}",
        f"Maximize F1 on holdout - {best_f1_thresh:.3f}",
        f"Maximize Accuracy on holdout - {best_acc_thresh:.3f}",
        "Custom threshold"
    ], index=0)
    if "Custom" in thresh_option:
        custom_thresh = st.sidebar.slider("Custom threshold", 0.0, 1.0, float(default_threshold), 0.01)
        active_threshold = float(custom_thresh)
    else:
        if "F1" in thresh_option:
            active_threshold = float(best_f1_thresh)
        elif "Accuracy" in thresh_option:
            active_threshold = float(best_acc_thresh)
        else:
            active_threshold = float(default_threshold)

    st.sidebar.markdown(f"**Active threshold:** {active_threshold:.3f}")

    # Single Order lookup
    st.sidebar.header("Single Order Lookup")
    order_id = st.sidebar.text_input("Enter Order_ID (exact)")
    if order_id:
        row = model_df[model_df['Order_ID'].astype(str) == str(order_id)]
        if row.empty:
            st.sidebar.error("Order ID not found.")
        else:
            r = row.iloc[0].to_dict()
            pred_rule, prob_rule, reason = hybrid_predict_single(r, artifacts)
            risk_flag = "High" if prob_rule >= active_threshold else "Low"
            st.sidebar.metric("Ensemble Prob", f"{prob_rule:.3f}", f"Risk: {risk_flag}")
            st.sidebar.markdown(f"**Predicted Delay (binary):** {int(pred_rule)}  \n**Reason:** {reason}")
            # Enhanced actions: Use pre-computed or generate full suggestion
            route_risk_thresh = model_df['route_risk'].quantile(0.75) if 'route_risk' in model_df else 0.5
            actions_text = r.get('corrective_actions', suggest_actions(r, importances, active_threshold, model_df['predicted_delay_prob'], route_risk_thresh=route_risk_thresh))
            st.sidebar.text_area("Suggested Actions", actions_text, height=120, key="actions_single")
            st.sidebar.caption(f"Origin: {r.get('Origin_raw','N/A')} | Carrier: {r.get('Carrier_raw','N/A')} | Priority: {r.get('Priority_raw','N/A')} | Weather: {r.get('Weather_Impact_raw','N/A')}")

    # Filters (use readable raw columns)
    st.sidebar.header("Filters")
    origins = sorted(model_df['Origin_raw'].dropna().unique())
    carriers = sorted(model_df['Carrier_raw'].dropna().unique())
    priorities = sorted(model_df['Priority_raw'].dropna().unique())
    selected_origin = st.sidebar.multiselect("Origins", options=origins, default=origins[:10] if len(origins)>10 else origins)
    selected_carrier = st.sidebar.multiselect("Carriers", options=carriers, default=carriers[:8] if len(carriers)>8 else carriers)
    selected_priority = st.sidebar.multiselect("Priorities", options=priorities, default=priorities)

    # Filter data
    mask = model_df['Origin_raw'].isin(selected_origin) & model_df['Carrier_raw'].isin(selected_carrier) & model_df['Priority_raw'].isin(selected_priority)
    filtered = model_df[mask].copy()
    filtered['is_high_risk'] = filtered['predicted_delay_prob'] >= active_threshold

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orders Shown", f"{len(filtered):,}")
    col2.metric("High Risk (pred)", f"{filtered['is_high_risk'].sum():,}")
    col3.metric("Actual Delays (label)", f"{int(filtered['is_delayed'].sum()):,}")
    potential_savings = filtered[filtered['is_high_risk']]['priority_factor'].multiply(filtered[filtered['is_high_risk']]['delay_days'].abs()).sum() * 100  # simple proxy
    col4.metric("Potential Savings (proxy)", f"${potential_savings:,.0f}")

    # Explain CV summary
    with st.expander("Model CV Summary (mean ± std)"):
        for m, s in cv_summary.items():
            st.write(f"**{m}** — F1: {s['f1'][0]:.3f} ± {s['f1'][1]:.3f} | Recall: {s['recall'][0]:.3f} ± {s['recall'][1]:.3f} | Precision: {s['precision'][0]:.3f} ± {s['precision'][1]:.3f}")

    # Accuracy/metrics on holdout shown
    # Recalculate holdout preds at active_threshold for display
    y_hold = artifacts['holdout_y_test']
    probs_hold = artifacts['holdout_probs']
    preds_hold = (probs_hold >= active_threshold).astype(int)
    if len(y_hold) > 0:
        acc_hold = accuracy_score(y_hold, preds_hold)
        prec_hold = precision_score(y_hold, preds_hold, zero_division=0)
        rec_hold = recall_score(y_hold, preds_hold, zero_division=0)
        f1_hold = f1_score(y_hold, preds_hold, zero_division=0)
    else:
        acc_hold = prec_hold = rec_hold = f1_hold = 0.0

    st.markdown("### Holdout metrics (using chosen threshold)")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Accuracy", f"{acc_hold:.2f}")
    mcol2.metric("Precision (delays)", f"{prec_hold:.2f}")
    mcol3.metric("Recall (delays)", f"{rec_hold:.2f}")
    mcol4.metric("F1 (delays)", f"{f1_hold:.2f}")

    # Visualizations
    st.header("Visualizations & Insights")

    # 1) Probability distribution histogram
    fig_hist = px.histogram(filtered, x='predicted_delay_prob', color='is_delayed', nbins=25,
                            title="Ensemble Predicted Delay Probability Distribution",
                            labels={'predicted_delay_prob': 'Predicted Delay Probability', 'is_delayed': 'Actual Delay'})
    fig_hist.add_vline(x=active_threshold, line_dash='dash', line_color='red', annotation_text=f"Threshold {active_threshold:.3f}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2) Feature importances (from XGB model)
    if hasattr(final_models['XGBoost'], 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': artifacts['feature_cols'],
            'importance': final_models['XGBoost'].feature_importances_
        }).sort_values('importance', ascending=False).head(12)
        fig_imp = px.bar(imp, x='importance', y='feature', orientation='h', title="Top XGBoost Feature Importances")
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    # 3) Delay days by Origin box
    if 'delay_days' in filtered.columns:
        fig_box = px.box(filtered, x='Origin_raw', y='delay_days', color='is_high_risk',
                         title="Delay Days by Origin (filtered)", labels={'Origin_raw':'Origin', 'delay_days':'Delay (days)'})
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

    # 4) Scatter: route_risk vs predicted prob
    fig_scatter = px.scatter(filtered, x='route_risk', y='predicted_delay_prob', size='Traffic_Delay_Minutes',
                             color='Carrier_raw', hover_data=['Order_ID', 'Origin_raw'], title="Route Risk vs Predicted Delay Prob")
    fig_scatter.add_hline(y=active_threshold, line_dash='dash', line_color='red')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 5) Pie: high risk by priority
    hr = filtered[filtered['is_high_risk']]
    if len(hr) > 0:
        pie = hr['Priority_raw'].value_counts()
        fig_pie = px.pie(values=pie.values, names=pie.index, title="High-Risk Orders by Priority")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No high-risk orders in current filter to show pie chart.")

    # 6) Confusion matrix for holdout
    if len(y_hold) > 0:
        cm = confusion_matrix(y_hold, preds_hold)
        cm_df = pd.DataFrame(cm, index=['NoDelay','Delay'], columns=['PredNoDelay','PredDelay'])
        st.subheader("Holdout Confusion Matrix")
        st.dataframe(cm_df)

    # Tables: top high-risk orders
    st.header("Top High-Risk Orders (filtered)")
    hr_table = filtered[filtered['is_high_risk']].sort_values('predicted_delay_prob', ascending=False)
    display_cols = ['Order_ID','Origin_raw','Destination_raw','Priority_raw','Carrier_raw','predicted_delay_prob','delay_days','compound_risk','corrective_actions']
    st.dataframe(hr_table[display_cols].head(50), use_container_width=True)

    # Origin risk summary
    st.subheader("Origin-level Risk Summary (filtered)")
    summary = filtered.groupby('Origin_raw').agg({
        'predicted_delay_prob':'mean',
        'is_high_risk':'mean',
        'delay_days':'mean',
        'compound_risk':'mean'
    }).round(3).sort_values('predicted_delay_prob', ascending=False)
    st.dataframe(summary, use_container_width=True)

    # Downloads
    st.header("Downloads & Exports")
    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf, index=False)
    st.download_button("Download filtered CSV", csv_buf.getvalue(), "filtered_delivery_risk.csv", "text/csv")
    full_buf = io.StringIO()
    model_df.to_csv(full_buf, index=False)
    st.download_button("Download full scored CSV", full_buf.getvalue(), "full_delivery_risk_scored.csv", "text/csv")

    # Save models again on demand
    if st.button("Save model artifacts to disk"):
        joblib.dump(final_models['RandomForest'], 'best_rf.pkl')
        joblib.dump(final_models['XGBoost'], 'best_xgb.pkl')
        joblib.dump(artifacts['scaler'], 'scaler.pkl')
        joblib.dump(artifacts['label_encoders'], 'label_encoders.pkl')
        st.success("Artifacts saved to disk.")

    st.markdown("---")
    st.caption("NEXGEN: K-Fold CV + SMOTE per fold | Features engineered from domain knowledge | Ensemble of RF + XGB | Hybrid rule fallback | Detailed corrective actions")

if __name__ == "__main__":
    main()
