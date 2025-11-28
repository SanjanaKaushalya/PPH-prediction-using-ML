"""
run_experiments_full.py

Full experiment script with many PNG plots:
- Preprocess dataset (auto-detect columns)
- For sizes = [300, 600, 1000, 1500, 2000, 2500, 3000, 3500]:
    - sample n rows
    - 80/20 stratified split
    - train RandomForest (class_weight='balanced')
    - train XGBoost (scale_pos_weight = neg/pos)
    - evaluate (accuracy, precision, recall, f1, roc_auc, pr_auc)
    - save models
    - save plots per model/size:
        - ROC curve (png)
        - Precision-Recall curve (png)
        - Confusion matrix (png)
        - Learning curve (png)
        - Feature importance (png)
- Save results_summary.csv and produce comparison plots (accuracy vs sample size)
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# try to import xgboost
try:
    import xgboost as xgb
except Exception as e:
    raise ImportError("xgboost is required. Install with: pip install xgboost") from e

# -----------------------
# Config
# -----------------------
CSV_PATH = Path("pph_sri_lanka_synthetic.csv")   # update if needed
OUT_DIR = Path("models")
PLOTS_DIR = Path("plots")
RESULTS_CSV = Path("results_summary.csv")
RANDOM_STATE = 42

# sizes (including the larger ones you requested)
sizes = [300, 600, 1000, 1500, 2000, 2500, 3000, 3500]

# create folders
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Load dataset
# -----------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found at: {CSV_PATH}. Put your data file there or change CSV_PATH.")

df = pd.read_csv(CSV_PATH)
print("Dataset loaded:", df.shape)
print("Columns:", list(df.columns))
print(df.head().iloc[:5, :].to_string(index=False))

# -----------------------
# Column mapping (auto-detect common names)
# -----------------------
col_candidates = {
    'pre_preg_weight': ['prepreg_weight_kg','prepreg_weight','Prepreg_weight_kg','Prepreg_weight'],
    'height': ['height_cm','Height_cm','height'],
    'bmi': ['prepreg_bmi','Prepreg_BMI','pre_preg_bmi','Prepreg BMI','BMI'],
    'prior_pph': ['history_of_pph','History_of_PPH','History_of_PPH','History_of_PPH'],
    'hb': ['hb_note_at_booking','Hb_note_at_booking','hb_booking','Hb_booking','Hb'],
    'gdm': ['history_of_gdm_or_t2dm','History_of_GDM_or_T2DM','History_of_GDM','GDM_history','History_of_GDM_or_T2DM'],
    'outcome': ['pph_outcome','PPH_outcome','PPH','outcome','PPH outcome']
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

mapped = {k: find_col(df, v) for k, v in col_candidates.items()}
print("\nDetected columns mapping:")
for k, v in mapped.items():
    print(f"  {k:<12} -> {v}")

# check required columns exist
required = ['pre_preg_weight', 'bmi', 'prior_pph', 'hb', 'gdm', 'outcome']
missing = [r for r in required if mapped.get(r) is None]
if missing:
    raise ValueError(f"Missing required columns in CSV (couldn't auto-detect): {missing}. Update CSV headers or mapping.")

# rename to standard short names
rename_map = {
    mapped['pre_preg_weight']: 'pre_preg_weight',
    mapped['bmi']: 'pre_preg_bmi',
    mapped['prior_pph']: 'prior_pph',
    mapped['hb']: 'hb_booking',
    mapped['gdm']: 'gest_diabetes',
    mapped['outcome']: 'PPH'
}
# height optional
if mapped.get('height'):
    rename_map[mapped['height']] = 'height_cm'

df = df.rename(columns=rename_map)

# -----------------------
# Convert categorical yes/no to 0/1 (robust)
# -----------------------
def map_yesno_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.astype(str).str.strip().str.lower().map({'yes':1,'y':1,'true':1,'1':1,'no':0,'n':0,'false':0,'0':0})

df['prior_pph'] = map_yesno_series(df['prior_pph']).fillna(0).astype(int)
df['gest_diabetes'] = map_yesno_series(df['gest_diabetes']).fillna(0).astype(int)
df['PPH'] = map_yesno_series(df['PPH'])

# If target missing rows exist, drop them (can't train without labels)
if df['PPH'].isna().any():
    n_before = len(df)
    df = df.dropna(subset=['PPH'])
    print(f"Dropped {n_before - len(df)} rows with missing target (PPH).")

df['PPH'] = df['PPH'].fillna(0).astype(int)

# ensure Hb numeric
if not pd.api.types.is_numeric_dtype(df['hb_booking']):
    df['hb_booking'] = pd.to_numeric(df['hb_booking'], errors='coerce')

# fill numeric missing with median (safe for tree models)
df = df.fillna(df.median(numeric_only=True))

# create anaemia_flag feature
df['anaemia_flag'] = (df['hb_booking'] < 11).astype(int)

# final feature list (only existing features)
features = ['pre_preg_weight', 'pre_preg_bmi', 'prior_pph', 'hb_booking', 'anaemia_flag', 'gest_diabetes']
# if height exists and you want to include, add it; current design uses BMI directly
if 'height_cm' in df.columns and 'pre_preg_bmi' not in df.columns:
    features.insert(1, 'height_cm')

print("\nUsing features:", features)
print("Target: PPH")

# check final datatypes
print("\nFeature dtypes:")
print(df[features].dtypes)
print("Target distribution:\n", df['PPH'].value_counts())

# -----------------------
# Helper functions: evaluation and plotting data storage
# -----------------------
def safe_predict_proba(model, X):
    """Return probability for positive class. Handle models that might not implement predict_proba."""
    try:
        probs = model.predict_proba(X)
        if probs.ndim == 1:
            return probs
        if probs.shape[1] == 1:
            return probs.ravel()
        return probs[:, 1]
    except Exception:
        try:
            scores = model.decision_function(X)
            probs = 1 / (1 + np.exp(-scores))
            return probs
        except Exception:
            preds = model.predict(X)
            return preds.astype(float)

def compute_metrics(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    probs = safe_predict_proba(model, X)
    roc = np.nan
    pr = np.nan
    try:
        if len(np.unique(y)) > 1:
            roc = roc_auc_score(y, probs)
    except Exception:
        roc = np.nan
    try:
        if len(np.unique(y)) > 1:
            pr = average_precision_score(y, probs)
    except Exception:
        pr = np.nan
    return acc, roc, pr

def plot_roc(y_true, probs, outpath, title=None):
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.figure(figsize=(6.5,4))
        plt.plot(fpr, tpr, marker='.', label=f"AUC={roc_auc_score(y_true, probs):.4f}")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title or "ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    except Exception as e:
        print("Failed to create ROC plot:", e)

def plot_pr(y_true, probs, outpath, title=None):
    try:
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs) if len(np.unique(y_true))>1 else np.nan
        plt.figure(figsize=(6.5,4))
        plt.plot(recall, precision, marker='.', label=f"AP={ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title or "Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    except Exception as e:
        print("Failed to create PR plot:", e)

def plot_confusion(y_true, y_pred, outpath, title=None):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title or "Confusion Matrix")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    except Exception as e:
        print("Failed to create Confusion Matrix plot:", e)

def plot_learning_curve(estimator, X, y, outpath, title=None):
    try:
        # use 5-fold CV if possible; if dataset small this may warn
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, train_sizes=np.linspace(0.1,1.0,5), scoring='accuracy', n_jobs=-1, random_state=RANDOM_STATE
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.figure(figsize=(6.5,4))
        plt.plot(train_sizes, train_mean, marker='o', label='Train score')
        plt.plot(train_sizes, test_mean, marker='o', label='CV score')
        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        plt.title(title or "Learning Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    except Exception as e:
        print("Failed to create learning curve plot:", e)

def plot_feature_importance(model, feature_names, outpath, title=None):
    try:
        # Try sklearn-style feature_importances_ first
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            inds = np.argsort(imp)[::-1]
            vals = imp[inds]
            names = [feature_names[i] for i in inds]
        else:
            # Try XGBoost booster get_score fallback
            try:
                booster = model.get_booster()
                score_dict = booster.get_score(importance_type='weight')
                # map scores to feature order
                vals = []
                names = []
                for f in feature_names:
                    key = f"f{feature_names.index(f)}"
                    v = score_dict.get(key, 0.0)
                    vals.append(v)
                    names.append(f)
                # sort descending
                inds = np.argsort(vals)[::-1]
                vals = np.array(vals)[inds]
                names = [names[i] for i in inds]
            except Exception:
                print("No feature importance available for this model.")
                return
        plt.figure(figsize=(6.5,4))
        sns.barplot(x=vals, y=names)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(title or "Feature Importance")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    except Exception as e:
        print("Failed to create feature importance plot:", e)

# collect results
results = []

# For combined ROC/PR plotting, keep per-model/per-size data (already used for per-size plots)
roc_data = {'rf':{}, 'xgb':{}}
pr_data = {'rf':{}, 'xgb':{}}

# Overfitting thresholds (tune as needed). These are absolute differences (train - test).
THRESHOLDS = {'accuracy': 0.05, 'roc_auc': 0.05, 'pr_auc': 0.05}

# -----------------------
# Main loop — experiments
# -----------------------
for size in sizes:
    print("\n" + "="*60)
    print(f"Running experiment with n = {size}")
    print("="*60)

    # sample with fixed random_state for reproducibility
    if size >= len(df):
        df_small = df.copy()
        print("Requested size >= dataset size; using full dataset.")
    else:
        df_small = df.sample(n=size, random_state=RANDOM_STATE)

    X_small = df_small[features]
    y_small = df_small['PPH'].astype(int)

    # ensure both classes present; if not, warn and skip
    if y_small.nunique() < 2:
        print(f"Warning: sample of size {size} contains only one class. Skipping.")
        continue

    # train/test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, stratify=y_small, random_state=RANDOM_STATE
    )

    print("Train samples:", len(X_train), "Test samples:", len(X_test))
    print("Train class distribution:\n", y_train.value_counts().to_dict())
    print("Test class distribution:\n", y_test.value_counts().to_dict())

    # -----------------------
    # Random Forest
    # -----------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # compute train metrics for overfitting check
    rf_train_acc, rf_train_roc, rf_train_pr = compute_metrics(rf, X_train, y_train)

    rf_preds = rf.predict(X_test)
    rf_probs = safe_predict_proba(rf, X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_roc = roc_auc_score(y_test, rf_probs) if len(np.unique(y_test))>1 else np.nan
    rf_pr = average_precision_score(y_test, rf_probs) if len(np.unique(y_test))>1 else np.nan

    print("\nRandom Forest results (n={}):".format(size))
    print(f"  Train Accuracy: {rf_train_acc*100:.2f}% | Test Accuracy: {rf_acc*100:.2f}%")
    if not np.isnan(rf_train_roc) and not np.isnan(rf_roc):
        print(f"  Train ROC AUC:  {rf_train_roc:.4f} | Test ROC AUC:  {rf_roc:.4f}")
    else:
        print(f"  Train ROC AUC:  {rf_train_roc} | Test ROC AUC:  {rf_roc}")
    if not np.isnan(rf_train_pr) and not np.isnan(rf_pr):
        print(f"  Train PR AUC:   {rf_train_pr:.4f} | Test PR AUC:   {rf_pr:.4f}")
    else:
        print(f"  Train PR AUC:   {rf_train_pr} | Test PR AUC:   {rf_pr}")

    print("\nClassification Report (RF):")
    print(classification_report(y_test, rf_preds, digits=4))

    # Overfitting decision (RF)
    overfit_reasons = []
    if rf_train_acc - rf_acc > THRESHOLDS['accuracy']:
        overfit_reasons.append(f"accuracy (train {rf_train_acc:.3f} vs test {rf_acc:.3f})")
    if (not np.isnan(rf_train_roc)) and (not np.isnan(rf_roc)) and (rf_train_roc - rf_roc > THRESHOLDS['roc_auc']):
        overfit_reasons.append(f"roc_auc (train {rf_train_roc:.3f} vs test {rf_roc:.3f})")
    if (not np.isnan(rf_train_pr)) and (not np.isnan(rf_pr)) and (rf_train_pr - rf_pr > THRESHOLDS['pr_auc']):
        overfit_reasons.append(f"pr_auc (train {rf_train_pr:.3f} vs test {rf_pr:.3f})")

    if overfit_reasons:
        print(">> RF Overfitting detected. Reasons:", "; ".join(overfit_reasons))
    else:
        print(">> RF No strong overfitting detected (within thresholds).")

    # save metric row
    results.append({
        'model':'RandomForest',
        'n':size,
        'accuracy': rf_acc,
        'roc_auc': rf_roc,
        'pr_auc': rf_pr,
        'n_train': len(X_train),
        'n_test': len(X_test)
    })

    # store ROC/PR curve points
    try:
        roc_data['rf'][size] = (roc_curve(y_test, rf_probs)[0], roc_curve(y_test, rf_probs)[1])
    except Exception:
        roc_data['rf'][size] = (np.array([]), np.array([]))
    try:
        pr_data['rf'][size] = (precision_recall_curve(y_test, rf_probs)[0], precision_recall_curve(y_test, rf_probs)[1])
    except Exception:
        pr_data['rf'][size] = (np.array([]), np.array([]))

    # save model
    rf_out = OUT_DIR / f"rf_{size}.joblib"
    joblib.dump(rf, rf_out)
    print("Saved RF model to:", rf_out)

    # -----------------------
    # Plots for RF (per-size)
    # -----------------------
    # ROC
    plot_roc(y_test, rf_probs, PLOTS_DIR / f"roc_rf_n{size}.png", title=f"RF ROC (n={size})")
    # PR
    plot_pr(y_test, rf_probs, PLOTS_DIR / f"pr_rf_n{size}.png", title=f"RF PR (n={size})")
    # Confusion matrix
    plot_confusion(y_test, rf_preds, PLOTS_DIR / f"cm_rf_n{size}.png", title=f"RF Confusion Matrix (n={size})")
    # Learning curve (use the fitted estimator object for curve shape)
    try:
        plot_learning_curve(RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
                            X_small, y_small, PLOTS_DIR / f"learning_rf_n{size}.png", title=f"RF Learning Curve (n={size})")
    except Exception as e:
        print("RF learning curve skipped:", e)
    # Feature importance
    plot_feature_importance(rf, features, PLOTS_DIR / f"featimp_rf_n{size}.png", title=f"RF Feature Importance (n={size})")

    # -----------------------
    # XGBoost
    # -----------------------
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    # compute train metrics for overfitting check
    xgb_train_acc, xgb_train_roc, xgb_train_pr = compute_metrics(xgb_model, X_train, y_train)

    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = safe_predict_proba(xgb_model, X_test)

    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_roc = roc_auc_score(y_test, xgb_probs) if len(np.unique(y_test))>1 else np.nan
    xgb_pr = average_precision_score(y_test, xgb_probs) if len(np.unique(y_test))>1 else np.nan

    print("\nXGBoost results (n={}):".format(size))
    print(f"  Train Accuracy: {xgb_train_acc*100:.2f}% | Test Accuracy: {xgb_acc*100:.2f}%")
    if not np.isnan(xgb_train_roc) and not np.isnan(xgb_roc):
        print(f"  Train ROC AUC:  {xgb_train_roc:.4f} | Test ROC AUC:  {xgb_roc:.4f}")
    else:
        print(f"  Train ROC AUC:  {xgb_train_roc} | Test ROC AUC:  {xgb_roc}")
    if not np.isnan(xgb_train_pr) and not np.isnan(xgb_pr):
        print(f"  Train PR AUC:   {xgb_train_pr:.4f} | Test PR AUC:   {xgb_pr:.4f}")
    else:
        print(f"  Train PR AUC:   {xgb_train_pr} | Test PR AUC:   {xgb_pr}")

    print("\nClassification Report (XGB):")
    print(classification_report(y_test, xgb_preds, digits=4))

    # Overfitting decision (XGB)
    overfit_reasons = []
    if xgb_train_acc - xgb_acc > THRESHOLDS['accuracy']:
        overfit_reasons.append(f"accuracy (train {xgb_train_acc:.3f} vs test {xgb_acc:.3f})")
    if (not np.isnan(xgb_train_roc)) and (not np.isnan(xgb_roc)) and (xgb_train_roc - xgb_roc > THRESHOLDS['roc_auc']):
        overfit_reasons.append(f"roc_auc (train {xgb_train_roc:.3f} vs test {xgb_roc:.3f})")
    if (not np.isnan(xgb_train_pr)) and (not np.isnan(xgb_pr)) and (xgb_train_pr - xgb_pr > THRESHOLDS['pr_auc']):
        overfit_reasons.append(f"pr_auc (train {xgb_train_pr:.3f} vs test {xgb_pr:.3f})")

    if overfit_reasons:
        print(">> XGBoost Overfitting detected. Reasons:", "; ".join(overfit_reasons))
    else:
        print(">> XGBoost No strong overfitting detected (within thresholds).")

    results.append({
        'model':'XGBoost',
        'n':size,
        'accuracy': xgb_acc,
        'roc_auc': xgb_roc,
        'pr_auc': xgb_pr,
        'n_train': len(X_train),
        'n_test': len(X_test)
    })

    # store ROC/PR curve points for xgb
    try:
        roc_data['xgb'][size] = (roc_curve(y_test, xgb_probs)[0], roc_curve(y_test, xgb_probs)[1])
    except Exception:
        roc_data['xgb'][size] = (np.array([]), np.array([]))
    try:
        pr_data['xgb'][size] = (precision_recall_curve(y_test, xgb_probs)[0], precision_recall_curve(y_test, xgb_probs)[1])
    except Exception:
        pr_data['xgb'][size] = (np.array([]), np.array([]))

    xgb_out = OUT_DIR / f"xgb_{size}.joblib"
    joblib.dump(xgb_model, xgb_out)
    print("Saved XGBoost model to:", xgb_out)

    # -----------------------
    # Plots for XGB (per-size)
    # -----------------------
    plot_roc(y_test, xgb_probs, PLOTS_DIR / f"roc_xgb_n{size}.png", title=f"XGB ROC (n={size})")
    plot_pr(y_test, xgb_probs, PLOTS_DIR / f"pr_xgb_n{size}.png", title=f"XGB PR (n={size})")
    plot_confusion(y_test, xgb_preds, PLOTS_DIR / f"cm_xgb_n{size}.png", title=f"XGB Confusion Matrix (n={size})")
    try:
        plot_learning_curve(xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE),
                            X_small, y_small, PLOTS_DIR / f"learning_xgb_n{size}.png", title=f"XGB Learning Curve (n={size})")
    except Exception as e:
        print("XGB learning curve skipped:", e)
    plot_feature_importance(xgb_model, features, PLOTS_DIR / f"featimp_xgb_n{size}.png", title=f"XGB Feature Importance (n={size})")

# -----------------------
# Save results summary
# -----------------------
results_df = pd.DataFrame(results)
results_df = results_df[['model','n','n_train','n_test','accuracy','roc_auc','pr_auc']]
results_df.to_csv(RESULTS_CSV, index=False)
print("\nSaved results summary to:", RESULTS_CSV)
print(results_df)

# -----------------------
# Plots: Accuracy vs Size (combined)
# -----------------------
plt.figure(figsize=(8,5))
sns.lineplot(data=results_df, x='n', y='accuracy', hue='model', marker='o')
plt.title("Accuracy vs Training Sample Size")
plt.xlabel("Sample size (total rows used)")
plt.ylabel("Accuracy")
plt.xticks(sorted(results_df['n'].unique()))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
acc_plot = PLOTS_DIR / "accuracy_vs_size.png"
plt.savefig(acc_plot, dpi=150)
plt.close()
print("Saved plot:", acc_plot)

# -----------------------
# Combined ROC plots per model (all sizes)
# -----------------------
for model_key, pretty_name in [('rf', 'RandomForest'), ('xgb', 'XGBoost')]:
    plt.figure(figsize=(10,6))
    for size in sizes:
        if size in roc_data[model_key] and len(roc_data[model_key][size][0])>0:
            fpr, tpr = roc_data[model_key][size]
            plt.plot(fpr, tpr, label=f"n={size}")
    plt.plot([0,1],[0,1],'--', color='gray', linewidth=0.8)
    plt.title(f"ROC Curves - {pretty_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    outp = PLOTS_DIR / f"roc_{model_key}.png"
    plt.savefig(outp, dpi=150)
    plt.close()
    print("Saved plot:", outp)

# -----------------------
# Combined PR curves per model (all sizes)
# -----------------------
for model_key, pretty_name in [('rf', 'RandomForest'), ('xgb', 'XGBoost')]:
    plt.figure(figsize=(10,6))
    for size in sizes:
        if size in pr_data[model_key] and len(pr_data[model_key][size][0])>0:
            precision, recall = pr_data[model_key][size]
            plt.plot(recall, precision, label=f"n={size}")
    plt.title(f"Precision-Recall Curves - {pretty_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    outp = PLOTS_DIR / f"pr_{model_key}.png"
    plt.savefig(outp, dpi=150)
    plt.close()
    print("Saved plot:", outp)

print("\nAll experiments finished. Summary saved to", RESULTS_CSV)
print("Model files saved to", OUT_DIR)
print("Plot files saved to", PLOTS_DIR)
