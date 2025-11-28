"""
RF_overfit_check.py

Usage: python RF_overfit_check.py
Requires: pandas, numpy, matplotlib, scikit-learn

This script:
- loads /mnt/data/pph_sri_lanka_synthetic.csv
- auto-detects the PPH target column
- maps 'yes'/'no' -> 1/0 and drops unknowns
- for sample sizes [300,700,1000,1500] (skips if not enough rows):
    - creates stratified sample (exact n)
    - splits 80/20 stratified
    - trains a RandomForest (n_estimators=200)
    - calculates train/test metrics (acc, AUC, precision, recall, f1)
    - produces plots: feature importances, confusion matrix, ROC, learning curve, validation curve (max_depth)
    - prints a diagnostic message "Overfitted / Underfitted / Not overfitted" with numeric gaps
- saves plots to the current folder (png files)
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = "pph_sri_lanka_synthetic.csv"
OUT_DIR = "./rf_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RND = 42

def safe_onehot_encoder():
    """Return a OneHotEncoder configured in a way that works across sklearn versions."""
    # try preferred newer param names; fallback gracefully
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown='ignore', sparse=False)
        except TypeError:
            # oldest versions: no sparse arg -> returns sparse matrix
            return OneHotEncoder(handle_unknown='ignore')

def detect_target_column(df):
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['pph','postpartum','post-partum','haemorr','hemorr','bleed','history','outcome','target','label'])]
    if candidates:
        return candidates[0]
    return df.columns[-1]

def map_target_series(s):
    s_clean = s.astype(str).str.strip().str.lower()
    m = {'yes':1,'y':1,'true':1,'1':1,'no':0,'n':0,'false':0,'0':0}
    mapped = s_clean.map(m)
    # fallback: if contains 'yes' or 'no' as substring
    mapped2 = s_clean.apply(lambda x: 1 if 'yes' in x else (0 if 'no' in x else np.nan))
    # choose mapped where available else mapped2
    final = mapped.copy()
    final[final.isna()] = mapped2[final.isna()]
    return final

def get_feature_names_from_column_transformer(ct, numeric_cols, categorical_cols):
    # After fitting, get names for numeric and categorical (onehot) transforms
    names = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        # try to extract categories from OneHotEncoder
        try:
            ohe = ct.named_transformers_['cat'].named_steps['onehot']
            for i, col in enumerate(categorical_cols):
                cats = ohe.categories_[i]
                names.extend([f"{col}__{c}" for c in cats])
        except Exception:
            names.extend(categorical_cols)
    return names

def plot_and_save(fig, fname):
    path = os.path.join(OUT_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")

# --- Load data
df = pd.read_csv(CSV_PATH)
print("Loaded CSV:", CSV_PATH, "shape:", df.shape)

# detect and map target
target_col = detect_target_column(df)
print("Detected target column:", target_col)
mapped_target = map_target_series(df[target_col])
print("Original target value counts (sample):")
print(df[target_col].value_counts(dropna=False).head(10))
print("Mapped target counts (including NaN):")
print(mapped_target.value_counts(dropna=False).to_dict())

# drop rows where mapping failed (unknown)
keep_mask = ~mapped_target.isna()
df = df.loc[keep_mask].copy()
df[target_col] = mapped_target.loc[keep_mask].astype(int)
print("Rows after dropping unknown target rows:", len(df))

# separate X,y and column types
y_all = df[target_col]
X_all = df.drop(columns=[target_col])
numeric_cols = X_all.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_all.select_dtypes(include=['object','category','bool']).columns.tolist()
print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols[:10])

# preprocessors
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', safe_onehot_encoder())])
preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_cols), ('cat', categorical_transformer, categorical_cols)], remainder='drop')

# sample sizes
sample_sizes = [300, 700, 1000, 1500]
results = []

for n in sample_sizes:
    print("\n" + "="*60)
    print("Processing sample size =", n)
    if len(df) < n:
        print("  Skip: dataset has only", len(df), "rows.")
        continue

    # produce stratified sample with approximate proportions, then exact n
    proportions = df[target_col].value_counts(normalize=True).to_dict()
    parts = []
    for cls, prop in proportions.items():
        k = int(round(prop * n))
        k = max(k, 1)  # ensure at least one per class
        cls_df = df[df[target_col] == cls]
        # if not enough examples of cls, sample with replacement
        replace = k > len(cls_df)
        parts.append(cls_df.sample(n=k, replace=replace, random_state=RND))
    sampled = pd.concat(parts).sample(n, random_state=RND)  # shuffle & exact size
    X = sampled.drop(columns=[target_col]).reset_index(drop=True)
    y = sampled[target_col].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=RND)
    print(" Train/Test shapes:", X_train.shape, X_test.shape)
    # Build pipeline (baseline RF without class balancing so we can detect overfit of baseline)
    rf = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
    pipe = Pipeline([('pre', preprocessor), ('clf', rf)])
    # fit
    pipe.fit(X_train, y_train)

    # predictions & metrics
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    y_proba_train = pipe.predict_proba(X_train)[:,1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
    y_proba_test = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_auc = roc_auc_score(y_train, y_proba_train) if y_proba_train is not None else math.nan
    test_auc = roc_auc_score(y_test, y_proba_test) if y_proba_test is not None else math.nan
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    # cross-val on training set (roc_auc)
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except Exception:
        cv_mean = np.nan; cv_std = np.nan

    # feature names and importances
    # we need to fit the preprocessor to extract the onehot categories (already fitted via pipe)
    try:
        feat_names = get_feature_names_from_column_transformer(pipe.named_steps['pre'], numeric_cols, categorical_cols)
    except Exception:
        feat_names = numeric_cols + categorical_cols
    importances = pipe.named_steps['clf'].feature_importances_ if hasattr(pipe.named_steps['clf'], 'feature_importances_') else None

    # --- Plots ---
    # 1) Feature importances (top 20)
    if importances is not None and len(importances) == len(feat_names):
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        topk = min(20, len(fi))
        fig, ax = plt.subplots(figsize=(10, max(3, topk*0.35)))
        fi.iloc[:topk].plot.bar(ax=ax)
        ax.set_title(f"Top {topk} Feature Importances (n={n})")
        ax.set_ylabel("Importance")
        plot_and_save(fig, f"feature_importances_n{n}.png")
    else:
        print("  Feature importance not available or mismatch in length.")

    # 2) Confusion matrix
    fig, ax = plt.subplots(figsize=(4,4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center')
    ax.set_title(f"Confusion Matrix (n={n})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plot_and_save(fig, f"confusion_matrix_n{n}.png")

    # 3) ROC curve (if probabilities available)
    if y_proba_test is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        roc_auc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}")
        ax.plot([0,1],[0,1],'--')
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve (n={n})")
        ax.legend()
        plot_and_save(fig, f"roc_curve_n{n}.png")
    else:
        print("  No predict_proba available to draw ROC")

    # 4) Learning curve (train vs validation) - using accuracy
    try:
        train_sizes, train_scores, val_scores = learning_curve(pipe, X_train, y_train, cv=5,
                                                               scoring='accuracy', train_sizes=np.linspace(0.1,1.0,5), n_jobs=-1)
        tr_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(train_sizes, tr_mean, marker='o', label='Train acc')
        ax.plot(train_sizes, val_mean, marker='o', label='Val acc')
        ax.set_xlabel("Training examples"); ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve (n={n})")
        ax.legend()
        plot_and_save(fig, f"learning_curve_n{n}.png")
    except Exception as e:
        print("  Learning curve failed:", e)

    # 5) Validation curve for max_depth
    try:
        param_range = np.arange(2, 21, 2)
        # create fresh RF wrappers for validation_curve (sklearn expects an unfitted estimator)
        def make_pipe_with_maxdepth(md):
            return Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, max_depth=md, random_state=RND))])
        train_scores_v, val_scores_v = validation_curve(make_pipe_with_maxdepth(5), X_train, y_train,
                                                       param_name='clf__max_depth', param_range=param_range,
                                                       cv=5, scoring='accuracy', n_jobs=-1)
        train_mean_v = train_scores_v.mean(axis=1)
        val_mean_v = val_scores_v.mean(axis=1)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(param_range, train_mean_v, marker='o', label='Train acc')
        ax.plot(param_range, val_mean_v, marker='o', label='Val acc')
        ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy")
        ax.set_title(f"Validation Curve (max_depth) n={n}")
        ax.legend()
        plot_and_save(fig, f"validation_curve_maxdepth_n{n}.png")
    except Exception as e:
        print("  Validation curve failed:", e)

    # --- Overfitting diagnostic ---
    # Define numeric gaps
    acc_gap = train_acc - test_acc
    auc_gap = (train_auc - test_auc) if (not math.isnan(train_auc) and not math.isnan(test_auc)) else math.nan

    # primary condition: if train acc much greater than test acc OR train_auc much greater than test_auc
    overfit = False
    underfit = False
    # thresholds (you can tune these)
    ACC_GAP_THRESH = 0.08   # 8% accuracy gap
    AUC_GAP_THRESH = 0.08   # 0.08 AUC gap

    if acc_gap > ACC_GAP_THRESH or (not math.isnan(auc_gap) and auc_gap > AUC_GAP_THRESH):
        overfit = True
    # underfit: both train and test are low (e.g., both < 0.65 accuracy)
    if train_acc < 0.65 and test_acc < 0.65:
        underfit = True

    # print results and conclusion
    print(f"Results n={n}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, acc_gap={acc_gap:.3f}")
    print(f"               train_auc={train_auc if not math.isnan(train_auc) else 'NA'}, test_auc={test_auc if not math.isnan(test_auc) else 'NA'}, auc_gap={auc_gap if not math.isnan(auc_gap) else 'NA'}")
    print(f"               precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}, cv_mean_auc={cv_mean:.3f} (+/- {cv_std:.3f})")
    if overfit:
        print(">>> DIAGNOSIS: Overfitted (model performs significantly better on train than test).")
    elif underfit:
        print(">>> DIAGNOSIS: Underfitted (both train and test accuracies are low).")
    else:
        print(">>> DIAGNOSIS: Not overfitted (train/test gap small and performance reasonable).")

    # store summary row
    results.append({
        'n': n,
        'train_acc': train_acc, 'test_acc': test_acc, 'acc_gap': acc_gap,
        'train_auc': train_auc, 'test_auc': test_auc, 'auc_gap': auc_gap,
        'precision': prec, 'recall': rec, 'f1': f1, 'cv_mean_auc': cv_mean
    })

# Final summary table
if results:
    summary_df = pd.DataFrame(results).sort_values('n')
    summary_csv = os.path.join(OUT_DIR, "rf_overfit_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\nSaved summary table to:", summary_csv)
    print(summary_df)
else:
    print("No results (no sample sizes processed).")
