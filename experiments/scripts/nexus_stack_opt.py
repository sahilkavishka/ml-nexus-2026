import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)
TARGET = "readmitted_30d"
ID_COL = "patient_id"

print("=" * 65)
print("ROUND 10: ADVANCED FEATURE INTERACTIONS + TARGET ENCODING + STACKING")
print("=" * 65)

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
y = train_raw[TARGET]

CAT_COLS = ["sex", "rurality", "hospital_type", "region",
            "discharge_disposition", "care_pathway"]
NUM_COLS = ["age", "socioeconomic_index", "prior_admissions_12m", "comorbidity_count",
            "diabetes", "hypertension", "chronic_kidney_disease", "heart_failure",
            "length_of_stay_days", "medication_count", "missed_appointments_12m",
            "followup_days", "hemoglobin_g_dl", "creatinine_mg_dl", "sodium_mmol_l",
            "heart_rate_bpm", "systolic_bp_mmhg"]

# 1. Feature Engineering with Suggested Clinical Interactions
def feature_engineer(df):
    df = df.copy()
    # Suggested interaction features
    df["shock_index"] = df["heart_rate_bpm"] / (df["systolic_bp_mmhg"] + 1e-5)
    df["age_comorbidity"] = df["age"] * df["comorbidity_count"]
    df["admit_stay_interaction"] = df["prior_admissions_12m"] * df["length_of_stay_days"]
    df["age_admissions"] = df["age"] * df["prior_admissions_12m"]
    df["stay_per_comorbidity"] = df["length_of_stay_days"] / (df["comorbidity_count"] + 1)
    df["admissions_per_age"] = df["prior_admissions_12m"] / (df["age"] / 10.0 + 1.0)
    
    # Clinical flags
    df["anemia_flag"] = (df["hemoglobin_g_dl"] < 11.0).astype(float)
    df["severe_anemia"] = (df["hemoglobin_g_dl"] < 9.0).astype(float)
    df["aki_flag"] = (df["creatinine_mg_dl"] > 1.5).astype(float)
    df["severe_aki"] = (df["creatinine_mg_dl"] > 3.0).astype(float)
    df["hyponatremia_flag"] = (df["sodium_mmol_l"] < 135).astype(float)
    df["tachy_flag"] = (df["heart_rate_bpm"] > 100).astype(float)
    df["htn_bp_flag"] = (df["systolic_bp_mmhg"] > 140).astype(float)

    # Missingness indicators
    lab_cols = ["hemoglobin_g_dl", "creatinine_mg_dl", "sodium_mmol_l",
                "heart_rate_bpm", "systolic_bp_mmhg", "followup_days"]
    for c in lab_cols:
        df[c + "_missing"] = df[c].isnull().astype(int)
    df["n_labs_missing"] = df[[c + "_missing" for c in lab_cols]].sum(axis=1)

    # Comorbidity interactions
    df["comorbidity_load"] = (df["diabetes"] + df["hypertension"] +
                               df["chronic_kidney_disease"] + df["heart_failure"])
    df["cardiorenal"] = (df["chronic_kidney_disease"] & df["heart_failure"]).astype(int)
    df["diabetic_hypertensive"] = (df["diabetes"] & df["hypertension"]).astype(int)
    df["high_comorbidity"] = (df["comorbidity_load"] >= 3).astype(int)
    df["polypharmacy"] = (df["medication_count"] >= 10).astype(int)
    df["frequent_admitter"] = (df["prior_admissions_12m"] >= 2).astype(int)
    df["very_frequent"] = (df["prior_admissions_12m"] >= 4).astype(int)
    df["long_stay"] = (df["length_of_stay_days"] >= 7).astype(int)
    df["missed_appt_flag"] = (df["missed_appointments_12m"] >= 2).astype(int)
    df["late_followup"] = ((df["followup_days"] > 14) | df["followup_days"].isnull()).astype(int)

    df["simple_risk_score"] = (df["frequent_admitter"] + df["long_stay"] +
                                df["high_comorbidity"] + df["polypharmacy"] +
                                df["missed_appt_flag"] + df["late_followup"] +
                                df["anemia_flag"] + df["aki_flag"])
    return df

train_fe = feature_engineer(train_raw)
test_fe = feature_engineer(test_raw)

# 2. Out-of-Fold Target Encoding for Categoricals
print("Calculating Out-of-Fold Target Encodings...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
global_mean = y.mean()
smoothing = 10.0

for col in CAT_COLS:
    train_fe[col + "_te"] = np.nan
    test_fe[col + "_te"] = np.zeros(len(test_fe))
    
    # Global counts for test set
    counts = train_fe[col].value_counts()
    sums = train_fe.groupby(col)[TARGET].sum()
    smoothed_te = (sums + smoothing * global_mean) / (counts + smoothing)
    test_fe[col + "_te"] = test_fe[col].map(smoothed_te).fillna(global_mean).values

    # OOF encoding for train set
    for ti, vi in skf.split(train_fe, y):
        tr_fold = train_fe.iloc[ti]
        counts_fold = tr_fold[col].value_counts()
        sums_fold = tr_fold.groupby(col)[TARGET].sum()
        fold_smoothed = (sums_fold + smoothing * global_mean) / (counts_fold + smoothing)
        train_fe.iloc[vi, train_fe.columns.get_loc(col + "_te")] = (
            train_fe.iloc[vi][col].map(fold_smoothed).fillna(global_mean).values
        )

TE_COLS = [c + "_te" for c in CAT_COLS]
NUMERIC_FEATURES = [c for c in train_fe.columns if c not in CAT_COLS + [ID_COL, TARGET]]
ALL_FEATURES = NUMERIC_FEATURES + CAT_COLS

print(f"Features: {len(ALL_FEATURES)} (Numerical: {len(NUMERIC_FEATURES)}, Categorical: {len(CAT_COLS)})")

# Native format for tree models
X_native = train_fe[ALL_FEATURES].copy()
X_test_native = test_fe[ALL_FEATURES].copy()
for c in CAT_COLS:
    X_native[c] = X_native[c].astype("category")
    X_test_native[c] = X_test_native[c].astype("category")

# Storage for Level-1 models
oof_dict = {}
test_dict = {}

# ── Model 1: CatBoost ─────────────────────────────────────────────────────────
print("\n--- Training CatBoost ---")
cb_oof = np.zeros(len(X_native))
cb_test = np.zeros(len(X_test_native))
for f, (ti, vi) in enumerate(skf.split(X_native, y)):
    cb = CatBoostClassifier(
        iterations=1800, learning_rate=0.025, depth=5, l2_leaf_reg=4,
        cat_features=CAT_COLS, eval_metric="Logloss",
        random_seed=SEED + f, early_stopping_rounds=100, verbose=False
    )
    cb.fit(X_native.iloc[ti], y.iloc[ti], eval_set=(X_native.iloc[vi], y.iloc[vi]))
    cb_oof[vi] = cb.predict_proba(X_native.iloc[vi])[:, 1]
    cb_test += cb.predict_proba(X_test_native)[:, 1] / N_FOLDS

print(f"  [CatBoost] OOF Log Loss: {log_loss(y, cb_oof):.5f} | AUC: {roc_auc_score(y, cb_oof):.5f}")
oof_dict["cat"] = cb_oof
test_dict["cat"] = cb_test

# ── Model 2: LightGBM (Tuned) ────────────────────────────────────────────────
print("\n--- Training LightGBM (Tuned) ---")
lgb_oof = np.zeros(len(X_native))
lgb_test = np.zeros(len(X_test_native))
LGB_P = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.022, num_leaves=22, max_depth=4,
    min_child_samples=45, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=5, lambda_l1=2.0, lambda_l2=2.5,
    n_estimators=2500, n_jobs=-1, random_state=SEED, verbose=-1
)
for f, (ti, vi) in enumerate(skf.split(X_native, y)):
    lg = lgb.LGBMClassifier(**LGB_P)
    lg.fit(X_native.iloc[ti], y.iloc[ti], eval_set=[(X_native.iloc[vi], y.iloc[vi])],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    lgb_oof[vi] = lg.predict_proba(X_native.iloc[vi])[:, 1]
    lgb_test += lg.predict_proba(X_test_native)[:, 1] / N_FOLDS

print(f"  [LightGBM] OOF Log Loss: {log_loss(y, lgb_oof):.5f} | AUC: {roc_auc_score(y, lgb_oof):.5f}")
oof_dict["lgb"] = lgb_oof
test_dict["lgb"] = lgb_test

# ── Model 3: XGBoost (hist tree method) ──────────────────────────────────────
print("\n--- Training XGBoost ---")
X_xgb = train_fe[NUMERIC_FEATURES].copy()
X_test_xgb = test_fe[NUMERIC_FEATURES].copy()

xgb_oof = np.zeros(len(X_xgb))
xgb_test = np.zeros(len(X_test_xgb))
XGB_P = dict(
    n_estimators=2000, learning_rate=0.025, max_depth=4,
    min_child_weight=6, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=1.5, reg_lambda=2.5, eval_metric="logloss",
    tree_method="hist", random_state=SEED, n_jobs=-1
)
for f, (ti, vi) in enumerate(skf.split(X_xgb, y)):
    xg = xgb.XGBClassifier(**XGB_P, early_stopping_rounds=100)
    xg.fit(X_xgb.iloc[ti], y.iloc[ti], eval_set=[(X_xgb.iloc[vi], y.iloc[vi])], verbose=False)
    xgb_oof[vi] = xg.predict_proba(X_xgb.iloc[vi])[:, 1]
    xgb_test += xg.predict_proba(X_test_xgb)[:, 1] / N_FOLDS

print(f"  [XGBoost] OOF Log Loss: {log_loss(y, xgb_oof):.5f} | AUC: {roc_auc_score(y, xgb_oof):.5f}")
oof_dict["xgb"] = xgb_oof
test_dict["xgb"] = xgb_test

# ── Model 4 & 5: Calibrated Lasso & Ridge Logistic Regression ────────────────
print("\n--- Training Calibrated Linear Models (Lasso & Ridge) ---")
X_lr_raw = train_fe[ALL_FEATURES].copy()
X_test_lr_raw = test_fe[ALL_FEATURES].copy()
for c in CAT_COLS:
    X_lr_raw[c] = X_lr_raw[c].astype(str)
    X_test_lr_raw[c] = X_test_lr_raw[c].astype(str)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_ohe = ohe.fit_transform(X_lr_raw[CAT_COLS])
test_ohe = ohe.transform(X_test_lr_raw[CAT_COLS])

imputer = IterativeImputer(random_state=SEED, max_iter=10, n_nearest_features=12)
train_num_imp = imputer.fit_transform(X_lr_raw[NUMERIC_FEATURES])
test_num_imp = imputer.transform(X_test_lr_raw[NUMERIC_FEATURES])

scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_num_imp)
test_num_scaled = scaler.transform(test_num_imp)

Xs = np.hstack([train_num_scaled, train_ohe])
Xs_t = np.hstack([test_num_scaled, test_ohe])

lasso_oof = np.zeros(len(Xs)); lasso_test = np.zeros(len(Xs_t))
ridge_oof = np.zeros(len(Xs)); ridge_test = np.zeros(len(Xs_t))

for f, (ti, vi) in enumerate(skf.split(Xs, y)):
    # Lasso L1
    m1 = CalibratedClassifierCV(LogisticRegression(C=0.15, penalty="l1", solver="saga", max_iter=2000, random_state=SEED), method="sigmoid", cv=3)
    m1.fit(Xs[ti], y.iloc[ti])
    lasso_oof[vi] = m1.predict_proba(Xs[vi])[:, 1]
    lasso_test += m1.predict_proba(Xs_t)[:, 1] / N_FOLDS
    
    # Ridge L2
    m2 = CalibratedClassifierCV(LogisticRegression(C=0.20, penalty="l2", solver="lbfgs", max_iter=2000, random_state=SEED), method="sigmoid", cv=3)
    m2.fit(Xs[ti], y.iloc[ti])
    ridge_oof[vi] = m2.predict_proba(Xs[vi])[:, 1]
    ridge_test += m2.predict_proba(Xs_t)[:, 1] / N_FOLDS

print(f"  [Lasso-LR] OOF Log Loss: {log_loss(y, lasso_oof):.5f} | AUC: {roc_auc_score(y, lasso_oof):.5f}")
print(f"  [Ridge-LR] OOF Log Loss: {log_loss(y, ridge_oof):.5f} | AUC: {roc_auc_score(y, ridge_oof):.5f}")
oof_dict["lasso"] = lasso_oof
test_dict["lasso"] = lasso_test
oof_dict["ridge"] = ridge_oof
test_dict["ridge"] = ridge_test

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOGISTIC REGRESSION META-LEARNER STACKING (With Cross-Validation)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TRAINING LOGISTIC REGRESSION META-LEARNER (CV-STACKING)")
print("=" * 65)

model_keys = list(oof_dict.keys())
stack_X = np.column_stack([oof_dict[k] for k in model_keys])
stack_X_test = np.column_stack([test_dict[k] for k in model_keys])

meta_oof = np.zeros(len(y))
meta_test = np.zeros(len(stack_X_test))

for ti, vi in skf.split(stack_X, y):
    meta = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    meta.fit(stack_X[ti], y.iloc[ti])
    meta_oof[vi] = meta.predict_proba(stack_X[vi])[:, 1]
    meta_test += meta.predict_proba(stack_X_test)[:, 1] / N_FOLDS

print(f"  >>> Stacked Meta-Learner OOF Log Loss: {log_loss(y, meta_oof):.5f}")
print(f"  >>> Stacked Meta-Learner OOF ROC-AUC : {roc_auc_score(y, meta_oof):.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MASTER BLEND WITH PROVEN BEST KAGGLE SUBMISSION (SUBMISSION 4)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- Creating Master Winning Blend ---")
sub4 = pd.read_csv("codewave submission 4.csv")["readmitted_30d"].values

# Blend: 50% Meta-Stacked Model + 50% Sub 4
master_blend = 0.50 * meta_test + 0.50 * sub4

# Bounded clinical safety limits
final_probs = np.clip(master_blend, 0.022, 0.728)

# Save as codewave submission 10 final.csv
sub = pd.DataFrame({
    "patient_id": test_raw[ID_COL].values,
    "readmitted_30d": final_probs
})
sub.to_csv("codewave submission 10 final.csv", index=False)

print(f"\n  Saved: 'codewave submission 10 final.csv' ({len(sub)} rows)")
print(f"  Min prob: {final_probs.min():.6f}")
print(f"  Max prob: {final_probs.max():.6f}")
print(f"  Mean prob: {final_probs.mean():.6f}")
print("\nFirst 10 rows:")
print(sub.head(10))

print("\n" + "=" * 65)
print("ROUND 10 COMPLETE!")
print("=" * 65)
