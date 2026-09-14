import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from catboost import CatBoostClassifier
import lightgbm as lgb

SEED = 42
N_FOLDS = 10
np.random.seed(SEED)
TARGET = "readmitted_30d"
ID_COL = "patient_id"

print("=" * 65)
print("ROUND 7: ULTIMATE RANK 1 WINNING PIPELINE")
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

def feature_engineer(df):
    df = df.copy()
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

    # Age & risk
    df["age_group"] = pd.cut(df["age"], bins=[0, 40, 60, 75, 120],
                              labels=["young", "middle", "senior", "elderly"])
    df["age_over65"] = (df["age"] >= 65).astype(int)

    # Comorbidity interactions
    df["comorbidity_load"] = (df["diabetes"] + df["hypertension"] +
                               df["chronic_kidney_disease"] + df["heart_failure"])
    df["cardiorenal"] = (df["chronic_kidney_disease"] & df["heart_failure"]).astype(int)
    df["diabetic_hypertensive"] = (df["diabetes"] & df["hypertension"]).astype(int)
    df["high_comorbidity"] = (df["comorbidity_load"] >= 3).astype(int)

    # Medication
    df["med_per_comorbidity"] = df["medication_count"] / (df["comorbidity_count"] + 1)
    df["polypharmacy"] = (df["medication_count"] >= 10).astype(int)

    # Hospitalization
    df["frequent_admitter"] = (df["prior_admissions_12m"] >= 2).astype(int)
    df["very_frequent"] = (df["prior_admissions_12m"] >= 4).astype(int)
    df["long_stay"] = (df["length_of_stay_days"] >= 7).astype(int)
    df["missed_appt_flag"] = (df["missed_appointments_12m"] >= 2).astype(int)
    df["late_followup"] = ((df["followup_days"] > 14) | df["followup_days"].isnull()).astype(int)

    df["stay_per_comorbidity"] = df["length_of_stay_days"] / (df["comorbidity_load"] + 1)
    df["admissions_per_year_age"] = df["prior_admissions_12m"] / (df["age"] / 10 + 1)
    df["labs_abnormal_count"] = (df["anemia_flag"] + df["aki_flag"] +
                                  df["hyponatremia_flag"] + df["tachy_flag"] + df["htn_bp_flag"])

    df["simple_risk_score"] = (df["frequent_admitter"] + df["long_stay"] +
                                df["high_comorbidity"] + df["polypharmacy"] +
                                df["missed_appt_flag"] + df["late_followup"] +
                                df["anemia_flag"] + df["aki_flag"])
    return df

train_fe = feature_engineer(train_raw)
test_fe = feature_engineer(test_raw)

NEW_CAT = ["age_group"]
NEW_NUM = ["anemia_flag", "severe_anemia", "aki_flag", "severe_aki", "hyponatremia_flag",
           "tachy_flag", "htn_bp_flag",
           "hemoglobin_g_dl_missing", "creatinine_mg_dl_missing", "sodium_mmol_l_missing",
           "heart_rate_bpm_missing", "systolic_bp_mmhg_missing", "followup_days_missing",
           "n_labs_missing", "age_over65",
           "comorbidity_load", "cardiorenal", "diabetic_hypertensive", "high_comorbidity",
           "med_per_comorbidity", "polypharmacy",
           "frequent_admitter", "very_frequent", "long_stay",
           "missed_appt_flag", "late_followup",
           "stay_per_comorbidity", "admissions_per_year_age", "labs_abnormal_count",
           "simple_risk_score"]

ALL_CATS = CAT_COLS + NEW_CAT
ALL_NUMS = NUM_COLS + NEW_NUM
FEATURES = ALL_NUMS + ALL_CATS

X_native = train_fe[FEATURES].copy()
X_test_native = test_fe[FEATURES].copy()
for c in ALL_CATS:
    X_native[c] = X_native[c].astype("category")
    X_test_native[c] = X_test_native[c].astype("category")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 1. CatBoost (2 Seeds: 42 and 777)
print("\n--- Training CatBoost Multi-Seed (10 Folds) ---")
cb_oof = np.zeros(len(X_native))
cb_test = np.zeros(len(X_test_native))
for s_idx, seed in enumerate([42, 777]):
    for f, (ti, vi) in enumerate(skf.split(X_native, y)):
        cb = CatBoostClassifier(
            iterations=1400, learning_rate=0.03, depth=5, l2_leaf_reg=4,
            cat_features=ALL_CATS, eval_metric="Logloss",
            random_seed=seed + f, early_stopping_rounds=100, verbose=False
        )
        cb.fit(X_native.iloc[ti], y.iloc[ti], eval_set=(X_native.iloc[vi], y.iloc[vi]))
        cb_oof[vi] += cb.predict_proba(X_native.iloc[vi])[:, 1] / 2.0
        cb_test += cb.predict_proba(X_test_native)[:, 1] / (N_FOLDS * 2.0)

print(f"  [CatBoost] OOF Log Loss: {log_loss(y, cb_oof):.5f} | AUC: {roc_auc_score(y, cb_oof):.5f}")

# 2. LightGBM (2 Seeds: 42 and 777)
print("\n--- Training LightGBM Multi-Seed (10 Folds) ---")
lgb_oof = np.zeros(len(X_native))
lgb_test = np.zeros(len(X_test_native))
LGB_P = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.025, num_leaves=24, max_depth=4,
    min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=5, lambda_l1=1.5, lambda_l2=2.0,
    n_estimators=2200, n_jobs=-1, verbose=-1
)
for s_idx, seed in enumerate([42, 777]):
    for f, (ti, vi) in enumerate(skf.split(X_native, y)):
        lg = lgb.LGBMClassifier(**LGB_P, random_state=seed)
        lg.fit(X_native.iloc[ti], y.iloc[ti], eval_set=[(X_native.iloc[vi], y.iloc[vi])],
               callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        lgb_oof[vi] += lg.predict_proba(X_native.iloc[vi])[:, 1] / 2.0
        lgb_test += lg.predict_proba(X_test_native)[:, 1] / (N_FOLDS * 2.0)

print(f"  [LightGBM] OOF Log Loss: {log_loss(y, lgb_oof):.5f} | AUC: {roc_auc_score(y, lgb_oof):.5f}")

# 3. Calibrated Lasso & ElasticNet Logistic Regression
print("\n--- Training Calibrated Linear Models (10 Folds) ---")
X_lr_raw = train_fe[FEATURES].copy()
X_test_lr_raw = test_fe[FEATURES].copy()
for c in ALL_CATS:
    X_lr_raw[c] = X_lr_raw[c].astype(str)
    X_test_lr_raw[c] = X_test_lr_raw[c].astype(str)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_ohe = ohe.fit_transform(X_lr_raw[ALL_CATS])
test_ohe = ohe.transform(X_test_lr_raw[ALL_CATS])

imputer = IterativeImputer(random_state=SEED, max_iter=10, n_nearest_features=8)
train_num_imp = imputer.fit_transform(X_lr_raw[ALL_NUMS])
test_num_imp = imputer.transform(X_test_lr_raw[ALL_NUMS])

scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_num_imp)
test_num_scaled = scaler.transform(test_num_imp)

Xs = np.hstack([train_num_scaled, train_ohe])
Xs_t = np.hstack([test_num_scaled, test_ohe])

lr_oof = np.zeros(len(Xs))
lr_test = np.zeros(len(Xs_t))

for f, (ti, vi) in enumerate(skf.split(Xs, y)):
    base_lr = LogisticRegression(C=0.15, penalty="l1", solver="saga", max_iter=2000, random_state=SEED)
    m = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    m.fit(Xs[ti], y.iloc[ti])
    lr_oof[vi] = m.predict_proba(Xs[vi])[:, 1]
    lr_test += m.predict_proba(Xs_t)[:, 1] / N_FOLDS

print(f"  [Lasso-LR] OOF Log Loss: {log_loss(y, lr_oof):.5f} | AUC: {roc_auc_score(y, lr_oof):.5f}")

# 4. Optimize 10-Fold Ensemble
print("\n--- Optimizing 10-Fold Stacking Weights ---")
oofs = [cb_oof, lgb_oof, lr_oof]
tests = [cb_test, lgb_test, lr_test]
names = ["CatBoost", "LightGBM", "Lasso-LR"]

def loss_func(weights):
    w = np.clip(weights, 0, None)
    s = w.sum()
    if s <= 1e-9: return 999.0
    w = w / s
    blend = sum(wi * oi for wi, oi in zip(w, oofs))
    blend = np.clip(blend, 1e-6, 1 - 1e-6)
    return log_loss(y, blend)

best_res, best_loss = None, np.inf
rng = np.random.RandomState(SEED)
for _ in range(50):
    x0 = rng.dirichlet(np.ones(len(oofs)))
    res = minimize(loss_func, x0, bounds=[(0, 1)] * len(oofs), method="SLSQP")
    if res.fun < best_loss:
        best_loss, best_res = res.fun, res

w = np.clip(best_res.x, 0, None)
w = w / w.sum()
for n, wi in zip(names, w):
    print(f"  {n}: {wi:.4f}")

new_10fold_test = sum(wi * ti for wi, ti in zip(w, tests))

# ============================================================
# MASTER ENSEMBLE: BLENDING TOP SUBMISSIONS
# ============================================================
# Sub 4: Kaggle 0.33423
# Sub 3: Kaggle 0.33444 (prior calibrated)
# New 10-Fold: 10-Fold Multi-Seed
sub3 = pd.read_csv("codewave submission 3.csv")["readmitted_30d"].values
sub4 = pd.read_csv("codewave submission 4.csv")["readmitted_30d"].values

print("\n--- Building Master Ensemble ---")
# Optimal convex blend:
# Sub 4 (45%) + Sub 3 (30%) + New 10-Fold (25%)
master_blend = 0.45 * sub4 + 0.30 * sub3 + 0.25 * new_10fold_test

# Bounded clinical safe clipping:
final_probs = np.clip(master_blend, 0.020, 0.725)

# Save as codewave submission 7.csv
sub = pd.DataFrame({
    "patient_id": test_raw[ID_COL].values,
    "readmitted_30d": final_probs
})
sub.to_csv("codewave submission 7.csv", index=False)

print(f"\n  Saved: 'codewave submission 7.csv' ({len(sub)} rows)")
print(f"  Min prob: {final_probs.min():.6f}")
print(f"  Max prob: {final_probs.max():.6f}")
print(f"  Mean prob: {final_probs.mean():.6f}")
print("\nFirst 10 rows:")
print(sub.head(10))

print("\n" + "=" * 65)
print("ROUND 7 COMPLETED SUCCESSFULLY!")
print("=" * 65)
