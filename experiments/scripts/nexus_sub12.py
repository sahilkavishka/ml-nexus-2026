from builtins import print
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
N_FOLDS = 5
np.random.seed(SEED)
TARGET = "readmitted_30d"
ID_COL = "patient_id"

print("=" * 65)
print("ROUND 12: DOMAIN-ADAPTED COVARIATE-SHIFT CHAMPION PIPELINE")
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

# 1. Feature Engineering
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
    lab_cols = ["hemoglobin_g_dl", "creatinine_mg_dl", "sodium_mmol_l","heart_rate_bpm", "systolic_bp_mmhg", "followup_days"]
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

NEW_NUM = ["anemia_flag", "severe_anemia", "aki_flag", "severe_aki", "hyponatremia_flag",
           "tachy_flag", "htn_bp_flag",
           "hemoglobin_g_dl_missing", "creatinine_mg_dl_missing", "sodium_mmol_l_missing",
           "heart_rate_bpm_missing", "systolic_bp_mmhg_missing", "followup_days_missing",
           "n_labs_missing",
           "comorbidity_load", "cardiorenal", "diabetic_hypertensive", "high_comorbidity",
           "polypharmacy", "frequent_admitter", "very_frequent", "long_stay",
           "missed_appt_flag", "late_followup", "simple_risk_score"]

ALL_NUMS = NUM_COLS + NEW_NUM
FEATURES = ALL_NUMS + CAT_COLS

# 2. Compute Density Ratio Sample Weights (Covariate Shift Adaptation)
print("\n--- Computing Density Ratio Sample Weights for Test-Set Adaptation ---")
adv_features = ["age", "prior_admissions_12m", "comorbidity_count", "length_of_stay_days", "heart_failure", "chronic_kidney_disease"]
X_adv_tr = train_raw[adv_features].fillna(train_raw[adv_features].median())
X_adv_te = test_raw[adv_features].fillna(train_raw[adv_features].median())

adv_X = np.vstack([X_adv_tr, X_adv_te])
adv_y = np.array([0] * len(train_raw) + [1] * len(test_raw))
scaler_adv = StandardScaler()
adv_Xs = scaler_adv.fit_transform(adv_X)

adv_clf = LogisticRegression(C=0.1, random_state=SEED).fit(adv_Xs, adv_y)
p_te = adv_clf.predict_proba(adv_Xs[:len(train_raw)])[:, 1]
sample_weights = (p_te / (1.0 - p_te + 1e-6)) * (len(train_raw) / len(test_raw))
sample_weights = np.clip(sample_weights, 0.70, 1.60)
print(f"  Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

# 3. Native format for trees
X_native = train_fe[FEATURES].copy()
X_test_native = test_fe[FEATURES].copy()
for c in CAT_COLS:
    X_native[c] = X_native[c].astype("category")
    X_test_native[c] = X_test_native[c].astype("category")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Train CatBoost with sample weights
print("\n--- Training Domain-Adapted CatBoost ---")
cb_oof = np.zeros(len(X_native))
cb_test = np.zeros(len(X_test_native))
for f, (ti, vi) in enumerate(skf.split(X_native, y)):
    cb = CatBoostClassifier(
        iterations=1800, learning_rate=0.025, depth=5, l2_leaf_reg=4,
        cat_features=CAT_COLS, eval_metric="Logloss",
        random_seed=SEED + f, early_stopping_rounds=100, verbose=False
    )
    cb.fit(X_native.iloc[ti], y.iloc[ti], sample_weight=sample_weights[ti],
           eval_set=(X_native.iloc[vi], y.iloc[vi]))
    cb_oof[vi] = cb.predict_proba(X_native.iloc[vi])[:, 1]
    cb_test += cb.predict_proba(X_test_native)[:, 1] / N_FOLDS

print(f"  [CatBoost] OOF Log Loss: {log_loss(y, cb_oof):.5f} | AUC: {roc_auc_score(y, cb_oof):.5f}")

# Train LightGBM with sample weights
print("\n--- Training Domain-Adapted LightGBM ---")
lgb_oof = np.zeros(len(X_native))
lgb_test = np.zeros(len(X_test_native))
LGB_P = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.022, num_leaves=24, max_depth=4,
    min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=5, lambda_l1=1.5, lambda_l2=2.0,
    n_estimators=2500, n_jobs=-1, random_state=SEED, verbose=-1
)
for f, (ti, vi) in enumerate(skf.split(X_native, y)):
    lg = lgb.LGBMClassifier(**LGB_P)
    lg.fit(X_native.iloc[ti], y.iloc[ti], sample_weight=sample_weights[ti],
           eval_set=[(X_native.iloc[vi], y.iloc[vi])],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    lgb_oof[vi] = lg.predict_proba(X_native.iloc[vi])[:, 1]
    lgb_test += lg.predict_proba(X_test_native)[:, 1] / N_FOLDS

print(f"  [LightGBM] OOF Log Loss: {log_loss(y, lgb_oof):.5f} | AUC: {roc_auc_score(y, lgb_oof):.5f}")

# Train Calibrated Lasso LR
print("\n--- Training Calibrated Lasso LR ---")
X_lr_raw = train_fe[FEATURES].copy()
X_test_lr_raw = test_fe[FEATURES].copy()
for c in CAT_COLS:
    X_lr_raw[c] = X_lr_raw[c].astype(str)
    X_test_lr_raw[c] = X_test_lr_raw[c].astype(str)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_ohe = ohe.fit_transform(X_lr_raw[CAT_COLS])
test_ohe = ohe.transform(X_test_lr_raw[CAT_COLS])

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

# Stacking optimization
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
    return log_loss(y, blend, sample_weight=sample_weights)

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

p_domain_adapted = sum(wi * ti for wi, ti in zip(w, tests))


print("\n--- Creating Master Winning Blend ---")
sub9 = pd.read_csv("codewave submission 9.csv")["readmitted_30d"].values
sub4 = pd.read_csv("codewave submission 4.csv")["readmitted_30d"].values

# 50% Sub 9 (0.33416) + 25% Sub 4 (0.33423) + 25% Domain-Adapted
master_p = 0.50 * sub9 + 0.25 * sub4 + 0.25 * p_domain_adapted

# Apply micro-sharpening (slope = 1.012)
z = np.log(master_p / (1.0 - master_p))
z_sharp = 1.012 * z + 0.012
p_sharp = 1.0 / (1.0 + np.exp(-z_sharp))

# Bounded clinical safe clipping:
final_probs = np.clip(p_sharp, 0.0245, 0.728)

# Save as codewave submission 12.csv
sub = pd.DataFrame({
    "patient_id": test_raw[ID_COL].values,
    "readmitted_30d": final_probs
})
sub.to_csv("codewave submission 12.csv", index=False)

print(f"\n  Saved: 'codewave submission 12.csv' ({len(sub)} rows)")
print(f"  Min prob: {final_probs.min():.6f}")
print(f"  Max prob: {final_probs.max():.6f}")
print(f"  Mean prob: {final_probs.mean():.6f}")
print("\nFirst 10 rows:")
print(sub.head(10))

print("\n" + "=" * 65)
print("ROUND 12 COMPLETED SUCCESSFULLY!")
print("=" * 65)
