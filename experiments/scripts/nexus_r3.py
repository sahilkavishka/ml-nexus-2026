import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from catboost import CatBoostClassifier
import lightgbm as lgb

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)
TARGET = "readmitted_30d"
ID_COL = "patient_id"

print("="*65)
print("ROUND 3: CATBOOST + LIGHTGBM + LASSO-LR OPTIMIZED ENSEMBLE")
print("="*65)

train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")

CAT_COLS = ["sex","rurality","hospital_type","region","discharge_disposition","care_pathway"]
NUM_COLS = ["age","socioeconomic_index","prior_admissions_12m","comorbidity_count",
            "diabetes","hypertension","chronic_kidney_disease","heart_failure",
            "length_of_stay_days","medication_count","missed_appointments_12m",
            "followup_days","hemoglobin_g_dl","creatinine_mg_dl","sodium_mmol_l",
            "heart_rate_bpm","systolic_bp_mmhg"]

def feature_engineer(df):
    df = df.copy()
    # Clinical flags
    df["anemia_flag"]        = (df["hemoglobin_g_dl"] < 11.0).astype(float)
    df["severe_anemia"]      = (df["hemoglobin_g_dl"] < 9.0).astype(float)
    df["aki_flag"]           = (df["creatinine_mg_dl"] > 1.5).astype(float)
    df["severe_aki"]         = (df["creatinine_mg_dl"] > 3.0).astype(float)
    df["hyponatremia_flag"]  = (df["sodium_mmol_l"] < 135).astype(float)
    df["tachy_flag"]         = (df["heart_rate_bpm"] > 100).astype(float)
    df["htn_bp_flag"]        = (df["systolic_bp_mmhg"] > 140).astype(float)
    # Missingness
    for c in ["hemoglobin_g_dl","creatinine_mg_dl","sodium_mmol_l","heart_rate_bpm","systolic_bp_mmhg","followup_days"]:
        df[c+"_missing"] = df[c].isnull().astype(int)
    df["n_labs_missing"]     = (df["hemoglobin_g_dl"].isnull().astype(int) +
                                df["creatinine_mg_dl"].isnull().astype(int) +
                                df["sodium_mmol_l"].isnull().astype(int) +
                                df["heart_rate_bpm"].isnull().astype(int) +
                                df["systolic_bp_mmhg"].isnull().astype(int))
    # Age & Risk
    df["age_group"]          = pd.cut(df["age"], bins=[0,40,60,75,120], labels=["young","middle","senior","elderly"])
    df["age_over65"]         = (df["age"] >= 65).astype(int)
    # Comorbidity interactions
    df["comorbidity_load"]   = df["diabetes"]+df["hypertension"]+df["chronic_kidney_disease"]+df["heart_failure"]
    df["cardiorenal"]        = df["chronic_kidney_disease"] & df["heart_failure"]
    df["diabetic_hypertensive"]= df["diabetes"] & df["hypertension"]
    df["high_comorbidity"]   = (df["comorbidity_load"] >= 3).astype(int)
    # Medication
    df["med_per_comorbidity"]= df["medication_count"]/(df["comorbidity_count"]+1)
    df["polypharmacy"]       = (df["medication_count"] >= 10).astype(int)
    # Hospitalization
    df["frequent_admitter"]  = (df["prior_admissions_12m"] >= 2).astype(int)
    df["very_frequent"]      = (df["prior_admissions_12m"] >= 4).astype(int)
    df["long_stay"]          = (df["length_of_stay_days"] >= 7).astype(int)
    df["missed_appt_flag"]   = (df["missed_appointments_12m"] >= 2).astype(int)
    df["late_followup"]      = ((df["followup_days"]>14)|df["followup_days"].isnull()).astype(int)
    df["simple_risk_score"]  = (df["frequent_admitter"] + df["long_stay"] +
                                df["high_comorbidity"] + df["polypharmacy"] +
                                df["missed_appt_flag"] + df["late_followup"] +
                                df["anemia_flag"] + df["aki_flag"])
    return df

train_fe = feature_engineer(train_raw)
test_fe  = feature_engineer(test_raw)

NEW_CAT = ["age_group"]
NEW_NUM = ["anemia_flag","severe_anemia","aki_flag","severe_aki","hyponatremia_flag",
           "tachy_flag","htn_bp_flag",
           "hemoglobin_g_dl_missing","creatinine_mg_dl_missing","sodium_mmol_l_missing",
           "heart_rate_bpm_missing","systolic_bp_mmhg_missing","followup_days_missing",
           "n_labs_missing","age_over65",
           "comorbidity_load","cardiorenal","diabetic_hypertensive","high_comorbidity",
           "med_per_comorbidity","polypharmacy",
           "frequent_admitter","very_frequent","long_stay",
           "missed_appt_flag","late_followup","simple_risk_score"]

ALL_CATS = CAT_COLS + NEW_CAT
ALL_NUMS = NUM_COLS + NEW_NUM

# Label encode for tree models & LR
for col in ALL_CATS:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)])
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

FEATURES = ALL_NUMS + ALL_CATS
X = train_fe[FEATURES]
y = train_fe[TARGET]
X_test = test_fe[FEATURES]

print(f"Features: {len(FEATURES)}")

# Impute
print("Imputing missing values...")
imputer    = IterativeImputer(random_state=SEED, max_iter=10, n_nearest_features=8)
X_imp      = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES, index=X.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
POS_WEIGHT = (y==0).sum() / (y==1).sum()

# ── Model 1: CatBoost ─────────────────────────────────────────────────────────
print("\n--- Training CatBoost ---")
cat_oof  = np.zeros(len(X_imp))
cat_test = np.zeros(len(X_test_imp))
cat_ll, cat_auc = [], []

for f, (ti, vi) in enumerate(skf.split(X_imp, y)):
    cb = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=5,
        l2_leaf_reg=4, eval_metric="Logloss",
        random_seed=SEED + f, early_stopping_rounds=100,
        verbose=False
    )
    cb.fit(X_imp.iloc[ti], y.iloc[ti], eval_set=(X_imp.iloc[vi], y.iloc[vi]))
    p = cb.predict_proba(X_imp.iloc[vi])[:, 1]
    cat_oof[vi] = p
    cat_test += cb.predict_proba(X_test_imp)[:, 1] / N_FOLDS
    cat_ll.append(log_loss(y.iloc[vi], p))
    cat_auc.append(roc_auc_score(y.iloc[vi], p))
    print(f"  Fold {f+1}: LL={cat_ll[-1]:.5f}  AUC={cat_auc[-1]:.5f}")

print(f"  [CatBoost]  Mean LL={np.mean(cat_ll):.5f}  AUC={np.mean(cat_auc):.5f}")

# ── Model 2: Tuned LightGBM ───────────────────────────────────────────────────
print("\n--- Training Tuned LightGBM ---")
lgb_oof  = np.zeros(len(X_imp))
lgb_test = np.zeros(len(X_test_imp))
lgb_ll, lgb_auc = [], []

LGB_P = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.025, num_leaves=24, max_depth=4,
    min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=5, lambda_l1=1.5, lambda_l2=2.0,
    n_estimators=2500, n_jobs=-1, random_state=SEED, verbose=-1
)

for f, (ti, vi) in enumerate(skf.split(X_imp, y)):
    lg = lgb.LGBMClassifier(**LGB_P)
    lg.fit(X_imp.iloc[ti], y.iloc[ti], eval_set=[(X_imp.iloc[vi], y.iloc[vi])],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    p = lg.predict_proba(X_imp.iloc[vi])[:, 1]
    lgb_oof[vi] = p
    lgb_test += lg.predict_proba(X_test_imp)[:, 1] / N_FOLDS
    lgb_ll.append(log_loss(y.iloc[vi], p))
    lgb_auc.append(roc_auc_score(y.iloc[vi], p))
    print(f"  Fold {f+1}: LL={lgb_ll[-1]:.5f}  AUC={lgb_auc[-1]:.5f}")

print(f"  [LightGBM]  Mean LL={np.mean(lgb_ll):.5f}  AUC={np.mean(lgb_auc):.5f}")

# ── Model 3: Calibrated Lasso Logistic Regression ─────────────────────────────
print("\n--- Training Calibrated Lasso LR ---")
scaler = StandardScaler()
Xs   = scaler.fit_transform(X_imp)
Xs_t = scaler.transform(X_test_imp)

lr_oof  = np.zeros(len(Xs))
lr_test = np.zeros(len(Xs_t))
lr_ll, lr_auc = [], []

for f, (ti, vi) in enumerate(skf.split(Xs, y)):
    base_lr = LogisticRegression(C=0.15, penalty="l1", solver="saga", max_iter=2000, random_state=SEED)
    m = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    m.fit(Xs[ti], y.iloc[ti])
    p = m.predict_proba(Xs[vi])[:, 1]
    lr_oof[vi] = p
    lr_test += m.predict_proba(Xs_t)[:, 1] / N_FOLDS
    lr_ll.append(log_loss(y.iloc[vi], p))
    lr_auc.append(roc_auc_score(y.iloc[vi], p))
    print(f"  Fold {f+1}: LL={lr_ll[-1]:.5f}  AUC={lr_auc[-1]:.5f}")

print(f"  [Lasso-LR]  Mean LL={np.mean(lr_ll):.5f}  AUC={np.mean(lr_auc):.5f}")

# ── OPTIMAL ENSEMBLE WEIGHTS VIA SCIPY OPTIMIZE ──────────────────────────────
print("\n" + "="*65)
print("OPTIMIZING ENSEMBLE WEIGHTS (MINIMIZING LOG LOSS)")
print("="*65)

# Objective function to minimize Log Loss directly
def loss_func(weights):
    w1, w2, w3 = weights
    w_sum = w1 + w2 + w3
    if w_sum <= 0:
        return 999.0
    w1, w2, w3 = w1/w_sum, w2/w_sum, w3/w_sum
    blend = w1 * cat_oof + w2 * lgb_oof + w3 * lr_oof
    blend = np.clip(blend, 1e-6, 1 - 1e-6)
    return log_loss(y, blend)

res = minimize(loss_func, [1/3, 1/3, 1/3], bounds=[(0,1), (0,1), (0,1)], method="SLSQP")
w = res.x / res.x.sum()
print(f"  Optimal Weights -> CatBoost: {w[0]:.4f}, LightGBM: {w[1]:.4f}, Lasso-LR: {w[2]:.4f}")

opt_oof  = w[0]*cat_oof  + w[1]*lgb_oof  + w[2]*lr_oof
opt_test = w[0]*cat_test + w[1]*lgb_test + w[2]*lr_test
print(f"  Optimized Ensemble OOF Log Loss: {log_loss(y, opt_oof):.5f}")
print(f"  Optimized Ensemble OOF ROC-AUC : {roc_auc_score(y, opt_oof):.5f}")

# ── PRIOR CALIBRATION (ODDS-RATIO SHIFT) ──────────────────────────────────────
# Align the prediction mean to the exact training prevalence (0.125857)
train_prev = y.mean()
test_mean  = opt_test.mean()
print(f"\n  Test predicted mean: {test_mean:.6f} | Target train prevalence: {train_prev:.6f}")

# Shift in log-odds space to match prevalence perfectly
def shift_log_odds(p, current_mean, target_mean):
    # Log-odds
    lo = np.log(p / (1.0 - p))
    # Shift intercept
    delta = np.log((target_mean / (1.0 - target_mean)) / (current_mean / (1.0 - current_mean)))
    new_lo = lo + delta
    return 1.0 / (1.0 + np.exp(-new_lo))

calibrated_test = shift_log_odds(opt_test, test_mean, train_prev)
print(f"  Calibrated Test mean after prior shift: {calibrated_test.mean():.6f}")

# Safety bounds: clip to [0.015, 0.78]
final_probs = np.clip(calibrated_test, 0.015, 0.78)

# ── SAVE SUBMISSION ───────────────────────────────────────────────────────────
sub = pd.DataFrame({
    "patient_id": test_raw[ID_COL].values,
    "readmitted_30d": final_probs
})
sub.to_csv("codewave submission 3.csv", index=False)
print(f"\n  Saved: 'codewave submission 3.csv' ({len(sub)} rows)")
print(f"  Min prob: {final_probs.min():.6f}, Max prob: {final_probs.max():.6f}")
print(f"  Mean prob: {final_probs.mean():.6f}")
print(sub.head(10))

print("\n" + "="*65)
print("ROUND 3 COMPLETE!")
print("="*65)
