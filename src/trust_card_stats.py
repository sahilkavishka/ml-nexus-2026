import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
import json

SEED = 42
N_FOLDS = 5
TARGET = "readmitted_30d"
ID_COL = "patient_id"

train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")
y = train_raw[TARGET]

CAT_COLS = ["sex", "rurality", "hospital_type", "region", "discharge_disposition", "care_pathway"]
NUM_COLS = ["age", "socioeconomic_index", "prior_admissions_12m", "comorbidity_count",
            "diabetes", "hypertension", "chronic_kidney_disease", "heart_failure",
            "length_of_stay_days", "medication_count", "missed_appointments_12m",
            "followup_days", "hemoglobin_g_dl", "creatinine_mg_dl", "sodium_mmol_l",
            "heart_rate_bpm", "systolic_bp_mmhg"]

def fe(df):
    df = df.copy()
    df["anemia_flag"] = (df["hemoglobin_g_dl"] < 11.0).astype(float)
    df["aki_flag"]    = (df["creatinine_mg_dl"] > 1.5).astype(float)
    df["tachy_flag"]  = (df["heart_rate_bpm"] > 100).astype(float)
    df["htn_bp_flag"] = (df["systolic_bp_mmhg"] > 140).astype(float)
    df["age_group"]   = pd.cut(df["age"], bins=[0,40,60,75,120], labels=["young","middle","senior","elderly"])
    df["age_over65"]  = (df["age"] >= 65).astype(int)
    df["comorbidity_load"] = df["diabetes"]+df["hypertension"]+df["chronic_kidney_disease"]+df["heart_failure"]
    df["cardiorenal"] = (df["chronic_kidney_disease"] & df["heart_failure"]).astype(int)
    df["frequent_admitter"] = (df["prior_admissions_12m"] >= 2).astype(int)
    df["long_stay"] = (df["length_of_stay_days"] >= 7).astype(int)
    df["late_followup"] = ((df["followup_days"] > 14) | df["followup_days"].isnull()).astype(int)
    df["polypharmacy"] = (df["medication_count"] >= 10).astype(int)
    return df

tr = fe(train_raw)
ALL_CATS = CAT_COLS + ["age_group"]
ALL_NUMS = NUM_COLS + ["anemia_flag", "aki_flag", "tachy_flag", "htn_bp_flag",
                       "age_over65", "comorbidity_load", "cardiorenal",
                       "frequent_admitter", "long_stay", "late_followup", "polypharmacy"]
FEATURES = ALL_NUMS + ALL_CATS

# Fast 5-fold evaluation to get exact OOF probabilities
X_lr = tr[FEATURES].copy()
for c in ALL_CATS: X_lr[c] = X_lr[c].astype(str)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_ohe = ohe.fit_transform(X_lr[ALL_CATS])
imp = IterativeImputer(random_state=SEED, max_iter=8, n_nearest_features=6)
X_num = imp.fit_transform(X_lr[ALL_NUMS])
sc = StandardScaler()
X_s = np.hstack([sc.fit_transform(X_num), X_ohe])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_p = np.zeros(len(y))
fold_metrics = {"log_loss": [], "auc": [], "brier": [], "sens": [], "spec": []}

threshold = 0.126 # Operating threshold set to disease prevalence

for ti, vi in skf.split(X_s, y):
    m = CalibratedClassifierCV(LogisticRegression(C=0.15, penalty="l1", solver="saga", max_iter=1500, random_state=SEED), method="sigmoid", cv=3)
    m.fit(X_s[ti], y.iloc[ti])
    p = m.predict_proba(X_s[vi])[:, 1]
    oof_p[vi] = p
    
    ll = log_loss(y.iloc[vi], p)
    auc = roc_auc_score(y.iloc[vi], p)
    br = brier_score_loss(y.iloc[vi], p)
    pred_binary = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.iloc[vi], pred_binary).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    fold_metrics["log_loss"].append(ll)
    fold_metrics["auc"].append(auc)
    fold_metrics["brier"].append(br)
    fold_metrics["sens"].append(sens)
    fold_metrics["spec"].append(spec)

print("=== OVERALL PERFORMANCE (Across 5 Folds) ===")
for k in fold_metrics:
    vals = fold_metrics[k]
    print(f"{k}: Mean = {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# Subgroup Audit
tr["oof_p"] = oof_p
print("\n=== SUBGROUP AUDIT ===")
subgroups = ["sex", "rurality", "age_group", "hospital_type"]
for sg in subgroups:
    print(f"\n--- Subgroup: {sg} ---")
    for val, grp in tr.groupby(sg):
        ll = log_loss(grp[TARGET], grp["oof_p"])
        auc = roc_auc_score(grp[TARGET], grp["oof_p"])
        br = brier_score_loss(grp[TARGET], grp["oof_p"])
        print(f"Group: {val:<12} | N={len(grp):<5} | Prev={grp[TARGET].mean():.3f} | LogLoss={ll:.4f} | AUC={auc:.4f} | Brier={br:.4f}")

# Uncertainty Referral (Abstaining on 10% most uncertain)
# Uncertainty is highest when predicted probability is closest to the 0.5 boundary or highest binary entropy
entropy = - (oof_p * np.log(oof_p + 1e-12) + (1 - oof_p) * np.log(1 - oof_p + 1e-12))
cutoff = np.percentile(entropy, 90)
certain_mask = entropy < cutoff
print("\n=== UNCERTAINTY / HUMAN REFERRAL (Top 10% Most Uncertain Deferred) ===")
print(f"Full 100% Population: N={len(y)} | Log Loss={log_loss(y, oof_p):.4f} | AUC={roc_auc_score(y, oof_p):.4f} | Brier={brier_score_loss(y, oof_p):.4f}")
print(f"Remaining 90% Patients: N={certain_mask.sum()} | Log Loss={log_loss(y[certain_mask], oof_p[certain_mask]):.4f} | AUC={roc_auc_score(y[certain_mask], oof_p[certain_mask]):.4f} | Brier={brier_score_loss(y[certain_mask], oof_p[certain_mask]):.4f}")

# Feature importances / coefficients
base = LogisticRegression(C=0.15, penalty="l1", solver="saga", max_iter=1500, random_state=SEED).fit(X_s, y)
all_feat_names = list(ALL_NUMS) + list(ohe.get_feature_names_out(ALL_CATS))
coefs = pd.Series(base.coef_[0], index=all_feat_names)
top_pos = coefs.nlargest(6)
top_neg = coefs.nsmallest(6)
print("\n=== TOP INFLUENTIAL PREDICTORS ===")
print("Top Positive (Increases Readmission Risk):")
print(top_pos)
print("\nTop Negative (Decreases Readmission Risk):")
print(top_neg)

# High risk vs Low risk examples
high_idx = np.argmax(oof_p)
low_idx = np.argmin(oof_p)
print("\n=== LOCAL EXPLANATION EXAMPLES ===")
print(f"High-Risk Patient ID: {tr.iloc[high_idx][ID_COL]} | Pred Prob: {oof_p[high_idx]:.4f} | Actual: {tr.iloc[high_idx][TARGET]}")
print(tr.iloc[high_idx][["age", "prior_admissions_12m", "comorbidity_count", "heart_failure", "chronic_kidney_disease", "length_of_stay_days", "creatinine_mg_dl"]].to_dict())
print(f"\nLow-Risk Patient ID: {tr.iloc[low_idx][ID_COL]} | Pred Prob: {oof_p[low_idx]:.4f} | Actual: {tr.iloc[low_idx][TARGET]}")
print(tr.iloc[low_idx][["age", "prior_admissions_12m", "comorbidity_count", "heart_failure", "chronic_kidney_disease", "length_of_stay_days", "creatinine_mg_dl"]].to_dict())
