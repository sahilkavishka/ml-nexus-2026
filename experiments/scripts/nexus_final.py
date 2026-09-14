import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)
TARGET = "readmitted_30d"
ID_COL = "patient_id"

print("="*65)
print("FINAL SUBMISSION - MEGA ENSEMBLE WITH MULTI-SEED DIVERSITY")
print("="*65)

train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")
y = train_raw[TARGET]

CAT_COLS = ["sex","rurality","hospital_type","region","discharge_disposition","care_pathway"]
NUM_COLS = ["age","socioeconomic_index","prior_admissions_12m","comorbidity_count",
            "diabetes","hypertension","chronic_kidney_disease","heart_failure",
            "length_of_stay_days","medication_count","missed_appointments_12m",
            "followup_days","hemoglobin_g_dl","creatinine_mg_dl","sodium_mmol_l",
            "heart_rate_bpm","systolic_bp_mmhg"]

def feature_engineer(df):
    df = df.copy()
    df["anemia_flag"]        = (df["hemoglobin_g_dl"] < 11.0).astype(float)
    df["severe_anemia"]      = (df["hemoglobin_g_dl"] < 9.0).astype(float)
    df["aki_flag"]           = (df["creatinine_mg_dl"] > 1.5).astype(float)
    df["severe_aki"]         = (df["creatinine_mg_dl"] > 3.0).astype(float)
    df["hyponatremia_flag"]  = (df["sodium_mmol_l"] < 135).astype(float)
    df["tachy_flag"]         = (df["heart_rate_bpm"] > 100).astype(float)
    df["htn_bp_flag"]        = (df["systolic_bp_mmhg"] > 140).astype(float)
    df["hypo_bp_flag"]       = (df["systolic_bp_mmhg"] < 90).astype(float)
    for c in ["hemoglobin_g_dl","creatinine_mg_dl","sodium_mmol_l","heart_rate_bpm","systolic_bp_mmhg","followup_days"]:
        df[c+"_missing"] = df[c].isnull().astype(int)
    df["n_labs_missing"]     = df["hemoglobin_g_dl"].isnull().astype(int)+df["creatinine_mg_dl"].isnull().astype(int)+df["sodium_mmol_l"].isnull().astype(int)+df["heart_rate_bpm"].isnull().astype(int)+df["systolic_bp_mmhg"].isnull().astype(int)
    df["age_group"]          = pd.cut(df["age"],bins=[0,40,60,75,120],labels=["young","middle","senior","elderly"])
    df["age_over65"]         = (df["age"]>=65).astype(int)
    df["age_over80"]         = (df["age"]>=80).astype(int)
    df["comorbidity_load"]   = df["diabetes"]+df["hypertension"]+df["chronic_kidney_disease"]+df["heart_failure"]
    df["cardiorenal"]        = df["chronic_kidney_disease"] & df["heart_failure"]
    df["diabetic_htn"]       = df["diabetes"] & df["hypertension"]
    df["high_comorbidity"]   = (df["comorbidity_load"]>=3).astype(int)
    df["med_per_comorbidity"]= df["medication_count"]/(df["comorbidity_count"]+1)
    df["polypharmacy"]       = (df["medication_count"]>=10).astype(int)
    df["frequent_admitter"]  = (df["prior_admissions_12m"]>=2).astype(int)
    df["very_frequent"]      = (df["prior_admissions_12m"]>=4).astype(int)
    df["long_stay"]          = (df["length_of_stay_days"]>=7).astype(int)
    df["missed_appt_flag"]   = (df["missed_appointments_12m"]>=2).astype(int)
    df["late_followup"]      = ((df["followup_days"]>14)|df["followup_days"].isnull()).astype(int)
    df["simple_risk_score"]  = df["frequent_admitter"]+df["long_stay"]+df["high_comorbidity"]+df["polypharmacy"]+df["missed_appt_flag"]+df["late_followup"]+df["anemia_flag"]+df["aki_flag"]
    df["low_ses"]            = (df["socioeconomic_index"]<df["socioeconomic_index"].quantile(0.25)).astype(int)
    return df

train_fe = feature_engineer(train_raw)
test_fe  = feature_engineer(test_raw)

NEW_CAT = ["age_group"]
ALL_CATS = CAT_COLS + NEW_CAT
ALL_NUMS = NUM_COLS + ["anemia_flag","severe_anemia","aki_flag","severe_aki","hyponatremia_flag","tachy_flag","htn_bp_flag","hypo_bp_flag",
           "hemoglobin_g_dl_missing","creatinine_mg_dl_missing","sodium_mmol_l_missing","heart_rate_bpm_missing","systolic_bp_mmhg_missing","followup_days_missing",
           "n_labs_missing","age_over65","age_over80","comorbidity_load","cardiorenal","diabetic_htn","high_comorbidity",
           "med_per_comorbidity","polypharmacy","frequent_admitter","very_frequent","long_stay","missed_appt_flag","late_followup","simple_risk_score","low_ses"]

for col in ALL_CATS:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)])
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

FEATURES = ALL_NUMS + ALL_CATS
X = train_fe[FEATURES]; X_test = test_fe[FEATURES]
print(f"Features: {len(FEATURES)}")

print("Imputing...")
imp = IterativeImputer(random_state=SEED, max_iter=10, n_nearest_features=8)
X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEATURES, index=X.index)
X_t   = pd.DataFrame(imp.transform(X_test), columns=FEATURES, index=X_test.index)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_oof = {}
all_test = {}

def run_model(name, model_fn, X_imp, X_t, y, skf):
    oof = np.zeros(len(X_imp)); test_p = np.zeros(len(X_t))
    lls = []
    for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
        m = model_fn(f)
        m.fit(X_imp.iloc[ti], y.iloc[ti])
        p = m.predict_proba(X_imp.iloc[vi])[:,1]
        oof[vi] = p; test_p += m.predict_proba(X_t)[:,1]/N_FOLDS
        lls.append(log_loss(y.iloc[vi],p))
        print(f"  [{name}] Fold {f+1}: LL={lls[-1]:.5f}")
    print(f"  [{name}] Mean LL={np.mean(lls):.5f} AUC={roc_auc_score(y,oof):.5f}")
    return oof, test_p, np.mean(lls)

# === CatBoost Seed A ===
print("\n--- CatBoost (Seed 42) ---")
oof1 = np.zeros(len(X_imp)); t1 = np.zeros(len(X_t)); ll1=[]
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    cb = CatBoostClassifier(iterations=1500,learning_rate=0.03,depth=5,l2_leaf_reg=4,
                            eval_metric="Logloss",random_seed=42+f,early_stopping_rounds=100,verbose=False)
    cb.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=(X_imp.iloc[vi],y.iloc[vi]))
    p=cb.predict_proba(X_imp.iloc[vi])[:,1]
    oof1[vi]=p; t1+=cb.predict_proba(X_t)[:,1]/N_FOLDS
    ll1.append(log_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll1[-1]:.5f}")
print(f"  [CatBoost-A] Mean LL={np.mean(ll1):.5f} AUC={roc_auc_score(y,oof1):.5f}")
all_oof["cb_a"]=oof1; all_test["cb_a"]=t1

# === CatBoost Seed B (different depth+lr) ===
print("\n--- CatBoost (Variant B: depth=6, lr=0.04) ---")
oof2 = np.zeros(len(X_imp)); t2 = np.zeros(len(X_t)); ll2=[]
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    cb = CatBoostClassifier(iterations=1500,learning_rate=0.04,depth=6,l2_leaf_reg=6,
                            eval_metric="Logloss",random_seed=100+f,early_stopping_rounds=100,verbose=False)
    cb.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=(X_imp.iloc[vi],y.iloc[vi]))
    p=cb.predict_proba(X_imp.iloc[vi])[:,1]
    oof2[vi]=p; t2+=cb.predict_proba(X_t)[:,1]/N_FOLDS
    ll2.append(log_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll2[-1]:.5f}")
print(f"  [CatBoost-B] Mean LL={np.mean(ll2):.5f} AUC={roc_auc_score(y,oof2):.5f}")
all_oof["cb_b"]=oof2; all_test["cb_b"]=t2

# === LightGBM A ===
print("\n--- LightGBM (Config A: leaves=24, lr=0.025) ---")
oof3 = np.zeros(len(X_imp)); t3 = np.zeros(len(X_t)); ll3=[]
LGB_A = dict(objective="binary",metric="binary_logloss",learning_rate=0.025,num_leaves=24,max_depth=4,
             min_child_samples=40,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=5,
             lambda_l1=1.5,lambda_l2=2.0,n_estimators=2500,n_jobs=-1,random_state=42,verbose=-1)
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    lg=lgb.LGBMClassifier(**LGB_A)
    lg.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=[(X_imp.iloc[vi],y.iloc[vi])],
           callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(-1)])
    p=lg.predict_proba(X_imp.iloc[vi])[:,1]
    oof3[vi]=p; t3+=lg.predict_proba(X_t)[:,1]/N_FOLDS
    ll3.append(log_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll3[-1]:.5f}")
print(f"  [LGB-A] Mean LL={np.mean(ll3):.5f} AUC={roc_auc_score(y,oof3):.5f}")
all_oof["lgb_a"]=oof3; all_test["lgb_a"]=t3

# === LightGBM B (different config) ===
print("\n--- LightGBM (Config B: leaves=31, lr=0.035) ---")
oof4 = np.zeros(len(X_imp)); t4 = np.zeros(len(X_t)); ll4=[]
LGB_B = dict(objective="binary",metric="binary_logloss",learning_rate=0.035,num_leaves=31,max_depth=5,
             min_child_samples=30,feature_fraction=0.75,bagging_fraction=0.75,bagging_freq=5,
             lambda_l1=1.0,lambda_l2=1.5,n_estimators=2000,n_jobs=-1,random_state=99,verbose=-1)
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    lg=lgb.LGBMClassifier(**LGB_B)
    lg.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=[(X_imp.iloc[vi],y.iloc[vi])],
           callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(-1)])
    p=lg.predict_proba(X_imp.iloc[vi])[:,1]
    oof4[vi]=p; t4+=lg.predict_proba(X_t)[:,1]/N_FOLDS
    ll4.append(log_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll4[-1]:.5f}")
print(f"  [LGB-B] Mean LL={np.mean(ll4):.5f} AUC={roc_auc_score(y,oof4):.5f}")
all_oof["lgb_b"]=oof4; all_test["lgb_b"]=t4

# === XGBoost ===
print("\n--- XGBoost ---")
oof5 = np.zeros(len(X_imp)); t5 = np.zeros(len(X_t)); ll5=[]
XGB_P = dict(objective="binary:logistic",eval_metric="logloss",learning_rate=0.035,max_depth=5,
             min_child_weight=5,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.5,reg_lambda=2.0,
             scale_pos_weight=(y==0).sum()/(y==1).sum(),n_estimators=2000,early_stopping_rounds=100,
             n_jobs=-1,random_state=SEED,verbosity=0)
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    xm=xgb.XGBClassifier(**XGB_P)
    xm.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=[(X_imp.iloc[vi],y.iloc[vi])],verbose=False)
    p=xm.predict_proba(X_imp.iloc[vi])[:,1]
    oof5[vi]=p; t5+=xm.predict_proba(X_t)[:,1]/N_FOLDS
    ll5.append(log_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll5[-1]:.5f}")
print(f"  [XGBoost] Mean LL={np.mean(ll5):.5f} AUC={roc_auc_score(y,oof5):.5f}")
all_oof["xgb"]=oof5; all_test["xgb"]=t5

# === Lasso LR (best from R2) ===
print("\n--- Calibrated Lasso LR ---")
scaler = StandardScaler()
Xs = scaler.fit_transform(X_imp); Xs_t = scaler.transform(X_t)
oof6 = np.zeros(len(Xs)); t6 = np.zeros(len(Xs_t)); ll6=[]
for f,(ti,vi) in enumerate(skf.split(Xs,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.15,penalty="l1",solver="saga",max_iter=2000,random_state=SEED),method="sigmoid",cv=3)
    m.fit(Xs[ti],y.iloc[ti])
    p=m.predict_proba(Xs[vi])[:,1]
    oof6[vi]=p; t6+=m.predict_proba(Xs_t)[:,1]/N_FOLDS
    ll6.append(log_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll6[-1]:.5f}")
print(f"  [LassoLR] Mean LL={np.mean(ll6):.5f} AUC={roc_auc_score(y,oof6):.5f}")
all_oof["lr"]=oof6; all_test["lr"]=t6

# === ElasticNet LR ===
print("\n--- Calibrated ElasticNet LR ---")
oof7 = np.zeros(len(Xs)); t7 = np.zeros(len(Xs_t)); ll7=[]
for f,(ti,vi) in enumerate(skf.split(Xs,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.2,penalty="elasticnet",l1_ratio=0.6,solver="saga",max_iter=2000,random_state=SEED),method="sigmoid",cv=3)
    m.fit(Xs[ti],y.iloc[ti])
    p=m.predict_proba(Xs[vi])[:,1]
    oof7[vi]=p; t7+=m.predict_proba(Xs_t)[:,1]/N_FOLDS
    ll7.append(log_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll7[-1]:.5f}")
print(f"  [ElasticNet] Mean LL={np.mean(ll7):.5f} AUC={roc_auc_score(y,oof7):.5f}")
all_oof["en"]=oof7; all_test["en"]=t7

# ============================================================
# OPTIMAL WEIGHT SEARCH VIA SCIPY (DIFFERENTIAL EVOLUTION)
# ============================================================
print("\n" + "="*65)
print("GLOBAL WEIGHT OPTIMIZATION (Differential Evolution)")
print("="*65)

keys = list(all_oof.keys())
oofs = np.column_stack([all_oof[k] for k in keys])
tests = np.column_stack([all_test[k] for k in keys])

def loss_fn(weights):
    weights = np.abs(weights)
    s = weights.sum()
    if s < 1e-9: return 9.0
    w = weights / s
    blend = oofs @ w
    blend = np.clip(blend, 1e-7, 1-1e-7)
    return log_loss(y, blend)

# Global search via Differential Evolution
print("Running global optimization...")
n = len(keys)
result = differential_evolution(loss_fn, bounds=[(0,1)]*n, seed=SEED, maxiter=2000, popsize=20, tol=1e-7)
w_opt = np.abs(result.x) / np.abs(result.x).sum()
print(f"  Global optimum LL: {result.fun:.5f}")
for k, w in zip(keys, w_opt):
    print(f"    {k}: {w:.4f}")

ens_oof  = oofs  @ w_opt
ens_test = tests @ w_opt

print(f"\n  Ensemble OOF LL:  {log_loss(y, ens_oof):.5f}")
print(f"  Ensemble OOF AUC: {roc_auc_score(y, ens_oof):.5f}")
print(f"  Ensemble OOF Brier: {brier_score_loss(y, ens_oof):.5f}")

# ============================================================
# CALIBRATION: ISOTONIC ON OOF + PRIOR LOG-ODDS SHIFT
# ============================================================
print("\n" + "="*65)
print("CALIBRATION")
print("="*65)

# Isotonic on OOF
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(ens_oof, y)
iso_oof  = iso.predict(ens_oof)
iso_test = iso.predict(ens_test)
print(f"  After Isotonic - OOF LL: {log_loss(y, iso_oof):.5f}")

# Prior log-odds shift to align test mean to training prevalence
train_prev = float(y.mean())

def shift_logodds(p, target_mean):
    lo = np.log(p / (1-p))
    cur_mean = p.mean()
    delta = np.log((target_mean/(1-target_mean)) / (cur_mean/(1-cur_mean)))
    new_lo = lo + delta
    return 1.0 / (1.0 + np.exp(-new_lo))

# Try the shift for OOF as well - pick best
raw_ll = log_loss(y, ens_oof)
iso_ll = log_loss(y, iso_oof)

# Shift isotonic test predictions
final_test_calib = shift_logodds(iso_test, train_prev)
print(f"  After prior shift - Test mean: {final_test_calib.mean():.6f} (target: {train_prev:.6f})")

# Also try just raw ensemble with prior shift  
raw_shifted = shift_logodds(ens_oof, train_prev)
raw_shifted_ll = log_loss(y, raw_shifted)
iso_shifted_ll = log_loss(y, shift_logodds(iso_oof, train_prev))
print(f"  Raw LL: {raw_ll:.5f}")
print(f"  Isotonic LL: {iso_ll:.5f}")
print(f"  Raw+Shift OOF LL: {raw_shifted_ll:.5f}")
print(f"  Iso+Shift OOF LL: {iso_shifted_ll:.5f}")

# Use best calibration approach
options = {
    "raw": (ens_test, raw_ll),
    "isotonic": (iso_test, iso_ll),
    "raw+shift_test": (shift_logodds(ens_test, train_prev), raw_shifted_ll),
    "iso+shift_test": (final_test_calib, iso_shifted_ll),
}
best_key = min(options, key=lambda k: options[k][1])
final_probs = options[best_key][0]
print(f"\n  >>> Best calibration: {best_key} (OOF proxy LL={options[best_key][1]:.5f})")

# Safety clip: no probabilities at extremes
final_probs = np.clip(final_probs, 0.015, 0.80)

print(f"\n  Final Test mean: {final_probs.mean():.6f}")
print(f"  Final Test range: [{final_probs.min():.4f}, {final_probs.max():.4f}]")

# Save
sub = pd.DataFrame({"patient_id": test_raw[ID_COL].values, "readmitted_30d": final_probs})
sub.to_csv("codewave final submission.csv", index=False)
print(f"\n  Saved: 'codewave final submission.csv' ({len(sub)} rows)")
print(sub.head(10))

print("\n" + "="*65)
print("FINAL PIPELINE COMPLETE!")
print("="*65)
