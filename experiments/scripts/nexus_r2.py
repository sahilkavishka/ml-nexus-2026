import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import lightgbm as lgb

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)
TARGET = "readmitted_30d"
ID_COL = "patient_id"

print("="*65)
print("ROUND 2: LR-OPTIMIZED PIPELINE")
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
    # Clinical abnormality flags
    df["anemia_flag"]       = (df["hemoglobin_g_dl"] < 11.0).astype(float)
    df["severe_anemia"]     = (df["hemoglobin_g_dl"] < 9.0).astype(float)
    df["aki_flag"]          = (df["creatinine_mg_dl"] > 1.5).astype(float)
    df["severe_aki"]        = (df["creatinine_mg_dl"] > 3.0).astype(float)
    df["hyponatremia_flag"] = (df["sodium_mmol_l"] < 135).astype(float)
    df["severe_hyponatremia"]=(df["sodium_mmol_l"] < 130).astype(float)
    df["tachy_flag"]        = (df["heart_rate_bpm"] > 100).astype(float)
    df["severe_tachy"]      = (df["heart_rate_bpm"] > 120).astype(float)
    df["htn_bp_flag"]       = (df["systolic_bp_mmhg"] > 140).astype(float)
    df["hypo_bp_flag"]      = (df["systolic_bp_mmhg"] < 90).astype(float)
    # Missing flags
    for c in ["hemoglobin_g_dl","creatinine_mg_dl","sodium_mmol_l","heart_rate_bpm","systolic_bp_mmhg","followup_days"]:
        df[c+"_missing"] = df[c].isnull().astype(int)
    # Count missing labs
    df["n_labs_missing"] = (df["hemoglobin_g_dl"].isnull().astype(int) +
                            df["creatinine_mg_dl"].isnull().astype(int) +
                            df["sodium_mmol_l"].isnull().astype(int) +
                            df["heart_rate_bpm"].isnull().astype(int) +
                            df["systolic_bp_mmhg"].isnull().astype(int))
    # Age groups & risk
    df["age_group"] = pd.cut(df["age"],bins=[0,40,60,75,120],labels=["young","middle","senior","elderly"])
    df["age_over65"]          = (df["age"] >= 65).astype(int)
    df["age_over80"]          = (df["age"] >= 80).astype(int)
    # Comorbidity features
    df["comorbidity_load"]    = df["diabetes"]+df["hypertension"]+df["chronic_kidney_disease"]+df["heart_failure"]
    df["cardiorenal"]         = df["chronic_kidney_disease"] & df["heart_failure"]
    df["diabetic_hypertensive"]= df["diabetes"] & df["hypertension"]
    df["high_comorbidity"]    = (df["comorbidity_load"] >= 3).astype(int)
    # Medication features
    df["med_per_comorbidity"] = df["medication_count"]/(df["comorbidity_count"]+1)
    df["polypharmacy"]        = (df["medication_count"] >= 10).astype(int)
    df["high_medication"]     = (df["medication_count"] >= 15).astype(int)
    # Admission history
    df["frequent_admitter"]   = (df["prior_admissions_12m"] >= 2).astype(int)
    df["very_frequent"]       = (df["prior_admissions_12m"] >= 4).astype(int)
    df["long_stay"]           = (df["length_of_stay_days"] >= 7).astype(int)
    df["very_long_stay"]      = (df["length_of_stay_days"] >= 14).astype(int)
    df["missed_appt_flag"]    = (df["missed_appointments_12m"] >= 2).astype(int)
    df["late_followup"]       = ((df["followup_days"]>14)|df["followup_days"].isnull()).astype(int)
    df["very_late_followup"]  = ((df["followup_days"]>30)|df["followup_days"].isnull()).astype(int)
    # Composite risk score (simple additive)
    df["simple_risk_score"]   = (df["frequent_admitter"] + df["long_stay"] +
                                  df["high_comorbidity"] + df["polypharmacy"] +
                                  df["missed_appt_flag"] + df["late_followup"] +
                                  df["anemia_flag"] + df["aki_flag"])
    # Socioeconomic interactions
    df["low_ses"] = (df["socioeconomic_index"] < df["socioeconomic_index"].quantile(0.25)).astype(int)
    df["ses_x_rural"] = df["low_ses"] * (df["rurality"].astype(str)=="Rural").astype(int)
    return df

train_fe = feature_engineer(train_raw)
test_fe  = feature_engineer(test_raw)

NEW_CAT = ["age_group"]
NEW_NUM = ["anemia_flag","severe_anemia","aki_flag","severe_aki","hyponatremia_flag",
           "severe_hyponatremia","tachy_flag","severe_tachy","htn_bp_flag","hypo_bp_flag",
           "hemoglobin_g_dl_missing","creatinine_mg_dl_missing","sodium_mmol_l_missing",
           "heart_rate_bpm_missing","systolic_bp_mmhg_missing","followup_days_missing",
           "n_labs_missing","age_over65","age_over80",
           "comorbidity_load","cardiorenal","diabetic_hypertensive","high_comorbidity",
           "med_per_comorbidity","polypharmacy","high_medication",
           "frequent_admitter","very_frequent","long_stay","very_long_stay",
           "missed_appt_flag","late_followup","very_late_followup",
           "simple_risk_score","low_ses","ses_x_rural"]

ALL_CATS = CAT_COLS + NEW_CAT
ALL_NUMS = NUM_COLS + NEW_NUM

for col in ALL_CATS:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col].astype(str), test_fe[col].astype(str)])
    le.fit(combined)
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

FEATURES = ALL_NUMS + ALL_CATS
X      = train_fe[FEATURES]
y      = train_fe[TARGET]
X_test = test_fe[FEATURES]
print(f"Total features after FE: {len(FEATURES)}")

# Impute
print("Imputing...")
imputer    = IterativeImputer(random_state=SEED, max_iter=10, n_nearest_features=8)
X_imp      = pd.DataFrame(imputer.fit_transform(X),      columns=FEATURES, index=X.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test),     columns=FEATURES, index=X_test.index)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("\n" + "="*65)
print("TRAINING LR VARIANTS + LGBM")
print("="*65)

oof_preds  = {}
test_preds = {}
cv_metrics = {}

def show(name, lls, aucs, brs):
    print(f"  [{name}]  LL={np.mean(lls):.5f}(+-{np.std(lls):.5f})  AUC={np.mean(aucs):.5f}  Brier={np.mean(brs):.5f}")

scaler = StandardScaler()
Xs   = scaler.fit_transform(X_imp)
Xs_t = scaler.transform(X_test_imp)

# ── Variant A: LR L2 (C=0.05 tight regularization) ───────────────────────────
print("\n--- LR-L2 C=0.05 (tight) ---")
o=np.zeros(len(Xs)); t=np.zeros(len(Xs_t)); ll_=[]; auc_=[]; br_=[]
for f,(ti,vi) in enumerate(skf.split(Xs,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.05,class_weight="balanced",max_iter=2000,random_state=SEED,solver="lbfgs"),method="isotonic",cv=3)
    m.fit(Xs[ti],y.iloc[ti])
    p=m.predict_proba(Xs[vi])[:,1]
    o[vi]=p; t+=m.predict_proba(Xs_t)[:,1]/N_FOLDS
    ll_.append(log_loss(y.iloc[vi],p)); auc_.append(roc_auc_score(y.iloc[vi],p)); br_.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_[-1]:.5f}  AUC={auc_[-1]:.5f}")
show("LR-L2-C0.05",ll_,auc_,br_)
oof_preds["lr_tight"]=o; test_preds["lr_tight"]=t
cv_metrics["lr_tight"]={"logloss":np.mean(ll_),"auc":np.mean(auc_),"brier":np.mean(br_)}

# ── Variant B: LR L2 (C=0.3) ─────────────────────────────────────────────────
print("\n--- LR-L2 C=0.3 ---")
o=np.zeros(len(Xs)); t=np.zeros(len(Xs_t)); ll_=[]; auc_=[]; br_=[]
for f,(ti,vi) in enumerate(skf.split(Xs,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.3,class_weight="balanced",max_iter=2000,random_state=SEED,solver="lbfgs"),method="isotonic",cv=3)
    m.fit(Xs[ti],y.iloc[ti])
    p=m.predict_proba(Xs[vi])[:,1]
    o[vi]=p; t+=m.predict_proba(Xs_t)[:,1]/N_FOLDS
    ll_.append(log_loss(y.iloc[vi],p)); auc_.append(roc_auc_score(y.iloc[vi],p)); br_.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_[-1]:.5f}  AUC={auc_[-1]:.5f}")
show("LR-L2-C0.3",ll_,auc_,br_)
oof_preds["lr_mid"]=o; test_preds["lr_mid"]=t
cv_metrics["lr_mid"]={"logloss":np.mean(ll_),"auc":np.mean(auc_),"brier":np.mean(br_)}

# ── Variant C: LR ElasticNet ──────────────────────────────────────────────────
print("\n--- LR ElasticNet (l1_ratio=0.5) ---")
o=np.zeros(len(Xs)); t=np.zeros(len(Xs_t)); ll_=[]; auc_=[]; br_=[]
for f,(ti,vi) in enumerate(skf.split(Xs,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.2,penalty="elasticnet",l1_ratio=0.5,class_weight="balanced",max_iter=2000,random_state=SEED,solver="saga"),method="isotonic",cv=3)
    m.fit(Xs[ti],y.iloc[ti])
    p=m.predict_proba(Xs[vi])[:,1]
    o[vi]=p; t+=m.predict_proba(Xs_t)[:,1]/N_FOLDS
    ll_.append(log_loss(y.iloc[vi],p)); auc_.append(roc_auc_score(y.iloc[vi],p)); br_.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_[-1]:.5f}  AUC={auc_[-1]:.5f}")
show("LR-ElasticNet",ll_,auc_,br_)
oof_preds["lr_en"]=o; test_preds["lr_en"]=t
cv_metrics["lr_en"]={"logloss":np.mean(ll_),"auc":np.mean(auc_),"brier":np.mean(br_)}

# ── Variant D: LR L1 (Lasso - sparse) ────────────────────────────────────────
print("\n--- LR L1 C=0.2 (Lasso/sparse) ---")
o=np.zeros(len(Xs)); t=np.zeros(len(Xs_t)); ll_=[]; auc_=[]; br_=[]
for f,(ti,vi) in enumerate(skf.split(Xs,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.2,penalty="l1",class_weight="balanced",max_iter=2000,random_state=SEED,solver="saga"),method="isotonic",cv=3)
    m.fit(Xs[ti],y.iloc[ti])
    p=m.predict_proba(Xs[vi])[:,1]
    o[vi]=p; t+=m.predict_proba(Xs_t)[:,1]/N_FOLDS
    ll_.append(log_loss(y.iloc[vi],p)); auc_.append(roc_auc_score(y.iloc[vi],p)); br_.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_[-1]:.5f}  AUC={auc_[-1]:.5f}")
show("LR-L1",ll_,auc_,br_)
oof_preds["lr_l1"]=o; test_preds["lr_l1"]=t
cv_metrics["lr_l1"]={"logloss":np.mean(ll_),"auc":np.mean(auc_),"brier":np.mean(br_)}

# ── Variant E: LR with 2-way Polynomial (select top features first) ───────────
print("\n--- LR + Polynomial Features (degree=2, top 30 LGB features) ---")
# First get feature importances from LightGBM to select top features
lgb_sel = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.1,num_leaves=31,
                               n_jobs=-1,random_state=SEED,verbose=-1)
lgb_sel.fit(X_imp, y)
imp = pd.Series(lgb_sel.feature_importances_, index=FEATURES)
top30 = imp.nlargest(30).index.tolist()
print(f"  Top 30 features selected for polynomial expansion")

Xs_top   = scaler.fit_transform(X_imp[top30])
Xs_t_top = scaler.transform(X_test_imp[top30])
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
Xp   = poly.fit_transform(Xs_top)
Xp_t = poly.transform(Xs_t_top)
print(f"  Polynomial feature count: {Xp.shape[1]}")

# Re-scale
scaler2 = StandardScaler()
Xp   = scaler2.fit_transform(Xp)
Xp_t = scaler2.transform(Xp_t)

o=np.zeros(len(Xp)); t=np.zeros(len(Xp_t)); ll_=[]; auc_=[]; br_=[]
for f,(ti,vi) in enumerate(skf.split(Xp,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.05,class_weight="balanced",max_iter=2000,random_state=SEED,solver="lbfgs"),method="isotonic",cv=3)
    m.fit(Xp[ti],y.iloc[ti])
    p=m.predict_proba(Xp[vi])[:,1]
    o[vi]=p; t+=m.predict_proba(Xp_t)[:,1]/N_FOLDS
    ll_.append(log_loss(y.iloc[vi],p)); auc_.append(roc_auc_score(y.iloc[vi],p)); br_.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_[-1]:.5f}  AUC={auc_[-1]:.5f}")
show("LR-Poly2",ll_,auc_,br_)
oof_preds["lr_poly"]=o; test_preds["lr_poly"]=t
cv_metrics["lr_poly"]={"logloss":np.mean(ll_),"auc":np.mean(auc_),"brier":np.mean(br_)}

# ── Variant F: LightGBM with more tuned hyperparams ──────────────────────────
print("\n--- LightGBM (Tuned, more leaves) ---")
LGB_P = dict(objective="binary",metric="binary_logloss",learning_rate=0.03,
             num_leaves=31,max_depth=5,min_child_samples=50,
             feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=5,
             lambda_l1=1.0,lambda_l2=1.0,min_gain_to_split=0.01,
             n_estimators=3000,n_jobs=-1,random_state=SEED,verbose=-1)
POS_WEIGHT = (y==0).sum()/(y==1).sum()
o=np.zeros(len(X_imp)); t=np.zeros(len(X_test_imp)); ll_=[]; auc_=[]; br_=[]
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    m=lgb.LGBMClassifier(**LGB_P)
    m.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=[(X_imp.iloc[vi],y.iloc[vi])],
          callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(-1)])
    p=m.predict_proba(X_imp.iloc[vi])[:,1]
    o[vi]=p; t+=m.predict_proba(X_test_imp)[:,1]/N_FOLDS
    ll_.append(log_loss(y.iloc[vi],p)); auc_.append(roc_auc_score(y.iloc[vi],p)); br_.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_[-1]:.5f}  AUC={auc_[-1]:.5f}")
show("LightGBM-Tuned",ll_,auc_,br_)
oof_preds["lgb"]=o; test_preds["lgb"]=t
cv_metrics["lgb"]={"logloss":np.mean(ll_),"auc":np.mean(auc_),"brier":np.mean(br_)}

# ── ENSEMBLE ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("OPTIMAL ENSEMBLE")
print("="*65)
inv={k:1.0/cv_metrics[k]["logloss"] for k in cv_metrics}
tot=sum(inv.values())
W={k:v/tot for k,v in inv.items()}
print("  Weights:", {k:round(v,4) for k,v in W.items()})
ens_oof  = sum(W[k]*oof_preds[k] for k in W)
ens_test = sum(W[k]*test_preds[k] for k in W)
ens_ll=log_loss(y,ens_oof); ens_auc=roc_auc_score(y,ens_oof); ens_br=brier_score_loss(y,ens_oof)
print(f"  Ensemble OOF: LL={ens_ll:.5f}  AUC={ens_auc:.5f}  Brier={ens_br:.5f}")

# ── CALIBRATION ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("CALIBRATION")
print("="*65)
iso=IsotonicRegression(out_of_bounds="clip")
iso.fit(ens_oof,y)
iso_oof=iso.predict(ens_oof); iso_test=iso.predict(ens_test)

platt=LogisticRegression()
platt.fit(ens_oof.reshape(-1,1),y)
platt_oof=platt.predict_proba(ens_oof.reshape(-1,1))[:,1]
platt_test=platt.predict_proba(ens_test.reshape(-1,1))[:,1]

ll_raw=log_loss(y,ens_oof); ll_iso=log_loss(y,iso_oof); ll_platt=log_loss(y,platt_oof)
print(f"  Raw Ensemble  LL: {ll_raw:.5f}")
print(f"  Isotonic      LL: {ll_iso:.5f}")
print(f"  Platt         LL: {ll_platt:.5f}")

# Also try: best single model (find the one with lowest OOF LL)
best_single_key = min(cv_metrics, key=lambda k: cv_metrics[k]["logloss"])
best_single_ll  = cv_metrics[best_single_key]["logloss"]
print(f"  Best single ({best_single_key}): LL={best_single_ll:.5f}")

# Calibrate best single model OOF
iso2=IsotonicRegression(out_of_bounds="clip")
iso2.fit(oof_preds[best_single_key],y)
bsm_oof_cal = iso2.predict(oof_preds[best_single_key])
bsm_test_cal= iso2.predict(test_preds[best_single_key])
ll_bsm_cal  = log_loss(y,bsm_oof_cal)
print(f"  Best single calibrated LL: {ll_bsm_cal:.5f}")

options = {
    "raw_ensemble": (ens_test, ll_raw),
    "iso_ensemble": (iso_test, ll_iso),
    "platt_ensemble": (platt_test, ll_platt),
    "best_single_cal": (bsm_test_cal, ll_bsm_cal),
}
chosen_key = min(options, key=lambda k: options[k][1])
final = options[chosen_key][0]
print(f"\n  >>> Selected: {chosen_key}  (LL={options[chosen_key][1]:.5f})")
final = np.clip(final, 1e-5, 1-1e-5)

# ── SUBMISSION ────────────────────────────────────────────────────────────────
sub=pd.DataFrame({"patient_id":test_raw[ID_COL].values,"readmitted_30d":final})
sub.to_csv("submission_r2.csv",index=False)
print(f"\n  Saved submission_r2.csv ({len(sub)} rows)")
print(f"  Prob range: [{final.min():.6f}, {final.max():.6f}]")
print(f"  Mean pred: {final.mean():.6f}  Train mean: {y.mean():.6f}")
print(sub.head(10))

print("\n" + "="*65)
print("FINAL COMPARISON")
print("="*65)
header = f"{'Model':<25} {'LogLoss':>10} {'AUC':>10} {'Brier':>10}"
print(header)
print("-"*60)
for k,m in sorted(cv_metrics.items(), key=lambda x: x[1]["logloss"]):
    flag = " <-- BEST" if k == min(cv_metrics, key=lambda x: cv_metrics[x]["logloss"]) else ""
    print(f"{k:<25} {m['logloss']:>10.5f} {m['auc']:>10.5f} {m['brier']:>10.5f}{flag}")
print(f"{'Ensemble':25} {ens_ll:>10.5f} {ens_auc:>10.5f} {ens_br:>10.5f}")
print(f"\nFinal selected: {chosen_key}  LL={options[chosen_key][1]:.5f}")
print("DONE!")
