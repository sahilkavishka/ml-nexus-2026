import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[WARNING] CatBoost not installed - skipping.")

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

TARGET = "readmitted_30d"
ID_COL = "patient_id"

print("="*65)
print("STEP 1 - LOADING DATA")
print("="*65)
train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")
print(f"Train: {train_raw.shape}   Test: {test_raw.shape}")
print(train_raw[TARGET].value_counts(normalize=True).round(4))

print("\n" + "="*65)
print("STEP 2 - FEATURE ENGINEERING & PREPROCESSING")
print("="*65)

CAT_COLS = ["sex","rurality","hospital_type","region","discharge_disposition","care_pathway"]
NUM_COLS = ["age","socioeconomic_index","prior_admissions_12m","comorbidity_count",
            "diabetes","hypertension","chronic_kidney_disease","heart_failure",
            "length_of_stay_days","medication_count","missed_appointments_12m",
            "followup_days","hemoglobin_g_dl","creatinine_mg_dl","sodium_mmol_l",
            "heart_rate_bpm","systolic_bp_mmhg"]

def feature_engineer(df):
    df = df.copy()
    df["anemia_flag"]       = (df["hemoglobin_g_dl"] < 11.0).astype(float)
    df["aki_flag"]          = (df["creatinine_mg_dl"] > 1.5).astype(float)
    df["hyponatremia_flag"] = (df["sodium_mmol_l"] < 135).astype(float)
    df["tachy_flag"]        = (df["heart_rate_bpm"] > 100).astype(float)
    df["htn_bp_flag"]       = (df["systolic_bp_mmhg"] > 140).astype(float)
    for c in ["hemoglobin_g_dl","creatinine_mg_dl","sodium_mmol_l","heart_rate_bpm","systolic_bp_mmhg","followup_days"]:
        df[c+"_missing"] = df[c].isnull().astype(int)
    df["age_group"]           = pd.cut(df["age"],bins=[0,40,60,75,120],labels=["young","middle","senior","elderly"])
    df["comorbidity_load"]    = df["diabetes"]+df["hypertension"]+df["chronic_kidney_disease"]+df["heart_failure"]
    df["med_per_comorbidity"] = df["medication_count"]/(df["comorbidity_count"]+1)
    df["frequent_admitter"]   = (df["prior_admissions_12m"]>=2).astype(int)
    df["long_stay"]           = (df["length_of_stay_days"]>=7).astype(int)
    df["missed_appt_flag"]    = (df["missed_appointments_12m"]>=2).astype(int)
    df["late_followup"]       = ((df["followup_days"]>14)|df["followup_days"].isnull()).astype(int)
    return df

train_fe = feature_engineer(train_raw)
test_fe  = feature_engineer(test_raw)

NEW_CAT = ["age_group"]
NEW_NUM = ["anemia_flag","aki_flag","hyponatremia_flag","tachy_flag","htn_bp_flag",
           "comorbidity_load","med_per_comorbidity","frequent_admitter","long_stay",
           "missed_appt_flag","late_followup",
           "hemoglobin_g_dl_missing","creatinine_mg_dl_missing","sodium_mmol_l_missing",
           "heart_rate_bpm_missing","systolic_bp_mmhg_missing","followup_days_missing"]

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
print(f"Total features: {len(FEATURES)}")

print("Imputing missing values (IterativeImputer)...")
imputer    = IterativeImputer(random_state=SEED, max_iter=10, n_nearest_features=8)
X_imp      = pd.DataFrame(imputer.fit_transform(X),      columns=FEATURES, index=X.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test),     columns=FEATURES, index=X_test.index)
print("Done.")

print("\n" + "="*65)
print("STEP 3 - CROSS-VALIDATION & MODEL TRAINING")
print("="*65)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
POS_WEIGHT = (y==0).sum()/(y==1).sum()
print(f"Imbalance ratio (neg/pos): {POS_WEIGHT:.2f}")

oof_preds  = {}
test_preds = {}
cv_metrics = {}

def show(name, lls, aucs, brs):
    print(f"  [{name}] LL={np.mean(lls):.5f}(+-{np.std(lls):.5f})  AUC={np.mean(aucs):.5f}  Brier={np.mean(brs):.5f}")

# LightGBM
print("\n--- LightGBM ---")
LGB_P = dict(objective="binary",metric="binary_logloss",learning_rate=0.05,
             num_leaves=63,min_child_samples=30,feature_fraction=0.8,bagging_fraction=0.8,
             bagging_freq=5,lambda_l1=0.1,lambda_l2=0.1,scale_pos_weight=POS_WEIGHT,
             n_estimators=2000,n_jobs=-1,random_state=SEED,verbose=-1)
lo=np.zeros(len(X_imp)); lt=np.zeros(len(X_test_imp))
ll_l=[]; auc_l=[]; br_l=[]
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    m=lgb.LGBMClassifier(**LGB_P)
    m.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=[(X_imp.iloc[vi],y.iloc[vi])],
          callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
    p=m.predict_proba(X_imp.iloc[vi])[:,1]
    lo[vi]=p; lt+=m.predict_proba(X_test_imp)[:,1]/N_FOLDS
    ll_l.append(log_loss(y.iloc[vi],p)); auc_l.append(roc_auc_score(y.iloc[vi],p)); br_l.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_l[-1]:.5f}  AUC={auc_l[-1]:.5f}")
show("LightGBM",ll_l,auc_l,br_l)
oof_preds["lgb"]=lo; test_preds["lgb"]=lt
cv_metrics["lgb"]={"logloss":np.mean(ll_l),"auc":np.mean(auc_l),"brier":np.mean(br_l)}

# XGBoost
print("\n--- XGBoost ---")
XGB_P = dict(objective="binary:logistic",eval_metric="logloss",learning_rate=0.05,
             max_depth=6,min_child_weight=5,subsample=0.8,colsample_bytree=0.8,
             reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=POS_WEIGHT,
             n_estimators=2000,early_stopping_rounds=50,
             n_jobs=-1,random_state=SEED,verbosity=0)
xo=np.zeros(len(X_imp)); xt=np.zeros(len(X_test_imp))
ll_x=[]; auc_x=[]; br_x=[]
for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
    m=xgb.XGBClassifier(**XGB_P)
    m.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=[(X_imp.iloc[vi],y.iloc[vi])],verbose=False)
    p=m.predict_proba(X_imp.iloc[vi])[:,1]
    xo[vi]=p; xt+=m.predict_proba(X_test_imp)[:,1]/N_FOLDS
    ll_x.append(log_loss(y.iloc[vi],p)); auc_x.append(roc_auc_score(y.iloc[vi],p)); br_x.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_x[-1]:.5f}  AUC={auc_x[-1]:.5f}")
show("XGBoost",ll_x,auc_x,br_x)
oof_preds["xgb"]=xo; test_preds["xgb"]=xt
cv_metrics["xgb"]={"logloss":np.mean(ll_x),"auc":np.mean(auc_x),"brier":np.mean(br_x)}

# CatBoost
if HAS_CATBOOST:
    print("\n--- CatBoost ---")
    co=np.zeros(len(X_imp)); ct=np.zeros(len(X_test_imp))
    ll_c=[]; auc_c=[]; br_c=[]
    for f,(ti,vi) in enumerate(skf.split(X_imp,y)):
        m=CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=6,l2_leaf_reg=5,
                             scale_pos_weight=POS_WEIGHT,eval_metric="Logloss",
                             random_seed=SEED,early_stopping_rounds=50,verbose=False)
        m.fit(X_imp.iloc[ti],y.iloc[ti],eval_set=(X_imp.iloc[vi],y.iloc[vi]))
        p=m.predict_proba(X_imp.iloc[vi])[:,1]
        co[vi]=p; ct+=m.predict_proba(X_test_imp)[:,1]/N_FOLDS
        ll_c.append(log_loss(y.iloc[vi],p)); auc_c.append(roc_auc_score(y.iloc[vi],p)); br_c.append(brier_score_loss(y.iloc[vi],p))
        print(f"  Fold {f+1}: LL={ll_c[-1]:.5f}  AUC={auc_c[-1]:.5f}")
    show("CatBoost",ll_c,auc_c,br_c)
    oof_preds["cat"]=co; test_preds["cat"]=ct
    cv_metrics["cat"]={"logloss":np.mean(ll_c),"auc":np.mean(auc_c),"brier":np.mean(br_c)}

# Logistic Regression
print("\n--- Logistic Regression (Baseline) ---")
scaler=StandardScaler()
Xs=scaler.fit_transform(X_imp); Xs_t=scaler.transform(X_test_imp)
lro=np.zeros(len(Xs)); lrt=np.zeros(len(Xs_t))
ll_r=[]; auc_r=[]; br_r=[]
for f,(ti,vi) in enumerate(skf.split(Xs,y)):
    m=CalibratedClassifierCV(LogisticRegression(C=0.5,class_weight="balanced",max_iter=1000,random_state=SEED),method="sigmoid",cv=3)
    m.fit(Xs[ti],y.iloc[ti])
    p=m.predict_proba(Xs[vi])[:,1]
    lro[vi]=p; lrt+=m.predict_proba(Xs_t)[:,1]/N_FOLDS
    ll_r.append(log_loss(y.iloc[vi],p)); auc_r.append(roc_auc_score(y.iloc[vi],p)); br_r.append(brier_score_loss(y.iloc[vi],p))
    print(f"  Fold {f+1}: LL={ll_r[-1]:.5f}  AUC={auc_r[-1]:.5f}")
show("LogReg",ll_r,auc_r,br_r)
oof_preds["lr"]=lro; test_preds["lr"]=lrt
cv_metrics["lr"]={"logloss":np.mean(ll_r),"auc":np.mean(auc_r),"brier":np.mean(br_r)}

# Weighted Ensemble
print("\n--- Weighted Ensemble ---")
inv={k:1.0/cv_metrics[k]["logloss"] for k in cv_metrics}
tot=sum(inv.values())
W={k:v/tot for k,v in inv.items()}
print("  Weights:", {k:round(v,4) for k,v in W.items()})
ens_oof  = sum(W[k]*oof_preds[k] for k in W)
ens_test = sum(W[k]*test_preds[k] for k in W)
ens_ll=log_loss(y,ens_oof); ens_auc=roc_auc_score(y,ens_oof); ens_br=brier_score_loss(y,ens_oof)
print(f"  Ensemble OOF: LL={ens_ll:.5f}  AUC={ens_auc:.5f}  Brier={ens_br:.5f}")

print("\n" + "="*65)
print("STEP 4 - PROBABILITY CALIBRATION")
print("="*65)

iso=IsotonicRegression(out_of_bounds="clip")
iso.fit(ens_oof,y)
iso_oof=iso.predict(ens_oof); iso_test=iso.predict(ens_test)

platt=LogisticRegression()
platt.fit(ens_oof.reshape(-1,1),y)
platt_oof=platt.predict_proba(ens_oof.reshape(-1,1))[:,1]
platt_test=platt.predict_proba(ens_test.reshape(-1,1))[:,1]

ll_raw=log_loss(y,ens_oof)
ll_iso=log_loss(y,iso_oof)
ll_platt=log_loss(y,platt_oof)
print(f"  Raw Ensemble  LL: {ll_raw:.5f}")
print(f"  Isotonic      LL: {ll_iso:.5f}")
print(f"  Platt         LL: {ll_platt:.5f}")

best=min(ll_raw,ll_iso,ll_platt)
if ll_iso==best:
    final=iso_test; calib="Isotonic"
elif ll_platt==best:
    final=platt_test; calib="Platt"
else:
    final=ens_test; calib="None"
print(f"  Selected: {calib}")
final=np.clip(final,1e-5,1-1e-5)

print("\n" + "="*65)
print("STEP 5 - SUBMISSION FILE")
print("="*65)
sub=pd.DataFrame({"patient_id":test_raw[ID_COL].values,"readmitted_30d":final})
assert (sub["readmitted_30d"]>0).all() and (sub["readmitted_30d"]<1).all()
sub.to_csv("submission.csv",index=False)
print(f"  Saved submission.csv  ({len(sub)} rows)")
print(f"  Probability range: [{final.min():.6f}, {final.max():.6f}]")
print(f"  Mean pred: {final.mean():.6f}  Train label mean: {y.mean():.6f}")
print(sub.head())

print("\n" + "="*65)
print("FINAL CV SUMMARY")
print("="*65)
header = f"{'Model':<25} {'LogLoss':>10} {'AUC':>10} {'Brier':>10}"
print(header)
print("-"*55)
for k,m in cv_metrics.items():
    print(f"{k:<25} {m['logloss']:>10.5f} {m['auc']:>10.5f} {m['brier']:>10.5f}")
print(f"{'Ensemble':25} {ens_ll:>10.5f} {ens_auc:>10.5f} {ens_br:>10.5f}")
print(f"\nCalibration: {calib}")
print("DONE!")
