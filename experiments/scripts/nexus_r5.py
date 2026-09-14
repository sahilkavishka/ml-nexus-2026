import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("optuna not installed -> run: pip install optuna --break-system-packages")
    print("Falling back to fixed hyperparameters (less optimal).")

SEED = 42
N_FOLDS = 7          # more folds -> steadier OOF estimate
N_TUNE_FOLDS = 3     # cheaper inner CV used only during hyperparameter search
N_TRIALS = 40        # raise to 100+ if you can afford the runtime
TARGET = "readmitted_30d"
ID_COL = "patient_id"
APPLY_PREVALENCE_SHIFT = False

print("=" * 65)
print("ROUND 5: TUNED CATBOOST + LGBM + XGB + LASSO-LR, STACKED")
print("=" * 65)

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")

CAT_COLS = ["sex", "rurality", "hospital_type", "region",
            "discharge_disposition", "care_pathway"]
NUM_COLS = ["age", "socioeconomic_index", "prior_admissions_12m", "comorbidity_count",
            "diabetes", "hypertension", "chronic_kidney_disease", "heart_failure",
            "length_of_stay_days", "medication_count", "missed_appointments_12m",
            "followup_days", "hemoglobin_g_dl", "creatinine_mg_dl", "sodium_mmol_l",
            "heart_rate_bpm", "systolic_bp_mmhg"]


def feature_engineer(df):
    df = df.copy()
    df["anemia_flag"] = (df["hemoglobin_g_dl"] < 11.0).astype(float)
    df["severe_anemia"] = (df["hemoglobin_g_dl"] < 9.0).astype(float)
    df["aki_flag"] = (df["creatinine_mg_dl"] > 1.5).astype(float)
    df["severe_aki"] = (df["creatinine_mg_dl"] > 3.0).astype(float)
    df["hyponatremia_flag"] = (df["sodium_mmol_l"] < 135).astype(float)
    df["tachy_flag"] = (df["heart_rate_bpm"] > 100).astype(float)
    df["htn_bp_flag"] = (df["systolic_bp_mmhg"] > 140).astype(float)

    lab_cols = ["hemoglobin_g_dl", "creatinine_mg_dl", "sodium_mmol_l",
                "heart_rate_bpm", "systolic_bp_mmhg", "followup_days"]
    for c in lab_cols:
        df[c + "_missing"] = df[c].isnull().astype(int)
    df["n_labs_missing"] = df[[c + "_missing" for c in lab_cols]].sum(axis=1)

    df["age_group"] = pd.cut(df["age"], bins=[0, 40, 60, 75, 120],
                              labels=["young", "middle", "senior", "elderly"])
    df["age_over65"] = (df["age"] >= 65).astype(int)

    df["comorbidity_load"] = (df["diabetes"] + df["hypertension"] +
                               df["chronic_kidney_disease"] + df["heart_failure"])
    df["cardiorenal"] = (df["chronic_kidney_disease"] & df["heart_failure"]).astype(int)
    df["diabetic_hypertensive"] = (df["diabetes"] & df["hypertension"]).astype(int)
    df["high_comorbidity"] = (df["comorbidity_load"] >= 3).astype(int)

    df["med_per_comorbidity"] = df["medication_count"] / (df["comorbidity_count"] + 1)
    df["polypharmacy"] = (df["medication_count"] >= 10).astype(int)

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
           "missed_appt_flag", "late_followup", "simple_risk_score",
           "stay_per_comorbidity", "admissions_per_year_age", "labs_abnormal_count"]

ALL_CATS = CAT_COLS + NEW_CAT
ALL_NUMS = NUM_COLS + NEW_NUM
FEATURES = ALL_NUMS + ALL_CATS

# Redundant/collinear pairs kept for trees (they can split around them fine)
# but dropped for the linear model, where duplicated signal just adds noise.
LR_DROP = ["severe_anemia", "severe_aki", "very_frequent"]
LR_NUMS = [c for c in ALL_NUMS if c not in LR_DROP]

y = train_fe[TARGET]
POS_WEIGHT = (y == 0).sum() / (y == 1).sum()

# ── Native (raw) frames for tree models: keep NaN, keep true categories ──────
X_native = train_fe[FEATURES].copy()
X_test_native = test_fe[FEATURES].copy()
for c in ALL_CATS:
    X_native[c] = X_native[c].astype(str).fillna("missing").astype("category")
    X_test_native[c] = X_test_native[c].astype(str).fillna("missing").astype("category")
    cats = pd.api.types.union_categoricals([X_native[c], X_test_native[c]]).categories
    X_native[c] = X_native[c].cat.set_categories(cats)
    X_test_native[c] = X_test_native[c].cat.set_categories(cats)

cat_feature_idx = [FEATURES.index(c) for c in ALL_CATS]
print(f"Features: {len(FEATURES)}  (categorical: {len(ALL_CATS)})")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
tune_skf = StratifiedKFold(n_splits=N_TUNE_FOLDS, shuffle=True, random_state=SEED)


def isotonic_calibrate_oof(oof_pred, y_true, test_pred, folds):
    """Out-of-fold isotonic calibration so trees' raw probabilities aren't
    used uncalibrated (gradient-boosted trees are often mildly overconfident
    at the tails, which log loss punishes heavily)."""
    calibrated_oof = np.zeros_like(oof_pred)
    calibrated_test = np.zeros_like(test_pred)
    for ti, vi in folds.split(oof_pred.reshape(-1, 1), y_true):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_pred[ti], y_true.iloc[ti])
        calibrated_oof[vi] = iso.predict(oof_pred[vi])
        calibrated_test += iso.predict(test_pred) / folds.n_splits
    return calibrated_oof, calibrated_test


# ── Model 1: CatBoost ────────────────────────────────────────────────────────
print("\n--- Tuning CatBoost ---" if HAS_OPTUNA else "\n--- Training CatBoost (fixed params) ---")

if HAS_OPTUNA:
    def cb_objective(trial):
        params = dict(
            iterations=1200,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
            eval_metric="Logloss", cat_features=cat_feature_idx,
            random_seed=SEED, verbose=False, early_stopping_rounds=80,
        )
        scores = []
        for ti, vi in tune_skf.split(X_native, y):
            cb = CatBoostClassifier(**params)
            cb.fit(X_native.iloc[ti], y.iloc[ti], eval_set=(X_native.iloc[vi], y.iloc[vi]))
            p = cb.predict_proba(X_native.iloc[vi])[:, 1]
            scores.append(log_loss(y.iloc[vi], p))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(cb_objective, n_trials=N_TRIALS, show_progress_bar=False)
    CB_BEST = study.best_params
    print(f"  Best CatBoost params: {CB_BEST}  (LL={study.best_value:.5f})")
else:
    CB_BEST = dict(learning_rate=0.03, depth=5, l2_leaf_reg=4,
                   bagging_temperature=0.5, random_strength=0.5)

cat_oof = np.zeros(len(X_native))
cat_test = np.zeros(len(X_test_native))
cat_ll, cat_auc = [], []
for f, (ti, vi) in enumerate(skf.split(X_native, y)):
    cb = CatBoostClassifier(
        iterations=1800, eval_metric="Logloss", cat_features=cat_feature_idx,
        random_seed=SEED + f, early_stopping_rounds=100, verbose=False, **CB_BEST,
    )
    cb.fit(X_native.iloc[ti], y.iloc[ti], eval_set=(X_native.iloc[vi], y.iloc[vi]))
    p = cb.predict_proba(X_native.iloc[vi])[:, 1]
    cat_oof[vi] = p
    cat_test += cb.predict_proba(X_test_native)[:, 1] / N_FOLDS
    cat_ll.append(log_loss(y.iloc[vi], p))
    cat_auc.append(roc_auc_score(y.iloc[vi], p))
print(f"  [CatBoost]  Mean LL={np.mean(cat_ll):.5f}  AUC={np.mean(cat_auc):.5f}")
cat_oof, cat_test = isotonic_calibrate_oof(cat_oof, y, cat_test, skf)
print(f"  [CatBoost calibrated]  LL={log_loss(y, cat_oof):.5f}")

# ── Model 2: LightGBM ────────────────────────────────────────────────────────
print("\n--- Tuning LightGBM ---" if HAS_OPTUNA else "\n--- Training LightGBM (fixed params) ---")

if HAS_OPTUNA:
    def lgb_objective(trial):
        params = dict(
            objective="binary", metric="binary_logloss", n_estimators=1500,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 12, 64),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 0.9),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=5,
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
            n_jobs=-1, random_state=SEED, verbose=-1,
        )
        scores = []
        for ti, vi in tune_skf.split(X_native, y):
            lg = lgb.LGBMClassifier(**params)
            lg.fit(X_native.iloc[ti], y.iloc[ti], eval_set=[(X_native.iloc[vi], y.iloc[vi])],
                   categorical_feature=ALL_CATS,
                   callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
            p = lg.predict_proba(X_native.iloc[vi])[:, 1]
            scores.append(log_loss(y.iloc[vi], p))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
    LGB_BEST = study.best_params
    print(f"  Best LightGBM params: {LGB_BEST}  (LL={study.best_value:.5f})")
else:
    LGB_BEST = dict(learning_rate=0.025, num_leaves=24, max_depth=4, min_child_samples=40,
                     feature_fraction=0.7, bagging_fraction=0.8, lambda_l1=1.5, lambda_l2=2.0)

lgb_oof = np.zeros(len(X_native))
lgb_test = np.zeros(len(X_test_native))
lgb_ll, lgb_auc = [], []
for f, (ti, vi) in enumerate(skf.split(X_native, y)):
    lg = lgb.LGBMClassifier(objective="binary", metric="binary_logloss", n_estimators=2500,
                             bagging_freq=5, n_jobs=-1, random_state=SEED, verbose=-1, **LGB_BEST)
    lg.fit(X_native.iloc[ti], y.iloc[ti], eval_set=[(X_native.iloc[vi], y.iloc[vi])],
           categorical_feature=ALL_CATS,
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    p = lg.predict_proba(X_native.iloc[vi])[:, 1]
    lgb_oof[vi] = p
    lgb_test += lg.predict_proba(X_test_native)[:, 1] / N_FOLDS
    lgb_ll.append(log_loss(y.iloc[vi], p))
    lgb_auc.append(roc_auc_score(y.iloc[vi], p))
print(f"  [LightGBM]  Mean LL={np.mean(lgb_ll):.5f}  AUC={np.mean(lgb_auc):.5f}")
lgb_oof, lgb_test = isotonic_calibrate_oof(lgb_oof, y, lgb_test, skf)
print(f"  [LightGBM calibrated]  LL={log_loss(y, lgb_oof):.5f}")

# ── Model 3: XGBoost ─────────────────────────────────────────────────────────
print("\n--- Tuning XGBoost ---" if HAS_OPTUNA else "\n--- Training XGBoost (fixed params) ---")
X_xgb = X_native.copy()
X_test_xgb = X_test_native.copy()
for c in ALL_CATS:
    X_xgb[c] = X_xgb[c].cat.codes.replace(-1, np.nan)
    X_test_xgb[c] = X_test_xgb[c].cat.codes.replace(-1, np.nan)

if HAS_OPTUNA:
    def xgb_objective(trial):
        params = dict(
            n_estimators=1200,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            eval_metric="logloss", tree_method="hist",
            random_state=SEED, n_jobs=-1, early_stopping_rounds=80,
        )
        scores = []
        for ti, vi in tune_skf.split(X_xgb, y):
            xg = xgb.XGBClassifier(**params)
            xg.fit(X_xgb.iloc[ti], y.iloc[ti], eval_set=[(X_xgb.iloc[vi], y.iloc[vi])], verbose=False)
            p = xg.predict_proba(X_xgb.iloc[vi])[:, 1]
            scores.append(log_loss(y.iloc[vi], p))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
    XGB_BEST = study.best_params
    print(f"  Best XGBoost params: {XGB_BEST}  (LL={study.best_value:.5f})")
else:
    XGB_BEST = dict(learning_rate=0.025, max_depth=4, min_child_weight=5,
                     subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0)

xgb_oof = np.zeros(len(X_xgb))
xgb_test = np.zeros(len(X_test_xgb))
xgb_ll, xgb_auc = [], []
for f, (ti, vi) in enumerate(skf.split(X_xgb, y)):
    xg = xgb.XGBClassifier(n_estimators=2000, eval_metric="logloss", tree_method="hist",
                            random_state=SEED, n_jobs=-1, early_stopping_rounds=100, **XGB_BEST)
    xg.fit(X_xgb.iloc[ti], y.iloc[ti], eval_set=[(X_xgb.iloc[vi], y.iloc[vi])], verbose=False)
    p = xg.predict_proba(X_xgb.iloc[vi])[:, 1]
    xgb_oof[vi] = p
    xgb_test += xg.predict_proba(X_test_xgb)[:, 1] / N_FOLDS
    xgb_ll.append(log_loss(y.iloc[vi], p))
    xgb_auc.append(roc_auc_score(y.iloc[vi], p))
print(f"  [XGBoost]   Mean LL={np.mean(xgb_ll):.5f}  AUC={np.mean(xgb_auc):.5f}")
xgb_oof, xgb_test = isotonic_calibrate_oof(xgb_oof, y, xgb_test, skf)
print(f"  [XGBoost calibrated]  LL={log_loss(y, xgb_oof):.5f}")

# ── Model 4: Calibrated Lasso Logistic Regression ────────────────────────────
print("\n--- Training Calibrated Lasso LR ---")
X_lr_raw = train_fe[LR_NUMS + ALL_CATS].copy()
X_test_lr_raw = test_fe[LR_NUMS + ALL_CATS].copy()
for c in ALL_CATS:
    X_lr_raw[c] = X_lr_raw[c].astype(str)
    X_test_lr_raw[c] = X_test_lr_raw[c].astype(str)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_ohe = ohe.fit_transform(X_lr_raw[ALL_CATS])
test_ohe = ohe.transform(X_test_lr_raw[ALL_CATS])

imputer = IterativeImputer(random_state=SEED, max_iter=10, n_nearest_features=8)
train_num_imp = imputer.fit_transform(X_lr_raw[LR_NUMS])
test_num_imp = imputer.transform(X_test_lr_raw[LR_NUMS])

scaler = StandardScaler()
train_num_scaled = scaler.fit_transform(train_num_imp)
test_num_scaled = scaler.transform(test_num_imp)

Xs = np.hstack([train_num_scaled, train_ohe])
Xs_t = np.hstack([test_num_scaled, test_ohe])

lr_oof = np.zeros(len(Xs))
lr_test = np.zeros(len(Xs_t))
lr_ll, lr_auc = [], []
for f, (ti, vi) in enumerate(skf.split(Xs, y)):
    base_lr = LogisticRegression(C=0.15, penalty="l1", solver="saga",
                                  max_iter=2000, random_state=SEED)
    m = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)
    m.fit(Xs[ti], y.iloc[ti])
    p = m.predict_proba(Xs[vi])[:, 1]
    lr_oof[vi] = p
    lr_test += m.predict_proba(Xs_t)[:, 1] / N_FOLDS
    lr_ll.append(log_loss(y.iloc[vi], p))
    lr_auc.append(roc_auc_score(y.iloc[vi], p))
print(f"  [Lasso-LR]  Mean LL={np.mean(lr_ll):.5f}  AUC={np.mean(lr_auc):.5f}")

# ── STACKING (logistic-regression meta-learner on OOF predictions) ──────────
print("\n" + "=" * 65)
print("STACKING BASE MODELS WITH A LOGISTIC-REGRESSION META-LEARNER")
print("=" * 65)

meta_X = np.column_stack([cat_oof, lgb_oof, xgb_oof, lr_oof])
meta_X_test = np.column_stack([cat_test, lgb_test, xgb_test, lr_test])

meta_oof = np.zeros(len(meta_X))
meta_test = np.zeros(len(meta_X_test))
for f, (ti, vi) in enumerate(skf.split(meta_X, y)):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(meta_X[ti], y.iloc[ti])
    meta_oof[vi] = meta.predict_proba(meta_X[vi])[:, 1]
    meta_test += meta.predict_proba(meta_X_test)[:, 1] / N_FOLDS

final_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(meta_X, y)
print(f"  Meta-model coefficients (Cat, LGB, XGB, LR): {final_meta.coef_[0]}")
print(f"  Stacked OOF Log Loss: {log_loss(y, meta_oof):.5f}")
print(f"  Stacked OOF ROC-AUC : {roc_auc_score(y, meta_oof):.5f}")

opt_oof, opt_test = meta_oof, meta_test

# ── Optional prevalence shift (off by default) ───────────────────────────────
train_prev = y.mean()
test_mean = opt_test.mean()
print(f"\n  OOF mean: {opt_oof.mean():.6f} | Test-pred mean: {test_mean:.6f} "
      f"| Train prevalence: {train_prev:.6f}")

if APPLY_PREVALENCE_SHIFT:
    def shift_log_odds(p, current_mean, target_mean):
        lo = np.log(p / (1.0 - p))
        delta = np.log((target_mean / (1.0 - target_mean)) /
                        (current_mean / (1.0 - current_mean)))
        return 1.0 / (1.0 + np.exp(-(lo + delta)))

    opt_test = shift_log_odds(opt_test, test_mean, train_prev)
    print(f"  Applied prevalence shift -> new test mean: {opt_test.mean():.6f}")
else:
    print("  Prevalence shift NOT applied (APPLY_PREVALENCE_SHIFT=False).")

final_probs = np.clip(opt_test, 0.005, 0.90)

# ── SAVE SUBMISSION ──────────────────────────────────────────────────────────
sub = pd.DataFrame({
    "patient_id": test_raw[ID_COL].values,
    "readmitted_30d": final_probs
})
sub.to_csv("codewave submission 5.csv", index=False)
print(f"\n  Saved: 'codewave submission 5.csv' ({len(sub)} rows)")
print(f"  Min prob: {final_probs.min():.6f}, Max prob: {final_probs.max():.6f}")
print(f"  Mean prob: {final_probs.mean():.6f}")
print(sub.head(10))

print("\n" + "=" * 65)
print("ROUND 5 COMPLETE!")
print("=" * 65)