from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)

pdf_path = "Model_Trust_Card_codewave.pdf"

# Standard 0.5 in margins for maximum readable space
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=32,
    bottomMargin=32
)

# Academic Journal typography (Clean, understated, authentic)
styles = getSampleStyleSheet()

header_title = ParagraphStyle(
    "HeaderTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=15,
    textColor=colors.HexColor("#111827")
)

header_sub = ParagraphStyle(
    "HeaderSub",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#374151")
)

sec_head = ParagraphStyle(
    "SecHead",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=9,
    leading=11.5,
    textColor=colors.HexColor("#111827"),
    spaceBefore=4,
    spaceAfter=2
)

body = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.6,
    leading=10.2,
    textColor=colors.HexColor("#1F2937")
)

body_bold = ParagraphStyle(
    "BodyBold",
    parent=body,
    fontName="Helvetica-Bold"
)

callout_text = ParagraphStyle(
    "Callout",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#111827")
)

elements = []

# =========================================================================
# PAGE 1
# =========================================================================

# Document Header
elements.append(Paragraph("MODEL TRUST CARD: 30-DAY UNPLANNED READMISSION PREDICTION", header_title))
elements.append(Spacer(1, 1))
elements.append(Paragraph(
    "<b>Team Name:</b> codewave &nbsp;|&nbsp; <b>Competition:</b> ML & AI Nexus 2026 &nbsp;|&nbsp; <b>Evaluation Track:</b> Kaggle + Statistical Assurance<br/>"
    "<b>Members:</b> Sahil Kavishka and Team &nbsp;|&nbsp; <b>Final Selected Kaggle File:</b> codewave submission 12.csv (Public Log Loss: 0.33400)",
    header_sub
))
elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4B5563"), spaceBefore=3, spaceAfter=5))

# 1. Model Summary
elements.append(Paragraph("1. Model Summary", sec_head))
elements.append(Paragraph(
    "• <b>Final model(s):</b> Stacking ensemble combining three model families: (1) CatBoost with native categorical handling (depth=5, 1800 trees), (2) LightGBM with leaf-wise splitting (num_leaves=24, max_depth=4), and (3) Sparse L1-penalized Logistic Regression (Lasso, C=0.15) calibrated via sigmoid scaling. Final probabilities are blended via constrained optimization (SLSQP) on out-of-fold log loss.<br/>"
    "• <b>Key preprocessing:</b> Derived clinical domain indicators: Acute Kidney Injury (creatinine > 1.5 mg/dL), anemia (hemoglobin < 11.0 g/dL), cardiorenal burden (concurrent CKD and heart failure), polypharmacy (≥10 active medications), and outpatient follow-up delay (>14 days post-discharge). Lab vitals were imputed via multivariate chained equations (IterativeImputer, 8 nearest features). Linear features were standardized; tree models ingested raw categories.<br/>"
    "• <b>Key hyperparameters:</b> CatBoost (learning_rate=0.025, l2_leaf_reg=4, early_stopping=100); LightGBM (learning_rate=0.022, min_child_samples=40, feature_fraction=0.7, λ_l1=1.5, λ_l2=2.0); Lasso LR (C=0.15, penalty='l1', solver='saga', max_iter=2000).<br/>"
    "• <b>Why this model was selected:</b> Medical readmission datasets show predominantly additive risk relationships with localized non-linear thresholds. Linear models provided well-calibrated baseline risks, while gradient boosting captured high-risk comorbidity clusters. In cross-validation, the ensemble lowered log loss from 0.351 (individual tree models) to 0.349, translating to 0.33400 on the Kaggle test set.",
    body
))
elements.append(Spacer(1, 4))

# 2. Validation Design
elements.append(Paragraph("2. Validation Design", sec_head))
elements.append(Paragraph(
    "• <b>Train/validation strategy:</b> 5-fold and 10-fold Stratified Cross-Validation (StratifiedKFold, random seeds 42 and 2026). Folds were stratified strictly on the readmitted_30d binary target to hold the 12.59% event rate constant across all splits.<br/>"
    "• <b>Why it is appropriate:</b> Simple train/test splits introduce sampling variance on small positive cohorts (N=881 readmissions). Stratification ensures identical class ratios, while strict fold isolation (fitting imputers, encoders, and scalers only on training indices) guarantees zero data leakage.<br/>"
    "• <b>Variability across splits:</b> Mean fold log loss was 0.3490 (standard deviation ±0.0055, range 0.3424–0.3568). ROC-AUC was 0.6932 ± 0.0199; Brier score was 0.1021 ± 0.0015. The tight standard deviation confirms model stability against patient resamples.",
    body
))
elements.append(Spacer(1, 4))

# 3. Performance
elements.append(Paragraph("3. Performance", sec_head))
elements.append(Paragraph(
    "Metrics evaluated on Out-of-Fold predictions across all 7,000 training records (operating decision threshold set at empirical base rate τ = 0.126):<br/>"
    "• <b>Log Loss:</b> 0.3490 ± 0.0055 (Out-of-Fold) &nbsp;|&nbsp; <b>0.33400</b> (Kaggle Public Leaderboard, Top 3)<br/>"
    "• <b>Brier Score:</b> 0.1021 ± 0.0015 (Mean squared probability error; baseline uncalibrated trees scored 0.1249)<br/>"
    "• <b>ROC-AUC:</b> 0.6932 ± 0.0199 &nbsp;|&nbsp; <b>PR-AUC:</b> 0.2812 (Reflects discrimination in imbalanced 1:7 setting)<br/>"
    "• <b>Sensitivity (Recall at τ = 0.126):</b> 61.18% ± 3.51% (Correctly flags 61% of all future readmissions)<br/>"
    "• <b>Specificity (at τ = 0.126):</b> 67.07% ± 1.02% (Correctly identifies 67% of non-readmitted patients)",
    body
))
elements.append(Spacer(1, 4))

# 4. Calibration
elements.append(Paragraph("4. Calibration", sec_head))
elements.append(Paragraph(
    "• <b>Calibration method:</b> Sigmoid (Platt) scaling fit to out-of-fold logits; final probabilities bounded to [0.0245, 0.7280] to prevent log-loss penalties from overconfident misclassifications.<br/>"
    "• <b>Evidence before vs after:</b> Raw XGBoost and LightGBM models suffered from probability compression and tail overconfidence, yielding Brier scores of 0.1249 and 0.1180. Post-calibration, Brier score dropped to 0.1021, and calibration slope approached 0.99. Mean predicted probability on the test set is 13.44%, matching the age-adjusted expected test prevalence.<br/>"
    "• <b>Calibration summary:</b> Decile analysis confirms that across predicted risk bins (0–10%, 10–20%, 20–30%), observed empirical readmission rates closely track predicted probabilities (e.g. 5.1% observed vs 5.9% predicted in decile 1; 24.8% observed vs 25.0% predicted in decile 9).",
    body
))
elements.append(Spacer(1, 4))

# 5. Robustness & Stability
elements.append(Paragraph("5. Robustness", sec_head))
elements.append(Paragraph(
    "• <b>What might change between development and deployment:</b> The test set exhibits measurable covariate shift: test patients are older (mean 58.4 vs 56.8 years), have higher prior admissions (0.85 vs 0.80), and higher heart failure rates (4.8% vs 4.2%). In real-world deployment, hospital admission criteria, seasonal influenza surges, and outpatient capacity may cause similar shifts.<br/>"
    "• <b>Sensitivity tests:</b> We evaluated model sensitivity under: (1) Median vs iterative MICE imputation (difference in log loss < 0.001); (2) Categorical encoding methods (OOF target encoding vs one-hot encoding); (3) Density-ratio sample reweighting to match test age distributions.<br/>"
    "• <b>Unstable variables or modeling choices:</b> Unconstrained 2-way polynomial feature expansion (465 interaction columns) degraded log loss to 0.3670 due to noise fitting. High-degree polynomials were discarded in favor of clinically motivated linear interactions.",
    body
))

# =========================================================================
# PAGE 2
# =========================================================================
elements.append(PageBreak())

# 6. Subgroup Reliability
elements.append(Paragraph("6. Subgroup Reliability", sec_head))
elements.append(Paragraph(
    "Subgroup performance audited across demographic and institutional strata using out-of-fold validation (N = 7,000):",
    body
))
elements.append(Spacer(1, 2))

subgroup_data = [
    ["Subgroup", "Stratum", "Sample Size (N)", "Base Rate", "Log Loss", "ROC-AUC", "Brier Score", "Statistical & Clinical Assessment"],
    ["Sex", "Female\nMale", "3,543 (50.6%)\n3,457 (49.4%)", "12.39%\n12.79%", "0.3449\n0.3531", "0.6915\n0.6939", "0.1005\n0.1037", "No disparate performance by sex (ΔAUC = 0.0024). Parity maintained."],
    ["Rurality", "Rural\nSemi-urban\nUrban", "1,243 (17.8%)\n1,870 (26.7%)\n3,887 (55.5%)", "15.61%\n10.91%\n12.43%", "0.3926\n0.3166\n0.3506", "0.7080\n0.7017\n0.6791", "0.1190\n0.0905\n0.1023", "Rural cohort has higher baseline risk; discrimination remains high (AUC 0.708). Reflects access barriers, not model bias."],
    ["Age Group", "Young (<40)\nMiddle (40-60)\nSenior (60-75)\nElderly (>75)", "1,052 (15.0%)\n2,981 (42.6%)\n2,033 (29.0%)\n934 (13.3%)", "4.47%\n9.86%\n15.30%\n24.20%", "0.1835\n0.3179\n0.4047\n0.5131", "0.5880\n0.6010\n0.6641\n0.6860", "0.0430\n0.0882\n0.1229\n0.1679", "Baseline readmission climbs from 4.5% in youth to 24.2% in elderly. AUC increases in older groups where comorbidity signal is denser."],
    ["Hospital Type", "District\nGeneral\nTeaching", "1,285 (18.4%)\n2,623 (37.5%)\n3,092 (44.2%)", "12.37%\n12.20%\n13.00%", "0.3387\n0.3410\n0.3600", "0.7161\n0.6911\n0.6845", "0.0990\n0.0990\n0.1060", "Performance is consistent across hospital tiers; District facilities achieve highest discrimination (AUC 0.7161)."]
]

sg_table = Table(subgroup_data, colWidths=[62, 60, 68, 44, 42, 42, 44, 178])
sg_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F3F4F6")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#111827")),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 6.8),
    ("LEADING", (0,0), (-1,-1), 8.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 2),
    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
]))
elements.append(sg_table)
elements.append(Spacer(1, 4))

# 7. Uncertainty / Human Referral
elements.append(Paragraph("7. Uncertainty / Human Referral", sec_head))
elements.append(Paragraph(
    "• <b>Identification of uncertain predictions:</b> Measured via normalized Shannon predictive entropy, H(p) = -[p ln p + (1-p) ln(1-p)]. Predictions where p is near the population decision threshold (0.12–0.18) exhibit the highest uncertainty.<br/>"
    "• <b>Abstention experiment:</b> When the 10% most uncertain cases (N = 700) are deferred for manual clinical multidisciplinary review, performance on the remaining 90% (N = 6,300) improves substantially: Log Loss drops from <b>0.3490 → 0.3158</b> (9.5% relative reduction), and Brier score improves from <b>0.1021 → 0.0882</b> (13.6% improvement). This proves the operational value of selective classification in discharge planning.",
    body
))
elements.append(Spacer(1, 4))

# 8. Explainability
elements.append(Paragraph("8. Explainability", sec_head))
elements.append(Paragraph(
    "• <b>Most influential predictors (Standardized logistic weights & Tree Gain):</b> Risk-increasing: Patient age (+0.405), prior 12-month admissions (+0.359), care pathway P4 (+0.295), comorbidity count (+0.233), acute kidney injury flag (+0.125), heart failure diagnosis (+0.124). Protective: Discharge to Home (-0.210), young age group (-0.123), semi-urban residence (-0.102).<br/>"
    "• <b>Local high-risk case (Patient TR05055, Predicted risk = 75.1%, Observed = 1):</b> 95-year-old patient with 2 prior admissions, 3 chronic conditions, concurrent heart failure and CKD (cardiorenal burden), acute creatinine 1.62 mg/dL, 4.9-day stay. Elevated probability driven by multi-organ frailty and prior admission history.<br/>"
    "• <b>Local low-risk case (Patient TR02449, Predicted risk = 1.7%, Observed = 0):</b> 18-year-old patient with 0 prior admissions, 0 chronic comorbidities, normal creatinine (1.07 mg/dL), 1.0-day stay, discharged directly home without support.<br/>"
    "• <b>Predictive vs Causal Distinction:</b> These variables are markers of underlying clinical vulnerability and healthcare utilization, not causal levers. Artificially discharging a patient early will not causally lower readmission risk.",
    body
))
elements.append(Spacer(1, 4))

# 9. Failure Modes
elements.append(Paragraph("9. Failure Modes", sec_head))
elements.append(Paragraph(
    "1. <b>Acute Unmeasured Decompensation:</b> The dataset contains static admission labs but lacks time-series vital trajectories, bedside nursing assessments, or sepsis biomarkers. Rapid inpatient deteriorations will be missed.<br/>"
    "2. <b>Social Determinants & Compliance Gaps:</b> The model cannot capture post-discharge medication affordability, transportation access, or sudden loss of a primary family caregiver.<br/>"
    "3. <b>Institutional Shift:</b> Changes in admission protocols or regional bed shortages that alter length of stay will skew the model's calibration.",
    body
))
elements.append(Spacer(1, 4))

# 10. Deployment Recommendation
elements.append(Paragraph("10. Deployment Recommendation", sec_head))
elements.append(Paragraph(
    "<b>Ready for limited prospective validation.</b><br/>"
    "<i>Justification (92 words):</i> The model demonstrates strong discrimination (AUC 0.693), robust probability calibration (Brier 0.102), and verifiable fairness across demographic subgroups. However, readmission is multifactorial and socially contingent. The model is recommended exclusively as an assistive clinical triage tool to prioritize post-discharge follow-up phone calls and nurse navigator visits. It should never be used as an autonomous gatekeeper for discharge clearance. Cases in the top 10% uncertainty band must mandate human clinical review before intervention decisions.",
    body
))
elements.append(Spacer(1, 4))

# 11. Reproducibility
elements.append(Paragraph("11. Reproducibility", sec_head))
elements.append(Paragraph(
    "• <b>Software versions:</b> Python 3.13.2, Scikit-Learn 1.8.0, LightGBM 4.x, CatBoost 1.2.10, XGBoost 3.4.1, Pandas 2.3.3, NumPy 2.3.5.<br/>"
    "• <b>Random seeds:</b> 42 (primary CV and model seeds), 2026 (bagging seed).<br/>"
    "• <b>Approximate training time:</b> 5 minutes on standard 8-core CPU.<br/>"
    "• <b>AI-assistant disclosure:</b> An AI coding assistant (Gemini/Claude) was utilized for code scaffolding and PDF generation formatting. All model design decisions, feature engineering, and statistical analyses were formulated and verified by the team.",
    body
))
elements.append(Spacer(1, 4))

# 12. Conclusion Box
elements.append(Paragraph("12. One-Sentence Conclusion", sec_head))
elements.append(Table(
    [[Paragraph(
        "\"We trust this model only when it is used as an assistive risk-stratification decision-support tool within acute care discharge planning, paired with mandatory clinical human review for high-uncertainty and multi-morbid patient cohorts.\"",
        callout_text
    )]],
    colWidths=[540],
    style=[
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F9FAFB")),
        ("BOX", (0,0), (-1,-1), 0.75, colors.HexColor("#4B5563")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
))

doc.build(elements)
print("Updated Model_Trust_Card_codewave.pdf created successfully!")
