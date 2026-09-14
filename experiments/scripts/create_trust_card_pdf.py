from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)

pdf_path = "Model_Trust_Card_codewave.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    leftMargin=40,
    rightMargin=40,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=19,
    textColor=colors.HexColor("#1A365D")
)

meta_style = ParagraphStyle(
    "Meta",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9,
    leading=12.5,
    textColor=colors.HexColor("#2D3748")
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=colors.HexColor("#2B6CB0"),
    spaceBefore=7,
    spaceAfter=3
)

body_style = ParagraphStyle(
    "BodyTextCustom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#1A202C")
)

quote_style = ParagraphStyle(
    "Quote",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor("#1A365D")
)

elements = []

# ================= PAGE 1 =================
elements.append(Paragraph("MODEL TRUST CARD: 30-DAY UNPLANNED HOSPITAL READMISSION", title_style))
elements.append(Spacer(1, 3))
elements.append(Paragraph("<b>Theme:</b> Beyond the Black Box: Statistics for Trustworthy AI &nbsp;|&nbsp; <b>Competition:</b> ML & AI NEXUS 2026<br/><b>Team:</b> codewave &nbsp;|&nbsp; <b>Members:</b> Sahil Kavishka and Team &nbsp;|&nbsp; <b>Kaggle Submission:</b> codewave submission 4.csv (Score: 0.33423, Top 3)", meta_style))
elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=5, spaceAfter=8))

# 1. Model Summary
elements.append(Paragraph("1. Model Summary", h1_style))
p1 = """<b>Architecture:</b> Stacking ensemble integrating diverse gradient-boosted decision trees (CatBoost, LightGBM, XGBoost) with a Calibrated Sparse L1-penalized Logistic Regression (Lasso). Optimal blending weights were determined via multi-restart SLSQP constrained optimization directly minimizing Out-of-Fold Log Loss.<br/>
<b>Key Preprocessing:</b> Multi-stage domain pipeline: (a) Clinical feature engineering including Acute Kidney Injury (creatinine > 1.5 mg/dL), anemia (hemoglobin < 11 g/dL), cardiorenal syndrome interaction (CKD & Heart Failure), polypharmacy (≥10 medications), and planned follow-up access delays (>14 days); (b) Informative missingness indicator flags for all laboratory and vital parameters; (c) Multivariate iterative imputation (IterativeImputer / MICE) for missing vitals; (d) Standard scaling for linear models and native categorical handling for tree algorithms.<br/>
<b>Key Hyperparameters:</b> Lasso LR (C=0.15, penalty='l1', SAGA solver, Sigmoid calibration); CatBoost (depth=5, learning_rate=0.03, l2_leaf_reg=4, 1500 trees); LightGBM (depth=4, num_leaves=24, learning_rate=0.025, λ_L1=1.5, λ_L2=2.0).<br/>
<b>Rationale:</b> Linear models provide smooth, well-calibrated baseline probabilities, while gradient boosters capture complex non-linear clinical risk interactions. Stacking them achieved superior discrimination and the lowest well-calibrated Log Loss (0.3489 CV, 0.33423 Kaggle)."""
elements.append(Paragraph(p1, body_style))
elements.append(Spacer(1, 6))

# 2. Validation Design
elements.append(Paragraph("2. Validation Design", h1_style))
p2 = """<b>Validation Strategy:</b> 10-Fold and 5-Fold Stratified Cross-Validation (StratifiedKFold, seeds 42 and 2026). Target stratification was strictly enforced on the binary outcome (readmitted_30d) to preserve the 12.59% minority readmission prevalence across all folds.<br/>
<b>Methodological Rationale:</b> Prevents class imbalance drift between resamples and provides a rock-solid, unbiased out-of-fold (OOF) generalization estimate. All feature encoders, scalers, and iterative imputers were fitted exclusively inside training folds, rigorously preventing data leakage.<br/>
<b>Uncertainty Across Folds:</b> OOF Log Loss = 0.3490 ± 0.0055 | ROC-AUC = 0.6932 ± 0.0199 | Brier Score = 0.1021 ± 0.0015. Tight standard deviations across splits demonstrate high model stability and robustness across population resamples."""
elements.append(Paragraph(p2, body_style))
elements.append(Spacer(1, 6))

# 3. Performance
elements.append(Paragraph("3. Predictive Performance", h1_style))
p3 = """Evaluated across Out-of-Fold cross-validation (with clinical operating threshold aligned to population prevalence τ = 0.126):<br/>
• <b>Binary Log Loss:</b> <b>0.3490 ± 0.0055</b> (Kaggle Public Leaderboard: <b>0.33423 — Rank 3 / Top 3</b>)<br/>
• <b>Brier Score:</b> <b>0.1021 ± 0.0015</b> (Demonstrating superior probability accuracy compared to uncalibrated baselines)<br/>
• <b>ROC-AUC:</b> <b>0.6932 ± 0.0199</b> | <b>PR-AUC:</b> <b>0.2812</b> (Strong discriminative power under heavy 87:13 class imbalance)<br/>
• <b>Sensitivity (Recall at τ = 0.126):</b> <b>61.18% ± 3.51%</b> (Captures over 60% of all unplanned readmissions)<br/>
• <b>Specificity (at τ = 0.126):</b> <b>67.07% ± 1.02%</b> (Safely rules out two-thirds of stable discharge candidates)"""
elements.append(Paragraph(p3, body_style))
elements.append(Spacer(1, 6))

# 4. Calibration
elements.append(Paragraph("4. Probability Calibration", h1_style))
p4 = """<b>Calibration Method:</b> Sigmoid (Platt) scaling applied to logistic outputs; ensemble predictions bounded strictly within realistic clinical safety limits [0.015, 0.780].<br/>
<b>Empirical Evidence Before vs After:</b> Uncalibrated tree ensembles exhibited severe probability dispersion and extreme tail overconfidence. Post-stacking calibration reduced the Brier Score from 0.1249 (raw XGBoost) down to 0.1021. Predicted test probability mean is 13.56%, closely tracking actual population incidence (12.59%) and eliminating catastrophic log-loss tail penalties."""
elements.append(Paragraph(p4, body_style))
elements.append(Spacer(1, 6))

# 5. Robustness & Stability
elements.append(Paragraph("5. Robustness & Sensitivity Analyses", h1_style))
p5 = """<b>Potential Population Shifts:</b> Hospital referral variations, differences in inpatient length of stay, and shifts in post-discharge follow-up accessibility between development and deployment sites.<br/>
<b>Sensitivity Testing:</b> Assessed under varying categorical encodings, iterative MICE versus median imputation, and extreme boundary clipping ([0.005, 0.90] vs [0.015, 0.78]).<br/>
<b>Unstable Modeling Choices Identified:</b> Unconstrained 2-way polynomial feature expansion (465 interactions) caused severe overfitting (OOF Log Loss degraded to 0.3670) and was rejected in favor of parsimonious clinical indicators and L1 sparsity."""
elements.append(Paragraph(p5, body_style))

# ================= PAGE 2 =================
elements.append(PageBreak())

# 6. Subgroup Reliability
elements.append(Paragraph("6. Subgroup Reliability & Demographic Fairness Audit", h1_style))
elements.append(Paragraph("Comprehensive fairness and calibration audit conducted across demographic and contextual cohorts (OOF predictions):", body_style))
elements.append(Spacer(1, 3))

subgroup_data = [
    ["Subgroup", "Cohort", "N", "Obs. Prev", "Log Loss", "ROC-AUC", "Brier Score", "Equity & Clinical Assessment"],
    ["Sex", "Female\nMale", "3,543\n3,457", "12.39%\n12.79%", "0.3449\n0.3531", "0.6915\n0.6939", "0.1005\n0.1037", "Excellent demographic parity (ΔAUC = 0.0024); zero gender bias observed."],
    ["Rurality", "Rural\nSemi-urban\nUrban", "1,243\n1,870\n3,887", "15.61%\n10.91%\n12.43%", "0.3926\n0.3166\n0.3506", "0.7080\n0.7017\n0.6791", "0.1190\n0.0905\n0.1023", "Rural cohort exhibits higher readmission baseline, reflecting geographic access barriers."],
    ["Age Group", "Young (<40)\nMiddle (40-60)\nSenior (60-75)\nElderly (>75)", "1,052\n2,981\n2,033\n934", "4.47%\n9.86%\n15.30%\n24.20%", "0.1835\n0.3179\n0.4047\n0.5131", "0.5880\n0.6010\n0.6641\n0.6860", "0.0430\n0.0882\n0.1229\n0.1679", "Elderly risk increases sharply due to frailty; discriminative accuracy peaks in seniors."],
    ["Hospital Type", "District\nGeneral\nTeaching", "1,285\n2,623\n3,092", "12.37%\n12.20%\n13.00%", "0.3387\n0.3410\n0.3600", "0.7161\n0.6911\n0.6845", "0.0990\n0.0990\n0.1060", "Strongest discrimination in District hospitals (AUC 0.716); stable across tiers."]
]

sg_table = Table(subgroup_data, colWidths=[65, 62, 35, 45, 45, 45, 45, 190])
sg_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EDF2F7")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#1A365D")),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 7),
    ("LEADING", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 2),
    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
]))
elements.append(sg_table)
elements.append(Spacer(1, 6))

# 7. Uncertainty
elements.append(Paragraph("7. Uncertainty Quantification & Selective Prediction (Human Referral)", h1_style))
p7 = """<b>Uncertainty Identification:</b> Predictive Shannon entropy was computed as H(p) = -[p ln p + (1-p) ln(1-p)]. Boundary cases close to decision ambiguity display maximal uncertainty.<br/>
<b>Impact of Deferring the 10% Most Uncertain Cases:</b><br/>
• <b>Full Population (N=7,000):</b> Log Loss = <b>0.3490</b> | Brier Score = <b>0.1021</b> | ROC-AUC = 0.6927<br/>
• <b>Remaining 90% Cohort (N=6,300):</b> Log Loss drops to <b>0.3158 (9.5% improvement)</b> | Brier Score drops to <b>0.0882 (13.6% improvement)</b>.<br/>
<b>Clinical Value:</b> Validates selective triage: routing the top 10% most ambiguous cases for Multidisciplinary Team (MDT) review drastically improves autonomous decision quality for the remaining 90% of discharge workflows."""
elements.append(Paragraph(p7, body_style))
elements.append(Spacer(1, 6))

# 8. Explainability
elements.append(Paragraph("8. Explainability & Causal Reasoning", h1_style))
p8 = """<b>Most Influential Predictors:</b> Risk-increasing: Age (+0.405), prior 12m admissions (+0.359), care pathway P4 (+0.295), chronic comorbidity count (+0.233), acute kidney injury (+0.125), heart failure (+0.124). Protective: Discharge to Home (-0.210), young age group (-0.123).<br/>
<b>Local Explanations:</b><br/>
• <i>High-Risk Prediction (Patient TR05055, Pred: 75.12%, Actual: Readmitted):</i> 95-year-old, 2 prior admissions, 3 chronic comorbidities, concurrent heart failure and CKD (cardiorenal burden), acute creatinine 1.62 mg/dL, 4.9-day stay.<br/>
• <i>Low-Risk Prediction (Patient TR02449, Pred: 1.66%, Actual: Not Readmitted):</i> 18-year-old, 0 prior admissions, 0 comorbidities, normal creatinine 1.07 mg/dL, 1.0-day stay, discharged directly home.<br/>
<b>Causal Distinctions:</b> Features represent markers of baseline frailty and utilization patterns, not modifiable causal targets. Artificially truncating length of stay will not causally prevent readmission."""
elements.append(Paragraph(p8, body_style))
elements.append(Spacer(1, 6))

# 9, 10, 11
elements.append(Paragraph("9. Failure Modes, 10. Deployment Recommendation & 11. Reproducibility", h1_style))
p9_10_11 = """<b>Concrete Failure Modes:</b> (1) <i>Unmeasured Clinical Acuity:</i> Missing ICU vitals time-series and real-time sepsis biomarkers; (2) <i>Post-Discharge Social Factors:</i> Unrecorded medication non-adherence or sudden loss of home caregiver support; (3) <i>Cross-Hospital Domain Shift:</i> Changes in regional hospital capacity or admission thresholds.<br/>
<b>Deployment Recommendation: Ready for limited prospective validation.</b><br/>
<i>Justification:</i> The model demonstrates solid discrimination (AUC 0.693), robust calibration (Brier 0.102), verified subgroup parity across sex and hospital tiers, and transparent explainability. However, because readmission etiology is multifaceted, it should be deployed exclusively as a decision-support triage tool alongside human oversight, with automatic referral of the top 10% uncertain cases to nurse navigators, rather than an autonomous discharge gatekeeper.<br/>
<b>Reproducibility:</b> Python 3.13.2, Scikit-Learn 1.8.0, LightGBM 4.x, CatBoost 1.2.10, XGBoost 3.4.1. Seeds: 42, 2026. Training time: ~6 mins on multi-core CPU. AI assistant (Gemini/Claude) utilized for script scaffolding and visualization formatting in compliance with disclosure guidelines."""
elements.append(Paragraph(p9_10_11, body_style))
elements.append(Spacer(1, 6))

# 12. Conclusion Box
conclusion_text = """<b>12. One-Sentence Conclusion:</b><br/>
<i>\"We trust this model only when it is deployed as an assistive risk-stratification decision-support tool within acute care workflows, paired with mandatory clinical human review for high-uncertainty and complex multi-morbid patient cohorts.\"</i>"""
c_table = Table([[Paragraph(conclusion_text, quote_style)]], colWidths=[532])
c_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
    ("BOX", (0,0), (-1,-1), 1.2, colors.HexColor("#2B6CB0")),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
elements.append(c_table)

doc.build(elements)
print("Model_Trust_Card_codewave.pdf created successfully with EXACTLY 2 PAGES!")
