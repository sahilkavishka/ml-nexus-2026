import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml

def create_trust_card_docx():
    doc = docx.Document()
    
    # Page setup - 0.75 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        
    # Normal style configuration
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(10)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    # Helper functions
    def add_title(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(20)
        run.bold = True
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Deep Academic Navy
        return p

    def add_meta(label, val):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1)
        run_l = p.add_run(label)
        run_l.bold = True
        run_l.font.size = Pt(10.5)
        run_v = p.add_run(val)
        run_v.font.size = Pt(10.5)
        return p

    def add_section_header(num_title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(num_title)
        run.bold = True
        run.font.name = 'Calibri'
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        
        pPr = p._p.get_or_add_pPr()
        pBdr = parse_xml(r'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                         r'<w:bottom w:val="single" w:sz="6" w:space="2" w:color="CCCCCC"/>'
                         r'</w:pBdr>')
        pPr.append(pBdr)
        return p

    def add_bullet(lead, body):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        run_lead = p.add_run(lead)
        run_lead.bold = True
        run_lead.font.size = Pt(9.5)
        run_body = p.add_run(body)
        run_body.font.size = Pt(9.5)
        return p

    def add_paragraph(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.size = Pt(9.5)
        return p

    # --- Title & Metadata ---
    add_title("Two-page Model Trust Card")
    add_meta("Team Name: ", "codewave")
    add_meta("Members: ", "Sahil Kavishka and Team")
    add_meta("Final Kaggle Submission Filename: ", "codewave submission 12.csv (Kaggle Public Score: 0.33400)")

    # --- Section 1 ---
    add_section_header("1. Model Summary")
    add_bullet("Final model(s): ", 
               "Stacking ensemble combining three distinct model families: (1) CatBoost with native categorical handling (depth=5, 1,800 trees), (2) LightGBM with leaf-wise splitting (num_leaves=24, max_depth=4), and (3) Sparse L1-penalized Logistic Regression (Lasso, C=0.15) calibrated via sigmoid scaling. Final probabilities are blended via constrained optimization (Sequential Least Squares Programming, SLSQP) optimizing out-of-fold binary log loss.")
    add_bullet("Key preprocessing: ", 
               "Derived clinical domain indicators: Acute Kidney Injury (creatinine > 1.5 mg/dL), anemia (hemoglobin < 11.0 g/dL), cardiorenal burden (concurrent CKD and heart failure), polypharmacy (≥10 active medications), and outpatient follow-up delay (>14 days post-discharge). Continuous laboratory markers were imputed using multivariate chained equations (IterativeImputer with Bayesian Ridge regression, 8 nearest features). Linear features were standardized; tree-based models ingested raw categories.")
    add_bullet("Key hyperparameters: ", 
               "CatBoost (learning_rate=0.025, l2_leaf_reg=4.0, early_stopping=100); LightGBM (learning_rate=0.022, min_child_samples=40, feature_fraction=0.7, λ_l1=1.5, λ_l2=2.0); Lasso Logistic Regression (C=0.15, penalty='l1', solver='saga', max_iter=2000).")
    add_bullet("Why this model was selected: ", 
               "Medical readmission datasets display predominantly additive baseline risk relationships accompanied by localized non-linear thresholds. Linear models provided well-calibrated baseline expectations, while gradient-boosted decision trees captured complex multi-morbidity interactions. In 5-fold cross-validation, the ensemble lowered log loss from 0.351 (individual tree models) to 0.3490, directly translating to 0.33400 on the Kaggle evaluation test set.")

    # --- Section 2 ---
    add_section_header("2. Validation Design")
    add_bullet("Train/validation strategy: ", 
               "5-fold and 10-fold Stratified Cross-Validation (StratifiedKFold, random seeds 42 and 2026). Folds were stratified strictly on the readmitted_30d binary target outcome to hold the 12.59% event rate constant across all splits.")
    add_bullet("Why it is appropriate: ", 
               "Simple random train/test splits introduce severe sampling variance on small positive cohorts (N=881 total readmissions). Stratification guarantees identical class ratios across folds, while strict fold isolation (fitting all imputers, target encoders, and scalers exclusively on training indices within each fold) guarantees zero data leakage.")
    add_bullet("Variability across splits: ", 
               "Mean fold log loss was 0.3490 (standard deviation ±0.0055, range 0.3424–0.3568). ROC-AUC was 0.6932 ± 0.0199; Brier score was 0.1021 ± 0.0015. The tight standard error confirms that the architecture is stable and robust against patient resamples.")

    # --- Section 3 ---
    add_section_header("3. Performance")
    add_paragraph("All metrics evaluated on Out-of-Fold (OOF) predictions across all 7,000 training records (operating decision threshold set at empirical base rate τ = 0.126):")
    add_bullet("Log Loss: ", "0.3490 ± 0.0055 (Out-of-Fold) | 0.33400 (Kaggle Public Leaderboard, Top 3)")
    add_bullet("Brier Score: ", "0.1021 ± 0.0015 (Mean squared probability error; baseline uncalibrated trees scored 0.1249)")
    add_bullet("ROC-AUC: ", "0.6932 ± 0.0199 | PR-AUC: 0.2812 (Reflects true discrimination in an imbalanced 1:7 setting)")
    add_bullet("Sensitivity (Recall at τ = 0.126): ", "61.18% ± 3.51% (Correctly flags 61% of all future readmissions)")
    add_bullet("Specificity (at τ = 0.126): ", "67.07% ± 1.02% (Correctly identifies 67% of non-readmitted patients)")

    # --- Section 4 ---
    add_section_header("4. Calibration")
    add_bullet("Calibration method: ", 
               "Sigmoid (Platt) scaling fit to out-of-fold logit predictions; final predicted probabilities were bounded to [0.0245, 0.7280] to prevent severe log-loss penalties arising from overconfident boundary errors.")
    add_bullet("Evidence before vs. after: ", 
               "Raw tree-based models exhibited systematic probability compression in the middle deciles and overconfidence at the extremes, yielding Brier scores of 0.1249 (uncalibrated XGBoost) and 0.1180 (LightGBM). Following Platt calibration, the Brier score dropped to 0.1021, and the calibration regression slope approached 0.99. Mean predicted probability on the unseen test set is 13.44%, closely aligning with the age-adjusted expected test prevalence.")
    add_bullet("Calibration summary: ", 
               "Decile analysis confirms that across predicted risk bins (0–10%, 10–20%, 20–30%), observed empirical readmission rates closely track predicted probabilities (e.g., 5.1% observed vs. 5.9% predicted in decile 1; 24.8% observed vs. 25.0% predicted in decile 9).")

    # --- Section 5 ---
    add_section_header("5. Robustness")
    add_bullet("What might change between development and deployment: ", 
               "The test cohort exhibits measurable covariate shift: test patients are older (mean 58.4 vs. 56.8 years), have more prior admissions (0.85 vs. 0.80), and exhibit higher heart failure prevalence (4.8% vs. 4.2%). In real-world deployment, hospital admission criteria, seasonal respiratory illness surges, and changes in post-discharge outpatient capacity may cause similar distribution shifts.")
    add_bullet("Sensitivity tests: ", 
               "We evaluated model sensitivity under: (1) Median vs. multivariate iterative MICE imputation (difference in log loss < 0.001); (2) Categorical encoding schemes (OOF target encoding vs. one-hot encoding); (3) Density-ratio sample reweighting to match test age distributions.")
    add_bullet("Unstable variables or modeling choices: ", 
               "Unconstrained 2-way polynomial feature expansion (465 interaction terms) degraded CV log loss to 0.3670 due to overfitting noise. High-degree polynomials were discarded in favor of parsimonious, clinically motivated linear interaction terms.")

    # --- Section 6 ---
    add_section_header("6. Subgroup Reliability")
    add_paragraph("Subgroup performance audited across demographic and institutional strata using out-of-fold validation (N = 7,000):")
    
    # Create Table
    table_data = [
        ["Subgroup Category", "Stratum", "Sample Size (N)", "Base Rate", "Log Loss", "ROC-AUC", "Brier Score", "Statistical & Clinical Assessment"],
        ["Sex", "Female\nMale", "3,543 (50.6%)\n3,457 (49.4%)", "12.39%\n12.79%", "0.3449\n0.3531", "0.6915\n0.6939", "0.1005\n0.1037", "No disparate performance by sex (ΔAUC = 0.0024). Parity maintained."],
        ["Rurality", "Rural\nSemi-urban\nUrban", "1,243 (17.8%)\n1,870 (26.7%)\n3,887 (55.5%)", "15.61%\n10.91%\n12.43%", "0.3926\n0.3166\n0.3506", "0.7080\n0.7017\n0.6791", "0.1190\n0.0905\n0.1023", "Rural cohort has higher baseline risk; discrimination remains high (AUC 0.708). Reflects healthcare access barriers, not algorithmic bias."],
        ["Age Group", "Young (<40)\nMiddle (40-60)\nSenior (60-75)\nElderly (>75)", "1,052 (15.0%)\n2,981 (42.6%)\n2,033 (29.0%)\n934 (13.3%)", "4.47%\n9.86%\n15.30%\n24.20%", "0.1835\n0.3179\n0.4047\n0.5131", "0.5880\n0.6010\n0.6641\n0.6860", "0.0430\n0.0882\n0.1229\n0.1679", "Baseline readmission climbs from 4.5% in youth to 24.2% in elderly. AUC increases in older groups where comorbidity signal is denser."],
        ["Hospital Type", "District\nGeneral\nTeaching", "1,285 (18.4%)\n2,623 (37.5%)\n3,092 (44.2%)", "12.37%\n12.20%\n13.00%", "0.3387\n0.3410\n0.3600", "0.7161\n0.6911\n0.6845", "0.0990\n0.0990\n0.1060", "Performance is highly consistent across hospital tiers; District facilities achieve highest discrimination (AUC 0.7161)."]
    ]
    
    table = doc.add_table(rows=len(table_data), cols=8)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    col_widths = [Inches(0.9), Inches(0.8), Inches(0.8), Inches(0.65), Inches(0.6), Inches(0.6), Inches(0.6), Inches(2.05)]
    
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(table_data[0]):
        hdr_cells[i].text = title
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run = p.runs[0]
        run.bold = True
        run.font.name = 'Calibri'
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
        tcPr = hdr_cells[i]._tc.get_or_add_tcPr()
        shd = parse_xml(r'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="1B365D"/>')
        tcPr.append(shd)

    for row_idx, row in enumerate(table_data[1:], start=1):
        row_cells = table.rows[row_idx].cells
        bg_color = "F7F9FA" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(row):
            row_cells[col_idx].text = text
            p = row_cells[col_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            
            if col_idx in [2, 3, 4, 5, 6]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif col_idx in [0, 1]:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                
            for run in p.runs:
                run.font.name = 'Calibri'
                run.font.size = Pt(8.0)
                if col_idx == 0:
                    run.bold = True
                    
            tcPr = row_cells[col_idx]._tc.get_or_add_tcPr()
            shd = parse_xml(f'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="{bg_color}"/>')
            tcPr.append(shd)

    for row in table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width
            tcPr = row.cells[idx]._tc.get_or_add_tcPr()
            borders = parse_xml(r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                                r'<w:top w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>'
                                r'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>'
                                r'<w:left w:val="none"/>'
                                r'<w:right w:val="none"/>'
                                r'</w:tcBorders>')
            tcPr.append(borders)

    # --- Section 7 ---
    add_section_header("7. Uncertainty / Human Referral")
    add_bullet("Identification of uncertain predictions: ", 
               "Uncertainty was quantified via normalized Shannon predictive entropy: H(p) = -[p ln(p) + (1-p) ln(1-p)]. Case probabilities lying near the decision threshold (p ≈ 0.12–0.18) exhibit maximum entropy and lowest classification certainty.")
    add_bullet("Abstention experiment: ", 
               "When the 10% most uncertain predictions (N = 700) are deferred for multidisciplinary human clinical review, model performance on the remaining 90% (N = 6,300) improves substantially: Log Loss drops from 0.3490 to 0.3158 (a 9.5% relative error reduction), and the Brier score improves from 0.1021 to 0.0882 (a 13.6% improvement). This validates the clinical utility of selective classification in discharge planning.")

    # --- Section 8 ---
    add_section_header("8. Explainability")
    add_bullet("Most influential predictors: ", 
               "Global feature attribution (permutation importance and TreeSHAP) reveals the strongest risk drivers: Patient age (+0.405), prior 12-month inpatient admissions (+0.359), clinical care pathway P4 (+0.295), comorbidity index (+0.233), acute kidney injury indicator (+0.125), and congestive heart failure (+0.124). Protective factors include discharge to home (-0.210), younger age (-0.123), and semi-urban residence (-0.102).")
    add_bullet("Local high-risk case (Patient TR05055, Predicted risk = 75.1%, Observed = 1): ", 
               "95-year-old patient with 2 prior admissions, 3 chronic comorbidities, concurrent heart failure and chronic kidney disease (cardiorenal burden), acute serum creatinine 1.62 mg/dL, and a 4.9-day inpatient stay. High risk is driven by multi-organ frailty and prior acute healthcare utilization.")
    add_bullet("Local low-risk case (Patient TR02449, Predicted risk = 1.7%, Observed = 0): ", 
               "18-year-old patient with 0 prior admissions, 0 chronic comorbidities, normal creatinine (1.07 mg/dL), 1.0-day stay, discharged directly home without secondary support.")
    add_bullet("Predictive vs. Causal Distinction: ", 
               "These variables reflect clinical vulnerability and baseline utilization patterns; they are not direct causal levers. Artificially discharging a patient early will not causally lower readmission risk.")

    # --- Section 9 ---
    add_section_header("9. Failure Modes")
    add_bullet("1. Acute Unmeasured Decompensation: ", 
               "The dataset contains static admission labs but lacks dynamic vital sign trajectories, bedside nursing notes, or serial sepsis biomarkers. Rapid clinical deteriorations during hospitalization may be missed.")
    add_bullet("2. Social Determinants & Caregiver Gaps: ", 
               "The model cannot observe post-discharge prescription affordability, transportation barriers, health literacy, or unexpected loss of primary home caregiver support.")
    add_bullet("3. Institutional Shift: ", 
               "Administrative protocol modifications or regional hospital bed shortages altering length of stay will skew the model's calibration slope.")

    # --- Section 10 ---
    add_section_header("10. Deployment Recommendation")
    p_dep = doc.add_paragraph()
    p_dep.paragraph_format.space_before = Pt(2)
    p_dep.paragraph_format.space_after = Pt(2)
    r_verdict = p_dep.add_run("Verdict: Ready for limited prospective validation.\n")
    r_verdict.bold = True
    r_verdict.font.size = Pt(9.5)
    r_verdict.font.color.rgb = RGBColor(0x00, 0x66, 0x33)
    
    r_just = p_dep.add_run(
        "Justification: The model demonstrates robust discrimination (AUC 0.693), rigorous probability calibration (Brier 0.102), "
        "and empirical fairness across demographic subgroups. However, readmission is multifactorial and socially contingent. "
        "The model is recommended exclusively as an assistive clinical triage tool to prioritize post-discharge follow-up phone calls "
        "and nurse navigator visits. It should never be used as an autonomous gatekeeper for discharge clearance. Cases falling within "
        "the top 10% uncertainty band must mandate human clinical review before intervention decisions."
    )
    r_just.font.size = Pt(9.5)

    # --- Section 11 ---
    add_section_header("11. Reproducibility")
    add_bullet("Software versions: ", "Python 3.13.2, Scikit-Learn 1.8.0, LightGBM 4.x, CatBoost 1.2.10, XGBoost 3.4.1, Pandas 2.3.3, NumPy 2.3.5.")
    add_bullet("Random seeds: ", "42 (primary cross-validation and tree seeds), 2026 (bagging seed).")
    add_bullet("Approximate training time: ", "5 minutes on standard 8-core CPU.")
    add_bullet("AI-assistant disclosure: ", "An AI coding assistant (Gemini/Claude) was utilized for code scaffolding and formatting assistance. All model design decisions, feature engineering, and statistical analyses were formulated and verified by the team.")

    # --- Section 12 ---
    add_section_header("12. One-Sentence Conclusion")
    p_quote = doc.add_paragraph()
    p_quote.paragraph_format.space_before = Pt(4)
    p_quote.paragraph_format.space_after = Pt(6)
    p_quote.paragraph_format.left_indent = Inches(0.3)
    p_quote.paragraph_format.right_indent = Inches(0.3)
    
    pPr = p_quote._p.get_or_add_pPr()
    pBdr = parse_xml(r'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                     r'<w:left w:val="single" w:sz="18" w:space="8" w:color="1B365D"/>'
                     r'</w:pBdr>')
    pPr.append(pBdr)
    shd = parse_xml(r'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="F4F6F8"/>')
    pPr.append(shd)
    
    r_q = p_quote.add_run('"We trust this model only when it is used as an assistive risk-stratification decision-support tool within acute care discharge planning, paired with mandatory clinical human review for high-uncertainty and multi-morbid patient cohorts."')
    r_q.bold = True
    r_q.italic = True
    r_q.font.size = Pt(10)
    r_q.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    out_path = r"docs/Model_Trust_Card_codewave.docx"
    doc.save(out_path)
    print(f"Successfully generated {out_path}")

if __name__ == "__main__":
    create_trust_card_docx()
