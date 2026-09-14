# 🏥 Beyond the Black Box: Statistics for Trustworthy AI

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Competition](https://img.shields.io/badge/Kaggle-ML%20%26%20AI%20Nexus%202026-20BEFF.svg?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/ml-nexus-2026)
[![Rank](https://img.shields.io/badge/Leaderboard-Rank%20%233%20(Top%203)-4CAF50.svg?style=flat)](#-key-results--benchmarks)
[![Log Loss](https://img.shields.io/badge/Best%20Log%20Loss-0.33400-brightgreen.svg?style=flat)](#-key-results--benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official repository of Team `codewave` for the ML & AI Nexus 2026 Hackathon.**  
> Predicting 30-day unplanned hospital readmission probabilities using a calibrated, uncertainty-aware, and trustworthy clinical AI ensemble.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Results & Benchmarks](#-key-results--benchmarks)
- [System Architecture](#-system-architecture)
- [Clinical Feature Engineering](#-clinical-feature-engineering)
- [Model Calibration & Uncertainty Referral](#-model-calibration--uncertainty-referral)
- [Subgroup Reliability & Algorithmic Fairness](#-subgroup-reliability--algorithmic-fairness)
- [Repository Structure](#-repository-structure)
- [Quickstart & Reproducibility](#-quickstart--reproducibility)
- [Model Trust Card](#-model-trust-card)
- [Team Members & Acknowledgments](#-team-members--acknowledgments)

---

## 🎯 Project Overview

Hospital readmission within 30 days is a pivotal quality and safety indicator in healthcare delivery. However, deploying predictive AI in acute hospital discharge planning presents severe operational and ethical challenges:
* **Asymmetric Risk:** Overconfident false negatives cause preventable clinical deterioration post-discharge.
* **Covariate Shift:** Patient demographics and comorbidity prevalence vary across cohorts and clinical settings.
* **The "Black Box" Problem:** Clinicians cannot act on uncalibrated, uninterpretable risk scores.

### The Challenge
Predict the true probability of 30-day unplanned readmission (`readmitted_30d`) evaluated under **Binary Log Loss**:
$$\mathcal{L}_{\log} = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \Big]$$

Rather than optimizing for raw accuracy with complex opaque models, our solution focuses on **statistical calibration**, **subgroup reliability**, **uncertainty quantification**, and **clinical decision support**.

---

## 🏆 Key Results & Benchmarks

| Metric | Out-of-Fold Cross-Validation (5-Fold) | Kaggle Public Test Set (30%) | Description |
| :--- | :---: | :---: | :--- |
| **Binary Log Loss** | **0.3490 ± 0.0055** | **0.33400** | **Rank #3 Globally** (Top Tier on Kaggle Leaderboard) |
| **Brier Score** | **0.1021 ± 0.0015** | — | Mean squared probability calibration error |
| **ROC-AUC** | **0.6932 ± 0.0199** | — | Multi-cohort discriminative capability |
| **PR-AUC** | **0.2812** | — | Precision-Recall trade-off at 1:7 class imbalance |
| **Sensitivity (Recall at $\tau = 0.126$)** | **61.18% ± 3.51%** | — | Correctly identifies ~61% of all readmissions |
| **Specificity (at $\tau = 0.126$)** | **67.07% ± 1.02%** | — | Correctly identifies ~67% of non-readmitted patients |

---

## 🔬 System Architecture

Our solution combines three fundamentally distinct model paradigms in an out-of-fold stacking ensemble, optimized via Sequential Least Squares Programming (SLSQP):

```
                       ┌─────────────────────────┐
                       │ Raw Patient EHR Records │
                       └────────────┬────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │ Clinical Domain Feature Engine    │
                  │ - Cardiorenal burden & AKI flags  │
                  │ - Polypharmacy & Care delays      │
                  │ - Multivariate MICE Lab Imputation│
                  └─────────────────┬─────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
  ┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
  │ CatBoost Classifier  ││ LightGBM Classifier  ││ Calibrated Lasso LR  │
  │ - Native Cat handling││ - Leaf-wise max_d=4  ││ - L1 Penalty (C=0.15)│
  │ - Depth=5, 1800 trees││ - num_leaves=24      ││ - Platt Sigmoid CV=3 │
  └──────────┬───────────┘└──────────┬───────────┘└──────────┬───────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │  Constrained SLSQP Stack Optimizer│
                   │  Out-of-Fold Log Loss Minimization│
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │  Platt Sigmoid Calibration &      │
                   │  Safe Range Clamping [0.025,0.728]│
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ Final Probabilities (Mean: 13.44%)│
                   └───────────────────────────────────┘
```

1. **CatBoost (Depth=5, 1800 trees):** Models complex interactions between discrete clinical categories (`care_pathway`, `discharge_disposition`) and laboratory vitals.
2. **LightGBM (Max Depth=4, Num Leaves=24):** Constrained shallow decision trees providing high-speed, non-linear risk stratification.
3. **L1-Penalized Lasso Logistic Regression (C=0.15):** Regularized linear anchor preventing tree over-reliance and stabilizing baseline probabilities.

---

## 🧪 Clinical Feature Engineering

All engineered features are grounded in evidence-based clinical guidelines:
* **Acute Kidney Injury (AKI):** Serum Creatinine $> 1.5$ mg/dL and severe AKI ($> 3.0$ mg/dL).
* **Anemia Stratification:** Hemoglobin $< 11.0$ g/dL (mild) and $< 9.0$ g/dL (severe).
* **Cardiorenal Syndrome:** Comorbid Chronic Kidney Disease (CKD) and Congestive Heart Failure (CHF).
* **Care Coordination Gaps:** Outpatient follow-up delay $> 14$ days post-discharge.
* **Polypharmacy Index:** Active medication count $\ge 10$.
* **Missingness Patterns:** Systematic missing-value indicators identifying unmonitored outpatient labs.

---

## 📊 Model Calibration & Uncertainty Referral

### Probability Calibration
Uncalibrated tree ensembles produce compressed risk probabilities. We applied **Platt Sigmoid Scaling** to fit out-of-fold logits, dropping the Brier score from **0.1249** to **0.1021**.

### Selective Classification (Human-in-the-Loop Abstention)
Using normalized Shannon Predictive Entropy:
$$H(p) = - \big[ p \ln p + (1 - p) \ln(1 - p) \big]$$

When the model defers the top 10% most ambiguous predictions ($p \approx 0.12 - 0.18$) for multidisciplinary human clinical review:
* **Log Loss drops from 0.3490 → 0.3158** (9.5% relative error reduction).
* **Brier score improves from 0.1021 → 0.0882** (13.6% improvement).

---

## ⚖️ Subgroup Reliability & Algorithmic Fairness

To ensure equity across diverse patient demographics, out-of-fold predictions were audited across strata:

| Subgroup Category | Stratum | N (%) | Base Rate | Log Loss | ROC-AUC | Brier Score | Equity Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sex** | Female<br>Male | 3,543 (50.6%)<br>3,457 (49.4%) | 12.39%<br>12.79% | 0.3449<br>0.3531 | 0.6915<br>0.6939 | 0.1005<br>0.1037 | **Parity Maintained** (ΔAUC = 0.0024) |
| **Rurality** | Rural<br>Semi-urban<br>Urban | 1,243 (17.8%)<br>1,870 (26.7%)<br>3,887 (55.5%) | 15.61%<br>10.91%<br>12.43% | 0.3926<br>0.3166<br>0.3506 | 0.7080<br>0.7017<br>0.6791 | 0.1190<br>0.0905<br>0.1023 | Higher rural baseline reflects access barriers, not model bias. |
| **Age Group** | Young (<40)<br>Middle (40-60)<br>Senior (60-75)<br>Elderly (>75) | 1,052 (15.0%)<br>2,981 (42.6%)<br>2,033 (29.0%)<br>934 (13.3%) | 4.47%<br>9.86%<br>15.30%<br>24.20% | 0.1835<br>0.3179<br>0.4047<br>0.5131 | 0.5880<br>0.6010<br>0.6641<br>0.6860 | AUC scales with comorbidity density in older cohorts. |
| **Hospital Type** | District<br>General<br>Teaching | 1,285 (18.4%)<br>2,623 (37.5%)<br>3,092 (44.2%) | 12.37%<br>12.20%<br>13.00% | 0.3387<br>0.3410<br>0.3600 | 0.7161<br>0.6911<br>0.6845 | 0.0990<br>0.0990<br>0.1060 | Consistent performance across all hospital tiers. |

---

## 📁 Repository Structure

```text
nexus/
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Production environment dependencies
├── .gitignore                         # Git exclusion rules
├── codewave_solution.ipynb            # Clean, end-to-end reproducible Jupyter Notebook
│
├── data/                              # Competition datasets
│   ├── train.csv                      # Training dataset (7,000 patient records)
│   ├── test.csv                       # Test dataset (3,000 patient records)
│   ├── data_dictionary.csv            # Variable definitions & clinical types
│   └── sample_submission.csv          # Submission file benchmark
│
├── docs/                              # Model Trust Card & Competition Briefs
│   ├── Model_Trust_Card_codewave.docx # Official 2-Page Model Trust Card (Word)
│   ├── Model_Trust_Card_codewave.pdf  # Official 2-Page Model Trust Card (PDF)
│   ├── Model_Trust_Card_codewave.md   # Model Trust Card Markdown text
│   ├── Competition_Overview.pdf       # Hackathon problem description
│   └── Competition_Guidelines.pdf     # Rules, presentation & evaluation criteria
│
├── src/                               # Modular python source code
│   ├── __init__.py
│   ├── pipeline.py                    # Production training, calibration & prediction script
│   ├── trust_card_stats.py            # Statistical audit (Fairness, Calibration, Abstention)
│   └── generate_trust_card.py         # Trust Card docx/pdf generation script
│
├── submissions/                       # Top verified Kaggle submission files
│   ├── codewave_submission_12.csv     # Winning Final Model (Score: 0.33400)
│   ├── codewave_submission_09.csv     # Conservative Multi-Model Blend (Score: 0.33416)
│   └── codewave_submission_11.csv     # Alternative High-performing Blend (Score: 0.33418)
│
└── experiments/                       # Development scratch iterations (Archived)
    ├── scripts/                       # Iterative experimental scripts
    └── submissions/                   # Early submission iterations
```

---

## 🚀 Quickstart & Reproducibility

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/YOUR_USERNAME/nexus-readmission-prediction.git
cd nexus-readmission-prediction

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Full Model Pipeline
To execute feature engineering, 5-fold cross-validation, and generate predictions:
```bash
python src/pipeline.py
```
*Output: `submissions/codewave_submission_12.csv`*

### 3. Run Statistical Audit & Trust Card Metrics
To re-evaluate subgroup reliability, calibration deciles, and the human referral abstention experiment:
```bash
python src/trust_card_stats.py
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook codewave_solution.ipynb
```

---

## 📄 Model Trust Card

Our comprehensive **Two-Page Model Trust Card** adheres to trustworthy AI governance standards.
* 📄 **PDF Version:** [`docs/Model_Trust_Card_codewave.pdf`](docs/Model_Trust_Card_codewave.pdf)
* 📝 **Word Version:** [`docs/Model_Trust_Card_codewave.docx`](docs/Model_Trust_Card_codewave.docx)
* 🌐 **Markdown:** [`docs/Model_Trust_Card_codewave.md`](docs/Model_Trust_Card_codewave.md)

### Deployment Recommendation
> **Ready for limited prospective validation as an assistive decision-support tool.**  
> The model is recommended exclusively to prioritize post-discharge nurse navigator follow-up calls. It must **never** be used as an autonomous discharge clearance gatekeeper. Patients falling into the top 10% uncertainty band mandate human clinical multidisciplinary review.

---

## 👥 Team Members & Acknowledgments

* **Team Name:** `codewave`
* **Lead Author & ML Engineer:** Sahil Kavishka (`22cds0428@ms.sab.ac.lk`)
* **Event:** ML & AI Nexus 2026 Hackathon
* **Theme:** *"Beyond the Black Box: Statistics for Trustworthy AI"*

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
