# 🎓 Project — Student Dropout Predictive Analytics

> **Production-grade EDA and machine learning pipeline exploring dropout behavior, engagement patterns, and risk segmentation on an online learning platform.**

---

## 📁 Repository Structure

```
Data-Analytics-
│
├── Project — Student Dropout Analysis/     ← Student Dropout Predictive Analytics
│   ├── Dataset/                            ← Raw dataset (not tracked — see Dataset Info)
│   ├── Notebook/                           ← Analysis scripts and ML pipeline
│   │   └── student_dropout_analysis.py     ← Production Python EDA + ML script
│   └── Outputs/                            ← Generated charts, plots, and reports
│       ├── fig1_overview.png               ← KPI dashboard + regional & temporal breakdown
│       ├── fig2_eda.png                    ← Correlation heatmap + deep EDA panels
│       ├── fig3_models.png                 ← ML model results, ROC curves, feature importance
│       └── student_dropout_analysis_report.pdf  ← Full executive PDF report
│
├── .gitignore                              ← Excludes large dataset files
├── requirements.txt                        ← Python dependencies
└── README.md                               ← You are here
```

---

## 📌 Project Overview

This project performs a **complete end-to-end Data Science workflow** on an online learning platform's student engagement dataset — covering EDA, feature engineering, behavioral segmentation, and predictive modeling to identify students at risk of dropping out before course completion.

Each record in the dataset represents a single student enrollment and captures:
- Academic engagement metrics: assignments completed, course completion rate
- Platform activity: login frequency, last activity timestamp, forum participation
- Enrollment context: region, enroll date, exam season flag
- Pre-computed risk signal: a continuous dropout score (0–1)

### 💼 Why This Analysis Matters

Student dropout is one of the most pressing problems in online education. With completion rates on major MOOC platforms averaging below 15%, identifying at-risk students early enables targeted intervention at scale.

| Business Metric | Impact |
|---|---|
| Dropout Rate | 65.9% of students failed to complete — a critical retention problem |
| Revenue Leakage | Every dropout represents lost lifetime value and platform credibility |
| Intervention ROI | Early identification reduces support costs vs. post-dropout recovery |
| Platform Reputation | Completion rates directly affect institutional and corporate partnerships |
| Personalization | Behavioral clusters enable tailored nudges rather than blanket campaigns |

---

## 📊 Dataset Info

| Property | Detail |
|---|---|
| **Source** | Student Dropout Dataset (CSV) |
| **Rows** | 5,000 students |
| **Columns** | 15 raw + 8 engineered features |
| **Format** | CSV |
| **Domain** | EdTech / Online Learning Behavior |
| **Cohort Period** | January 2024 — December 2024 |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `student_id` | string | Unique identifier per student |
| `age` | int | Student age (17–40) |
| `region` | categorical | City of enrollment (10 MENA cities) |
| `enroll_date` | date | Date of course enrollment |
| `exam_season` | binary | 1 = enrolled during exam season |
| `courses_enrolled` | int | Number of courses enrolled in |
| `completed_assignments` | int | Number of assignments submitted |
| `completion_rate` | float | Proportion of course content completed (0–1) |
| `login_frequency` | float | Average logins per week |
| `last_activity_days_ago` | int | Days since last platform interaction |
| `forum_posts_count` | int | Number of forum contributions |
| `dropout_score` | float | Pre-computed dropout risk signal (0 = low risk, 1 = high risk) |
| `label` | binary | Target — 1 = dropout, 0 = active |
| `label_multiclass` | int | 0 = active, 1 = at-risk, 2 = dropped |
| `label_name` | categorical | Human-readable label: active / at-risk / dropped |

> ⚠️ **Dataset not tracked by Git.** Place the CSV file inside `Dataset/` before running the notebook.

---

## 🔬 Analysis Walkthrough

The notebook follows a structured, 10-section analytics workflow:

### 1. 📋 Dataset Understanding
Business framing, column classification (numerical / categorical / ordinal / target), and domain context.

### 2. 🔍 Data Quality Assessment
- Zero missing values confirmed across all 15 columns
- Zero duplicate student records
- IQR-based outlier detection across all numerical features
- Datatype correctness validated

### 3. 🧹 Data Cleaning Plan
- No imputation required (complete dataset)
- Outliers retained — flagged as valid behavioral extremes (e.g. highly inactive students)
- Label encoding for categorical variables
- RobustScaler applied in ML pipeline to handle skewed engagement distributions

### 4. 📈 Exploratory Data Analysis

| Chart | What It Shows |
|---|---|
| `fig1_overview.png` | 4-card KPI summary, label distribution pie, regional dropout rates, monthly enrollment trend, completion/login/inactivity distributions, forum & engagement comparison |
| `fig2_eda.png` | Full correlation heatmap, dropout score distributions, age distributions, engagement vs. dropout scatter, login vs. completion scatter, assignments-per-course violin plots |
| `fig3_models.png` | ROC curves (3 models), model performance bar chart, feature importance (RF), binary & multiclass confusion matrices, calibration plot |

### 5. 🛠️ Feature Engineering

Eight new analytical features derived from raw columns:

| Feature | Formula / Logic | Purpose |
|---|---|---|
| `enroll_month` | `enroll_date.dt.month` | Captures seasonal enrollment patterns |
| `enroll_quarter` | `enroll_date.dt.quarter` | Quarter-level temporal signal |
| `assignments_per_course` | `completed_assignments / courses_enrolled` | Normalised academic effort |
| `engagement_score` | `completion_rate×0.4 + login_norm×0.35 + forum_norm×0.25` | Composite behavioral engagement index |
| `recency_risk` | `last_activity_days_ago / max` | Normalised inactivity risk signal |
| `is_inactive` | `last_activity_days_ago > 30 → 1` | Binary hard-inactivity flag |
| `zero_logins` | `login_frequency < 1 → 1` | Ghost enrollment indicator |
| `high_performer` | `completion_rate > 0.7 AND login_freq > 5 → 1` | Top student identifier |

### 6. 🤖 Predictive Modeling

Three classifiers evaluated on both binary (dropout / active) and multiclass (active / at-risk / dropped) tasks:

| Model | Accuracy | AUC-ROC |
|---|---|---|
| **Random Forest** ⭐ | **93.4%** | **0.982** |
| Gradient Boosting | ~91% | ~0.970 |
| Logistic Regression | ~88% | ~0.950 |

**Multiclass Random Forest Performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Active | 0.91 | 0.87 | 0.89 |
| At-Risk | 0.77 | 0.77 | 0.77 |
| Dropped | 0.86 | 0.90 | 0.88 |
| **Overall** | **0.85** | **0.84** | **0.84** |

**Top Predictive Features (Random Forest Importance):**
1. `dropout_score` — strongest single predictor
2. `engagement_score` — engineered composite feature
3. `completion_rate` — core academic signal
4. `login_frequency` — platform stickiness
5. `forum_posts_count` — community participation

### 7. 💡 Business Insights

8 structured, actionable findings — see [Key Findings](#-key-findings) below.

### 8. 🏗️ Production Pipeline Design
Data ingestion → validation → feature store → model training → monitoring recommendations for a real production environment.

### 9. 🐍 Python Implementation
Clean, modular, production-style code with full sklearn Pipeline objects, RobustScaler, stratified cross-validation, and artifact export.

### 10. 📄 Executive Summary
Board-ready takeaways for both engineering and business stakeholders.

---

## 📸 Output Gallery

<table>
  <tr>
    <td><img src="Outputs/fig1_overview.png" width="380"/><br><sub>KPI Overview & Regional Breakdown</sub></td>
    <td><img src="Outputs/fig2_eda.png" width="380"/><br><sub>Correlation Heatmap & Deep EDA</sub></td>
  </tr>
  <tr>
    <td><img src="Outputs/fig3_models.png" width="380"/><br><sub>ML Model Results & Feature Importance</sub></td>
    <td><img src="Outputs/student_dropout_analysis_report.pdf" width="380"/><br><sub>Full Executive PDF Report</sub></td>
  </tr>
</table>

---

## 🔑 Key Findings

1. **65.9% overall dropout rate** — only 1 in 3 students completes their enrolled courses, representing a systemic retention failure requiring urgent intervention.
2. **Engagement score is the single strongest behavioral predictor** — completion rate, login frequency, and forum participation together explain the vast majority of dropout variance, with AUC of 0.982 on the binary task.
3. **Dropped students log in 7.7× less than active students** — login frequency averages 6.67/week for active students vs. 0.86/week for dropped students, making it the most operationally actionable early-warning signal.
4. **Inactivity beyond 30 days is near-certain dropout** — students flagged by `is_inactive` show dramatically higher dropout rates; a 30-day inactivity threshold is a reliable automated alert trigger.
5. **Regional dropout rates vary by up to 8.6 percentage points** — Doha (69.5%) and Beirut (69.3%) significantly outpace Alexandria (60.9%), suggesting region-specific UX, support, or curriculum factors.
6. **October and April have the highest dropout rates by enrollment month** — likely aligned with academic exam cycles, validating the `exam_season` feature as a confounding factor worth controlling for in interventions.
7. **At-risk students are the hardest to classify** (F1 = 0.77) — the boundary between at-risk and dropped is blurry, suggesting a need for a continuous risk score rather than discrete labels in production systems.
8. **Zero-login students represent a distinct ghost cohort** — the `zero_logins` flag identifies students who never engaged post-enrollment, a segment better addressed through onboarding redesign than retention campaigns.

---

## 🛠️ Tech Stack

```
Python 3.12
├── pandas          — data manipulation, cleaning, and feature engineering
├── numpy           — numerical operations and derived metrics
├── matplotlib      — base charting engine and dashboard layout
├── seaborn         — statistical visualizations and correlation heatmaps
└── scikit-learn    — Pipeline, RobustScaler, RandomForest, GradientBoosting,
                      LogisticRegression, ROC-AUC, ConfusionMatrix
```

---

## ⚡ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Data-Analytics-.git
cd "Data-Analytics-/Project — Student Dropout Analysis"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place student_dropout_dataset.csv inside Dataset/

# 4. Run the full analysis
python Notebook/student_dropout_analysis.py

# Charts will be saved to Outputs/
```

---

## 📦 requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
reportlab>=4.0.0
```

---

## 🗺️ Dashboard Recommendations

This analysis is ready to be operationalised into a live intervention and monitoring layer:

| Tool | Suggested Dashboard |
|---|---|
| **Streamlit** | Real-time dropout risk scorer — input student metrics, output risk probability |
| **Power BI** | Regional dropout heatmap + weekly engagement trend by cohort |
| **Tableau** | Student funnel by label stage with engagement drill-through |
| **Looker** | Cluster segmentation dashboard for academic advisor CRM targeting |

---

## 🚀 Future Work

| Direction | Description |
|---|---|
| 🎯 Real-Time Risk Scoring | Deploy Random Forest as a REST API to score students daily on latest activity data |
| 🔔 Automated Intervention Engine | Trigger email/SMS nudges at 7-day, 14-day, and 30-day inactivity thresholds |
| 🧪 A/B Testing Framework | Test intervention formats (peer nudge vs. instructor message vs. discount offer) per segment |
| 🔄 Sequence Modelling | LSTM on weekly engagement time series to capture deteriorating momentum before dropout |
| 🌍 Region-Specific Models | Train separate models per city to capture local dropout drivers and cultural context |
| 📡 Live Pipeline | Airflow + Snowflake pipeline feeding a Streamlit early-warning dashboard for academic advisors |

---

## 👤 Author

**Senior Data Scientist & Analytics Engineer**
Portfolio project demonstrating production-grade EDA, feature engineering, multi-model evaluation, and executive reporting on real-world EdTech data.

---

## 📄 License

This project is licensed under the MIT License.

---

*Part of the [Data-Analytics- Portfolio](../README.md) — a collection of end-to-end analytics projects.*
