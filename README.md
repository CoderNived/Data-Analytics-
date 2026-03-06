# 📊 Data Analytics Portfolio — Exploratory Data Analysis & Insights

> *Turning raw data into actionable insights through rigorous EDA, statistical analysis, and professional-grade visualizations.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()

---

## 🧭 Overview

This repository is a **professional data analytics portfolio** showcasing end-to-end exploratory data analysis across real-world datasets. Each project follows a structured analytical pipeline — from raw data ingestion and cleaning through to deep statistical exploration and visual storytelling.

| What | Detail |
|---|---|
| **Purpose** | Demonstrate production-quality EDA and analytical thinking |
| **Problems Solved** | Uncovering patterns, trends, and outliers hidden in complex datasets |
| **Datasets Used** | Large-scale open-source and public datasets (e.g. GitHub repositories metadata) |
| **Goal** | Extract meaningful insights that support data-driven decision making |

---

## 🗂️ Repository Structure

```
Data-Analytics-
│
├── Project 1/                        ← GitHub Open-Source Ecosystem Analysis
│   ├── Dataset/                      ← Raw dataset (not tracked — see Dataset Info)
│   ├── Notebook/                     ← Jupyter notebooks with full EDA walkthrough
│   │   └── github_eda_analysis.py    ← Production Python EDA script
│   └── Outputs/                      ← Generated charts, plots, and reports
│       ├── 01_distributions.png
│       ├── 02_language_analysis.png
│       ├── 03_top_repos.png
│       ├── 04_correlation_heatmap.png
│       ├── 05_scatter_relationships.png
│       ├── 06_boxplots_language_stars.png
│       ├── 07_activity_popularity.png
│       ├── 08_size_bucket_analysis.png
│       └── interactive_dashboard.html
│
├── .gitignore                        ← Excludes large dataset files
├── requirements.txt                  ← Python dependencies
└── README.md                         ← You are here
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core programming language for all analysis and scripting |
| **Pandas** | Data loading, cleaning, transformation, and aggregation |
| **NumPy** | Numerical operations, log transformations, and statistical computing |
| **Matplotlib** | Base layer for all static chart rendering and figure management |
| **Seaborn** | High-level statistical visualizations — heatmaps, boxplots, distributions |
| **Plotly** | Interactive HTML dashboards for exploratory deep-dives |
| **Jupyter Notebook** | Iterative, narrative-driven analysis environment |

---

## 🔄 Data Analysis Workflow

Every project in this repository follows a consistent, professional pipeline:

```
Raw Data → Cleaning → Feature Engineering → EDA → Visualization → Insights
```

**1. Data Collection**
Sourcing structured datasets from public repositories, Kaggle, or APIs. Data is assessed for completeness, relevance, and quality before analysis begins.

**2. Data Cleaning**
Handling missing values, parsing datetime fields, removing duplicates, standardising categorical columns (language, license), and filtering irrelevant records (e.g. forked repositories).

**3. Feature Engineering**
Creating derived metrics that unlock deeper analytical value — repository age, days since last update, stars-per-year growth velocity, composite popularity scores, and activity indices.

**4. Exploratory Data Analysis (EDA)**
Distribution profiling, group-level aggregations, correlation analysis, outlier identification, and language/ecosystem breakdowns — all interpreted through a business lens.

**5. Data Visualization**
Generating publication-quality static charts (Matplotlib/Seaborn) and interactive dashboards (Plotly) to communicate findings clearly and compellingly.

**6. Insights & Conclusions**
Synthesising findings into an executive summary with key takeaways, ecosystem observations, and actionable recommendations.

---

## ✨ Key Features

- **End-to-end EDA pipeline** structured across 9 analytical stages
- **Composite metric design** — custom popularity and activity scores built from first principles
- **Correlation & statistical analysis** across numerical, categorical, and datetime features
- **Outlier detection** using the IQR method to surface statistically anomalous repositories
- **Language ecosystem analysis** — which technologies dominate open source by stars, forks, and repo count
- **Interactive Plotly dashboard** with multi-panel layout exportable as a standalone HTML file
- **Professional dark-theme visualizations** styled for portfolio and presentation use
- **Modular, production-ready Python script** — fully commented and easily extensible

---

## 📈 Example Visualizations

The following chart types are generated across the projects:

| Visualization | Insight Delivered |
|---|---|
| **Distribution histograms** | Skewness and spread of stars, forks, repo sizes, and age |
| **Correlation heatmap** | Pairwise relationships between all numerical features |
| **Language bar charts** | Repo count, median stars, and forks broken down by language |
| **Top-N horizontal bars** | Most starred and most forked repositories at a glance |
| **Scatter plots with trendlines** | Forks vs Stars, Issues vs Forks, Age vs Popularity |
| **Box plots** | Star distribution variance across programming languages |
| **Activity vs Popularity** | Impact of active maintenance on community engagement |
| **Size bucket analysis** | Whether repository size influences star acquisition |

> All output charts are saved to `Project 1/Outputs/` and the interactive dashboard is available as `interactive_dashboard.html`.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/CoderNived/Data-Analytics-.git
cd Data-Analytics-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The dataset is not included in this repository due to its size (see [Dataset Information](#-dataset-information) below).
Download it separately and place it at:

```
Project 1/Dataset/repositories.csv
```

### 4. Run the EDA script

```bash
cd "Project 1/Notebook"
python github_eda_analysis.py
```

All charts will be saved automatically to `Project 1/Outputs/`.

### 5. (Optional) Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebook file inside `Project 1/Notebook/`.

---

## 📦 Dataset Information

### Project 1 — GitHub Repositories Metadata

| Attribute | Detail |
|---|---|
| **Source** | Kaggle — GitHub Popular Repositories Dataset |
| **Size** | ~445 MB (raw CSV) |
| **Rows** | 100,000+ repository records |
| **Key Columns** | Name, Stars, Forks, Issues, Language, License, Topics, Created At, Updated At |

**Why this dataset?**
The GitHub ecosystem is one of the richest publicly available proxies for open-source software trends. Analysing it reveals which technologies dominate developer mindshare, what separates thriving projects from abandoned ones, and what patterns characterise highly starred repositories.

> ⚠️ **Large files are not stored in this repository.** Due to GitHub's 100 MB file size limit, the raw dataset must be downloaded separately from Kaggle and placed in the `Dataset/` folder before running the analysis.

---

## 🔮 Future Improvements

- [ ] **Machine Learning** — Build predictive models to forecast a repository's star trajectory
- [ ] **Interactive Dashboard** — Deploy a Streamlit or Dash web app for live exploration
- [ ] **Automated Data Pipeline** — Schedule data refresh via GitHub Actions or Apache Airflow
- [ ] **NLP on Descriptions** — Topic modelling and sentiment analysis on repository descriptions
- [ ] **Time Series Analysis** — Track growth trends and seasonal patterns in open-source activity
- [ ] **Multi-project expansion** — Add EDA projects across other domains (finance, health, social)
- [ ] **requirements.txt generator** — Auto-generate environment files per project subfolder

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <b>Nived Shenoy</b><br/>
      Electronics & Telecommunication Engineering Student<br/>
      Aspiring Data Scientist & AI Engineer<br/><br/>
      <a href="https://github.com/CoderNived">
        <img src="https://img.shields.io/badge/GitHub-CoderNived-181717?style=flat&logo=github" />
      </a>
    </td>
  </tr>
</table>

*Building a data-driven future — one dataset at a time.*

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this work with attribution.

See the [LICENSE](LICENSE) file for full details.

---

<div align="center">

*If you found this useful, consider leaving a ⭐ — it helps others discover the project.*

</div>
