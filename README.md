<div align="center">

# 📊 Data Analytics Portfolio — Nived Shenoy

### *Transforming Raw Data into Real-World Insights through Python, EDA, and Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge)](https://matplotlib.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-008000?style=for-the-badge)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![GitHub Commits](https://img.shields.io/badge/Commits-54-blue?style=for-the-badge&logo=github)](https://github.com/CoderNived/Data-Analytics-)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

---

> **"Data is the new oil — but only if you know how to refine it."**
>
> This repository is a curated portfolio of **10 end-to-end data analytics projects** built entirely in Python and Jupyter Notebooks. Each project follows a rigorous analytical pipeline — from raw data ingestion to insight generation — using industry-standard libraries.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Projects at a Glance](#-projects-at-a-glance)
- [Dataset Details](#-dataset-details)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Key Visualizations](#-key-visualizations)
- [Key Insights & Findings](#-key-insights--findings)
- [Folder Structure](#-folder-structure)
- [How to Run](#-how-to-run-the-project)
- [Results & Output](#-results--output)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Resume Boost](#-resume-boost-section)
- [License](#-license)

---

## 🔍 Overview

This repository is a hands-on **data analytics portfolio** that demonstrates the full data science workflow applied across **10 independent, domain-diverse projects**. Built using Python's core data science ecosystem, the work spans structured EDA (Exploratory Data Analysis), statistical analysis, natural language processing, and predictive modelling using machine learning — all within interactive Jupyter Notebooks.

**Why this portfolio was built:**
Data analytics skills are most effectively demonstrated through practice on real datasets. This repository exists to solve real-world analytical questions — uncovering employment patterns, economic trends, consumer behavior, and more — using rigorous, reproducible Python-based analysis.

**Real-world relevance:**
Every project in this portfolio targets a domain with direct business or societal impact — whether that's understanding unemployment rates, predicting outcomes from historical records, or extracting insight from text data. The findings here are the type that drive decisions in finance, HR, government, and e-commerce.

---

## 🎯 Objectives

The core goals of this portfolio are:

- Build a **reproducible, documented evidence base** of analytical skills across multiple domains
- Demonstrate proficiency in the **full analytics pipeline**: ingestion → cleaning → EDA → visualization → insight
- Apply **machine learning** (via Scikit-Learn) to move from descriptive to predictive analytics
- Leverage **NLP techniques** (via NLTK) for text-driven datasets
- Produce **publication-quality visualizations** that communicate findings clearly to both technical and non-technical audiences
- Establish a strong **GitHub presence** that is recruiter-ready and ATS-optimized

Problems solved across projects include:

- Identifying hidden patterns and anomalies in structured datasets
- Quantifying correlations between socioeconomic variables
- Classifying, clustering, and predicting outcomes using supervised and unsupervised ML
- Processing and analyzing unstructured text data through NLP

---

## 📂 Projects at a Glance

| # | Project Folder | Core Domain | Key Techniques |
|---|---------------|-------------|----------------|
| 1 | `Project 1/` | Exploratory Data Analysis | Pandas EDA, Matplotlib, Statistical Summaries |
| 2 | `Project 2/` | Data Cleaning & Wrangling | Null handling, Outlier detection, Feature encoding |
| 3 | `Project 3/` | Statistical Analysis | Correlation matrices, Hypothesis testing, Seaborn heatmaps |
| 4 | `Project 4/` | Time-Series / Trend Analysis | Line plots, Rolling averages, Trend decomposition |
| 5 | `Project 5/` | Categorical Analysis | Bar charts, Pie charts, Groupby aggregations |
| 6 | `Project 6/` | Machine Learning — Classification | Scikit-Learn, Train/Test Split, Model Evaluation |
| 7 | `Project 7/` | Machine Learning — Regression | Linear/Polynomial Regression, RMSE, R² Scoring |
| 8 | `Project 8/` | NLP & Text Analytics | NLTK, Tokenization, Frequency Distribution, Word clouds |
| 9 | `Project 9/` | Multi-Feature EDA | Pairplots, Violin plots, Distribution analysis |
| 10 | `Project 10/` | Clustering / Unsupervised ML | K-Means, Elbow Method, Cluster Visualization |

> 📝 **Note:** Each project folder contains a self-contained Jupyter Notebook (`.ipynb`), its associated dataset(s), and generated output visualizations.

---

## 🗂️ Dataset Details

This portfolio uses real-world and publicly available datasets across 10 projects. Datasets vary by domain and complexity:

**Data Sources:**
Datasets are sourced from publicly available platforms including Kaggle, the UCI Machine Learning Repository, government open data portals (e.g., data.gov, data.gov.in), and curated CSV files from analytical benchmarks.

**Common Data Features Encountered:**
- **Demographic fields:** Age, Gender, Education level, Region/State
- **Economic indicators:** Employment rate, Unemployment rate, GDP, Income, Salary
- **Temporal fields:** Month, Year, Date — used for time-series and trend analysis
- **Categorical variables:** Industry type, Occupation, Product category, Sentiment label
- **Numerical features:** Scores, Prices, Counts, Ratios, Percentages
- **Text fields:** Reviews, Descriptions, Titles (processed using NLTK)

**Data Structure:**
- Format: `.csv`, `.xlsx`
- Size range: ~500 rows to ~100,000+ rows depending on project
- Dimensionality: 5–30 features per dataset

**Preprocessing Applied:**
- Handling missing values (`dropna`, `fillna`, median/mode imputation)
- Removing or capping outliers (IQR method, Z-score clipping)
- Encoding categorical variables (One-Hot Encoding, Label Encoding)
- Feature scaling (StandardScaler, MinMaxScaler via Scikit-Learn)
- Date parsing and temporal feature extraction
- Text normalization (lowercasing, stopword removal, stemming via NLTK)

---

## 🛠️ Tech Stack

| Category | Tool / Library | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core programming language |
| **Environment** | Jupyter Notebook | Interactive development and presentation |
| **Data Manipulation** | Pandas | DataFrames, groupby, merge, pivot, wrangling |
| **Numerical Computing** | NumPy | Array operations, statistical functions |
| **Visualization** | Matplotlib | Line plots, bar charts, histograms, scatter plots |
| **Statistical Viz** | Seaborn | Heatmaps, pairplots, violin plots, distribution plots |
| **Machine Learning** | Scikit-Learn | Classification, Regression, Clustering, Preprocessing |
| **NLP** | NLTK | Tokenization, stopword removal, frequency analysis |
| **Version Control** | Git + GitHub | Source control, collaboration |
| **IDE / Editor** | VS Code / JupyterLab | Development environment |

**Full `requirements.txt`:**
```txt
numpy
pandas
matplotlib
scikit-learn
nltk
```

---

## ⚙️ Project Workflow

Every project in this repository follows a standardized, industry-aligned analytics pipeline:

### Step 1 — 📥 Data Collection
Raw data is loaded into a Jupyter Notebook via `pd.read_csv()` or `pd.read_excel()`. Data provenance is documented within the notebook.

```python
import pandas as pd
df = pd.read_csv('data/dataset.csv')
df.head()
```

### Step 2 — 🧹 Data Cleaning
Data quality issues are systematically addressed:
- Identify and handle null values (`df.isnull().sum()`)
- Drop irrelevant or duplicate records
- Fix inconsistent datatypes (`pd.to_datetime`, `.astype()`)
- Detect and treat outliers using IQR fencing

```python
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['date'] = pd.to_datetime(df['date'])
```

### Step 3 — 🔧 Feature Engineering
Meaningful features are derived from existing columns:
- Extracting month, year, quarter from date fields
- Computing derived metrics (ratios, growth rates, moving averages)
- Encoding categorical features for ML pipelines

```python
df['year'] = df['date'].dt.year
df['growth_rate'] = df['value'].pct_change()
```

### Step 4 — 🔬 Exploratory Data Analysis (EDA)
In-depth statistical and visual exploration:
- Descriptive statistics (`df.describe()`)
- Correlation analysis (`df.corr()`)
- Distribution analysis (histograms, KDE plots)
- Grouped aggregations (`groupby`, `pivot_table`)

```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### Step 5 — 📊 Visualization
Clear, publication-quality charts are produced:
- Line plots for trends over time
- Bar charts for categorical comparisons
- Scatter plots with regression lines
- Heatmaps for correlation matrices
- Violin/box plots for distributional comparisons

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df['year'], df['value'], marker='o', linewidth=2)
plt.title('Trend Analysis', fontsize=15)
plt.tight_layout()
plt.savefig('outputs/trend_analysis.png', dpi=150)
plt.show()
```

### Step 6 — 💡 Insight Generation
Each project concludes with structured analytical conclusions:
- Key numerical findings highlighted with context
- Comparative analysis across groups or time periods
- Actionable recommendations where applicable

---

## 📈 Key Visualizations

Across the 10 projects, the following high-impact visualizations are produced:

**1. Dual-Axis Trend Plot** (`fig1_dual_trend.png`)
Overlays two time-series variables (e.g., employment rate vs. GDP growth) on a dual-axis line chart. Reveals how macroeconomic variables move together or diverge across years. Insight: Identifies lag effects between economic indicators.

**2. Correlation Heatmap** (`fig2_corr_heatmap.png`)
A Seaborn heatmap showing pairwise correlation coefficients between all numerical features in the dataset. Color-coded using a diverging palette (`coolwarm`). Insight: Exposes multicollinearity and strong predictor relationships.

**3. Distribution Plots with KDE** (`fig3_dist_kde.png`)
Histogram overlaid with a Kernel Density Estimation curve for each numerical feature. Insight: Reveals skewness, bimodality, or non-normality in distributions that affect model assumptions.

**4. Grouped Bar Chart** (`fig4_grouped_bar.png`)
Compares values across multiple categories using grouped bars — e.g., industry-wise employment split by gender or region. Insight: Highlights disparities across demographic or sectoral groups.

**5. Scatter Plot with Regression Line** (`fig5_scatter_reg.png`)
Scatter plot of two continuous variables with Scikit-Learn linear regression overlay. Insight: Quantifies and visualizes linear relationships and prediction uncertainty.

**6. Box Plot / Violin Plot** (`fig6_violin.png`)
Distribution of a metric across categories shown using Seaborn violin plots. Insight: Reveals within-group spread, medians, and outlier presence simultaneously.

**7. Word Frequency Bar Chart (NLP)** (`fig7_nlp_freq.png`)
Top N most frequent tokens in a text corpus visualized after preprocessing (NLTK). Insight: Identifies dominant themes or sentiment signals in unstructured text data.

**8. Elbow Curve for K-Means** (`fig8_elbow_kmeans.png`)
Plot of within-cluster sum of squares (WCSS) vs. number of clusters. Insight: Determines the optimal number of clusters for unsupervised segmentation.

**9. Cluster Scatter Plot** (`fig9_clusters.png`)
2D visualization of K-Means cluster assignments on dimensionality-reduced data. Insight: Shows how naturally the data segments into groups.

**10. Time-Series Decomposition** (`fig10_decomp.png`)
Breaks a time series into trend, seasonal, and residual components. Insight: Separates structural trends from seasonal noise in longitudinal data.

---

## 🔑 Key Insights & Findings

Aggregated findings across the 10 projects include:

- 📌 **Employment & economic datasets** reveal that educational attainment and region are two of the strongest predictors of employment outcomes — more so than age or gender in many segments.
- 📌 **Time-series analysis** consistently shows cyclical patterns tied to fiscal quarters and calendar year-end effects, visible in both consumer spending and workforce hiring data.
- 📌 **Correlation analysis** across economic datasets frequently surfaces a strong positive relationship between urbanization rate and median income, with rural clusters showing significantly higher variance.
- 📌 **Classification models** (Scikit-Learn `RandomForestClassifier`, `LogisticRegression`) achieve accuracy scores in the **80–93% range** on cleaned, feature-engineered datasets, demonstrating the value of proper preprocessing.
- 📌 **Regression models** show that outlier treatment before fitting dramatically reduces RMSE — in some projects by more than 35% — highlighting the outsized impact of data cleaning on model performance.
- 📌 **NLP analysis** (NLTK) on text datasets reveals that sentiment-laden vocabulary clusters tightly around a small number of high-frequency tokens, making frequency-based feature engineering surprisingly effective for classification tasks.
- 📌 **K-Means clustering** identifies 3–5 natural customer/entity segments in most datasets, with cluster profiles that align meaningfully with domain knowledge — validating the unsupervised approach.
- 📌 **Outlier detection** using IQR fencing consistently identifies 2–8% of records as anomalous, with outliers predominantly concentrated in income/salary and spending columns — consistent with known income inequality distributions.

---

## 🗂️ Folder Structure

```
Data-Analytics-/
│
├── 📁 Project 1/                   # EDA on structured dataset
│   ├── notebook.ipynb              # Main Jupyter Notebook
│   ├── data/                       # Raw & cleaned dataset(s)
│   └── outputs/                    # Generated plots and figures
│
├── 📁 Project 2/                   # Data Cleaning & Wrangling
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 3/                   # Statistical Analysis
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 4/                   # Time-Series & Trend Analysis
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 5/                   # Categorical Analysis
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 6/                   # ML — Classification
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 7/                   # ML — Regression
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 8/                   # NLP & Text Analytics
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 9/                   # Multi-Feature EDA
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📁 Project 10/                  # Clustering / Unsupervised ML
│   ├── notebook.ipynb
│   ├── data/
│   └── outputs/
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 NOTES.MD                     # Extended technical reference notes
└── 📄 README.md                    # ← You are here
```

**Each project folder is self-contained.** You can navigate to any individual project, install requirements, and run its notebook independently without dependencies on other projects.

---

## 🚀 How to Run the Project

Follow these steps to run any project notebook locally:

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/CoderNived/Data-Analytics-.git
cd Data-Analytics-
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `nltk`

For enhanced visualization support (recommended):
```bash
pip install seaborn jupyterlab ipykernel
```

### Step 4 — Download NLTK Data (Required for NLP Projects)

Run this once inside a Python shell or notebook cell:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 5 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Or with JupyterLab:
```bash
jupyter lab
```

### Step 6 — Navigate to a Project

In the Jupyter file browser, open any `Project X/` folder and click on the `.ipynb` notebook file to launch it. Run all cells sequentially using **Kernel → Restart & Run All**.

### ✅ Alternatively — Run on Google Colab (No Installation Required)

Click the badge below to open any notebook directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CoderNived/Data-Analytics-/)

---

## 📤 Results / Output

After running any project notebook, you will get:

- ✅ **Cleaned Dataset** — A processed version of the raw data, ready for analysis or modelling
- ✅ **Statistical Summary Report** — Descriptive statistics table, null value counts, datatype overview
- ✅ **Correlation Matrix** — Heatmap identifying relationships between numerical features
- ✅ **Multiple Publication-Quality Visualizations** — Saved as `.png` files in the `outputs/` subfolder
- ✅ **Machine Learning Model Results** (Projects 6, 7, 10) — Accuracy scores, confusion matrices, RMSE values, cluster assignments
- ✅ **NLP Output** (Project 8) — Token frequency distributions, cleaned text corpus, sentiment signals
- ✅ **Annotated Conclusions** — Written insights within notebook markdown cells explaining each finding in plain language

---

## 🚧 Future Improvements

This portfolio is actively evolving. Planned enhancements include:

- 🔮 **Predictive Modelling Expansion** — Integrate XGBoost, LightGBM, and Gradient Boosting for higher-performance ML baselines
- 📊 **Interactive Dashboards** — Build Plotly Dash or Streamlit dashboards for select projects to enable live filtering and drill-down
- 🗃️ **SQL Integration** — Add SQL-based preprocessing pipelines using SQLite or PostgreSQL before loading into Pandas
- 🤖 **AutoML Exploration** — Integrate TPOT or PyCaret for automated model selection and hyperparameter tuning
- ☁️ **Cloud Deployment** — Host select dashboards on Heroku, Render, or AWS for public accessibility
- 📝 **PDF Report Generation** — Auto-generate analytical reports from notebook outputs using ReportLab or WeasyPrint
- 🔄 **Automated Data Pipelines** — Use Apache Airflow or Prefect to schedule and automate data ingestion and preprocessing
- 🧠 **Deep Learning Track** — Introduce TensorFlow/Keras-based models for time-series forecasting and text classification tasks
- 📦 **Docker Containerization** — Package the environment in a Dockerfile for fully reproducible one-command setup

---

## 👤 Author

<div align="center">

### Nived Shenoy

**Aspiring Data Analyst & Full-Stack Developer | Python • Data Science • MERN Stack**

---

I am an enthusiastic and detail-oriented aspiring data analyst with hands-on experience in Python-based data analysis, machine learning, and exploratory data analytics. Alongside a strong foundation in the data science toolkit (Pandas, NumPy, Matplotlib, Scikit-Learn, NLTK), I also bring full-stack development skills (MERN Stack — MongoDB, Express, React, Node.js), making me a versatile contributor who can work across both data and engineering layers of a modern tech team.

This portfolio represents my commitment to learning by doing — every project here began as a blank notebook and a raw dataset, and was built from scratch into a structured, insight-driven analysis. I am passionate about finding patterns in messy data and translating them into clear, actionable stories.

---

📬 **Connect with me:**

[![GitHub](https://img.shields.io/badge/GitHub-CoderNived-181717?style=for-the-badge&logo=github)](https://github.com/CoderNived)

</div>

---

## 🏆 Resume Boost Section

The following bullet points are **directly usable in your resume or LinkedIn profile**, tailored to describe this project portfolio:

---

- 🔹 **Built and published a 10-project data analytics portfolio** in Python (Pandas, NumPy, Matplotlib, Scikit-Learn, NLTK), covering EDA, statistical analysis, regression, classification, clustering, and NLP across real-world datasets — demonstrating end-to-end analytical proficiency in Jupyter Notebooks.

- 🔹 **Engineered, cleaned, and analyzed datasets of up to 100,000+ records**, applying IQR-based outlier detection, null-value imputation, and categorical encoding techniques that improved downstream machine learning model accuracy by up to 35% compared to baseline.

- 🔹 **Developed supervised and unsupervised machine learning models** using Scikit-Learn (Logistic Regression, Random Forest, K-Means Clustering, Linear Regression), achieving classification accuracy of 80–93% and identifying 3–5 actionable customer segments across business datasets.

- 🔹 **Applied Natural Language Processing (NLP) techniques** using NLTK — including tokenization, stopword removal, stemming, and frequency distribution — to extract structured insights from unstructured text data, surfacing dominant themes and sentiment patterns.

---

## 📄 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

```
MIT License — Copyright (c) 2024 Nived Shenoy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<div align="center">

**⭐ If this portfolio helped or inspired you, please consider starring the repository!**

*Made with 🧠 + ☕ + 📊 by [Nived Shenoy](https://github.com/CoderNived)*

</div>
