# 📊 Employment Trend Analysis (2015–2024)

> A production-grade data analytics project examining European employment dynamics across 35 countries, 6 age groups, and 2 genders — spanning a decade of economic transformation including the COVID-19 shock and recovery.

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-Eurostat%20LFS-blue?style=flat-square)]()

---

## 📁 Repository Structure

```
Employment-Analytics/
│
├── Dataset/
│   └── occupazione.csv                  # Raw employment dataset (35 countries, 2015–2024)
│
├── Notebook/
│   └── employment_analysis.py           # Main analysis pipeline (cleaning → EDA → visualisation)
│
├── Outputs/
│   ├── fig1_headline_trend.png          # Aggregate employment trend line chart
│   ├── fig2_age_trends.png              # Multi-cohort employment trends by age group
│   ├── fig3_country_ranking.png         # Country comparison bar chart (2024)
│   ├── fig4_gender_gap_heatmap.png      # Gender gap heatmap (country × year)
│   ├── fig5_covid_shock.png             # COVID-19 employment shock analysis
│   ├── fig6_senior_growth.png           # Senior cohort (55-64) growth by country
│   ├── fig7_boxplot_age.png             # Employment distribution by age group
│   ├── fig8_country_heatmap.png         # Full country × year employment heatmap
│   └── summary_report.txt              # Statistical summary output
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation (this file)
```

---

## 🔍 Project Overview

### What Is Employment Rate Analysis?

The **employment rate** measures the percentage of the working-age population that is currently employed. It is one of the most fundamental indicators of economic health — more granular and structurally meaningful than headline unemployment figures alone.

This project performs a **full-spectrum analytical pipeline** on a decade of European employment data, from raw data ingestion through to statistical modelling and insight extraction.

### Why Does It Matter?

| Domain | Relevance |
|--------|-----------|
| 🏛️ **Economic Policy** | Governments use employment trends to calibrate fiscal stimulus, pension reform, and active labour market programs. A 1 pp gain in employment rate can represent hundreds of thousands of jobs and billions in tax revenue. |
| 📈 **Labour Market Analysis** | Policymakers need to understand whether employment gains are broad-based or concentrated — by gender, age, and geography — to design targeted interventions. |
| 🏢 **Workforce Planning** | HR strategists and business leaders use macro employment signals to forecast talent availability, wage inflation risk, and expansion feasibility in specific markets. |
| 🎓 **Academic Research** | Employment data disaggregated by cohort and country supports research in economics, demography, sociology, and public policy. |

### Scope

| Attribute | Detail |
|-----------|--------|
| Countries | 35 European nations (EU, EEA, and candidate states) |
| Time Period | 2015 – 2024 (10 years) |
| Granularity | Country × Gender × Age Group × Year |
| Key Event Captured | COVID-19 labour market shock (2020) and recovery (2021–2024) |
| Source | Eurostat Labour Force Survey (LFS) / National Statistical Offices |

---

## 📂 Dataset Information

**File:** `Dataset/occupazione.csv`
**Format:** Wide (years as columns), pivoted to long during preprocessing
**Dimensions:** 420 rows × 13 columns (raw)

### Column Reference

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `SEX` | `str` | Gender identifier | `F` (Female), `M` (Male) |
| `AGE` | `str` | Age band of the cohort | `15-24`, `25-54`, `55-64`, `15-64`, `20-64`, `15-29` |
| `ISO` | `str` | ISO 3166-1 alpha-2 country code | `DE`, `FR`, `IT`, `TR` |
| `2015` … `2024` | `float64` | Employment rate (%) for the given year | `68.5`, `71.2`, `NaN` |

### Age Group Definitions

| Age Group | Description | Policy Relevance |
|-----------|-------------|-----------------|
| `15-24` | Youth cohort | Key youth unemployment indicator; NEET risk group |
| `15-29` | Extended youth | Captures graduates entering the labour market |
| `25-54` | Prime working age | Core labour force; most internationally comparable |
| `55-64` | Older workers / pre-retirement | Sensitive to pension reform and active ageing policy |
| `15-64` | **Headline Eurostat rate** | Standard international benchmark |
| `20-64` | EU 2020/2030 Strategy target group | EU target: 78% employment rate by 2030 |

### Missing Data Summary

```
Country  | Missing Years     | Reason
---------|-------------------|-------------------------------------------
BA       | 2015–2020         | Data collection began in 2021
ME       | 2022–2024         | Statistical reporting discontinued
All others | None            | Complete 10-year records
```

---

## 🧹 Data Cleaning

All preprocessing steps are implemented in `Notebook/employment_analysis.py`. Each decision is documented inline with rationale comments.

### Step 1 — Missing Value Detection & Handling

```python
# Audit NaN distribution per year column
print(df[year_cols].isnull().sum())

# Identify affected countries
nan_by_iso = (df.set_index(['SEX', 'AGE', 'ISO'])[year_cols]
                .isnull().any(axis=1)
                .groupby('ISO').sum())
```

**Decision:** Missing values are **not imputed**. Employment rates are highly country-specific and context-sensitive. Imputation would introduce statistical bias into country-level aggregations. Countries with missing years (BA, ME) are excluded from analyses that require full 2015–2024 records.

### Step 2 — Wide-to-Long (Tidy) Transformation

```python
df_long = df.melt(
    id_vars=['SEX', 'AGE', 'ISO'],
    value_vars=[str(y) for y in range(2015, 2025)],
    var_name='Year',
    value_name='Rate'
)
df_long['Year'] = df_long['Year'].astype(int)
df_long['Rate'] = pd.to_numeric(df_long['Rate'], errors='coerce')
```

**Why:** Tidy (long) format is required for `groupby` time-series aggregations, `seaborn` faceted plotting, and cross-dataset joins with unemployment data. The wide format is retained for heatmap visualisations.

### Step 3 — Type Standardisation

```python
# Cast all rate columns to float64; coerce non-numeric to NaN
for col in year_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

### Step 4 — Duplicate Check

```python
dupes = df_long.duplicated(subset=['SEX', 'AGE', 'ISO', 'Year']).sum()
assert dupes == 0, f"Found {dupes} duplicate rows"
# Result: 0 duplicates confirmed
```

### Step 5 — Column Naming Standardisation

All column names are standardised to uppercase (`SEX`, `AGE`, `ISO`, `YEAR`, `RATE`) for consistent programmatic access throughout the pipeline.

### Cleaning Summary

| Step | Input | Output | Records Affected |
|------|-------|--------|-----------------|
| Wide → Long melt | 420 rows × 13 cols | 4,200 rows × 5 cols | All |
| Type casting | Mixed types | `int64` / `float64` | Year + Rate columns |
| NaN handling | 120 missing cells | Excluded from agg. | BA (12 rows), ME (12 rows) |
| Deduplication | 4,200 rows | 4,200 rows (no change) | 0 removed |

---

## 🔬 Exploratory Data Analysis

### 1. Employment Trends 2015–2024

The aggregate European employment rate (15-64) grew from **64.5%** (2015) to **71.3%** (2024) — a gain of **+6.8 percentage points** over ten years. The only interruption was COVID-19, which caused a −1.2 pp contraction in 2020. Recovery was complete by 2022 and the decade closed at an all-time high.

```python
# Year-over-year trend for headline metric (15-64)
yoy = df_long[df_long['AGE'] == '15-64'].groupby('YEAR')['RATE'].mean()
yoy_change = yoy.diff()
```

| Year | Mean Rate (%) | YoY Change (pp) |
|------|--------------|-----------------|
| 2015 | 64.5 | — |
| 2016 | 65.6 | +1.1 |
| 2017 | 66.9 | +1.3 |
| 2018 | 68.1 | +1.2 |
| 2019 | 68.9 | +0.8 |
| **2020** | **67.7** | **−1.2 ⚠️** |
| 2021 | 68.8 | +1.1 |
| 2022 | 70.5 | +1.7 |
| 2023 | 70.9 | +0.4 |
| 2024 | 71.3 | +0.5 |

### 2. Gender Employment Comparison

Male employment consistently exceeds female across all years and countries. The mean M−F gap was **10.2 pp in 2015** and has narrowed to **9.1 pp by 2024**.

```python
gender_trend = (df_long[df_long['AGE'] == '15-64']
    .groupby(['YEAR', 'SEX'])['RATE'].mean()
    .unstack()
    .assign(gap=lambda x: x['M'] - x['F']))
```

Country-level extremes in 2024:

| Highest Gap | Country | Gap (pp) | Lowest Gap | Country | Gap (pp) |
|-------------|---------|----------|------------|---------|----------|
| 🔴 Worst | Turkey (TR) | 36.3 | 🟢 Best | Finland (FI) | 0.4 |
| | Bosnia (BA) | 25.5 | | Estonia (EE) | 1.0 |
| | Italy (IT) | 17.8 | | Lithuania (LT) | 1.2 |

### 3. Age-Group Employment Distribution

Employment varies enormously by cohort. Prime working-age adults (25-54) consistently show rates above 80%, while youth (15-24) average just 34.7%.

```python
age_stats = df_long.groupby('AGE')['RATE'].describe()
```

| Age Group | Mean (%) | Std Dev | Min (%) | Max (%) |
|-----------|----------|---------|---------|---------|
| 15-24 | 34.7 | 15.7 | 10.4 | 78.2 |
| 15-29 | 49.4 | 13.0 | 23.0 | 82.5 |
| 25-54 | 80.2 | 9.7 | 35.7 | 94.8 |
| 55-64 | 58.2 | 14.8 | 16.7 | 89.7 |
| 15-64 | 68.3 | 10.3 | 29.7 | 89.4 |
| 20-64 | 72.8 | 10.4 | 32.0 | 91.1 |

**Notable trend:** The 55-64 cohort grew by **+13.3 pp** over the decade — the single largest structural gain — driven by pension age reforms across Europe.

### 4. Country-Level Patterns

```python
country_2024 = (df_long[(df_long['AGE'] == '15-64') & (df_long['YEAR'] == 2024)]
    .groupby('ISO')['RATE'].mean()
    .sort_values(ascending=False))
```

**Top 5 — 2024:** Iceland (85.2%), Netherlands (82.3%), Switzerland (80.4%), Malta (78.5%), Germany (77.4%)

**Bottom 5 — 2024:** Italy (62.2%), North Macedonia (57.8%), Turkey (55.1%), Bosnia (53.8%), Montenegro (n/a)

**Fastest growing 2015→2024:** Serbia (+15.3 pp), Malta (+13.8 pp), Cyprus (+12.9 pp), Greece (+12.5 pp), Croatia (+12.3 pp)

---

## 📐 Statistical Analysis

```python
import numpy as np

# 1. Mean employment rate per country (all years, 15-64)
country_means = df_long[df_long['AGE']=='15-64'].groupby('ISO')['RATE'].mean()

# 2. Compound annual growth rate (CAGR) approximation
pivot = df_long[df_long['AGE']=='15-64'].groupby(['ISO','YEAR'])['RATE'].mean().unstack()
pivot['CAGR'] = ((pivot[2024] / pivot[2015]) ** (1/9) - 1) * 100

# 3. Gender variance comparison
gender_var = df_long[df_long['AGE']=='15-64'].groupby('SEX')['RATE'].var()

# 4. COVID shock magnitude
covid_shock = (pivot[2020] - pivot[2019]).sort_values()

# 5. Decade growth
total_growth = (pivot[2024] - pivot[2015]).sort_values(ascending=False)
```

### Key Statistical Outputs

| Metric | Female | Male |
|--------|--------|------|
| Mean Employment Rate (15-64, all years) | 63.4% | 73.3% |
| Standard Deviation | 11.0 pp | 6.5 pp |
| Min (2024 cross-section) | 36.9% (TR) | 56.0% (multiple) |
| Max (2024 cross-section) | 82.3% (IS) | 89.4% (IS) |

> **Interpretation:** Female employment shows **higher variance** (σ = 11.0 vs 6.5), reflecting greater cross-country heterogeneity driven by cultural, structural, and policy differences. Male employment is more uniform across Europe.

### COVID-19 Shock — Top 5 Most Affected Countries

```
Montenegro (ME):  −5.75 pp
Iceland (IS):     −3.85 pp
Ireland (IE):     −2.90 pp
Turkey (TR):      −2.80 pp
Greece (GR):      −2.45 pp
```

### Countries Resilient to COVID (2019→2020)

```
Serbia (RS):  +0.70 pp    (employment gained)
Malta (MT):   +0.50 pp
Poland (PL):  +0.20 pp
Croatia (HR): +0.20 pp
```

---

## 📊 Visualisations

All charts are saved to `Outputs/` at 150 DPI.

### Figure 1 — Headline Employment Trend + Gender Split
**File:** `fig1_headline_trend.png`
**Type:** Dual-panel line chart
**Reveals:** Long-run employment growth trajectory, COVID-19 dip, and persistent M−F gap. The left panel shows the aggregate; the right panel disaggregates by gender, making the narrowing gap visually clear.

---

### Figure 2 — Employment Trends by Age Group
**File:** `fig2_age_trends.png`
**Type:** Multi-series line chart (6 cohorts)
**Reveals:** Differential recovery rates across cohorts. The 55-64 senior boom is the steepest positive trajectory; youth (15-24) is the most COVID-volatile. Prime-age (25-54) is the most stable.

---

### Figure 3 — Country Ranking Bar Chart
**File:** `fig3_country_ranking.png`
**Type:** Horizontal bar chart (colour-coded)
**Reveals:** Full ranking of 33 countries by 2024 employment rate. Colour coding (green ≥ 72%, orange 62–72%, red < 62%) allows instant identification of high, medium, and low performers.

---

### Figure 4 — Gender Gap Heatmap
**File:** `fig4_gender_gap_heatmap.png`
**Type:** Annotated heatmap (country × year)
**Reveals:** Which countries have large, persistent gender employment gaps vs. those converging toward parity. Turkey and the Balkans show persistent deep red; Nordics and Baltics show near-white (parity).

---

### Figure 5 — COVID-19 Employment Shock
**File:** `fig5_covid_shock.png`
**Type:** Bar chart (positive/negative)
**Reveals:** Asymmetric pandemic impact — tourism-dependent economies (IS, IE, GR, ES) suffered the largest drops; manufacturing and informal-economy countries (PL, HR, RS) were largely immune.

---

### Figure 6 — Senior Employment Growth
**File:** `fig6_senior_growth.png`
**Type:** Bar chart with mean reference line
**Reveals:** The 55-64 cohort's decade-long structural transformation. Every single country in the dataset shows positive growth — a universal policy effect from pension reform.

---

### Figure 7 — Age Group Distribution (Box Plot)
**File:** `fig7_boxplot_age.png`
**Type:** Box-and-whisker plot
**Reveals:** Distribution shape, median, and outliers across all six age cohorts. Wide IQR for 15-24 and 55-64 signals high cross-country heterogeneity in these groups.

---

### Figure 8 — Full Country × Year Heatmap
**File:** `fig8_country_heatmap.png`
**Type:** Annotated heatmap (33 countries × 10 years)
**Reveals:** The complete employment landscape at a glance. Each cell shows the exact rate. The north-to-south gradient and the 2020 cooling band are immediately visible.

---

## 💡 Key Insights

> The following insights are derived from the statistical and visual analysis. Each is framed as a stakeholder-ready finding.

**1. Europe's Labour Market Reached a Decade High in 2024**
The mean European employment rate (15-64) hit **71.3%** in 2024 — the highest value in the entire dataset — demonstrating that the post-GFC recovery continued beyond the COVID disruption.

**2. COVID-19 Was a Shock, Not a Structural Break**
Despite causing a −1.2 pp employment drop in 2020 (the only annual decline in the dataset), the labour market returned to its pre-pandemic trajectory within two years. Furlough schemes and fiscal stabilisers prevented the scarring effect seen after 2008.

**3. Senior Employment Is the Decade's Biggest Structural Story**
Workers aged 55-64 saw average employment grow from 51.1% to 64.4% — a **+13.3 pp surge** that dwarfs any other cohort gain. This is the most direct evidence in the dataset that **pension reform works**.

**4. Turkey's Gender Gap Is a Category of Its Own**
At **36.3 pp** (male minus female employment in 2024), Turkey's gender gap is more than double that of the next-highest country (Bosnia, 25.5 pp) and more than four times the European average (9.1 pp). This represents an enormous, structural economic inefficiency.

**5. Nordic-Baltic States Are the Global Benchmark for Gender Parity**
Finland (0.4 pp), Estonia (1.0 pp), Lithuania (1.2 pp), Latvia (2.3 pp) and Sweden (2.9 pp) have essentially closed the gender employment gap. Their model — universal childcare, equal pay legislation, and flexible work culture — is directly transferable.

**6. Eastern European Convergence Is Accelerating**
Serbia (+15.3 pp), Malta (+13.8 pp), Cyprus (+12.9 pp) and Greece (+12.5 pp) are the fastest-improving economies in the dataset. EU labour market integration and structural funds are producing measurable results, validating Single Market membership.

**7. Youth Employment Remains Structurally Fragile**
Despite a decade of EU Youth Guarantee schemes, youth (15-24) employment averages only 34.7% — and was disproportionately hit by COVID. Countries like Greece (16.8% for females), Italy (15.1%), and Romania (14.7%) show that youth employment challenges are structural, not cyclical.

**8. High Achievers Are Near the Ceiling**
Iceland (85.2%), Netherlands (82.3%), and Switzerland (80.4%) show minimal room for further improvement — their growth 2015→2024 was only +0.5 to +1.2 pp. Policy energy must focus on the convergence countries, where each invested euro has the highest marginal impact on aggregate European employment.

---

## 🛠️ Tech Stack

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.0 | Data ingestion, reshaping, aggregation, groupby operations |
| `numpy` | ≥ 1.24 | Numerical computation, growth rate calculations, correlation |
| `matplotlib` | ≥ 3.7 | Base plotting engine; figure/axis management |
| `seaborn` | ≥ 0.12 | Statistical visualisation; heatmaps, box plots, themes |
| `scikit-learn` *(optional)* | ≥ 1.3 | Linear trend fitting, outlier detection, preprocessing pipelines |

### Environment

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| OS | Linux / macOS / Windows |
| IDE | VS Code / JupyterLab / PyCharm |

---

## ⚡ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Employment-Analytics.git
cd Employment-Analytics
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Analysis Pipeline

```bash
python Notebook/employment_analysis.py
```

All outputs (charts + summary report) will be written to `Outputs/`.

### 5. Inspect Results

```
Outputs/
├── fig1_headline_trend.png       ← Overall employment trend
├── fig2_age_trends.png           ← Cohort breakdown
├── fig3_country_ranking.png      ← Country comparison
├── fig4_gender_gap_heatmap.png   ← Gender gap matrix
├── fig5_covid_shock.png          ← Pandemic impact
├── fig6_senior_growth.png        ← Senior cohort analysis
├── fig7_boxplot_age.png          ← Distribution analysis
└── fig8_country_heatmap.png      ← Full data heatmap
```

### `requirements.txt`

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## 🚀 Future Improvements

### Short-Term (v1.1)

- [ ] **Unemployment Integration** — Merge with `disoccupazione.csv` to enable joint employment/unemployment analysis and compute true labour market slack metrics
- [ ] **Interactive Dashboard** — Rebuild visualisations in `Plotly` or `Dash` for browser-based interactivity and country-level drill-down
- [ ] **Automated Statistical Report** — Export a formatted PDF report using `reportlab` or `WeasyPrint`, triggered as part of the analysis pipeline

### Medium-Term (v1.2)

- [ ] **Time-Series Forecasting** — Apply ARIMA, Prophet, or LSTM models to project 2025–2030 employment rates under baseline, optimistic, and recessionary scenarios
- [ ] **GDP & Macro Correlation** — Enrich the dataset with Eurostat GDP, productivity, and wage data to quantify the employment-growth elasticity per country
- [ ] **Feature Engineering** — Derive composite indicators: employment-to-population ratios, gender parity indices, decade momentum scores, and COVID recovery speed metrics

### Long-Term (v2.0)

- [ ] **ML Clustering** — Apply `k-means` or hierarchical clustering to group countries into labour market archetypes (Nordic model, Southern European model, transition economies, etc.)
- [ ] **Causal Inference** — Use difference-in-differences methodology to evaluate the impact of specific policy events (pension reforms, youth guarantee launches, minimum wage changes)
- [ ] **Real-Time Data Pipeline** — Connect to the Eurostat REST API for automated monthly data refreshes and drift alerting when employment rates deviate from expected trend bands
- [ ] **Multi-Language Support** — Internationalise the report outputs for EU institutional use

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Data Source:** [Eurostat Labour Force Survey (LFS)](https://ec.europa.eu/eurostat/web/lfs) and national statistical offices
- **Geographic Coverage:** EU-27 member states + Iceland, Norway, Switzerland, Turkey, Bosnia & Herzegovina, Montenegro, North Macedonia, Serbia
- **Methodology Reference:** [Eurostat Employment Statistics Methodology](https://ec.europa.eu/eurostat/statistics-explained/index.php/Employment_statistics)

---

<div align="center">

**Built with 🐍 Python · 🐼 pandas · 📊 seaborn · 📐 numpy**

*Senior Data Engineer & Data Science Portfolio Project*

</div>
