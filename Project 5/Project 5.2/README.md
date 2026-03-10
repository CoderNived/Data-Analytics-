# 📉 Global Unemployment Trend Analysis (2015–2024)

> A professional-grade data analytics project dissecting European unemployment dynamics across 35 countries, 35 age cohorts, and 2 genders over a decade — capturing structural shifts, the COVID-19 shock, youth labour market fragility, and the road to recovery.

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scipy](https://img.shields.io/badge/scipy-1.11%2B-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Source-Eurostat%20LFS-0ea5e9?style=flat-square)]()

---

## 📁 Repository Structure

```
Unemployment-Analytics/
│
├── Dataset/
│   └── disoccupazione.csv               # Raw unemployment dataset (35 countries, 2015–2024)
│
├── Notebook/
│   └── unemployment_analysis.py         # Full analysis pipeline (preprocessing → EDA → insights)
│
├── Outputs/
│   ├── fig1_dual_trend.png              # Employment & unemployment dual-axis trend
│   ├── fig2_gender_split.png            # Gender-disaggregated unemployment trends
│   ├── fig3_scatter_correlation.png     # Employment vs unemployment scatter + OLS
│   ├── fig4_youth_unemployment.png      # Youth (15-24) unemployment country ranking
│   ├── fig5_covid_double_shock.png      # COVID-19 shock: employment + unemployment
│   ├── fig6_unemployment_heatmap.png    # Country × year unemployment heatmap
│   ├── fig7_unemployment_by_age.png     # Unemployment trends by age cohort
│   ├── fig8_youth_change.png            # Youth unemployment change 2015→2024
│   ├── fig9_gender_gap_country.png      # Unemployment gender gap per country
│   └── analytics_report.txt            # Statistical summary output
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation (this file)
```

---

## 🔍 Project Overview

### What Is Unemployment Rate Analysis?

The **unemployment rate** measures the percentage of the active labour force (those employed plus those actively seeking work) who are without a job but available and looking for one. Unlike the employment rate, it focuses specifically on labour market *failure* — the gap between labour supply and demand.

This project delivers a **complete analytical pipeline** on a decade of disaggregated European unemployment data, from raw data ingestion through statistical testing to policy-grade insight extraction.

### Why Does It Matter?

| Domain | Why Unemployment Analysis Is Critical |
|--------|---------------------------------------|
| 📡 **Economic Health Monitoring** | Unemployment is a lagging economic indicator that confirms whether GDP growth is translating into real job creation. Rising unemployment signals recession; sustained decline marks genuine recovery. |
| 🏛️ **Government Policy Decisions** | Policymakers use unemployment data to calibrate benefit spending, justify stimulus packages, design active labour market programs (ALMPs), and set pension reform timelines. |
| 🎓 **Labour Market Research** | Economists study unemployment disaggregated by age, gender, and geography to identify structural barriers, skill mismatches, and the effectiveness of vocational training investments. |
| 🏢 **Corporate Workforce Strategy** | Talent acquisition teams and HR strategists use national unemployment rates as a proxy for talent availability, hiring competition, and wage inflation risk in target markets. |
| 📊 **Social Policy & Welfare** | Sustained youth unemployment correlates with long-term earnings scarring (the "scarring effect"), increased NEET rates, and elevated social welfare costs — making early intervention critical. |

### Scope

| Attribute | Detail |
|-----------|--------|
| Countries | 35 European nations (EU, EEA, candidate states) |
| Time Period | 2015 – 2024 (10 years) |
| Age Cohorts | 35 distinct bands from `15-19` through `70-74` |
| Granularity | Country × Gender × Age Group × Year |
| Key Event | COVID-19 labour market shock (2020) and full recovery analysis |
| Source | Eurostat Labour Force Survey (LFS) / National Statistical Offices |

---

## 📂 Dataset Information

**File:** `Dataset/disoccupazione.csv`
**Format:** Wide (years as columns), reshaped to long format during preprocessing
**Dimensions:** 2,450 rows × 13 columns (raw wide format)

### Column Reference

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `SEX` | `str` | Gender identifier | `F` (Female), `M` (Male) |
| `AGE` | `str` | Age band of the cohort | `15-24`, `25-54`, `55-64`, `15-74`, `70-74`… (35 groups) |
| `ISO` | `str` | ISO 3166-1 alpha-2 country code | `DE`, `ES`, `GR`, `TR` |
| `2015` … `2024` | `float64` | Unemployment rate (%) for that year | `12.3`, `6.8`, `NaN` |

### Age Cohort Reference

The unemployment dataset contains **35 age bands** — far more granular than the companion employment dataset. This enables detailed lifecycle unemployment analysis.

| Category | Age Bands Included |
|----------|--------------------|
| **Youth** | `15-19`, `15-24`, `20-24`, `20-29`, `25-29` |
| **Early Career** | `15-29`, `15-39`, `25-49` |
| **Prime Working Age** | `25-54`, `30-34`, `30-54`, `35-39`, `40-44`, `40-59`, `45-49` |
| **Mid Career** | `50-54`, `50-59`, `50-64`, `50-74` |
| **Senior / Pre-Retirement** | `55-59`, `55-64`, `60-64` |
| **Post-Retirement** | `65-69`, `65-74`, `70-74` |
| **Headline Aggregates** | `15-59`, `15-64`, `15-74`, `20-64`, `25-64`, `25-74`, `30-64`, `30-74`, `40-64` |

> **Key metric used in this analysis:** `15-74` is the primary headline aggregate, aligning with Eurostat's standard international unemployment reporting convention.

### Missing Data Profile

```
Country  | Affected Years    | Pattern                         | Impact
---------|-------------------|---------------------------------|---------------------------
BA       | Partial (various) | Reporting inconsistencies       | Excluded from decade calcs
ME       | Partial (various) | Statistical reporting gaps      | Excluded from 2024 sections
Various  | ~12–15% of cells  | Small population cell suppression | Excluded from group means
```

---

## 🧹 Data Cleaning & Transformation

All preprocessing steps are fully implemented and commented in `Notebook/unemployment_analysis.py`.

### Step 1 — Missing Value Detection & Audit

```python
# Quantify NaN per year column
print(df[year_cols].isnull().sum())

# Sample output:
# 2015    273
# 2016    291
# 2017    313  ← Higher NaN in later years due to new small-band suppression rules
# ...

# Identify which countries carry missing data
nan_by_iso = (df.set_index(['SEX', 'AGE', 'ISO'])[year_cols]
                .isnull().any(axis=1)
                .groupby('ISO')
                .sum())
print(nan_by_iso[nan_by_iso > 0])
```

**Decision:** NaN values are **excluded from aggregations**, not imputed. Unemployment rates are highly sensitive to local labour market conditions; mean or interpolation-based imputation would distort country-level statistics. Small-cell suppressions (e.g. `70-74` age band in small countries) are expected by design in Eurostat data.

### Step 2 — Wide-to-Long (Tidy) Transformation

```python
YEAR_COLS = [str(y) for y in range(2015, 2025)]

df_long = (df
    .melt(
        id_vars=['SEX', 'AGE', 'ISO'],
        value_vars=YEAR_COLS,
        var_name='YEAR',
        value_name='URATE'
    )
    .assign(
        YEAR=lambda x: x['YEAR'].astype(int),
        URATE=lambda x: pd.to_numeric(x['URATE'], errors='coerce')
    )
)
# Result: 24,500 rows × 4 columns — tidy format ready for analysis
```

**Why tidy format:** The long format enables `groupby` time-series aggregations, `seaborn` faceted visualisations, cross-dataset joins with employment data, and consistent filter patterns (`df_long[df_long['AGE'] == '15-24']`).

### Step 3 — Demographic Field Normalisation

```python
# Validate SEX only contains expected values
assert set(df['SEX'].unique()) == {'F', 'M'}, "Unexpected SEX values found"

# Validate ISO codes against known set
expected_isos = {'AT','BA','BE','BG','CH','CY','CZ','DE','DK','EE',
                 'ES','FI','FR','GR','HR','HU','IE','IS','IT','LT',
                 'LU','LV','ME','MK','MT','NL','NO','PL','PT','RO',
                 'RS','SE','SI','SK','TR'}
found_isos = set(df['ISO'].unique())
assert found_isos == expected_isos, f"Unexpected ISOs: {found_isos - expected_isos}"

# Validate AGE bands are non-empty strings
assert df['AGE'].notna().all(), "Null AGE values detected"
```

### Step 4 — Duplicate Check

```python
composite_key = ['SEX', 'AGE', 'ISO', 'YEAR']
duplicates = df_long.duplicated(subset=composite_key).sum()
assert duplicates == 0, f"Found {duplicates} duplicates on composite key"
# Confirmed: 0 duplicates
```

### Step 5 — Convenience Analytical Slices

```python
# Pre-filtered views used throughout the analysis
u_1574 = df_long[df_long['AGE'] == '15-74']   # Headline unemployment
u_1524 = df_long[df_long['AGE'] == '15-24']   # Youth
u_5564 = df_long[df_long['AGE'] == '55-64']   # Senior / pre-retirement
u_2554 = df_long[df_long['AGE'] == '25-54']   # Prime working age
```

### Cleaning Pipeline Summary

| Step | Input Shape | Output Shape | Key Action |
|------|------------|-------------|------------|
| Load raw CSV | 2,450 × 13 | 2,450 × 13 | Type inference |
| Melt to long format | 2,450 × 13 | 24,500 × 4 | Structural reshape |
| Type casting | Mixed | `int64` / `float64` | Precision + safety |
| NaN audit | ~3,100 NaN cells | Flagged, not removed | Analytical transparency |
| Deduplication | 24,500 rows | 24,500 rows | 0 removed |
| Validation asserts | — | All passed | Data quality gate |

---

## 🔬 Exploratory Data Analysis

### 1. Unemployment Trends Over Time (2015–2024)

European unemployment (15-74 aggregate) fell by **4.1 percentage points** over the decade, from 10.3% in 2015 to 6.3% in 2024 — a structural improvement representing millions of workers moving from unemployment into employment.

```python
# Headline trend
unemp_yoy = u_1574.groupby('YEAR')['URATE'].mean()
yoy_change = unemp_yoy.diff().rename('YoY Change (pp)')
```

| Year | Mean Rate (%) | YoY Change (pp) |
|------|--------------|-----------------|
| 2015 | 10.33 | — |
| 2016 | 9.42 | −0.91 |
| 2017 | 8.39 | −1.03 |
| 2018 | 7.42 | −0.97 |
| 2019 | 6.88 | −0.54 |
| **2020** | **7.54** | **+0.66 ⚠️** |
| 2021 | 7.43 | −0.11 |
| 2022 | 6.43 | −1.00 |
| 2023 | 6.30 | −0.13 |
| 2024 | 6.26 | −0.04 |

> **COVID footnote:** The 2020 spike (+0.66 pp) was substantially smaller than the employment rate contraction, because many discouraged workers exited the labour force entirely — they were no longer counted as *seeking* work, and thus not classified as unemployed. This labour force withdrawal effect is a known statistical artefact of pandemic-era data.

### 2. Gender Unemployment Gap

```python
gender_gap = (u_1574
    .groupby(['YEAR', 'SEX'])['URATE'].mean()
    .unstack()
    .assign(gap_FM=lambda x: x['F'] - x['M']))
```

Female unemployment exceeds male unemployment, but by a small and declining margin at the European aggregate level (0.3–0.6 pp). This aggregate masks extreme country-level divergence:

| Country | F Rate (%) | M Rate (%) | F−M Gap (pp) | Pattern |
|---------|-----------|-----------|-------------|---------|
| 🔴 Bosnia (BA) | 16.6 | 10.1 | +6.5 | Strong female disadvantage |
| 🔴 Greece (GR) | 12.8 | 8.0 | +4.8 | Mediterranean structural pattern |
| 🔴 Turkey (TR) | 11.8 | 7.1 | +4.7 | Cultural + structural barriers |
| 🟡 Spain (ES) | 12.7 | 10.2 | +2.5 | Dual labour market effects |
| 🟢 Finland (FI) | 7.6 | 9.2 | −1.6 | Men more affected |
| 🟢 Latvia (LV) | 5.8 | 8.0 | −2.2 | Industry-driven male disadvantage |
| 🟢 Lithuania (LT) | 6.5 | 7.8 | −1.3 | Baltic pattern: male sectors hit harder |

### 3. Youth Unemployment Patterns (15-24)

Youth unemployment is the most structurally acute problem in the dataset. The average youth unemployment rate across Europe in 2024 stands at **17.3%** — nearly three times the prime-age rate (6.0%). In 28 of 34 countries, youth unemployment exceeds 10%.

```python
youth_2024 = (u_1524[u_1524['YEAR'] == 2024]
    .groupby('ISO')['URATE'].mean()
    .sort_values(ascending=False))
```

**2024 Youth Unemployment Spectrum:**

```
CRITICAL (≥ 20%):   BA (31.6%), MK (28.7%), ES (26.6%), SE (24.3%), RO (24.2%)
HIGH (15–20%):      GR (22.6%), RS (23.1%), PT (21.6%), LU (21.3%), IT (20.7%)
MODERATE (10–15%):  SK, EE, FI, FR, TR, BE, HR, LT, HU, DK, LV, CY, BG, NO
LOW (< 10%):        SI, PL, IE, AT, CZ, IS, MT, NL, CH, DE
```

**Decade improvement leaders (2015→2024):**
Greece −27.5 pp · Croatia −25.5 pp · Spain −21.8 pp · Serbia −21.2 pp · Italy −20.0 pp

**Decade deterioration (2015→2024):**
Estonia +4.9 pp · Luxembourg +4.0 pp · Sweden +3.9 pp · Denmark +2.5 pp

### 4. Country Unemployment Comparisons (2024)

```python
# Full country ranking — lowest to highest unemployment
country_rank = (u_1574[u_1574['YEAR'] == 2024]
    .groupby('ISO')['URATE'].mean()
    .dropna()
    .sort_values())
```

**Best performers (2024):** Czech Republic (2.7%), Poland (2.9%), Malta (3.2%), Germany (3.4%), Iceland (3.6%), Slovenia (3.8%)

**Worst performers (2024):** Bosnia (13.4%), North Macedonia (12.2%), Spain (11.5%), Greece (10.4%), Turkey (9.5%)

---

## 📐 Statistical Analysis

```python
import numpy as np
from scipy import stats

# ── 1. Mean unemployment by age group ───────────────────────
age_means = df_long.groupby('AGE')['URATE'].mean().sort_values(ascending=False)

# ── 2. Year-over-year percent change ────────────────────────
yoy_pct = unemp_yoy.pct_change() * 100

# ── 3. Gender variance comparison ───────────────────────────
f_rates = u_1574[u_1574['SEX'] == 'F']['URATE'].dropna()
m_rates = u_1574[u_1574['SEX'] == 'M']['URATE'].dropna()

print(f"Female std dev: {f_rates.std():.2f} pp")
print(f"Male std dev:   {m_rates.std():.2f} pp")

# ── 4. Levene's test for variance equality ──────────────────
stat, p_value = stats.levene(f_rates, m_rates)
print(f"Levene test: stat={stat:.3f}, p={p_value:.4f}")
# p < 0.05 → variances are significantly different

# ── 5. COVID shock magnitude per country ────────────────────
u_pivot = u_1574.groupby(['ISO', 'YEAR'])['URATE'].mean().unstack()
covid_shock = (u_pivot[2020] - u_pivot[2019]).sort_values(ascending=False)

# ── 6. Outlier detection via IQR ────────────────────────────
Q1 = u_1574['URATE'].quantile(0.25)
Q3 = u_1574['URATE'].quantile(0.75)
IQR = Q3 - Q1
outliers = u_1574[
    (u_1574['URATE'] < Q1 - 1.5 * IQR) |
    (u_1574['URATE'] > Q3 + 1.5 * IQR)
]
print(f"Outlier observations: {len(outliers)} ({len(outliers)/len(u_1574)*100:.1f}%)")
```

### Key Statistical Outputs

#### Unemployment Rates by Key Age Group (2015–2024)

| Age Group | Mean (%) | Std Dev | Min (%) | Max (%) | Trend |
|-----------|----------|---------|---------|---------|-------|
| `15-24` Youth | 18.3 | 9.7 | 4.8 | 55.0 | Improving |
| `20-24` | 16.5 | 9.6 | 4.2 | 53.4 | Improving |
| `25-54` Prime age | 6.9 | 4.5 | 1.5 | 28.7 | Stable |
| `55-64` Senior | 6.0 | 3.5 | 1.2 | 22.6 | Declining |
| `15-74` Headline | 7.6 | 4.5 | 1.7 | 28.9 | Declining |

#### Gender Statistical Comparison (15-74, All Years)

| Metric | Female | Male | Difference |
|--------|--------|------|------------|
| Mean Rate (%) | 8.08 | 7.23 | F +0.85 pp |
| Median Rate (%) | 6.65 | 5.90 | F +0.75 pp |
| Std Deviation (pp) | 4.62 | 4.47 | F more variable |
| Maximum Observed (%) | 28.9 (GR, 2015) | 28.9 (GR, 2015) | Tied |
| Minimum Observed (%) | 1.7 | 1.7 | Tied |

#### COVID-19 Shock — Unemployment Rise (2019 → 2020)

```
Largest Rises:                    Smallest Rises / Declines:
  Montenegro (ME):  +2.75 pp        Poland (PL):     −0.05 pp  ← resilient
  Estonia (EE):     +2.45 pp        France (FR):     −0.40 pp
  Lithuania (LT):   +2.20 pp        Italy (IT):      −0.70 pp  ← labour force exit
  Iceland (IS):     +2.00 pp        Turkey (TR):     −0.80 pp
  Latvia (LV):      +1.80 pp        Greece (GR):     −1.05 pp  ← discouraged workers
```

> **Analytical note:** Countries showing *falling* unemployment in 2020 (IT, GR, TR) do not represent improvement — they reflect the statistical effect of mass labour force exit. When people stop looking for work, they are no longer counted as unemployed.

#### Outlier Detection Summary

| Method | Threshold | Outlier Count | Notable Outliers |
|--------|-----------|--------------|-----------------|
| IQR (×1.5) | Rate > 15.6% | ~12% of observations | GR (2015–2017), ES (2015–2016), BA |
| Z-score (> 3σ) | Rate > 21.8% | ~4% of observations | GR youth (2015): 55.0%, ES youth: 48%+ |

---

## 📊 Visualisations

All charts are saved to `Outputs/` at 150 DPI and labelled with figure numbers matching the analysis script.

### Figure 1 — Dual-Axis Trend: Employment & Unemployment
**File:** `fig1_dual_trend.png`
**Type:** Dual Y-axis line chart
**Reveals:** The inverse relationship between employment growth (+6.8 pp) and unemployment decline (−4.1 pp) over the decade. The synchronised COVID spike (2020) and rapid recovery are clearly visible. The correlation coefficient (r = −0.73) is displayed on the chart.

---

### Figure 2 — Gender Split: Unemployment Trends
**File:** `fig2_gender_split.png`
**Type:** Dual-panel line chart (Employment + Unemployment side by side)
**Reveals:** Near-identical female and male trajectories at the aggregate level, with a persistent but small female disadvantage (~0.5 pp). Diverges significantly at country level.

---

### Figure 3 — Employment vs. Unemployment Scatter (2024)
**File:** `fig3_scatter_correlation.png`
**Type:** Scatter plot with OLS regression line
**Reveals:** Each dot is a country. The downward-sloping trendline (r = −0.73) confirms that high-employment countries systematically achieve low unemployment. Labelled data points identify outliers (TR, BA, ES vs. IS, NL, CZ).

---

### Figure 4 — Youth Unemployment Country Ranking
**File:** `fig4_youth_unemployment.png`
**Type:** Horizontal bar chart (colour-coded: red ≥ 20%, orange ≥ 12%, green < 12%)
**Reveals:** The full spectrum of youth unemployment performance. Germany (6.5%) sits at the top; Bosnia (31.6%) at the bottom. The mean line (~17%) illustrates how many countries fall above the European average.

---

### Figure 5 — COVID-19 Double Shock
**File:** `fig5_covid_double_shock.png`
**Type:** Side-by-side bar charts
**Reveals:** Left panel shows employment change 2019→2020; right panel shows unemployment change. The asymmetry — some countries saw employment fall without unemployment rising — reveals the labour force exit effect. Key outliers are labelled.

---

### Figure 6 — Country × Year Unemployment Heatmap
**File:** `fig6_unemployment_heatmap.png`
**Type:** Annotated heatmap (34 countries × 10 years)
**Colour scale:** Red-Yellow-Green (RdYlGn_r): red = high unemployment, green = low
**Reveals:** Persistent red zones (BA, MK, GR, ES, TR) vs. consistently green performers (CZ, PL, MT, DE). The 2020 warming band is visible as a horizontal stripe of elevated values.

---

### Figure 7 — Unemployment Trends by Age Cohort
**File:** `fig7_unemployment_by_age.png`
**Type:** Multi-series line chart (4 cohorts: 15-24, 25-54, 55-64, 15-74)
**Reveals:** Youth unemployment is both the highest and most volatile cohort. The 55-64 cohort shows the steepest structural decline. The 2020 spike is proportionally largest for youth and oldest workers.

---

### Figure 8 — Youth Unemployment Change 2015→2024
**File:** `fig8_youth_change.png`
**Type:** Bar chart (green = improved, red = deteriorated)
**Reveals:** The enormous variation in youth unemployment trajectories. Southern European post-crisis recoveries (GR: −27.5 pp, HR: −25.5 pp, ES: −21.8 pp) contrast sharply with Northern European youth market deterioration (EE: +4.9 pp, SE: +3.9 pp).

---

### Figure 9 — Unemployment Gender Gap by Country (2024)
**File:** `fig9_gender_gap_country.png`
**Type:** Bar chart (red = women disadvantaged, green = men disadvantaged)
**Reveals:** Country-level divergence masked by the aggregate near-parity. Turkey, Bosnia, and Greece show women severely disadvantaged; Lithuania, Latvia, and Finland show the reverse — male-dominated industries (manufacturing, construction) drive male disadvantage in these economies.

---

## 💡 Key Findings

> Each finding is framed as a stakeholder-ready insight, grounded in specific data values from the analysis.

**1. European Unemployment Has Halved Its 2015 Peak — But Unevenly**
The mean European unemployment rate fell from **10.3% (2015)** to **6.3% (2024)**, a structural improvement driven by sustained economic expansion, labour market reform, and EU labour mobility. However, this aggregate masks a 10× spread between the best performer (Czech Republic: 2.7%) and the worst (Bosnia: 13.4%) — confirming that "European average" unemployment is a political construct, not a lived reality for workers in weaker economies.

**2. COVID-19 Was a Labour Force Exit Event, Not Just an Unemployment Event**
In 2020, several countries (Italy, Greece, Turkey) showed *falling* unemployment alongside falling employment — a statistical paradox explained by mass labour force withdrawal. Workers who stopped looking for jobs disappeared from the unemployment count. Any analysis treating 2020 unemployment figures at face value without accounting for this effect will reach incorrect conclusions.

**3. Youth Unemployment Is a Crisis in Two-Thirds of Europe**
28 of 34 countries recorded double-digit youth (15-24) unemployment in 2024. Bosnia (31.6%), North Macedonia (28.7%) and Spain (26.6%) are in structural crisis. Critically, even high-employment countries — Sweden (24.3%), Romania (24.2%), Finland (18.8%) — carry alarming youth unemployment rates, pointing to labour market dualisation that protects incumbents at the expense of young entrants.

**4. Germany's Apprenticeship System Is the Most Replicable Policy Insight**
At **6.5% youth unemployment**, Germany outperforms the European average by a factor of nearly 3. The causal mechanism is well-documented: the dual VET (Vocational Education & Training) system creates a direct school-to-work pipeline that eliminates the skill mismatch at the root of youth unemployment. No other country in the dataset approaches Germany's youth unemployment performance through a different institutional model.

**5. Southern Europe Has Made the Greatest Recovery — But From the Greatest Depth**
Greece (−27.5 pp), Croatia (−25.5 pp), Spain (−21.8 pp), and Italy (−20.0 pp) achieved the largest youth unemployment improvements since 2015. These gains are real but contextually important: they represent recovery from post-2008 GFC peaks of 55–60% youth unemployment, not structural transformation. These countries remain structurally above the European average and are vulnerable to the next economic cycle.

**6. Turkey's Labour Market Gender Inequality Is the Dataset's Most Extreme Outlier**
Turkey's female unemployment rate (11.8%) is **4.7 pp above its male rate** (7.1%) — the largest F−M gap in the dataset. Combined with the companion employment data showing a 36.3 pp employment gender gap, Turkey presents a comprehensive picture of structural female labour market exclusion driven by cultural norms, childcare infrastructure gaps, and legal barriers. This is the largest single opportunity for GDP improvement available to any country in the dataset.

**7. Northern European Youth Markets Are Silently Deteriorating**
While Southern Europe recovered dramatically, Sweden (+3.9 pp), Estonia (+4.9 pp), Denmark (+2.5 pp), and Luxembourg (+4.0 pp) saw youth unemployment *increase* between 2015 and 2024. These countries have strong overall labour markets but are experiencing labour market dualisation: high housing costs, insider-outsider dynamics, and limited youth-specific pathways are pricing young workers out of the most productive urban labour markets.

**8. Czech Republic and Poland Are the Underappreciated Success Stories**
With youth unemployment of 9.3% and 10.8% respectively in 2024, and overall unemployment of just 2.7% and 2.9%, Czechia and Poland combine strong youth outcomes with near-full adult employment — outperforming many wealthier Nordic economies on youth metrics. Their success is attributed to flexible secondary education pathways, strong manufacturing job creation, and effective school-to-work transition programs.

---

## 🛠️ Tech Stack

### Core Dependencies

| Library | Version | Role in This Project |
|---------|---------|---------------------|
| `pandas` | ≥ 2.0 | Data ingestion, wide-to-long reshaping, groupby aggregations, pivot tables |
| `numpy` | ≥ 1.24 | Numerical operations, growth rate calculations, IQR outlier bounds |
| `matplotlib` | ≥ 3.7 | Base figure/axis management; dual-axis charts; custom colour coding |
| `seaborn` | ≥ 0.12 | Heatmaps, box plots, regression scatter plots, statistical themes |
| `scipy` | ≥ 1.11 | Levene's test, Pearson/Spearman correlation, statistical significance testing |
| `scikit-learn` *(optional)* | ≥ 1.3 | Linear trend fitting, OLS regression, preprocessing pipelines |

### Development Environment

| Tool | Version / Detail |
|------|-----------------|
| Python | 3.10+ |
| OS | Linux / macOS / Windows |
| Recommended IDE | VS Code with Python + Jupyter extensions |
| Notebook alternative | JupyterLab (convert `.py` to `.ipynb` with `jupytext`) |

---

## ⚡ Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Unemployment-Analytics.git
cd Unemployment-Analytics
```

### 2. Create & Activate a Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install All Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Full Analysis Pipeline

```bash
python Notebook/unemployment_analysis.py
```

All figures and the statistical summary report will be saved to `Outputs/`.

### 5. Review Outputs

```
Outputs/
├── fig1_dual_trend.png           ← Employment + unemployment dual-axis
├── fig2_gender_split.png         ← Gender-disaggregated comparison
├── fig3_scatter_correlation.png  ← Country-level correlation scatter
├── fig4_youth_unemployment.png   ← Youth unemployment rankings
├── fig5_covid_double_shock.png   ← Pandemic shock analysis
├── fig6_unemployment_heatmap.png ← Full country × year matrix
├── fig7_unemployment_by_age.png  ← Cohort trend comparison
├── fig8_youth_change.png         ← Youth improvement 2015→2024
├── fig9_gender_gap_country.png   ← Country gender gap breakdown
└── analytics_report.txt          ← Full statistical output
```

### `requirements.txt`

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
scikit-learn>=1.3.0
```

---

## 🚀 Future Work

### Short-Term (v1.1)

- [ ] **Employment Dataset Integration** — Merge with `occupazione.csv` on `(SEX, AGE, ISO, YEAR)` to compute combined labour market dashboards, employment-unemployment correlation heatmaps, and composite labour market health scores per country
- [ ] **Interactive Plotly Dashboard** — Rebuild all 9 static charts in `Plotly` or `Dash` for browser-based interactivity, country drill-down, and year-range sliders
- [ ] **Automated PDF Report** — Generate a formatted analytical report using `reportlab` or `WeasyPrint` as a pipeline artifact, enabling scheduled delivery to stakeholders

### Medium-Term (v1.2)

- [ ] **Time-Series Forecasting** — Apply `Prophet`, `ARIMA`, or `Exponential Smoothing` to project unemployment trajectories to 2027 under baseline, optimistic, and recessionary scenarios; quantify uncertainty bands
- [ ] **Macroeconomic Correlation Analysis** — Enrich with Eurostat GDP growth, productivity, and wage data to calculate Okun's Law coefficients per country and identify where the growth-unemployment relationship has broken down
- [ ] **Youth NEET Rate Modelling** — Combine unemployment data with education participation rates to estimate NEET (Not in Education, Employment, or Training) risk by country and age band; flag at-risk cohorts

### Long-Term (v2.0)

- [ ] **Machine Learning Predictions** — Train gradient boosting models (`XGBoost`, `LightGBM`) to predict next-year country unemployment rates using lagged macroeconomic features; evaluate against naive and ARIMA baselines
- [ ] **Policy Intervention Causal Analysis** — Use difference-in-differences methodology to evaluate the employment effect of specific policy events: EU Youth Guarantee launch dates, minimum wage changes, pension reform implementations
- [ ] **Eurostat API Integration** — Connect to the live Eurostat REST API for automated monthly data refreshes; implement drift alerting when unemployment rates deviate from forecasted trend bands by more than 0.5 pp
- [ ] **Labour Market Archetype Clustering** — Apply `k-means` or DBSCAN clustering to group countries into unemployment archetypes (post-crisis recovery, structural fragility, near-full-employment, gender-stratified) and track cluster membership changes over time

---

## 🤝 Contributing

Contributions are welcome. To contribute:

```bash
# Fork the repository, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines and include docstrings for all new functions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Dataset sourced from Eurostat under the [Eurostat data reuse policy](https://ec.europa.eu/eurostat/about-us/policies/copyright).

---

## 🙏 Acknowledgements

- **Primary Data Source:** [Eurostat Labour Force Survey (LFS)](https://ec.europa.eu/eurostat/web/lfs)
- **Geographic Coverage:** EU-27 + Iceland, Norway, Switzerland, Turkey, Bosnia & Herzegovina, Montenegro, North Macedonia, Serbia
- **Methodology:** [Eurostat Unemployment Statistics Methodology](https://ec.europa.eu/eurostat/statistics-explained/index.php/Unemployment_statistics)
- **Companion Project:** [Employment Trend Analysis (2015–2024)](../Employment-Analytics/README.md) — `occupazione.csv`

---

<div align="center">

**Built with 🐍 Python · 🐼 pandas · 📊 seaborn · 🔬 scipy**

*Senior Data Scientist Portfolio Project — Labour Market Analytics*

</div>
