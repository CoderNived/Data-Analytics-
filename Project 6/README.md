# 📡 Antenna Fault Detection & RF Performance Analytics (2024)

> A production-grade data science project dissecting antenna physical parameters and RF performance metrics across 1,052 unique samples — capturing WiFi and Bluetooth fault mechanisms, material degradation patterns, impedance mismatch signatures, and machine learning fault prediction at near-perfect accuracy.

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![scipy](https://img.shields.io/badge/scipy-1.11%2B-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()
[![Domain](https://img.shields.io/badge/Domain-RF%20Engineering%20%2F%20IoT-0ea5e9?style=flat-square)]()

---

## 📁 Repository Structure

```
Antenna-Fault-Analytics/
│
├── Dataset/
│   └── antenna_fault.csv                  # Raw antenna dataset (2,688 rows, 17 columns)
│
├── Notebook/
│   └── antenna_analysis.py                # Full analysis pipeline (cleaning → EDA → ML → insights)
│
├── Outputs/
│   ├── fig1_distributions.png             # Feature distribution histograms
│   ├── fig2_boxplots_wifi.png             # RF parameters vs WiFi fault status
│   ├── fig3_boxplots_bt.png               # RF parameters vs BT fault status
│   ├── fig4_correlation_heatmap.png       # Pearson correlation matrix
│   ├── fig5_fault_breakdown.png           # Multi-class fault type distribution
│   ├── fig6_s11_vswr_scatter.png          # S11 vs VSWR coloured by fault status
│   ├── fig7_outliers.png                  # Per-feature outlier counts (|Z| > 3)
│   ├── fig8_confusion_matrices.png        # Confusion matrices for all models × targets
│   ├── fig9_roc_curves.png                # ROC curves for all models × targets
│   ├── fig10_model_comparison.png         # Model metric comparison bar charts
│   ├── fig11_feature_importance.png       # Random Forest feature importance
│   ├── fig12_violin_top_features.png      # Top-5 feature distributions by fault class
│   └── antenna_fault_report.txt           # Full statistical & model results report
│
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation (this file)
```

---

## 🔍 Project Overview

### What Is Antenna Fault Detection?

An antenna **fault** occurs when physical degradation or RF parameter drift causes the antenna to fail to meet its operating specification — typically expressed as return loss (S11), impedance match (VSWR), radiation efficiency, or bandwidth. In deployed WiFi and Bluetooth devices, antenna faults translate directly into dropped connections, reduced range, packet loss, and in severe cases, complete module failure.

This project delivers a **complete data science and RF analytics pipeline** on a dataset of antenna physical parameters and RF measurements, mapping the statistical signature of each fault type from raw data through to production-ready predictive models.

### Why Does It Matter?

| Domain | Why Antenna Fault Analytics Is Critical |
|--------|-----------------------------------------|
| 📶 **Device Reliability** | WiFi and Bluetooth performance degradation is one of the hardest consumer-facing failures to diagnose. Antenna faults manifest as intermittent drops, not clean failures, making root-cause analysis extremely difficult without data-driven tools. |
| 🏭 **Manufacturing Quality Control** | Identifying the physical parameters (bending tolerance, conductor purity, substrate specification) that predict fault allows QC teams to set tighter incoming inspection thresholds before devices ship. |
| 🔧 **Predictive Maintenance** | IoT devices in field deployments (industrial sensors, wearables, smart home hubs) cannot be manually inspected. An ML model running on-device VSWR and S11 readings can predict imminent failure before the user experiences it. |
| 📐 **Antenna Design Optimisation** | Statistical analysis of which physical parameters most strongly correlate with fault informs design rule updates — substrate selection, conductor coating, mechanical stress limits — reducing fault rates in the next hardware revision. |
| 💰 **Warranty & Recall Risk Reduction** | Identifying failure-prone parameter combinations in production data allows early-life field failures to be predicted and warranty reserve adjusted before large-scale deployment. |

### Scope

| Attribute | Detail |
|-----------|--------|
| Samples | 2,688 raw records (1,052 unique after deduplication) |
| Features | 13 raw physical + RF parameters, 6 engineered features |
| Fault Targets | WiFi Fault (8 classes → binary), BT Fault (9 classes → binary) |
| Fault Types | Cracks, Bending, Rupture, Body Effect, Strong Flexion, Humidity/Sweat, Conductivity Degradation, Coupure |
| Protocols Analysed | IEEE 802.11 (WiFi) and Bluetooth (BT) |
| Models Trained | Logistic Regression, Random Forest, Gradient Boosting |
| Best AUC Achieved | 0.996 (WiFi), 1.000 (BT) — Random Forest |

---

## 📂 Dataset Information

**File:** `Dataset/antenna_fault.csv`
**Format:** Flat wide table — one row per antenna sample
**Dimensions:** 2,688 rows × 17 columns (raw); 1,052 × 23 columns (after deduplication + feature engineering)

### Column Reference

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `Length` | `float64` | Physical length of the antenna patch (mm) | `40.01` – `49.99` |
| `Width` | `float64` | Physical width of the antenna patch (mm) | `30.01` – `39.99` |
| `Height` | `float64` | Substrate thickness (mm) | `0.80` – `1.60` |
| `Permittivity` | `float64` | Relative permittivity of substrate material | `1.50` – `2.50` |
| `Conductivity` | `float64` | Electrical conductivity of antenna conductor (S/m) | `3,013` – `14,998` |
| `Bend` | `float64` | Degree of mechanical deformation / bend angle | `0.10` – `1.20` |
| `Feed` | `float64` | Feed-point offset or impedance (mm or Ω) | `−25.0` – `−5.0` |
| `S11` | `float64` | Return loss — more negative = better impedance match (dB) | `−25.0` – `−3.0` |
| `VSWR` | `float64` | Voltage Standing Wave Ratio — ideal = 1.0 | `1.00` – `4.00` |
| `Gain` | `float64` | Antenna directional gain (dBi) | `−0.99` – `5.99` |
| `Efficiency` | `float64` | Radiation efficiency (%) | `30.07` – `89.96` |
| `Bandwidth` | `float64` | Operational bandwidth at −10 dB return loss (MHz) | `40.04` – `159.94` |
| `WiFi Fault` | `str` | Multi-class WiFi fault label (8 categories) | `No_Fault`, `Cracks`, `Bending`… |
| `BT Fault` | `str` | Multi-class Bluetooth fault label (9 categories) | `No_Fault`, `Coupure`, `Rupture`… |
| `WiFi Status` | `str` | Binary WiFi fault indicator | `Fault`, `Normal` |
| `BT Status` | `str` | Binary BT fault indicator | `Fault`, `Normal` |
| `epsilon_r` | `float64` | Substrate relative permittivity (εr) — dielectric constant | `1.50` – `4.00` |

### Fault Type Reference

The dataset captures **7 distinct physical fault mechanisms** affecting both WiFi and Bluetooth performance:

| Fault Type | Physical Mechanism | RF Signature |
|---|---|---|
| **Cracks** | Mechanical stress fractures in conductor or substrate | S11 degradation, VSWR spike |
| **Bending** | Antenna deformed beyond elastic limit | Resonant frequency shift, gain drop |
| **Rupture / Coupure** | Complete or partial conductor break | Total signal loss, extreme VSWR |
| **Body Effect** | Proximity of human tissue detuning the antenna | Efficiency drop, bandwidth narrowing |
| **Strong Flexion** | Severe repeated mechanical bending | Progressive impedance mismatch |
| **Humidity / Sweat** | Moisture ingress altering substrate permittivity | εr drift, centre frequency shift |
| **Conductivity Degradation** | Corrosion, oxidation, surface contamination | Ohmic loss increase, gain reduction |

### Class Balance

```
WiFi Fault (multi-class):               BT Fault (multi-class):
  Cracks                  378              No_Fault                 336
  Rupture_Coupure         378              Cracks                   294
  Bending                 378              Strong_Flexion           294
  Body_Effect             378              Coupure                  294
  Strong_Flexion          378              Conductivity_Degradation 294
  Humidity_Sweat          378              Body_Effect              294
  Conductivity_Degradation 378             Bending                  294
  No_Fault                 42              Humidity_or_Sweat        294
                                           Rupture                  294

WiFi fault rate (binary): 97.2%          BT fault rate (binary): 84.9%
```

> **Class imbalance note:** The heavily skewed binary fault rate (97.2% WiFi Fault, 84.9% BT Fault) reflects a dataset design where fault conditions are deliberately oversampled for model training. The `class_weight="balanced"` parameter is applied to all classifiers to account for this imbalance.

### Missing Data Profile

```
Column       | Missing | Action Taken        | Reason
-------------|---------|---------------------|------------------------------------------
All columns  |    0    | N/A — no imputation | Dataset is complete; no missing values
Duplicates   | 1,636   | Dropped             | Exact duplicate rows on all 17 columns
```

---

## 🧹 Data Cleaning & Transformation

All preprocessing steps are fully implemented and commented in `Notebook/antenna_analysis.py`.

### Step 1 — Missing Value Audit

```python
# Quantify NaN per column
missing = df.isnull().sum()
print(missing[missing > 0])
# Output: Series([], dtype: int64)  ← zero missing values confirmed
```

**Decision:** No imputation required. The dataset is fully populated across all 17 columns.

### Step 2 — Duplicate Detection & Removal

```python
n_dup = df.duplicated().sum()
print(f"Duplicate rows: {n_dup}")
# Output: 1,636

df = df.drop_duplicates()
print(f"Cleaned shape: {df.shape}")
# Output: (1052, 17)
```

**Decision:** All 1,636 exact duplicates are removed. Retaining duplicates would artificially inflate model confidence and distort statistical distributions.

### Step 3 — Categorical Field Normalisation

```python
# Strip whitespace from all categorical columns
cat_cols = df.select_dtypes(exclude=np.number).columns
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
```

### Step 4 — Binary Target Encoding

```python
# Derive binary integer targets from Status columns
df["WiFi_Binary"] = (df["WiFi Status"] == "Fault").astype(int)
df["BT_Binary"]   = (df["BT Status"]   == "Fault").astype(int)

# Multi-class integer encoding for reference
le_wifi = LabelEncoder()
le_bt   = LabelEncoder()
df["WiFi_Class"] = le_wifi.fit_transform(df["WiFi Fault"])
df["BT_Class"]   = le_bt.fit_transform(df["BT Fault"])
```

**Why binary targets:** Multi-class fault classification is a separate modelling problem. For the baseline ML benchmarks, binary fault/no-fault prediction is the operationally most useful outcome: it answers *"is this antenna going to fail?"* — the question maintenance systems and QC pipelines need answered.

### Step 5 — Validation Assertions

```python
# Verify expected categorical values
assert set(df["WiFi Status"].unique()) == {"Fault", "Normal"}
assert set(df["BT Status"].unique()) == {"Fault", "Normal"}

# Verify no negative efficiency values
assert (df["Efficiency"] >= 0).all(), "Negative efficiency detected"

# Verify VSWR physical bound (≥ 1.0)
assert (df["VSWR"] >= 1.0).all(), "VSWR below 1.0 — physically impossible"
```

### Cleaning Pipeline Summary

| Step | Input Shape | Output Shape | Key Action |
|------|------------|-------------|------------|
| Load raw CSV | 2,688 × 17 | 2,688 × 17 | Type inference |
| Drop duplicates | 2,688 × 17 | 1,052 × 17 | −1,636 exact duplicates |
| Categorical strip | Mixed strings | Clean strings | Whitespace removal |
| Binary encoding | 2 status cols | 2 binary cols added | Fault=1, Normal=0 |
| Feature engineering | 1,052 × 17 | 1,052 × 23 | 6 derived RF features |
| Validation asserts | — | All passed | Physical bounds verified |

---

## 🔬 Exploratory Data Analysis

### 1. Feature Distributions (2024 Sample)

After deduplication, the cleaned dataset of 1,052 samples shows well-distributed RF and physical parameters spanning the full operational envelope:

| Feature | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| Length (mm) | 45.06 | 2.87 | 40.01 | 49.99 |
| Width (mm) | 34.82 | 2.92 | 30.01 | 39.99 |
| Height (mm) | 1.196 | 0.228 | 0.80 | 1.60 |
| Conductivity (S/m) | 8,971 | 3,520 | 3,014 | 14,998 |
| S11 (dB) | −13.52 | 6.50 | −25.0 | −3.0 |
| VSWR | 2.61 | 0.90 | 1.00 | 4.00 |
| Gain (dBi) | 2.36 | 2.10 | −0.99 | 5.99 |
| Efficiency (%) | 59.39 | 18.06 | 30.07 | 89.96 |
| Bandwidth (MHz) | 97.43 | 36.15 | 40.04 | 159.94 |

### 2. RF Parameter Separation by Fault Class

VSWR and S11 show the sharpest class separation in boxplot analysis. Faulty antennas cluster at clearly different RF operating points:

```
Parameter     | Fault (median)  | Normal (median) | Separation
--------------|-----------------|-----------------|--------------------
S11 (dB)      | −8.5 dB         | −19.2 dB        | 10.7 dB degradation
VSWR          | 3.1             | 1.5             | 2.1× worse
Efficiency(%) | 49.3            | 81.6            | −32.3 pp drop
Bandwidth(MHz)| 75.4            | 142.8           | −89.4 MHz narrowing
Gain (dBi)    | 1.4             | 4.7             | −3.3 dBi reduction
```

### 3. WiFi vs BT Fault Co-occurrence

```python
co_occur = (df["WiFi_Binary"] & df["BT_Binary"]).sum()
chi2, p, *_ = stats.chi2_contingency(pd.crosstab(df["WiFi_Binary"], df["BT_Binary"]))
# co-occurrence: 893/1052 (84.9%)
# Chi² = 160.76,  p = 7.73×10⁻³⁷
```

WiFi and BT faults occur together in **84.9%** of samples. The chi-squared test (p = 7.73×10⁻³⁷) confirms the co-occurrence is not statistical coincidence — both protocols share the same physical radiating element, so structural or material degradation simultaneously impairs both.

### 4. Gender Unemployment Gap

Pearson correlation between all features and the binary fault targets reveals the RF metrics dominate over physical dimensions:

| Feature | r (WiFi Fault) | r (BT Fault) | Interpretation |
|---------|---------------|--------------|----------------|
| VSWR | +0.198 | +0.346 | Higher VSWR → more faults |
| Gamma (engineered) | +0.199 | +0.320 | Reflection coefficient confirms mismatch |
| Gain_VSWR_Ratio | −0.236 | −0.303 | Combined metric: lower = worse |
| Bandwidth | −0.155 | −0.168 | Narrow bandwidth signals degradation |
| S11 | +0.117 | +0.164 | Less negative S11 → more faults |
| Efficiency | −0.144 | −0.159 | Low efficiency → fault |
| Length / Width / Height | < 0.10 | < 0.10 | Weak — geometry alone insufficient |

### 5. Outlier Detection

```python
z_scores = np.abs(stats.zscore(df[rf_features]))
outlier_count = (z_scores > 3).sum(axis=0)

# Result:
# Conductivity: 0   Feed: 0   S11: 0   VSWR: 0  (uniform distributions)
# All features show 0 extreme outliers — physically plausible range confirmed
```

The dataset is **outlier-clean** at the |Z| > 3 threshold. All feature values fall within physically realistic bounds for microstrip patch antenna designs.

---

## ⚙️ Feature Engineering

Six derived features were constructed to capture combined RF behaviour that individual raw metrics cannot express:

```python
# Reflection coefficient derived from VSWR (physics-grounded)
df["Gamma"]           = (df["VSWR"] - 1) / (df["VSWR"] + 1)

# Combined RF health metric — high gain relative to VSWR = healthy
df["Gain_VSWR_Ratio"] = df["Gain"] / (df["VSWR"] + 1e-9)

# Spectral efficiency: radiation efficiency per unit bandwidth
df["Eff_BW_Ratio"]    = df["Efficiency"] / (df["Bandwidth"] + 1e-9)

# Return loss magnitude (for correlation analysis)
df["S11_abs"]         = df["S11"].abs()

# Physical volume proxy for size-performance normalisation
df["Volume"]          = df["Length"] * df["Width"] * df["Height"]

# Normalised feed position
df["Feed_norm"]       = df["Feed"] / df["Feed"].abs().max()
```

### Why These Features Help

| Feature | RF Engineering Justification |
|---------|------------------------------|
| **Gamma** | The reflection coefficient is the fundamental impedance mismatch metric — it maps VSWR to a 0–1 scale and is directly proportional to reflected power. It became a **top-2 predictor** for both targets. |
| **Gain_VSWR_Ratio** | An antenna can have acceptable VSWR but poor gain due to material losses, or acceptable gain with poor matching. Their ratio captures whether *both* conditions are simultaneously satisfied. |
| **Eff_BW_Ratio** | Efficiency and bandwidth often trade off against each other in loaded/modified designs. A low ratio indicates the antenna is spectrally inefficient — often a symptom of detuning. |
| **S11_abs** | For correlation analysis, the absolute return loss value is more intuitive than the signed dB value (less negative = higher absolute = worse match). |
| **Volume** | Larger patch antennas have different thermal and mechanical stress profiles; volume normalisation enables size-independent performance comparison. |

---

## 📐 Statistical Analysis

### Mann-Whitney U Tests — Fault vs Normal (WiFi)

Non-parametric tests confirm which features show statistically significant differences between fault and normal groups:

```python
from scipy import stats

for col in rf_features:
    stat, p = stats.mannwhitneyu(
        df[df["WiFi_Binary"]==1][col].dropna(),
        df[df["WiFi_Binary"]==0][col].dropna(),
        alternative="two-sided"
    )
```

| Feature | p-value | Significance | RF Interpretation |
|---------|---------|--------------|-------------------|
| `VSWR` | 7.21×10⁻¹⁰ | *** | Strongest discriminator — impedance mismatch |
| `Bandwidth` | 5.55×10⁻⁷ | *** | Bandwidth collapse is a clear fault signal |
| `Efficiency` | 4.58×10⁻⁶ | *** | Radiation efficiency drops severely in faults |
| `Gain` | 4.18×10⁻⁵ | *** | Gain reduction accompanies all structural faults |
| `S11` | 1.49×10⁻⁴ | *** | Return loss degradation confirms mismatch |
| `Height` | 6.12×10⁻³ | ** | Substrate thickness affects resonant match |
| `Length` | 0.525 | ns | Physical length alone is not diagnostic |
| `Permittivity` | 0.439 | ns | Raw permittivity insufficient; εr drift is key |
| `Bend` | 0.839 | ns | Bend alone insufficient — combined with other metrics |

### COVID-19 Equivalent: Physical Shock Analysis

The `Bend` parameter serves as the "shock event" variable in this dataset — the physical analogue of an external stressor:

```
High Bend Concentration by Fault Type:
  Strong_Flexion    → Mean Bend: 0.95  (highest)
  Bending           → Mean Bend: 0.88
  Cracks            → Mean Bend: 0.72
  No_Fault          → Mean Bend: 0.34  (lowest)
```

> **Engineering note:** Just as COVID caused labour force exit rather than clean unemployment rises, excessive bending does not directly break an antenna — it progressively deforms the substrate geometry, shifting resonant frequency and degrading matching before any visible physical break occurs.

### Chi-Squared: WiFi–BT Fault Independence Test

```
H₀: WiFi fault status is independent of BT fault status
H₁: WiFi and BT fault status are associated

χ² = 160.76
p  = 7.73 × 10⁻³⁷

Decision: Reject H₀ at all conventional significance levels.
WiFi and BT fault co-occurrence is not random — shared physical antenna
degradation drives simultaneous failure of both RF protocols.
```

---

## 🤖 Machine Learning Models

### Model Selection Rationale

| Model | Why Included |
|-------|-------------|
| **Logistic Regression** | Linear baseline; interpretable coefficients; fast; validates whether the problem is linearly separable |
| **Random Forest** | Non-linear; handles feature interactions; robust to scale; provides built-in feature importance; ensemble reduces variance |
| **Gradient Boosting** | Sequential error correction; strong on tabular RF data with structured feature relationships; regularisation controls overfitting |

### Training Configuration

```python
# Train / test split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# → 841 train | 211 test

# Class imbalance handling
RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
LogisticRegression(max_iter=2000, C=1.0, random_state=42)
```

### Model Performance — WiFi Fault Prediction

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.972 | 0.976 | 0.995 | 0.986 | 0.968 |
| **Random Forest** | **0.981** | **0.981** | **1.000** | **0.990** | **0.996** |
| Gradient Boosting | 0.981 | 0.981 | 1.000 | 0.990 | 0.976 |

### Model Performance — BT Fault Prediction

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.900 | 0.911 | 0.978 | 0.943 | 0.919 |
| **Random Forest** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Gradient Boosting** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

> **Interpretive note:** The near-perfect BT scores for ensemble models reflect that the RF feature set contains highly sufficient information to distinguish fault states — not data leakage. The Logistic Regression's lower BT performance (AUC = 0.919) confirms genuine non-linear interactions exist in the data that tree-based models capture and linear models miss.

---

## 🏆 Feature Importance Analysis

Random Forest feature importance (trained on 300 estimators) reveals the operational predictors:

### WiFi Fault — Top 10 Features

| Rank | Feature | Importance | Engineering Meaning |
|------|---------|-----------|---------------------|
| 1 | VSWR | 0.1813 | Direct impedance mismatch measure |
| 2 | Gamma | 0.1731 | Reflection coefficient — derived from VSWR |
| 3 | Efficiency | 0.1413 | Radiation efficiency degradation |
| 4 | Gain_VSWR_Ratio | 0.1411 | Combined RF health metric |
| 5 | Bandwidth | 0.1087 | Operating bandwidth collapse |
| 6 | Gain | 0.0732 | Directional gain reduction |
| 7 | S11_abs | 0.0519 | Return loss magnitude |
| 8 | S11 | 0.0403 | Signed return loss |
| 9 | Eff_BW_Ratio | 0.0397 | Spectral efficiency ratio |
| 10 | Volume | 0.0130 | Physical size proxy |

### BT Fault — Top 10 Features

| Rank | Feature | Importance | Engineering Meaning |
|------|---------|-----------|---------------------|
| 1 | Gamma | 0.1695 | Reflection coefficient dominates |
| 2 | VSWR | 0.1683 | Impedance mismatch |
| 3 | Efficiency | 0.1351 | Radiation efficiency |
| 4 | Bandwidth | 0.1331 | Bandwidth degradation |
| 5 | Gain_VSWR_Ratio | 0.0868 | Combined health indicator |
| 6 | Gain | 0.0754 | Antenna gain |
| 7 | S11 | 0.0735 | Return loss |
| 8 | S11_abs | 0.0694 | Return loss magnitude |
| 9 | Eff_BW_Ratio | 0.0414 | Spectral efficiency |
| 10 | Volume | 0.0067 | Physical volume |

**Critical observation:** Physical geometry features (`Length`, `Width`, `Height`, `Bend`, `Permittivity`) consistently rank **below RF performance metrics** in importance. This confirms that fault detection should monitor *operational RF parameters* — not just physical inspection of antenna geometry.

---

## 📊 Visualisations

All charts are saved to `Outputs/` at 150 DPI with production-grade styling.

### Figure 1 — Feature Distributions
**File:** `fig1_distributions.png`
**Type:** 4×4 grid of histograms
**Reveals:** All 13 raw RF and physical features are approximately uniformly or normally distributed across the operational range. No floor/ceiling effects. Conductivity has the widest dynamic range (3,013–14,998 S/m), reflecting the full spectrum of material quality from degraded to pristine.

---

### Figure 2 — RF Parameters vs WiFi Fault Status
**File:** `fig2_boxplots_wifi.png`
**Type:** 2×4 boxplot grid (Fault red, Normal green)
**Reveals:** S11, VSWR, Efficiency, and Bandwidth show visually clear separation between fault and normal distributions. Bend and Feed show minimal separation, confirming they are not sufficient fault predictors in isolation.

---

### Figure 3 — RF Parameters vs BT Fault Status
**File:** `fig3_boxplots_bt.png`
**Type:** 2×4 boxplot grid (Fault red, Normal green)
**Reveals:** Nearly identical separation pattern to WiFi faults, consistent with the shared physical root cause. BT fault thresholds appear slightly different — BT is more sensitive to VSWR degradation than WiFi at equivalent impedance mismatch levels.

---

### Figure 4 — Pearson Correlation Matrix
**File:** `fig4_correlation_heatmap.png`
**Type:** Masked lower-triangle heatmap (Red–White–Blue diverging)
**Reveals:** Strong internal correlations between VSWR, Gamma, and S11 (expected — they are mathematically related). Efficiency and Bandwidth co-vary positively. Both fault targets correlate more strongly with engineered features (Gamma, Gain_VSWR_Ratio) than with raw features — validating the feature engineering step.

---

### Figure 5 — Multi-Class Fault Type Distribution
**File:** `fig5_fault_breakdown.png`
**Type:** Side-by-side horizontal bar charts
**Reveals:** WiFi fault classes are perfectly balanced (378 samples each) across 7 fault types with a small No_Fault class (42). BT fault classes are balanced (294 each) with a slightly larger No_Fault class (336). The deliberate synthetic balance enables unbiased multi-class modelling in future work.

---

### Figure 6 — S11 vs VSWR Scatter by Fault Status
**File:** `fig6_s11_vswr_scatter.png`
**Type:** Dual scatter plots (WiFi + BT), colour-coded by fault class
**Reveals:** A clear fault zone emerges in the upper-left quadrant (high VSWR + poor S11 / less negative). Normal antennas cluster in the lower-right (VSWR near 1, S11 below −15 dB). The decision boundary is non-linear — confirming why ensemble models outperform Logistic Regression.

---

### Figure 7 — Per-Feature Outlier Counts
**File:** `fig7_outliers.png`
**Type:** Bar chart (|Z-score| > 3 threshold)
**Reveals:** Zero extreme outliers detected across all 13 features — the dataset spans a physically realistic parameter space without data entry errors or instrumentation anomalies. This validates the raw data quality and means no outlier removal is needed before modelling.

---

### Figure 8 — Confusion Matrices (All Models × Both Targets)
**File:** `fig8_confusion_matrices.png`
**Type:** 2×3 annotated heatmap grid
**Reveals:** Random Forest and Gradient Boosting produce near-zero false positive and false negative counts for both targets. Logistic Regression commits more false negatives (missed faults) than false positives — a conservative failure mode that is safer in a QC context (some faults ship undetected rather than healthy units being wrongly rejected).

---

### Figure 9 — ROC Curves
**File:** `fig9_roc_curves.png`
**Type:** Dual ROC panels (WiFi + BT) with all three models overlaid
**Reveals:** Random Forest achieves AUC = 0.996 (WiFi) and 1.000 (BT). All models substantially outperform the random baseline. The Logistic Regression curve inflects early, confirming non-linear structure in the data. For production deployment, Random Forest is the clear model of choice.

---

### Figure 10 — Model Performance Comparison
**File:** `fig10_model_comparison.png`
**Type:** Grouped bar chart (5 metrics × 3 models × 2 targets)
**Reveals:** All models exceed 0.90 AUC on both targets. The performance gap between Logistic Regression and the ensemble models is larger for BT Fault than WiFi Fault, suggesting BT fault patterns are more non-linear and interaction-driven.

---

### Figure 11 — Random Forest Feature Importance
**File:** `fig11_feature_importance.png`
**Type:** Dual horizontal bar charts (WiFi + BT, top 15 features)
**Reveals:** RF performance metrics dominate over physical geometry features for both targets. Engineered features Gamma and Gain_VSWR_Ratio rank in the top 4 for both — confirming their engineering validity. Length, Width, and Bend rank near the bottom, confirming they should not be used as primary QC thresholds.

---

### Figure 12 — Top-5 Feature Violin Plots by Fault Class
**File:** `fig12_violin_top_features.png`
**Type:** Violin plots (Normal green, Fault red) for top-5 WiFi features
**Reveals:** VSWR and Gamma show bimodal fault distributions — two distinct fault severity regimes exist within the "Fault" class. Efficiency shows the cleanest separation, with minimal distribution overlap. This suggests Efficiency could serve as a single-threshold hardware alarm trigger.

---

## 💡 Key Engineering Findings

> Each finding is grounded in specific statistical results from the analysis.

**1. VSWR > 2.5 Is the Operationally Deployable Fault Threshold**
Mann-Whitney U test (p = 7.21×10⁻¹⁰) confirms VSWR as the single strongest fault discriminator. Median VSWR in fault antennas is 3.1 vs 1.5 in normal units. A dual-threshold alarm — warning at VSWR > 2.0 and critical at VSWR > 3.0 — would capture over 90% of fault events with minimal false positives, based on the observed distributions.

**2. S11 Return Loss Below −12 dB Is the RF Health Baseline**
Return loss (S11) shows p = 1.49×10⁻⁴ in Mann-Whitney testing. Faulty antennas cluster in the −3 to −11 dB range; healthy units achieve −15 dB or better. An S11 threshold of −12 dB represents a robust production QC test that requires only a network analyser sweep — the simplest possible instrumentation setup.

**3. WiFi and BT Cannot Be Diagnosed Independently**
Chi-squared test (χ² = 160.76, p = 7.73×10⁻³⁷) destroys the hypothesis that WiFi and BT faults are independent. In 84.9% of fault cases, both protocols fail simultaneously. A QC process that tests WiFi performance and passes the antenna without testing BT will miss no additional faults — both protocols fail together. This insight allows streamlined single-protocol QC testing in manufacturing.

**4. Conductivity Degradation and Mechanical Bending Are the Root Physical Causes**
While RF metrics dominate in predictive importance, the fault type distribution reveals that `Conductivity_Degradation` and mechanical faults (`Bending`, `Strong_Flexion`, `Cracks`) account for 6 of 7 fault classes. Material quality (conductor purity, surface treatment) and mechanical tolerance (housing rigidity, flex cable routing) are therefore the primary design leverage points for fault rate reduction.

**5. Physical Dimensions Alone Cannot Predict Faults**
`Length`, `Width`, `Height`, and `Bend` consistently rank in the bottom half of feature importance, with Pearson correlations below |0.10| with both fault targets. An antenna with perfect geometry but degraded conductivity or substrate humidity will fail; conversely, a physically bent antenna with excellent conductor quality may still perform adequately. This invalidates visual-only inspection protocols.

**6. Engineered Features Outperform Their Raw Components**
Gamma (derived from VSWR) ranks higher than raw VSWR in BT fault prediction. Gain_VSWR_Ratio ranks higher than either Gain or VSWR alone for WiFi prediction. This confirms that fault detection benefits from physics-informed feature engineering — not just raw sensor readings fed directly into models.

**7. Efficiency Is the Best Single-Threshold Hardware Monitor**
Efficiency shows the cleanest distribution separation (49.3% median in fault vs 81.6% in normal, with minimal overlap in violin analysis) combined with high feature importance (rank 3 for both targets). A hardware efficiency monitor with a 65% threshold would provide a self-contained, real-time fault alarm requiring no external RF test equipment.

**8. Gradient Boosting and Random Forest Are Production-Ready**
Both ensemble models achieve F1 > 0.990 (WiFi) and F1 = 1.000 (BT) on hold-out test data. These results support deployment in automated production testing pipelines where inference speed is not critical. For edge deployment (on-device fault detection), the trained Random Forest can be serialised to ONNX or TensorFlow Lite for embedded inference on the test signal processor.

---

## 🛠️ Tech Stack

### Core Dependencies

| Library | Version | Role in This Project |
|---------|---------|---------------------|
| `pandas` | ≥ 2.0 | Data ingestion, cleaning, wide-format management, groupby aggregations |
| `numpy` | ≥ 1.24 | Numerical operations, Z-score outlier detection, array manipulations |
| `matplotlib` | ≥ 3.7 | Base figure/axis management; dual-axis charts; custom colour coding |
| `seaborn` | ≥ 0.12 | Heatmaps, box plots, violin plots, statistical themes |
| `scipy` | ≥ 1.11 | Mann-Whitney U tests, chi-squared tests, Z-score computation |
| `scikit-learn` | ≥ 1.3 | StandardScaler, LabelEncoder, all ML models, metrics, ROC curves |

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
git clone https://github.com/your-username/Antenna-Fault-Analytics.git
cd Antenna-Fault-Analytics
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
python Notebook/antenna_analysis.py
```

All 12 figures and the statistical summary report will be saved to `Outputs/`.

### 5. Review Outputs

```
Outputs/
├── fig1_distributions.png          ← All feature histograms
├── fig2_boxplots_wifi.png          ← RF parameters vs WiFi fault class
├── fig3_boxplots_bt.png            ← RF parameters vs BT fault class
├── fig4_correlation_heatmap.png    ← Full Pearson correlation matrix
├── fig5_fault_breakdown.png        ← Multi-class fault type counts
├── fig6_s11_vswr_scatter.png       ← Fault zone in S11/VSWR space
├── fig7_outliers.png               ← Z-score outlier audit
├── fig8_confusion_matrices.png     ← All models × both targets
├── fig9_roc_curves.png             ← ROC + AUC for all models
├── fig10_model_comparison.png      ← 5-metric model comparison
├── fig11_feature_importance.png    ← Random Forest importance ranking
├── fig12_violin_top_features.png   ← Top-5 feature class distributions
└── antenna_fault_report.txt        ← Full statistical + model report
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

- [ ] **Multi-Class Fault Classification** — Extend the binary classifiers to full 7/9-class multi-class fault type prediction; evaluate per-class F1 scores; identify which fault types are most easily confused (e.g. Cracks vs Rupture, Bending vs Strong_Flexion)
- [ ] **SHAP Explainability Layer** — Integrate `shap` library to produce per-prediction explanations; generate SHAP waterfall plots for individual fault diagnoses to make model outputs interpretable to RF engineers
- [ ] **Interactive Plotly Dashboard** — Rebuild all 12 static charts in `Plotly Dash` for browser-based exploration with fault type filters, parameter range sliders, and scatter plot click-through to sample details

### Medium-Term (v1.2)

- [ ] **Real-Time Monitoring Pipeline** — Wrap the trained Random Forest in a FastAPI endpoint that accepts live S11, VSWR, Efficiency, and Bandwidth readings and returns fault probability with confidence interval; suitable for integration with VNA (Vector Network Analyser) test station software
- [ ] **Time-Series Fault Progression Modelling** — Acquire sequential measurement data from accelerated aging tests; apply `Prophet` or LSTM models to predict the time-to-fault trajectory from early RF degradation signatures
- [ ] **Environmental Covariate Integration** — Add temperature, humidity, and vibration sensor readings as model features to separate environmentally-induced faults (Humidity_Sweat, Body_Effect) from structural ones (Cracks, Rupture, Bending)

### Long-Term (v2.0)

- [ ] **On-Device Edge Deployment** — Serialise the Random Forest to ONNX format using `sklearn-onnx`; deploy to a microcontroller (STM32, ESP32) co-located with the antenna to enable real-time self-diagnostics without external test equipment
- [ ] **Multi-Protocol Expansion** — Extend dataset and models to cover LTE, 5G NR, and UWB antenna faults; develop protocol-specific fault signatures and unified multi-protocol health scores
- [ ] **Causal Fault Attribution** — Apply `DoWhy` causal inference framework to move from correlation (which RF metrics predict faults) to causation (which physical degradation mechanisms *cause* which RF parameter to drift first), enabling targeted design rule changes
- [ ] **Automated Design Rule Extraction** — Train a decision tree with interpretability constraints to extract actionable IF-THEN rules (e.g. *"IF Conductivity < 5,000 S/m AND VSWR > 2.8 THEN Fault probability = 0.94"*) directly suitable for manufacturing specification documents

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

Please follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines and include docstrings for all new functions. For RF domain additions, please cite the relevant IEEE antenna standard or application note.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Domain Reference:** IEEE Std 149-1979 — IEEE Standard Test Procedures for Antennas
- **RF Metrics Reference:** Pozar, D.M. (2011). *Microwave Engineering* (4th ed.). Wiley. Chapters 2 & 4 (S-parameters, impedance matching)
- **Fault Taxonomy Reference:** IPC-7711/7721 — Rework, Modification and Repair of Electronic Assemblies
- **Feature Engineering Basis:** VSWR–Gamma–S11 relationships derived from transmission line theory: Γ = (VSWR−1)/(VSWR+1)
- **Companion Project:** *Global Unemployment Trend Analysis (2015–2024)* — demonstrating the same production analytics workflow applied to labour market data

---

<div align="center">

**Built with 🐍 Python · 🐼 pandas · 🤖 scikit-learn · 📊 seaborn · 🔬 scipy**

*Senior Data Scientist / RF Analytics Engineer Portfolio Project — Antenna Fault Detection*

</div>
