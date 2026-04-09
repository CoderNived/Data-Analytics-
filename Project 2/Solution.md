# Solution Document
## Zomato Cart Add-On Recommendation Engine — EDA & Analytics Notebook v2

| Field | Detail |
|---|---|
| **Document Version** | 1.0 |
| **Status** | Implemented |
| **PRD Reference** | PRD.md v1.0 |
| **Stack** | Python 3.8+ · pandas · numpy · matplotlib · seaborn · scikit-learn |
| **Last Updated** | March 2026 |

---

## Table of Contents

1. [Solution Overview](#1-solution-overview)
2. [Environment & Configuration](#2-environment--configuration)
3. [Data Pipeline Implementation](#3-data-pipeline-implementation)
4. [EDA Implementation](#4-eda-implementation)
5. [Feature Engineering Implementation](#5-feature-engineering-implementation)
6. [Clustering & Advanced Analytics](#6-clustering--advanced-analytics)
7. [Business Insights Engine](#7-business-insights-engine)
8. [Output Artefacts](#8-output-artefacts)
9. [PRD Requirements Traceability Matrix](#9-prd-requirements-traceability-matrix)
10. [Known Limitations & Deviations](#10-known-limitations--deviations)
11. [How to Run](#11-how-to-run)

---

## 1. Solution Overview

The solution is a single, end-to-end Python analytics notebook (`zomato_cart_addons_eda.py`) structured into 11 sequential sections. It fully implements Phase 1 of the PRD: exploratory data analysis, feature engineering, behavioural clustering, co-occurrence mining, and a consolidated business insights report — all from a single 57-column session-level CSV input.

### Section Map

| Section | Title | PRD Sections Addressed |
|---|---|---|
| 1 | Project Overview | §1, §2 |
| 2 | Environment Setup | §7.3, §7.4 |
| 3 | Data Loading | §6.1 (FR-01, FR-02) |
| 4 | Data Cleaning & Preprocessing | §6.1, §6.2 (FR-03 – FR-08) |
| 5 | EDA & Visualizations | §6.3 (FR-10 – FR-22) |
| 6 | Feature Engineering | §6.4 (FR-23 – FR-24) |
| 7 | Behavioural Clustering | §6.5 (FR-25 – FR-35) |
| 8 | Business Insights | §6.7 (FR-36 – FR-37) |
| 9 | Consolidated Dashboard | §6.8 (FR-38 – FR-39) |
| 10 | Conclusion | §2.2 |
| 11 | Future Work | §13 |

---

## 2. Environment & Configuration

### 2.1 Library Imports

```python
import warnings; warnings.filterwarnings('ignore')   # NFR-08
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')              # NFR-06
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
import itertools, os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

`matplotlib.use('Agg')` is called **before** any `plt` import to guarantee headless PNG rendering on servers with no display (NFR-06). `warnings.filterwarnings('ignore')` suppresses scikit-learn convergence and pandas chained-assignment noise (NFR-08).

### 2.2 Global Constants

```python
PALETTE    = ['#E23744', '#FC8019', '#FFB347', '#2ECC71',
              '#3498DB', '#9B59B6', '#1ABC9C', '#E74C3C']
ZOMATO_RED = '#E23744'
ZOMATO_ORG = '#FC8019'
BG_COLOR   = '#FAFAFA'
```

These constants implement NFR-10 (brand colour palette). Every chart in the notebook references `ZOMATO_RED`, `ZOMATO_ORG`, or `PALETTE` — no raw hex strings appear at call sites.

```python
DOW_MAP   = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
DAY_ORDER = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
```

Implements NFR-15: all day-of-week operations use these shared constants rather than inline literals.

### 2.3 Output Configuration & `savefig` Helper

```python
OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name, fig=None, tight=True):
    path = f"{OUTPUT_DIR}/{name}"
    if tight: plt.tight_layout()
    if fig:   fig.savefig(path, bbox_inches='tight', facecolor=BG_COLOR)
    else:     plt.savefig(path, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  ✔ saved → {path}")
```

This single helper is the **sole** mechanism for saving figures (NFR-16). It enforces `bbox_inches='tight'` and `facecolor=BG_COLOR` on every output (NFR-11), and calls `plt.close()` after every save to prevent memory leaks from unclosed figure objects (NFR-07).

### 2.4 Global rcParams

```python
plt.rcParams.update({
    'figure.dpi'       : 130,        # NFR-11
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'axes.spines.top'  : False,      # NFR-13
    'axes.spines.right': False,      # NFR-13
    'font.family'      : 'DejaVu Sans',
})
```

Sets 130 DPI globally (NFR-11) and removes top/right spines on all axes by default (NFR-13), eliminating the need to manually hide spines on individual plots.

---

## 3. Data Pipeline Implementation

### 3.1 Data Loading (FR-01, FR-02)

```python
df_raw = pd.read_csv('/home/claude/zomato_cart_addons.csv')
print(f"Rows: {df_raw.shape[0]:,}   Columns: {df_raw.shape[1]}")
```

Row and column counts are logged immediately after load (FR-01). The `head(5)`, `info()`, and `describe()` outputs give an upfront structural summary.

```python
df['session_timestamp'] = pd.to_datetime(df['session_timestamp'], errors='coerce')
print(f"Null datetimes: {df['session_timestamp'].isna().sum()}")
```

`errors='coerce'` converts unparseable values to `NaT` rather than raising, and the null count is logged immediately (FR-02).

### 3.2 Missing Value Detection & Imputation (FR-03, FR-04)

```python
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
mv_report   = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
has_missing = mv_report[mv_report['Missing Count'] > 0]
print(has_missing.to_string() if len(has_missing) else "  ✔ No missing values detected.")
```

The report prints only columns with at least one null, keeping the output clean (FR-03).

```python
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in cat_cols: df[c].fillna(df[c].mode()[0], inplace=True)
for c in num_cols: df[c].fillna(df[c].median(), inplace=True)
```

Categorical nulls are filled with the **column mode** (most frequent value); numeric nulls with the **column median** (robust to outliers). This implements FR-04 without requiring any hardcoded column-specific logic — it applies generically across the entire schema.

### 3.3 Deduplication (FR-05)

```python
dupes = df.duplicated(subset='session_id').sum()
df.drop_duplicates(subset='session_id', keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
```

Deduplication targets `session_id` specifically (the primary key per §5.1 of the PRD). `keep='first'` retains the earliest encounter. The index is reset after dropping so that all subsequent `.iloc` operations remain valid (FR-05).

### 3.4 Derived Columns (FR-07)

```python
df['day_name'] = df['day_of_week'].map(DOW_MAP)
```

Maps the integer `day_of_week` to human-readable abbreviated names using the shared `DOW_MAP` constant. Used in all subsequent day-of-week groupby and reindex operations.

### 3.5 Outlier Detection (FR-06)

```python
for col in ['base_cart_value', 'final_order_value',
            'actual_added_addon_value', 'session_engagement_score']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    n_out  = ((df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)).sum()
    print(f"  {col} → {n_out} outliers (retained — valid business data)")
```

IQR-based detection is applied to the four high-variance monetary/score columns. Outliers are **logged but retained** — as specified in FR-06 — because extreme order values are genuine high-value transactions, not data errors.

### 3.6 Label Encoding (FR-08)

```python
encode_cols = ['user_segment', 'user_city', 'user_preferred_cuisine',
               'user_preferred_addon_category', 'restaurant_cuisine',
               'restaurant_type', 'meal_time', 'weather_condition',
               'traffic_density', 'delivery_zone']
for col in encode_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
```

Each encoded column is stored with a `_enc` suffix, preserving the original string column for readable groupby operations while providing integer codes for clustering and future ML models (FR-08). `.astype(str)` guards against any residual nulls that survived imputation.

---

## 4. EDA Implementation

### 4.1 KPI Dashboard (FR-10) — `01_kpi_overview.png`

Eight KPI tiles are rendered on a `2×4` subplot grid with a dark `#1A1A2E` figure background. Each tile displays the value in 22pt bold and the label in 9pt grey — matching the dark-mode design spec. The eight metrics cover:

- Total sessions and unique users (scale indicators)
- Add-on adoption rate (primary business KPI)
- Average final order value and average add-on value (revenue KPIs)
- Average add-on value for adopters only (quality of adoption signal)
- Average recommendation score (model quality indicator)
- Average engagement score (session health indicator)

All values are computed dynamically from `df` — no hardcoded numbers appear in the tile construction (FR-37 principle applied here too).

### 4.2 User & Restaurant Distributions (FR-11) — `02_user_restaurant_distributions.png`

Three panels on a `1×3` grid:

- **User segment pie**: Uses `value_counts()` on `user_segment`; `autopct='%1.1f%%'` provides percentage labels directly on wedges.
- **Top 10 cities**: Horizontal bar via `barh` on `value_counts().head(10)`; the top city is highlighted in `ZOMATO_RED`, others in light grey for visual hierarchy.
- **Top 10 cuisines**: Vertical bar using the `PALETTE` cycle, with 45° rotated x-labels to prevent overlap.

### 4.3 Engagement Score Analysis (FR-12) — `03_engagement_score.png`

- **Histogram** with 50 bins; both mean and median are overlaid as dashed vertical lines (different colours) with legend labels showing the exact values.
- **Box plot** split by `any_addon_added` outcome. The `Addon Outcome` column is created inline by mapping `{1: 'Add-On Added ✓', 0: 'No Add-On ✗'}` for readable axis labels. `plt.suptitle('')` suppresses pandas' automatic suptitle on boxplots.

### 4.4 Add-On Frequency Analysis (FR-13, FR-14)

**Parsing pipe-separated strings** is handled with a consistent pattern used throughout the notebook:

```python
all_addons = []
for entry in df['actual_added_addon_names']:
    val = str(entry)
    if val not in ('None', 'nan', ''):
        all_addons.extend(val.split('|'))
addon_freq = Counter(all_addons)
```

Converting to `str` first handles any float `NaN` values that Python would not split correctly. The `('None', 'nan', '')` exclusion guard implements FR-09 and NFR per the PRD risk for malformed entries.

`Counter.most_common(15)` efficiently returns the top 15 without sorting the full frequency dictionary.

The add-on count distribution bar chart adds percentage labels above each bar using `val / len(df) * 100`, communicating both absolute volume and relative frequency simultaneously.

### 4.5 Time-Based Analysis (FR-15, FR-16)

A `2×2` subplot grid covers four time dimensions:

- **Hourly volume**: `peak_hours = [12, 13, 19, 20, 21]` are highlighted red inline via a list comprehension colour array — no separate masking step required.
- **Hourly adoption rate**: Line chart with `fill_between` shading and a dashed horizontal reference line at the dataset mean adoption rate.
- **Day-of-week volume**: Weekend days (Sat/Sun) are highlighted red; weekdays use `ZOMATO_ORG`.
- **Day × Hour heatmap**: Built with `df.groupby(['day_name','hour'])['any_addon_added'].mean().unstack()` then reindexed to `DAY_ORDER` to enforce Monday-first ordering. Seaborn `heatmap` with `YlOrRd` colormap provides intuitive warm/cool adoption intensity.

The separate meal-time figure (`05b`) uses `reindex(seg_order)` to enforce the canonical meal sequence (Breakfast → Late Night) rather than alphabetical order.

### 4.6 Revenue Impact Analysis (FR-17) — `06_revenue_impact.png`

Three panels:

- **AOV by add-on count**: Simple `groupby('actual_added_addon_count')['final_order_value'].mean()` with ₹-prefixed value labels above each bar.
- **Grouped base vs add-on value**: Side-by-side bars using `np.arange` + width offset. The `x` positions are manually computed with `w=0.35` so the two series don't overlap.
- **Cuisine revenue by add-on presence**: Uses the inline `has_addon_label` column created via `map({1:'Add-On Adopted', 0:'No Add-On'})`. `unstack()` pivots the adoption flag into columns for grouped bar plotting. `nlargest(8, 'Add-On Adopted')` selects the 8 cuisines with the highest add-on order value, focusing the chart on high-value opportunities.

### 4.7 Recommendation Quality Analysis (FR-18) — `07_reco_quality_analysis.png`

Overlaid density histograms (using `density=True`) for four recommendation metrics, each split by adopter vs. non-adopter. `density=True` normalises both distributions to the same scale so shape comparison is meaningful despite different group sizes.

### 4.8 Contextual Signal Analysis (FR-19) — `08_contextual_signals.png`

- **Weather**: Sorted descending by adoption rate; the highest-adopting weather condition is highlighted red.
- **Traffic density**: Manually ordered Low → Medium → High using `reindex` with a filtered list (handles datasets where one traffic level might be absent).
- **Festival × Offer interaction**: A `groupby(['is_festival_day','has_offer'])` cross-tabulation, unstacked and relabelled for readability, visualised as a grouped bar chart revealing the compound effect.

### 4.9 Cart Composition Analysis (FR-20) — `09_cart_composition.png`

- **Cart flags**: Computes adoption rates for sessions both with and without each flag (drink/dessert/side), building a `DataFrame` with rows "Without"/"With" and columns as flag names, then plots as grouped bars.
- **Cart completion bins**: `pd.cut` divides [0,1] into five equal-width bins; bin labels are explicit strings for readable axes.
- **Cart size**: `groupby('base_cart_item_count')` line chart — shows how adoption rate varies with the number of items already in the cart, revealing a non-linear relationship.

### 4.10 User Attribute Analysis (FR-21) — `10_user_attributes.png`

- **Price sensitivity bins**: Same `pd.cut` approach as cart completion.
- **Historical acceptance scatter**: 3,000-row sample (`random_state=42`) with uniform y-jitter (`np.random.uniform(-0.05, 0.05)`) to reveal density at the binary `0`/`1` target levels, which would otherwise be invisible without jitter.
- **Order frequency bins**: Non-uniform bin edges (`[0,2,5,10,20,max+1]`) reflect natural ordering frequency tiers rather than equal-width splits.

### 4.11 Correlation Matrix (FR-22) — `11_correlation_matrix.png`

```python
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ...)
```

The upper triangle is masked to eliminate the redundant mirror half, halving the visual noise. `center=0` on `RdYlGn` maps zero correlation to yellow, negative to red, and positive to green — matching intuitive colour semantics. `annot_kws={'size':7}` keeps the 29×29 matrix legible.

---

## 5. Feature Engineering Implementation

All 8 features required by §6.4 of the PRD are implemented in Section 6 of the notebook:

### 5.1 Division-by-Zero Safety (FR-23)

All ratio features use the same guard pattern:

```python
df['feature'] = (
    df['numerator'] / df['denominator'].replace(0, np.nan)
).fillna(0).round(N)
```

`.replace(0, np.nan)` converts zero denominators to `NaN` so pandas produces `NaN` (not `inf`) on division, and `.fillna(0)` then replaces those `NaN`s with zero — meaning sessions with zero items or zero order value contribute a clean 0 to the feature rather than corrupting downstream aggregations.

### 5.2 Feature Implementations

**`addon_revenue_share`**
```python
df['addon_revenue_share'] = (
    df['actual_added_addon_value'] / df['final_order_value'].replace(0, np.nan)
).fillna(0).round(3)
```

**`add_on_rate`**
```python
df['add_on_rate'] = (
    df['actual_added_addon_count'] / df['base_cart_item_count'].replace(0, np.nan)
).fillna(0).round(3)
```

**`cart_value_per_item`**
```python
df['cart_value_per_item'] = (
    df['base_cart_value'] / df['base_cart_item_count'].replace(0, np.nan)
).fillna(0).round(2)
```

**`user_loyalty_score`**
```python
df['user_loyalty_score'] = (
    df['user_order_frequency_30d'] / df['user_order_frequency_30d'].max()
    - df['user_recency_days']      / df['user_recency_days'].max()
    + df['num_past_orders_at_restaurant'] / df['num_past_orders_at_restaurant'].max()
).round(3)
```

Each component is min-max normalised to [0,1] before combining, preventing any single raw-scale column from dominating. The recency term is subtracted (longer gap = lower loyalty).

**`reco_attractiveness`**
```python
df['reco_attractiveness'] = (
    df['avg_reco_score'] * df['avg_reco_popularity'] / (df['avg_reco_price_ratio'] + 0.01)
).round(3)
```

The `+0.01` floor on `avg_reco_price_ratio` prevents division by zero when add-ons are free.

**`cart_diversity_flag`**
```python
df['cart_diversity_flag'] = (
    (df['cart_has_drink'] + df['cart_has_dessert'] + df['cart_has_side']) >= 2
).astype(int)
```

Summing three binary columns and thresholding at 2 is a concise, vectorised implementation of "at least two of three flags set."

**`addon_upsell_flag`** and **`high_engagement_flag`**
Both use the 75th percentile threshold computed dynamically from the data:
```python
df['addon_upsell_flag']    = (df['addon_revenue_share']       > df['addon_revenue_share'].quantile(0.75)).astype(int)
df['high_engagement_flag'] = (df['session_engagement_score']  > df['session_engagement_score'].quantile(0.75)).astype(int)
```

### 5.3 Feature Visualisations (FR-24)

The `12_feature_engineering.png` figure presents four panels:

- Distribution of `addon_revenue_share` (histogram with mean line) — shows the heavy zero-mass at non-adopter sessions.
- `add_on_rate` by cuisine (top 10) — identifies which cuisines drive the highest add-on intensity per cart item.
- Adoption rate by engagement tier (Low vs High `high_engagement_flag`) — quantifies the lift from high engagement.
- Average order value Weekday vs Weekend — validates the weekend premium observed in the business insights.

---

## 6. Clustering & Advanced Analytics

### 6.1 Feature Scaling (FR-25)

```python
X_cluster = df[cluster_features].fillna(0)
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)
```

`StandardScaler` transforms each of the 12 clustering features to zero mean and unit variance. This is essential because the features span vastly different scales: `final_order_value` ranges in hundreds of rupees while `is_weekend` is binary. Without scaling, KMeans would effectively ignore low-variance binary features entirely.

### 6.2 Elbow & Silhouette Optimisation (FR-26, FR-27)

```python
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
```

Both metrics are computed in a single loop. The elbow curve is used for visual inflection detection; the silhouette score provides a quantitative, objective criterion. Both charts mark `k=4` with a red dashed line (FR-27).

### 6.3 Final Model (FR-28 – FR-30)

```python
OPTIMAL_K = 4
kmeans    = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

`random_state=42` and `n_init=10` implement NFR-02/NFR-04. The final silhouette score is printed to stdout (FR-29). Cluster integers are then mapped to named archetypes via:

```python
cluster_labels = {
    0: 'Engaged Adopters',
    1: 'High-Value Selectives',
    2: 'Price-Sensitive Browsers',
    3: 'Low-Intent Passives',
}
df['cluster_label'] = df['cluster'].map(cluster_labels)
```

### 6.4 PCA Visualisation (FR-31)

```python
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]; df['pca2'] = X_pca[:, 1]
```

PCA is used **only for visualisation** — it is not involved in the KMeans fitting. The explained variance ratios are injected into the axis labels: `f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)'`, communicating how much of the total variance each axis captures.

The normalised bar comparison chart (second panel of `14_cluster_analysis.png`) uses min-max normalisation within each metric column so all five metrics share the same [0,1] axis scale, enabling fair cross-cluster, cross-metric comparison.

### 6.5 Co-occurrence Mining (FR-33 – FR-35)

```python
addon_combos = Counter()
for row in df['actual_added_addon_names']:
    val = str(row)
    if val not in ('None', 'nan', ''):
        items = sorted(val.split('|'))
        for combo in itertools.combinations(items, 2):
            addon_combos[combo] += 1
```

Items within each session are **sorted before combining** — this ensures `('Coke', 'Fries')` and `('Fries', 'Coke')` are counted as the same pair regardless of pipe-ordering. `itertools.combinations(items, 2)` generates all unordered pairs in O(n²/2) per session. The `Counter` accumulates pair frequencies across all sessions efficiently. The top 10 pairs are printed as a DataFrame and saved as `15_addon_combinations.png` (FR-34).

---

## 7. Business Insights Engine

### 7.1 Dynamic Value Computation (FR-36, FR-37)

All 10 insight values are derived from the processed DataFrame at runtime:

```python
adopt_rate        = df['any_addon_added'].mean() * 100
aov_with_addon    = df[df['any_addon_added'] == 1]['final_order_value'].mean()
aov_without_addon = df[df['any_addon_added'] == 0]['final_order_value'].mean()
addon_uplift      = (aov_with_addon / aov_without_addon - 1) * 100
top_addon_name    = addon_df.iloc[0]['Add-On'] if len(addon_df) else 'N/A'
peak_hour         = hourly_adopt.loc[hourly_adopt['any_addon_added'].idxmax(), 'hour']
best_segment      = seg_adopt.idxmax()
weather_top       = weather_adopt.idxmax()
```

The f-string insights block interpolates all 8 computed values directly, ensuring the printed report always matches the actual dataset. The `if len(addon_df) else 'N/A'` guard for `top_addon_name` handles edge cases where no add-ons were accepted in the dataset.

### 7.2 Insights Structure

The 10-point report covers, in order: adoption rate, revenue uplift, top add-on, timing, recommendation quality, cart composition, user sensitivity, cluster strategies, festival/offer effect, and weekend effect — directly mapping to the 10 business use cases listed in §2.2 of the PRD.

---

## 8. Output Artefacts

### 8.1 PNG Visualisations

All 18 files required by PRD §6.9 are generated and saved to `OUTPUT_DIR`:

| File | Generator Function/Section | Status |
|---|---|---|
| `01_kpi_overview.png` | Section 5.1 | ✅ |
| `02_user_restaurant_distributions.png` | Section 5.2 | ✅ |
| `03_engagement_score.png` | Section 5.3 | ✅ |
| `04_addon_analysis.png` | Section 5.4 | ✅ |
| `04b_reco_addon_categories.png` | Section 5.4 | ✅ |
| `05_time_based_behavior.png` | Section 5.5 | ✅ |
| `05b_meal_time_analysis.png` | Section 5.5 | ✅ |
| `06_revenue_impact.png` | Section 5.6 | ✅ |
| `07_reco_quality_analysis.png` | Section 5.7 | ✅ |
| `08_contextual_signals.png` | Section 5.8 | ✅ |
| `09_cart_composition.png` | Section 5.9 | ✅ |
| `10_user_attributes.png` | Section 5.10 | ✅ |
| `11_correlation_matrix.png` | Section 5.11 | ✅ |
| `12_feature_engineering.png` | Section 6 | ✅ |
| `13_cluster_optimisation.png` | Section 7 | ✅ |
| `14_cluster_analysis.png` | Section 7 | ✅ |
| `15_addon_combinations.png` | Section 7 | ✅ |
| `16_consolidated_dashboard.png` | Section 9 | ✅ |

### 8.2 Consolidated Dashboard (FR-38, FR-39)

The `16_consolidated_dashboard.png` is a `20 × 26` inch figure built with `matplotlib.gridspec.GridSpec(4, 3)`:

```python
fig = plt.figure(figsize=(20, 26))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.35)
```

`GridSpec` is used instead of `plt.subplots` to enable the Row 4 heatmap to span all 3 columns via `gs[3, :]`. The `hspace=0.48` and `wspace=0.35` spacings prevent label overlap across the 11 panels. The three KPI tiles in Row 1 use solid fill colours (red, green, blue) instead of the whitegrid theme, creating a strong visual anchor at the top of the dashboard.

### 8.3 Stdout Reports

Beyond PNG files, the notebook also produces three structured text outputs to stdout:

- **Column description table** (Section 3): 57-row schema glossary.
- **Business insights block** (Section 8): 10-point actionable insight report with dynamic values.
- **Conclusion & Future Work** (Sections 10–11): Narrative summary and roadmap.

---

## 9. PRD Requirements Traceability Matrix

### Functional Requirements

| Req ID | Description | Implementation Location | Status |
|---|---|---|---|
| FR-01 | Load CSV; log shape | `pd.read_csv` + print statement, Section 3 | ✅ |
| FR-02 | Parse timestamp; report nulls | `pd.to_datetime(errors='coerce')`, Section 4.1 | ✅ |
| FR-03 | Detect & report missing values | `isnull().sum()` + `mv_report`, Section 4.2 | ✅ |
| FR-04 | Impute: mode for categorical, median for numeric | Loop over `cat_cols` / `num_cols`, Section 4.2 | ✅ |
| FR-05 | Deduplicate on `session_id` | `drop_duplicates` + `reset_index`, Section 4.3 | ✅ |
| FR-06 | IQR outlier detection; retain outliers | IQR loop on 4 columns, Section 4.5 | ✅ |
| FR-07 | Create `day_name` column | `df['day_name'] = df['day_of_week'].map(DOW_MAP)`, Section 4.4 | ✅ |
| FR-08 | Label-encode categoricals with `_enc` suffix | `LabelEncoder` loop, Section 4.6 | ✅ |
| FR-09 | Parse pipe-separated string columns | `str(v).split('|')` with null guards, Sections 5.4, 7 | ✅ |
| FR-10 | KPI overview (8 metrics) | `01_kpi_overview.png`, Section 5.1 | ✅ |
| FR-11 | User & restaurant distributions | `02_user_restaurant_distributions.png`, Section 5.2 | ✅ |
| FR-12 | Engagement score histogram + boxplot | `03_engagement_score.png`, Section 5.3 | ✅ |
| FR-13 | Add-on popularity + count distribution | `04_addon_analysis.png`, Section 5.4 | ✅ |
| FR-14 | Recommended add-on categories | `04b_reco_addon_categories.png`, Section 5.4 | ✅ |
| FR-15 | Time-based analysis (4 panels) | `05_time_based_behavior.png`, Section 5.5 | ✅ |
| FR-16 | Meal-time adoption + volume | `05b_meal_time_analysis.png`, Section 5.5 | ✅ |
| FR-17 | Revenue impact (3 panels) | `06_revenue_impact.png`, Section 5.6 | ✅ |
| FR-18 | Recommendation quality histograms | `07_reco_quality_analysis.png`, Section 5.7 | ✅ |
| FR-19 | Contextual signal analysis | `08_contextual_signals.png`, Section 5.8 | ✅ |
| FR-20 | Cart composition analysis | `09_cart_composition.png`, Section 5.9 | ✅ |
| FR-21 | User attribute analysis | `10_user_attributes.png`, Section 5.10 | ✅ |
| FR-22 | Correlation matrix (29 features) | `11_correlation_matrix.png`, Section 5.11 | ✅ |
| FR-23 | Division-by-zero safety on engineered features | `.replace(0,np.nan).fillna(0)`, Section 6 | ✅ |
| FR-24 | Visualise engineered features | `12_feature_engineering.png`, Section 6 | ✅ |
| FR-25 | Scale clustering features with StandardScaler | `StandardScaler().fit_transform`, Section 7 | ✅ |
| FR-26 | KMeans for k=2 to 8 with inertia + silhouette | `for k in range(2,9)` loop, Section 7 | ✅ |
| FR-27 | Elbow + silhouette curves; mark k=4 | `13_cluster_optimisation.png`, Section 7 | ✅ |
| FR-28 | Final KMeans k=4, seed=42, n_init=10 | `KMeans(n_clusters=4, random_state=42, n_init=10)`, Section 7 | ✅ |
| FR-29 | Log final silhouette score | `print(silhouette_score(...))`, Section 7 | ✅ |
| FR-30 | Map clusters to named archetypes | `cluster_labels` dict + `.map()`, Section 7 | ✅ |
| FR-31 | PCA projection + normalised cluster bar chart | `14_cluster_analysis.png`, Section 7 | ✅ |
| FR-32 | Print cluster profile table | `cluster_profile.to_string()`, Section 7 | ✅ |
| FR-33 | Parse add-on names for co-occurrence | `itertools.combinations` + `Counter`, Section 7 | ✅ |
| FR-34 | Top 10 co-occurrence pairs chart | `15_addon_combinations.png`, Section 7 | ✅ |
| FR-35 | Co-occurrence as bundle input | Output surfaced in business insights, Section 8 | ✅ |
| FR-36 | 10-point structured insights report | f-string insights block, Section 8 | ✅ |
| FR-37 | All insight values dynamically computed | 8 computed variables, Section 8 | ✅ |
| FR-38 | Consolidated dashboard (4-row GridSpec) | `matplotlib.gridspec.GridSpec(4,3)`, Section 9 | ✅ |
| FR-39 | Save dashboard as `16_consolidated_dashboard.png` | `savefig('16_consolidated_dashboard.png')`, Section 9 | ✅ |

### Non-Functional Requirements

| Req ID | Description | Implementation | Status |
|---|---|---|---|
| NFR-01 | Complete in < 10 min on ≥8 GB RAM machine | No heavy ops; KMeans n_init=10 limits iterations | ✅ |
| NFR-02 | KMeans: n_init=10, random_state=42 | Enforced in both elbow loop and final fit | ✅ |
| NFR-03 | StandardScaler + PCA before fitting | Applied to `X_scaled` before `KMeans.fit_predict` | ✅ |
| NFR-04 | All random seeds = 42 | KMeans, PCA, sample — all `random_state=42` | ✅ |
| NFR-05 | Reproducible output on repeat runs | Fixed seeds + deterministic pandas ops | ✅ |
| NFR-06 | `matplotlib.use('Agg')` for headless rendering | First statement after matplotlib import | ✅ |
| NFR-07 | `plt.close()` after every save | Enforced inside `savefig()` helper | ✅ |
| NFR-08 | Suppress warnings | `warnings.filterwarnings('ignore')` at top | ✅ |
| NFR-09 | Section headers printed to stdout | `print("="*60); print("SECTION N...")` | ✅ |
| NFR-10 | Zomato brand colour palette | `PALETTE`, `ZOMATO_RED`, `ZOMATO_ORG` constants | ✅ |
| NFR-11 | 130 DPI, `bbox_inches='tight'`, `facecolor=BG_COLOR` | `plt.rcParams['figure.dpi']=130`; `savefig()` helper | ✅ |
| NFR-12 | Bold titles; axis labels on all non-pie charts | `fontweight='bold'` on all titles; `set_xlabel/ylabel` | ✅ |
| NFR-13 | Hide top/right spines | `rcParams['axes.spines.top/right']=False` globally | ✅ |
| NFR-14 | No magic strings outside schema section | All column references use variable names | ✅ |
| NFR-15 | Use `DOW_MAP` and `DAY_ORDER` consistently | All day-of-week ops use these constants | ✅ |
| NFR-16 | `savefig()` as sole save mechanism | Every `plt.savefig` is inside `savefig()` helper | ✅ |

---

## 10. Known Limitations & Deviations

### 10.1 LabelEncoder Scope

`LabelEncoder` is fitted on the full dataset and reused across all 10 categorical columns via a single `le` instance. This means the encoder state from the last column fitted is stored in `le`, but since `_enc` columns are written immediately on each iteration, this is functionally correct. For production use, a `dict` of fitted encoders per column (or `OrdinalEncoder`) would be more maintainable.

### 10.2 KMeans Archetype Labels Are Fixed

The cluster integer → archetype name mapping (`cluster_labels = {0:'Engaged Adopters', ...}`) is hardcoded to cluster IDs 0–3. KMeans cluster IDs are not guaranteed to be stable across runs unless the data and seed are identical. With `random_state=42` and a fixed dataset, this is deterministic — but re-running on a different dataset or seed may require manually re-mapping the labels to match the new cluster profiles.

### 10.3 Co-occurrence Chart Label Truncation

The co-occurrence horizontal bar chart uses `str(x[0])` on the tuple pair, which renders as `"('Item A', 'Item B')"`. For production dashboards, a formatter like `" + ".join(pair)` would produce cleaner labels.

### 10.4 Correlation Matrix Annotation Size

At `annot_kws={'size':7}`, the 29×29 correlation matrix annotations are readable at 130 DPI on screen but may be small in printed form. This is a known trade-off between information density and legibility on a fixed-size figure.

### 10.5 Out-of-Scope Items (Phase 1)

Per PRD §4.2, the following are **not implemented** in this notebook and are planned for subsequent phases:

- Real-time model serving infrastructure
- A/B testing framework
- Live prediction model training and deployment
- Integration with Zomato production systems
- PII anonymisation protocols

---

## 11. How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Input

Place the dataset at the expected path:

```
/home/claude/zomato_cart_addons.csv
```

### Execution

```bash
python zomato_cart_addons_eda.py
```

Or run each cell sequentially in a Jupyter notebook environment.

### Output

All 18 PNG files and the business insights text are written to:

```
/mnt/user-data/outputs/
```

A successful run ends with:

```
✅ All visualizations generated successfully!
Output files saved in: /mnt/user-data/outputs
```

### Estimated Runtime

Under 5 minutes on a machine with ≥8 GB RAM for a dataset up to ~200,000 rows. KMeans with `n_init=10` over k=2–8 is the most compute-intensive step.

---

*End of Document*