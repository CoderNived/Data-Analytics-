# 🍕 Project 2 — Zomato Cart Add-On Sessions Analysis

> **Production-grade EDA exploring cart behavior, add-on conversion, and revenue drivers on a food delivery platform.**

---

## 📁 Repository Structure

```
Data-Analytics-
│
├── Project 2/                              ← Zomato Cart Add-On Sessions Analysis
│   ├── Dataset/                            ← Raw dataset (not tracked — see Dataset Info)
│   ├── Notebook/                           ← Jupyter notebooks with full EDA walkthrough
│   │   └── zomato_eda_notebook.py          ← Production Python EDA script
│   └── Outputs/                            ← Generated charts, plots, and reports
│       ├── 01_kpi_overview.png
│       ├── 02_user_behavior_distributions.png
│       ├── 03_session_duration.png
│       ├── 04_addon_analysis.png
│       ├── 05_time_based_behavior.png
│       ├── 06_revenue_impact.png
│       ├── 07_cart_abandonment.png
│       ├── 08_correlation_matrix.png
│       ├── 09_feature_engineering.png
│       ├── 10_cluster_optimisation.png
│       ├── 11_cluster_analysis.png
│       ├── 12_addon_combinations.png
│       └── 13_consolidated_dashboard.png
│
├── .gitignore                              ← Excludes large dataset files
├── requirements.txt                        ← Python dependencies
└── README.md                               ← You are here
```

---

## 📌 Project Overview

This project performs a **complete Exploratory Data Analysis (EDA)** on a food delivery platform's cart add-on session data — modelled after real-world Zomato user interaction logs.

Each record in the dataset represents a single user session and captures:
- What the user added to their cart and which add-ons they selected
- Whether the session converted to a placed order or was abandoned
- Financial signals: base order value, add-on value, discounts
- Contextual metadata: platform, city, cuisine type, hour of day, day of week

### 💼 Why This Analysis Matters

Add-ons are one of the highest-leverage levers in food delivery economics. They are low-friction, high-margin upsells that directly affect:

| Business Metric | Impact |
|---|---|
| Average Order Value (AOV) | Add-ons contribute a measurable uplift per session |
| Cart Conversion Rate | Sessions with add-ons show lower abandonment |
| Recommendation Systems | Co-occurrence patterns reveal bundling opportunities |
| Upselling Strategy | Timing and platform data guide intervention design |

---

## 📊 Dataset Info

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Zomato Cart Add-On Sessions Dataset](https://www.kaggle.com/datasets/abdullahsafwan333/zomato-cart-add-on-sessions-dataset) |
| **Rows** | 5,000 sessions (post-cleaning) |
| **Columns** | 18 raw + 8 engineered features |
| **Format** | CSV |
| **Domain** | Food Delivery / E-Commerce Behavior |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `session_id` | string | Unique identifier per browsing session |
| `user_id` | string | Anonymised user identifier |
| `platform` | categorical | Android / iOS / Web |
| `city` | categorical | City of the session |
| `cuisine_type` | categorical | Primary cuisine category |
| `session_duration_sec` | int | Session length in seconds |
| `items_in_cart` | int | Number of main items added |
| `addons_selected_count` | int | Number of add-ons selected |
| `addon_names` | string | Pipe-separated list of selected add-ons |
| `base_order_value` | float | Value of main items (₹) |
| `addon_value` | float | Revenue from add-ons (₹) |
| `total_order_value` | float | Final order value incl. delivery fee (₹) |
| `order_placed` | binary | 1 = order placed, 0 = abandoned |
| `cart_abandoned` | binary | 1 = cart abandoned, 0 = purchased |
| `discount_applied` | binary | 1 = coupon used |
| `discount_pct` | int | Discount percentage (0 if none) |
| `hour_of_day` | int | Hour (0–23) session started |
| `day_of_week` | categorical | Mon–Sun |

> ⚠️ **Dataset not tracked by Git.** Download from the Kaggle link above and place inside `Dataset/`.

---

## 🔬 Analysis Walkthrough

The notebook follows a structured, 11-section analytics workflow:

### 1. 📋 Project Overview
Business framing, use cases, and analytical objectives.

### 2. ⚙️ Environment Setup
Full library imports with aesthetic configuration for reproducible, publication-quality charts.

### 3. 📥 Data Loading
`head()`, `info()`, `describe()`, and a human-readable column dictionary.

### 4. 🧹 Data Cleaning & Preprocessing
- Missing value detection and mode/median imputation
- Duplicate session removal (30 found and dropped)
- Data type enforcement
- IQR-based outlier flagging (retained — valid business data)
- Label encoding for ML-ready categoricals

### 5. 📈 Exploratory Data Analysis

| Chart | What It Shows |
|---|---|
| `01_kpi_overview.png` | 8-card KPI summary dashboard |
| `02_user_behavior_distributions.png` | Platform share, city volume, cuisine breakdown |
| `03_session_duration.png` | Session length histogram + outcome box plot |
| `04_addon_analysis.png` | Top 15 add-ons by frequency + per-session distribution |
| `05_time_based_behavior.png` | Hourly volume, hourly conversion, day-of-week patterns, Day×Hour heatmap |
| `06_revenue_impact.png` | AOV by add-on count, base vs add-on value, cuisine-level comparison |
| `07_cart_abandonment.png` | Abandonment by platform, by add-on count, by discount status |
| `08_correlation_matrix.png` | Lower-triangle heatmap of all numeric features |

### 6. 🛠️ Feature Engineering

Eight new analytical features created from raw columns:

| Feature | Formula / Logic | Purpose |
|---|---|---|
| `session_duration_mins` | `duration_sec / 60` | Human-readable duration |
| `add_on_rate` | `addons / items_in_cart` | Add-on attachment density |
| `cart_value_per_item` | `base_value / items` | Spend per item signal |
| `addon_revenue_share` | `addon_value / total_value` | Add-on contribution to bill |
| `session_efficiency` | `1000 / (duration + 1)` if converted | Speed-to-conversion signal |
| `time_segment` | Bucketed by hour | Breakfast / Lunch / Snack / Dinner / Late Night |
| `is_weekend` | Sat or Sun → 1 | Weekend ordering flag |
| `addon_upsell_flag` | Top 25% `addon_revenue_share` | High-upsell session indicator |

### 7. 🤖 Advanced Analysis — Behavioral Clustering

K-Means clustering applied to 8 normalized session features:

- **Elbow curve** + **Silhouette scores** used to select optimal `k = 4`
- **PCA projection** (2D) visualizes cluster separation
- **Cluster profile comparison** (normalized bar chart) across key metrics

| Cluster | Label | Characteristics |
|---|---|---|
| 0 | Casual Browsers | Low AOV, low add-ons, high abandonment |
| 1 | High-Value Converters | High spend, strong conversion, few add-ons |
| 2 | Quick Add-On Buyers | Multiple add-ons, moderate order value |
| 3 | Discount Seekers | Price-sensitive, coupon-driven, moderate conversion |

- **Add-on co-occurrence analysis** identifies top 10 natural product bundles (e.g. Cold Drink + Garlic Bread, Gulab Jamun + Lassi)

### 8. 💡 Business Insights
8 structured, actionable findings — see [Key Findings](#-key-findings) below.

### 9. 📊 Dashboard Suggestions
Recommendations for operationalising insights in Power BI / Tableau / Streamlit.

### 10. ✅ Conclusion
Five core takeaways summarising the most impactful patterns.

### 11. 🚀 Future Work
Six next-step project directions including recommendation systems, predictive models, and A/B testing frameworks.

---

## 📸 Output Gallery

<table>
  <tr>
    <td><img src="Outputs/01_kpi_overview.png" width="380"/><br><sub>KPI Overview Dashboard</sub></td>
    <td><img src="Outputs/05_time_based_behavior.png" width="380"/><br><sub>Time-Based Ordering Behavior</sub></td>
  </tr>
  <tr>
    <td><img src="Outputs/06_revenue_impact.png" width="380"/><br><sub>Revenue Impact of Add-Ons</sub></td>
    <td><img src="Outputs/07_cart_abandonment.png" width="380"/><br><sub>Cart Abandonment Deep-Dive</sub></td>
  </tr>
  <tr>
    <td><img src="Outputs/11_cluster_analysis.png" width="380"/><br><sub>Behavioral Segmentation (K-Means)</sub></td>
    <td><img src="Outputs/13_consolidated_dashboard.png" width="380"/><br><sub>Consolidated Analytics Dashboard</sub></td>
  </tr>
</table>

---

## 🔑 Key Findings

1. **Add-ons drive a +9.6% AOV uplift** — sessions with at least one add-on average ₹1,130 vs ₹1,032 without.
2. **Add-ons reduce cart abandonment** — each additional add-on selected correlates with a measurable drop in abandonment rate, suggesting add-ons act as a commitment device.
3. **Dinner + Late Night windows are the highest-converting** — peak conversion occurs between 19:00–22:00; push notifications and in-app prompts during these hours would yield the highest ROI.
4. **Web platform has the highest abandonment rate** — Android and iOS users convert significantly better, pointing to a Web UX gap in the add-on presentation flow.
5. **Weekend sessions show elevated AOV and add-on attach rates** — a natural window for weekend-exclusive bundle deals.
6. **Top bundling opportunities identified via co-occurrence**: Cold Drink + Garlic Bread, Gulab Jamun + Lassi, and Chocolate Brownie + Naan emerge as natural pre-built combo packs.
7. **4 distinct behavioral archetypes** — each requiring a tailored upsell strategy rather than a one-size-fits-all approach.
8. **Discounts boost conversion but reduce net revenue** — discount + add-on combo targeting of price-sensitive sessions can partially offset margin compression.

---

## 🛠️ Tech Stack

```
Python 3.12
├── pandas          — data manipulation and cleaning
├── numpy           — numerical operations and feature engineering
├── matplotlib      — base charting engine
├── seaborn         — statistical visualizations and heatmaps
└── scikit-learn    — StandardScaler, KMeans, PCA, silhouette_score
```

---

## ⚡ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Data-Analytics-.git
cd "Data-Analytics-/Project 2"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download from Kaggle and place CSV in Dataset/

# 4. Run the analysis
python Notebook/zomato_eda_notebook.py

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
```

---

## 🗺️ Dashboard Recommendations

This analysis is ready to be operationalised into a live BI layer:

| Tool | Suggested Dashboard |
|---|---|
| **Streamlit** | Real-time session funnel with add-on toggle filters |
| **Power BI** | Add-on popularity heatmap + AOV trend by week |
| **Tableau** | City-level conversion funnel with cuisine drill-through |
| **Looker** | Cluster segmentation dashboard for CRM targeting |

---

## 🚀 Future Work

| Direction | Description |
|---|---|
| 🤝 Recommendation System | Train Association Rules / ALS Collaborative Filter on add-on co-occurrence |
| 🎯 Conversion Prediction | LightGBM / XGBoost model to score session conversion probability in real-time |
| 🧪 A/B Testing Framework | Test add-on presentation formats (banner vs inline vs modal) per platform |
| 🔄 Sequence Modelling | LSTM on session click streams to pre-load personalised add-ons |
| 💰 Price Elasticity | Analyse add-on uptake sensitivity to pricing across user segments |
| 📡 Live Pipeline | Airflow + BigQuery pipeline feeding a Streamlit dashboard |

---

## 👤 Author

**Senior Data Scientist & Analytics Engineer**
Portfolio project demonstrating production-grade EDA, feature engineering, and behavioral segmentation on real-world food delivery data.

---

## 📄 License

This project is licensed under the MIT License.

---

*Part of the [Data-Analytics- Portfolio](../README.md) — a collection of end-to-end analytics projects.*
