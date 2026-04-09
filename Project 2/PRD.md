# Product Requirements Document
## Zomato Cart Add-On Recommendation Engine

| Field | Detail |
|---|---|
| **Document Version** | 1.0 |
| **Status** | Draft |
| **Owner** | Senior Data Scientist |
| **Last Updated** | March 2026 |
| **Stakeholders** | Product, Data Science, Engineering, Growth, Marketing |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Success Metrics](#3-goals--success-metrics)
4. [Scope](#4-scope)
5. [Dataset & Schema](#5-dataset--schema)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [System Architecture](#8-system-architecture)
9. [Feature Engineering Specification](#9-feature-engineering-specification)
10. [Model & Analytics Requirements](#10-model--analytics-requirements)
11. [Business Rules & Logic](#11-business-rules--logic)
12. [Visualizations & Reporting](#12-visualizations--reporting)
13. [Future Work](#13-future-work)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [Glossary](#15-glossary)

---

## 1. Executive Summary

This document specifies the requirements for the **Zomato Cart Add-On Recommendation Engine** — an analytics and machine learning system designed to increase add-on item adoption rates during the cart checkout session on the Zomato platform.

The system ingests a 57-column session-level dataset capturing user signals, restaurant attributes, cart composition, contextual factors, and recommendation metadata. It delivers exploratory insights, behavioural clustering, recommendation quality analysis, and a roadmap of ML-powered personalisation features.

The primary goal is to grow the overall add-on adoption rate and quantify the resulting revenue uplift per session.

---

## 2. Problem Statement

### 2.1 Background

Zomato surfaces add-on item recommendations (e.g., drinks, sides, desserts) to users at the cart stage of the ordering flow. Despite a recommendation engine being in place, adoption rates and the downstream revenue contribution of add-ons remain below their potential.

### 2.2 Core Questions

- What user, restaurant, cart, and contextual signals most strongly predict add-on acceptance?
- How can the recommendation system rank and personalise add-on suggestions more effectively?
- Which user archetypes exist, and what upsell strategy is appropriate for each?
- How do contextual triggers (weather, meal time, festival day, active offer) modulate adoption?
- What is the measurable revenue impact of improving the add-on adoption rate?

### 2.3 Current Pain Points

- Recommendation relevance is not uniformly high across sessions; low `avg_reco_score` sessions rarely convert.
- Price-sensitive users are served the same add-on prompts as non-sensitive users, leading to poor conversion and negative UX.
- No systematic use of contextual signals (weather, traffic, festival) to time or adjust promotions.
- Cart composition (drink/dessert/side flags, completion score) is not yet used as a real-time trigger for "complete your meal" prompts.
- There is no operational dashboard giving growth and product teams live visibility into adoption metrics.

---

## 3. Goals & Success Metrics

### 3.1 Business Goals

| Goal | Description |
|---|---|
| **Increase Add-On Adoption Rate** | Lift the overall add-on adoption rate by a measurable percentage through improved recommendations |
| **Revenue Uplift** | Grow average final order value via accepted add-ons |
| **Personalisation at Scale** | Serve each user archetype a contextually appropriate recommendation strategy |
| **Operational Visibility** | Provide stakeholders with a live, self-serve analytics dashboard |

### 3.2 Key Performance Indicators (KPIs)

| KPI | Definition | Target Direction |
|---|---|---|
| Add-On Adoption Rate | `any_addon_added == 1` sessions / total sessions | ↑ Increase |
| Avg Add-On Revenue per Session | `actual_added_addon_value` mean across all sessions | ↑ Increase |
| Add-On Revenue Uplift | % difference in `final_order_value` between adopters and non-adopters | ↑ Increase |
| Avg Recommendation Score | Mean `avg_reco_score` across served recommendations | ↑ Increase |
| Cluster Conversion Rate | Adoption rate per behavioural segment | ↑ Increase for each |
| Price-Sensitive User Adoption | Adoption rate where `user_price_sensitivity > 0.6` | ↑ Increase (currently lagging) |

### 3.3 Analytical Success Criteria

- EDA fully covers all 57 input columns with statistical summaries and charts.
- At least 15 publication-quality visualisations produced and saved.
- K-Means clustering yields a silhouette score ≥ 0.20 with meaningful cluster separation.
- Feature engineering produces ≥ 8 derived signals with documented business rationale.
- All business insights are grounded in data and expressed with quantified values.

---

## 4. Scope

### 4.1 In Scope

- Exploratory Data Analysis (EDA) on the full 57-column, session-level dataset.
- Data cleaning, deduplication, and missing-value imputation.
- Feature engineering of derived signals.
- K-Means behavioural clustering with PCA projection.
- Add-on co-occurrence and bundling opportunity analysis.
- Correlation analysis across all numeric features.
- Contextual signal analysis (weather, traffic, festival, meal time, weekend).
- Business insights report with actionable recommendations.
- Consolidated analytics dashboard visualisation.

### 4.2 Out of Scope (for this phase)

- Real-time serving infrastructure (see Future Work §13).
- A/B testing framework implementation.
- Training and deployment of a live ML prediction model.
- Integration with Zomato production backend systems.
- PII handling and user data anonymisation protocols (assumed pre-processed upstream).

---

## 5. Dataset & Schema

### 5.1 Dataset Overview

| Property | Value |
|---|---|
| **Granularity** | One row = one recommendation session |
| **Total Columns** | 57 |
| **Key Identifier** | `session_id` (must be unique; duplicates dropped) |
| **Target Variable** | `any_addon_added` (binary: 1 = at least one add-on accepted) |
| **Source File** | `zomato_cart_addons.csv` |

### 5.2 Column Catalogue

#### Identifiers & Timestamp

| Column | Type | Description |
|---|---|---|
| `session_id` | String | Unique session identifier |
| `session_timestamp` | Datetime | Session start datetime (parsed to `pd.Timestamp`) |
| `user_id` | String | Anonymised user identifier |
| `restaurant_id` | String | Anonymised restaurant identifier |
| `restaurant_name` | String | Restaurant display name |

#### User Attributes

| Column | Type | Description |
|---|---|---|
| `user_segment` | Categorical | CRM tier (e.g., Gold, Silver) |
| `user_city` | Categorical | User's city |
| `user_preferred_cuisine` | Categorical | Historically preferred cuisine |
| `user_veg_preference` | Binary | 1 = prefers vegetarian |
| `user_price_sensitivity` | Float [0–1] | Higher = more price sensitive |
| `user_order_frequency_30d` | Integer | Orders placed in past 30 days |
| `user_avg_order_value` | Float (₹) | Historical average order value |
| `user_recency_days` | Integer | Days since last order |
| `num_past_orders_at_restaurant` | Integer | Prior visits to this restaurant |
| `user_addon_acceptance_rate` | Float [0–1] | Historical add-on acceptance rate |
| `user_preferred_addon_category` | Categorical | Most-accepted add-on category |

#### Restaurant Attributes

| Column | Type | Description |
|---|---|---|
| `restaurant_city` | Categorical | Restaurant's city |
| `restaurant_cuisine` | Categorical | Primary cuisine served |
| `restaurant_type` | Categorical | Casual / Fine Dining / QSR / etc. |
| `restaurant_online_order` | String | Yes / No |
| `restaurant_price_tier` | Integer [1–4] | 1 = cheapest, 4 = premium |
| `restaurant_rating` | Float [0–5] | Platform rating |
| `restaurant_is_chain` | Binary | 1 = chain, 0 = standalone |
| `restaurant_delivery_time_avg` | Integer (min) | Average historical delivery time |
| `restaurant_avg_orders_per_day` | Float | Historical daily order volume |

#### Contextual Signals

| Column | Type | Description |
|---|---|---|
| `hour` | Integer [0–23] | Hour of day |
| `day_of_week` | Integer [0–6] | 0 = Monday, 6 = Sunday |
| `meal_time` | Categorical | Breakfast / Lunch / Snack / Dinner / Late Night |
| `is_weekend` | Binary | 1 = Saturday or Sunday |
| `has_offer` | Binary | 1 = active discount on session |
| `weather_condition` | Categorical | Sunny / Rainy / Cloudy / etc. |
| `traffic_density` | Categorical | Low / Medium / High |
| `is_festival_day` | Binary | 1 = national or regional festival |
| `estimated_delivery_time` | Integer (min) | Displayed ETA |
| `delivery_zone` | Categorical | Delivery zone identifier |

#### Cart Composition

| Column | Type | Description |
|---|---|---|
| `session_engagement_score` | Float | Composite user engagement score for the session |
| `base_cart_item_names` | String (pipe-sep) | Main item names in cart |
| `base_cart_item_categories` | String (pipe-sep) | Item categories |
| `base_cart_item_count` | Integer | Number of main cart items |
| `base_cart_value` | Float (₹) | Cart value before add-ons |
| `cart_has_drink` | Binary | 1 = cart contains a drink |
| `cart_has_dessert` | Binary | 1 = cart contains a dessert |
| `cart_has_side` | Binary | 1 = cart contains a side |
| `cart_completion_score` | Float [0–1] | How "complete" the meal is |

#### Recommendation Metadata

| Column | Type | Description |
|---|---|---|
| `recommended_addon_names` | String (pipe-sep) | Recommended add-on names |
| `recommended_addon_categories` | String (pipe-sep) | Recommended add-on categories |
| `recommended_addon_prices` | String (pipe-sep) | Prices of recommended add-ons |
| `avg_reco_score` | Float | Average relevance score of recommendations |
| `avg_reco_price_ratio` | Float | Add-on price / base item price |
| `avg_reco_popularity` | Float | Popularity percentile of recommended add-ons |
| `avg_reco_is_complementary` | Float [0–1] | Fraction of recommendations that are complementary |

#### Outcomes

| Column | Type | Description |
|---|---|---|
| `actual_added_addon_names` | String (pipe-sep) | Add-on names actually accepted |
| `actual_added_addon_categories` | String (pipe-sep) | Categories of accepted add-ons |
| `actual_added_addon_count` | Integer | Number of add-ons accepted |
| `actual_added_addon_value` | Float (₹) | Revenue from accepted add-ons |
| `any_addon_added` | Binary | **Primary target variable** |
| `final_order_value` | Float (₹) | Total order value including add-ons |

---

## 6. Functional Requirements

### 6.1 Data Loading & Validation

- **FR-01** Load the raw CSV from the specified path; log row and column counts.
- **FR-02** Parse `session_timestamp` as a datetime column; report null count post-parse.
- **FR-03** Detect and report all columns with missing values and their percentages.
- **FR-04** Impute missing categorical values with column mode; numeric with column median.
- **FR-05** Drop duplicate `session_id` rows, retaining the first occurrence; log count removed.
- **FR-06** Apply IQR-based outlier detection to `base_cart_value`, `final_order_value`, `actual_added_addon_value`, and `session_engagement_score`; retain outliers (they represent valid high-value orders) but log counts.

### 6.2 Preprocessing

- **FR-07** Create `day_name` column by mapping `day_of_week` integer to abbreviated name (Mon–Sun).
- **FR-08** Apply `LabelEncoder` to all categorical columns used in ML features; store as `<column>_enc` suffix to preserve originals.
- **FR-09** Parse all pipe-separated string columns (`actual_added_addon_names`, `recommended_addon_categories`, etc.) when performing frequency analysis.

### 6.3 Exploratory Data Analysis

The EDA module must produce the following analytical outputs:

- **FR-10** KPI overview: total sessions, unique users, add-on adoption rate, average final order value, average add-on value (all sessions and adopters only), average recommendation score, average engagement score.
- **FR-11** User and restaurant distributions: sessions by user segment (pie chart), top 10 cities by session volume (horizontal bar), top 10 cuisines by session count (bar chart).
- **FR-12** Session engagement score: histogram with mean/median lines, box plot split by add-on outcome.
- **FR-13** Add-on popularity: top 15 most-accepted add-on names by frequency (horizontal bar), distribution of add-on count per session (bar chart with percentage labels).
- **FR-14** Recommended add-on categories: top 10 recommended categories by recommendation count.
- **FR-15** Time-based analysis: hourly session volume (peak hours highlighted), hourly adoption rate line chart with average reference line, session volume by day of week, day × hour adoption rate heatmap.
- **FR-16** Meal-time analysis: adoption rate and session volume by meal time segment (Breakfast, Lunch, Snack, Dinner, Late Night).
- **FR-17** Revenue impact: average final order value by add-on count, stacked base vs. add-on value by add-on count, order value by cuisine broken down by add-on presence.
- **FR-18** Recommendation quality: overlaid histograms of `avg_reco_score`, `avg_reco_price_ratio`, `avg_reco_popularity`, and `avg_reco_is_complementary` for adopters vs. non-adopters.
- **FR-19** Contextual signal analysis: adoption rate by weather condition, by traffic density, and by festival × offer interaction.
- **FR-20** Cart composition analysis: adoption rate by drink/dessert/side flag presence, by cart completion score bin, and by base cart item count.
- **FR-21** User attribute analysis: adoption by price sensitivity bin, scatter of historical acceptance rate vs. current session outcome, adoption by 30-day order frequency bin.
- **FR-22** Correlation matrix: lower-triangle heatmap across all 29 key numeric features with annotations.

### 6.4 Feature Engineering

The system must derive the following 8 features:

| Feature | Formula | Purpose |
|---|---|---|
| `addon_revenue_share` | `actual_added_addon_value / final_order_value` | Add-on's fraction of total revenue |
| `add_on_rate` | `actual_added_addon_count / base_cart_item_count` | Add-ons per cart item |
| `cart_value_per_item` | `base_cart_value / base_cart_item_count` | Average unit value of base cart |
| `user_loyalty_score` | Normalised composite of frequency, recency (inverse), past restaurant orders | User loyalty proxy |
| `reco_attractiveness` | `avg_reco_score × avg_reco_popularity / (avg_reco_price_ratio + 0.01)` | Combined reco appeal signal |
| `cart_diversity_flag` | 1 if ≥ 2 of `cart_has_drink`, `cart_has_dessert`, `cart_has_side` are set | Cart completeness flag |
| `addon_upsell_flag` | 1 if `addon_revenue_share` > 75th percentile | High add-on contribution session |
| `high_engagement_flag` | 1 if `session_engagement_score` > 75th percentile | Highly engaged user flag |

- **FR-23** All engineered features must handle division by zero via `.replace(0, np.nan).fillna(0)`.
- **FR-24** Engineered features must be visualised: distribution of `addon_revenue_share`, add-on rate by cuisine, adoption rate by engagement tier, and order value by weekday/weekend.

### 6.5 Behavioural Clustering

- **FR-25** Scale all 12 clustering input features using `StandardScaler`.
- **FR-26** Run KMeans for k = 2 through 8; record inertia and silhouette score for each k.
- **FR-27** Generate and save an elbow curve and silhouette score curve; mark the selected k = 4.
- **FR-28** Fit final KMeans model with k = 4, random_state = 42, n_init = 10; assign cluster labels to all rows.
- **FR-29** Log final silhouette score; target ≥ 0.15.
- **FR-30** Map cluster integers to named archetypes: Engaged Adopters, High-Value Selectives, Price-Sensitive Browsers, Low-Intent Passives.
- **FR-31** Run PCA (2 components) for visualisation; plot colour-coded scatter and normalised bar comparison across clusters.
- **FR-32** Print full cluster profile table (mean of all clustering features by cluster).

### 6.6 Add-On Co-occurrence Analysis

- **FR-33** Parse all `actual_added_addon_names` entries and count pair co-occurrences using `itertools.combinations`.
- **FR-34** Display and save a horizontal bar chart of the top 10 most co-occurring add-on pairs.
- **FR-35** Co-occurrence data to be used as input for bundle/combo pack recommendations.

### 6.7 Business Insights Report

- **FR-36** Print a structured 10-point insights summary including: adoption rate, revenue uplift, top add-on name, peak adoption hour, best meal segment, weather condition with highest adoption, cluster strategy recommendations, festival × offer effect, and weekend effect.
- **FR-37** All insight values must be dynamically computed from the dataset (no hardcoded values).

### 6.8 Consolidated Dashboard

- **FR-38** Produce a single 20 × 26 inch multi-panel dashboard figure containing: 3 KPI tiles, user segment pie, hourly volume bar, top accepted add-ons bar, avg order value by add-on count, hourly adoption rate line, cluster size bar, and day × hour adoption heatmap.
- **FR-39** Save dashboard as `16_consolidated_dashboard.png`.

### 6.9 Output Files

All visualisations must be saved as PNG files to the configured `OUTPUT_DIR`:

| File | Section |
|---|---|
| `01_kpi_overview.png` | KPI Dashboard |
| `02_user_restaurant_distributions.png` | User & Restaurant |
| `03_engagement_score.png` | Engagement Score |
| `04_addon_analysis.png` | Add-On Popularity |
| `04b_reco_addon_categories.png` | Recommended Categories |
| `05_time_based_behavior.png` | Time Analysis |
| `05b_meal_time_analysis.png` | Meal Time |
| `06_revenue_impact.png` | Revenue Impact |
| `07_reco_quality_analysis.png` | Recommendation Quality |
| `08_contextual_signals.png` | Contextual Signals |
| `09_cart_composition.png` | Cart Composition |
| `10_user_attributes.png` | User Attributes |
| `11_correlation_matrix.png` | Correlation Matrix |
| `12_feature_engineering.png` | Feature Engineering |
| `13_cluster_optimisation.png` | Cluster Optimisation |
| `14_cluster_analysis.png` | Cluster Profiles |
| `15_addon_combinations.png` | Co-occurrence Pairs |
| `16_consolidated_dashboard.png` | Final Dashboard |

---

## 7. Non-Functional Requirements

### 7.1 Performance

- **NFR-01** The full notebook must complete execution in under 10 minutes on a standard data science workstation (≥8 GB RAM, no GPU required).
- **NFR-02** KMeans fitting must use `n_init=10` and `random_state=42` for reproducibility.
- **NFR-03** PCA and StandardScaler must be applied before cluster fitting to ensure stable convergence.

### 7.2 Reproducibility

- **NFR-04** All random seeds must be set to 42.
- **NFR-05** The notebook must produce identical outputs on consecutive runs against the same input CSV.

### 7.3 Code Quality

- **NFR-06** All chart rendering must use `matplotlib.use('Agg')` to support headless server execution.
- **NFR-07** All figures must be closed after saving (`plt.close()`) to prevent memory leaks.
- **NFR-08** Warnings must be suppressed via `warnings.filterwarnings('ignore')` for clean output.
- **NFR-09** Section headers must be printed to stdout at the start of each major section.

### 7.4 Visualisation Standards

- **NFR-10** All charts must use the defined `PALETTE` with Zomato brand colours (`ZOMATO_RED = #E23744`, `ZOMATO_ORG = #FC8019`).
- **NFR-11** All figures must be saved at 130 DPI with `bbox_inches='tight'` and `facecolor=BG_COLOR`.
- **NFR-12** Chart titles must be bold; axis labels must be present on all non-pie charts.
- **NFR-13** Spines (top and right) must be removed from all standard axes plots.

### 7.5 Maintainability

- **NFR-14** All column names must be referenced via the defined schema; no magic strings outside the schema section.
- **NFR-15** The `DOW_MAP` and `DAY_ORDER` constants must be used consistently for all day-of-week operations.
- **NFR-16** The `savefig()` helper function must be the sole mechanism for saving figures.

---

## 8. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                │
│   zomato_cart_addons.csv  (57 columns, session-level)          │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                  DATA PIPELINE (Sections 3–4)                  │
│  Load → Parse Timestamps → Impute → Deduplicate → Encode       │
└────────────────────────┬───────────────────────────────────────┘
                         │
           ┌─────────────┴──────────────┐
           ▼                            ▼
┌──────────────────┐         ┌──────────────────────────┐
│  EDA MODULE      │         │  FEATURE ENGINEERING     │
│  (Section 5)     │         │  (Section 6)             │
│  15+ charts      │         │  8 derived features      │
└──────────────────┘         └──────────────┬───────────┘
                                            │
                                            ▼
                             ┌──────────────────────────┐
                             │  ANALYTICS MODULE        │
                             │  (Section 7)             │
                             │  K-Means Clustering      │
                             │  PCA Projection          │
                             │  Co-occurrence Analysis  │
                             └──────────────┬───────────┘
                                            │
                                            ▼
                             ┌──────────────────────────┐
                             │  OUTPUT LAYER            │
                             │  18 PNG visualisations   │
                             │  Business Insights Print │
                             │  Consolidated Dashboard  │
                             └──────────────────────────┘
```

### Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Data Manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn (KMeans, PCA, StandardScaler, LabelEncoder) |
| Utility | collections.Counter, itertools |

---

## 9. Feature Engineering Specification

### 9.1 Feature Definitions (Detailed)

**`addon_revenue_share`**
Add-on value as a proportion of the final order. A value of 0 means no add-ons were taken; values approaching 1 indicate most of the order came from add-ons. Used to identify high-upsell sessions.

**`add_on_rate`**
The ratio of accepted add-ons to base cart items. Normalises add-on count for cart size, enabling fair comparison across sessions with different cart sizes.

**`user_loyalty_score`**
A composite normalised score combining frequency (positive contribution), recency (negative contribution — longer gap = lower score), and past orders at this restaurant (positive contribution). Range approximately −1 to +1.

**`reco_attractiveness`**
Combines recommendation relevance (`avg_reco_score`), popularity (`avg_reco_popularity`), and affordability (inverse of `avg_reco_price_ratio`). Higher values indicate a more attractive recommendation set. The `+0.01` floor prevents division by zero.

**`cart_diversity_flag`**
Binary indicator that the cart already contains at least two distinct meal component types. Research shows diverse carts have higher add-on acceptance; this flag can be used as a real-time trigger for "complete your meal" prompts.

**`high_engagement_flag`**
Flags sessions in the top quartile of engagement score. High-engagement users are prime candidates for premium or exclusive add-on offers.

### 9.2 Encoding Strategy

All categorical features used in clustering or ML must be label-encoded. Original categorical columns are preserved alongside their `_enc` suffixed counterparts to allow both interpretability and numeric operations.

---

## 10. Model & Analytics Requirements

### 10.1 Clustering Model

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | KMeans | Efficient, interpretable, scalable |
| k (clusters) | 4 | Elbow + silhouette optimisation |
| Scaler | StandardScaler | Required; features have different units |
| n_init | 10 | Guards against poor centroid initialisation |
| random_state | 42 | Reproducibility |
| Dimensionality reduction | PCA (2 components) | Visualisation only; not used in fitting |

### 10.2 Clustering Feature Set

The following 12 features are used as inputs:

- `session_engagement_score`
- `base_cart_item_count`
- `actual_added_addon_count`
- `final_order_value`
- `addon_revenue_share`
- `add_on_rate`
- `any_addon_added`
- `is_weekend`
- `user_price_sensitivity`
- `user_addon_acceptance_rate`
- `avg_reco_score`
- `cart_completion_score`

### 10.3 Cluster Archetypes

| Cluster ID | Label | Characteristics | Recommended Strategy |
|---|---|---|---|
| 0 | Engaged Adopters | High engagement, high adoption rate, above-average cart value | Reinforce with premium or exclusive add-ons; loyalty rewards |
| 1 | High-Value Selectives | High order value, selective add-on acceptance, low price sensitivity | Quality-focused bundles; curated premium packs |
| 2 | Price-Sensitive Browsers | High price sensitivity, lower adoption rate, smaller carts | Discounted add-ons; value-bundle framing; "great deal" badges |
| 3 | Low-Intent Passives | Low engagement, near-zero adoption, minimal cart diversity | Re-engagement nudges; social proof ("Most popular with this order"); minimal intrusive prompts |

### 10.4 Co-occurrence Analysis

Add-on pairs are extracted from `actual_added_addon_names` by splitting on `|`, sorting the items alphabetically, and counting all 2-combinations per session using `itertools.combinations`. The top 10 pairs are surfaced as bundle candidates. This provides a data-driven foundation for "Combo Add-On Packs" product features.

---

## 11. Business Rules & Logic

### 11.1 Contextual Promotion Triggers

| Condition | Recommended Action |
|---|---|
| `meal_time == 'Dinner'` or `'Late Night'` | Activate premium add-on recommendation cards |
| `weather_condition == 'Rainy'` | Prioritise hot beverage and comfort food add-ons |
| `is_festival_day == 1` AND `has_offer == 1` | Deploy curated festival add-on packs with offer overlay |
| `is_weekend == 1` | Launch weekend-exclusive add-on deals |
| `cart_diversity_flag == 0` | Trigger "Complete your meal" prompt inline in cart |
| `high_engagement_flag == 1` | Show premium/exclusive add-on options |

### 11.2 User Sensitivity Routing

- Users with `user_price_sensitivity > 0.6` must be served **discounted or value-bundled** add-on recommendations rather than standard upsell prompts.
- The add-on price ratio (`avg_reco_price_ratio`) for price-sensitive users should be capped; add-ons priced above a threshold relative to the base item should be deprioritised in the ranking.

### 11.3 Recommendation Quality Floor

- Sessions where `avg_reco_score` falls below the dataset median should not surface add-on recommendation cards until the relevance model is improved. Low-relevance recommendations have been shown to produce lower adoption and may degrade user trust.

### 11.4 Timing Rules

- Add-on recommendation cards should be surfaced within the first 10 seconds of the cart view to maximise visibility before purchase intent resolves.
- Push notifications and in-app banners should be scheduled during the identified peak adoption hour and the best-converting meal time segment, as computed dynamically from the dataset.

---

## 12. Visualizations & Reporting

### 12.1 Chart Standards

All charts must adhere to the following:

- **Colour palette**: `['#E23744', '#FC8019', '#FFB347', '#2ECC71', '#3498DB', '#9B59B6', '#1ABC9C', '#E74C3C']`
- **Background**: `#FAFAFA` (light mode); KPI tiles use `#1A1A2E` (dark) with white text
- **DPI**: 130
- **Top/right spines**: Always hidden
- **Bar charts**: White edge colour on all bars
- **Value labels**: Displayed on all bar/horizontal bar charts (₹ prefix for monetary values, % suffix for rate values)

### 12.2 Dashboard Layout

The consolidated dashboard (Section 9) uses a 4-row GridSpec layout:

- **Row 1** (3 panels): Total Sessions KPI, Add-On Adoption Rate KPI, Revenue Uplift KPI
- **Row 2** (3 panels): User segment pie, Hourly session bar, Top add-ons horizontal bar
- **Row 3** (3 panels): AOV by add-on count, Hourly adoption line, Cluster size horizontal bar
- **Row 4** (full width): Day × Hour adoption heatmap

---

## 13. Future Work

The following capabilities are planned for subsequent product phases:

### Phase 2 — Prediction Model

Train a gradient boosting classifier (XGBoost or LightGBM) using all 57 raw features plus the 8 engineered features to predict `any_addon_added` in real-time. The model output probability score will gate whether to show an add-on recommendation card, with a configurable threshold per user segment.

### Phase 3 — Personalised Recommendation Ranker

Implement a LambdaMART or Two-Tower neural network to rank add-on candidates per session, utilising user history and real-time cart state alongside recommendation metadata features.

### Phase 4 — A/B Testing Framework

Design and execute controlled experiments testing: add-on card placement (inline vs. modal vs. banner), copy variations by meal time and user segment, and price framing strategies for price-sensitive users.

### Phase 5 — Price Elasticity Modelling

Analyse `avg_reco_price_ratio` sensitivity across `user_price_sensitivity` deciles to determine optimal add-on price points per cluster, enabling dynamic pricing of recommendations.

### Phase 6 — Contextual Bandit / Reinforcement Learning

Deploy a contextual bandit (LinUCB or Thompson Sampling) for real-time add-on selection, incorporating weather, meal time, traffic density, and live cart state as context features.

### Phase 7 — Live Operational Dashboard

Build a production dashboard (Streamlit or Power BI) fed by an Airflow + BigQuery pipeline, giving growth and product teams real-time visibility into adoption KPIs, cluster distributions, and recommendation quality metrics.

---

## 14. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dataset schema changes (column additions/removals) | Medium | High | Validate column presence at load time; fail fast with descriptive error messages |
| Clustering instability across runs | Low | Medium | Fixed `random_state=42` and `n_init=10`; document expected silhouette score range |
| High missing-value rate in key features | Medium | High | Monitor missing % report at Section 4; escalate if any key feature exceeds 10% missing |
| Low silhouette score (< 0.15) | Low | Medium | Revisit feature selection; consider DBSCAN or hierarchical clustering as fallbacks |
| Overlapping cluster profiles | Medium | Medium | Review PCA scatter for separation; adjust k or feature set |
| Pipe-separated columns containing malformed entries | Low | Low | Null/`'nan'`/`''` guards applied at all parse sites |
| Visualisation rendering failures in headless environment | Low | Medium | `matplotlib.use('Agg')` enforced at import time |

---

## 15. Glossary

| Term | Definition |
|---|---|
| **Add-On** | A supplementary item (drink, side, dessert) recommended to a user on top of their base cart |
| **Adoption Rate** | Fraction of sessions where at least one add-on was accepted (`any_addon_added == 1`) |
| **Base Cart** | The set of items the user added before any add-on recommendation was made |
| **Cart Completion Score** | A platform-computed [0–1] score indicating how nutritionally or meal-wise complete the current cart is |
| **Cluster / Archetype** | A group of sessions (and associated users) sharing similar behavioural characteristics, derived via K-Means |
| **Co-occurrence** | Two add-on items being accepted together in the same session |
| **Contextual Signal** | Any session-level feature reflecting external conditions: weather, time, traffic, festival status |
| **EDA** | Exploratory Data Analysis — statistical and visual examination of raw data |
| **Engagement Score** | Composite score reflecting how actively the user interacted with the app during the session |
| **Festival Day** | A national or regional public holiday or celebration |
| **KPI** | Key Performance Indicator |
| **LabelEncoder** | Scikit-learn transformer converting categorical strings to integer codes |
| **Meal Time** | Named time-of-day segment: Breakfast, Lunch, Snack, Dinner, or Late Night |
| **PCA** | Principal Component Analysis — dimensionality reduction for visualisation |
| **Price Sensitivity** | A [0–1] score where higher values indicate greater reluctance to spend on extras |
| **Reco Score** | A platform-generated relevance score for a recommended add-on item |
| **Revenue Uplift** | Percentage increase in `final_order_value` attributable to accepted add-ons |
| **Session** | A single user visit to the cart page where add-on recommendations are served |
| **Silhouette Score** | A clustering quality metric ranging from −1 to +1; higher is better |
| **User Segment** | CRM-assigned tier (e.g., Gold, Silver) reflecting a user's value or behaviour |

---

*End of Document*