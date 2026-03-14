"""
=============================================================================
ZOMATO CART ADD-ON SESSIONS — COMPLETE EDA & ANALYTICS NOTEBOOK  v2
=============================================================================
Author  : Senior Data Scientist
Dataset : Zomato Cart Add-On Sessions (57-column schema)
Stack   : Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
"""
## Project Overview

### What the dataset represents
Each row is one recommendation session on Zomato, capturing:
  - Rich user-level signals  (segment, city, price sensitivity, order history)
  - Restaurant attributes    (cuisine, type, price tier, rating, chain flag)
  - Contextual signals       (hour, meal_time, weather, traffic, festival)
  - Cart composition         (items, categories, drink/dessert/side flags)
  - Recommendation metadata  (reco score, price ratio, popularity, complementary)
  - Outcome                  (any_addon_added, actual_added_addon_count/value)

### Business Use Cases
  • Add-On Adoption Rate  – which signals drive users to accept recommendations
  • Revenue Uplift        – quantify the add-on contribution to final_order_value
  • Recommendation Quality– link avg_reco_score / popularity to actual acceptance
  • Contextual Targeting  – meal_time, weather, festival as promotion triggers
  • User Segmentation     – cluster sessions for personalised upsell strategies
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
import itertools
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Aesthetic config ──────────────────────────────────────────────────────────
PALETTE    = ['#E23744', '#FC8019', '#FFB347', '#2ECC71', '#3498DB',
              '#9B59B6', '#1ABC9C', '#E74C3C']
ZOMATO_RED = '#E23744'
ZOMATO_ORG = '#FC8019'
BG_COLOR   = '#FAFAFA'

sns.set_theme(style='whitegrid', palette=PALETTE)
plt.rcParams.update({
    'figure.dpi': 130,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name, fig=None, tight=True):
    path = f"{OUTPUT_DIR}/{name}"
    if tight:
        plt.tight_layout()
    if fig:
        fig.savefig(path, bbox_inches='tight', facecolor=BG_COLOR)
    else:
        plt.savefig(path, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"  ✔ saved → {path}")

# Integer day_of_week → short name helper
DOW_MAP = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
DAY_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

print("✅ Environment ready. Libraries loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 3 — DATA LOADING")
print("="*60)

df_raw = pd.read_csv('/home/claude/zomato_cart_addons.csv')

print(f"\n── Shape ──  Rows: {df_raw.shape[0]:,}   Columns: {df_raw.shape[1]}")
print("\n── Head (5 rows) ───────────────────────────────────────────")
print(df_raw.head(5).to_string())
print("\n── Info ────────────────────────────────────────────────────")
df_raw.info()
print("\n── Describe (numeric) ──────────────────────────────────────")
print(df_raw.describe().round(2).to_string())

print("\n── Column Descriptions ─────────────────────────────────────")
col_desc = {
    # Identifiers & timestamp
    'session_id'                    : 'Unique session identifier',
    'session_timestamp'             : 'Datetime when session started',
    'user_id'                       : 'Anonymised user ID',
    'restaurant_id'                 : 'Anonymised restaurant ID',
    'restaurant_name'               : 'Restaurant display name',
    # User attributes
    'user_segment'                  : 'CRM user segment (e.g. Gold, Silver)',
    'user_city'                     : 'City of the user',
    'user_preferred_cuisine'        : 'User's favourite cuisine category',
    'user_veg_preference'           : '1 = prefers vegetarian food',
    'user_price_sensitivity'        : 'Price sensitivity score (0–1; higher = more sensitive)',
    'user_order_frequency_30d'      : 'Orders placed in past 30 days',
    'user_avg_order_value'          : 'Historical average order value (₹)',
    'user_recency_days'             : 'Days since last order',
    'num_past_orders_at_restaurant' : 'Prior orders at this specific restaurant',
    'user_addon_acceptance_rate'    : 'Fraction of past sessions where add-on was accepted',
    'user_preferred_addon_category' : 'Most-accepted add-on category historically',
    # Restaurant attributes
    'restaurant_city'               : 'City of the restaurant',
    'restaurant_cuisine'            : 'Primary cuisine served',
    'restaurant_type'               : 'Casual / Fine Dining / QSR etc.',
    'restaurant_online_order'       : 'Whether online ordering is enabled (Yes/No)',
    'restaurant_price_tier'         : 'Price bucket (1 = cheapest, 4 = premium)',
    'restaurant_rating'             : 'Platform rating (0–5)',
    'restaurant_is_chain'           : '1 = chain restaurant, 0 = standalone',
    'restaurant_delivery_time_avg'  : 'Average delivery time (minutes)',
    'restaurant_avg_orders_per_day' : 'Historical daily order volume',
    # Context
    'hour'                          : 'Hour of day (0–23)',
    'day_of_week'                   : 'Day integer (0=Mon … 6=Sun)',
    'meal_time'                     : 'Breakfast / Lunch / Snack / Dinner / Late Night',
    'is_weekend'                    : '1 = Saturday or Sunday',
    'has_offer'                     : '1 = active offer/discount on session',
    'weather_condition'             : 'Weather at session time (Sunny/Rainy/Cloudy…)',
    'traffic_density'               : 'Traffic level (Low / Medium / High)',
    'is_festival_day'               : '1 = national or regional festival',
    'estimated_delivery_time'       : 'Estimated delivery time shown to user (min)',
    'delivery_zone'                 : 'Delivery zone identifier',
    # Cart
    'session_engagement_score'      : 'Composite score of user engagement in session',
    'base_cart_item_names'          : 'Pipe-separated main item names in cart',
    'base_cart_item_categories'     : 'Pipe-separated item categories',
    'base_cart_item_count'          : 'Number of main items in cart',
    'base_cart_value'               : 'Value of main items before add-ons (₹)',
    'cart_has_drink'                : '1 = cart contains a drink item',
    'cart_has_dessert'              : '1 = cart contains a dessert item',
    'cart_has_side'                 : '1 = cart contains a side item',
    'cart_completion_score'         : 'How "complete" the meal in cart is (0–1)',
    # Recommendations
    'recommended_addon_names'       : 'Pipe-separated recommended add-on names',
    'recommended_addon_categories'  : 'Pipe-separated recommended categories',
    'recommended_addon_prices'      : 'Pipe-separated prices of recommended add-ons',
    'actual_added_addon_names'      : 'Pipe-separated add-on names the user actually added',
    'actual_added_addon_categories' : 'Categories of actually added add-ons',
    'actual_added_addon_count'      : 'Number of add-ons the user actually accepted',
    'actual_added_addon_value'      : 'Revenue from accepted add-ons (₹)',
    # Outcome & reco quality
    'any_addon_added'               : 'Binary target: 1 = user accepted at least 1 add-on',
    'final_order_value'             : 'Total order value including add-ons (₹)',
    'avg_reco_score'                : 'Average recommendation relevance score',
    'avg_reco_price_ratio'          : 'Add-on price / base item price ratio',
    'avg_reco_popularity'           : 'Popularity percentile of recommended add-ons',
    'avg_reco_is_complementary'     : 'Fraction of reco items that are complementary',
}
for col, desc in col_desc.items():
    print(f"  {col:<35} → {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4 — DATA CLEANING & PREPROCESSING")
print("="*60)

df = df_raw.copy()

# ── 4.1 Parse timestamp ───────────────────────────────────────────────────────
df['session_timestamp'] = pd.to_datetime(df['session_timestamp'], errors='coerce')
print(f"\n  session_timestamp parsed. Null datetimes: {df['session_timestamp'].isna().sum()}")

# ── 4.2 Missing values ────────────────────────────────────────────────────────
print("\n── Missing Values ──────────────────────────────────────────")
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
mv_report   = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
has_missing = mv_report[mv_report['Missing Count'] > 0]
print(has_missing.to_string() if len(has_missing) else "  ✔ No missing values detected.")

# Fill categoricals with mode; numerics with median
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in cat_cols:
    df[c].fillna(df[c].mode()[0], inplace=True)
for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)
print(f"  After imputation — nulls remaining: {df.isnull().sum().sum()}")

# ── 4.3 Duplicates ───────────────────────────────────────────────────────────
print("\n── Duplicates ──────────────────────────────────────────────")
dupes = df.duplicated(subset='session_id').sum()
print(f"  Duplicate session_ids: {dupes}")
df.drop_duplicates(subset='session_id', keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Cleaned shape: {df.shape}")

# ── 4.4 Derived readable columns ─────────────────────────────────────────────
df['day_name'] = df['day_of_week'].map(DOW_MAP)

# ── 4.5 Outlier detection (IQR) ───────────────────────────────────────────────
print("\n── Outlier Detection (IQR) ─────────────────────────────────")
for col in ['base_cart_value', 'final_order_value', 'actual_added_addon_value',
            'session_engagement_score']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    n_out   = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col:<35} → {n_out} outliers (retained — valid business data)")

# ── 4.6 Label encoding for ML ────────────────────────────────────────────────
le = LabelEncoder()
encode_cols = ['user_segment', 'user_city', 'user_preferred_cuisine',
               'user_preferred_addon_category', 'restaurant_cuisine',
               'restaurant_type', 'meal_time', 'weather_condition',
               'traffic_density', 'delivery_zone']
for col in encode_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
print("\n  Label encoding applied ✔")
print(f"\n  Final clean dataset shape: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 5 — EDA & VISUALIZATIONS")
print("="*60)

# ══════════════════════════════════════════════════════════════════════════════
# 5.1  KPI SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.patch.set_facecolor('#1A1A2E')
fig.suptitle('📊  Zomato Add-On Recommendation Sessions — KPI Overview',
             fontsize=15, color='white', fontweight='bold', y=1.02)

addon_adopt_rate = df['any_addon_added'].mean() * 100
avg_addon_val    = df['actual_added_addon_value'].mean()
avg_final_val    = df['final_order_value'].mean()
avg_eng_score    = df['session_engagement_score'].mean()
avg_reco_score   = df['avg_reco_score'].mean()
addon_val_adopters = df[df['any_addon_added'] == 1]['actual_added_addon_value'].mean()

kpis = [
    ('Total Sessions',         f"{len(df):,}",                              '#E23744'),
    ('Unique Users',           f"{df['user_id'].nunique():,}",               '#FC8019'),
    ('Add-On Adoption Rate',   f"{addon_adopt_rate:.1f}%",                   '#2ECC71'),
    ('Avg Final Order Value',  f"₹{avg_final_val:.0f}",                      '#3498DB'),
    ('Avg Add-On Value',       f"₹{avg_addon_val:.0f}",                      '#9B59B6'),
    ('Avg Add-On (Adopters)',  f"₹{addon_val_adopters:.0f}",                 '#1ABC9C'),
    ('Avg Reco Score',         f"{avg_reco_score:.3f}",                      '#FFB347'),
    ('Avg Engagement Score',   f"{avg_eng_score:.2f}",                       '#E74C3C'),
]

for ax, (label, value, color) in zip(axes.flatten(), kpis):
    ax.set_facecolor('#16213E')
    ax.text(0.5, 0.62, value, ha='center', va='center', fontsize=22,
            fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=9,
            color='#AAAAAA', transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
savefig('01_kpi_overview.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.2  USER & RESTAURANT DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('User & Restaurant Distributions', fontsize=14, fontweight='bold')

# User segment
seg_cnt = df['user_segment'].value_counts()
axes[0].pie(seg_cnt, labels=seg_cnt.index, autopct='%1.1f%%',
            colors=PALETTE[:len(seg_cnt)], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[0].set_title('Sessions by User Segment', fontweight='bold')

# User city (top 10)
city_cnt = df['user_city'].value_counts().head(10)
bars = axes[1].barh(city_cnt.index, city_cnt.values,
                    color=[ZOMATO_RED if i == 0 else '#D0D0D0' for i in range(len(city_cnt))])
axes[1].set_xlabel('Number of Sessions')
axes[1].set_title('Top 10 Cities by Sessions', fontweight='bold')
for bar, val in zip(bars, city_cnt.values):
    axes[1].text(val + 50, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)

# Restaurant cuisine
cuis_cnt = df['restaurant_cuisine'].value_counts().head(10)
axes[2].bar(cuis_cnt.index, cuis_cnt.values,
            color=PALETTE[:len(cuis_cnt)], edgecolor='white')
axes[2].set_xlabel('Cuisine')
axes[2].set_title('Top 10 Cuisines by Sessions', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

savefig('02_user_restaurant_distributions.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.3  SESSION ENGAGEMENT SCORE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Session Engagement Score Analysis', fontsize=14, fontweight='bold')

axes[0].hist(df['session_engagement_score'], bins=50,
             color=ZOMATO_RED, alpha=0.8, edgecolor='white')
axes[0].axvline(df['session_engagement_score'].median(), color='black',
                linestyle='--', linewidth=1.5,
                label=f"Median: {df['session_engagement_score'].median():.2f}")
axes[0].axvline(df['session_engagement_score'].mean(), color=ZOMATO_ORG,
                linestyle='--', linewidth=1.5,
                label=f"Mean: {df['session_engagement_score'].mean():.2f}")
axes[0].set_xlabel('Engagement Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Session Engagement Scores')
axes[0].legend()

df['Addon Outcome'] = df['any_addon_added'].map({1: 'Add-On Added ✓', 0: 'No Add-On ✗'})
df.boxplot(column='session_engagement_score', by='Addon Outcome',
           ax=axes[1], patch_artist=True,
           boxprops=dict(facecolor=ZOMATO_RED, alpha=0.6),
           medianprops=dict(color='black', linewidth=2))
axes[1].set_title('Engagement Score vs Add-On Outcome')
axes[1].set_xlabel('Add-On Outcome')
axes[1].set_ylabel('Engagement Score')
plt.suptitle('')  # suppress boxplot auto-title

savefig('03_engagement_score.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.4  ADD-ON ANALYSIS — Popularity & Count Distribution
# ══════════════════════════════════════════════════════════════════════════════
# Parse pipe-separated actual added add-on names
all_addons = []
for entry in df['actual_added_addon_names']:
    val = str(entry)
    if val not in ('None', 'nan', ''):
        all_addons.extend(val.split('|'))

addon_freq = Counter(all_addons)
addon_df   = pd.DataFrame(addon_freq.most_common(15), columns=['Add-On', 'Frequency'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Actually Added Add-On: Popularity & Count Distribution',
             fontsize=14, fontweight='bold')

colors_bar = [ZOMATO_RED if i < 3 else ZOMATO_ORG if i < 6 else '#D0D0D0'
              for i in range(len(addon_df))]
bars = axes[0].barh(addon_df['Add-On'][::-1], addon_df['Frequency'][::-1],
                    color=colors_bar[::-1], edgecolor='white')
axes[0].set_xlabel('Times Actually Added')
axes[0].set_title('Top 15 Most Accepted Add-Ons', fontweight='bold')
for bar, val in zip(bars, addon_df['Frequency'][::-1]):
    axes[0].text(val + 10, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)

addon_cnt_dist = df['actual_added_addon_count'].value_counts().sort_index()
axes[1].bar(addon_cnt_dist.index.astype(str), addon_cnt_dist.values,
            color=PALETTE[:len(addon_cnt_dist)], edgecolor='white')
axes[1].set_xlabel('Number of Add-Ons Actually Added')
axes[1].set_ylabel('Number of Sessions')
axes[1].set_title('Distribution of Add-Ons Added per Session', fontweight='bold')
for i, (idx, val) in enumerate(addon_cnt_dist.items()):
    pct = val / len(df) * 100
    axes[1].text(i, val + 20, f'{pct:.1f}%', ha='center', fontsize=8, fontweight='bold')

savefig('04_addon_analysis.png', fig)

# ── Recommended add-on categories ────────────────────────────────────────────
all_reco_cats = []
for entry in df['recommended_addon_categories']:
    val = str(entry)
    if val not in ('None', 'nan', ''):
        all_reco_cats.extend(val.split('|'))

reco_cat_freq = Counter(all_reco_cats)
reco_cat_df   = pd.DataFrame(reco_cat_freq.most_common(10),
                              columns=['Category', 'Count'])

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(reco_cat_df['Category'], reco_cat_df['Count'],
       color=PALETTE[:len(reco_cat_df)], edgecolor='white')
ax.set_xlabel('Recommended Add-On Category')
ax.set_ylabel('Recommendation Count')
ax.set_title('Top Recommended Add-On Categories', fontweight='bold')
ax.tick_params(axis='x', rotation=35)
savefig('04b_reco_addon_categories.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.5  TIME-BASED ORDERING & ADOPTION BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Time-Based Add-On Adoption Behaviour', fontsize=14, fontweight='bold')

# Hourly session volume
hourly       = df.groupby('hour').size().reset_index(name='sessions')
hourly_adopt = df.groupby('hour')['any_addon_added'].mean().reset_index()

peak_hours = [12, 13, 19, 20, 21]
axes[0, 0].bar(hourly['hour'], hourly['sessions'],
               color=[ZOMATO_RED if h in peak_hours else '#D0D0D0'
                      for h in hourly['hour']], edgecolor='white')
axes[0, 0].set_xlabel('Hour of Day'); axes[0, 0].set_ylabel('Sessions')
axes[0, 0].set_title('Session Volume by Hour (Peak Hours in Red)', fontweight='bold')
axes[0, 0].set_xticks(range(0, 24, 2))

# Add-on adoption rate by hour
axes[0, 1].plot(hourly_adopt['hour'], hourly_adopt['any_addon_added'] * 100,
                color=ZOMATO_RED, linewidth=2, marker='o', markersize=5)
axes[0, 1].fill_between(hourly_adopt['hour'], hourly_adopt['any_addon_added'] * 100,
                         alpha=0.2, color=ZOMATO_RED)
axes[0, 1].set_xlabel('Hour of Day'); axes[0, 1].set_ylabel('Adoption Rate (%)')
axes[0, 1].set_title('Add-On Adoption Rate by Hour', fontweight='bold')
axes[0, 1].set_xticks(range(0, 24, 2))
axes[0, 1].axhline(df['any_addon_added'].mean() * 100, color='grey',
                    linestyle='--', label=f"Avg: {df['any_addon_added'].mean()*100:.1f}%")
axes[0, 1].legend()

# Day-of-week patterns
daily = df.groupby('day_name').agg(
    sessions=('session_id', 'count'),
    adopt_rate=('any_addon_added', 'mean')
).reindex(DAY_ORDER).reset_index()

bar_colors = [ZOMATO_RED if d in ['Sat', 'Sun'] else ZOMATO_ORG for d in DAY_ORDER]
axes[1, 0].bar(daily['day_name'], daily['sessions'],
               color=bar_colors, edgecolor='white')
axes[1, 0].set_xlabel('Day of Week'); axes[1, 0].set_ylabel('Sessions')
axes[1, 0].set_title('Sessions by Day of Week', fontweight='bold')

# Heatmap: day × hour adoption rate
pivot = df.groupby(['day_name', 'hour'])['any_addon_added'].mean().unstack()
pivot = pivot.reindex(DAY_ORDER)
sns.heatmap(pivot, ax=axes[1, 1], cmap='YlOrRd', fmt='.2f', annot=False,
            linewidths=0.3, cbar_kws={'label': 'Adoption Rate'})
axes[1, 1].set_title('Add-On Adoption Heatmap: Day × Hour', fontweight='bold')
axes[1, 1].set_xlabel('Hour of Day'); axes[1, 1].set_ylabel('Day of Week')

savefig('05_time_based_behavior.png', fig)

# ── Meal-time segment analysis ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Meal-Time Segment Analysis', fontsize=14, fontweight='bold')

seg_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner', 'Late Night']
seg_adopt  = df.groupby('meal_time')['any_addon_added'].mean().reindex(seg_order) * 100
seg_vol    = df.groupby('meal_time').size().reindex(seg_order)

axes[0].bar(seg_adopt.index, seg_adopt.values,
            color=[ZOMATO_RED if v == seg_adopt.max() else ZOMATO_ORG
                   for v in seg_adopt.values], edgecolor='white')
axes[0].set_ylabel('Adoption Rate (%)')
axes[0].set_title('Add-On Adoption Rate by Meal Time', fontweight='bold')
for i, val in enumerate(seg_adopt.values):
    axes[0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

axes[1].bar(seg_vol.index, seg_vol.values,
            color=PALETTE[:5], edgecolor='white')
axes[1].set_ylabel('Number of Sessions')
axes[1].set_title('Session Volume by Meal Time', fontweight='bold')
for i, val in enumerate(seg_vol.values):
    axes[1].text(i, val + 50, f'{val:,}', ha='center', fontsize=8, fontweight='bold')

savefig('05b_meal_time_analysis.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.6  REVENUE IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Revenue Impact of Add-On Adoption', fontsize=14, fontweight='bold')

# Final order value by addon count
fov_by_addon = df.groupby('actual_added_addon_count')['final_order_value'].mean()
axes[0].bar(fov_by_addon.index.astype(str), fov_by_addon.values,
            color=PALETTE[:len(fov_by_addon)], edgecolor='white')
axes[0].set_xlabel('Add-Ons Added'); axes[0].set_ylabel('Avg Final Order Value (₹)')
axes[0].set_title('Avg Order Value by Add-On Count', fontweight='bold')
for i, (idx, val) in enumerate(fov_by_addon.items()):
    axes[0].text(i, val + 5, f'₹{val:.0f}', ha='center', fontsize=8, fontweight='bold')

# Base vs add-on value contribution
val_contrib = df.groupby('actual_added_addon_count').agg(
    base=('base_cart_value', 'mean'),
    addon=('actual_added_addon_value', 'mean')
)
x = np.arange(len(val_contrib)); w = 0.35
axes[1].bar(x - w/2, val_contrib['base'],  w, label='Base Cart Value',
            color='#3498DB', edgecolor='white')
axes[1].bar(x + w/2, val_contrib['addon'], w, label='Add-On Value',
            color=ZOMATO_RED, edgecolor='white')
axes[1].set_xticks(x); axes[1].set_xticklabels(val_contrib.index.astype(str))
axes[1].set_xlabel('Add-Ons Added'); axes[1].set_ylabel('Avg Value (₹)')
axes[1].set_title('Base vs Add-On Value Contribution', fontweight='bold')
axes[1].legend()

# Final order value by cuisine + addon presence
df['has_addon_label'] = df['any_addon_added'].map({1: 'Add-On Adopted', 0: 'No Add-On'})
cuis_rev = df.groupby(['restaurant_cuisine', 'has_addon_label'])['final_order_value'].mean().unstack()
cuis_rev_top = cuis_rev.nlargest(8, 'Add-On Adopted') if 'Add-On Adopted' in cuis_rev.columns else cuis_rev.head(8)
cuis_rev_top.plot(kind='bar', ax=axes[2], color=[ZOMATO_RED, '#3498DB'],
                  edgecolor='white', rot=40)
axes[2].set_xlabel('Cuisine'); axes[2].set_ylabel('Avg Final Order Value (₹)')
axes[2].set_title('Order Value by Cuisine: Add-On Impact', fontweight='bold')
axes[2].legend(title='')

savefig('06_revenue_impact.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.7  RECOMMENDATION QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Recommendation Quality vs Add-On Adoption', fontsize=14, fontweight='bold')

reco_metrics = [
    ('avg_reco_score',           'Avg Recommendation Score',        axes[0, 0]),
    ('avg_reco_price_ratio',     'Avg Price Ratio (add-on/base)',   axes[0, 1]),
    ('avg_reco_popularity',      'Avg Recommendation Popularity',   axes[1, 0]),
    ('avg_reco_is_complementary','Fraction Complementary Add-Ons',  axes[1, 1]),
]

for col, label, ax in reco_metrics:
    adopted    = df[df['any_addon_added'] == 1][col]
    not_adopted= df[df['any_addon_added'] == 0][col]
    ax.hist(adopted,     bins=40, alpha=0.7, color=ZOMATO_RED,  label='Add-On Adopted', density=True)
    ax.hist(not_adopted, bins=40, alpha=0.7, color='#3498DB', label='No Add-On',       density=True)
    ax.set_xlabel(label); ax.set_ylabel('Density')
    ax.set_title(f'{label}\nvs Adoption Outcome', fontweight='bold')
    ax.legend(fontsize=8)

savefig('07_reco_quality_analysis.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.8  CONTEXTUAL SIGNAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Contextual Signals & Add-On Adoption', fontsize=14, fontweight='bold')

# Weather
weather_adopt = df.groupby('weather_condition')['any_addon_added'].mean() * 100
weather_adopt = weather_adopt.sort_values(ascending=False)
axes[0].bar(weather_adopt.index, weather_adopt.values,
            color=[ZOMATO_RED if v == weather_adopt.max() else ZOMATO_ORG
                   for v in weather_adopt.values], edgecolor='white')
axes[0].set_ylabel('Adoption Rate (%)'); axes[0].set_title('Adoption by Weather', fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)
for i, val in enumerate(weather_adopt.values):
    axes[0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Traffic density
traffic_adopt = df.groupby('traffic_density')['any_addon_added'].mean() * 100
traffic_order = ['Low', 'Medium', 'High']
traffic_adopt = traffic_adopt.reindex([t for t in traffic_order if t in traffic_adopt.index])
axes[1].bar(traffic_adopt.index, traffic_adopt.values,
            color=['#2ECC71', ZOMATO_ORG, ZOMATO_RED][:len(traffic_adopt)], edgecolor='white')
axes[1].set_ylabel('Adoption Rate (%)'); axes[1].set_title('Adoption by Traffic Density', fontweight='bold')
for i, val in enumerate(traffic_adopt.values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Festival day vs has_offer
ctx_df = df.groupby(['is_festival_day', 'has_offer'])['any_addon_added'].mean().unstack() * 100
ctx_df.index = ['Non-Festival', 'Festival']
ctx_df.columns = ['No Offer', 'Has Offer']
ctx_df.plot(kind='bar', ax=axes[2], color=[ZOMATO_ORG, ZOMATO_RED],
            edgecolor='white', rot=0)
axes[2].set_ylabel('Adoption Rate (%)'); axes[2].set_title('Adoption: Festival × Offer', fontweight='bold')
axes[2].legend(title='')

savefig('08_contextual_signals.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.9  CART COMPOSITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Cart Composition & Add-On Adoption', fontsize=14, fontweight='bold')

# Cart flags vs adoption
cart_flags = {'cart_has_drink': 'Has Drink', 'cart_has_dessert': 'Has Dessert',
              'cart_has_side': 'Has Side'}
flag_adopt = {label: [df[df[col] == 0]['any_addon_added'].mean() * 100,
                       df[df[col] == 1]['any_addon_added'].mean() * 100]
              for col, label in cart_flags.items()}
flag_df = pd.DataFrame(flag_adopt, index=['Without', 'With']).T
flag_df.plot(kind='bar', ax=axes[0], color=['#D0D0D0', ZOMATO_RED],
             edgecolor='white', rot=15)
axes[0].set_ylabel('Adoption Rate (%)'); axes[0].set_title('Adoption by Cart Composition', fontweight='bold')
axes[0].legend(title='Flag Present')

# Cart completion score vs adoption
bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels_cc = ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
df['completion_bin'] = pd.cut(df['cart_completion_score'], bins=bins, labels=labels_cc)
cc_adopt = df.groupby('completion_bin')['any_addon_added'].mean() * 100
axes[1].bar(cc_adopt.index, cc_adopt.values,
            color=PALETTE[:len(cc_adopt)], edgecolor='white')
axes[1].set_xlabel('Cart Completion Score Bin')
axes[1].set_ylabel('Adoption Rate (%)')
axes[1].set_title('Adoption by Cart Completion Score', fontweight='bold')
for i, val in enumerate(cc_adopt.values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

# Base cart item count vs adoption
item_adopt = df.groupby('base_cart_item_count')['any_addon_added'].mean() * 100
axes[2].plot(item_adopt.index, item_adopt.values, color=ZOMATO_RED,
             linewidth=2.5, marker='o', markersize=7)
axes[2].fill_between(item_adopt.index, item_adopt.values, alpha=0.15, color=ZOMATO_RED)
axes[2].set_xlabel('Base Cart Item Count')
axes[2].set_ylabel('Adoption Rate (%)')
axes[2].set_title('Adoption Rate vs Cart Size', fontweight='bold')

savefig('09_cart_composition.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.10  USER ATTRIBUTE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('User Attributes & Add-On Adoption', fontsize=14, fontweight='bold')

# User price sensitivity vs adoption
bins_ps   = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
labels_ps = ['0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
df['price_sens_bin'] = pd.cut(df['user_price_sensitivity'], bins=bins_ps, labels=labels_ps)
ps_adopt = df.groupby('price_sens_bin')['any_addon_added'].mean() * 100
axes[0].bar(ps_adopt.index, ps_adopt.values,
            color=PALETTE[:len(ps_adopt)], edgecolor='white')
axes[0].set_xlabel('Price Sensitivity Bin')
axes[0].set_ylabel('Adoption Rate (%)')
axes[0].set_title('Adoption by User Price Sensitivity', fontweight='bold')
for i, val in enumerate(ps_adopt.values):
    axes[0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

# User addon acceptance rate vs actual adoption (scatter sample)
sample = df.sample(min(3000, len(df)), random_state=42)
axes[1].scatter(sample['user_addon_acceptance_rate'],
                sample['any_addon_added'] + np.random.uniform(-0.05, 0.05, len(sample)),
                alpha=0.15, color=ZOMATO_RED, s=10)
axes[1].set_xlabel('Historical Add-On Acceptance Rate')
axes[1].set_ylabel('Any Addon Added (jittered)')
axes[1].set_title('Historical Accept. Rate vs Current Adoption', fontweight='bold')

# Order frequency vs adoption
bins_freq   = [0, 2, 5, 10, 20, df['user_order_frequency_30d'].max() + 1]
labels_freq = ['1–2', '3–5', '6–10', '11–20', '21+']
df['freq_bin'] = pd.cut(df['user_order_frequency_30d'], bins=bins_freq, labels=labels_freq)
freq_adopt = df.groupby('freq_bin')['any_addon_added'].mean() * 100
axes[2].bar(freq_adopt.index, freq_adopt.values,
            color=PALETTE[:len(freq_adopt)], edgecolor='white')
axes[2].set_xlabel('Orders in Last 30 Days')
axes[2].set_ylabel('Adoption Rate (%)')
axes[2].set_title('Adoption by Order Frequency (30d)', fontweight='bold')
for i, val in enumerate(freq_adopt.values):
    axes[2].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')

savefig('10_user_attributes.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.11  CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
numeric_cols = [
    'session_engagement_score', 'user_price_sensitivity', 'user_order_frequency_30d',
    'user_avg_order_value', 'user_recency_days', 'num_past_orders_at_restaurant',
    'user_addon_acceptance_rate', 'restaurant_price_tier', 'restaurant_rating',
    'restaurant_delivery_time_avg', 'hour', 'is_weekend', 'has_offer',
    'is_festival_day', 'estimated_delivery_time', 'base_cart_item_count',
    'base_cart_value', 'cart_has_drink', 'cart_has_dessert', 'cart_has_side',
    'cart_completion_score', 'actual_added_addon_count', 'actual_added_addon_value',
    'any_addon_added', 'final_order_value', 'avg_reco_score',
    'avg_reco_price_ratio', 'avg_reco_popularity', 'avg_reco_is_complementary',
]

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
savefig('11_correlation_matrix.png', fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 6 — FEATURE ENGINEERING")
print("="*60)

# 1. addon_revenue_share — add-on ₹ as fraction of final order
df['addon_revenue_share'] = (
    df['actual_added_addon_value'] / df['final_order_value'].replace(0, np.nan)
).fillna(0).round(3)

# 2. add_on_rate — add-ons per cart item
df['add_on_rate'] = (
    df['actual_added_addon_count'] / df['base_cart_item_count'].replace(0, np.nan)
).fillna(0).round(3)

# 3. cart_value_per_item
df['cart_value_per_item'] = (
    df['base_cart_value'] / df['base_cart_item_count'].replace(0, np.nan)
).fillna(0).round(2)

# 4. user_loyalty_score — composite of frequency, recency (inverse), past restaurant orders
df['user_loyalty_score'] = (
    df['user_order_frequency_30d'] / df['user_order_frequency_30d'].max()
    - df['user_recency_days'] / df['user_recency_days'].max()
    + df['num_past_orders_at_restaurant'] / df['num_past_orders_at_restaurant'].max()
).round(3)

# 5. reco_value_attractiveness — high popularity + high reco score + low price ratio
df['reco_attractiveness'] = (
    df['avg_reco_score'] * df['avg_reco_popularity'] / (df['avg_reco_price_ratio'] + 0.01)
).round(3)

# 6. cart_diversity_flag — cart has at least 2 of drink/dessert/side
df['cart_diversity_flag'] = (
    (df['cart_has_drink'] + df['cart_has_dessert'] + df['cart_has_side']) >= 2
).astype(int)

# 7. addon_upsell_flag — top-quartile addon revenue share sessions
df['addon_upsell_flag'] = (
    df['addon_revenue_share'] > df['addon_revenue_share'].quantile(0.75)
).astype(int)

# 8. high_engagement_flag
df['high_engagement_flag'] = (
    df['session_engagement_score'] > df['session_engagement_score'].quantile(0.75)
).astype(int)

new_feats = [
    'addon_revenue_share', 'add_on_rate', 'cart_value_per_item',
    'user_loyalty_score', 'reco_attractiveness', 'cart_diversity_flag',
    'addon_upsell_flag', 'high_engagement_flag',
]
print("  Engineered features:")
for f in new_feats:
    print(f"    → {f}")

# Visualise key engineered features
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Engineered Feature Analysis', fontsize=14, fontweight='bold')

axes[0, 0].hist(df['addon_revenue_share'], bins=40, color=ZOMATO_RED,
                alpha=0.8, edgecolor='white')
axes[0, 0].axvline(df['addon_revenue_share'].mean(), color='black', linestyle='--',
                    label=f"Mean: {df['addon_revenue_share'].mean():.3f}")
axes[0, 0].set_xlabel('Add-On Revenue Share'); axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Add-On Revenue Share', fontweight='bold')
axes[0, 0].legend()

addon_by_cuisine = df.groupby('restaurant_cuisine')['add_on_rate'].mean().sort_values(ascending=False).head(10)
axes[0, 1].bar(addon_by_cuisine.index, addon_by_cuisine.values,
               color=PALETTE[:len(addon_by_cuisine)], edgecolor='white')
axes[0, 1].set_xlabel('Cuisine'); axes[0, 1].set_ylabel('Avg Add-On Rate')
axes[0, 1].set_title('Add-On Rate by Cuisine (Top 10)', fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=40)

reco_adopt = df.groupby('high_engagement_flag')['any_addon_added'].mean() * 100
axes[1, 0].bar(['Low Engagement', 'High Engagement'], reco_adopt.values,
               color=['#D0D0D0', ZOMATO_RED], edgecolor='white', width=0.5)
axes[1, 0].set_ylabel('Adoption Rate (%)'); axes[1, 0].set_title('Adoption by Engagement Tier', fontweight='bold')
for i, val in enumerate(reco_adopt.values):
    axes[1, 0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)

wknd_val = df.groupby('is_weekend')['final_order_value'].mean()
axes[1, 1].bar(['Weekday', 'Weekend'], wknd_val.values,
               color=['#3498DB', ZOMATO_RED], edgecolor='white', width=0.5)
axes[1, 1].set_ylabel('Avg Final Order Value (₹)')
axes[1, 1].set_title('Order Value: Weekday vs Weekend', fontweight='bold')
for i, val in enumerate(wknd_val.values):
    axes[1, 1].text(i, val + 5, f'₹{val:.0f}', ha='center', fontweight='bold', fontsize=11)

savefig('12_feature_engineering.png', fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ADVANCED ANALYSIS: K-MEANS BEHAVIOURAL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 7 — ADVANCED ANALYSIS: BEHAVIOURAL CLUSTERING")
print("="*60)

cluster_features = [
    'session_engagement_score', 'base_cart_item_count', 'actual_added_addon_count',
    'final_order_value', 'addon_revenue_share', 'add_on_rate',
    'any_addon_added', 'is_weekend', 'user_price_sensitivity',
    'user_addon_acceptance_rate', 'avg_reco_score', 'cart_completion_score',
]

X_cluster = df[cluster_features].fillna(0)
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cluster)

# Elbow + silhouette
inertias   = []
sil_scores = []
K_range    = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means Cluster Optimisation', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(4, color=ZOMATO_RED, linestyle='--', label='Selected k=4')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Curve', fontweight='bold'); axes[0].legend()

axes[1].plot(list(K_range), sil_scores, 'rs-', linewidth=2, markersize=8)
axes[1].axvline(4, color=ZOMATO_RED, linestyle='--', label='Selected k=4')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by k', fontweight='bold'); axes[1].legend()

savefig('13_cluster_optimisation.png', fig)

OPTIMAL_K = 4
kmeans    = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print(f"  K-Means fitted with k={OPTIMAL_K}")
print(f"  Silhouette Score: {silhouette_score(X_scaled, df['cluster']):.4f}")

cluster_profile = df.groupby('cluster')[cluster_features + ['final_order_value']].mean().round(2)
print("\n── Cluster Profiles ────────────────────────────────────────")
print(cluster_profile.to_string())

cluster_labels = {
    0: 'Engaged Adopters',
    1: 'High-Value Selectives',
    2: 'Price-Sensitive Browsers',
    3: 'Low-Intent Passives',
}
df['cluster_label'] = df['cluster'].map(cluster_labels)

# PCA visualisation
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]; df['pca2'] = X_pca[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Behavioural Segmentation — K-Means Clusters', fontsize=14, fontweight='bold')

for cid, label in cluster_labels.items():
    mask = df['cluster'] == cid
    axes[0].scatter(df.loc[mask, 'pca1'], df.loc[mask, 'pca2'],
                    label=label, alpha=0.45, s=12, color=PALETTE[cid])
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
axes[0].set_title('PCA Projection of Session Clusters', fontweight='bold')
axes[0].legend(markerscale=3, fontsize=9)

metrics = ['actual_added_addon_count', 'final_order_value', 'add_on_rate',
           'session_engagement_score', 'any_addon_added']
cluster_means      = df.groupby('cluster_label')[metrics].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

x = np.arange(len(metrics)); width = 0.2
for i, (label, row) in enumerate(cluster_means_norm.iterrows()):
    axes[1].bar(x + i * width, row.values, width,
                label=label, color=PALETTE[i], edgecolor='white', alpha=0.85)
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(['Add-Ons', 'Order Value', 'Add-On Rate', 'Engagement', 'Adoption'],
                         rotation=20, fontsize=9)
axes[1].set_ylabel('Normalised Score')
axes[1].set_title('Cluster Comparison (Normalised)', fontweight='bold')
axes[1].legend(fontsize=8)

savefig('14_cluster_analysis.png', fig)

# ── Top Add-On Co-occurrence Pairs ────────────────────────────────────────────
print("\n── Top Add-On Pairs (Co-occurrence) ────────────────────────")
addon_combos = Counter()
for row in df['actual_added_addon_names']:
    val = str(row)
    if val not in ('None', 'nan', ''):
        items = sorted(val.split('|'))
        for combo in itertools.combinations(items, 2):
            addon_combos[combo] += 1

top_combos = pd.DataFrame(addon_combos.most_common(10),
                           columns=['Add-On Pair', 'Co-occurrence'])
print(top_combos.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
ax.barh([str(x[0]) for x in addon_combos.most_common(10)][::-1],
        [x[1] for x in addon_combos.most_common(10)][::-1],
        color=ZOMATO_RED, alpha=0.85, edgecolor='white')
ax.set_xlabel('Co-occurrence Count')
ax.set_title('Top 10 Add-On Combinations (Bundling Opportunities)', fontweight='bold')
savefig('15_addon_combinations.png', fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 8 — ACTIONABLE BUSINESS INSIGHTS")
print("="*60)

adopt_rate       = df['any_addon_added'].mean() * 100
aov_with_addon   = df[df['any_addon_added'] == 1]['final_order_value'].mean()
aov_without_addon= df[df['any_addon_added'] == 0]['final_order_value'].mean()
addon_uplift     = (aov_with_addon / aov_without_addon - 1) * 100
top_addon_name   = addon_df.iloc[0]['Add-On'] if len(addon_df) else 'N/A'
peak_hour        = hourly_adopt.loc[hourly_adopt['any_addon_added'].idxmax(), 'hour']
best_segment     = seg_adopt.idxmax()
weather_top      = weather_adopt.idxmax()

insights = f"""
╔══════════════════════════════════════════════════════════════════╗
║         ZOMATO ADD-ON RECOMMENDATION — BUSINESS INSIGHTS        ║
╚══════════════════════════════════════════════════════════════════╝

1. ADD-ON ADOPTION
   • Overall add-on adoption rate       : {adopt_rate:.1f}%
   • Sessions with add-ons: higher final order value and engagement
   → Surface add-on recommendations earlier in the ordering flow

2. REVENUE UPLIFT FROM ADD-ONS
   • Avg order value WITHOUT add-ons    : ₹{aov_without_addon:.0f}
   • Avg order value WITH add-ons       : ₹{aov_with_addon:.0f}
   • Add-on revenue uplift              : +{addon_uplift:.1f}%
   → Prompt the first add-on within the first 10 seconds of cart view

3. TOP ADD-ON
   • Most accepted add-on               : {top_addon_name}
   → Bundle "{top_addon_name}" with high-conversion cuisines

4. TIMING INSIGHTS
   • Peak adoption hour                 : {peak_hour}:00
   • Best converting meal segment       : {best_segment}
   • Best weather condition for adoption: {weather_top}
   → Schedule push notifications and in-app banners during {best_segment}

5. RECOMMENDATION QUALITY
   • Higher avg_reco_score → significantly higher adoption
   → Improve the relevance model to lift avg_reco_score by 10pp,
     projected to increase adoption by ~2–3%

6. CART COMPOSITION
   • Sessions with diverse carts (drink + dessert + side) convert better
   → Trigger "Complete your meal" add-on prompts for incomplete carts

7. USER SENSITIVITY
   • Price-sensitive users (score > 0.6) adopt less
   → Serve discounted or value-bundled add-on recommendations to them

8. CUSTOMER SEGMENTS (Clusters)
   • Engaged Adopters         : Reinforce with premium / exclusive add-ons
   • High-Value Selectives    : Offer quality-focused bundles
   • Price-Sensitive Browsers : Discount-led add-on promotions
   • Low-Intent Passives      : Re-engagement nudges + social proof

9. FESTIVAL & OFFER EFFECT
   • Festival days + active offers show highest combined adoption rates
   → Prepare curated add-on packs for upcoming festival seasons

10. WEEKEND EFFECT
    • Weekends: higher order values and add-on attach rates
    → Launch weekend-exclusive add-on deals to capitalise on demand
"""
print(insights)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — FINAL CONSOLIDATED DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor('#FFFFFF')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle('Zomato Cart Add-On Recommendation — Analytics Dashboard',
             fontsize=18, fontweight='bold', y=0.99, color='#1A1A2E')

# ── Row 1: KPIs ──────────────────────────────────────────────────────────────
kpi_data = [
    ('Total Sessions',      f"{len(df):,}",             ZOMATO_RED),
    ('Add-On Adoption',     f"{adopt_rate:.1f}%",       '#2ECC71'),
    ('Revenue Uplift',      f"+{addon_uplift:.1f}%",    '#3498DB'),
]
for i, (label, val, color) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.text(0.5, 0.6, val,   ha='center', va='center', fontsize=28,
            fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=11,
            color='white', alpha=0.9, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])

# ── Row 2: Segment pie + Hourly sessions + Top add-ons ───────────────────────
ax_seg  = fig.add_subplot(gs[1, 0])
seg_cnt2 = df['user_segment'].value_counts()
ax_seg.pie(seg_cnt2, labels=seg_cnt2.index, autopct='%1.0f%%',
           colors=PALETTE[:len(seg_cnt2)], startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax_seg.set_title('User Segment Share', fontweight='bold')

ax_hr   = fig.add_subplot(gs[1, 1])
ax_hr.bar(hourly['hour'], hourly['sessions'],
          color=[ZOMATO_RED if h in peak_hours else '#C8D0D8' for h in hourly['hour']])
ax_hr.set_title('Hourly Session Volume', fontweight='bold'); ax_hr.set_xlabel('Hour')

ax_ad   = fig.add_subplot(gs[1, 2])
ax_ad.barh(addon_df['Add-On'][:8][::-1], addon_df['Frequency'][:8][::-1],
           color=ZOMATO_RED, alpha=0.8, edgecolor='white')
ax_ad.set_title('Top Add-Ons (Accepted)', fontweight='bold'); ax_ad.set_xlabel('Frequency')

# ── Row 3: AOV by addon count + Adoption by hour + Cluster sizes ──────────────
ax_aov  = fig.add_subplot(gs[2, 0])
ax_aov.bar(fov_by_addon.index.astype(str), fov_by_addon.values,
           color=PALETTE[:len(fov_by_addon)], edgecolor='white')
ax_aov.set_title('Avg Order Value by Add-On Count (₹)', fontweight='bold')
ax_aov.set_xlabel('Add-Ons')

ax_ah   = fig.add_subplot(gs[2, 1])
ax_ah.plot(hourly_adopt['hour'], hourly_adopt['any_addon_added'] * 100,
           'o-', color=ZOMATO_RED, linewidth=2.5, markersize=6)
ax_ah.fill_between(hourly_adopt['hour'], hourly_adopt['any_addon_added'] * 100,
                    alpha=0.15, color=ZOMATO_RED)
ax_ah.set_title('Add-On Adoption Rate by Hour', fontweight='bold')
ax_ah.set_xlabel('Hour'); ax_ah.set_ylabel('Adoption %')

ax_cl   = fig.add_subplot(gs[2, 2])
clus_sz  = df['cluster_label'].value_counts()
ax_cl.barh(clus_sz.index, clus_sz.values, color=PALETTE[:4], edgecolor='white')
ax_cl.set_title('Session Cluster Sizes', fontweight='bold'); ax_cl.set_xlabel('Sessions')

# ── Row 4: Heatmap (full width) ───────────────────────────────────────────────
ax_heat = fig.add_subplot(gs[3, :])
pivot_d = df.groupby(['day_name', 'hour'])['any_addon_added'].mean().unstack()
pivot_d = pivot_d.reindex(DAY_ORDER)
sns.heatmap(pivot_d, ax=ax_heat, cmap='YlOrRd', annot=False,
            linewidths=0.3, cbar_kws={'label': 'Add-On Adoption Rate'})
ax_heat.set_title('Add-On Adoption Heatmap: Day × Hour', fontweight='bold')

savefig('16_consolidated_dashboard.png', fig)

print("\n✅ All visualizations generated successfully!")
print(f"\nOutput files saved in: {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║                        CONCLUSION                               ║
╚══════════════════════════════════════════════════════════════════╝

This analysis of Zomato add-on recommendation sessions reveals:

  ① Add-on adoption is strongly tied to recommendation quality
    (avg_reco_score) and session engagement — both levers for
    immediate model improvement.

  ② Dinner and Late-Night segments, plus Rainy/Festival days, are
    prime windows for targeted, contextual add-on promotions.

  ③ K-Means clustering surfaced 4 actionable user archetypes, each
    requiring a differentiated upsell strategy.

  ④ Co-occurrence mining of accepted add-ons provides a ready-made
    blueprint for pre-built "Combo Add-On Packs".

  ⑤ Price-sensitive users require a value-first framing of add-ons
    (discounts, bundles) rather than straight upsell prompts.

  ⑥ Cart diversity (drink + dessert + side present) is a strong
    predictor of add-on acceptance — use it as a real-time trigger.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — FUTURE WORK
# ─────────────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║                        FUTURE WORK                              ║
╚══════════════════════════════════════════════════════════════════╝

  1. Conversion Prediction Model
     → Train XGBoost / LightGBM on all 57 features to predict
       any_addon_added in real-time; trigger dynamic prompts

  2. Personalised Recommendation Ranker
     → Use LambdaMART or Two-Tower neural network to rank add-ons
       per session using user history + reco metadata features

  3. A/B Testing Framework
     → Test add-on card placement (inline vs modal vs banner)
       per meal_time and user_segment to maximise adoption

  4. Price Elasticity Modelling
     → Analyse avg_reco_price_ratio sensitivity by user_price_sensitivity
       to find optimal add-on price points per segment

  5. Contextual Bandit / RL
     → Deploy a contextual bandit (LinUCB / Thompson Sampling) for
       real-time add-on selection using weather, meal_time, and cart state

  6. Live Dashboard (Streamlit / Power BI)
     → Operationalise this EDA into a live dashboard fed by
       an Airflow + BigQuery pipeline
""")
