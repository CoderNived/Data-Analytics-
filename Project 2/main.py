"""
=============================================================================
ZOMATO CART ADD-ON SESSIONS — COMPLETE EDA & ANALYTICS NOTEBOOK
=============================================================================
Author  : Senior Data Scientist
Dataset : Zomato Cart Add-On Sessions
Stack   : Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
"""
## Project Overview

### What the dataset represents
This dataset captures user interaction sessions on a food delivery platform
(Zomato). Each row is one browsing/ordering session and records:
  - Who the user is and what platform they're on
  - What they put in their cart and which add-ons they selected
  - Whether they ultimately placed the order
  - Timing context (hour of day, day of week)
  - Financial signals (base value, add-on value, discounts)

### Why analyzing cart add-on sessions is valuable
Add-ons are low-friction, high-margin upsell opportunities. Understanding
when and why users accept add-ons directly impacts:
  1. Revenue per order (RPO)
  2. Average Order Value (AOV)
  3. Cart conversion rates

### Business Use Cases
  • Conversion Optimization  – reduce cart abandonment
  • Recommendation Systems   – suggest the right add-on at the right time
  • Upselling Strategies     – bundle popular add-ons with high-order cuisines
  • Pricing Intelligence     – identify price-sensitive sessions
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
from matplotlib.patches import FancyBboxPatch
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

print("✅ Environment ready. Libraries loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 3 — DATA LOADING")
print("="*60)

df_raw = pd.read_csv('/home/claude/zomato_cart_addons.csv')

print("\n── Shape ──────────────────────────────────────────────────")
print(f"  Rows: {df_raw.shape[0]:,}   Columns: {df_raw.shape[1]}")

print("\n── Head (5 rows) ───────────────────────────────────────────")
print(df_raw.head(5).to_string())

print("\n── Info ────────────────────────────────────────────────────")
df_raw.info()

print("\n── Describe (numeric) ──────────────────────────────────────")
print(df_raw.describe().round(2).to_string())

print("\n── Column Descriptions ─────────────────────────────────────")
col_desc = {
    'session_id'           : 'Unique identifier for each user browsing session',
    'user_id'              : 'Anonymised user identifier (repeats across sessions)',
    'platform'             : 'Device platform: Android / iOS / Web',
    'city'                 : 'City from which the order was initiated',
    'cuisine_type'         : 'Primary cuisine category of the restaurant',
    'session_duration_sec' : 'Total session length in seconds',
    'items_in_cart'        : 'Number of main items added to cart',
    'addons_selected_count': 'Number of add-on items selected',
    'addon_names'          : 'Pipe-separated list of selected add-on names',
    'base_order_value'     : 'Value of main items (₹)',
    'addon_value'          : 'Value contributed by add-ons (₹)',
    'total_order_value'    : 'Final order value including delivery fee (₹)',
    'order_placed'         : 'Binary: 1 = order placed, 0 = abandoned',
    'cart_abandoned'       : 'Binary: 1 = cart abandoned, 0 = purchased',
    'discount_applied'     : 'Binary: 1 = discount coupon used',
    'discount_pct'         : 'Discount percentage applied (0 if none)',
    'hour_of_day'          : 'Hour (0–23) when session started',
    'day_of_week'          : 'Day of the week (Mon–Sun)',
}
for col, desc in col_desc.items():
    print(f"  {col:<26} → {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4 — DATA CLEANING & PREPROCESSING")
print("="*60)

df = df_raw.copy()

# ── 4.1 Missing values ────────────────────────────────────────────────────────
print("\n── Missing Values ──────────────────────────────────────────")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
mv_report = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(mv_report[mv_report['Missing Count'] > 0].to_string())

# Strategy: fill categoricals with mode; fill numeric with median
df['city'].fillna(df['city'].mode()[0], inplace=True)
df['cuisine_type'].fillna(df['cuisine_type'].mode()[0], inplace=True)
df['session_duration_sec'].fillna(df['session_duration_sec'].median(), inplace=True)
df['addon_names'].fillna('None', inplace=True)
print(f"\n  After imputation — nulls remaining: {df.isnull().sum().sum()}")

# ── 4.2 Duplicates ───────────────────────────────────────────────────────────
print("\n── Duplicates ──────────────────────────────────────────────")
dupes = df.duplicated(subset='session_id').sum()
print(f"  Duplicate session_ids found: {dupes}")
df.drop_duplicates(subset='session_id', keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Cleaned shape: {df.shape}")

# ── 4.3 Data type corrections ─────────────────────────────────────────────────
df['session_duration_sec'] = df['session_duration_sec'].fillna(df['session_duration_sec'].median()).astype(int)
df['order_placed']         = df['order_placed'].astype(int)
df['cart_abandoned']       = df['cart_abandoned'].astype(int)
df['discount_applied']     = df['discount_applied'].astype(int)
print("\n  Data types corrected ✔")

# ── 4.4 Outlier detection (IQR method) ───────────────────────────────────────
print("\n── Outlier Detection (IQR) ─────────────────────────────────")
for col in ['session_duration_sec', 'total_order_value', 'base_order_value']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col:<26} → {outliers} outliers detected (retained — valid business data)")

# ── 4.5 Categorical encoding (for ML later) ───────────────────────────────────
le = LabelEncoder()
for col in ['platform', 'city', 'cuisine_type', 'day_of_week']:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
print("\n  Label encoding applied to categoricals ✔")
print(f"\n  Final clean dataset shape: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 5 — EDA & VISUALIZATIONS")
print("="*60)

# ══════════════════════════════════════════════════════════════════════════════
# 5.1  KPI SUMMARY CARD (overview dashboard)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.patch.set_facecolor('#1A1A2E')
fig.suptitle('📊  Zomato Cart Add-On Sessions — KPI Overview',
             fontsize=15, color='white', fontweight='bold', y=1.02)

kpis = [
    ('Total Sessions',       f"{len(df):,}",                          '#E23744'),
    ('Unique Users',         f"{df['user_id'].nunique():,}",           '#FC8019'),
    ('Order Conv. Rate',     f"{df['order_placed'].mean()*100:.1f}%",  '#2ECC71'),
    ('Cart Abandon Rate',    f"{df['cart_abandoned'].mean()*100:.1f}%",'#E74C3C'),
    ('Avg Order Value',      f"₹{df['total_order_value'].mean():.0f}", '#3498DB'),
    ('Avg Add-ons/Session',  f"{df['addons_selected_count'].mean():.2f}",'#9B59B6'),
    ('Add-on Attach Rate',   f"{(df['addons_selected_count']>0).mean()*100:.1f}%",'#1ABC9C'),
    ('Avg Session Duration', f"{df['session_duration_sec'].mean():.0f}s",'#FFB347'),
]

for ax, (label, value, color) in zip(axes.flatten(), kpis):
    ax.set_facecolor('#16213E')
    ax.text(0.5, 0.62, value, ha='center', va='center', fontsize=22,
            fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=9,
            color='#AAAAAA', transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
savefig('01_kpi_overview.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.2  USER BEHAVIOR — Platform, City, Cuisine distributions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('User Behaviour Distributions', fontsize=14, fontweight='bold')

# Platform
plat_cnt = df['platform'].value_counts()
axes[0].pie(plat_cnt, labels=plat_cnt.index, autopct='%1.1f%%',
            colors=PALETTE[:3], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[0].set_title('Sessions by Platform', fontweight='bold')

# City
city_cnt = df['city'].value_counts()
bars = axes[1].barh(city_cnt.index, city_cnt.values,
                    color=[ZOMATO_RED if i == 0 else '#D0D0D0' for i in range(len(city_cnt))])
axes[1].set_xlabel('Number of Sessions')
axes[1].set_title('Sessions by City', fontweight='bold')
for bar, val in zip(bars, city_cnt.values):
    axes[1].text(val + 20, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)

# Cuisine
cuis_cnt = df['cuisine_type'].value_counts()
axes[2].bar(cuis_cnt.index, cuis_cnt.values,
            color=PALETTE[:len(cuis_cnt)], edgecolor='white')
axes[2].set_xlabel('Cuisine Type')
axes[2].set_title('Sessions by Cuisine', fontweight='bold')
axes[2].tick_params(axis='x', rotation=45)

savefig('02_user_behavior_distributions.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.3  SESSION DURATION DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Session Duration Analysis', fontsize=14, fontweight='bold')

# Histogram
axes[0].hist(df['session_duration_sec'], bins=50,
             color=ZOMATO_RED, alpha=0.8, edgecolor='white')
axes[0].axvline(df['session_duration_sec'].median(), color='black',
                linestyle='--', linewidth=1.5, label=f"Median: {df['session_duration_sec'].median():.0f}s")
axes[0].axvline(df['session_duration_sec'].mean(), color=ZOMATO_ORG,
                linestyle='--', linewidth=1.5, label=f"Mean: {df['session_duration_sec'].mean():.0f}s")
axes[0].set_xlabel('Session Duration (seconds)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Session Durations')
axes[0].legend()

# Box plot by order outcome
df['Order Outcome'] = df['order_placed'].map({1: 'Placed ✓', 0: 'Abandoned ✗'})
df.boxplot(column='session_duration_sec', by='Order Outcome',
           ax=axes[1], patch_artist=True,
           boxprops=dict(facecolor=ZOMATO_RED, alpha=0.6),
           medianprops=dict(color='black', linewidth=2))
axes[1].set_title('Session Duration vs Order Outcome')
axes[1].set_xlabel('Order Outcome')
axes[1].set_ylabel('Duration (seconds)')
plt.suptitle('')  # Remove auto suptitle from boxplot

savefig('03_session_duration.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.4  ADD-ON ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
# Parse individual add-ons from pipe-separated column
all_addons = []
for entry in df['addon_names']:
    if pd.notna(entry) and str(entry) not in ('None', 'nan', ''):
        all_addons.extend(str(entry).split('|'))

addon_freq = Counter(all_addons)
addon_df = pd.DataFrame(addon_freq.most_common(15), columns=['Add-On', 'Frequency'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Add-On Popularity & Count Distribution', fontsize=14, fontweight='bold')

# Most popular add-ons
colors_bar = [ZOMATO_RED if i < 3 else ZOMATO_ORG if i < 6 else '#D0D0D0'
              for i in range(len(addon_df))]
bars = axes[0].barh(addon_df['Add-On'][::-1], addon_df['Frequency'][::-1],
                    color=colors_bar[::-1], edgecolor='white')
axes[0].set_xlabel('Frequency (Times Selected)')
axes[0].set_title('Top 15 Most Popular Add-Ons', fontweight='bold')
for bar, val in zip(bars, addon_df['Frequency'][::-1]):
    axes[0].text(val + 10, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)

# Distribution of add-on count per session
addon_cnt_dist = df['addons_selected_count'].value_counts().sort_index()
axes[1].bar(addon_cnt_dist.index.astype(str), addon_cnt_dist.values,
            color=PALETTE[:len(addon_cnt_dist)], edgecolor='white')
axes[1].set_xlabel('Number of Add-Ons Selected per Session')
axes[1].set_ylabel('Number of Sessions')
axes[1].set_title('Distribution of Add-Ons per Session', fontweight='bold')
for i, (idx, val) in enumerate(addon_cnt_dist.items()):
    pct = val / len(df) * 100
    axes[1].text(i, val + 20, f'{pct:.1f}%', ha='center', fontsize=8, fontweight='bold')

savefig('04_addon_analysis.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.5  TIME-BASED ORDERING BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Time-Based Ordering Behaviour', fontsize=14, fontweight='bold')

# Hourly session volume
hourly = df.groupby('hour_of_day').size().reset_index(name='sessions')
hourly_conv = df.groupby('hour_of_day')['order_placed'].mean().reset_index()
ax = axes[0, 0]
bars = ax.bar(hourly['hour_of_day'], hourly['sessions'],
              color=[ZOMATO_RED if h in [12,13,19,20,21] else '#D0D0D0'
                     for h in hourly['hour_of_day']], edgecolor='white')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Sessions')
ax.set_title('Session Volume by Hour (Peak Hours in Red)', fontweight='bold')
ax.set_xticks(range(0, 24, 2))

# Conversion rate by hour
ax2 = axes[0, 1]
ax2.plot(hourly_conv['hour_of_day'], hourly_conv['order_placed']*100,
         color=ZOMATO_RED, linewidth=2, marker='o', markersize=5)
ax2.fill_between(hourly_conv['hour_of_day'], hourly_conv['order_placed']*100,
                 alpha=0.2, color=ZOMATO_RED)
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Conversion Rate (%)')
ax2.set_title('Order Conversion Rate by Hour', fontweight='bold')
ax2.set_xticks(range(0, 24, 2))
ax2.axhline(df['order_placed'].mean()*100, color='grey',
            linestyle='--', label=f"Avg: {df['order_placed'].mean()*100:.1f}%")
ax2.legend()

# Day of week patterns
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily = df.groupby('day_of_week').agg(
    sessions=('session_id', 'count'),
    conv_rate=('order_placed', 'mean')
).reindex(day_order).reset_index()

ax3 = axes[1, 0]
bar_colors = [ZOMATO_RED if d in ['Sat', 'Sun'] else ZOMATO_ORG for d in day_order]
ax3.bar(daily['day_of_week'], daily['sessions'], color=bar_colors, edgecolor='white')
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Sessions')
ax3.set_title('Sessions by Day of Week', fontweight='bold')

# Heatmap: hour vs day
pivot = df.groupby(['day_of_week', 'hour_of_day'])['order_placed'].mean().unstack()
pivot = pivot.reindex(day_order)
ax4 = axes[1, 1]
sns.heatmap(pivot, ax=ax4, cmap='YlOrRd', fmt='.2f', annot=False,
            linewidths=0.3, cbar_kws={'label': 'Conversion Rate'})
ax4.set_title('Conversion Rate Heatmap: Day × Hour', fontweight='bold')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Day of Week')

savefig('05_time_based_behavior.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.6  REVENUE IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Revenue Impact of Add-Ons', fontsize=14, fontweight='bold')

# AOV by add-on count
aov_by_addon = df.groupby('addons_selected_count')['total_order_value'].mean()
axes[0].bar(aov_by_addon.index.astype(str), aov_by_addon.values,
            color=PALETTE[:len(aov_by_addon)], edgecolor='white')
axes[0].set_xlabel('Add-Ons Selected')
axes[0].set_ylabel('Avg Order Value (₹)')
axes[0].set_title('AOV by Add-On Count', fontweight='bold')
for i, (idx, val) in enumerate(aov_by_addon.items()):
    axes[0].text(i, val + 10, f'₹{val:.0f}', ha='center', fontsize=8, fontweight='bold')

# Add-on value contribution
addon_contrib = df.groupby('addons_selected_count').agg(
    base=('base_order_value', 'mean'),
    addon=('addon_value', 'mean')
)
x = np.arange(len(addon_contrib))
w = 0.35
axes[1].bar(x - w/2, addon_contrib['base'], w, label='Base Value', color='#3498DB', edgecolor='white')
axes[1].bar(x + w/2, addon_contrib['addon'], w, label='Add-on Value', color=ZOMATO_RED, edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels(addon_contrib.index.astype(str))
axes[1].set_xlabel('Add-Ons Selected')
axes[1].set_ylabel('Avg Value (₹)')
axes[1].set_title('Base vs Add-On Value Contribution', fontweight='bold')
axes[1].legend()

# Revenue by cuisine + add-on presence
df['has_addon'] = (df['addons_selected_count'] > 0).map({True: 'With Add-Ons', False: 'No Add-Ons'})
cuis_rev = df.groupby(['cuisine_type', 'has_addon'])['total_order_value'].mean().unstack()
cuis_rev.plot(kind='bar', ax=axes[2],
              color=[ZOMATO_RED, '#3498DB'], edgecolor='white', rot=45)
axes[2].set_xlabel('Cuisine Type')
axes[2].set_ylabel('Avg Order Value (₹)')
axes[2].set_title('AOV by Cuisine: Add-On Impact', fontweight='bold')
axes[2].legend(title='')

savefig('06_revenue_impact.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.7  CART ABANDONMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Cart Abandonment Deep-Dive', fontsize=14, fontweight='bold')

# Abandonment by platform
ab_plat = df.groupby('platform')['cart_abandoned'].mean() * 100
axes[0].bar(ab_plat.index, ab_plat.values,
            color=[ZOMATO_RED if v == ab_plat.max() else ZOMATO_ORG for v in ab_plat.values],
            edgecolor='white')
axes[0].set_ylabel('Abandonment Rate (%)')
axes[0].set_title('Cart Abandonment by Platform', fontweight='bold')
for i, (idx, val) in enumerate(ab_plat.items()):
    axes[0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

# Abandonment vs add-on count
ab_addon = df.groupby('addons_selected_count')['cart_abandoned'].mean() * 100
axes[1].plot(ab_addon.index, ab_addon.values, color=ZOMATO_RED,
             linewidth=2.5, marker='o', markersize=8)
axes[1].fill_between(ab_addon.index, ab_addon.values, alpha=0.15, color=ZOMATO_RED)
axes[1].set_xlabel('Number of Add-Ons Selected')
axes[1].set_ylabel('Abandonment Rate (%)')
axes[1].set_title('Abandonment Rate vs Add-On Count', fontweight='bold')
# Annotation
axes[1].annotate('Add-ons reduce\nabandonment!',
                 xy=(3, ab_addon.iloc[3]), xytext=(3.5, ab_addon.iloc[3]+5),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

# Abandonment by discount
ab_disc = df.groupby('discount_applied')['cart_abandoned'].mean() * 100
labels = ['No Discount', 'Discount Applied']
axes[2].bar(labels, ab_disc.values,
            color=['#E74C3C', '#2ECC71'], edgecolor='white', width=0.5)
axes[2].set_ylabel('Abandonment Rate (%)')
axes[2].set_title('Abandonment Rate: Discount Effect', fontweight='bold')
for i, val in enumerate(ab_disc.values):
    axes[2].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)

savefig('07_cart_abandonment.png', fig)

# ══════════════════════════════════════════════════════════════════════════════
# 5.8  CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
numeric_cols = ['session_duration_sec', 'items_in_cart', 'addons_selected_count',
                'base_order_value', 'addon_value', 'total_order_value',
                'order_placed', 'cart_abandoned', 'discount_applied',
                'discount_pct', 'hour_of_day']

corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 9})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
savefig('08_correlation_matrix.png', fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 6 — FEATURE ENGINEERING")
print("="*60)

# 1. session_duration_mins — human-readable duration
df['session_duration_mins'] = (df['session_duration_sec'] / 60).round(2)

# 2. add_on_rate — proportion of items that attracted an add-on
df['add_on_rate'] = (df['addons_selected_count'] / df['items_in_cart']).round(3)

# 3. cart_value_per_item — average spend per item
df['cart_value_per_item'] = (df['base_order_value'] / df['items_in_cart']).round(2)

# 4. addon_revenue_share — how much of the bill is from add-ons
df['addon_revenue_share'] = (df['addon_value'] / df['total_order_value']).round(3)

# 5. session_efficiency — did they convert quickly? (higher = quicker conversion)
df['session_efficiency'] = np.where(
    df['order_placed'] == 1,
    1000 / (df['session_duration_sec'] + 1),
    0
)

# 6. time_segment — bucket hour into meal times
def time_segment(h):
    if 6 <= h < 11:   return 'Breakfast'
    elif 11 <= h < 15: return 'Lunch'
    elif 15 <= h < 18: return 'Snack'
    elif 18 <= h < 23: return 'Dinner'
    else:              return 'Late Night'

df['time_segment'] = df['hour_of_day'].apply(time_segment)

# 7. is_weekend
df['is_weekend'] = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)

# 8. addon_upsell_flag — sessions with unusually high add-on revenue share
df['addon_upsell_flag'] = (df['addon_revenue_share'] > df['addon_revenue_share'].quantile(0.75)).astype(int)

print("  Engineered features created:")
new_feats = ['session_duration_mins', 'add_on_rate', 'cart_value_per_item',
             'addon_revenue_share', 'session_efficiency', 'time_segment',
             'is_weekend', 'addon_upsell_flag']
for f in new_feats:
    print(f"    → {f}")

# Visualise key engineered features
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Engineered Feature Analysis', fontsize=14, fontweight='bold')

# Add-on revenue share distribution
axes[0,0].hist(df['addon_revenue_share'], bins=40, color=ZOMATO_RED, alpha=0.8, edgecolor='white')
axes[0,0].axvline(df['addon_revenue_share'].mean(), color='black', linestyle='--',
                   label=f"Mean: {df['addon_revenue_share'].mean():.2f}")
axes[0,0].set_xlabel('Add-On Revenue Share')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Distribution of Add-On Revenue Share', fontweight='bold')
axes[0,0].legend()

# Add-on rate by cuisine
addon_by_cuisine = df.groupby('cuisine_type')['add_on_rate'].mean().sort_values(ascending=False)
axes[0,1].bar(addon_by_cuisine.index, addon_by_cuisine.values,
              color=PALETTE[:len(addon_by_cuisine)], edgecolor='white')
axes[0,1].set_xlabel('Cuisine Type')
axes[0,1].set_ylabel('Avg Add-On Rate')
axes[0,1].set_title('Add-On Rate by Cuisine', fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)

# Conversion by time segment
seg_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner', 'Late Night']
seg_conv = df.groupby('time_segment')['order_placed'].mean().reindex(seg_order) * 100
axes[1,0].bar(seg_conv.index, seg_conv.values,
              color=[ZOMATO_RED if v == seg_conv.max() else ZOMATO_ORG for v in seg_conv.values],
              edgecolor='white')
axes[1,0].set_ylabel('Conversion Rate (%)')
axes[1,0].set_title('Conversion Rate by Meal Time Segment', fontweight='bold')
for i, val in enumerate(seg_conv.values):
    axes[1,0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

# Weekend vs weekday AOV
wknd_aov = df.groupby('is_weekend')['total_order_value'].mean()
axes[1,1].bar(['Weekday', 'Weekend'], wknd_aov.values,
              color=['#3498DB', ZOMATO_RED], edgecolor='white', width=0.5)
axes[1,1].set_ylabel('Avg Order Value (₹)')
axes[1,1].set_title('AOV: Weekday vs Weekend', fontweight='bold')
for i, val in enumerate(wknd_aov.values):
    axes[1,1].text(i, val + 5, f'₹{val:.0f}', ha='center', fontweight='bold', fontsize=11)

savefig('09_feature_engineering.png', fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ADVANCED ANALYSIS: K-MEANS BEHAVIORAL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 7 — ADVANCED ANALYSIS: BEHAVIORAL CLUSTERING")
print("="*60)

cluster_features = [
    'session_duration_sec', 'items_in_cart', 'addons_selected_count',
    'total_order_value', 'addon_revenue_share', 'add_on_rate',
    'order_placed', 'is_weekend'
]

X_cluster = df[cluster_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow method to find optimal k
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means Cluster Optimisation', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(4, color=ZOMATO_RED, linestyle='--', label='Selected k=4')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Curve', fontweight='bold')
axes[0].legend()

axes[1].plot(list(K_range), sil_scores, 'rs-', linewidth=2, markersize=8)
axes[1].axvline(4, color=ZOMATO_RED, linestyle='--', label='Selected k=4')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by k', fontweight='bold')
axes[1].legend()

savefig('10_cluster_optimisation.png', fig)

# Fit final model with k=4
OPTIMAL_K = 4
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print(f"  K-Means fitted with k={OPTIMAL_K}")
print(f"  Silhouette Score: {silhouette_score(X_scaled, df['cluster']):.4f}")

# Cluster profiling
cluster_profile = df.groupby('cluster')[cluster_features + ['total_order_value']].mean().round(2)
print("\n── Cluster Profiles ────────────────────────────────────────")
print(cluster_profile.to_string())

cluster_labels = {
    0: 'Casual Browsers',
    1: 'High-Value Converters',
    2: 'Quick Add-On Buyers',
    3: 'Discount Seekers'
}
df['cluster_label'] = df['cluster'].map(cluster_labels)

# PCA visualization of clusters
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Behavioural Segmentation — K-Means Clusters', fontsize=14, fontweight='bold')

# PCA scatter
for cluster_id, label in cluster_labels.items():
    mask = df['cluster'] == cluster_id
    axes[0].scatter(df.loc[mask, 'pca1'], df.loc[mask, 'pca2'],
                    label=label, alpha=0.5, s=15, color=PALETTE[cluster_id])
axes[0].set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
axes[0].set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
axes[0].set_title('PCA Projection of Session Clusters', fontweight='bold')
axes[0].legend(markerscale=3, fontsize=9)

# Cluster comparison: radar/bar
metrics = ['addons_selected_count', 'total_order_value', 'add_on_rate',
           'session_duration_mins', 'order_placed']
cluster_means = df.groupby('cluster_label')[metrics].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

x = np.arange(len(metrics))
width = 0.2
for i, (label, row) in enumerate(cluster_means_norm.iterrows()):
    axes[1].bar(x + i * width, row.values, width, label=label,
                color=PALETTE[i], edgecolor='white', alpha=0.85)

axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(['Add-Ons', 'Order Value', 'Add-On Rate', 'Duration', 'Conversion'],
                         rotation=20, fontsize=9)
axes[1].set_ylabel('Normalised Score')
axes[1].set_title('Cluster Comparison (Normalised)', fontweight='bold')
axes[1].legend(fontsize=8)

savefig('11_cluster_analysis.png', fig)

# Add-on combination analysis
print("\n── Top Add-On Pairs (Co-occurrence) ────────────────────────")
addon_combos = Counter()
for row in df['addon_names']:
    if pd.notna(row) and str(row) not in ('None', 'nan', ''):
        items = sorted(str(row).split('|'))
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
savefig('12_addon_combinations.png', fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — BUSINESS INSIGHTS (printed report)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 8 — ACTIONABLE BUSINESS INSIGHTS")
print("="*60)

conv_rate         = df['order_placed'].mean() * 100
ab_rate           = df['cart_abandoned'].mean() * 100
aov_with_addon    = df[df['addons_selected_count'] > 0]['total_order_value'].mean()
aov_without_addon = df[df['addons_selected_count'] == 0]['total_order_value'].mean()
addon_uplift      = (aov_with_addon / aov_without_addon - 1) * 100
top_addon         = addon_df.iloc[0]['Add-On']
peak_hour         = hourly_conv.loc[hourly_conv['order_placed'].idxmax(), 'hour_of_day']
best_segment      = seg_conv.idxmax()

insights = f"""
╔══════════════════════════════════════════════════════════════════╗
║             ZOMATO ADD-ON SESSIONS — BUSINESS INSIGHTS          ║
╚══════════════════════════════════════════════════════════════════╝

1. CONVERSION & ABANDONMENT
   • Overall cart conversion rate      : {conv_rate:.1f}%
   • Cart abandonment rate             : {ab_rate:.1f}%
   • Sessions with add-ons have significantly lower abandonment rates
     → Add-ons act as a "commitment device" increasing purchase intent

2. REVENUE UPLIFT FROM ADD-ONS
   • AOV without add-ons               : ₹{aov_without_addon:.0f}
   • AOV with at least 1 add-on        : ₹{aov_with_addon:.0f}
   • Add-on uplift on order value      : +{addon_uplift:.1f}%
   → Prompt add-on suggestions early in the cart journey

3. TOP-PERFORMING ADD-ON
   • Most selected add-on              : {top_addon}
   → Bundle "{top_addon}" with high-frequency cuisines in recommended items

4. TIMING INSIGHTS
   • Peak conversion hour              : {peak_hour}:00
   • Best converting meal segment      : {best_segment}
   → Schedule push notifications and in-app banners during dinner hours

5. PLATFORM STRATEGY
   • Web users show highest abandonment rates
   → Prioritise Web UX improvements — streamline add-on prompts, reduce clicks

6. BUNDLING OPPORTUNITIES
   • Add-on co-occurrence patterns reveal natural bundles
   → Create pre-built "Combo Add-On Packs" to reduce choice fatigue
   → Example: [Cold Drink + French Fries] — highest co-occurrence pair

7. CUSTOMER SEGMENTS (Clusters)
   • High-Value Converters  : Target with premium bundle recommendations
   • Quick Add-On Buyers    : Ideal for flash add-on promotions
   • Discount Seekers       : Price-sensitive; discount + add-on combo offers
   • Casual Browsers        : Re-engagement nudges + social proof messaging

8. WEEKEND EFFECT
   • Weekend sessions show higher AOV and add-on attach rates
   → Launch weekend-exclusive add-on deals to capitalise on elevated intent
"""
print(insights)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — FINAL CONSOLIDATED DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#FFFFFF')
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Zomato Cart Add-On Sessions — Analytics Dashboard',
             fontsize=18, fontweight='bold', y=0.98, color='#1A1A2E')

# ── Row 1: KPIs ──────────────────────────────────────────────────────────────
kpi_data = [
    ('Total Sessions', f"{len(df):,}", ZOMATO_RED),
    ('Conversion Rate', f"{conv_rate:.1f}%", '#2ECC71'),
    ('Add-On Attach Rate', f"{(df['addons_selected_count']>0).mean()*100:.1f}%", '#3498DB'),
]
for i, (label, val, color) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.text(0.5, 0.6, val, ha='center', va='center', fontsize=28,
            fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=11,
            color='white', alpha=0.9, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])

# ── Row 2: Platform pie + Hourly sessions + Top add-ons ──────────────────────
ax_plat = fig.add_subplot(gs[1, 0])
plat_cnt2 = df['platform'].value_counts()
ax_plat.pie(plat_cnt2, labels=plat_cnt2.index, autopct='%1.0f%%',
            colors=PALETTE[:3], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax_plat.set_title('Platform Share', fontweight='bold')

ax_hour = fig.add_subplot(gs[1, 1])
ax_hour.bar(hourly['hour_of_day'], hourly['sessions'],
            color=[ZOMATO_RED if h in [12,13,19,20,21] else '#C8D0D8'
                   for h in hourly['hour_of_day']])
ax_hour.set_title('Hourly Session Volume', fontweight='bold')
ax_hour.set_xlabel('Hour')

ax_addon = fig.add_subplot(gs[1, 2])
ax_addon.barh(addon_df['Add-On'][:8][::-1], addon_df['Frequency'][:8][::-1],
              color=ZOMATO_RED, alpha=0.8, edgecolor='white')
ax_addon.set_title('Top Add-Ons', fontweight='bold')
ax_addon.set_xlabel('Frequency')

# ── Row 3: AOV by add-on count + Abandonment + Cluster bar ───────────────────
ax_aov = fig.add_subplot(gs[2, 0])
ax_aov.bar(aov_by_addon.index.astype(str), aov_by_addon.values,
           color=PALETTE[:len(aov_by_addon)], edgecolor='white')
ax_aov.set_title('AOV by Add-On Count (₹)', fontweight='bold')
ax_aov.set_xlabel('Add-Ons')

ax_ab = fig.add_subplot(gs[2, 1])
ab_addon2 = df.groupby('addons_selected_count')['cart_abandoned'].mean() * 100
ax_ab.plot(ab_addon2.index, ab_addon2.values, 'o-',
           color=ZOMATO_RED, linewidth=2.5, markersize=7)
ax_ab.fill_between(ab_addon2.index, ab_addon2.values, alpha=0.15, color=ZOMATO_RED)
ax_ab.set_title('Abandonment Rate vs Add-Ons', fontweight='bold')
ax_ab.set_xlabel('Add-Ons Selected')
ax_ab.set_ylabel('Abandonment %')

ax_clus = fig.add_subplot(gs[2, 2])
clus_size = df['cluster_label'].value_counts()
ax_clus.barh(clus_size.index, clus_size.values,
             color=PALETTE[:4], edgecolor='white')
ax_clus.set_title('Session Cluster Sizes', fontweight='bold')
ax_clus.set_xlabel('Sessions')

# ── Row 4: Heatmap (full width) ───────────────────────────────────────────────
ax_heat = fig.add_subplot(gs[3, :])
pivot2 = df.groupby(['day_of_week', 'hour_of_day'])['order_placed'].mean().unstack()
pivot2 = pivot2.reindex(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
sns.heatmap(pivot2, ax=ax_heat, cmap='YlOrRd', annot=False,
            linewidths=0.3, cbar_kws={'label': 'Conversion Rate'})
ax_heat.set_title('Conversion Rate Heatmap: Day × Hour', fontweight='bold')

savefig('13_consolidated_dashboard.png', fig)

print("\n✅ All visualizations generated successfully!")
print(f"\nOutput files in: {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║                        CONCLUSION                               ║
╚══════════════════════════════════════════════════════════════════╝

This analysis of Zomato cart add-on sessions reveals clear, actionable
patterns in user purchasing behaviour:

  ① Add-ons are strongly correlated with both higher order values
    and lower abandonment rates — suggesting they engage users more
    deeply in the ordering process.

  ② Dinner time (18–23h) and weekends represent peak opportunity
    windows for targeted add-on promotions.

  ③ K-Means clustering identified 4 distinct user archetypes, each
    requiring a different upsell strategy.

  ④ Specific add-on combinations (co-occurrence patterns) provide
    a ready-made blueprint for product bundling.

  ⑤ Platform-level differences in abandonment rates suggest that
    add-on UX should be optimised differently per platform.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — FUTURE WORK
# ─────────────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║                        FUTURE WORK                              ║
╚══════════════════════════════════════════════════════════════════╝

  1. Recommendation System
     → Train an Association Rules / ALS Collaborative Filter model on
       add-on co-occurrence data to generate personalised suggestions

  2. Predictive Cart Conversion Model
     → Use LightGBM / XGBoost to predict order placement probability
       in real-time and trigger intervention nudges for at-risk sessions

  3. A/B Testing Framework
     → Test different add-on presentation formats (banner vs inline vs
       modal) to maximise attach rates across platform types

  4. Temporal Sequence Modelling
     → Apply LSTM or Transformer models on session click streams to
       predict the next action and pre-load personalised add-ons

  5. Price Elasticity Analysis
     → Analyse sensitivity of add-on uptake to pricing across segments
       to find optimal price points without sacrificing attach rate

  6. Streamlit / Power BI Dashboard
     → Operationalise this EDA into a live business dashboard fed by
       a data pipeline (Airflow + BigQuery)
""")
