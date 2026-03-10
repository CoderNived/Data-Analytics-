"""
============================================================
  EUROPEAN EMPLOYMENT DATA ANALYSIS — occupazione.csv
  Senior Data Engineer & Data Scientist Report
  Dataset: Employment Rates by Country, Gender, Age (2015–2024)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Aesthetic defaults ──────────────────────────────────────
PALETTE   = sns.color_palette("tab10")
BLUE      = "#2563EB"
RED       = "#DC2626"
GREEN     = "#16A34A"
ORANGE    = "#D97706"
GRAY      = "#6B7280"
sns.set_theme(style="whitegrid", font_scale=1.05)

# ==============================================================
# 1. DATA LOADING & UNDERSTANDING
# ==============================================================
print("=" * 60)
print("1. DATA LOADING & UNDERSTANDING")
print("=" * 60)

df = pd.read_csv("occupazione.csv")
year_cols = [str(y) for y in range(2015, 2025)]

print(f"\nShape      : {df.shape}")
print(f"Columns    : {df.columns.tolist()}")
print(f"\nSEX values : {sorted(df['SEX'].unique().tolist())}")
print(f"AGE groups : {sorted(df['AGE'].unique().tolist())}")
print(f"Countries  : {len(df['ISO'].unique())} — {sorted(df['ISO'].unique().tolist())}")
print("\nFirst 5 rows:")
print(df.head())

# ==============================================================
# 2. DATA CLEANING & PREPROCESSING
# ==============================================================
print("\n" + "=" * 60)
print("2. DATA CLEANING & PREPROCESSING")
print("=" * 60)

# ── 2a. Missing value audit ──────────────────────────────────
print("\n[NaN per year column]")
print(df[year_cols].isnull().sum())

nan_iso = (df.set_index(['SEX', 'AGE', 'ISO'])[year_cols]
             .isnull().any(axis=1)
             .groupby('ISO').sum())
print("\n[Countries with any missing data]")
print(nan_iso[nan_iso > 0])

# ── 2b. Convert to long (tidy) format ───────────────────────
df_long = (df
    .melt(id_vars=['SEX', 'AGE', 'ISO'],
          value_vars=year_cols,
          var_name='Year',
          value_name='Rate')
    .assign(Year=lambda x: x['Year'].astype(int),
            Rate=lambda x: pd.to_numeric(x['Rate'], errors='coerce'))
    .rename(columns=str.upper)
)
# Standardise column capitalisation for convenience
df_long.columns = ['SEX', 'AGE', 'ISO', 'YEAR', 'RATE']

print(f"\nLong-format shape : {df_long.shape}")
print(f"Total NaN in RATE : {df_long['RATE'].isna().sum()} "
      f"({df_long['RATE'].isna().mean()*100:.1f}%)")

# ── 2c. Duplicate check ──────────────────────────────────────
dupes = df_long.duplicated(subset=['SEX', 'AGE', 'ISO', 'YEAR']).sum()
print(f"Duplicate rows    : {dupes}")

# ── 2d. Helper filters ───────────────────────────────────────
def age_filter(age: str):
    return df_long[df_long['AGE'] == age]

emp_1564  = age_filter('15-64')   # Headline employment metric
emp_1524  = age_filter('15-24')   # Youth
emp_5564  = age_filter('55-64')   # Senior
emp_2554  = age_filter('25-54')   # Prime age

# ==============================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================
print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 3a. Descriptive statistics ───────────────────────────────
print("\n[Employment rate stats — all age groups]")
print(df_long.groupby('AGE')['RATE'].describe().round(2))

print("\n[Employment rate stats — by gender (15-64)]")
print(emp_1564.groupby('SEX')['RATE'].describe().round(2))

# ── 3b. Year-over-year trend (15-64) ────────────────────────
yoy = emp_1564.groupby('YEAR')['RATE'].mean()
yoy_chg = yoy.diff().rename('YoY Change')
yoy_df = pd.DataFrame({'Mean Rate': yoy.round(2), 'YoY Change': yoy_chg.round(2)})
print("\n[YoY employment trend (15-64 aggregate)]")
print(yoy_df)

# ── 3c. COVID shock quantification ──────────────────────────
covid_pivot = (emp_1564
    .groupby(['ISO', 'YEAR'])['RATE'].mean()
    .unstack())
covid_drop = (covid_pivot[2020] - covid_pivot[2019]).dropna().sort_values()
print("\n[Top 8 COVID-impacted countries — 2019→2020 drop (pp)]")
print(covid_drop.head(8).round(2))
print("\n[Most resilient countries — 2019→2020]")
print(covid_drop.tail(5).round(2))

# ── 3d. Recovery: 2020→2024 growth ──────────────────────────
recovery = (covid_pivot[2024] - covid_pivot[2020]).dropna().sort_values(ascending=False)
print("\n[Recovery gain 2020→2024 (pp)]")
print(recovery.head(8).round(2))

# ── 3e. Overall growth 2015→2024 ────────────────────────────
total_growth = (covid_pivot[2024] - covid_pivot[2015]).dropna().sort_values(ascending=False)
print("\n[Total growth 2015→2024 (pp)]")
print(total_growth.round(2))

# ── 3f. Senior employment surge (55-64) ─────────────────────
print("\n[Senior (55-64) employment trend]")
print(emp_5564.groupby('YEAR')['RATE'].mean().round(2))

# ==============================================================
# 4. GENDER ANALYSIS
# ==============================================================
print("\n" + "=" * 60)
print("4. GENDER ANALYSIS")
print("=" * 60)

gender_trend = (emp_1564
    .groupby(['YEAR', 'SEX'])['RATE'].mean()
    .unstack()
    .assign(gap=lambda x: x['M'] - x['F']))
print("\n[Gender employment rates and M-F gap (15-64)]")
print(gender_trend.round(2))

gap_2024 = (emp_1564[emp_1564['YEAR'] == 2024]
    .groupby(['ISO', 'SEX'])['RATE'].mean()
    .unstack()
    .assign(gap=lambda x: (x['M'] - x['F']).round(2))
    .sort_values('gap', ascending=False))
print("\n[Gender gap by country (2024)]")
print(gap_2024.round(2))

# Correlation M↔F
corr_mf = gap_2024['M'].corr(gap_2024['F'])
print(f"\nCorrelation M vs F employment (2024): {corr_mf:.3f}")

# ==============================================================
# 5. COUNTRY RANKING (2024, 15-64)
# ==============================================================
print("\n" + "=" * 60)
print("5. COUNTRY RANKINGS — 2024")
print("=" * 60)

rank_2024 = (emp_1564[emp_1564['YEAR'] == 2024]
    .groupby('ISO')['RATE'].mean()
    .dropna()
    .sort_values(ascending=False))
print("\n[Full country ranking — mean employment rate 2024]")
print(rank_2024.round(2))

print(f"\nTOP 5    : {rank_2024.head(5).index.tolist()}")
print(f"BOTTOM 5 : {rank_2024.tail(5).index.tolist()}")

# ==============================================================
# 6. AGE-GROUP DEEP DIVE
# ==============================================================
print("\n" + "=" * 60)
print("6. AGE-GROUP DEEP DIVE")
print("=" * 60)

youth_trend = (emp_1524
    .groupby(['YEAR', 'SEX'])['RATE'].mean()
    .unstack())
print("\n[Youth (15-24) employment by year & gender]")
print(youth_trend.round(2))

prime_2024 = (emp_2554[emp_2554['YEAR'] == 2024]
    .groupby('ISO')['RATE'].mean()
    .dropna().sort_values(ascending=False))
print("\n[Prime age (25-54) employment ranking — 2024]")
print(prime_2024.round(2))

# ==============================================================
# 7. VISUALISATIONS (saved to disk)
# ==============================================================
print("\n" + "=" * 60)
print("7. GENERATING VISUALISATIONS …")
print("=" * 60)

# ── FIG 1: Headline trend + gender ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: aggregate trend
ax = axes[0]
ax.plot(yoy.index, yoy.values, color=BLUE, lw=2.5, marker='o', markersize=5)
ax.axvspan(2019.5, 2020.5, color='red', alpha=0.12, label='COVID-19 (2020)')
ax.set_title("Mean Employment Rate (15-64)\nAll Countries, 2015–2024", fontweight='bold')
ax.set_xlabel("Year")
ax.set_ylabel("Employment Rate (%)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax.legend()

# Right: gender split
ax = axes[1]
for sex, color, lbl in [('F', RED, 'Female'), ('M', BLUE, 'Male')]:
    s = emp_1564.groupby(['YEAR', 'SEX'])['RATE'].mean().unstack()[sex]
    ax.plot(s.index, s.values, color=color, lw=2.5, marker='o', markersize=5, label=lbl)
ax.axvspan(2019.5, 2020.5, color='gray', alpha=0.12, label='COVID-19')
ax.set_title("Employment Rate by Gender (15-64)\n2015–2024", fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Employment Rate (%)")
ax.legend()

plt.tight_layout()
plt.savefig("fig1_headline_trend.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig1_headline_trend.png")

# ── FIG 2: Age-group trends ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
age_order = ['15-24', '15-29', '25-54', '55-64', '15-64', '20-64']
colors    = sns.color_palette("tab10", len(age_order))
for age, color in zip(age_order, colors):
    s = df_long[df_long['AGE'] == age].groupby('YEAR')['RATE'].mean()
    ax.plot(s.index, s.values, lw=2.2, marker='o', markersize=4, label=age, color=color)
ax.axvspan(2019.5, 2020.5, color='red', alpha=0.1, label='COVID-19')
ax.set_title("Employment Rate by Age Group — 2015–2024", fontweight='bold', fontsize=14)
ax.set_xlabel("Year"); ax.set_ylabel("Employment Rate (%)")
ax.legend(title="Age Group", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig("fig2_age_trends.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig2_age_trends.png")

# ── FIG 3: Country ranking bar chart (2024, 15-64) ───────────
fig, ax = plt.subplots(figsize=(12, 8))
colors_bar = [GREEN if r >= 72 else (ORANGE if r >= 62 else RED) for r in rank_2024.values]
ax.barh(rank_2024.index[::-1], rank_2024.values[::-1], color=colors_bar[::-1], edgecolor='white')
ax.axvline(rank_2024.mean(), color='navy', lw=1.5, linestyle='--', label=f'Mean={rank_2024.mean():.1f}%')
ax.set_title("Employment Rate by Country — 2024 (15-64, avg M+F)", fontweight='bold', fontsize=13)
ax.set_xlabel("Employment Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig("fig3_country_ranking.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig3_country_ranking.png")

# ── FIG 4: Gender gap heatmap ────────────────────────────────
gap_matrix = (emp_1564
    .groupby(['ISO', 'YEAR', 'SEX'])['RATE'].mean()
    .unstack('SEX')
    .assign(gap=lambda x: x['M'] - x['F'])['gap']
    .unstack('YEAR')
    .dropna(how='all'))
fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(gap_matrix, annot=True, fmt=".1f", cmap="RdYlGn_r",
            linewidths=0.4, ax=ax, cbar_kws={'label': 'M−F gap (pp)'})
ax.set_title("Gender Employment Gap (M − F) by Country & Year\n(15-64)", fontweight='bold', fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Country")
plt.tight_layout()
plt.savefig("fig4_gender_gap_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig4_gender_gap_heatmap.png")

# ── FIG 5: COVID shock bar chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors_shock = [RED if v < 0 else GREEN for v in covid_drop.values]
ax.bar(covid_drop.index, covid_drop.values, color=colors_shock, edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_title("COVID-19 Employment Shock: Change 2019→2020 (pp)\n(15-64, avg M+F)",
             fontweight='bold', fontsize=13)
ax.set_ylabel("Percentage-point change"); ax.set_xlabel("Country")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("fig5_covid_shock.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig5_covid_shock.png")

# ── FIG 6: Senior employment surge ───────────────────────────
senior_country = (emp_5564
    .groupby(['ISO', 'YEAR'])['RATE'].mean()
    .unstack()
    .dropna(subset=[2015, 2024]))
senior_country['growth'] = senior_country[2024] - senior_country[2015]
sc = senior_country['growth'].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(sc.index, sc.values, color=BLUE, edgecolor='white')
ax.axhline(sc.mean(), color=ORANGE, lw=1.5, linestyle='--',
           label=f'Mean growth={sc.mean():.1f} pp')
ax.set_title("Senior (55-64) Employment Growth 2015→2024 by Country (pp)",
             fontweight='bold', fontsize=13)
ax.set_ylabel("Percentage-point change"); ax.set_xlabel("Country")
ax.legend(); plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("fig6_senior_growth.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig6_senior_growth.png")

# ── FIG 7: Box plots by age group ────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
order = ['15-24', '15-29', '25-54', '55-64', '15-64', '20-64']
data_box = [df_long[df_long['AGE'] == a]['RATE'].dropna().values for a in order]
bp = ax.boxplot(data_box, labels=order, patch_artist=True,
                medianprops=dict(color='black', lw=2))
colors_box = sns.color_palette("pastel", len(order))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax.set_title("Employment Rate Distribution by Age Group (2015–2024)",
             fontweight='bold', fontsize=13)
ax.set_xlabel("Age Group"); ax.set_ylabel("Employment Rate (%)")
plt.tight_layout()
plt.savefig("fig7_boxplot_age.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig7_boxplot_age.png")

# ── FIG 8: Country heatmap (15-64, avg) ──────────────────────
heat_data = (emp_1564
    .groupby(['ISO', 'YEAR'])['RATE'].mean()
    .unstack()
    .dropna(how='all')
    .sort_values(2024, ascending=False))
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlGn",
            linewidths=0.3, ax=ax, cbar_kws={'label': 'Employment Rate (%)'})
ax.set_title("Employment Rate Heatmap by Country & Year (15-64, avg M+F)",
             fontweight='bold', fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Country")
plt.tight_layout()
plt.savefig("fig8_country_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ fig8_country_heatmap.png")

# ==============================================================
# 8. STATISTICAL SUMMARY TABLE
# ==============================================================
print("\n" + "=" * 60)
print("8. FINAL STATISTICAL SUMMARY")
print("=" * 60)

summary = pd.DataFrame({
    'Mean 2015': emp_1564[emp_1564['YEAR']==2015].groupby('ISO')['RATE'].mean().round(1),
    'Mean 2024': emp_1564[emp_1564['YEAR']==2024].groupby('ISO')['RATE'].mean().round(1),
    'Growth pp': total_growth.round(1),
    'Gap M-F 2024 pp': gap_2024['gap'].round(1),
}).dropna(subset=['Mean 2015','Mean 2024'])
print(summary.to_string())

print("\n✅ Analysis complete. All figures saved.")
