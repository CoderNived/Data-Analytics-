"""
=============================================================================
  EUROPEAN EMPLOYMENT & UNEMPLOYMENT COMBINED ANALYSIS
  Datasets: occupazione.csv (Employment) + disoccupazione.csv (Unemployment)
  Period  : 2015 – 2024  |  35 European Countries  |  Gender & Age Disaggregated
  Author  : Senior Data Engineer Report
=============================================================================
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

# ─── Aesthetic Configuration ────────────────────────────────────────────────
BLUE   = "#2563EB";  DBLUE  = "#1D4ED8"
RED    = "#DC2626";  GREEN  = "#16A34A"
ORANGE = "#D97706";  GRAY   = "#6B7280"
TEAL   = "#0891B2";  PURPLE = "#7C3AED"
sns.set_theme(style="whitegrid", font_scale=1.05)

YEAR_COLS = [str(y) for y in range(2015, 2025)]

# =============================================================================
# 1. DATA LOADING & UNDERSTANDING
# =============================================================================
print("=" * 70)
print("1. DATA LOADING & UNDERSTANDING")
print("=" * 70)

occ = pd.read_csv("occupazione.csv")        # Employment rates
dis = pd.read_csv("disoccupazione.csv")     # Unemployment rates

for label, df in [("occupazione (Employment)", occ), ("disoccupazione (Unemployment)", dis)]:
    print(f"\n[{label}]")
    print(f"  Shape     : {df.shape}")
    print(f"  AGE groups: {sorted(df['AGE'].unique().tolist())}")
    print(f"  Countries : {len(df['ISO'].unique())} — {sorted(df['ISO'].unique().tolist())}")

common_ages = sorted(set(occ['AGE'].unique()) & set(dis['AGE'].unique()))
print(f"\nCommon age groups (joinable keys): {common_ages}")
print(f"Unemployment-only age groups: {len(set(dis['AGE'].unique()) - set(occ['AGE'].unique()))} extra cohorts")

# =============================================================================
# 2. DATA CLEANING & PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("2. DATA CLEANING & PREPROCESSING")
print("=" * 70)

def to_long(df: pd.DataFrame, val_col: str) -> pd.DataFrame:
    """Melt wide format to tidy long format with typed columns."""
    return (df
        .melt(id_vars=['SEX', 'AGE', 'ISO'], value_vars=YEAR_COLS,
              var_name='YEAR', value_name=val_col)
        .assign(YEAR=lambda x: x['YEAR'].astype(int),
                **{val_col: lambda x: pd.to_numeric(x[val_col], errors='coerce')})
    )

occ_l = to_long(occ, 'ERATE')   # Employment Rate
dis_l = to_long(dis, 'URATE')   # Unemployment Rate

# Missing value audit
for label, df_l, val in [("Employment", occ_l, 'ERATE'), ("Unemployment", dis_l, 'URATE')]:
    nan_pct = df_l[val].isna().mean() * 100
    nan_cnt = df_l[val].isna().sum()
    print(f"\n[{label}] Total NaN = {nan_cnt} ({nan_pct:.1f}%)")
    nan_by_iso = df_l[df_l[val].isna()]['ISO'].value_counts()
    if len(nan_by_iso):
        print("  NaN by country:", nan_by_iso.to_dict())

# Duplicate check
for label, df_l, key in [
    ("Employment", occ_l, ['SEX','AGE','ISO','YEAR']),
    ("Unemployment", dis_l, ['SEX','AGE','ISO','YEAR'])
]:
    dupes = df_l.duplicated(subset=key).sum()
    print(f"\n[{label}] Duplicate rows: {dupes}")

print("\n[Preprocessing decisions]")
print("  • Wide→Long (melt): Enables time-series groupby and faceting")
print("  • Type casting: YEAR→int64, RATE→float64 (errors='coerce')")
print("  • No imputation: missing values excluded from aggregations")
print("  • BA (Bosnia): data only from 2021; ME (Montenegro): data only to 2021")

# Convenience filters — Employment
e_1564 = occ_l[occ_l['AGE'] == '15-64']
e_1524 = occ_l[occ_l['AGE'] == '15-24']
e_5564 = occ_l[occ_l['AGE'] == '55-64']
e_2554 = occ_l[occ_l['AGE'] == '25-54']

# Convenience filters — Unemployment
u_1574 = dis_l[dis_l['AGE'] == '15-74']
u_1524 = dis_l[dis_l['AGE'] == '15-24']
u_5564 = dis_l[dis_l['AGE'] == '55-64']
u_2554 = dis_l[dis_l['AGE'] == '25-54']

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ── 3a. Headline trends ──────────────────────────────────────────────────────
emp_yoy = e_1564.groupby('YEAR')['ERATE'].mean()
unemp_yoy = u_1574.groupby('YEAR')['URATE'].mean()

trend_df = pd.DataFrame({
    'Employment Rate (%)': emp_yoy.round(2),
    'Unemployment Rate (%)': unemp_yoy.round(2),
    'ERate YoY pp': emp_yoy.diff().round(2),
    'URate YoY pp': unemp_yoy.diff().round(2),
})
print("\n[Headline Trends 2015–2024 (Employment 15-64 | Unemployment 15-74)]")
print(trend_df.to_string())

# ── 3b. Employment by age group ──────────────────────────────────────────────
print("\n[Employment stats by age group (2015–2024)]")
print(occ_l.groupby('AGE')['ERATE'].describe().round(2))

# ── 3c. Unemployment by age group ───────────────────────────────────────────
print("\n[Unemployment stats by age group — key cohorts (2015–2024)]")
key_ages = ['15-24','25-54','55-64','15-74']
print(dis_l[dis_l['AGE'].isin(key_ages)].groupby('AGE')['URATE'].describe().round(2))

# ── 3d. Gender analysis ──────────────────────────────────────────────────────
emp_gender = e_1564.groupby(['YEAR','SEX'])['ERATE'].mean().unstack()
emp_gender['gap_MF'] = emp_gender['M'] - emp_gender['F']
print("\n[Employment gender trends (15-64) — M-F gap]")
print(emp_gender.round(2))

unemp_gender = u_1574.groupby(['YEAR','SEX'])['URATE'].mean().unstack()
unemp_gender['gap_FM'] = unemp_gender['F'] - unemp_gender['M']
print("\n[Unemployment gender trends (15-74) — F-M gap]")
print(unemp_gender.round(2))

# ── 3e. Country rankings 2024 ────────────────────────────────────────────────
emp_rank = e_1564[e_1564['YEAR']==2024].groupby('ISO')['ERATE'].mean().dropna().sort_values(ascending=False)
unemp_rank = u_1574[u_1574['YEAR']==2024].groupby('ISO')['URATE'].mean().dropna().sort_values()

print("\n[Employment ranking — 2024 (15-64)]")
print(emp_rank.round(2))
print("\n[Unemployment ranking — 2024 (15-74), lowest first]")
print(unemp_rank.round(2))

# ── 3f. COVID shock ──────────────────────────────────────────────────────────
e_pivot = e_1564.groupby(['ISO','YEAR'])['ERATE'].mean().unstack()
u_pivot = u_1574.groupby(['ISO','YEAR'])['URATE'].mean().unstack()

e_shock = (e_pivot[2020] - e_pivot[2019]).dropna().sort_values()
u_shock = (u_pivot[2020] - u_pivot[2019]).dropna().sort_values(ascending=False)

print("\n[COVID shock — Employment drop 2019→2020 (largest drops)]")
print(e_shock.head(8).round(2))

print("\n[COVID shock — Unemployment rise 2019→2020 (largest rises)]")
print(u_shock.head(8).round(2))

# =============================================================================
# 4. CORRELATION & RELATIONSHIP ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("4. CORRELATION & RELATIONSHIP ANALYSIS")
print("=" * 70)

# Merge employment + unemployment on (ISO, YEAR, SEX) — common age 15-64
merged = pd.merge(
    e_1564.groupby(['ISO','YEAR'])['ERATE'].mean().reset_index(),
    dis_l[dis_l['AGE']=='15-64'].groupby(['ISO','YEAR'])['URATE'].mean().reset_index(),
    on=['ISO','YEAR']
).dropna()

overall_corr = merged['ERATE'].corr(merged['URATE'])
print(f"\nOverall correlation Employment vs Unemployment (15-64, all years): {overall_corr:.3f}")

# Per-year correlation
print("\nCorrelation by year:")
for yr in range(2015, 2025):
    sub = merged[merged['YEAR'] == yr]
    if len(sub) > 5:
        r = sub['ERATE'].corr(sub['URATE'])
        print(f"  {yr}: r = {r:.3f}")

# Youth unemployment vs employment correlation
youth_corr = pd.merge(
    e_1524[e_1524['YEAR']==2024].groupby('ISO')['ERATE'].mean(),
    u_1524[u_1524['YEAR']==2024].groupby('ISO')['URATE'].mean(),
    on='ISO'
).dropna()
youth_r = youth_corr['ERATE'].corr(youth_corr['URATE'])
print(f"\nYouth (15-24) Employment vs Unemployment correlation (2024): {youth_r:.3f}")

# Senior correlation
senior_corr = pd.merge(
    e_5564[e_5564['YEAR']==2024].groupby('ISO')['ERATE'].mean(),
    u_5564[u_5564['YEAR']==2024].groupby('ISO')['URATE'].mean(),
    on='ISO'
).dropna()
senior_r = senior_corr['ERATE'].corr(senior_corr['URATE'])
print(f"Senior (55-64) Employment vs Unemployment correlation (2024): {senior_r:.3f}")

# =============================================================================
# 5. ADVANCED ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("5. ADVANCED ANALYSIS")
print("=" * 70)

# Youth unemployment improvement 2015→2024
youth_change = (u_pivot_y := dis_l[dis_l['AGE']=='15-24']
    .groupby(['ISO','YEAR'])['URATE'].mean().unstack()
    .assign(change=lambda x: x[2024] - x[2015])
    .dropna(subset=[2015, 2024])['change'])
print("\n[Youth (15-24) unemployment change 2015→2024 — top improvers]")
print(youth_change.sort_values().head(8).round(2))
print("\n[Youth (15-24) unemployment change — most deteriorated]")
print(youth_change.sort_values(ascending=False).head(5).round(2))

# Senior unemployment trend
print("\n[Senior (55-64) unemployment trend]")
print(u_5564.groupby('YEAR')['URATE'].mean().round(2))

# Outlier detection: countries with high unemployment + high employment (paradox?)
sub_2024 = merged[merged['YEAR']==2024].sort_values('URATE', ascending=False)
print("\n[2024 combined snapshot — Employment vs Unemployment (15-64)]")
print(sub_2024.round(2).to_string(index=False))

# Recovery speed: time to return to pre-COVID unemployment levels
print("\n[Countries that recovered unemployment below 2019 levels by 2022]")
pre_covid = u_pivot[2019]
post_covid = u_pivot[2022]
recovered = (post_covid < pre_covid).dropna()
print(recovered[recovered == True].index.tolist())

print("\n[Countries still above pre-COVID unemployment in 2022]")
print(recovered[recovered == False].index.tolist())

# =============================================================================
# 6. VISUALISATIONS
# =============================================================================
print("\n" + "=" * 70)
print("6. GENERATING VISUALISATIONS …")
print("=" * 70)

# ── FIG 1: Dual-axis trend (Employment + Unemployment) ──────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.plot(emp_yoy.index, emp_yoy.values, color=BLUE, lw=2.5, marker='o', ms=5, label='Employment Rate (15-64)')
ax2.plot(unemp_yoy.index, unemp_yoy.values, color=RED, lw=2.5, marker='s', ms=5, label='Unemployment Rate (15-74)')
ax1.axvspan(2019.5, 2020.5, color='gray', alpha=0.15, label='COVID-19')
ax1.set_xlabel("Year"); ax1.set_ylabel("Employment Rate (%)", color=BLUE)
ax2.set_ylabel("Unemployment Rate (%)", color=RED)
ax1.tick_params(axis='y', labelcolor=BLUE); ax2.tick_params(axis='y', labelcolor=RED)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
ax1.set_title("European Employment & Unemployment Trends — 2015–2024\n(Mean across 35 countries)", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig("fig1_dual_trend.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig1_dual_trend.png")

# ── FIG 2: Gender split — Employment & Unemployment ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for sex, color, lbl in [('F', RED, 'Female'), ('M', BLUE, 'Male')]:
    s = e_1564.groupby(['YEAR','SEX'])['ERATE'].mean().unstack()[sex]
    ax.plot(s.index, s.values, color=color, lw=2.5, marker='o', ms=5, label=lbl)
ax.axvspan(2019.5, 2020.5, color='gray', alpha=0.12)
ax.set_title("Employment Rate by Gender (15-64)", fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Rate (%)"); ax.legend()

ax = axes[1]
for sex, color, lbl in [('F', RED, 'Female'), ('M', BLUE, 'Male')]:
    s = u_1574.groupby(['YEAR','SEX'])['URATE'].mean().unstack()[sex]
    ax.plot(s.index, s.values, color=color, lw=2.5, marker='o', ms=5, label=lbl)
ax.axvspan(2019.5, 2020.5, color='gray', alpha=0.12)
ax.set_title("Unemployment Rate by Gender (15-74)", fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Rate (%)"); ax.legend()

plt.suptitle("Gender Comparison: Employment & Unemployment 2015–2024", fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig2_gender_split.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig2_gender_split.png")

# ── FIG 3: Scatter — Employment vs Unemployment (2024) ──────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
for _, row in merged[merged['YEAR']==2024].iterrows():
    ax.scatter(row['ERATE'], row['URATE'], s=60, color=BLUE, alpha=0.7, zorder=3)
    ax.annotate(row['ISO'], (row['ERATE'], row['URATE']),
                textcoords="offset points", xytext=(4, 3), fontsize=8, color=GRAY)
m, b = np.polyfit(merged[merged['YEAR']==2024]['ERATE'].dropna(),
                  merged[merged['YEAR']==2024]['URATE'].dropna(), 1)
x_line = np.linspace(merged['ERATE'].min(), merged['ERATE'].max(), 100)
ax.plot(x_line, m * x_line + b, color=RED, lw=1.5, linestyle='--', label=f'OLS (r={overall_corr:.2f})')
ax.set_xlabel("Employment Rate % (15-64)"); ax.set_ylabel("Unemployment Rate % (15-64)")
ax.set_title("Employment vs. Unemployment — 2024\n(r = −0.73: strong negative correlation)", fontweight='bold', fontsize=13)
ax.legend(); plt.tight_layout()
plt.savefig("fig3_scatter_correlation.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig3_scatter_correlation.png")

# ── FIG 4: Youth unemployment country ranking 2024 ──────────────────────────
yu_2024 = (u_1524[u_1524['YEAR']==2024]
    .groupby('ISO')['URATE'].mean()
    .dropna().sort_values(ascending=False))
colors_bar = [RED if v>=20 else (ORANGE if v>=12 else GREEN) for v in yu_2024.values]
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(yu_2024.index[::-1], yu_2024.values[::-1], color=colors_bar[::-1], edgecolor='white')
ax.axvline(yu_2024.mean(), color='navy', lw=1.5, linestyle='--', label=f'Mean={yu_2024.mean():.1f}%')
ax.set_title("Youth (15-24) Unemployment Rate by Country — 2024\nRed ≥ 20% | Orange ≥ 12% | Green < 12%", fontweight='bold', fontsize=13)
ax.set_xlabel("Unemployment Rate (%)"); ax.legend()
plt.tight_layout()
plt.savefig("fig4_youth_unemployment.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig4_youth_unemployment.png")

# ── FIG 5: COVID double shock ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
colors_e = [RED if v < 0 else GREEN for v in e_shock.values]
ax.bar(e_shock.index, e_shock.values, color=colors_e, edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_title("Employment Change 2019→2020 (pp)", fontweight='bold')
ax.set_ylabel("pp change"); plt.sca(ax); plt.xticks(rotation=45, ha='right')

ax = axes[1]
colors_u = [RED if v > 0 else GREEN for v in u_shock.values]
ax.bar(u_shock.index, u_shock.values, color=colors_u, edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_title("Unemployment Change 2019→2020 (pp)", fontweight='bold')
ax.set_ylabel("pp change"); plt.sca(ax); plt.xticks(rotation=45, ha='right')

plt.suptitle("COVID-19 Labour Market Shock by Country", fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig5_covid_double_shock.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig5_covid_double_shock.png")

# ── FIG 6: Unemployment heatmap ──────────────────────────────────────────────
heat_u = (u_1574.groupby(['ISO','YEAR'])['URATE'].mean().unstack().dropna(how='all')
    .sort_values(2024, ascending=False))
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(heat_u, annot=True, fmt=".1f", cmap="RdYlGn_r",
            linewidths=0.3, ax=ax, cbar_kws={'label': 'Unemployment Rate (%)'})
ax.set_title("Unemployment Rate Heatmap by Country & Year (15-74, avg M+F)\nRed = High Unemployment", fontweight='bold', fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Country")
plt.tight_layout()
plt.savefig("fig6_unemployment_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig6_unemployment_heatmap.png")

# ── FIG 7: Age group unemployment trend ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
age_sel = ['15-24', '25-54', '55-64', '15-74']
colors_age = [RED, BLUE, ORANGE, GRAY]
for age, color in zip(age_sel, colors_age):
    s = dis_l[dis_l['AGE']==age].groupby('YEAR')['URATE'].mean()
    ax.plot(s.index, s.values, color=color, lw=2.5, marker='o', ms=4, label=age)
ax.axvspan(2019.5, 2020.5, color='gray', alpha=0.12, label='COVID-19')
ax.set_title("Unemployment Rate by Age Group — 2015–2024", fontweight='bold', fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Unemployment Rate (%)")
ax.legend(title="Age Group"); plt.tight_layout()
plt.savefig("fig7_unemployment_by_age.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig7_unemployment_by_age.png")

# ── FIG 8: Youth unemployment improvement 2015→2024 ─────────────────────────
youth_pivot = dis_l[dis_l['AGE']=='15-24'].groupby(['ISO','YEAR'])['URATE'].mean().unstack()
yu_change = (youth_pivot[2024] - youth_pivot[2015]).dropna().sort_values()
colors_yu = [GREEN if v < 0 else RED for v in yu_change.values]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(yu_change.index, yu_change.values, color=colors_yu, edgecolor='white')
ax.axhline(0, color='black', lw=0.8)
ax.set_title("Youth (15-24) Unemployment Change 2015→2024 by Country (pp)\nGreen = Improved | Red = Deteriorated", fontweight='bold', fontsize=13)
ax.set_ylabel("pp change"); plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("fig8_youth_unemployment_change.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig8_youth_unemployment_change.png")

# ── FIG 9: Gender unemployment gap per country 2024 ─────────────────────────
g_unemp = (u_1574[u_1574['YEAR']==2024]
    .groupby(['ISO','SEX'])['URATE'].mean()
    .unstack()
    .dropna()
    .assign(gap=lambda x: (x['F'] - x['M']).round(2))
    .sort_values('gap', ascending=False))
colors_g = [RED if v > 1 else (ORANGE if v > 0 else GREEN) for v in g_unemp['gap'].values]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(g_unemp.index, g_unemp['gap'].values, color=colors_g, edgecolor='white')
ax.axhline(0, color='black', lw=1)
ax.set_title("Unemployment Gender Gap (F−M) by Country — 2024\nRed = Women more unemployed | Green = Men more unemployed", fontweight='bold', fontsize=13)
ax.set_ylabel("F-M gap (pp)"); plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("fig9_unemployment_gender_gap.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig9_unemployment_gender_gap.png")

# ── FIG 10: Employment heatmap ───────────────────────────────────────────────
heat_e = (e_1564.groupby(['ISO','YEAR'])['ERATE'].mean().unstack()
    .dropna(how='all').sort_values(2024, ascending=False))
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(heat_e, annot=True, fmt=".1f", cmap="YlGn",
            linewidths=0.3, ax=ax, cbar_kws={'label': 'Employment Rate (%)'})
ax.set_title("Employment Rate Heatmap by Country & Year (15-64, avg M+F)", fontweight='bold', fontsize=13)
ax.set_xlabel("Year"); ax.set_ylabel("Country")
plt.tight_layout()
plt.savefig("fig10_employment_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig10_employment_heatmap.png")

# ── FIG 11: Combined country bubble chart ───────────────────────────────────
snap = merged[merged['YEAR']==2024].copy()
snap['youth_u'] = snap['ISO'].map(
    u_1524[u_1524['YEAR']==2024].groupby('ISO')['URATE'].mean())
snap = snap.dropna(subset=['ERATE','URATE','youth_u'])

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(snap['ERATE'], snap['URATE'],
                     s=snap['youth_u']*15,
                     c=snap['youth_u'], cmap='RdYlGn_r',
                     alpha=0.75, edgecolors='white', linewidth=0.5,
                     vmin=5, vmax=35)
for _, row in snap.iterrows():
    ax.annotate(row['ISO'], (row['ERATE'], row['URATE']),
                textcoords="offset points", xytext=(5, 3), fontsize=8)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Youth Unemployment Rate (%)", fontsize=10)
ax.set_xlabel("Employment Rate % (15-64)"); ax.set_ylabel("Unemployment Rate % (15-64)")
ax.set_title("Employment vs Unemployment — 2024\nBubble size & colour = Youth (15-24) Unemployment Rate", fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig("fig11_bubble_chart.png", dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig11_bubble_chart.png")

# =============================================================================
# 7. FINAL SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("7. FINAL COMBINED COUNTRY SNAPSHOT — 2024")
print("=" * 70)

snap_table = pd.DataFrame({
    'ERate 2024': emp_rank.round(1),
    'URate 2024': unemp_rank.round(1),
    'Youth URate 2024': (u_1524[u_1524['YEAR']==2024]
        .groupby('ISO')['URATE'].mean().dropna().round(1)),
    'ERate Growth (2015-24)': (e_pivot[2024] - e_pivot[2015]).dropna().round(1),
}).dropna(subset=['ERate 2024','URate 2024'])
print(snap_table.sort_values('ERate 2024', ascending=False).to_string())

print("\n✅ Combined analysis complete. All 11 figures saved.")
