import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              mean_squared_error, mean_absolute_error, r2_score,
                              classification_report, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d27',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'text.color': '#e0e0e0',
    'grid.color': '#2a2d3a',
    'grid.linewidth': 0.6,
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#ffffff',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#3a3d4a',
})

PALETTE = ['#4fc3f7','#ab47bc','#66bb6a','#ffa726','#ef5350','#26c6da','#ff7043','#42a5f5','#ec407a','#8d6e63']
ACCENT = '#4fc3f7'
HIGHLIGHT = '#ffa726'

def save_fig(name, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(f'/home/claude/{name}.png', dpi=150, bbox_inches='tight',
                facecolor='#0f1117', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: {name}.png")

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 1 — DATA UNDERSTANDING")
print("="*60)

df = pd.read_csv('/mnt/user-data/uploads/synthetic_fooddelivery_dataset.csv')
print(f"\nShape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum()/1024:.1f} KB")
print("\nColumn dtypes:")
print(df.dtypes)
print("\nSample (5 rows):")
print(df.head())
print("\nBasic describe:")
print(df.describe(include='all').T.to_string())

# ═══════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 2 — DATA CLEANING")
print("="*60)

# Missing values
print("\nMissing values (count & %):")
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df = pd.DataFrame({'Missing': miss, 'Pct%': miss_pct}).query('Missing > 0')
print(miss_df)

# Duplicates
dups = df.duplicated().sum()
print(f"\nDuplicate rows: {dups}")

# Parse datetime
df['Waktu_Transaksi'] = pd.to_datetime(df['Waktu_Transaksi'], format='mixed', dayfirst=False)

# Extract temporal features
df['Hour']       = df['Waktu_Transaksi'].dt.hour
df['DayOfWeek']  = df['Waktu_Transaksi'].dt.dayofweek
df['DayName']    = df['Waktu_Transaksi'].dt.day_name()
df['Month']      = df['Waktu_Transaksi'].dt.month
df['IsWeekend']  = df['DayOfWeek'].isin([5,6]).astype(int)

# Fill numeric missing values with medians (robust to outliers)
for col in ['Jarak_Kirim_KM', 'Waktu_Tunggu_Menit', 'Rating_Pelanggan', 'Harga_Pesanan']:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  → {col}: filled {df[col].isnull().sum()} NaN with median={median_val:.2f}")

# Fill categorical NaN
df['Ulasan_Teks'].fillna('No Review', inplace=True)
df['Tingkat_Keluhan'].fillna('Tidak Ada', inplace=True)

# Validate categoricals
print("\nUnique values per categorical:")
for col in ['Kategori_Menu','Status_Promo','Tingkat_Keluhan','Status_Pesanan']:
    print(f"  {col}: {sorted(df[col].astype(str).unique())}")

# Outlier detection (IQR)
print("\nOutlier detection (IQR method):")
for col in ['Harga_Pesanan','Jarak_Kirim_KM','Waktu_Tunggu_Menit','Rating_Pelanggan']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df[(df[col] < lo) | (df[col] > hi)]
    print(f"  {col}: {len(outliers)} outliers (range [{lo:.2f}, {hi:.2f}])")

print(f"\nClean dataset shape: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. EDA — UNIVARIATE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 3 — EDA")
print("="*60)

# --- 3A: Numeric distributions ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Univariate Distribution — Numeric Features', fontsize=14, fontweight='bold', color='white', y=0.98)

numeric_cols = [('Harga_Pesanan','Order Price (IDR)', '#4fc3f7'),
                ('Jarak_Kirim_KM','Delivery Distance (KM)', '#ab47bc'),
                ('Waktu_Tunggu_Menit','Wait Time (Minutes)', '#66bb6a'),
                ('Rating_Pelanggan','Customer Rating', '#ffa726')]

for ax, (col, label, color) in zip(axes.flat, numeric_cols):
    ax.set_facecolor('#1a1d27')
    data = df[col].dropna()
    ax.hist(data, bins=35, color=color, alpha=0.85, edgecolor='#0f1117', linewidth=0.3)
    ax.axvline(data.mean(), color='white', lw=1.5, ls='--', label=f'Mean: {data.mean():.1f}')
    ax.axvline(data.median(), color=HIGHLIGHT, lw=1.5, ls=':', label=f'Median: {data.median():.1f}')
    ax.set_title(label, color='white')
    ax.set_xlabel(label, color='#aaaaaa')
    ax.set_ylabel('Count', color='#aaaaaa')
    ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
    ax.grid(axis='y', alpha=0.3)

save_fig('fig1_univariate_numeric')
print("  Fig 1: Univariate Numeric Distributions")

# --- 3B: Categorical distributions ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Univariate Distribution — Categorical Features', fontsize=14, fontweight='bold', color='white', y=0.98)

cats = [('Kategori_Menu','Menu Category'),
        ('Status_Promo','Promo Status'),
        ('Tingkat_Keluhan','Complaint Level'),
        ('Status_Pesanan','Order Status')]

for ax, (col, label) in zip(axes.flat, cats):
    ax.set_facecolor('#1a1d27')
    vc = df[col].astype(str).value_counts()
    colors = PALETTE[:len(vc)]
    bars = ax.bar(range(len(vc)), vc.values, color=colors, edgecolor='#0f1117', linewidth=0.5)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=35, ha='right', fontsize=8)
    ax.set_title(label, color='white')
    ax.set_ylabel('Count', color='#aaaaaa')
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                f'{val:,}', ha='center', va='bottom', fontsize=7, color='#cccccc')
    ax.grid(axis='y', alpha=0.3)

save_fig('fig2_univariate_categorical')
print("  Fig 2: Univariate Categorical Distributions")

# --- 3C: Time Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Temporal Ordering Patterns', fontsize=14, fontweight='bold', color='white')

# Hourly
ax = axes[0]
ax.set_facecolor('#1a1d27')
hourly = df.groupby('Hour').size()
ax.bar(hourly.index, hourly.values, color=ACCENT, alpha=0.85, edgecolor='#0f1117')
ax.set_title('Orders by Hour of Day', color='white')
ax.set_xlabel('Hour', color='#aaaaaa')
ax.set_ylabel('Order Count', color='#aaaaaa')
ax.grid(axis='y', alpha=0.3)
peak_hour = hourly.idxmax()
ax.axvline(peak_hour, color=HIGHLIGHT, lw=2, ls='--', label=f'Peak: {peak_hour}:00')
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white')

# Day of week
ax = axes[1]
ax.set_facecolor('#1a1d27')
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
daily = df.groupby('DayName').size().reindex(day_order)
colors_day = [HIGHLIGHT if d in ['Saturday','Sunday'] else ACCENT for d in day_order]
ax.bar(range(7), daily.values, color=colors_day, alpha=0.85, edgecolor='#0f1117')
ax.set_xticks(range(7))
ax.set_xticklabels([d[:3] for d in day_order])
ax.set_title('Orders by Day of Week', color='white')
ax.set_xlabel('Day', color='#aaaaaa')
ax.set_ylabel('Order Count', color='#aaaaaa')
ax.grid(axis='y', alpha=0.3)

save_fig('fig3_time_analysis')
print("  Fig 3: Temporal Patterns")

# ═══════════════════════════════════════════════════════════════════════════
# 4. BIVARIATE & MULTIVARIATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 4 — BIVARIATE & MULTIVARIATE ANALYSIS")
print("="*60)

# Correlation matrix
num_df = df[['Harga_Pesanan','Jarak_Kirim_KM','Waktu_Tunggu_Menit',
             'Rating_Pelanggan','Hour','DayOfWeek','IsWeekend']].dropna()
corr = num_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', linewidths=0.5, linecolor='#0f1117',
            annot_kws={'size': 9, 'color': 'white'},
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix — Numeric Features', color='white', fontsize=12, fontweight='bold')
plt.xticks(rotation=30, ha='right', color='#e0e0e0')
plt.yticks(color='#e0e0e0')
ax.collections[0].colorbar.ax.tick_params(colors='white')
save_fig('fig4_correlation_heatmap')
print("  Fig 4: Correlation Heatmap")

print("\nCorrelation with Rating_Pelanggan:")
print(corr['Rating_Pelanggan'].drop('Rating_Pelanggan').sort_values(ascending=False).to_string())

# Key bivariate: Distance vs WaitTime vs Rating
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Key Bivariate Relationships', fontsize=13, fontweight='bold', color='white')

plots = [
    ('Jarak_Kirim_KM', 'Waktu_Tunggu_Menit', 'Distance vs Wait Time', '#4fc3f7'),
    ('Waktu_Tunggu_Menit', 'Rating_Pelanggan', 'Wait Time vs Rating', '#ab47bc'),
    ('Harga_Pesanan', 'Rating_Pelanggan', 'Order Price vs Rating', '#66bb6a'),
]
for ax, (x, y, title, color) in zip(axes, plots):
    ax.set_facecolor('#1a1d27')
    ax.scatter(df[x], df[y], alpha=0.25, s=12, color=color)
    # regression line
    idx = df[[x,y]].dropna().index
    if len(idx) > 10:
        m, b, r, p, _ = stats.linregress(df.loc[idx, x], df.loc[idx, y])
        xs = np.linspace(df[x].min(), df[x].max(), 100)
        ax.plot(xs, m*xs+b, color=HIGHLIGHT, lw=2, label=f'r={r:.2f}, p={p:.3f}')
    ax.set_title(title, color='white')
    ax.set_xlabel(x, color='#aaaaaa', fontsize=8)
    ax.set_ylabel(y, color='#aaaaaa', fontsize=8)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
    ax.grid(alpha=0.25)

save_fig('fig5_bivariate')
print("  Fig 5: Bivariate Scatter Plots")

# Promo vs Price
print("\nPromo vs Non-Promo — Price Statistics:")
promo_stats = df.groupby('Status_Promo')['Harga_Pesanan'].agg(['mean','median','std','count'])
print(promo_stats)

# Complaints vs Rating
print("\nComplaints vs Rating:")
comp_rating = df.groupby('Tingkat_Keluhan')['Rating_Pelanggan'].agg(['mean','median','count'])
print(comp_rating.sort_values('mean'))

# Menu category vs Price
print("\nMenu Category vs Price (mean):")
menu_price = df.groupby('Kategori_Menu')['Harga_Pesanan'].agg(['mean','median','count']).sort_values('mean', ascending=False)
print(menu_price.to_string())

# Boxplot: Menu category vs price
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Price & Wait Time by Menu Category', fontsize=13, fontweight='bold', color='white')

for ax, col, label in zip(axes, ['Harga_Pesanan','Waktu_Tunggu_Menit'], ['Order Price (IDR)','Wait Time (min)']):
    ax.set_facecolor('#1a1d27')
    cats_sorted = df.groupby('Kategori_Menu')[col].median().sort_values(ascending=False).index
    data_box = [df[df['Kategori_Menu']==c][col].dropna().values for c in cats_sorted]
    bp = ax.boxplot(data_box, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markerfacecolor='#555555', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for el in ['whiskers','caps','medians']:
        for item in bp[el]:
            item.set_color('white')
            item.set_linewidth(1)
    ax.set_xticks(range(1, len(cats_sorted)+1))
    ax.set_xticklabels(cats_sorted, rotation=35, ha='right', fontsize=8)
    ax.set_title(f'{label} by Category', color='white')
    ax.set_ylabel(label, color='#aaaaaa')
    ax.grid(axis='y', alpha=0.3)

save_fig('fig6_category_boxplot')
print("  Fig 6: Category Boxplots")

# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 5 — FEATURE ENGINEERING")
print("="*60)

df['PricePerKM'] = df['Harga_Pesanan'] / (df['Jarak_Kirim_KM'] + 0.01)
df['IsPeakHour'] = df['Hour'].apply(lambda h: 1 if (11<=h<=13 or 17<=h<=20) else 0)
df['WaitTimeCategory'] = pd.cut(df['Waktu_Tunggu_Menit'],
                                 bins=[0,15,30,45,200],
                                 labels=['Fast','Normal','Slow','Very Slow'])
df['DistanceBucket'] = pd.cut(df['Jarak_Kirim_KM'],
                               bins=[0,2,5,10,100],
                               labels=['Near','Mid','Far','Very Far'])
df['HasReview'] = (df['Ulasan_Teks'] != 'No Review').astype(int)
df['IsHighValue'] = (df['Harga_Pesanan'] > df['Harga_Pesanan'].quantile(0.75)).astype(int)

# Sentiment proxy from review
positive_words = ['enak','mantap','bagus','puas','cepat','suka','lezat','oke','good','great','tepat','sesuai']
negative_words = ['lama','lambat','kecewa','buruk','jelek','salah','basi','dingin','terlambat','tidak','kurang','parah']

def simple_sentiment(text):
    if not isinstance(text, str) or text == 'No Review':
        return 0
    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos > neg: return 1
    if neg > pos: return -1
    return 0

df['SentimentScore'] = df['Ulasan_Teks'].apply(simple_sentiment)
df['IsPromo'] = (df['Status_Promo'] == True).astype(int)

print("\nNew features created:")
new_feats = ['PricePerKM','IsPeakHour','WaitTimeCategory','DistanceBucket',
             'HasReview','IsHighValue','SentimentScore','IsPromo']
print(df[new_feats].describe(include='all').T[['count','mean','min','max']].to_string())

# ═══════════════════════════════════════════════════════════════════════════
# 6. STATISTICAL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 6 — STATISTICAL INSIGHTS")
print("="*60)

# T-test: Promo vs Non-promo price
promo_prices = df[df['Status_Promo']==True]['Harga_Pesanan'].dropna()
nopr_prices  = df[df['Status_Promo']==False]['Harga_Pesanan'].dropna()
t_stat, p_val = stats.ttest_ind(promo_prices, nopr_prices)
print(f"\nHypothesis: Do promotions affect order price?")
print(f"  Promo mean   : {promo_prices.mean():.2f}")
print(f"  No-Promo mean: {nopr_prices.mean():.2f}")
print(f"  t-statistic  : {t_stat:.4f}")
print(f"  p-value      : {p_val:.6f}")
print(f"  Result       : {'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'Not significant'}")

# Pearson: Distance vs Rating
valid = df[['Jarak_Kirim_KM','Rating_Pelanggan']].dropna()
r, p = stats.pearsonr(valid['Jarak_Kirim_KM'], valid['Rating_Pelanggan'])
print(f"\nHypothesis: Does distance impact customer rating?")
print(f"  Pearson r : {r:.4f}")
print(f"  p-value   : {p:.6f}")
print(f"  Result    : {'SIGNIFICANT' if p < 0.05 else 'Not significant'}")

# Kruskal-Wallis: complaint level vs rating
groups = [df[df['Tingkat_Keluhan']==c]['Rating_Pelanggan'].dropna() 
          for c in df['Tingkat_Keluhan'].unique()]
h_stat, p_kw = stats.kruskal(*groups)
print(f"\nKruskal-Wallis: Complaint level vs Rating")
print(f"  H-stat: {h_stat:.4f}, p-value: {p_kw:.6f}")
print(f"  Result: {'SIGNIFICANT difference across complaint groups' if p_kw < 0.05 else 'No significant difference'}")

# ═══════════════════════════════════════════════════════════════════════════
# 7+8. ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 7+8 — MACHINE LEARNING")
print("="*60)

# --- Encode categoricals ---
df_ml = df.copy()
le = LabelEncoder()
for col in ['Kategori_Menu','Status_Promo','Status_Pesanan','DayName',
            'WaitTimeCategory','DistanceBucket']:
    df_ml[col+'_enc'] = le.fit_transform(df_ml[col].astype(str))

# Target 1: Complaint Level Classification (binary: complaint vs no complaint)
df_ml['HasComplaint'] = (df_ml['Tingkat_Keluhan'] != 'Tidak Ada').astype(int)

features = ['Harga_Pesanan','Jarak_Kirim_KM','Waktu_Tunggu_Menit',
            'Hour','DayOfWeek','IsWeekend','IsPeakHour',
            'IsPromo','IsHighValue','SentimentScore','HasReview',
            'Kategori_Menu_enc','WaitTimeCategory_enc','DistanceBucket_enc']

# Drop rows with NaN in features
df_sub = df_ml[features + ['HasComplaint','Rating_Pelanggan']].dropna()
X_comp = df_sub[features]
y_comp = df_sub['HasComplaint']

X_train, X_test, y_train, y_test = train_test_split(X_comp, y_comp, test_size=0.2, random_state=42, stratify=y_comp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models_cls = {
    'Logistic Regression': LogisticRegression(max_iter=500, C=1.0),
    'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("\n=== Classification: Complaint Prediction ===")
cls_results = {}
for name, model in models_cls.items():
    X_tr = X_train_s if 'Regression' in name else X_train
    X_te = X_test_s  if 'Regression' in name else X_test
    model.fit(X_tr, y_train)
    pred = model.predict(X_te)
    acc  = accuracy_score(y_test, pred)
    f1   = f1_score(y_test, pred, average='weighted')
    prec = precision_score(y_test, pred, average='weighted')
    rec  = recall_score(y_test, pred, average='weighted')
    cls_results[name] = {'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec}
    print(f"\n{name}:")
    print(f"  Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

# Target 2: Rating Regression
y_rating = df_sub['Rating_Pelanggan']
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_comp, y_rating, test_size=0.2, random_state=42)
X_tr2s = scaler.fit_transform(X_tr2)
X_te2s = scaler.transform(X_te2)

reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Random Forest Reg': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting Reg': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("\n=== Regression: Rating Prediction ===")
reg_results = {}
for name, model in [('Linear Regression', LinearRegression()),
                     ('Ridge Regression', Ridge(alpha=1.0)),
                     ('Random Forest Reg', RandomForestRegressor(n_estimators=100, random_state=42))]:
    X_tr_use = X_tr2s if 'Linear' in name or 'Ridge' in name else X_tr2
    X_te_use = X_te2s if 'Linear' in name or 'Ridge' in name else X_te2
    model.fit(X_tr_use, y_tr2)
    pred = model.predict(X_te_use)
    rmse = np.sqrt(mean_squared_error(y_te2, pred))
    mae  = mean_absolute_error(y_te2, pred)
    r2   = r2_score(y_te2, pred)
    reg_results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"\n{name}:")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

# Feature importance from RF classifier
rf = models_cls['Random Forest']
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#1a1d27')
colors_bar = [HIGHLIGHT if v >= feat_imp.quantile(0.75) else ACCENT for v in feat_imp.values]
ax.barh(feat_imp.index, feat_imp.values, color=colors_bar, edgecolor='#0f1117')
ax.set_title('Feature Importance — Random Forest (Complaint Prediction)', color='white', fontweight='bold')
ax.set_xlabel('Importance Score', color='#aaaaaa')
ax.grid(axis='x', alpha=0.3)
save_fig('fig7_feature_importance')
print("\n  Fig 7: Feature Importance")

# Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Model Performance Comparison', fontsize=13, fontweight='bold', color='white')

# Classification
ax = axes[0]
ax.set_facecolor('#1a1d27')
model_names = list(cls_results.keys())
accs = [cls_results[m]['Accuracy'] for m in model_names]
f1s  = [cls_results[m]['F1'] for m in model_names]
x = np.arange(len(model_names))
w = 0.35
ax.bar(x - w/2, accs, w, label='Accuracy', color=ACCENT, alpha=0.85)
ax.bar(x + w/2, f1s, w, label='F1-Score', color=HIGHLIGHT, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' ','\n') for m in model_names], fontsize=7)
ax.set_title('Classification — Complaint Prediction', color='white')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white')
ax.grid(axis='y', alpha=0.3)
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax.text(i-w/2, a+0.01, f'{a:.2f}', ha='center', fontsize=7, color='white')
    ax.text(i+w/2, f+0.01, f'{f:.2f}', ha='center', fontsize=7, color='white')

# Regression
ax = axes[1]
ax.set_facecolor('#1a1d27')
reg_names = list(reg_results.keys())
rmses = [reg_results[m]['RMSE'] for m in reg_names]
r2s   = [reg_results[m]['R2'] for m in reg_names]
x = np.arange(len(reg_names))
ax.bar(x - w/2, rmses, w, label='RMSE', color='#ef5350', alpha=0.85)
ax.bar(x + w/2, r2s, w, label='R²', color='#66bb6a', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' ','\n') for m in reg_names], fontsize=7)
ax.set_title('Regression — Rating Prediction', color='white')
ax.legend(fontsize=9, framealpha=0.2, labelcolor='white')
ax.grid(axis='y', alpha=0.3)
for i, (r, r2) in enumerate(zip(rmses, r2s)):
    ax.text(i-w/2, r+0.01, f'{r:.2f}', ha='center', fontsize=7, color='white')
    ax.text(i+w/2, r2+0.01, f'{r2:.2f}', ha='center', fontsize=7, color='white')

save_fig('fig8_model_comparison')
print("  Fig 8: Model Comparison")

# ═══════════════════════════════════════════════════════════════════════════
# 9. BUSINESS INSIGHTS — visual
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Business Intelligence Dashboard', fontsize=14, fontweight='bold', color='white', y=0.98)

# Revenue by category
ax = axes[0,0]
ax.set_facecolor('#1a1d27')
rev = df.groupby('Kategori_Menu')['Harga_Pesanan'].sum().sort_values(ascending=True)
colors_rev = PALETTE[:len(rev)]
ax.barh(rev.index, rev.values/1e6, color=colors_rev, edgecolor='#0f1117')
ax.set_title('Total Revenue by Menu Category (M IDR)', color='white')
ax.set_xlabel('Revenue (Million IDR)', color='#aaaaaa')
ax.grid(axis='x', alpha=0.3)
for i, val in enumerate(rev.values):
    ax.text(val/1e6 + 0.5, i, f'{val/1e6:.1f}M', va='center', fontsize=8, color='#cccccc')

# Rating by Complaint Level
ax = axes[0,1]
ax.set_facecolor('#1a1d27')
comp_order = df.groupby('Tingkat_Keluhan')['Rating_Pelanggan'].mean().sort_values(ascending=False)
ax.bar(range(len(comp_order)), comp_order.values, color=PALETTE[:len(comp_order)], alpha=0.85)
ax.set_xticks(range(len(comp_order)))
ax.set_xticklabels(comp_order.index, rotation=20, ha='right')
ax.set_title('Average Rating by Complaint Level', color='white')
ax.set_ylabel('Avg Rating', color='#aaaaaa')
ax.set_ylim(0, 5.5)
ax.grid(axis='y', alpha=0.3)
for i, val in enumerate(comp_order.values):
    ax.text(i, val+0.05, f'{val:.2f}', ha='center', fontsize=9, color='white')

# Peak hour vs avg wait time
ax = axes[1,0]
ax.set_facecolor('#1a1d27')
peak_wait = df.groupby('Hour')['Waktu_Tunggu_Menit'].mean()
ax.plot(peak_wait.index, peak_wait.values, color=ACCENT, lw=2, marker='o', markersize=4)
ax.fill_between(peak_wait.index, peak_wait.values, alpha=0.15, color=ACCENT)
ax.axhspan(peak_wait.mean(), peak_wait.mean()+50, alpha=0.1, color=HIGHLIGHT, label='Above avg wait')
ax.set_title('Average Wait Time by Hour', color='white')
ax.set_xlabel('Hour of Day', color='#aaaaaa')
ax.set_ylabel('Avg Wait Time (min)', color='#aaaaaa')
ax.grid(alpha=0.3)
ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')

# Promo impact on order completion rate
ax = axes[1,1]
ax.set_facecolor('#1a1d27')
promo_comp = df.groupby(['Status_Promo','Status_Pesanan']).size().unstack(fill_value=0)
promo_pct = promo_comp.div(promo_comp.sum(axis=1), axis=0) * 100
labels = ['No Promo', 'Promo']
bottoms = np.zeros(2)
for i, status in enumerate(promo_pct.columns):
    vals = promo_pct[status].values
    ax.bar(range(2), vals, bottom=bottoms, label=status, color=PALETTE[i], alpha=0.85)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 3:
            ax.text(j, b + v/2, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    bottoms += vals
ax.set_xticks([0,1])
ax.set_xticklabels(labels)
ax.set_title('Order Status by Promo Usage (%)', color='white')
ax.set_ylabel('%', color='#aaaaaa')
ax.legend(fontsize=8, framealpha=0.2, labelcolor='white', bbox_to_anchor=(1,1))
ax.grid(axis='y', alpha=0.3)

save_fig('fig9_business_dashboard')
print("  Fig 9: Business Intelligence Dashboard")

# ═══════════════════════════════════════════════════════════════════════════
# 10. ADVANCED: Rating distribution by all categorical segments
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('Rating Distribution Across Key Segments', fontsize=13, fontweight='bold', color='white')

segments = [('WaitTimeCategory','Wait Time Category'),
            ('DistanceBucket','Distance Bucket'),
            ('Kategori_Menu','Menu Category')]

for ax, (col, label) in zip(axes, segments):
    ax.set_facecolor('#1a1d27')
    cats_u = df[col].dropna().unique()
    for i, cat in enumerate(cats_u):
        data = df[df[col]==cat]['Rating_Pelanggan'].dropna()
        ax.hist(data, bins=12, alpha=0.55, label=str(cat), color=PALETTE[i], density=True)
    ax.set_title(f'Rating by {label}', color='white')
    ax.set_xlabel('Rating', color='#aaaaaa')
    ax.set_ylabel('Density', color='#aaaaaa')
    ax.legend(fontsize=7, framealpha=0.2, labelcolor='white')
    ax.grid(alpha=0.3)

save_fig('fig10_rating_segments')
print("  Fig 10: Rating by Segments")

print("\n" + "="*60)
print("  ALL ANALYSIS COMPLETE")
print("="*60)

# Print summary stats for report
print("\n=== KEY METRICS SUMMARY ===")
print(f"Total Orders   : {len(df):,}")
print(f"Total Revenue  : IDR {df['Harga_Pesanan'].sum():,.0f}")
print(f"Avg Rating     : {df['Rating_Pelanggan'].mean():.3f}")
print(f"Avg Wait Time  : {df['Waktu_Tunggu_Menit'].mean():.1f} min")
print(f"Avg Distance   : {df['Jarak_Kirim_KM'].mean():.2f} km")
print(f"Promo Rate     : {df['Status_Promo'].mean()*100:.1f}%")
print(f"Complaint Rate : {(df['Tingkat_Keluhan'] != 'Tidak Ada').mean()*100:.1f}%")
print(f"Cancellation   : {(df['Status_Pesanan'] == 'Dibatalkan').mean()*100:.1f}%")
print(f"Peak hour      : {df.groupby('Hour').size().idxmax()}:00")

# Best classification model
best_cls = max(cls_results, key=lambda m: cls_results[m]['F1'])
print(f"\nBest classifier : {best_cls} (F1={cls_results[best_cls]['F1']:.4f}, Acc={cls_results[best_cls]['Accuracy']:.4f})")
best_reg = max(reg_results, key=lambda m: reg_results[m]['R2'])
print(f"Best regressor  : {best_reg} (R2={reg_results[best_reg]['R2']:.4f}, RMSE={reg_results[best_reg]['RMSE']:.4f})")
