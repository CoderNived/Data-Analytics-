"""
Student Dropout Analysis — Production-Grade Pipeline
Senior Data Engineer / Analyst Workflow
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay,
                             accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    'active':   '#2ECC71',
    'at-risk':  '#F39C12',
    'dropped':  '#E74C3C',
    'bg':       '#0F1117',
    'panel':    '#1A1D27',
    'text':     '#E8EAF0',
    'muted':    '#7F8C9A',
    'accent':   '#5B8DEF',
    'accent2':  '#A855F7',
    'border':   '#2A2D3A',
}
CAT_COLORS = [PALETTE['active'], PALETTE['at-risk'], PALETTE['dropped']]

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['panel'],
    'axes.edgecolor':    PALETTE['border'],
    'axes.labelcolor':   PALETTE['text'],
    'axes.titlecolor':   PALETTE['text'],
    'text.color':        PALETTE['text'],
    'xtick.color':       PALETTE['muted'],
    'ytick.color':       PALETTE['muted'],
    'grid.color':        PALETTE['border'],
    'grid.alpha':        0.6,
    'legend.facecolor':  PALETTE['panel'],
    'legend.edgecolor':  PALETTE['border'],
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('/mnt/user-data/uploads/student_dropout_dataset.csv')
df = df_raw.copy()
df['enroll_date'] = pd.to_datetime(df['enroll_date'])

# Feature Engineering
df['enroll_month']      = df['enroll_date'].dt.month
df['enroll_quarter']    = df['enroll_date'].dt.quarter
df['assignments_per_course'] = df['completed_assignments'] / df['courses_enrolled'].replace(0, np.nan)
df['engagement_score']  = (
    df['completion_rate'] * 0.40 +
    (df['login_frequency'] / df['login_frequency'].max()) * 0.35 +
    (df['forum_posts_count'] / df['forum_posts_count'].max()) * 0.25
)
df['recency_risk']      = (df['last_activity_days_ago'] / df['last_activity_days_ago'].max())
df['is_inactive']       = (df['last_activity_days_ago'] > 30).astype(int)
df['zero_logins']       = (df['login_frequency'] < 1).astype(int)
df['high_performer']    = ((df['completion_rate'] > 0.7) & (df['login_frequency'] > 5)).astype(int)

num_features = ['age','courses_enrolled','completed_assignments','completion_rate',
                'login_frequency','last_activity_days_ago','forum_posts_count',
                'dropout_score','assignments_per_course','engagement_score']

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor(PALETTE['bg'])

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4,
                       top=0.90, bottom=0.06, left=0.06, right=0.97)

fig.text(0.5, 0.96, 'STUDENT DROPOUT — OVERVIEW DASHBOARD',
         ha='center', va='top', fontsize=20, fontweight='bold',
         color=PALETTE['text'], fontfamily='DejaVu Sans')
fig.text(0.5, 0.93, 'n = 5,000 students · 10 regions · 2024 cohort',
         ha='center', va='top', fontsize=11, color=PALETTE['muted'])

label_counts = df['label_name'].value_counts()[['active','at-risk','dropped']]

# ── KPI tiles ─────────────────────────────────────────────────────────────────
kpis = [
    ('5,000',   'Total Students',    PALETTE['accent']),
    (f"{label_counts['dropped']:,}", 'Dropped Out',      PALETTE['dropped']),
    (f"{label_counts['at-risk']:,}", 'At-Risk',          PALETTE['at-risk']),
    (f"{label_counts['active']:,}",  'Active',           PALETTE['active']),
]
for i, (val, lbl, col) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(PALETTE['panel'])
    for spine in ax.spines.values(): spine.set_color(col); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.58, val, ha='center', va='center', transform=ax.transAxes,
            fontsize=26, fontweight='bold', color=col)
    ax.text(0.5, 0.22, lbl, ha='center', va='center', transform=ax.transAxes,
            fontsize=10, color=PALETTE['muted'])

# ── Pie chart ────────────────────────────────────────────────────────────────
ax_pie = fig.add_subplot(gs[1, 0])
wedges, texts, autotexts = ax_pie.pie(
    label_counts.values,
    labels=label_counts.index,
    colors=CAT_COLORS,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': PALETTE['bg'], 'linewidth': 2},
    textprops={'color': PALETTE['text'], 'fontsize': 9}
)
for at in autotexts: at.set_color(PALETTE['bg']); at.set_fontweight('bold')
ax_pie.set_title('Label Distribution', pad=10)

# ── Dropout rate by region ────────────────────────────────────────────────────
ax_reg = fig.add_subplot(gs[1, 1:3])
reg_drop = df.groupby('region')['label'].mean().sort_values(ascending=True)
bars = ax_reg.barh(reg_drop.index, reg_drop.values,
                   color=[PALETTE['dropped'] if v > 0.66 else PALETTE['at-risk'] if v > 0.63
                          else PALETTE['active'] for v in reg_drop.values],
                   edgecolor=PALETTE['bg'], linewidth=0.8, height=0.65)
ax_reg.axvline(reg_drop.mean(), color=PALETTE['muted'], linestyle='--', lw=1.2, label=f'Avg {reg_drop.mean():.2f}')
ax_reg.set_xlabel('Dropout Rate')
ax_reg.set_title('Dropout Rate by Region')
ax_reg.legend(fontsize=8); ax_reg.grid(axis='x')
for bar, val in zip(bars, reg_drop.values):
    ax_reg.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.2%}', va='center', fontsize=8, color=PALETTE['text'])

# ── Monthly enrollment & dropout ─────────────────────────────────────────────
ax_mon = fig.add_subplot(gs[1, 3])
monthly = df.groupby('enroll_month').agg(students=('label','count'), dropout_rate=('label','mean')).reset_index()
ax_mon.bar(monthly['enroll_month'], monthly['students'], color=PALETTE['accent'], alpha=0.5, label='Students')
ax2 = ax_mon.twinx()
ax2.plot(monthly['enroll_month'], monthly['dropout_rate'], color=PALETTE['dropped'],
         marker='o', markersize=4, lw=2, label='Dropout Rate')
ax_mon.set_xlabel('Month'); ax_mon.set_ylabel('Students', color=PALETTE['accent'])
ax2.set_ylabel('Dropout Rate', color=PALETTE['dropped'])
ax2.tick_params(colors=PALETTE['muted'])
ax_mon.set_title('Monthly Enrollment')
ax_mon.set_xticks(range(1,13))
ax_mon.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

# ── Distribution: completion_rate ─────────────────────────────────────────────
ax_cr = fig.add_subplot(gs[2, 0])
for lbl, col in zip(['active','at-risk','dropped'], CAT_COLORS):
    subset = df[df['label_name'] == lbl]['completion_rate']
    ax_cr.hist(subset, bins=30, alpha=0.65, color=col, label=lbl, edgecolor='none')
ax_cr.set_xlabel('Completion Rate'); ax_cr.set_title('Completion Rate by Label')
ax_cr.legend(fontsize=8); ax_cr.grid(axis='y')

# ── Distribution: login_frequency ────────────────────────────────────────────
ax_lf = fig.add_subplot(gs[2, 1])
for lbl, col in zip(['active','at-risk','dropped'], CAT_COLORS):
    subset = df[df['label_name'] == lbl]['login_frequency']
    ax_lf.hist(subset, bins=30, alpha=0.65, color=col, label=lbl, edgecolor='none')
ax_lf.set_xlabel('Login Frequency'); ax_lf.set_title('Login Frequency by Label')
ax_lf.legend(fontsize=8); ax_lf.grid(axis='y')

# ── Box: last_activity_days_ago ───────────────────────────────────────────────
ax_la = fig.add_subplot(gs[2, 2])
data_box = [df[df['label_name'] == l]['last_activity_days_ago'].values for l in ['active','at-risk','dropped']]
bp = ax_la.boxplot(data_box, patch_artist=True, labels=['Active','At-Risk','Dropped'],
                   medianprops={'color': PALETTE['bg'], 'linewidth': 2})
for patch, col in zip(bp['boxes'], CAT_COLORS):
    patch.set_facecolor(col); patch.set_alpha(0.8)
for element in ['whiskers','caps','fliers']:
    for item in bp[element]: item.set_color(PALETTE['muted'])
ax_la.set_ylabel('Days Since Last Activity')
ax_la.set_title('Inactivity by Label'); ax_la.grid(axis='y')

# ── Bar: forum_posts ──────────────────────────────────────────────────────────
ax_fp = fig.add_subplot(gs[2, 3])
seg_means = df.groupby('label_name')[['forum_posts_count','engagement_score']].mean()
x = np.arange(3)
w = 0.35
bars1 = ax_fp.bar(x - w/2, seg_means.loc[['active','at-risk','dropped'],'forum_posts_count'],
                  width=w, color=CAT_COLORS, alpha=0.85, label='Forum Posts')
ax_fp2 = ax_fp.twinx()
ax_fp2.plot(x, seg_means.loc[['active','at-risk','dropped'],'engagement_score'],
            color=PALETTE['accent2'], marker='D', ms=6, lw=2, label='Engagement')
ax_fp.set_xticks(x); ax_fp.set_xticklabels(['Active','At-Risk','Dropped'], fontsize=9)
ax_fp.set_ylabel('Avg Forum Posts')
ax_fp2.set_ylabel('Avg Engagement Score', color=PALETTE['accent2'])
ax_fp2.tick_params(colors=PALETTE['muted'])
ax_fp.set_title('Forum Posts & Engagement')
ax_fp.grid(axis='y')

plt.savefig('/home/claude/fig1_overview.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print('Fig 1 saved.')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — CORRELATION & DEEP EDA
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
fig2.patch.set_facecolor(PALETTE['bg'])
fig2.suptitle('CORRELATION ANALYSIS & DEEP EDA', fontsize=18, fontweight='bold',
              color=PALETTE['text'], y=0.98)

# Heatmap
corr_cols = ['age','courses_enrolled','completed_assignments','completion_rate',
             'login_frequency','last_activity_days_ago','forum_posts_count',
             'dropout_score','engagement_score','label']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(10, 150, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=axes2[0,0], cmap=cmap, center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor=PALETTE['bg'],
            cbar_kws={'shrink': 0.8})
axes2[0,0].set_title('Feature Correlation Heatmap')
axes2[0,0].tick_params(labelsize=8, rotation=30)

# Dropout score distribution by label
ax = axes2[0,1]
for lbl, col in zip(['active','at-risk','dropped'], CAT_COLORS):
    vals = df[df['label_name']==lbl]['dropout_score']
    ax.hist(vals, bins=40, alpha=0.7, color=col, label=lbl, edgecolor='none', density=True)
ax.set_xlabel('Dropout Score (0-1)')
ax.set_title('Dropout Score Distribution')
ax.legend(fontsize=9); ax.grid(axis='y')

# Age distribution
ax = axes2[0,2]
for lbl, col in zip(['active','at-risk','dropped'], CAT_COLORS):
    vals = df[df['label_name']==lbl]['age']
    ax.hist(vals, bins=20, alpha=0.65, color=col, label=lbl, edgecolor='none', density=True)
ax.set_xlabel('Age'); ax.set_title('Age Distribution by Label')
ax.legend(fontsize=9); ax.grid(axis='y')

# Engagement score vs dropout score scatter
ax = axes2[1,0]
colors_map = {'active': PALETTE['active'], 'at-risk': PALETTE['at-risk'], 'dropped': PALETTE['dropped']}
for lbl in ['dropped','at-risk','active']:
    sub = df[df['label_name']==lbl]
    ax.scatter(sub['engagement_score'], sub['dropout_score'],
               c=colors_map[lbl], alpha=0.25, s=12, label=lbl, edgecolors='none')
ax.set_xlabel('Engagement Score (engineered)')
ax.set_ylabel('Dropout Score')
ax.set_title('Engagement vs Dropout Score')
ax.legend(fontsize=9, markerscale=2); ax.grid(True)

# Completion rate vs login frequency
ax = axes2[1,1]
for lbl in ['dropped','at-risk','active']:
    sub = df[df['label_name']==lbl]
    ax.scatter(sub['login_frequency'], sub['completion_rate'],
               c=colors_map[lbl], alpha=0.2, s=12, label=lbl, edgecolors='none')
ax.set_xlabel('Login Frequency'); ax.set_ylabel('Completion Rate')
ax.set_title('Login Frequency vs Completion Rate')
ax.legend(fontsize=9, markerscale=2); ax.grid(True)

# Assignments per course violin
ax = axes2[1,2]
data_vio = [df[df['label_name']==l]['assignments_per_course'].dropna().values
            for l in ['active','at-risk','dropped']]
parts = ax.violinplot(data_vio, positions=[1,2,3], showmedians=True, showextrema=True)
for pc, col in zip(parts['bodies'], CAT_COLORS):
    pc.set_facecolor(col); pc.set_alpha(0.7)
parts['cmedians'].set_color(PALETTE['text'])
parts['cmaxes'].set_color(PALETTE['muted'])
parts['cmins'].set_color(PALETTE['muted'])
parts['cbars'].set_color(PALETTE['muted'])
ax.set_xticks([1,2,3]); ax.set_xticklabels(['Active','At-Risk','Dropped'])
ax.set_ylabel('Assignments per Course')
ax.set_title('Assignments/Course Distribution'); ax.grid(axis='y')

fig2.tight_layout(rect=[0,0,1,0.96])
plt.savefig('/home/claude/fig2_eda.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print('Fig 2 saved.')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — ML MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
feature_cols = ['age','courses_enrolled','completed_assignments','completion_rate',
                'login_frequency','last_activity_days_ago','forum_posts_count',
                'enroll_month','enroll_quarter','assignments_per_course',
                'engagement_score','recency_risk','is_inactive','zero_logins',
                'exam_season']
X = df[feature_cols].fillna(df[feature_cols].median())
y_binary = df['label']
y_multi  = df['label_multiclass']

X_tr, X_te, y_tr, y_te = train_test_split(X, y_binary, test_size=0.2,
                                            random_state=42, stratify=y_binary)
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X, y_multi, test_size=0.2,
                                                    random_state=42, stratify=y_multi)

# Binary
pipe_rf = Pipeline([('scaler', RobustScaler()),
                    ('clf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                                   random_state=42, n_jobs=-1))])
pipe_lr = Pipeline([('scaler', RobustScaler()),
                    ('clf', LogisticRegression(max_iter=500, random_state=42))])
pipe_gb = Pipeline([('scaler', RobustScaler()),
                    ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                       random_state=42))])
models = {'Random Forest': pipe_rf, 'Logistic Reg.': pipe_lr, 'Gradient Boost': pipe_gb}

results = {}
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:,1]
    results[name] = {
        'accuracy': accuracy_score(y_te, y_pred),
        'auc':      roc_auc_score(y_te, y_prob),
        'fpr':      roc_curve(y_te, y_prob)[0],
        'tpr':      roc_curve(y_te, y_prob)[1],
        'cm':       confusion_matrix(y_te, y_pred),
        'report':   classification_report(y_te, y_pred, output_dict=True),
    }

# Feature importance from RF
rf_model    = pipe_rf.named_steps['clf']
feat_imp    = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)

# Multiclass RF
pipe_rf_m = Pipeline([('scaler', RobustScaler()),
                       ('clf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                                      random_state=42, n_jobs=-1))])
pipe_rf_m.fit(X_tr_m, y_tr_m)
y_pred_m = pipe_rf_m.predict(X_te_m)

fig3, axes3 = plt.subplots(2, 3, figsize=(20, 12))
fig3.patch.set_facecolor(PALETTE['bg'])
fig3.suptitle('PREDICTIVE MODELING — RANDOM FOREST & COMPARISON', fontsize=18,
              fontweight='bold', color=PALETTE['text'], y=0.98)

# ROC curves
ax = axes3[0,0]
colors_roc = [PALETTE['accent'], PALETTE['accent2'], PALETTE['at-risk']]
for (name, res), col in zip(results.items(), colors_roc):
    ax.plot(res['fpr'], res['tpr'], lw=2, color=col,
            label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Binary Classification')
ax.legend(fontsize=9); ax.grid(True)

# Model comparison bar
ax = axes3[0,1]
names  = list(results.keys())
accs   = [results[n]['accuracy'] for n in names]
aucs   = [results[n]['auc'] for n in names]
x      = np.arange(len(names))
w      = 0.35
ax.bar(x - w/2, accs, width=w, color=PALETTE['accent'],  alpha=0.85, label='Accuracy')
ax.bar(x + w/2, aucs, width=w, color=PALETTE['accent2'], alpha=0.85, label='AUC-ROC')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.7, 1.0); ax.set_title('Model Performance Comparison')
ax.legend(fontsize=9); ax.grid(axis='y')
for bar in ax.patches:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# Feature importance (top 12)
ax = axes3[0,2]
top12 = feat_imp.tail(12)
bars = ax.barh(top12.index, top12.values,
               color=[PALETTE['accent'] if v > top12.median() else PALETTE['accent2'] for v in top12.values],
               edgecolor=PALETTE['bg'], height=0.65)
ax.set_xlabel('Importance'); ax.set_title('Feature Importance (Random Forest)')
ax.grid(axis='x')

# Confusion Matrix — Binary RF
ax = axes3[1,0]
cm_rf = results['Random Forest']['cm']
sns.heatmap(cm_rf, annot=True, fmt='d', ax=ax,
            cmap=sns.light_palette(PALETTE['accent'], as_cmap=True),
            linewidths=1, linecolor=PALETTE['bg'],
            xticklabels=['Active','Dropout'], yticklabels=['Active','Dropout'])
ax.set_title('Confusion Matrix — RF Binary'); ax.set_ylabel('True'); ax.set_xlabel('Predicted')

# Confusion Matrix — Multiclass RF
ax = axes3[1,1]
cm_m = confusion_matrix(y_te_m, y_pred_m)
sns.heatmap(cm_m, annot=True, fmt='d', ax=ax,
            cmap=sns.light_palette(PALETTE['accent2'], as_cmap=True),
            linewidths=1, linecolor=PALETTE['bg'],
            xticklabels=['Active','At-Risk','Dropped'],
            yticklabels=['Active','At-Risk','Dropped'])
ax.set_title('Confusion Matrix — RF Multiclass')
ax.set_ylabel('True'); ax.set_xlabel('Predicted')

# Dropout score vs predicted probability
ax = axes3[1,2]
probs = pipe_rf.predict_proba(X_te)[:,1]
ax.scatter(df.loc[X_te.index, 'dropout_score'], probs,
           c=[PALETTE['dropped'] if t==1 else PALETTE['active'] for t in y_te.values],
           alpha=0.25, s=10, edgecolors='none')
ax.plot([0,1],[0,1], 'w--', lw=1.5, alpha=0.5, label='Perfect calibration')
ax.set_xlabel('True Dropout Score (label)')
ax.set_ylabel('RF Predicted Probability')
ax.set_title('Calibration: True vs Predicted')
patches = [mpatches.Patch(color=PALETTE['dropped'], label='Dropout'),
           mpatches.Patch(color=PALETTE['active'],  label='Active')]
ax.legend(handles=patches, fontsize=9); ax.grid(True)

fig3.tight_layout(rect=[0,0,1,0.96])
plt.savefig('/home/claude/fig3_models.png', dpi=150, bbox_inches='tight',
            facecolor=PALETTE['bg'])
plt.close()
print('Fig 3 saved.')
print('\nBinary RF Accuracy:', results['Random Forest']['accuracy'])
print('Binary RF AUC:     ', results['Random Forest']['auc'])
print('\nMulticlass RF Report:')
print(classification_report(y_te_m, y_pred_m, target_names=['Active','At-Risk','Dropped']))
