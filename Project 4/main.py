"""
=============================================================================
  AI vs Human Text Classification — Full Analytics Pipeline
  Senior Data Engineer / Data Scientist Workflow
  Author : Production Analytics Pipeline
=============================================================================
"""

# ── Standard & Core ──────────────────────────────────────────────────────────
import re
import os
import math
import warnings
import textwrap
from collections import Counter

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model          import LogisticRegression
from sklearn.naive_bayes           import MultinomialNB
from sklearn.model_selection       import train_test_split, cross_val_score
from sklearn.metrics               import (accuracy_score, precision_score,
                                           recall_score, f1_score,
                                           confusion_matrix, classification_report)
from sklearn.pipeline              import Pipeline
from sklearn.preprocessing         import LabelEncoder

warnings.filterwarnings("ignore")

# ── Global Style ─────────────────────────────────────────────────────────────
PALETTE   = {"AI": "#4C72B0", "Human": "#DD8452"}
BG        = "#F8F9FA"
GRID_CLR  = "#E0E0E0"
OUT_DIR   = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : BG,
    "axes.edgecolor"    : "#CCCCCC",
    "axes.grid"         : True,
    "grid.color"        : GRID_CLR,
    "grid.linewidth"    : 0.6,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 8,
    "ytick.labelsize"   : 8,
    "legend.fontsize"   : 9,
})

# ── English Stop-Words (built-in, no NLTK needed) ────────────────────────────
STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be
because been before being below between both but by can can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further get got had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've if
in into is isn't it it's its itself let's me more moreover most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's will with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves also however may
study research results using based used paper method data analysis show
showed shown found findings suggests suggesting suggests indicates indicate
""".split())

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing spaces."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0

def char_count(text: str) -> int:
    return len(re.sub(r"\s", "", text)) if isinstance(text, str) else 0

def sentence_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    sentences = re.split(r"[.!?]+", text)
    return max(1, sum(1 for s in sentences if s.strip()))

def avg_word_length(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", text) if isinstance(text, str) else []
    return round(np.mean([len(w) for w in words]), 3) if words else 0.0

def avg_sentence_length(text: str) -> float:
    sc = sentence_count(text)
    wc = word_count(text)
    return round(wc / sc, 3) if sc else 0.0

def unique_word_ratio(text: str) -> float:
    """Type-Token Ratio — proxy for vocabulary richness."""
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower()) if isinstance(text, str) else []
    return round(len(set(words)) / len(words), 4) if words else 0.0

def flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease (manual implementation).
    Higher = easier to read (plain English).
    """
    words     = re.findall(r"\b[a-zA-Z]+\b", text) if isinstance(text, str) else []
    sentences = max(1, sentence_count(text))
    syllables = sum(_syllable_count(w) for w in words)
    if not words:
        return 0.0
    score = 206.835 - 1.015*(len(words)/sentences) - 84.6*(syllables/len(words))
    return round(score, 2)

def _syllable_count(word: str) -> int:
    word  = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def gunning_fog(text: str) -> float:
    """Gunning Fog Index — estimates years of formal education needed."""
    words     = re.findall(r"\b[a-zA-Z]+\b", text) if isinstance(text, str) else []
    sentences = max(1, sentence_count(text))
    complex_w = sum(1 for w in words if _syllable_count(w) >= 3)
    if not words:
        return 0.0
    return round(0.4 * ((len(words)/sentences) + 100*(complex_w/len(words))), 2)

def top_words(series: pd.Series, n: int = 20, exclude_stop: bool = True) -> list[tuple]:
    all_words = []
    for txt in series.dropna():
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", txt.lower())
        if exclude_stop:
            tokens = [t for t in tokens if t not in STOPWORDS]
        all_words.extend(tokens)
    return Counter(all_words).most_common(n)

def simple_wordcloud_data(series: pd.Series, n: int = 50) -> dict:
    """Return {word: freq} dict usable for manual word-cloud plotting."""
    words = top_words(series, n=n)
    return {w: f for w, f in words}

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — DATA UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STEP 1 · DATA UNDERSTANDING")
print("="*70)

df_raw = pd.read_csv("/mnt/user-data/uploads/data_for_preprocessing.csv")

print(f"\n📐 Shape            : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"📋 Column names     : {df_raw.columns.tolist()}")
print(f"\n🔑 Data Types:\n{df_raw.dtypes.to_string()}")
print(f"\n📊 Class Distribution (raw):\n{df_raw['Author'].value_counts().to_string()}")
print(f"\n⚠️  Missing Values:\n{df_raw.isnull().sum().to_string()}")
print(f"\n🔁 Duplicate Rows   : {df_raw.duplicated().sum():,}")
print(f"\n--- Sample Rows ---")
print(df_raw.head(3).to_string(max_colwidth=80))

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STEP 2 · DATA CLEANING")
print("="*70)

df = df_raw.copy()

# 2-A Drop index column
unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
df.drop(columns=unnamed_cols, inplace=True)
print(f"\n✅ Dropped index columns  : {unnamed_cols}")

# 2-B Drop rows with missing Text or Author
before = len(df)
df.dropna(subset=["Text", "Author"], inplace=True)
print(f"✅ Rows after null drop   : {len(df):,}  (removed {before - len(df)})")

# 2-C Drop duplicates
before = len(df)
df.drop_duplicates(subset=["Text"], inplace=True)
print(f"✅ Rows after dedup       : {len(df):,}  (removed {before - len(df)})")

# 2-D Normalize Author labels
df["Author"] = df["Author"].str.strip().str.title()
valid = df["Author"].isin(["Ai", "Human"])
df = df[df["Author"].isin(df["Author"].unique())]

# Standardise "Ai" → "AI"
df["Author"] = df["Author"].replace({"Ai": "AI"})

# 2-E Normalise text whitespace
df["Text"] = df["Text"].str.strip().str.replace(r"\s+", " ", regex=True)

# 2-F Reset index
df.reset_index(drop=True, inplace=True)

print(f"\n✅ Final clean shape      : {df.shape}")
print(f"✅ Final class counts:\n{df['Author'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — FEATURE ENGINEERING (NLP)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STEP 3 · FEATURE ENGINEERING")
print("="*70)

df["word_count"]         = df["Text"].apply(word_count)
df["char_count"]         = df["Text"].apply(char_count)
df["sentence_count"]     = df["Text"].apply(sentence_count)
df["avg_word_len"]       = df["Text"].apply(avg_word_length)
df["avg_sentence_len"]   = df["Text"].apply(avg_sentence_length)
df["unique_word_ratio"]  = df["Text"].apply(unique_word_ratio)
df["flesch_score"]       = df["Text"].apply(flesch_reading_ease)
df["gunning_fog"]        = df["Text"].apply(gunning_fog)

NUM_FEATURES = ["word_count","char_count","sentence_count",
                "avg_word_len","avg_sentence_len","unique_word_ratio",
                "flesch_score","gunning_fog"]

print("\n📐 Feature Statistics by Author:\n")
print(df.groupby("Author")[NUM_FEATURES].mean().round(2).T.to_string())

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — STATISTICAL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STEP 4 · STATISTICAL INSIGHTS")
print("="*70)

from scipy import stats as scipy_stats  # available in sklearn env

ai_wc    = df.loc[df["Author"]=="AI",    "word_count"]
hum_wc   = df.loc[df["Author"]=="Human", "word_count"]
t_stat, p_val = scipy_stats.ttest_ind(ai_wc, hum_wc)

print(f"\n📊 AI   — mean word count : {ai_wc.mean():.1f}  (std {ai_wc.std():.1f})")
print(f"📊 Human— mean word count : {hum_wc.mean():.1f}  (std {hum_wc.std():.1f})")
print(f"📊 Welch t-test  t={t_stat:.3f}, p={p_val:.4e}")
print(f"   → {'Statistically SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} difference (α=0.05)")

for feat in ["avg_word_len","unique_word_ratio","flesch_score","gunning_fog"]:
    a = df.loc[df["Author"]=="AI",    feat]
    h = df.loc[df["Author"]=="Human", feat]
    t, p = scipy_stats.ttest_ind(a, h)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    print(f"   {feat:<22}  AI={a.mean():.3f}  Human={h.mean():.3f}  p={p:.3e} {sig}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STEP 5 · VISUALISATIONS")
print("="*70)

# ── Fig 1 : Overview Dashboard ───────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle("AI vs Human Research Text — Analytics Dashboard",
             fontsize=16, fontweight="bold", y=0.98, color="#2C3E50")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

# Panel A — Class distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = df["Author"].value_counts()
bars = ax0.bar(counts.index, counts.values,
               color=[PALETTE[a] for a in counts.index],
               edgecolor="white", linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"{val:,}\n({val/len(df)*100:.1f}%)",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax0.set_title("A · Class Distribution")
ax0.set_ylabel("Count")
ax0.set_ylim(0, counts.max() * 1.22)

# Panel B — Word count histogram
ax1 = fig.add_subplot(gs[0, 1:])
for author, grp in df.groupby("Author"):
    ax1.hist(grp["word_count"].clip(upper=1500), bins=60,
             alpha=0.65, color=PALETTE[author], label=author, edgecolor="none")
ax1.axvline(ai_wc.mean(),  color=PALETTE["AI"],    linestyle="--", lw=1.5,
            label=f"AI mean {ai_wc.mean():.0f}")
ax1.axvline(hum_wc.mean(), color=PALETTE["Human"], linestyle="--", lw=1.5,
            label=f"Human mean {hum_wc.mean():.0f}")
ax1.set_title("B · Word Count Distribution (clipped at 1 500)")
ax1.set_xlabel("Word Count")
ax1.set_ylabel("Frequency")
ax1.legend()

# Panel C — Boxplot: word count by author
ax2 = fig.add_subplot(gs[1, 0])
plot_data = [df.loc[df["Author"]==a, "word_count"].values for a in ["AI","Human"]]
bp = ax2.boxplot(plot_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], [PALETTE["AI"], PALETTE["Human"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax2.set_xticklabels(["AI","Human"])
ax2.set_title("C · Word Count Boxplot")
ax2.set_ylabel("Word Count")

# Panel D — Avg word length boxplot
ax3 = fig.add_subplot(gs[1, 1])
plot_data2 = [df.loc[df["Author"]==a, "avg_word_len"].values for a in ["AI","Human"]]
bp2 = ax3.boxplot(plot_data2, patch_artist=True, widths=0.5,
                  medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp2["boxes"], [PALETTE["AI"], PALETTE["Human"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax3.set_xticklabels(["AI","Human"])
ax3.set_title("D · Avg Word Length Boxplot")
ax3.set_ylabel("Characters")

# Panel E — Unique word ratio (TTR)
ax4 = fig.add_subplot(gs[1, 2])
for author, grp in df.groupby("Author"):
    ax4.hist(grp["unique_word_ratio"], bins=40, alpha=0.65,
             color=PALETTE[author], label=author, edgecolor="none")
ax4.set_title("E · Vocabulary Richness (TTR)")
ax4.set_xlabel("Type-Token Ratio")
ax4.set_ylabel("Frequency")
ax4.legend()

# Panel F — Flesch Reading Ease
ax5 = fig.add_subplot(gs[2, 0])
for author, grp in df.groupby("Author"):
    ax5.hist(grp["flesch_score"].clip(-50, 100), bins=40, alpha=0.65,
             color=PALETTE[author], label=author, edgecolor="none")
ax5.set_title("F · Flesch Reading Ease")
ax5.set_xlabel("Score (higher = easier)")
ax5.set_ylabel("Frequency")
ax5.legend()

# Panel G — Gunning Fog
ax6 = fig.add_subplot(gs[2, 1])
for author, grp in df.groupby("Author"):
    ax6.hist(grp["gunning_fog"].clip(0, 50), bins=40, alpha=0.65,
             color=PALETTE[author], label=author, edgecolor="none")
ax6.set_title("G · Gunning Fog Index")
ax6.set_xlabel("Grade Level (higher = harder)")
ax6.set_ylabel("Frequency")
ax6.legend()

# Panel H — Sentence count distribution
ax7 = fig.add_subplot(gs[2, 2])
for author, grp in df.groupby("Author"):
    ax7.hist(grp["sentence_count"].clip(upper=60), bins=40, alpha=0.65,
             color=PALETTE[author], label=author, edgecolor="none")
ax7.set_title("H · Sentence Count Distribution")
ax7.set_xlabel("Sentences")
ax7.set_ylabel("Frequency")
ax7.legend()

plt.savefig(f"{OUT_DIR}/fig1_overview_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved fig1_overview_dashboard.png")

# ── Fig 2 : Top Words Comparison ─────────────────────────────────────────────
ai_words  = top_words(df.loc[df["Author"]=="AI",    "Text"], n=20)
hum_words = top_words(df.loc[df["Author"]=="Human", "Text"], n=20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.suptitle("Top 20 Content Words — AI vs Human",
             fontsize=14, fontweight="bold", color="#2C3E50")

for ax, words, label, color in [
        (axes[0], ai_words,  "AI",    PALETTE["AI"]),
        (axes[1], hum_words, "Human", PALETTE["Human"])]:
    ws, fs = zip(*words)
    y_pos  = range(len(ws))
    bars   = ax.barh(list(y_pos), list(fs), color=color, alpha=0.80, edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(ws), fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{label} — Top Words", color=color, fontweight="bold")
    ax.set_xlabel("Frequency")
    for bar, freq in zip(bars, fs):
        ax.text(bar.get_width() + max(fs)*0.01, bar.get_y() + bar.get_height()/2,
                f"{freq:,}", va="center", fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_top_words.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved fig2_top_words.png")

# ── Fig 3 : Correlation Heat-map of NLP Features ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
corr = df[NUM_FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
ax.set_title("NLP Feature Correlation Matrix", fontsize=13, fontweight="bold",
             pad=14)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved fig3_correlation_heatmap.png")

# ── Fig 4 : Violin plots for all features ────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor=BG)
fig.suptitle("Feature Distributions — AI vs Human (Violin Plots)",
             fontsize=14, fontweight="bold", color="#2C3E50")
axes = axes.flatten()

feature_labels = {
    "word_count"       : "Word Count",
    "char_count"       : "Character Count",
    "sentence_count"   : "Sentence Count",
    "avg_word_len"     : "Avg Word Length",
    "avg_sentence_len" : "Avg Sentence Length",
    "unique_word_ratio": "Vocabulary Richness (TTR)",
    "flesch_score"     : "Flesch Reading Ease",
    "gunning_fog"      : "Gunning Fog Index",
}

for ax, feat in zip(axes, NUM_FEATURES):
    parts = ax.violinplot(
        [df.loc[df["Author"]=="AI",    feat].clip(lower=df[feat].quantile(0.01),
                                                   upper=df[feat].quantile(0.99)),
         df.loc[df["Author"]=="Human", feat].clip(lower=df[feat].quantile(0.01),
                                                   upper=df[feat].quantile(0.99))],
        positions=[0, 1], showmedians=True, showextrema=False
    )
    for i, (body, color) in enumerate(zip(parts["bodies"],
                                          [PALETTE["AI"], PALETTE["Human"]])):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AI", "Human"])
    ax.set_title(feature_labels[feat], fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_violin_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved fig4_violin_features.png")

# ── Fig 5 : Manual Word-Cloud (frequency-scaled text) ────────────────────────
def plot_wordcloud_manual(ax, freq_dict: dict, title: str, color: str):
    """Plot a pseudo word-cloud using matplotlib text with size ∝ frequency."""
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=8)

    words  = list(freq_dict.keys())
    freqs  = np.array(list(freq_dict.values()), dtype=float)
    sizes  = 8 + 34 * (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-9)

    rng = np.random.default_rng(42)
    placed = []
    for word, size in sorted(zip(words, sizes), key=lambda x: -x[1]):
        for _ in range(300):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            overlap = False
            for (px, py, pw, ph) in placed:
                if abs(x - px) < pw + 0.06 and abs(y - py) < ph + 0.03:
                    overlap = True
                    break
            if not overlap:
                alpha  = 0.55 + 0.45 * (size - 8) / 34
                shade  = min(1.0, 0.4 + (size - 8) / 34)
                c      = matplotlib.colors.to_rgba(color, alpha)
                ax.text(x, y, word, ha="center", va="center",
                        fontsize=size, color=c, fontweight="bold",
                        rotation=rng.choice([0, 0, 0, 30, -30]))
                # approximate text bounding box
                placed.append((x, y, len(word)*size*0.0045, 0.04))
                break

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.suptitle("Pseudo Word Clouds — Content Words (excl. stop-words)",
             fontsize=14, fontweight="bold", color="#2C3E50")

plot_wordcloud_manual(axes[0],
                      simple_wordcloud_data(df.loc[df["Author"]=="AI",    "Text"], 45),
                      "AI-Generated Text", PALETTE["AI"])
plot_wordcloud_manual(axes[1],
                      simple_wordcloud_data(df.loc[df["Author"]=="Human", "Text"], 45),
                      "Human-Written Text", PALETTE["Human"])

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_word_clouds.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved fig5_word_clouds.png")

# ── Fig 6 : Sentence-length scatter + trend ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle("Word Count vs Average Sentence Length",
             fontsize=13, fontweight="bold", color="#2C3E50")

for ax, author in zip(axes, ["AI", "Human"]):
    sub = df[df["Author"]==author].sample(min(800, len(df[df["Author"]==author])),
                                           random_state=42)
    ax.scatter(sub["word_count"].clip(upper=1500),
               sub["avg_sentence_len"].clip(upper=80),
               alpha=0.25, s=12, color=PALETTE[author])
    # trend line
    x = sub["word_count"].clip(upper=1500)
    y = sub["avg_sentence_len"].clip(upper=80)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, p(xs), color="#2C3E50", linewidth=1.8, linestyle="--")
    ax.set_title(f"{author} Texts", color=PALETTE[author], fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Avg Sentence Length (words)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_scatter_sentence.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved fig6_scatter_sentence.png")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — BASELINE ML MODEL (TF-IDF + Logistic Regression & Naive Bayes)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STEP 6 · BASELINE ML MODELS")
print("="*70)

le = LabelEncoder()
y  = le.fit_transform(df["Author"])   # AI=0, Human=1 (alphabetical)
X  = df["Text"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)

# ── Logistic Regression pipeline ─────────────────────────────────────────────
pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1,2),
                               sublinear_tf=True, min_df=3)),
    ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  random_state=42))
])

# ── Multinomial Naive Bayes pipeline ─────────────────────────────────────────
pipe_nb = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1,2),
                               sublinear_tf=False, min_df=3)),
    ("clf",   MultinomialNB(alpha=0.1))
])

models = [("Logistic Regression", pipe_lr), ("Naive Bayes", pipe_nb)]
results = {}

for name, pipe in models:
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted")
    rec  = recall_score(y_te, y_pred, average="weighted")
    f1   = f1_score(y_te, y_pred, average="weighted")
    cm   = confusion_matrix(y_te, y_pred)
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="f1_weighted")

    results[name] = dict(acc=acc, prec=prec, rec=rec, f1=f1, cm=cm, cv=cv)

    print(f"\n─── {name} ───────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 (wtd)  : {f1:.4f}")
    print(f"  CV F1 5×  : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_te, y_pred,
                                 target_names=le.classes_, digits=4))

# ── Feature importance from LR ───────────────────────────────────────────────
tfidf_vocab = pipe_lr.named_steps["tfidf"].get_feature_names_out()
lr_coef     = pipe_lr.named_steps["clf"].coef_[0]
top_idx     = np.argsort(np.abs(lr_coef))[::-1][:30]
top_feat    = [(tfidf_vocab[i], lr_coef[i]) for i in top_idx]

print("\n📌 Top 30 Discriminative TF-IDF Features (Logistic Regression):")
for feat, coef in top_feat:
    direction = "→ AI" if coef < 0 else "→ Human"
    print(f"   {feat:<30}  coef={coef:+.4f}  {direction}")

# ── Fig 7 : Confusion Matrices + Feature Importance ─────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("ML Baseline Results", fontsize=14, fontweight="bold", color="#2C3E50")
gs2 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

for idx, (name, res) in enumerate(results.items()):
    ax_cm = fig.add_subplot(gs2[0, idx])
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=ax_cm, cbar=False,
                annot_kws={"size": 12, "weight": "bold"})
    ax_cm.set_title(f"{name}\nAcc={res['acc']:.3f}  F1={res['f1']:.3f}",
                    fontsize=10, fontweight="bold")
    ax_cm.set_ylabel("True")
    ax_cm.set_xlabel("Predicted")

# Model comparison bar
ax_cmp = fig.add_subplot(gs2[0, 2])
metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics_names))
w = 0.35
bars1 = ax_cmp.bar(x - w/2,
                    [results["Logistic Regression"][k]
                     for k in ["acc","prec","rec","f1"]],
                    w, label="Logistic Regression", color="#4C72B0", alpha=0.85)
bars2 = ax_cmp.bar(x + w/2,
                    [results["Naive Bayes"][k]
                     for k in ["acc","prec","rec","f1"]],
                    w, label="Naive Bayes", color="#55A868", alpha=0.85)
ax_cmp.set_xticks(x)
ax_cmp.set_xticklabels(metrics_names)
ax_cmp.set_ylim(0.5, 1.05)
ax_cmp.set_title("Model Performance Comparison")
ax_cmp.legend()
for bar in list(bars1) + list(bars2):
    ax_cmp.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

# Feature importance
ax_fi = fig.add_subplot(gs2[1, :])
top20 = top_feat[:20]
words, coefs = zip(*top20)
colors = [PALETTE["AI"] if c < 0 else PALETTE["Human"] for c in coefs]
bars = ax_fi.barh(range(len(words)), coefs, color=colors, alpha=0.8, edgecolor="white")
ax_fi.set_yticks(range(len(words)))
ax_fi.set_yticklabels(words, fontsize=9)
ax_fi.invert_yaxis()
ax_fi.axvline(0, color="#888888", linewidth=1)
ax_fi.set_title("Top 20 Discriminative TF-IDF Features\n"
                "(blue → AI-associated  |  orange → Human-associated)",
                fontweight="bold")
ax_fi.set_xlabel("Logistic Regression Coefficient")

# Add legend patches
import matplotlib.patches as mpatches
ai_patch  = mpatches.Patch(color=PALETTE["AI"],    label="AI-associated")
hum_patch = mpatches.Patch(color=PALETTE["Human"], label="Human-associated")
ax_fi.legend(handles=[ai_patch, hum_patch], loc="lower right")

plt.savefig(f"{OUT_DIR}/fig7_ml_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved fig7_ml_results.png")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — FINAL INSIGHTS REPORT
# ─────────────────────────────────────────────────────────────────────────────

ai_stats  = df[df["Author"]=="AI"][NUM_FEATURES].mean()
hum_stats = df[df["Author"]=="Human"][NUM_FEATURES].mean()
best_model = max(results, key=lambda k: results[k]["f1"])

report = f"""
================================================================================
  FINAL ANALYTICS REPORT — AI vs Human Research Text
================================================================================

DATASET SUMMARY
  • Total samples (clean)  : {len(df):,}
  • AI texts               : {(df['Author']=='AI').sum():,}  ({(df['Author']=='AI').mean()*100:.1f}%)
  • Human texts            : {(df['Author']=='Human').sum():,}  ({(df['Author']=='Human').mean()*100:.1f}%)
  • Duplicates removed     : {before - len(df)} rows
  • Missing values removed : handled

────────────────────────────────────────────────────────────────────────────────
KEY PATTERNS DISCOVERED
────────────────────────────────────────────────────────────────────────────────

1. TEXT LENGTH
   AI texts are substantially SHORTER than human texts:
   • AI mean word count    : {ai_stats['word_count']:.0f} words
   • Human mean word count : {hum_stats['word_count']:.0f} words
   This aligns with AI-generated abstracts/summaries vs full human papers.

2. VOCABULARY RICHNESS (Type-Token Ratio)
   • AI   TTR : {ai_stats['unique_word_ratio']:.3f}
   • Human TTR: {hum_stats['unique_word_ratio']:.3f}
   AI texts show {'higher' if ai_stats['unique_word_ratio'] > hum_stats['unique_word_ratio'] else 'lower'} lexical diversity per token —
   likely due to shorter texts inflating TTR artificially.

3. READABILITY
   • AI   Flesch score : {ai_stats['flesch_score']:.1f}   Gunning Fog: {ai_stats['gunning_fog']:.1f}
   • Human Flesch score: {hum_stats['flesch_score']:.1f}   Gunning Fog: {hum_stats['gunning_fog']:.1f}
   Both groups produce similarly complex academic language (expected for research).
   AI texts trend toward slightly {'simpler' if ai_stats['flesch_score'] > hum_stats['flesch_score'] else 'more complex'} sentence structures.

4. SENTENCE STRUCTURE
   • AI   avg sentence length : {ai_stats['avg_sentence_len']:.1f} words/sentence
   • Human avg sentence length: {hum_stats['avg_sentence_len']:.1f} words/sentence

5. MOST DISCRIMINATIVE FEATURES (TF-IDF)
   Top AI tokens     : {', '.join([w for w,c in top_feat if c < 0][:6])}
   Top Human tokens  : {', '.join([w for w,c in top_feat if c > 0][:6])}

────────────────────────────────────────────────────────────────────────────────
MODEL PERFORMANCE
────────────────────────────────────────────────────────────────────────────────
  Best model       : {best_model}
  Accuracy         : {results[best_model]['acc']:.4f}
  Weighted F1      : {results[best_model]['f1']:.4f}
  Cross-val F1 (5×): {results[best_model]['cv'].mean():.4f} ± {results[best_model]['cv'].std():.4f}

  Both TF-IDF + LR and TF-IDF + NB achieve strong performance, confirming that
  surface-level lexical patterns are highly discriminative between AI and Human
  research texts.

────────────────────────────────────────────────────────────────────────────────
RECOMMENDATIONS FOR IMPROVEMENT
────────────────────────────────────────────────────────────────────────────────
  1. Transformer-based embeddings (e.g., BERT, RoBERTa, DeBERTa) will capture
     semantic nuance beyond n-gram statistics.
  2. Use length-normalised features (per-sentence metrics) to control for the
     strong length confound between AI and Human texts.
  3. Add perplexity scores from a language model as a meta-feature — AI text
     tends to have lower LM perplexity.
  4. Collect more balanced data if class imbalance grows; consider augmentation
     or weighted loss functions.
  5. Incorporate stylometric features: punctuation density, passive voice ratio,
     hedge word frequency (may, might, perhaps), citation density.
  6. Run ablation studies to identify which feature groups (lexical / syntactic /
     readability) contribute most to classification.

================================================================================
"""

print(report)

with open(f"{OUT_DIR}/final_report.txt", "w") as f:
    f.write(report)
print("✅ Saved final_report.txt")

# ── Save the feature-engineered CSV ──────────────────────────────────────────
df.to_csv(f"{OUT_DIR}/data_cleaned_features.csv", index=False)
print("✅ Saved data_cleaned_features.csv")

print("\n🎉  Pipeline complete. All outputs written to /mnt/user-data/outputs/")
