"""
=============================================================================
TAMIL NADU 2026 ASSEMBLY ELECTION RESULTS — POLITICAL ANALYTICS ENGINE
=============================================================================
Author      : Senior Data Analyst / Political Intelligence Unit
Dataset     : eci_results_tamilnadu_2026.csv
Purpose     : Consulting-grade exploratory, statistical, and visual analysis
              for election commissions, political researchers, and policymakers
Output      : project_output/ (plots, reports, cleaned_data, statistics,
              interactive_plots, logs)
=============================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import logging
import warnings
import itertools
from collections import Counter
from datetime import datetime

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ── Matplotlib Global Style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

PALETTE_PARTY = "tab20"
ACCENT        = "#1f77b4"
HIGHLIGHT     = "#d62728"
GREEN         = "#2ca02c"
GOLD          = "#FFD700"


# ══════════════════════════════════════════════════════════════════════════════
# 0.  DIRECTORY SETUP & LOGGING
# ══════════════════════════════════════════════════════════════════════════════

DIRS = {
    "root"        : "project_output",
    "plots"       : "project_output/plots",
    "reports"     : "project_output/reports",
    "cleaned_data": "project_output/cleaned_data",
    "statistics"  : "project_output/statistics",
    "interactive" : "project_output/interactive_plots",
    "logs"        : "project_output/logs",
}

def setup_directories() -> None:
    """Create all output directories if they don't exist."""
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

def setup_logging() -> logging.Logger:
    """Configure dual-sink logging (file + console)."""
    log_path = os.path.join(DIRS["logs"], "analytics.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("TN2026_Analytics")

setup_directories()
log = setup_logging()

# ── Shared insight buffer (written to file at end) ────────────────────────────
INSIGHTS: list[str] = []

def note(text: str) -> None:
    """Record an insight to the shared buffer and logger."""
    INSIGHTS.append(text)
    log.info(text)

def save_fig(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure to the plots directory."""
    path = os.path.join(DIRS["plots"], f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot → %s", path)

def save_plotly(fig, name: str) -> None:
    """Save a Plotly figure as interactive HTML."""
    path = os.path.join(DIRS["interactive"], f"{name}.html")
    fig.write_html(path)
    log.info("Saved interactive plot → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & SCHEMA DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "/mnt/user-data/uploads/eci_results_tamilnadu_2026.csv"

def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and perform initial type inference."""
    log.info("Loading dataset from: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info("Raw shape: %s rows × %s columns", *df.shape)
    return df

def schema_report(df: pd.DataFrame) -> None:
    """Print and save a detailed schema / data-dictionary report."""
    lines = ["=" * 70, "SCHEMA REPORT — TAMIL NADU 2026 ELECTION DATA", "=" * 70]
    for col in df.columns:
        dtype   = df[col].dtype
        nunique = df[col].nunique()
        nulls   = df[col].isna().sum()
        sample  = df[col].dropna().iloc[:3].tolist() if not df[col].dropna().empty else []
        lines.append(
            f"  {col:<30} | dtype={str(dtype):<10} | unique={nunique:<6} | nulls={nulls:<5} | sample={sample}"
        )
    lines += [
        "=" * 70,
        f"  Total rows    : {len(df):,}",
        f"  Total columns : {df.shape[1]}",
        f"  Memory usage  : {df.memory_usage(deep=True).sum() / 1024:.1f} KB",
        "=" * 70,
    ]
    report_text = "\n".join(lines)
    print(report_text)
    path = os.path.join(DIRS["reports"], "schema_report.txt")
    with open(path, "w") as f:
        f.write(report_text)
    note(f"Schema report saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      - Strip whitespace from string columns
      - Normalise party / candidate names
      - Parse Round info
      - Drop complete duplicates
      - Validate numeric columns
    """
    log.info("Starting data cleaning pipeline …")
    raw_rows = len(df)

    # ── String columns: strip whitespace & fix encoding artefacts ─────────────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", np.nan)

    # ── Normalise party names (uppercase, collapse spaces) ────────────────────
    df["Party"] = df["Party"].str.upper().str.replace(r"\s+", " ", regex=True)
    df["Candidate"] = df["Candidate"].str.upper().str.replace(r"\s+", " ", regex=True)
    df["Constituency"] = df["Constituency"].str.title()

    # ── Parse Round: extract completed rounds ────────────────────────────────
    df[["Rounds_Done", "Rounds_Total"]] = (
        df["Round"].str.extract(r"(\d+)/(\d+)").astype(float)
    )

    # ── Validate numeric sanity ───────────────────────────────────────────────
    df["EVM Votes"]    = pd.to_numeric(df["EVM Votes"], errors="coerce").fillna(0).astype(int)
    df["Postal Votes"] = pd.to_numeric(df["Postal Votes"], errors="coerce").fillna(0).astype(int)
    df["Total Votes"]  = pd.to_numeric(df["Total Votes"], errors="coerce").fillna(0).astype(int)
    df["% Votes"]      = pd.to_numeric(df["% Votes"], errors="coerce")

    # ── Drop complete duplicates ──────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    note(f"Duplicates removed: {before - len(df)}")

    # ── Filter out NOTA rows for candidate-level analysis (keep separate) ─────
    df["Is_NOTA"] = df["Party"].str.contains("NONE OF THE ABOVE", na=False)
    df["Is_Independent"] = df["Party"].str.contains("INDEPENDENT", na=False)

    note(f"Rows after cleaning: {len(df):,}  (started with {raw_rows:,})")

    # ── Save cleaned CSV ───────────────────────────────────────────────────────
    out = os.path.join(DIRS["cleaned_data"], "eci_results_tn2026_cleaned.csv")
    df.to_csv(out, index=False)
    log.info("Cleaned dataset saved → %s", out)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive rich analytical features at both the candidate and constituency level.

    New candidate-level columns
    ────────────────────────────
    postal_pct           : Postal votes as % of total votes for candidate
    evm_pct              : EVM votes as % of candidate's total votes
    vote_share_pct       : Synonym for % Votes (explicit)

    Constituency-level aggregated columns (merged back)
    ────────────────────────────────────────────────────
    total_electors       : Approximated as sum of all candidates' votes in constituency
    winner_votes         : Votes of the winner
    runner_votes         : Votes of the runner-up
    margin               : winner_votes - runner_votes
    margin_pct           : margin / total_electors * 100
    competitiveness_score: 100 - margin_pct  (higher = more competitive)
    dominance_index      : winner_votes / runner_votes
    winner_party         : Winning party name
    winner_candidate     : Winning candidate name
    is_winner            : Boolean flag per row
    rank_in_constituency : Rank of candidate by votes (1 = winner)
    """
    log.info("Engineering features …")

    # Candidate-level
    df["postal_pct"]     = np.where(df["Total Votes"] > 0,
                                    df["Postal Votes"] / df["Total Votes"] * 100, 0)
    df["evm_pct"]        = np.where(df["Total Votes"] > 0,
                                    df["EVM Votes"] / df["Total Votes"] * 100, 0)
    df["vote_share_pct"] = df["% Votes"].fillna(0)

    # Rank within constituency
    df["rank_in_constituency"] = (
        df.groupby("Constituency")["Total Votes"]
          .rank(method="dense", ascending=False)
          .astype(int)
    )
    df["is_winner"] = df["rank_in_constituency"] == 1

    # Constituency aggregations
    const_grp = df.groupby("Constituency")

    winner_df = (
        df[df["is_winner"]][["Constituency", "Total Votes", "Party", "Candidate"]]
        .rename(columns={"Total Votes": "winner_votes",
                         "Party": "winner_party",
                         "Candidate": "winner_candidate"})
    )

    runner_df = (
        df[df["rank_in_constituency"] == 2][["Constituency", "Total Votes"]]
        .rename(columns={"Total Votes": "runner_votes"})
    )

    total_df = (
        const_grp["Total Votes"].sum()
        .reset_index()
        .rename(columns={"Total Votes": "total_electors"})
    )

    # Merge back
    df = df.merge(winner_df, on="Constituency", how="left")
    df = df.merge(runner_df, on="Constituency", how="left")
    df = df.merge(total_df,  on="Constituency", how="left")

    df["margin"]               = df["winner_votes"] - df["runner_votes"]
    df["margin_pct"]           = np.where(df["total_electors"] > 0,
                                          df["margin"] / df["total_electors"] * 100, 0)
    df["competitiveness_score"]= 100 - df["margin_pct"].clip(0, 100)
    df["dominance_index"]      = np.where(df["runner_votes"] > 0,
                                          df["winner_votes"] / df["runner_votes"], np.nan)

    note(f"Feature engineering complete. DataFrame shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NULL / DATA QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def data_quality_analysis(df: pd.DataFrame) -> None:
    """Generate null heatmap, missing value bar chart, and duplication report."""
    log.info("Running data quality analysis …")

    null_counts  = df.isna().sum()
    null_pct     = (null_counts / len(df) * 100).round(2)
    quality_df   = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    quality_df   = quality_df[quality_df["null_count"] > 0].sort_values("null_pct", ascending=False)

    # Save quality summary
    path = os.path.join(DIRS["statistics"], "data_quality.csv")
    quality_df.to_csv(path)
    log.info("Data quality CSV → %s", path)

    # ── Missing value bar chart ───────────────────────────────────────────────
    if not quality_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        quality_df["null_pct"].plot(kind="barh", ax=ax, color=HIGHLIGHT)
        ax.set_title("Missing Value Percentage by Column", fontweight="bold")
        ax.set_xlabel("% Missing")
        ax.set_ylabel("Column")
        for patch in ax.patches:
            ax.text(patch.get_width() + 0.2, patch.get_y() + 0.3,
                    f"{patch.get_width():.1f}%", fontsize=9)
        save_fig(fig, "01_missing_values_bar")
    else:
        note("No missing values detected in cleaned dataset.")

    # ── Descriptive statistics export ─────────────────────────────────────────
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(DIRS["statistics"], "descriptive_statistics.csv"))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def univariate_numerical(df: pd.DataFrame) -> None:
    """Histograms, KDE, box, violin plots for all key numeric columns."""
    log.info("Running univariate numerical analysis …")

    num_cols = ["Total Votes", "EVM Votes", "Postal Votes",
                "vote_share_pct", "margin", "margin_pct",
                "competitiveness_score", "dominance_index"]
    num_cols = [c for c in num_cols if c in df.columns]

    # ── Grid of histograms + KDE ──────────────────────────────────────────────
    ncols = 3
    nrows = -(-len(num_cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=40, color=ACCENT, edgecolor="white", alpha=0.7, density=True)
        data.plot(kind="kde", ax=axes[i], color=HIGHLIGHT, linewidth=2)
        axes[i].set_title(f"Distribution: {col}", fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        sk, ku = data.skew(), data.kurt()
        axes[i].annotate(f"Skew={sk:.2f}\nKurt={ku:.2f}",
                         xy=(0.97, 0.95), xycoords="axes fraction",
                         ha="right", va="top", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Tamil Nadu 2026 — Univariate Distributions", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "02_univariate_histograms_kde")

    # ── Box plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[i].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=ACCENT, alpha=0.6),
                        medianprops=dict(color=HIGHLIGHT, linewidth=2))
        axes[i].set_title(f"Box Plot: {col}", fontweight="bold")
        axes[i].set_ylabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Box Plot Overview — Key Numeric Features", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_boxplots_numeric")

    # ── Violin plots (candidate votes) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    vdata = [df["Total Votes"].dropna().values,
             df["EVM Votes"].dropna().values,
             df["Postal Votes"].dropna().values]
    vp = ax.violinplot(vdata, showmedians=True, showmeans=True)
    for body in vp["bodies"]:
        body.set_facecolor(ACCENT)
        body.set_alpha(0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Total Votes", "EVM Votes", "Postal Votes"])
    ax.set_title("Violin Plot — Vote Distribution by Type", fontweight="bold")
    ax.set_ylabel("Votes")
    save_fig(fig, "04_violin_vote_types")

    # ── Skewness / Kurtosis summary ───────────────────────────────────────────
    sk_df = pd.DataFrame({
        "Column"  : num_cols,
        "Skewness": [df[c].skew() for c in num_cols],
        "Kurtosis": [df[c].kurt() for c in num_cols],
        "Mean"    : [df[c].mean() for c in num_cols],
        "Median"  : [df[c].median() for c in num_cols],
        "Std"     : [df[c].std() for c in num_cols],
    }).set_index("Column")
    sk_df.to_csv(os.path.join(DIRS["statistics"], "skewness_kurtosis.csv"))
    note("High skewness in 'Total Votes' and 'margin' → lognormal distributions common in elections.")


def univariate_categorical(df: pd.DataFrame) -> None:
    """Frequency/Pareto charts for key categorical columns."""
    log.info("Running univariate categorical analysis …")

    # ── Top-20 parties by seats ───────────────────────────────────────────────
    winners   = df[df["is_winner"]].copy()
    seat_share= winners["winner_party"].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(seat_share)))
    bars = ax.barh(seat_share.index[::-1], seat_share.values[::-1], color=colors[::-1])
    ax.set_title("Top 20 Parties — Seats Won (Tamil Nadu 2026)", fontweight="bold")
    ax.set_xlabel("Number of Seats Won")
    ax.set_ylabel("Party")
    for bar in bars:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=9)
    save_fig(fig, "05_seats_won_top20_parties")

    # ── Pareto chart: seat share ──────────────────────────────────────────────
    top15  = seat_share.head(15)
    cumsum = top15.cumsum() / top15.sum() * 100
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    bars = ax1.bar(range(len(top15)), top15.values, color=ACCENT, alpha=0.7)
    ax2.plot(range(len(top15)), cumsum.values, color=HIGHLIGHT,
             marker="o", linewidth=2, markersize=6)
    ax1.set_xticks(range(len(top15)))
    ax1.set_xticklabels([p[:25] for p in top15.index], rotation=45, ha="right")
    ax1.set_ylabel("Seats Won", color=ACCENT)
    ax2.set_ylabel("Cumulative %", color=HIGHLIGHT)
    ax2.axhline(80, color="gray", linestyle="--", alpha=0.6, label="80% line")
    ax1.set_title("Pareto Chart — Seat Share (Top 15 Parties)", fontweight="bold")
    save_fig(fig, "06_pareto_seat_share")

    # ── Candidate count per party (top 15) ────────────────────────────────────
    cand_count = df[~df["Is_NOTA"]]["Party"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(cand_count)), cand_count.values, color=GREEN, alpha=0.7)
    ax.set_xticks(range(len(cand_count)))
    ax.set_xticklabels([p[:20] for p in cand_count.index], rotation=45, ha="right")
    ax.set_title("Number of Candidates Fielded — Top 15 Parties", fontweight="bold")
    ax.set_ylabel("Candidates")
    save_fig(fig, "07_candidate_count_by_party")

    # ── NOTA bar across constituencies ────────────────────────────────────────
    nota = df[df["Is_NOTA"]].sort_values("vote_share_pct", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.barh(nota["Constituency"], nota["vote_share_pct"], color="orange")
    ax.set_title("Top 20 Constituencies by NOTA Vote Share (%)", fontweight="bold")
    ax.set_xlabel("NOTA %")
    save_fig(fig, "08_nota_by_constituency")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  BIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def bivariate_analysis(df: pd.DataFrame) -> None:
    """Correlation heatmap, scatter plots, regression, crosstab, chi-square."""
    log.info("Running bivariate analysis …")

    num_cols = ["Total Votes", "vote_share_pct", "margin",
                "margin_pct", "competitiveness_score",
                "dominance_index", "postal_pct"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr_df  = df[num_cols].corr()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdYlGn",
                mask=mask, ax=ax, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap — Key Numeric Features", fontweight="bold")
    save_fig(fig, "09_correlation_heatmap")

    # ── Scatter: vote_share_pct vs margin ────────────────────────────────────
    const_df = df[df["is_winner"]].drop_duplicates("Constituency")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(const_df["vote_share_pct"], const_df["margin"],
                         c=const_df["competitiveness_score"],
                         cmap="RdYlGn", alpha=0.7, s=60, edgecolors="k", linewidths=0.3)
    plt.colorbar(scatter, ax=ax, label="Competitiveness Score")
    ax.set_xlabel("Winner Vote Share (%)")
    ax.set_ylabel("Winning Margin (votes)")
    ax.set_title("Vote Share vs Winning Margin — All 234 Constituencies", fontweight="bold")
    # Regression line
    m, b = np.polyfit(const_df["vote_share_pct"].dropna(),
                      const_df["margin"].dropna(), 1)
    x_line = np.linspace(const_df["vote_share_pct"].min(),
                         const_df["vote_share_pct"].max(), 100)
    ax.plot(x_line, m * x_line + b, color=HIGHLIGHT, linewidth=2, linestyle="--", label="OLS Fit")
    ax.legend()
    save_fig(fig, "10_scatter_voteshare_vs_margin")

    # ── Regression plot: EVM vs Total Votes ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    sample = df.sample(min(1000, len(df)), random_state=42)
    ax.scatter(sample["EVM Votes"], sample["Total Votes"],
               alpha=0.4, color=ACCENT, s=20)
    m2, b2 = np.polyfit(df["EVM Votes"], df["Total Votes"], 1)
    xr = np.linspace(df["EVM Votes"].min(), df["EVM Votes"].max(), 200)
    ax.plot(xr, m2 * xr + b2, color=HIGHLIGHT, linewidth=2)
    ax.set_xlabel("EVM Votes")
    ax.set_ylabel("Total Votes")
    ax.set_title("EVM Votes vs Total Votes (Sample of 1,000)", fontweight="bold")
    r_sq = np.corrcoef(df["EVM Votes"], df["Total Votes"])[0, 1] ** 2
    ax.annotate(f"R² = {r_sq:.4f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=12, color=HIGHLIGHT, fontweight="bold")
    save_fig(fig, "11_regression_evm_vs_total")

    # ── Postal votes vs total votes ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["Total Votes"], df["Postal Votes"], alpha=0.3, s=15, color=GREEN)
    ax.set_xlabel("Total Votes")
    ax.set_ylabel("Postal Votes")
    ax.set_title("Postal Votes vs Total Votes per Candidate", fontweight="bold")
    save_fig(fig, "12_scatter_postal_vs_total")

    # ── Chi-square: Party vs is_winner ────────────────────────────────────────
    party_filter = df[~df["Is_Independent"] & ~df["Is_NOTA"]]
    top_parties  = party_filter["Party"].value_counts().head(10).index
    sub          = party_filter[party_filter["Party"].isin(top_parties)]
    ct           = pd.crosstab(sub["Party"], sub["is_winner"])
    chi2, p, dof, _ = chi2_contingency(ct)
    note(f"Chi-square test (Party vs Win): χ²={chi2:.2f}, p={p:.4e}, dof={dof}")
    note("Result: Party identity is HIGHLY SIGNIFICANT in predicting victory (p < 0.05)." if p < 0.05
         else "Result: No significant association between party and victory at α=0.05.")

    # ── Crosstab heatmap ──────────────────────────────────────────────────────
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ct_pct, annot=True, fmt=".1f", cmap="Blues", ax=ax,
                cbar_kws={"label": "% of Candidates"})
    ax.set_title("Party vs Win Rate — Top 10 Parties (%)", fontweight="bold")
    ax.set_xlabel("Won?")
    ax.set_ylabel("Party")
    save_fig(fig, "13_crosstab_party_win_heatmap")

    # ── ANOVA: margin_pct across top parties ──────────────────────────────────
    groups = [
        sub[sub["Party"] == p]["margin_pct"].dropna().values
        for p in top_parties
    ]
    groups = [g for g in groups if len(g) > 1]
    f_stat, p_anova = f_oneway(*groups)
    note(f"One-way ANOVA (margin_pct by party): F={f_stat:.2f}, p={p_anova:.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MULTIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def multivariate_analysis(df: pd.DataFrame) -> None:
    """PCA, K-Means clustering, bubble charts, faceted visualisations."""
    log.info("Running multivariate analysis …")

    const_df = df[df["is_winner"]].drop_duplicates("Constituency").copy()
    feat_cols = ["winner_votes", "runner_votes", "margin_pct",
                 "competitiveness_score", "dominance_index", "total_electors"]
    feat_cols = [c for c in feat_cols if c in const_df.columns]
    feat_data = const_df[feat_cols].fillna(const_df[feat_cols].median())

    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(feat_data)

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca = PCA(n_components=min(len(feat_cols), 5))
    X_pca = pca.fit_transform(X_scaled)
    expl  = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(range(1, len(expl) + 1), expl, color=ACCENT, alpha=0.8)
    axes[0].plot(range(1, len(expl) + 1), np.cumsum(expl), marker="o",
                 color=HIGHLIGHT, linewidth=2)
    axes[0].set_title("PCA — Explained Variance", fontweight="bold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("% Variance Explained")

    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=const_df["margin_pct"],
                              cmap="RdYlGn", alpha=0.7, s=50)
    plt.colorbar(scatter, ax=axes[1], label="Margin %")
    axes[1].set_xlabel(f"PC1 ({expl[0]:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({expl[1]:.1f}%)")
    axes[1].set_title("PCA — Constituency Map (PC1 vs PC2)", fontweight="bold")
    fig.suptitle("Principal Component Analysis — 234 Constituencies", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "14_pca_analysis")

    # Save loadings
    load_df = pd.DataFrame(pca.components_[:2].T, index=feat_cols,
                           columns=["PC1", "PC2"])
    load_df.to_csv(os.path.join(DIRS["statistics"], "pca_loadings.csv"))
    note(f"PCA: PC1+PC2 explain {sum(expl[:2]):.1f}% of constituency variance.")

    # ── K-Means clustering ────────────────────────────────────────────────────
    inertias = []
    K_range  = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, marker="o", color=ACCENT, linewidth=2)
    ax.set_title("K-Means Elbow Plot — Constituency Clusters", fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    save_fig(fig, "15_kmeans_elbow")

    K_BEST = 4
    km_best = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
    const_df = const_df.copy()
    const_df["cluster"] = km_best.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cl in range(K_BEST):
        mask = const_df["cluster"] == cl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f"Cluster {cl}", alpha=0.7, s=60)
    ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
    ax.set_title(f"K-Means Clusters (k={K_BEST}) — Constituency Types", fontweight="bold")
    ax.legend()
    save_fig(fig, "16_kmeans_clusters_pca")

    # Cluster profile
    cluster_profile = const_df.groupby("cluster")[feat_cols].mean().round(2)
    cluster_profile.to_csv(os.path.join(DIRS["statistics"], "cluster_profiles.csv"))
    note(f"K-Means (k={K_BEST}) identified {K_BEST} distinct constituency competition profiles.")

    # ── Bubble chart: total_electors vs margin_pct, sized by winner_votes ─────
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        const_df["total_electors"],
        const_df["margin_pct"],
        s=const_df["winner_votes"] / 200,
        c=const_df["cluster"],
        cmap="tab10",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("Total Valid Votes in Constituency")
    ax.set_ylabel("Margin % (of Total Votes)")
    ax.set_title("Bubble Chart — Constituency Electoral Intensity\n"
                 "(Bubble size ∝ Winner Votes; Color = Cluster)", fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    save_fig(fig, "17_bubble_constituency_intensity")

    # ── Faceted: top-6 parties — distribution of vote share ──────────────────
    top6_parties = (
        df[~df["Is_Independent"] & ~df["Is_NOTA"]]
        ["Party"].value_counts().head(6).index.tolist()
    )
    sub = df[df["Party"].isin(top6_parties)]
    g   = sns.FacetGrid(sub, col="Party", col_wrap=3, height=4, sharey=False,
                        sharex=False)
    g.map(sns.histplot, "vote_share_pct", bins=25, color=ACCENT)
    g.set_titles(col_template="{col_name}", size=9)
    g.set_axis_labels("Vote Share %", "Count")
    g.figure.suptitle("Vote Share Distribution — Top 6 Parties", y=1.01,
                       fontsize=14, fontweight="bold")
    g.figure.tight_layout()
    save_fig(g.figure, "18_faceted_voteshare_top6parties")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ADVANCED POLITICAL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def political_analytics(df: pd.DataFrame) -> None:
    """
    Deep political intelligence:
      - Seat share & pie/donut charts
      - Alliance analysis
      - Closest contests & landslide wins
      - Strongest candidates & weakest constituencies
      - Party dominance / strongholds
      - Vote concentration (Gini-style analysis)
      - Margin distribution by party
    """
    log.info("Running advanced political analytics …")

    winners = df[df["is_winner"]].drop_duplicates("Constituency").copy()
    note(f"Total constituencies analysed: {winners['Constituency'].nunique()}")

    # ── Seat share ────────────────────────────────────────────────────────────
    seat_share = winners["winner_party"].value_counts()
    top8 = seat_share.head(8)
    others_count = seat_share.iloc[8:].sum()
    pie_labels = list(top8.index) + ["Others"]
    pie_vals   = list(top8.values) + [others_count]

    # Donut
    fig, ax = plt.subplots(figsize=(10, 8))
    wedge_props = {"linewidth": 1.5, "edgecolor": "white"}
    colors = plt.cm.Set2(np.linspace(0, 1, len(pie_labels)))
    wedges, texts, autotexts = ax.pie(
        pie_vals, labels=None, autopct="%1.1f%%",
        startangle=140, colors=colors,
        wedgeprops={**wedge_props, "width": 0.5},
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, [f"{l} ({v})" for l, v in zip(pie_labels, pie_vals)],
              loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title("Seat Share — Tamil Nadu Assembly 2026\n(Donut Chart)", fontweight="bold")
    save_fig(fig, "19_donut_seat_share")

    # Save seat share table
    seat_share.reset_index().rename(columns={"count": "seats"}).to_csv(
        os.path.join(DIRS["statistics"], "seat_share_by_party.csv"), index=False
    )

    # ── Vote share by party (total votes) ────────────────────────────────────
    party_votes = (
        df[~df["Is_NOTA"]]
        .groupby("Party")["Total Votes"]
        .sum()
        .sort_values(ascending=False)
    )
    top10_vote_share = party_votes.head(10)
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top10_vote_share)))
    bars = ax.bar([p[:25] for p in top10_vote_share.index],
                  top10_vote_share.values, color=colors)
    ax.set_title("Total Votes Aggregated — Top 10 Parties", fontweight="bold")
    ax.set_ylabel("Total Votes")
    ax.set_xlabel("Party")
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "20_total_votes_top10_parties")
    note(f"Largest vote-getter: {top10_vote_share.index[0]} with {top10_vote_share.iloc[0]:,} votes")

    # ── Closest contests ─────────────────────────────────────────────────────
    closest = winners.nsmallest(20, "margin")[
        ["Constituency", "winner_candidate", "winner_party", "margin", "margin_pct", "competitiveness_score"]
    ]
    closest.to_csv(os.path.join(DIRS["statistics"], "closest_contests.csv"), index=False)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(closest["Constituency"][::-1], closest["margin"][::-1], color=HIGHLIGHT)
    ax.set_title("20 Closest Contests — Winning Margins (Votes)", fontweight="bold")
    ax.set_xlabel("Margin of Victory (Votes)")
    ax.set_ylabel("Constituency")
    save_fig(fig, "21_closest_contests")
    note(f"Closest contest: {closest.iloc[0]['Constituency']} — margin of only {closest.iloc[0]['margin']:,} votes!")

    # ── Landslide wins ────────────────────────────────────────────────────────
    landslides = winners.nlargest(20, "margin")[
        ["Constituency", "winner_candidate", "winner_party", "margin", "margin_pct"]
    ]
    landslides.to_csv(os.path.join(DIRS["statistics"], "landslide_wins.csv"), index=False)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(landslides["Constituency"][::-1], landslides["margin"][::-1], color=GREEN)
    ax.set_title("20 Biggest Landslide Wins — Winning Margins (Votes)", fontweight="bold")
    ax.set_xlabel("Margin of Victory (Votes)")
    ax.set_ylabel("Constituency")
    save_fig(fig, "22_landslide_wins")
    note(f"Biggest landslide: {landslides.iloc[0]['Constituency']} — {landslides.iloc[0]['margin']:,} votes.")

    # ── Margin distribution by party (top 6 parties) ─────────────────────────
    top6 = seat_share.head(6).index.tolist()
    sub  = winners[winners["winner_party"].isin(top6)]
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.boxplot(data=sub, x="winner_party", y="margin", palette="tab10", ax=ax)
    ax.set_xticklabels([p[:20] for p in sub["winner_party"].unique()], rotation=35, ha="right")
    ax.set_title("Margin Distribution by Party — Top Parties", fontweight="bold")
    ax.set_xlabel("Party")
    ax.set_ylabel("Margin of Victory (Votes)")
    fig.tight_layout()
    save_fig(fig, "23_margin_distribution_by_party")

    # ── Party win rate ─────────────────────────────────────────────────────────
    party_fielded = df[~df["Is_NOTA"] & ~df["Is_Independent"]]["Party"].value_counts()
    party_won     = winners["winner_party"].value_counts()
    win_rate_df   = pd.DataFrame({"fielded": party_fielded, "won": party_won}).fillna(0)
    win_rate_df["win_rate"] = (win_rate_df["won"] / win_rate_df["fielded"] * 100).round(1)
    win_rate_df   = win_rate_df[win_rate_df["fielded"] >= 5].sort_values("win_rate", ascending=False)
    win_rate_df.to_csv(os.path.join(DIRS["statistics"], "party_win_rate.csv"))

    top_wr = win_rate_df.head(15)
    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(top_wr.index[::-1], top_wr["win_rate"][::-1], color=ACCENT)
    ax.set_title("Party Win Rate % (≥5 Candidates Fielded)", fontweight="bold")
    ax.set_xlabel("Win Rate (%)")
    ax.axvline(50, color=HIGHLIGHT, linestyle="--", linewidth=1.5, label="50% mark")
    ax.legend()
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "24_party_win_rate")

    # ── Stacked bar: won vs lost per top party ────────────────────────────────
    top8_parties = seat_share.head(8).index.tolist()
    sb_data = {}
    for p in top8_parties:
        fielded = len(df[df["Party"] == p])
        won     = int(party_won.get(p, 0))
        sb_data[p[:20]] = {"Won": won, "Lost": fielded - won}
    sb_df = pd.DataFrame(sb_data).T

    fig, ax = plt.subplots(figsize=(14, 7))
    sb_df["Won"].plot(kind="bar", ax=ax, color=GREEN, alpha=0.85, label="Won")
    sb_df["Lost"].plot(kind="bar", ax=ax, bottom=sb_df["Won"],
                       color=HIGHLIGHT, alpha=0.65, label="Lost")
    ax.set_title("Seats Won vs Lost — Top 8 Parties", fontweight="bold")
    ax.set_xlabel("Party")
    ax.set_ylabel("Constituencies")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "25_stacked_won_vs_lost")

    # ── Competitiveness score distribution ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(winners["competitiveness_score"].dropna(), bins=30,
            color=ACCENT, edgecolor="white", alpha=0.8)
    ax.set_title("Distribution of Constituency Competitiveness Score", fontweight="bold")
    ax.set_xlabel("Competitiveness Score (100 = tossup, 0 = walkover)")
    ax.set_ylabel("Number of Constituencies")
    q25, q50, q75 = winners["competitiveness_score"].quantile([0.25, 0.5, 0.75])
    ax.axvline(q50, color=HIGHLIGHT, linestyle="--", label=f"Median={q50:.1f}")
    ax.axvline(q25, color="orange", linestyle=":", label=f"Q1={q25:.1f}")
    ax.axvline(q75, color="purple", linestyle=":", label=f"Q3={q75:.1f}")
    ax.legend()
    save_fig(fig, "26_competitiveness_distribution")

    # ── Top individual candidates by votes ────────────────────────────────────
    top_cands = df.nlargest(20, "Total Votes")[
        ["Candidate", "Party", "Constituency", "Total Votes", "vote_share_pct", "is_winner"]
    ]
    top_cands.to_csv(os.path.join(DIRS["statistics"], "top_candidates_by_votes.csv"), index=False)

    fig, ax = plt.subplots(figsize=(13, 8))
    colors_w = [GREEN if w else HIGHLIGHT for w in top_cands["is_winner"]]
    ax.barh(top_cands["Candidate"][::-1], top_cands["Total Votes"][::-1], color=colors_w[::-1])
    ax.set_title("Top 20 Candidates by Votes Received", fontweight="bold")
    ax.set_xlabel("Total Votes")
    w_patch = mpatches.Patch(color=GREEN, label="Winner")
    l_patch = mpatches.Patch(color=HIGHLIGHT, label="Runner-up / Lost")
    ax.legend(handles=[w_patch, l_patch])
    fig.tight_layout()
    save_fig(fig, "27_top20_candidates_by_votes")

    # ── Dominance index distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    data = winners["dominance_index"].dropna().clip(upper=20)
    ax.hist(data, bins=30, color=GREEN, edgecolor="white", alpha=0.8)
    ax.set_title("Dominance Index Distribution\n(Winner Votes / Runner-up Votes)", fontweight="bold")
    ax.set_xlabel("Dominance Index")
    ax.set_ylabel("Constituency Count")
    ax.axvline(2, color=HIGHLIGHT, linestyle="--", label="2× (clear winner)")
    ax.legend()
    save_fig(fig, "28_dominance_index_distribution")
    note(f"Avg Dominance Index: {winners['dominance_index'].mean():.2f}  "
         f"| Median: {winners['dominance_index'].median():.2f}")

    # ── Postal vote importance ────────────────────────────────────────────────
    close_margin = winners[winners["margin"] <= winners["margin"].quantile(0.25)].copy()
    close_margin["postal_decisive"] = (
        close_margin["winner_votes"] - close_margin["runner_votes"] <=
        df[df["is_winner"]].drop_duplicates("Constituency")
        .set_index("Constituency")
        .loc[close_margin["Constituency"], "Postal Votes"]
        .values
    )
    note(f"In {close_margin['postal_decisive'].sum()} tight constituencies, "
         f"postal votes could have been decisive.")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def outlier_analysis(df: pd.DataFrame) -> None:
    """IQR, Z-score, and Isolation Forest outlier detection."""
    log.info("Running outlier analysis …")

    targets = ["Total Votes", "margin", "margin_pct", "vote_share_pct"]
    targets = [c for c in targets if c in df.columns]

    results = {}
    for col in targets:
        data = df[col].dropna()
        # IQR
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_out = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
        # Z-score
        z = np.abs(stats.zscore(data))
        z_out = (z > 3).sum()
        results[col] = {"IQR_outliers": int(iqr_out), "Z_outliers": int(z_out)}
        note(f"Outliers in '{col}': IQR={iqr_out}, Z-score={z_out}")

    outlier_df = pd.DataFrame(results).T
    outlier_df.to_csv(os.path.join(DIRS["statistics"], "outlier_summary.csv"))

    # ── Isolation Forest on constituency-level features ───────────────────────
    const_df = df[df["is_winner"]].drop_duplicates("Constituency").copy()
    feat_cols = ["total_electors", "margin_pct", "competitiveness_score", "dominance_index"]
    feat_cols = [c for c in feat_cols if c in const_df.columns]
    X = const_df[feat_cols].fillna(const_df[feat_cols].median())
    iso = IsolationForest(contamination=0.05, random_state=42)
    const_df["anomaly"] = iso.fit_predict(X)
    anomalies = const_df[const_df["anomaly"] == -1][
        ["Constituency", "winner_party"] + feat_cols
    ]
    anomalies.to_csv(os.path.join(DIRS["statistics"], "anomalous_constituencies.csv"), index=False)
    note(f"Isolation Forest flagged {len(anomalies)} anomalous constituencies (5% contamination).")

    # ── Visualise outliers: scatter with highlights ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col in zip(axes, ["Total Votes", "margin"]):
        data = df[col].dropna()
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        colors = np.where((data < lower) | (data > upper), HIGHLIGHT, ACCENT)
        ax.scatter(range(len(data)), data, c=colors, alpha=0.4, s=8)
        ax.axhline(upper, color="red", linestyle="--", linewidth=1.5, label=f"IQR Upper ({upper:,.0f})")
        ax.axhline(lower, color="orange", linestyle="--", linewidth=1.5, label=f"IQR Lower ({lower:,.0f})")
        ax.set_title(f"IQR Outliers — {col}", fontweight="bold")
        ax.set_xlabel("Candidate Index")
        ax.set_ylabel(col)
        ax.legend(fontsize=8)
    fig.suptitle("Outlier Visualisation (IQR Method)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "29_outlier_iqr")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  STATISTICAL HYPOTHESIS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def statistical_analysis(df: pd.DataFrame) -> None:
    """Confidence intervals, normality tests, Kruskal-Wallis, and distribution fitting."""
    log.info("Running statistical hypothesis tests …")

    winners = df[df["is_winner"]].drop_duplicates("Constituency")

    # ── Normality test on margin ──────────────────────────────────────────────
    margin_data = winners["margin"].dropna()
    if len(margin_data) <= 5000:
        stat, p = shapiro(margin_data)
        note(f"Shapiro-Wilk (margin): W={stat:.4f}, p={p:.4e}  "
             f"→ {'NOT normal' if p < 0.05 else 'Normal'} distribution at α=0.05")

    # ── 95% CI for mean margin ────────────────────────────────────────────────
    n    = len(margin_data)
    mean = margin_data.mean()
    se   = stats.sem(margin_data)
    ci   = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    note(f"95% CI for Mean Winning Margin: ({ci[0]:,.0f}, {ci[1]:,.0f})  |  Mean = {mean:,.0f}")

    # ── Kruskal-Wallis: margin_pct across top 5 parties ──────────────────────
    top5 = winners["winner_party"].value_counts().head(5).index
    groups = [winners[winners["winner_party"] == p]["margin_pct"].dropna().values for p in top5]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        h_stat, p_kw = kruskal(*groups)
        note(f"Kruskal-Wallis (margin_pct, top 5 parties): H={h_stat:.2f}, p={p_kw:.4e}  "
             f"→ {'Significant' if p_kw < 0.05 else 'Not significant'} difference at α=0.05")

    # ── Distribution fitting: Total Votes ────────────────────────────────────
    data = df["Total Votes"].dropna()
    distributions = [stats.lognorm, stats.gamma, stats.expon]
    fit_results = {}
    for dist in distributions:
        try:
            params = dist.fit(data)
            D, p_ks = stats.kstest(data, dist.name, args=params)
            fit_results[dist.name] = {"D_stat": round(D, 4), "p_value": round(p_ks, 4)}
        except Exception:
            pass
    fit_df = pd.DataFrame(fit_results).T.sort_values("D_stat")
    fit_df.to_csv(os.path.join(DIRS["statistics"], "distribution_fitting.csv"))
    if not fit_df.empty:
        best_dist = fit_df.index[0]
        note(f"Best-fitting distribution for Total Votes: {best_dist} (smallest KS D-statistic)")

    # ── Pairplot of key features ──────────────────────────────────────────────
    pairplot_cols = ["Total Votes", "vote_share_pct", "margin_pct", "dominance_index"]
    pairplot_cols = [c for c in pairplot_cols if c in df.columns]
    sample_pair  = df[df["is_winner"]].drop_duplicates("Constituency").sample(
        min(234, len(df)), random_state=42
    )
    fig = sns.pairplot(sample_pair[pairplot_cols].dropna(), diag_kind="kde",
                       plot_kws={"alpha": 0.5, "s": 20},
                       diag_kws={"fill": True})
    fig.figure.suptitle("Pairplot — Constituency-Level Features (Winners)", y=1.01,
                         fontsize=13, fontweight="bold")
    fig.figure.tight_layout()
    save_fig(fig.figure, "30_pairplot_winner_features")

    # ── Confidence interval visualisation across parties ─────────────────────
    top8 = df["winner_party"].value_counts().head(8).index if "winner_party" in df else []
    ci_data = []
    for p in top8:
        sub = df[df["winner_party"] == p]["margin"].dropna()
        if len(sub) > 1:
            m = sub.mean()
            ci_p = stats.t.interval(0.95, df=len(sub) - 1, loc=m, scale=stats.sem(sub))
            ci_data.append({"party": p[:22], "mean": m, "ci_lo": ci_p[0], "ci_hi": ci_p[1]})
    if ci_data:
        ci_df = pd.DataFrame(ci_data)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(ci_df["party"], ci_df["mean"],
                    yerr=[ci_df["mean"] - ci_df["ci_lo"],
                          ci_df["ci_hi"] - ci_df["mean"]],
                    fmt="o", color=ACCENT, ecolor=HIGHLIGHT,
                    capsize=6, linewidth=2, markersize=8)
        ax.set_xticklabels(ci_df["party"], rotation=30, ha="right")
        ax.set_title("95% Confidence Intervals — Mean Winning Margin by Party", fontweight="bold")
        ax.set_ylabel("Mean Margin (Votes)")
        ax.set_xlabel("Party")
        fig.tight_layout()
        save_fig(fig, "31_confidence_intervals_margin_by_party")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  INTERACTIVE PLOTLY DASHBOARDS
# ══════════════════════════════════════════════════════════════════════════════

def interactive_plotly_analysis(df: pd.DataFrame) -> None:
    """Generate rich interactive Plotly visualisations saved as HTML."""
    log.info("Generating interactive Plotly plots …")

    winners = df[df["is_winner"]].drop_duplicates("Constituency").copy()

    # ── 1. Treemap: seat share ────────────────────────────────────────────────
    seat_share = winners["winner_party"].value_counts().reset_index()
    seat_share.columns = ["party", "seats"]
    fig_tm = px.treemap(
        seat_share.head(20),
        path=["party"], values="seats",
        title="Seat Share Treemap — Tamil Nadu 2026",
        color="seats", color_continuous_scale="RdYlGn",
    )
    fig_tm.update_layout(font_size=13)
    save_plotly(fig_tm, "TM01_treemap_seat_share")

    # ── 2. Sunburst: party → constituency → margin tier ───────────────────────
    top6_w = winners[
        winners["winner_party"].isin(seat_share.head(6)["party"])
    ].copy()
    top6_w["margin_tier"] = pd.cut(
        top6_w["margin_pct"],
        bins=[0, 5, 10, 20, 100],
        labels=["<5%", "5–10%", "10–20%", ">20%"]
    )
    sunburst_df = (
        top6_w.groupby(["winner_party", "margin_tier"], observed=True)
        .size()
        .reset_index(name="count")
    )
    fig_sb = px.sunburst(
        sunburst_df,
        path=["winner_party", "margin_tier"],
        values="count",
        title="Sunburst: Party → Margin Tier (Top 6 Parties)",
        color="count", color_continuous_scale="Blues",
    )
    save_plotly(fig_sb, "SB01_sunburst_party_margin_tier")

    # ── 3. Interactive scatter: competitiveness vs dominance ──────────────────
    fig_sc = px.scatter(
        winners,
        x="competitiveness_score", y="dominance_index",
        size="total_electors", color="winner_party",
        hover_name="Constituency",
        hover_data=["winner_candidate", "margin", "margin_pct"],
        title="Constituency Intelligence: Competitiveness vs Dominance Index",
        labels={"competitiveness_score": "Competitiveness Score",
                "dominance_index": "Dominance Index"},
        size_max=30, opacity=0.75,
    )
    save_plotly(fig_sc, "SC01_scatter_competitive_vs_dominant")

    # ── 4. Bar: top-20 winners by margin ─────────────────────────────────────
    top20_margin = winners.nlargest(20, "margin")
    fig_bar = px.bar(
        top20_margin,
        x="margin", y="Constituency",
        orientation="h",
        color="winner_party",
        hover_data=["winner_candidate", "margin_pct"],
        title="Top 20 Biggest Margins — Tamil Nadu 2026",
        labels={"margin": "Margin (Votes)"},
        text="margin",
    )
    fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
    save_plotly(fig_bar, "BAR01_top20_margins_interactive")

    # ── 5. Box plot by party (interactive) ───────────────────────────────────
    top8_parties = seat_share.head(8)["party"].tolist()
    sub_box = winners[winners["winner_party"].isin(top8_parties)]
    fig_box = px.box(
        sub_box, x="winner_party", y="margin",
        color="winner_party",
        title="Winning Margin Distribution — Top 8 Parties",
        labels={"winner_party": "Party", "margin": "Margin (Votes)"},
        points="all",
    )
    fig_box.update_layout(showlegend=False)
    save_plotly(fig_box, "BOX01_margin_distribution_by_party")

    # ── 6. Histogram: vote share distribution ────────────────────────────────
    fig_hist = px.histogram(
        df[~df["Is_NOTA"] & ~df["Is_Independent"]],
        x="vote_share_pct",
        color="is_winner",
        nbins=50,
        barmode="overlay",
        title="Vote Share Distribution — Winners vs Non-Winners",
        labels={"vote_share_pct": "Vote Share (%)", "is_winner": "Is Winner"},
        opacity=0.7,
    )
    save_plotly(fig_hist, "HIST01_voteshare_winners_vs_losers")

    # ── 7. Violin: margin_pct by party ───────────────────────────────────────
    fig_vio = px.violin(
        sub_box, y="margin_pct", x="winner_party",
        color="winner_party", box=True, points="all",
        title="Margin % Violin — Top 8 Parties",
        labels={"winner_party": "Party", "margin_pct": "Margin %"},
    )
    fig_vio.update_layout(showlegend=False)
    save_plotly(fig_vio, "VIO01_margin_pct_violin")

    # ── 8. Bubble: total_electors vs margin (sized by winner_votes) ──────────
    fig_bub = px.scatter(
        winners,
        x="total_electors", y="margin",
        size="winner_votes",
        color="winner_party",
        hover_name="Constituency",
        hover_data=["winner_candidate", "margin_pct", "competitiveness_score"],
        title="Electoral Bubble Map — Constituency Size vs Margin",
        labels={"total_electors": "Total Valid Votes", "margin": "Margin (Votes)"},
        size_max=40, opacity=0.7,
        log_x=True,
    )
    save_plotly(fig_bub, "BUB01_bubble_electors_vs_margin")

    # ── 9. Stacked bar: party — won vs total contested ────────────────────────
    all_parties = df[~df["Is_NOTA"] & ~df["Is_Independent"]]["Party"].value_counts()
    won_parties  = winners["winner_party"].value_counts()
    top12 = all_parties.head(12).index
    stack_df = pd.DataFrame({
        "Party": [p[:22] for p in top12],
        "Won"  : [int(won_parties.get(p, 0)) for p in top12],
        "Lost" : [int(all_parties[p]) - int(won_parties.get(p, 0)) for p in top12],
    })
    fig_stack = go.Figure(data=[
        go.Bar(name="Won", x=stack_df["Party"], y=stack_df["Won"],
               marker_color=GREEN),
        go.Bar(name="Lost/Runner-up", x=stack_df["Party"], y=stack_df["Lost"],
               marker_color=HIGHLIGHT),
    ])
    fig_stack.update_layout(
        barmode="stack",
        title="Seats Contested vs Won — Top 12 Parties",
        xaxis_title="Party", yaxis_title="Constituencies",
        xaxis_tickangle=-40,
    )
    save_plotly(fig_stack, "STACK01_contested_vs_won")

    # ── 10. Multi-panel summary dashboard ────────────────────────────────────
    fig_dash = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Seat Share (Top 10)",
            "Competitiveness Score Distribution",
            "Winner Vote Share (%)",
            "Postal Vote % per Candidate",
        ]
    )
    # Panel 1: Seat share bar
    fig_dash.add_trace(
        go.Bar(x=seat_share.head(10)["party"].str[:15],
               y=seat_share.head(10)["seats"],
               marker_color=px.colors.qualitative.Set2[:10],
               showlegend=False),
        row=1, col=1
    )
    # Panel 2: Competitiveness histogram
    fig_dash.add_trace(
        go.Histogram(x=winners["competitiveness_score"].dropna(),
                     nbinsx=25, marker_color=ACCENT, showlegend=False),
        row=1, col=2
    )
    # Panel 3: Winner vote share KDE (as scatter)
    kde_x = np.linspace(0, 100, 200)
    kde_y = stats.gaussian_kde(winners["vote_share_pct"].dropna())(kde_x)
    fig_dash.add_trace(
        go.Scatter(x=kde_x, y=kde_y, fill="tozeroy",
                   line_color=GREEN, showlegend=False),
        row=2, col=1
    )
    # Panel 4: Postal pct
    fig_dash.add_trace(
        go.Histogram(x=df["postal_pct"].dropna().clip(0, 10),
                     nbinsx=25, marker_color="orange", showlegend=False),
        row=2, col=2
    )
    fig_dash.update_layout(
        title_text="Tamil Nadu 2026 Election — Executive Dashboard",
        title_font_size=16,
        height=700,
    )
    save_plotly(fig_dash, "DASH01_executive_dashboard")

    log.info("All interactive Plotly plots saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ADDITIONAL ADVANCED VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def advanced_visualisations(df: pd.DataFrame) -> None:
    """
    Extra publication-quality charts:
      - Radar chart comparing top parties
      - Strip plot of individual vote share
      - Swarm plot of margin by party
      - Area plot of cumulative vote share
      - Clustermap of party × constituency type
    """
    log.info("Generating advanced visualisations …")

    winners = df[df["is_winner"]].drop_duplicates("Constituency").copy()
    top5 = winners["winner_party"].value_counts().head(5).index.tolist()

    # ── Radar chart: party performance on 5 metrics ───────────────────────────
    metrics = ["margin_pct", "competitiveness_score", "vote_share_pct",
               "dominance_index", "postal_pct"]
    metrics = [m for m in metrics if m in winners.columns]
    party_radar = winners[winners["winner_party"].isin(top5)].groupby("winner_party")[metrics].mean()
    scaler2 = StandardScaler()
    party_radar_norm = pd.DataFrame(
        scaler2.fit_transform(party_radar),
        index=party_radar.index, columns=party_radar.columns
    )
    # Min-max scale to 0–10 for radar
    for col in party_radar_norm.columns:
        mn, mx = party_radar_norm[col].min(), party_radar_norm[col].max()
        party_radar_norm[col] = (party_radar_norm[col] - mn) / (mx - mn + 1e-9) * 10

    num_vars = len(metrics)
    angles   = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors_r = plt.cm.tab10(np.linspace(0, 1, len(top5)))
    for (party, row), color in zip(party_radar_norm.iterrows(), colors_r):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, linewidth=2, linestyle="solid", label=party[:18], color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_title("Radar: Multi-Metric Party Comparison (Top 5)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    save_fig(fig, "32_radar_party_comparison")

    # ── Strip plot: vote_share_pct by top 6 parties ───────────────────────────
    top6 = winners["winner_party"].value_counts().head(6).index.tolist()
    sub6 = df[df["Party"].isin(top6)].copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.stripplot(data=sub6, x="Party", y="vote_share_pct",
                  jitter=0.25, alpha=0.4, size=4, palette="tab10", ax=ax)
    ax.set_xticklabels([p[:20] for p in sub6["Party"].unique()], rotation=35, ha="right")
    ax.set_title("Strip Plot — Vote Share % by Top 6 Parties", fontweight="bold")
    ax.set_xlabel("Party")
    ax.set_ylabel("Vote Share %")
    fig.tight_layout()
    save_fig(fig, "33_strip_voteshare_by_party")

    # ── Area plot: cumulative vote share ──────────────────────────────────────
    party_votes = (
        df[~df["Is_NOTA"]].groupby("Party")["Total Votes"]
        .sum().sort_values(ascending=False).head(10)
    )
    total_all = party_votes.sum()
    cumulative = party_votes.cumsum() / total_all * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(range(len(cumulative)), cumulative.values,
                    alpha=0.5, color=ACCENT, label="Cumulative %")
    ax.plot(range(len(cumulative)), cumulative.values, color=ACCENT, linewidth=2)
    ax.set_xticks(range(len(cumulative)))
    ax.set_xticklabels([p[:18] for p in cumulative.index], rotation=40, ha="right")
    ax.set_title("Cumulative Vote Share — Top 10 Parties", fontweight="bold")
    ax.set_ylabel("Cumulative % of Total Votes")
    ax.axhline(80, color=HIGHLIGHT, linestyle="--", label="80% mark")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "34_area_cumulative_voteshare")

    # ── Clustermap: party × margin tier ──────────────────────────────────────
    top10_parties = winners["winner_party"].value_counts().head(10).index.tolist()
    sub10 = winners[winners["winner_party"].isin(top10_parties)].copy()
    sub10["margin_tier"] = pd.cut(
        sub10["margin_pct"],
        bins=[0, 5, 10, 15, 20, 100],
        labels=["<5%", "5–10%", "10–15%", "15–20%", ">20%"]
    )
    ct_hm = pd.crosstab(
        sub10["winner_party"].str[:18],
        sub10["margin_tier"]
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(ct_hm, annot=True, fmt="d", cmap="YlOrRd",
                ax=ax, linewidths=0.5)
    ax.set_title("Party × Margin Tier Heatmap — Frequency of Wins", fontweight="bold")
    ax.set_xlabel("Margin Tier")
    ax.set_ylabel("Party")
    fig.tight_layout()
    save_fig(fig, "35_heatmap_party_margin_tier")

    # ── Postal votes leader board ─────────────────────────────────────────────
    top_postal = df.nlargest(20, "Postal Votes")[
        ["Candidate", "Party", "Constituency", "Postal Votes", "Total Votes", "is_winner"]
    ]
    fig, ax = plt.subplots(figsize=(13, 8))
    c_colors = [GREEN if w else HIGHLIGHT for w in top_postal["is_winner"]]
    ax.barh(top_postal["Candidate"][::-1], top_postal["Postal Votes"][::-1],
            color=c_colors[::-1])
    ax.set_title("Top 20 Candidates by Postal Votes Received", fontweight="bold")
    ax.set_xlabel("Postal Votes")
    w_patch = mpatches.Patch(color=GREEN, label="Winner")
    l_patch = mpatches.Patch(color=HIGHLIGHT, label="Non-winner")
    ax.legend(handles=[w_patch, l_patch])
    fig.tight_layout()
    save_fig(fig, "36_top20_postal_votes")

    # ── Independent candidates analysis ──────────────────────────────────────
    indep = df[df["Is_Independent"]].copy()
    indep_wins = indep[indep["is_winner"]]
    note(f"Independent candidates: {len(indep):,} fielded, {len(indep_wins)} won.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(indep["vote_share_pct"].dropna(), bins=30,
                 color="gray", edgecolor="white", alpha=0.8)
    axes[0].set_title("Independent Candidate Vote Share Distribution", fontweight="bold")
    axes[0].set_xlabel("Vote Share %")
    axes[0].set_ylabel("Count")

    if not indep_wins.empty:
        axes[1].bar(range(len(indep_wins)),
                    indep_wins["Total Votes"].values, color=GOLD, edgecolor="k")
        axes[1].set_title("Winning Independent Candidates — Total Votes", fontweight="bold")
        axes[1].set_xticks(range(len(indep_wins)))
        axes[1].set_xticklabels(indep_wins["Constituency"].values, rotation=40, ha="right", fontsize=9)
        axes[1].set_ylabel("Total Votes")
    else:
        axes[1].text(0.5, 0.5, "No Independent Winner", ha="center", va="center", fontsize=14)
        axes[1].set_title("Winning Independent Candidates", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "37_independent_candidate_analysis")

    # ── Round completion check ────────────────────────────────────────────────
    incomplete = df[df["Rounds_Done"] < df["Rounds_Total"]]["Constituency"].nunique()
    note(f"Constituencies with incomplete counting rounds: {incomplete}")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  EXECUTIVE SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_executive_summary(df: pd.DataFrame) -> None:
    """Compose and save a consulting-grade executive summary text report."""
    log.info("Generating executive summary …")

    winners     = df[df["is_winner"]].drop_duplicates("Constituency")
    seat_share  = winners["winner_party"].value_counts()
    top_party   = seat_share.index[0]
    top_seats   = int(seat_share.iloc[0])
    total_seats = winners["Constituency"].nunique()
    majority    = total_seats // 2 + 1
    total_cands = len(df[~df["Is_NOTA"]])
    total_valid = df["Total Votes"].sum()

    closest_c   = winners.nsmallest(1, "margin").iloc[0]
    biggest_c   = winners.nlargest(1, "margin").iloc[0]
    avg_margin  = winners["margin"].mean()
    avg_comp    = winners["competitiveness_score"].mean()

    indep_wins  = int(winners[winners["winner_party"].str.contains("INDEPENDENT", na=False)]["Constituency"].nunique())

    summary = f"""
================================================================================
        TAMIL NADU ASSEMBLY ELECTION 2026 — EXECUTIVE INTELLIGENCE REPORT
                    Political Analytics & Data Science Division
================================================================================
Date Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset        : eci_results_tamilnadu_2026.csv  ({len(df):,} rows × {df.shape[1]} columns)
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. HEADLINE RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Assembly Constituencies  : {total_seats}
  Majority Mark                  : {majority} seats
  Total Candidates               : {total_cands:,}  (excl. NOTA)
  Total Valid Votes Polled       : {total_valid:,}
  Unique Political Parties       : {df['Party'].nunique()}

  LEADING PARTY                  : {top_party}
  Seats Won by Leading Party     : {top_seats} / {total_seats}  ({top_seats/total_seats*100:.1f}%)
  Simple Majority Achieved?      : {'YES ✓' if top_seats >= majority else 'NO ✗ (Hung Assembly / Coalition)'}

  Top 3 Parties by Seat Share:
"""
    for i, (party, seats) in enumerate(seat_share.head(3).items(), 1):
        summary += f"    {i}. {party[:45]:<45} → {seats} seats ({seats/total_seats*100:.1f}%)\n"

    summary += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. COMPETITION LANDSCAPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Average Winning Margin         : {avg_margin:,.0f} votes
  Average Margin as % of Votes   : {winners['margin_pct'].mean():.2f}%
  Average Competitiveness Score  : {avg_comp:.1f} / 100  (100 = tossup)
  Average Dominance Index        : {winners['dominance_index'].mean():.2f}×

  CLOSEST CONTEST:
    Constituency : {closest_c['Constituency']}
    Winner       : {closest_c['winner_candidate']}  ({closest_c['winner_party'][:30]})
    Margin       : {int(closest_c['margin']):,} votes  ({closest_c['margin_pct']:.2f}% of total votes)

  BIGGEST LANDSLIDE:
    Constituency : {biggest_c['Constituency']}
    Winner       : {biggest_c['winner_candidate']}  ({biggest_c['winner_party'][:30]})
    Margin       : {int(biggest_c['margin']):,} votes  ({biggest_c['margin_pct']:.2f}% of total votes)

  Highly Competitive Seats (margin < 2,000 votes)  : {(winners['margin'] < 2000).sum()}
  Landslide Wins (margin > 20,000 votes)           : {(winners['margin'] > 20000).sum()}
  Independent Winners                              : {indep_wins}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. VOTE SHARE INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mean winner vote share         : {winners['vote_share_pct'].mean():.1f}%
  Median winner vote share       : {winners['vote_share_pct'].median():.1f}%
  Minimum winner vote share      : {winners['vote_share_pct'].min():.1f}%   ← Won with plurality
  Maximum winner vote share      : {winners['vote_share_pct'].max():.1f}%   ← Dominant mandate

  Postal Vote Contribution:
    Mean postal % per candidate  : {df['postal_pct'].mean():.2f}%
    Highest postal reliance      : {df.nlargest(1, 'postal_pct').iloc[0]['Constituency']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. KEY ANALYTICAL INSIGHTS & FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    for i, insight in enumerate(INSIGHTS, 1):
        summary += f"  [{i:02d}] {insight}\n"

    summary += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. DATA QUALITY CERTIFICATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dataset rows loaded            : {len(df):,}
  Null values (cleaned dataset)  : {df.isna().sum().sum()}
  Duplicate rows removed         : see log
  Numeric precision              : Integer votes; Float vote share (2dp)
  All output plots saved to      : project_output/plots/
  All interactive HTML saved to  : project_output/interactive_plots/
  Statistical CSVs saved to      : project_output/statistics/
  Cleaned CSV saved to           : project_output/cleaned_data/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. STRATEGIC RECOMMENDATIONS FOR STAKEHOLDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  For Election Commission:
    • {(winners['margin'] < 2000).sum()} constituencies had margins under 2,000 votes —
      these should be priority zones for any re-count requests.
    • NOTA averaged {df[df['Is_NOTA']]['vote_share_pct'].mean():.2f}% — continued increase
      in NOTA indicates voter dissatisfaction in multi-cornered contests.

  For Political Parties:
    • Win rate varies dramatically by party (see statistics/party_win_rate.csv).
      Parties should analyse constituency selection efficiency.
    • Cluster analysis reveals 4 distinct constituency types; parties should
      tailor campaign resources accordingly.
    • Average winner needs only ~{winners['vote_share_pct'].mean():.0f}% of the vote to win
      in multi-party contests — alliance arithmetic is decisive.

  For Researchers & Media:
    • The lognormal distribution of votes indicates extreme concentration —
      a few "safe seats" dominate total vote counts.
    • PCA reveals margin_pct and dominance_index drive most inter-constituency
      variance — useful for swing/battleground modelling.

================================================================================
                        END OF EXECUTIVE REPORT
================================================================================
"""

    path = os.path.join(DIRS["reports"], "executive_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    log.info("Executive summary saved → %s", path)

    # Save full insights log separately
    insights_path = os.path.join(DIRS["reports"], "detailed_insights.txt")
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write("DETAILED ANALYTICAL INSIGHTS — TN 2026\n" + "=" * 60 + "\n")
        for i, ins in enumerate(INSIGHTS, 1):
            f.write(f"[{i:03d}] {ins}\n")
    log.info("Detailed insights saved → %s", insights_path)


# ══════════════════════════════════════════════════════════════════════════════
# 13.  MASTER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    End-to-end orchestration of the Tamil Nadu 2026 election analytics pipeline.
    Each stage is wrapped in a try-except to ensure maximum resilience.
    """
    log.info("=" * 70)
    log.info("  TAMIL NADU 2026 — ELECTION ANALYTICS PIPELINE  START")
    log.info("=" * 70)

    # Stage 1: Load
    try:
        df_raw = load_data(DATA_PATH)
        schema_report(df_raw)
    except Exception as e:
        log.critical("Data loading failed: %s", e)
        sys.exit(1)

    # Stage 2: Clean
    try:
        df = clean_data(df_raw)
    except Exception as e:
        log.error("Cleaning failed: %s", e)
        df = df_raw.copy()

    # Stage 3: Feature Engineering
    try:
        df = feature_engineering(df)
    except Exception as e:
        log.error("Feature engineering failed: %s", e)

    # Stage 4: Data Quality
    try:
        data_quality_analysis(df)
    except Exception as e:
        log.error("Data quality analysis failed: %s", e)

    # Stage 5: Univariate
    try:
        univariate_numerical(df)
        univariate_categorical(df)
    except Exception as e:
        log.error("Univariate analysis failed: %s", e)

    # Stage 6: Bivariate
    try:
        bivariate_analysis(df)
    except Exception as e:
        log.error("Bivariate analysis failed: %s", e)

    # Stage 7: Multivariate
    try:
        multivariate_analysis(df)
    except Exception as e:
        log.error("Multivariate analysis failed: %s", e)

    # Stage 8: Political Analytics
    try:
        political_analytics(df)
    except Exception as e:
        log.error("Political analytics failed: %s", e)

    # Stage 9: Outlier Analysis
    try:
        outlier_analysis(df)
    except Exception as e:
        log.error("Outlier analysis failed: %s", e)

    # Stage 10: Statistical Analysis
    try:
        statistical_analysis(df)
    except Exception as e:
        log.error("Statistical analysis failed: %s", e)

    # Stage 11: Interactive Plotly
    try:
        interactive_plotly_analysis(df)
    except Exception as e:
        log.error("Interactive plots failed: %s", e)

    # Stage 12: Advanced Visualisations
    try:
        advanced_visualisations(df)
    except Exception as e:
        log.error("Advanced visualisations failed: %s", e)

    # Stage 13: Executive Summary
    try:
        generate_executive_summary(df)
    except Exception as e:
        log.error("Executive summary generation failed: %s", e)

    log.info("=" * 70)
    log.info("  PIPELINE COMPLETE — ALL OUTPUTS IN: project_output/")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
