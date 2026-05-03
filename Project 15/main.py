"""
FIFA Player Data Analysis — Production-Quality Script
======================================================
Dataset columns:
    Name, Country, Position, Age, Overall_Rating,
    Future Potential, Team, Value Per M$, Total_Stats Score

Run:
    python main.py
"""

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # Non-interactive backend — safe for scripts

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH    = "Fifa.csv"
OUTPUT_DIR  = "outputs"
RANDOM_SEED = 42

# Palette used across all charts
PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, filename: str) -> None:
    """Save a figure to OUTPUT_DIR and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


def _section(title: str) -> None:
    """Print a formatted section header."""
    bar = "─" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the FIFA CSV, enforce clean column names, and return a raw DataFrame.
    """
    _section("1 · Loading Data")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Normalise column names: strip whitespace, replace spaces with underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. CLEAN DATA
# ──────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Rename columns to consistent snake_case names.
    • Cast numeric columns.
    • Handle missing values.
    • Remove impossible / corrupt rows.
    • Engineer new features.
    """
    _section("2 · Cleaning & Feature Engineering")

    # ── Robust column rename (handles slight naming variations) ─────────────
    rename_map = {
        "Name"                : "name",
        "Country"             : "country",
        "Position"            : "position",
        "Age"                 : "age",
        "Overall_Rating"      : "overall_rating",
        "Future_Potential"    : "potential",
        "FuturePotential"     : "potential",   # fallback
        "Team"                : "team",
        "Value_Per_M"         : "value_m",
        "ValuePerM"           : "value_m",
        "Total_Stats_Score"   : "total_stats",
        "TotalStatsScore"     : "total_stats",
    }
    # Apply only the renames that exist in the dataframe
    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    # After rename, try to infer remaining columns positionally if needed
    expected = ["name", "country", "position", "age",
                "overall_rating", "potential", "team", "value_m", "total_stats"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"  [warn] Columns still missing after rename: {missing}")

    # ── Numeric casts ────────────────────────────────────────────────────────
    num_cols = ["age", "overall_rating", "potential", "value_m", "total_stats"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Missing value audit ──────────────────────────────────────────────────
    miss = df.isnull().sum()
    if miss.any():
        print("  Missing values per column:")
        print(miss[miss > 0].to_string(header=False))

    # Drop rows where core numeric fields are entirely absent
    df.dropna(subset=["overall_rating", "age"], inplace=True)

    # Fill remaining numeric NaNs with column median
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill object NaNs with "Unknown"
    for col in df.select_dtypes("object").columns:
        df[col].fillna("Unknown", inplace=True)

    # ── Domain sanity checks ─────────────────────────────────────────────────
    df = df[(df["age"] >= 15) & (df["age"] <= 50)]
    df = df[(df["overall_rating"] >= 40) & (df["overall_rating"] <= 99)]
    if "potential" in df.columns:
        df = df[(df["potential"] >= 40) & (df["potential"] <= 99)]

    # ── Outlier removal: value_m (IQR-based) ─────────────────────────────────
    if "value_m" in df.columns:
        q1, q3 = df["value_m"].quantile([0.25, 0.75])
        iqr     = q3 - q1
        upper   = q3 + 6 * iqr   # generous fence — star players have huge values
        df      = df[df["value_m"] >= 0]          # no negative values
        df_no_outlier = df[df["value_m"] <= upper]
        removed = len(df) - len(df_no_outlier)
        print(f"  Outlier rows removed (value_m > {upper:.1f}M): {removed}")
        df = df_no_outlier

    # ── Feature engineering ──────────────────────────────────────────────────
    if "potential" in df.columns:
        df["potential_gap"]   = df["potential"] - df["overall_rating"]   # room to grow
    if "value_m" in df.columns and "overall_rating" in df.columns:
        df["value_per_rating"] = np.where(
            df["overall_rating"] > 0,
            df["value_m"] / df["overall_rating"],
            np.nan
        )

    # Position grouping: slim down the many positional labels
    pos_group = {
        "GK" : "Goalkeeper",
        "CB" : "Defender",  "LB" : "Defender",  "RB" : "Defender",
        "LWB": "Defender",  "RWB": "Defender",
        "CDM": "Midfielder","CM" : "Midfielder", "CAM": "Midfielder",
        "LM" : "Midfielder","RM" : "Midfielder",
        "LW" : "Forward",   "RW" : "Forward",    "LF" : "Forward",
        "RF" : "Forward",   "CF" : "Forward",    "ST" : "Forward",
    }
    if "position" in df.columns:
        df["position_group"] = df["position"].map(pos_group).fillna("Other")

    # Age bands
    df["age_band"] = pd.cut(
        df["age"],
        bins=[14, 20, 25, 30, 35, 50],
        labels=["U-21", "21-25", "26-30", "31-35", "35+"],
    )

    df.reset_index(drop=True, inplace=True)
    print(f"\n  Clean dataset shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def perform_eda(df: pd.DataFrame) -> None:
    """Print summary statistics, distributions, and correlations."""
    _section("3 · Exploratory Data Analysis")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n  Numeric summary:\n")
    print(df[num_cols].describe().round(2).to_string())

    # ── Correlation matrix (console) ─────────────────────────────────────────
    corr_cols = [c for c in
                 ["overall_rating", "potential", "age", "value_m", "total_stats"]
                 if c in df.columns]
    print("\n  Pearson correlation matrix:\n")
    print(df[corr_cols].corr().round(3).to_string())

    # ── Top countries by player count ─────────────────────────────────────────
    if "country" in df.columns:
        top_countries = df["country"].value_counts().head(10)
        print("\n  Top 10 countries by player count:")
        print(top_countries.to_string())

    # ── Position-group breakdown ──────────────────────────────────────────────
    if "position_group" in df.columns:
        print("\n  Players per position group:")
        print(df["position_group"].value_counts().to_string())

    # ── Age-band vs average rating ────────────────────────────────────────────
    if "age_band" in df.columns:
        print("\n  Mean overall_rating by age band:")
        print(df.groupby("age_band", observed=True)["overall_rating"]
                .mean().round(2).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 4. STATISTICAL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def statistical_analysis(df: pd.DataFrame) -> None:
    """
    • One-way ANOVA: does mean overall rating differ by position group?
    • Spearman correlations (non-parametric, robust to outliers).
    • Simple OLS: value_m ~ overall_rating + age.
    • 95 % confidence intervals for mean overall_rating per position group.
    """
    _section("4 · Statistical Analysis")

    # ── ANOVA: overall rating across position groups ──────────────────────────
    if "position_group" in df.columns:
        groups = [
            g["overall_rating"].dropna().values
            for _, g in df.groupby("position_group", observed=True)
            if len(g) >= 5
        ]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"\n  ANOVA (overall_rating ~ position_group)")
            print(f"    F-stat = {f_stat:.3f},  p-value = {p_val:.4e}")
            if p_val < 0.05:
                print("    → Significant difference in ratings across positions (p < 0.05)")
            else:
                print("    → No significant difference detected")

    # ── Spearman correlations ─────────────────────────────────────────────────
    pairs = [
        ("overall_rating", "value_m"),
        ("overall_rating", "total_stats"),
        ("age",            "potential"),
        ("potential_gap",  "value_m"),
    ]
    print("\n  Spearman rank correlations:")
    for col_a, col_b in pairs:
        if col_a in df.columns and col_b in df.columns:
            sub = df[[col_a, col_b]].dropna()
            rho, p = stats.spearmanr(sub[col_a], sub[col_b])
            print(f"    {col_a:20s} ↔ {col_b:18s}  ρ={rho:+.3f}  p={p:.3e}")

    # ── OLS regression: value_m ~ overall_rating + age ────────────────────────
    if all(c in df.columns for c in ["value_m", "overall_rating", "age"]):
        sub = df[["value_m", "overall_rating", "age"]].dropna()
        # Design matrix with intercept
        X = np.column_stack([
            np.ones(len(sub)),
            sub["overall_rating"].values,
            sub["age"].values,
        ])
        y = sub["value_m"].values
        # Least-squares fit
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        y_pred   = X @ coeffs
        ss_res   = np.sum((y - y_pred) ** 2)
        ss_tot   = np.sum((y - y.mean()) ** 2)
        r_sq     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"\n  OLS: value_m ~ intercept + overall_rating + age")
        print(f"    Intercept      = {coeffs[0]:+.4f}")
        print(f"    overall_rating = {coeffs[1]:+.4f}  (Δvalue per rating point)")
        print(f"    age            = {coeffs[2]:+.4f}  (Δvalue per year of age)")
        print(f"    R²             = {r_sq:.4f}")

    # ── 95% CI for mean overall_rating by position group ─────────────────────
    if "position_group" in df.columns:
        print("\n  95% CI for mean overall_rating by position group:")
        for grp, sub in df.groupby("position_group", observed=True):
            vals = sub["overall_rating"].dropna()
            if len(vals) < 2:
                continue
            lo, hi = stats.t.interval(
                0.95, df=len(vals) - 1,
                loc=vals.mean(),
                scale=stats.sem(vals)
            )
            print(f"    {grp:12s}  n={len(vals):5,}  "
                  f"mean={vals.mean():.2f}  95%CI=[{lo:.2f}, {hi:.2f}]")


# ──────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
def visualize_data(df: pd.DataFrame) -> None:
    """
    Generate and save all plots to OUTPUT_DIR:
      01 – Distribution: overall_rating histogram
      02 – Distribution: age histogram
      03 – Distribution: value_m histogram
      04 – Boxplot: overall_rating by position group
      05 – Boxplot: value_m by position group
      06 – Correlation heatmap
      07 – Scatter: overall_rating vs value_m
      08 – Scatter: age vs overall_rating (coloured by potential_gap)
      09 – Bar: top 15 countries by average overall_rating
      10 – Bar: top 15 most represented countries
      11 – Line: mean rating by age band
      12 – Boxplot: potential_gap by position group
    """
    _section("5 · Visualisations")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 01 Overall-rating distribution ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["overall_rating"], bins=40, kde=True, color="#4C72B0", ax=ax)
    ax.set(title="Distribution of Overall Rating",
           xlabel="Overall Rating", ylabel="Count")
    ax.axvline(df["overall_rating"].mean(), color="red",
               linestyle="--", label=f"Mean={df['overall_rating'].mean():.1f}")
    ax.legend()
    _save(fig, "01_overall_rating_distribution.png")

    # ── 02 Age distribution ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["age"], bins=35, kde=True, color="#55A868", ax=ax)
    ax.set(title="Age Distribution of FIFA Players",
           xlabel="Age", ylabel="Count")
    ax.axvline(df["age"].mean(), color="red", linestyle="--",
               label=f"Mean={df['age'].mean():.1f}")
    ax.legend()
    _save(fig, "02_age_distribution.png")

    # ── 03 Market value distribution (log scale) ──────────────────────────────
    if "value_m" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df["value_m"], bins=60, kde=True,
                     color="#C44E52", ax=axes[0])
        axes[0].set(title="Market Value Distribution (Linear)",
                    xlabel="Value (M$)", ylabel="Count")

        pos_vals = df[df["value_m"] > 0]["value_m"]
        sns.histplot(np.log1p(pos_vals), bins=60, kde=True,
                     color="#8172B2", ax=axes[1])
        axes[1].set(title="Market Value Distribution (log1p-transformed)",
                    xlabel="log(1 + Value M$)", ylabel="Count")
        fig.suptitle("Market Value Distribution", fontsize=13, fontweight="bold")
        _save(fig, "03_market_value_distribution.png")

    # ── 04 Overall rating by position group ──────────────────────────────────
    if "position_group" in df.columns:
        order = (df.groupby("position_group", observed=True)["overall_rating"]
                   .median().sort_values(ascending=False).index.tolist())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="position_group", y="overall_rating",
                    order=order, palette="Set2", ax=ax)
        ax.set(title="Overall Rating by Position Group",
               xlabel="Position Group", ylabel="Overall Rating")
        _save(fig, "04_rating_by_position.png")

    # ── 05 Market value by position group ─────────────────────────────────────
    if "position_group" in df.columns and "value_m" in df.columns:
        order = (df.groupby("position_group", observed=True)["value_m"]
                   .median().sort_values(ascending=False).index.tolist())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="position_group", y="value_m",
                    order=order, palette="Set3", ax=ax)
        ax.set(title="Market Value by Position Group",
               xlabel="Position Group", ylabel="Value (M$)")
        _save(fig, "05_value_by_position.png")

    # ── 06 Correlation heatmap ─────────────────────────────────────────────────
    corr_cols = [c for c in
                 ["overall_rating", "potential", "potential_gap",
                  "age", "value_m", "total_stats", "value_per_rating"]
                 if c in df.columns]
    corr_mat = df[corr_cols].corr()
    fig, ax  = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 10})
    ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
    _save(fig, "06_correlation_heatmap.png")

    # ── 07 Scatter: overall_rating vs value_m ─────────────────────────────────
    if "value_m" in df.columns:
        # Sample for clarity (dataset can be huge)
        sample = df[df["value_m"] > 0].sample(
            min(3000, len(df)), random_state=RANDOM_SEED
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            sample["overall_rating"], sample["value_m"],
            c=sample["age"], cmap="plasma", alpha=0.45, s=18, edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, label="Age")
        ax.set(title="Overall Rating vs Market Value",
               xlabel="Overall Rating", ylabel="Market Value (M$)")
        # Annotate a few extreme-value players
        top5 = df.nlargest(5, "value_m")
        for _, row in top5.iterrows():
            ax.annotate(
                row.get("name", "")[:20],
                xy=(row["overall_rating"], row["value_m"]),
                fontsize=7, alpha=0.8,
                xytext=(4, 4), textcoords="offset points"
            )
        _save(fig, "07_rating_vs_value_scatter.png")

    # ── 08 Scatter: age vs overall_rating ─────────────────────────────────────
    if "potential_gap" in df.columns:
        sample = df.sample(min(3000, len(df)), random_state=RANDOM_SEED)
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            sample["age"], sample["overall_rating"],
            c=sample["potential_gap"], cmap="RdYlGn",
            alpha=0.45, s=18, edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, label="Potential Gap (potential − rating)")
        ax.set(title="Age vs Overall Rating (coloured by Potential Gap)",
               xlabel="Age", ylabel="Overall Rating")
        _save(fig, "08_age_vs_rating_potential.png")

    # ── 09 Top 15 countries by mean overall rating ─────────────────────────────
    if "country" in df.columns:
        min_players = 50
        country_stats = (
            df.groupby("country")
              .agg(mean_rating=("overall_rating", "mean"),
                   count=("overall_rating", "count"))
              .query("count >= @min_players")
              .nlargest(15, "mean_rating")
              .reset_index()
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(country_stats["country"], country_stats["mean_rating"],
                       color=sns.color_palette("viridis", len(country_stats)))
        ax.set(title=f"Top 15 Countries by Mean Overall Rating (≥{min_players} players)",
               xlabel="Mean Overall Rating")
        ax.invert_yaxis()
        for bar, val in zip(bars, country_stats["mean_rating"]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9)
        _save(fig, "09_top_countries_mean_rating.png")

    # ── 10 Top 15 most represented countries ─────────────────────────────────
    if "country" in df.columns:
        top_countries = df["country"].value_counts().head(15).reset_index()
        top_countries.columns = ["country", "count"]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(top_countries["country"], top_countries["count"],
                       color=sns.color_palette("magma", len(top_countries)))
        ax.set(title="Top 15 Most Represented Countries",
               xlabel="Player Count")
        ax.invert_yaxis()
        for bar, val in zip(bars, top_countries["count"]):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        _save(fig, "10_top_countries_player_count.png")

    # ── 11 Mean rating by age band ────────────────────────────────────────────
    if "age_band" in df.columns:
        age_stats = (
            df.groupby("age_band", observed=True)["overall_rating"]
              .agg(["mean", "sem"])
              .reset_index()
        )
        age_stats["ci95"] = age_stats["sem"] * 1.96
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(age_stats["age_band"].astype(str), age_stats["mean"],
                marker="o", linewidth=2, color="#4C72B0")
        ax.fill_between(
            age_stats["age_band"].astype(str),
            age_stats["mean"] - age_stats["ci95"],
            age_stats["mean"] + age_stats["ci95"],
            alpha=0.25, color="#4C72B0", label="95% CI"
        )
        ax.set(title="Mean Overall Rating by Age Band",
               xlabel="Age Band", ylabel="Mean Overall Rating")
        ax.legend()
        _save(fig, "11_rating_by_age_band.png")

    # ── 12 Potential gap by position group ────────────────────────────────────
    if "potential_gap" in df.columns and "position_group" in df.columns:
        order = (df.groupby("position_group", observed=True)["potential_gap"]
                   .median().sort_values(ascending=False).index.tolist())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="position_group", y="potential_gap",
                    order=order, palette="coolwarm", ax=ax)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set(title="Potential Gap (potential − rating) by Position Group",
               xlabel="Position Group", ylabel="Potential Gap")
        _save(fig, "12_potential_gap_by_position.png")

    print(f"\n  All plots saved to /{OUTPUT_DIR}/")


# ──────────────────────────────────────────────────────────────────────────────
# 6. INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame) -> None:
    """Print high-value, non-obvious findings derived from the data."""
    _section("6 · Key Insights")

    insights = []

    # ── Rating distribution ───────────────────────────────────────────────────
    mu  = df["overall_rating"].mean()
    med = df["overall_rating"].median()
    sk  = df["overall_rating"].skew()
    insights.append(
        f"Rating distribution: mean={mu:.1f}, median={med:.1f}, "
        f"skew={sk:.2f}. {'Right-skewed — most players cluster below the mean.' if sk > 0.3 else 'Roughly symmetric.'}"
    )

    # ── Age at peak rating ────────────────────────────────────────────────────
    if "age" in df.columns:
        peak_age = (
            df.groupby("age")["overall_rating"].mean().idxmax()
        )
        insights.append(
            f"Players peak in overall rating at age {peak_age}. "
            "Talent scouting should track players 3–5 years before this window."
        )

    # ── Young high-potential gems ──────────────────────────────────────────────
    if "potential_gap" in df.columns and "age" in df.columns:
        gems = df[(df["age"] <= 21) & (df["potential_gap"] >= 10)]
        insights.append(
            f"There are {len(gems):,} players aged ≤21 with a potential gap ≥10 "
            "points — strong targets for future investment."
        )

    # ── Over/under-valued players ─────────────────────────────────────────────
    if "value_per_rating" in df.columns:
        high_val = df[df["value_per_rating"] > df["value_per_rating"].quantile(0.95)]
        low_val  = df[(df["value_per_rating"] < df["value_per_rating"].quantile(0.10))
                      & (df["value_m"] > 0)]
        insights.append(
            f"{len(high_val):,} players appear significantly overvalued relative to "
            "their rating (top 5% cost-per-rating). Conversely, "
            f"{len(low_val):,} players in the bottom 10% may represent bargains."
        )

    # ── Top earning positions ──────────────────────────────────────────────────
    if "position_group" in df.columns and "value_m" in df.columns:
        top_pos = (
            df.groupby("position_group", observed=True)["value_m"]
              .median().sort_values(ascending=False).idxmax()
        )
        insights.append(
            f"The position group commanding the highest median market value "
            f"is: {top_pos}."
        )

    # ── Veteran high-performers ────────────────────────────────────────────────
    if "age" in df.columns:
        vets = df[(df["age"] >= 35) & (df["overall_rating"] >= 80)]
        insights.append(
            f"{len(vets):,} players aged 35+ still maintain an overall rating ≥80, "
            "indicating elite longevity. These players can be valuable mentors / "
            "squad fillers."
        )

    # ── Country-level talent concentration ───────────────────────────────────
    if "country" in df.columns:
        top_elite_country = (
            df[df["overall_rating"] >= 80]
              .groupby("country").size()
              .sort_values(ascending=False)
              .idxmax()
        )
        insights.append(
            f"The country producing the most elite-rated players (≥80) "
            f"is: {top_elite_country}."
        )

    # ── Print all insights ─────────────────────────────────────────────────────
    for i, txt in enumerate(insights, 1):
        print(f"\n  [{i}] " + "\n      ".join(textwrap.wrap(txt, width=72)))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw   = load_data()
    df_clean = clean_data(df_raw)
    perform_eda(df_clean)
    statistical_analysis(df_clean)
    visualize_data(df_clean)
    generate_insights(df_clean)

    _section("✓ Analysis Complete")
    print(f"  All outputs written to /{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
