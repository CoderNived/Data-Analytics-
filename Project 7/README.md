# 🌐 Global B2B Lead Intelligence — Analytics Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-27AE60?style=for-the-badge"/>
</p>

<p align="center">
  <em>A senior-grade, end-to-end data analytics pipeline that transforms raw B2B contact data into structured sales intelligence — with automated cleaning, ML-powered segmentation, and actionable business recommendations.</em>
</p>

---

## 📌 Overview

In B2B sales, **data quality and prospect intelligence are everything.** Most CRMs are filled with raw, unstructured contact lists — inconsistent country names, ambiguous job titles, personal email addresses mixed with business ones, and zero segmentation.

This project solves that problem.

The **Global B2B Lead Intelligence Pipeline** ingests raw B2B contact data from a CSV file, runs it through a multi-stage analytics pipeline, and outputs a fully enriched, ML-segmented lead dataset — paired with 6 publication-quality visual dashboards and a prioritized outreach strategy.

**Business impact:**
- Identify the highest-value buyer personas in your dataset instantly
- Know which industries and countries to prioritize for ABM (Account-Based Marketing)
- Segment leads into distinct buyer clusters to personalize outreach at scale
- Eliminate bad data (invalid emails, duplicates, unknown entries) before it enters your CRM

> 📁 GitHub Repository: [CoderNived / Data-Analytics / Project 7](https://github.com/CoderNived/Data-Analytics-/tree/main/Project%207)

---

## 🧠 Key Features

| Feature | Description |
|---|---|
| 🧹 **Data Cleaning** | Deduplication, whitespace normalization, country name standardization |
| 📧 **Email Validation** | Regex-based email format verification + Business vs Personal domain classification |
| 👔 **Role Classification** | Rule-based NLP parsing of job titles into 8 structured role categories |
| 🏷️ **Seniority Mapping** | Hierarchical seniority scoring (Executive → Senior → Mid-Level → Junior) |
| 🗺️ **Region Engineering** | Maps 40+ countries to 6 global regions for geo-targeting |
| 🤖 **KMeans Clustering** | Unsupervised ML segmentation into 5 named B2B buyer personas |
| 📉 **PCA Visualization** | 2D dimensionality reduction to visualize cluster separation |
| 📊 **6 Visual Dashboards** | Production-quality charts covering industry, geography, email, and segmentation |
| 💡 **Sales Intelligence Report** | Tier-based lead prioritization strategy printed as a structured output |
| 📤 **Enriched Export** | Clean, enriched CSV ready for CRM import or further modeling |

---

## 🛠️ Tech Stack

```
Language          : Python 3.10+
Data Layer        : Pandas, NumPy
Visualization     : Matplotlib, Seaborn
Machine Learning  : Scikit-learn (KMeans, PCA, LabelEncoder)
Data Quality      : re (Regex), custom validation functions
Output            : CSV export + PNG dashboards
```

---

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              GLOBAL B2B LEAD INTELLIGENCE PIPELINE              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  SECTION 1       │────▶│  SECTION 2       │────▶│  SECTION 3       │
│  Data Loading    │     │  Data Cleaning   │     │  EDA             │
│  & Inspection    │     │  & Preprocessing │     │  & Profiling     │
│                  │     │                  │     │                  │
│ • Shape / types  │     │ • Deduplication  │     │ • Role counts    │
│ • Missing values │     │ • Standardize    │     │ • Industry dist. │
│ • Duplicate check│     │   countries      │     │ • Country dist.  │
│ • Email regex    │     │ • Normalize      │     │                  │
│   validation     │     │   titles &       │     │                  │
│ • Special chars  │     │   industries     │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  SECTION 4       │────▶│  SECTION 5       │────▶│  SECTION 6       │
│  Sales           │     │  Feature         │     │  Clustering &    │
│  Intelligence    │     │  Engineering     │     │  Segmentation    │
│                  │     │                  │     │                  │
│ • Top industries │     │ • Seniority level│     │ • LabelEncoding  │
│ • Target markets │     │ • Email domain   │     │ • KMeans (k=5)   │
│ • Role profiles  │     │ • Email type     │     │ • PCA projection │
│ • C-Level %      │     │   (Biz/Personal) │     │ • Persona naming │
│                  │     │ • Region mapping │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  SECTION 7       │────▶│  SECTION 8       │────▶│  SECTION 9       │
│  Visualization   │     │  Business        │     │  Enriched CSV    │
│  Dashboards      │     │  Recommendations │     │  Export          │
│                  │     │                  │     │                  │
│ • 6 figures      │     │ • Tier-1/2/3     │     │ • 12 enriched    │
│   saved as PNGs  │     │   lead strategy  │     │   feature cols   │
│                  │     │ • Buyer personas │     │ • CRM-ready      │
│                  │     │ • Country targets│     │   output file    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 📈 Visual Outputs

The pipeline generates **6 production-quality figures**, each saved as a high-resolution PNG.

### `fig1_overview_dashboard.png` — Overview Dashboard
A 2×3 grid master dashboard featuring:
- Horizontal bar chart of Top 15 Industries
- Role distribution pie chart
- Top 15 Countries bar chart
- Regional breakdown horizontal bar

### `fig2_industry_role_heatmap.png` — Industry × Role Matrix
A custom-colormap heatmap showing the intersection of Top 10 Industries with each leadership role category — perfect for identifying high-executive-density verticals at a glance.

### `fig3_seniority_email_intel.png` — Email & Seniority Intelligence
Three-panel chart covering:
- Seniority level donut chart
- Business vs Personal email type bar chart
- Top 12 business email domains

### `fig4_clustering_segmentation.png` — B2B Lead Segmentation
- Named buyer segment distribution bar chart
- PCA 2D scatter plot with annotated cluster centroids (% variance explained labeled on axes)

### `fig5_geo_industry_intelligence.png` — Geographic Intelligence
A 2×2 dashboard containing:
- Country stacked bar (colored by region)
- Role mix by region (100% stacked %)
- Industry × seniority breakdown
- Email type by region (%)

### `fig6_lead_prioritization_matrix.png` — Lead Prioritization Matrix
Bubble chart with:
- X-axis: Total decision-makers per industry
- Y-axis: Executive-level percentage per industry
- Bubble size: Volume indicator
- Quadrant lines at medians for instant tier classification

---

## 💡 Key Insights

> Derived from the analytics pipeline across all processed records.

**🏭 Industry Concentration**
- Financial Services, Technology, and Professional Services consistently rank as the top 3 industries by decision-maker density
- These verticals also carry the highest executive-level concentration — making them the highest-ROI targets for ABM campaigns

**🌍 Geographic Distribution**
- The dataset is heavily weighted toward **North America and Europe**, making it ideal for English-language outbound and inbound campaigns
- Asia-Pacific represents an emerging growth region with strong mid-market representation

**👔 Buyer Persona Breakdown**
- **C-Level contacts** hold direct budget authority — ideal for enterprise deals with high ACV
- **Founders/Owners** offer the fastest sales cycles, particularly for SMB or PLG-motion products
- **Directors** in Healthcare and Professional Services signal long-term contract potential

**📧 Email Quality**
- The majority of contacts carry verifiable **business email addresses** — the dataset is outreach-ready
- A meaningful minority use personal domains (Gmail, Yahoo, Outlook) — these should be deprioritized or enriched before outreach

---

## 🎯 Business Recommendations

### Lead Prioritization Tiers

| Tier | Criteria | Action |
|---|---|---|
| **Tier 1** 🔥 | C-Level + Business Email + Top 5 Industry | Immediate personalized outreach |
| **Tier 2** 🟡 | Director / VP + Business Email + Any Industry | Structured nurture sequence |
| **Tier 3** 🔵 | Manager-level + Any Industry | Volume-based automated campaign |
| **De-prioritize** ⚪ | Personal email domain or Unknown Industry | Enrich before contact |

### Target Buyer Segments

- **Enterprise Buyers** — C-Suite in large established industries; high ACV potential
- **Startup Founders** — Owner-operated businesses; short sales cycles, decision-maker is also the buyer
- **Technical Leadership** — Architects, Engineers, Scientists; champion-building and bottom-up adoption plays
- **Operations / Mid-Market** — Managers and Heads; strong for expansion and upsell motions
- **Emerging Market Leaders** — High-growth regions; invest in localization and regional partnerships

---

## 📂 Project Structure

```
project-7-b2b-lead-intelligence/
│
├── 📄 b2b_pipeline.py                        # Main analytics pipeline script
├── 📄 README.md                              # Project documentation
│
├── 📁 data/
│   └── globalb2bdataset.csv                  # Raw input dataset
│
└── 📁 outputs/
    ├── b2b_enriched_leads.csv                # Enriched & segmented lead export
    ├── fig1_overview_dashboard.png           # Master overview dashboard
    ├── fig2_industry_role_heatmap.png        # Industry × Role heatmap
    ├── fig3_seniority_email_intel.png        # Seniority & email intelligence
    ├── fig4_clustering_segmentation.png      # KMeans cluster visualization
    ├── fig5_geo_industry_intelligence.png    # Geographic intelligence dashboard
    └── fig6_lead_prioritization_matrix.png  # Bubble chart prioritization matrix
```

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/CoderNived/Data-Analytics-.git
cd "Data-Analytics-/Project 7"
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or via a requirements file:
```bash
pip install -r requirements.txt
```

### 3. Add Your Dataset
Place your CSV file at the expected input path:
```
data/globalb2bdataset.csv
```

Ensure the CSV contains at minimum these columns:

| Column | Description |
|---|---|
| `Decision Maker Name` | Full name of the contact |
| `Decision Maker Title` | Job title (used for role classification) |
| `Email Address` | Contact email (validated + typed) |
| `Industry` | Company industry vertical |
| `Country` | Country of operation |

### 4. Run the Pipeline
```bash
python b2b_pipeline.py
```

### 5. View Outputs
All figures and the enriched CSV are saved to:
```
outputs/
```

---

## 📌 Future Improvements

- [ ] **Predictive Lead Scoring** — Train an XGBoost or logistic regression model on historical conversion data to assign a propensity score to each lead
- [ ] **Firmographic Enrichment** — Integrate Clearbit, ZoomInfo, or Apollo APIs to append company size, revenue, and LinkedIn URLs
- [ ] **Intent Data Layer** — Overlay third-party intent signals (G2, Bombora) for real-time buying signal detection
- [ ] **Real-Time Pipeline** — Migrate to Apache Airflow or Prefect for scheduled, automated pipeline runs on new data drops
- [ ] **Interactive Dashboard** — Port visualizations to Plotly Dash or Streamlit for a shareable, interactive web app
- [ ] **NLP Title Parser** — Replace rule-based role classification with a fine-tuned NLP model for higher accuracy on non-standard titles
- [ ] **CRM Integration** — Add direct push to Salesforce, HubSpot, or Pipedrive via their APIs post-scoring

---

## 👤 Author

**Nived**
Senior Data Analytics Portfolio — [GitHub @CoderNived](https://github.com/CoderNived)

> *This project is part of a series of production-grade data analytics case studies demonstrating real-world data engineering and data science skills — covering the full pipeline from raw data ingestion to business-ready intelligence.*

---

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Python-3776AB?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/Domain-B2B%20Sales%20Intelligence-1A3C5E?style=flat-square"/>
  <img src="https://img.shields.io/badge/Type-Portfolio%20Project-27AE60?style=flat-square"/>
</p>
