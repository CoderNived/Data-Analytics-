# 🤖 Project — AI vs Human Research Text Classification

> **Production-grade NLP analytics and machine learning pipeline for detecting AI-generated vs. human-written academic research text — covering full EDA, feature engineering, statistical analysis, and baseline classification modeling.**

---

## 📁 Repository Structure

```
Data-Analytics-
│
├── Project — AI vs Human Text Classification/    ← NLP Text Classification Pipeline
│   ├── Dataset/                                  ← Raw dataset (not tracked — see Dataset Info)
│   ├── Notebook/                                 ← Analysis scripts and ML pipeline
│   │   └── analysis_pipeline.py                 ← Production Python NLP + ML script
│   └── Outputs/                                  ← Generated charts, plots, and reports
│       ├── fig1_overview_dashboard.png           ← 8-panel KPI dashboard + distributions
│       ├── fig2_top_words.png                    ← Top 20 content words — AI vs Human
│       ├── fig3_correlation_heatmap.png          ← NLP feature correlation matrix
│       ├── fig4_violin_features.png              ← Feature distributions (violin plots)
│       ├── fig5_word_clouds.png                  ← Pseudo word clouds — AI vs Human
│       ├── fig6_scatter_sentence.png             ← Word count vs sentence length scatter
│       ├── fig7_ml_results.png                   ← Confusion matrices + feature importance
│       ├── data_cleaned_features.csv             ← Clean dataset with all engineered features
│       └── final_report.txt                      ← Full analytics summary report
│
├── .gitignore                                    ← Excludes large dataset files
├── requirements.txt                              ← Python dependencies
└── README.md                                     ← You are here
```

---

## 📌 Project Overview

This project performs a **complete end-to-end NLP Data Science workflow** on a corpus of academic research texts, distinguishing between AI-generated and human-written entries — covering text preprocessing, statistical EDA, NLP feature engineering, readability analysis, and TF-IDF-based classification modeling.

Each record in the dataset represents a single research text entry and captures:
- Raw text content: full abstract or paper section
- Authorship label: binary classification target (AI or Human)

### 💼 Why This Analysis Matters

The proliferation of AI-generated content in academic publishing is one of the most pressing integrity challenges in modern research. As LLMs become increasingly capable of producing fluent, domain-specific prose, distinguishing AI from human authorship has significant implications for peer review, plagiarism detection, and scholarly trust.

| Business Metric | Impact |
|---|---|
| Classification Accuracy | 99.3% — highly reliable automated detection |
| Dataset Balance | Near-perfectly balanced: AI 50.6% vs Human 49.4% |
| Discriminability | A simple TF-IDF signal achieves AUC > 0.99 |
| Research Integrity | Early detection of AI content protects academic credibility |
| Scalability | Pipeline is modular and production-deployable as a REST API |
| Feature Insight | Statistical patterns reveal structural differences in writing style |

---

## 📊 Dataset Info

| Property | Detail |
|---|---|
| **Source** | AI vs Human Research Text Dataset (CSV) |
| **Rows** | 6,069 (after cleaning) |
| **Columns** | 2 raw + 8 engineered NLP features |
| **Format** | CSV |
| **Domain** | NLP / Academic Text / AI Detection |
| **Labels** | Binary — AI, Human |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `Text` | string | Full research text (abstract or paper section) |
| `Author` | categorical | Target label — `AI` or `Human` |
| `word_count` | int | Total number of words in the text |
| `char_count` | int | Total non-whitespace character count |
| `sentence_count` | int | Number of sentences detected |
| `avg_word_len` | float | Mean character length per word |
| `avg_sentence_len` | float | Mean word count per sentence |
| `unique_word_ratio` | float | Type-Token Ratio — vocabulary richness proxy |
| `flesch_score` | float | Flesch Reading Ease score (higher = simpler) |
| `gunning_fog` | float | Gunning Fog Index — estimated grade level required |

> ⚠️ **Dataset not tracked by Git.** Place the CSV file inside `Dataset/` before running the notebook.

---

## 🔬 Analysis Walkthrough

The notebook follows a structured, 10-section analytics workflow:

### 1. 📋 Data Understanding
Business framing, column classification (raw text / label / engineered), shape inspection, data type validation, class distribution, and detection of potential data quality issues.

### 2. 🔍 Data Cleaning
- Unnamed index column detected and dropped
- Zero missing values confirmed across all columns
- Zero duplicate records found
- Author labels standardised (`Ai` → `AI`)
- Whitespace normalised across all text entries

### 3. 🛠️ Feature Engineering

Eight NLP features engineered from raw text without reliance on external NLP libraries:

| Feature | Formula / Logic | Purpose |
|---|---|---|
| `word_count` | `len(text.split())` | Measures text length — primary discriminator |
| `char_count` | `len(re.sub(r"\s","",text))` | Character-level volume metric |
| `sentence_count` | Split on `.!?` punctuation | Structural complexity proxy |
| `avg_word_len` | Mean character count per word | Vocabulary sophistication signal |
| `avg_sentence_len` | `word_count / sentence_count` | Sentence complexity measure |
| `unique_word_ratio` | `unique_words / total_words` (TTR) | Vocabulary richness / lexical diversity |
| `flesch_score` | Flesch Reading Ease formula (manual) | Readability — higher = more accessible |
| `gunning_fog` | Gunning Fog Index formula (manual) | Grade level required to understand text |

### 4. 📈 Exploratory Data Analysis

| Chart | What It Shows |
|---|---|
| `fig1_overview_dashboard.png` | Class distribution, word count histogram, boxplots, TTR histogram, Flesch/Fog distributions, sentence count distribution |
| `fig2_top_words.png` | Top 20 stop-word-filtered content words per author class |
| `fig3_correlation_heatmap.png` | Lower-triangle Pearson correlation heatmap across all 8 NLP features |
| `fig4_violin_features.png` | Violin plots for all features, split by AI vs Human |
| `fig5_word_clouds.png` | Frequency-scaled pseudo word clouds for each author class |
| `fig6_scatter_sentence.png` | Word count vs. average sentence length with fitted trend line, per class |
| `fig7_ml_results.png` | Confusion matrices, model performance bar chart, TF-IDF feature importance |

### 5. 📐 Statistical Insights

Welch's independent t-tests performed across all engineered features:

| Feature | AI Mean | Human Mean | p-value | Significant? |
|---|---|---|---|---|
| `word_count` | 44 | 271 | ≈ 0 | ✅ ** |
| `sentence_count` | 3.1 | 16.8 | ≈ 0 | ✅ ** |
| `avg_sentence_len` | 15.5 | 18.6 | < 0.01 | ✅ ** |
| `unique_word_ratio` | 0.862 | 0.571 | ≈ 0 | ✅ ** |
| `flesch_score` | 18.9 | 16.4 | < 0.01 | ✅ ** |
| `gunning_fog` | 18.5 | 19.8 | < 0.01 | ✅ ** |
| `avg_word_len` | 5.89 | 5.91 | 0.088 | ❌ ns |

### 6. 🤖 Predictive Modeling

Two classifiers evaluated on a stratified 80/20 train-test split using TF-IDF (unigrams + bigrams, 20K features):

| Model | Accuracy | Weighted F1 | CV F1 (5-fold) |
|---|---|---|---|
| **TF-IDF + Logistic Regression** ⭐ | **99.3%** | **99.3%** | **99.5% ± 0.2%** |
| TF-IDF + Naive Bayes | 97.5% | 97.5% | 98.2% ± 0.4% |

**Logistic Regression — Per-Class Performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AI | 0.9856 | 1.0000 | 0.9927 |
| Human | 1.0000 | 0.9850 | 0.9924 |
| **Overall** | **0.9927** | **0.9926** | **0.9926** |

**Top Discriminative TF-IDF Features (Logistic Regression Coefficients):**

| Token | Direction | Coefficient |
|---|---|---|
| `we` | → AI | −5.14 |
| `and` | → Human | +4.42 |
| `were` | → Human | +4.10 |
| `was` | → Human | +3.80 |
| `this study` | → Human | +2.27 |
| `uses` | → AI | −1.89 |

---

## 🔑 Key Findings

1. **AI texts are 6× shorter than human texts** — AI entries average 44 words vs 271 for human entries, revealing the dataset captures short AI-generated summaries against full human-authored sections. Text length alone is a near-perfect discriminator.

2. **TF-IDF + Logistic Regression achieves 99.3% accuracy** — confirming that surface-level lexical patterns are highly discriminative. Cross-validated F1 of 99.5% demonstrates robust generalisability beyond the test split.

3. **AI texts avoid the first person** — `"we"` is the single strongest negative coefficient (→ AI), suggesting AI-generated academic text systematically avoids first-person plural language, a key stylometric tell.

4. **Human texts use richer connective language** — tokens like `and`, `were`, `was`, `however`, `findings`, and `this study` are strongly human-associated, reflecting fuller narrative structure and retrospective reporting.

5. **Vocabulary richness (TTR) is inflated by short AI texts** — AI's higher TTR (0.862 vs 0.571) is an artifact of short text length, not genuine lexical diversity. Length-normalised comparisons are essential for fair assessment.

6. **Readability scores are statistically different but practically similar** — both groups produce dense, graduate-level academic language (Flesch ≈ 17–19, Fog ≈ 18–20), making readability a weak standalone classifier.

7. **Average word length is NOT a significant discriminator** (p = 0.088) — both AI and human academic writers use vocabulary of near-identical average length, confirming domain calibration overrides authorship style at the word level.

8. **At 30 days of inactivity the at-risk boundary blurs** — the `avg_sentence_len` gap (15.5 vs 18.6 words/sentence) is statistically significant, suggesting AI text favours shorter, more direct sentence constructions even in academic register.

---

## 🛠️ Tech Stack

```
Python 3.12
├── pandas          — data loading, cleaning, and feature storage
├── numpy           — numerical operations and vectorised feature computation
├── matplotlib      — multi-panel dashboard layout and custom visualisations
├── seaborn         — correlation heatmaps and statistical plot overlays
└── scikit-learn    — TfidfVectorizer, Pipeline, LogisticRegression,
                      MultinomialNB, cross_val_score, confusion_matrix, ROC
```

> All NLP utilities (tokenisation, syllable counting, readability metrics, stop-word filtering) implemented natively — **no NLTK or spaCy dependency required.**

---

## ⚡ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Data-Analytics-.git
cd "Data-Analytics-/Project — AI vs Human Text Classification"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place data_for_preprocessing.csv inside Dataset/

# 4. Run the full analysis
python Notebook/analysis_pipeline.py

# All charts and reports will be saved to Outputs/
```

---

## 📦 requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

---

## 🗺️ Dashboard Recommendations

This pipeline is ready to be operationalised as a live AI-content detection and monitoring layer:

| Tool | Suggested Dashboard |
|---|---|
| **Streamlit** | Real-time text classifier — paste a text excerpt, receive AI probability score |
| **FastAPI** | REST endpoint for batch classification of submitted manuscripts or assignments |
| **Power BI** | Institutional dashboard tracking AI content rates across departments over time |
| **Gradio** | Lightweight public demo with per-token feature attribution via SHAP |

---

## 🚀 Future Work

| Direction | Description |
|---|---|
| 🎯 Transformer-Based Models | Fine-tune DeBERTa or RoBERTa for richer semantic-level classification beyond TF-IDF |
| 📏 Length De-Confounding | Train and evaluate on length-matched subsets to isolate pure stylistic signal |
| 🔬 Perplexity Features | Add LM-based perplexity as a meta-feature — AI text consistently scores lower |
| ✍️ Stylometric Expansion | Incorporate passive voice ratio, hedge word density, punctuation patterns, citation frequency |
| 🧪 Ablation Studies | Measure marginal contribution of each feature group (lexical / syntactic / readability) |
| 🌐 Cross-Domain Validation | Test pipeline on non-academic text (news, social media, blog posts) to assess generalisability |
| 📡 Live Pipeline | Airflow-orchestrated scoring pipeline for real-time manuscript screening at submission |

---

## 👤 Author

**Senior Data Scientist & NLP Analytics Engineer**
Portfolio project demonstrating production-grade text analytics, native NLP feature engineering, multi-model evaluation, and statistical hypothesis testing on a real-world AI detection task.

---

## 📄 License

This project is licensed under the MIT License.

---

*Part of the [Data-Analytics- Portfolio](../README.md) — a collection of end-to-end analytics projects.*
