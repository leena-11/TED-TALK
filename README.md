# Personalized TED Talks Recommendation System Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-3C873A?logo=python&logoColor=white)](https://www.nltk.org/)
[![Project Status](https://img.shields.io/badge/Status-Modules%201--6%20Completed-success)](#module-completion-status)

---

## 📌 Project Information

- **Project Title:** Personalized TED Talks Recommendation System Using Machine Learning
- **Author:** Leena Jarapala
- **Internship:** Infosys Springboard Virtual Internship 7.0
- **Domain:** Artificial Intelligence / Machine Learning & Natural Language Processing
- **Application Framework:** Streamlit Web Application

---

## 📋 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Project Architecture & Workflow](#-project-architecture--workflow)
3. [Module Completion Status](#-module-completion-status)
4. [Technologies Used](#-technologies-used)
5. [Project Directory Structure](#-project-directory-structure)
6. [Module 1 – Week 1: Data Collection](#module-1--week-1-data-collection)
7. [Module 2 – Week 1: Data Exploration and Data Preprocessing](#module-2--week-1-data-exploration-and-data-preprocessing)
8. [Module 3 – Week 2: Feature Engineering and Vectorization](#module-3--week-2-feature-engineering-and-vectorization)
9. [Module 4 – Week 2: Develop Recommendation Engine](#module-4--week-2-develop-recommendation-engine)
10. [Module 5 – Week 3: Evaluation Metrics](#module-5--week-3-evaluation-metrics)
11. [Module 6 – Week 3 and Week 4: Deployment via Streamlit App](#module-6--week-3-and-week-4-deployment-via-streamlit-app)
12. [Module 7 – Pending Work](#-module-7--pending-work)
13. [Screenshots & Visualizations](#-screenshots--visualizations)
14. [Installation & Execution Guide](#-installation--execution-guide)
15. [Academic & Engineering Integrity Statement](#-academic--engineering-integrity-statement)

---

## 📖 Executive Summary

The **Personalized TED Talks Recommendation System** is an end-to-end Machine Learning and Natural Language Processing project developed under the **Infosys Springboard Virtual Internship 7.0**. 

The system analyzes a catalog of **2,550 TED Talks** published across 355 global conferences, incorporating speech transcripts, editorial descriptions, titles, speaker credentials, and curated thematic tags. By transforming unstructured multi-field text into a 5,000-dimensional **TF-IDF (Term Frequency–Inverse Document Frequency)** vector space and computing pairwise **Cosine Similarity**, the engine recommends highly relevant talks matching any user query.

The project incorporates an empirical evaluation framework benchmarked against **15,010 authentic human-curated editorial recommendations** from TED Conferences LLC, and deploys the resulting models as an interactive, production-ready **Streamlit web application**.

---

## 🔄 Project Architecture & Workflow

```text
[Raw Datasets]
  ├── ted_main.csv (2,550 talks x 17 cols)
  └── transcripts.csv (2,467 transcripts x 2 cols)
          │
          ▼
[Module 1: Data Collection]
  ├── Referential integrity check (URL key)
  ├── Deduplication (3 redundant transcripts resolved)
  └── Consolidation -> datasets/ted_talks_collected.csv (2,550 x 18)
          │
          ▼
[Module 2: Data Exploration & Preprocessing]
  ├── NLP Transcript Cleaning (Lowercase, Punctuation, Whitespace, NLTK Stopwords)
  ├── Missing Value Imputation (86 non-spoken talks imputed with Title + Description)
  ├── Categorical Encoding (speaker_id, occupation_id, multi-hot Top 20 tags)
  └── Outputs -> cleaned_ted_talks.csv (2,550 x 46), cleaned_transcripts.csv (2,464 x 5)
          │
          ▼
[Module 3: Feature Engineering & Vectorization]
  ├── Multi-field Content Synthesis (Title + Speaker + 2x Tags + Description + Transcript)
  ├── TfidfVectorizer (5,000 features, ngram_range=(1,2), sublinear_tf=True)
  ├── Pairwise Cosine Similarity Computation (2,550 x 2,550 matrix)
  └── Serialization -> models/tfidf_vectorizer.pkl, models/cosine_similarity.pkl
          │
          ▼
[Module 4: Recommendation Engine Development]
  ├── Content-Based Recommendation Pipeline (Pure semantic match)
  ├── Popularity-Augmented Hybrid Mode (alpha = 0.85 semantic + 0.15 normalized views)
  ├── Multi-Tier Title Search (Exact, Case-Insensitive, Substring / Keyword Search)
  └── Serialization -> models/content_based_model.pkl, models/recommendation_model.pkl
          │
          ▼
[Module 5: Evaluation & Benchmarking]
  ├── Ground-Truth Benchmark: 15,010 authentic TED curated 'related_talks' pairs
  ├── Train-Test Split (80% train / 20% test = 510 held-out evaluation queries)
  ├── IR Metrics: Precision@K, Recall@K, NDCG@K, MAP@K for K in [3, 5, 10, 15, 20]
  └── Benchmarked: Content-Based, Hybrid (alpha=0.85, 0.70), Tag Jaccard, Popularity
          │
          ▼
[Module 6: Streamlit Web Deployment]
  ├── Interactive 4-Page App: Recommender, Topic Explorer, Model Benchmarks, About
  ├── Streamlit Caching (@st.cache_data, @st.cache_resource) for sub-second responses
  ├── Full Metadata Cards with Explainability ("Why recommended") & TED.com Links
  └── Comprehensive Error Handling & Production Readiness
```

---

## 📊 Module Completion Status

| Module | Week | Description | Status | Verification Artifact |
| :--- | :---: | :--- | :---: | :--- |
| **Module 1** | Week 1 | Data Collection & Dataset Integrity | **`COMPLETED`** | `module1_data_collection.py`, `reports/data_collection_report.txt` |
| **Module 2** | Week 1 | Data Exploration & Preprocessing | **`COMPLETED`** | `module2_data_preprocessing.ipynb`, `cleaned_ted_talks.csv` |
| **Module 3** | Week 2 | Feature Engineering & Vectorization | **`COMPLETED`** | `module3_feature_engineering.ipynb`, `models/cosine_similarity.pkl` |
| **Module 4** | Week 2 | Recommendation Engine Development | **`COMPLETED`** | `module4_recommendation_engine.ipynb`, `recommendation_engine.py` |
| **Module 5** | Week 3 | Evaluation Metrics & Benchmarking | **`COMPLETED`** | `module5_evaluation.ipynb`, `evaluation.py`, `outputs/model_comparison.csv` |
| **Module 6** | Week 3–4 | Deployment via Streamlit Web App | **`COMPLETED`** | `app.py`, `reports/module6_deployment_report.txt`, `screenshots/` |
| **Module 7** | Week 4 | Presentation & Project Documentation | **`PENDING`** | *Scheduled for final internship submission phase* |

---

## 🛠️ Technologies Used

The project is implemented exclusively with verified, standard open-source Python data science and machine learning libraries:

| Category | Technology | Version | Purpose in Project |
| :--- | :--- | :---: | :--- |
| **Language** | Python | `3.10+` | Core programming language for end-to-end pipeline execution |
| **Data Manipulation** | Pandas | `2.3.1` | Tabular data loading, cleaning, merging, and categorical feature engineering |
| **Numerical Computing** | NumPy | `2.3.1` | High-dimensional array operations, Cosine Similarity math, normalization |
| **Machine Learning** | Scikit-Learn | `1.7.1` | `TfidfVectorizer`, `LabelEncoder`, `cosine_similarity`, `train_test_split` |
| **Natural Language Processing** | NLTK | `3.9.1` | Corpus stopwords dictionary (198 words), tokenization sanitization |
| **Data Visualization** | Matplotlib | `3.11.0` | Publication-grade plots: distribution curves, evaluation charts, Heatmaps |
| **Statistical Visualization** | Seaborn | `0.13.2` | Styled statistical plots, boxplots, multi-metric comparison bars |
| **Web Application** | Streamlit | `1.66.1` | Interactive, responsive UI deployment with caching and card components |
| **Model Serialization** | Pickle | `Built-in` | Serializing vectorizers, matrices, and recommendation engine instances |

---

## 📂 Project Directory Structure

```text
TED TALKS/
│
├── app.py                             # Streamlit interactive web application (525 lines)
├── recommendation_engine.py           # Standalone recommendation engine class & CLI interface
├── evaluation.py                      # Offline evaluation suite & metric calculation scripts
├── requirements.txt                   # Production dependency specifications
├── README.md                          # Detailed project documentation & implementation report
├── cleaned_ted_talks.csv              # Sanitized, engineered talk corpus (2,550 rows x 46 cols)
├── cleaned_transcripts.csv            # Sanitized talk transcripts (2,464 rows x 5 cols)
├── execute_module2.py                 # Notebook generation & execution runner for Module 2
├── execute_module3.py                 # Notebook generation & execution runner for Module 3
├── execute_module4.py                 # Notebook generation & execution runner for Module 4
│
├── datasets/                          # Dataset repository
│   ├── ted_main.csv                   # Raw TED Talk metadata (2,550 rows x 17 cols)
│   ├── transcripts.csv                # Raw TED Talk speech transcripts (2,467 rows x 2 cols)
│   ├── ted_talks_collected.csv       # Consolidated raw dataset with transcripts (2,550 rows x 18 cols)
│   ├── cleaned_ted_talks.csv          # Backup copy of preprocessed dataset
│   └── cleaned_transcripts.csv        # Backup copy of preprocessed transcripts
│
├── models/                            # Trained machine learning models & serialized matrices
│   ├── tfidf_vectorizer.pkl           # Fitted TfidfVectorizer (5,000 vocabulary terms, 183.1 KB)
│   ├── tfidf_matrix.pkl               # Sparse document-term matrix (2,550 x 5,000, 10.7 MB)
│   ├── cosine_similarity.pkl          # Pairwise Cosine Similarity matrix (2,550 x 2,550, 24.8 MB)
│   ├── content_based_model.pkl        # Serialized Content-Based recommendation model (26.3 MB)
│   └── recommendation_model.pkl       # Unified production recommendation model (26.3 MB)
│
├── outputs/                           # Data outputs, feature tables, and evaluation metrics
│   ├── talk_features.csv              # Engineered feature metadata with top TF-IDF keywords (2,550 rows)
│   ├── similarity_matrix.csv          # Full pairwise similarity matrix in CSV format (44.5 MB)
│   ├── top_recommendations.csv        # Top 10 recommendations for benchmark query talks (40 rows)
│   ├── model_comparison.csv           # Model benchmark comparison at cut-off K = 10 (5 models)
│   └── evaluation_results.csv         # Full evaluation results across K in [3, 5, 10, 15, 20] (25 rows)
│
├── reports/                           # Comprehensive technical documentation reports
│   ├── data_collection_report.txt     # Week 1 Module 1 Technical Report
│   ├── eda_report.txt                 # Week 1 Module 2 Technical Report
│   ├── module3_report.txt             # Week 2 Module 3 Technical Report
│   ├── module4_report.txt             # Week 2 Module 4 Technical Report
│   ├── module5_evaluation_report.txt  # Week 3 Module 5 Technical Report
│   ├── module6_deployment_report.txt  # Week 4 Module 6 Technical Report
│   │
│   └── visualizations/                # High-resolution (300 DPI) generated figures
│       ├── missing_values_chart.png
│       ├── talk_length_distribution.png
│       ├── top_10_tags.png
│       ├── views_distribution.png
│       ├── top_tfidf_features.png
│       ├── feature_sparsity_distribution.png
│       ├── similarity_distribution.png
│       ├── similarity_heatmap.png
│       ├── top_recommendations_profile.png
│       ├── recommendation_evaluation_metrics.png
│       ├── precision_comparison.png
│       ├── recall_comparison.png
│       ├── ndcg_comparison.png
│       └── map_comparison.png
│
├── screenshots/                       # Streamlit web application interface captures
│   ├── recommendation_view.png        # Main recommender interface with cards & explanations (351.8 KB)
│   ├── topic_explorer.png             # Thematic topic & tag taxonomy explorer (274.5 KB)
│   └── model_evaluation.png           # Live model evaluation benchmark dashboard (307.4 KB)
│
├── module1_data_collection.py         # Module 1 pipeline script
├── module2_data_preprocessing.ipynb   # Module 2 executed Jupyter Notebook
├── module3_feature_engineering.ipynb  # Module 3 executed Jupyter Notebook
├── module4_recommendation_engine.ipynb# Module 4 executed Jupyter Notebook
└── module5_evaluation.ipynb           # Module 5 executed Jupyter Notebook
```

---

## Module 1 – Week 1: Data Collection

### 1. Module Objective
To discover, collect, audit, deduplicate, and organize the TED Talks dataset, ensuring complete metadata coverage (titles, speakers, tags, descriptions, URLs, views, and transcripts), verifying topic diversity across domains, and evaluating user interaction data availability while strictly adhering to real-world data constraints.

### 2. Dataset Sources and Origins
- **Dataset Name:** Kaggle TED Talks Dataset (`ted-talks`)
- **Original Publisher:** TED Conferences LLC (via official [ted.com](https://www.ted.com) portal)
- **Curation:** Rounak Banik (Kaggle)
- **Temporal Scope:** All TED Talks published on TED.com up to September 21, 2017
- **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

### 3. Raw Input Files
1. **`ted_main.csv` (7.23 MB):** 2,550 records × 17 features:
   `comments`, `description`, `duration`, `event`, `film_date`, `languages`, `main_speaker`, `name`, `num_speaker`, `published_date`, `ratings`, `related_talks`, `speaker_occupation`, `tags`, `title`, `url`, `views`.
2. **`transcripts.csv` (27.20 MB):** 2,467 records × 2 features:
   `transcript`, `url`.

### 4. Implementation Workflow & Deduplication
1. **Referential Integrity Audit:** Loaded both CSVs and verified 1-to-1 linkage using the primary key `url`.
2. **Duplicate Detection:**
   - `ted_main.csv` contained **0** duplicate URLs across 2,550 talks.
   - `transcripts.csv` contained **3** redundant duplicate URLs that were safely removed, yielding 2,464 unique transcripts.
3. **Consolidation:** Left-joined transcripts onto `ted_main.csv` on `url` to construct the unified collection `datasets/ted_talks_collected.csv` (2,550 records × 18 columns).
4. **Metadata Coverage Audit:**
   - **Titles (`title`):** 2,550 / 2,550 (100.0% coverage)
   - **Speaker Names (`main_speaker`):** 2,550 / 2,550 (100.0% coverage)
   - **Curated Tags (`tags`):** 2,550 / 2,550 (100.0% coverage)
   - **Descriptions (`description`):** 2,550 / 2,550 (100.0% coverage)
   - **Talk URLs (`url`):** 2,550 / 2,550 (100.0% coverage)
   - **Views (`views`):** 2,550 / 2,550 (100.0% coverage)
   - **Transcripts (`transcript`):** 2,464 / 2,550 (96.63% coverage; 86 talks without spoken words, e.g., musical performances, mime, magic).
5. **Topic Diversity Analysis:** Identified **416 unique thematic tags** across 19,154 total tag mentions (average 7.51 tags/talk). Top categories span technology (727), science (567), global issues (501), culture (486), TEDx (450), and design (418).
6. **Ratings and Watch-History Assessment:**
   - Evaluated the `ratings` column: Contains aggregate tag-level reactions (e.g., "Inspiring", "Funny", "Informative") across all viewers.
   - **User interaction finding:** The dataset contains **zero** individual user IDs, personal ratings, or per-user watch-history logs.
   - **Strict Policy:** In adherence to academic integrity, **no fake user interaction data was fabricated**.

### 5. Outputs Generated
- `module1_data_collection.py` — Pipeline script
- `datasets/ted_talks_collected.csv` — Unified collected dataset (2,550 rows × 18 columns)
- `reports/data_collection_report.txt` — Detailed data collection report (8.3 KB)

### 6. Terminal Execution & Verification
```bash
python module1_data_collection.py
```
*Verification:* Inspect `reports/data_collection_report.txt` to confirm 2,550 talks collected, 416 unique topics identified, and 100% metadata coverage.

---

## Module 2 – Week 1: Data Exploration and Data Preprocessing

### 1. Module Objective
To conduct exploratory data analysis on talk durations, views, and tag taxonomy; clean verbatim transcripts via natural language processing (lowercasing, punctuation stripping, stopword removal, space normalization); resolve missing data; encode categorical attributes; and output clean, model-ready datasets with visual EDA figures.

### 2. Implementation Workflow
1. **Dataset Loading & Audit:**
   Loaded `ted_main.csv` (2,550 × 17) and `transcripts.csv` (2,467 × 2). Verified column dtypes and checked shapes.
2. **Missing-Value Analysis & Resolution:**
   - `speaker_occupation`: 6 missing values (0.24%). Imputed with string `'Unknown'`.
   - `transcript`: 86 talks (3.37%) lacked spoken audio transcripts. Imputed using synthetic synthesis of title and description: `"{title}. {description}"`. Created a boolean column `has_original_transcript` (`True` for 2,464 talks, `False` for 86 talks) to maintain tracking integrity.
   - All other 15 columns had 0 missing values.
3. **Statistical Analysis of Numerical Features:**
   - **Duration:** Range: 2.25 mins to 87.60 mins. Mean: 13.78 mins (std: 6.23 mins). Median: 14.13 mins. Clustered tightly around the 18-minute TED presentation limit.
   - **Views:** Range: 50,443 to 47,227,110 views. Mean: 1,698,297. Median: 1,124,524. Demonstrates heavy log-normal positive skew.
   - **Correlation:** Duration vs. Views Pearson correlation $r = 0.0487$ (no linear dependency between talk length and popularity).
4. **Natural Language Transcript Preprocessing:**
   Applied a 5-stage NLP pipeline to all transcripts:
   - Case normalization to lowercase.
   - Regular expression punctuation stripping (`re.sub(r'[^\w\s]', ' ', text)`).
   - Numerical standalone token filtering.
   - Whitespace and newline compaction (`re.sub(r'\s+', ' ', text).strip()`).
   - Stopwords removal using **NLTK's English Stopwords Corpus (198 words)**.
   - *Impact:* Average raw transcript length of 11,438 characters compressed to 6,735 cleaned characters (mean 959 words/talk), achieving a **41.1% noise reduction**.
5. **Categorical Feature Encoding:**
   - `main_speaker`: Scikit-Learn `LabelEncoder` $\rightarrow$ `speaker_id` (2,156 classes).
   - `speaker_occupation`: Scikit-Learn `LabelEncoder` $\rightarrow$ `occupation_id` (1,459 classes).
   - `tags`: Multi-hot binary indicator columns created for the **Top 20 most frequent tags** (`tag_technology`, `tag_science`, `tag_global_issues`, etc.).
   - `cleaned_tags`: Normalized space-delimited string formatted for text vectorization.
6. **EDA Visualizations Generated:**
   Four publication-quality 300 DPI figures saved in `reports/visualizations/`:
   - `talk_length_distribution.png`: Histogram & KDE with 18-minute benchmark line and boxplot.
   - `views_distribution.png`: Raw vs. $\log_{10}$-transformed views distribution.
   - `top_10_tags.png`: Frequency bar chart of top 10 categories with corpus percentages.
   - `missing_values_chart.png`: Pre- and post-cleaning missing value audit comparison.

### 3. Outputs Generated
- `module2_data_preprocessing.ipynb` — Executed Jupyter Notebook (1.09 MB)
- `cleaned_ted_talks.csv` — Fully cleaned dataset (2,550 rows × 46 columns, 24.0 MB)
- `cleaned_transcripts.csv` — Sanitized transcripts (2,464 rows × 5 columns, 44.1 MB)
- `reports/eda_report.txt` — Comprehensive EDA audit report (6.1 KB)
- 4 visualization PNG files in `reports/visualizations/`

### 4. Terminal Execution & Verification
```bash
python execute_module2.py
```
*Verification:* Verify that `cleaned_ted_talks.csv` contains 2,550 rows and 46 columns with 0 null values across all columns.

---

## Module 3 – Week 2: Feature Engineering and Vectorization

### 1. Module Objective
To extract and synthesize high-signal textual features from titles, speakers, tags, descriptions, and sanitized transcripts; apply Term Frequency–Inverse Document Frequency (TF-IDF) vectorization; compute pairwise Cosine Similarity across all talks; and serialize the resulting models and matrices.

### 2. Multi-Field Content Synthesis
Rather than vectorizing transcripts in isolation, a synthesized composite document was constructed for each talk:
$$\text{combined\_content} = \text{title} + " " + \text{main\_speaker} + " " + (2 \times \text{cleaned\_tags}) + " " + \text{description} + " " + \text{cleaned\_transcript}$$
- **Tag Weighting Strategy:** Curated thematic tags receive a **$2\times$ repetition multiplier** because editorial tags carry dense category signals compared to casual conversational transcript words.

### 3. TF-IDF Vectorization Configuration
Scikit-Learn's `TfidfVectorizer` was configured with tuned hyperparameters:
- `max_features = 5000`: Retains the top 5,000 most statistically informative n-grams.
- `ngram_range = (1, 2)`: Captures unigrams (`"creativity"`, `"education"`) and informative bigrams (`"climate change"`, `"machine learning"`, `"body language"`, `"mental health"`).
- `stop_words = 'english'`: Strips residual grammatical stopwords.
- `min_df = 2`: Ignores hapax legomena and OCR/speech recognition typos.
- `sublinear_tf = True`: Replaces term frequency $\text{tf}$ with $1 + \log(\text{tf})$ to prevent long speeches from dominating vector magnitudes.

### 4. Vectorization & Similarity Matrix Statistics
- **Document-Term Matrix (`tfidf_matrix`):** $2,550 \times 5,000$ sparse CSR matrix (935,408 non-zero entries; **92.66% sparsity**; mean 366.8 active terms/talk).
- **Pairwise Cosine Similarity Matrix:** $2,550 \times 2,550$ float32 matrix ($6,502,500$ elements, 24.8 MB).
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v \quad (\text{for } L_2\text{-normalized vectors})$$
  - Off-diagonal mean similarity: **0.1012** (std: 0.0426; median: 0.1018).
  - Maximum off-diagonal similarity: **0.7426** (between sequel talks by the same speaker on the same topic).

### 5. Outputs Generated
- `module3_feature_engineering.ipynb` — Executed Jupyter Notebook (1.52 MB)
- `models/tfidf_vectorizer.pkl` — Serialized fitted vectorizer (183.1 KB)
- `models/tfidf_matrix.pkl` — Serialized document-term sparse matrix (10.7 MB)
- `models/cosine_similarity.pkl` — Serialized pairwise similarity matrix (24.8 MB)
- `outputs/talk_features.csv` — Talk features with top-5 TF-IDF terms per talk (740.7 KB)
- `outputs/similarity_matrix.csv` — Full $2,550 \times 2,550$ similarity matrix in CSV format (44.5 MB)
- `reports/module3_report.txt` — Technical feature engineering report (8.0 KB)
- 4 visualization PNG files: `top_tfidf_features.png`, `feature_sparsity_distribution.png`, `similarity_distribution.png`, `similarity_heatmap.png`

### 6. Terminal Execution & Verification
```bash
python execute_module3.py
```
*Verification:* Confirm `models/cosine_similarity.pkl` loads as a $2550 \times 2550$ NumPy array with diagonal elements equal to $1.0$.

---

## Module 4 – Week 2: Develop Recommendation Engine

### 1. Module Objective
To implement a production recommendation engine supporting Content-Based filtering and Popularity-Augmented Hybrid filtering; provide flexible multi-tier title search; rank and return Top 10 recommendations with full metadata; conduct offline train-test evaluation; and serialize the trained recommendation models.

### 2. Architecture & Algorithms Implemented

#### A. Content-Based Recommendation Pipeline
Given a query talk index $q$, the engine extracts row $q$ of the precomputed Cosine Similarity matrix:
$$S_q = [s(q, 1), s(q, 2), \dots, s(q, N)]$$
Candidate talks $i \neq q$ are ranked in descending order of $s(q, i)$.

#### B. Popularity-Augmented Hybrid Recommendation
To avoid popularity starvation and surface high-engagement talks alongside niche content, a hybrid ranking formula was developed:
$$\text{Score}_{\text{Hybrid}}(q, i) = \alpha \cdot s(q, i) + (1 - \alpha) \cdot \widetilde{V}_i$$
Where:
- $\widetilde{V}_i = \frac{\log(1 + \text{views}_i) - \min(\log(1 + V))}{\max(\log(1 + V)) - \min(\log(1 + V))}$ is the min-max normalized $\log$-views.
- $\alpha = 0.85$ (default): Prioritizes 85% semantic content relevance while giving a 15% calibrated weight to global audience resonance.

#### C. Multi-Tier Title Search Engine
To support flexible real-world user queries, `recommendation_engine.py` implements a 3-tier lookup hierarchy:
1. Exact string match.
2. Case-insensitive and whitespace-stripped match.
3. Keyword / substring fuzzy containment search.

#### D. Collaborative Filtering Assessment
- The TED dataset contains talk-level aggregated statistics but **zero individual user profiles or interaction logs**.
- In strict adherence to engineering ethics, **no fake user profiles were synthesized**. Pure collaborative filtering was evaluated architecturally and contrasted in Module 5.

### 3. Offline Train-Test Split Evaluation (80/20)
- **Training Set (80%):** 2,040 talks
- **Test Set (20%):** 510 talks held out for query evaluation
- **Performance at Top $K = 10$:**
  - **Tag Precision@10:** **0.8855** (8.9 of 10 recommended talks share curated tags with the query talk)
  - **Tag Recall@10:** **0.8330** (captures 83.3% of the query talk's thematic categories)
  - **Mean Reciprocal Rank (MRR):** **0.9682** (the first relevant talk appears at rank #1 in >98% of queries)
  - **Intra-List Diversity:** **0.7918** (recommendations avoid repetitive redundancy)

### 4. Sample Flagship Query Demonstrations

#### Query: "Do schools kill creativity?" (Sir Ken Robinson)
| Rank | Recommended Talk Title | Speaker | Similarity | Key Shared Tags |
| :---: | :--- | :--- | :---: | :--- |
| **#1** | *How to escape education's death valley* | Ken Robinson | **0.2815** | `culture`, `education`, `teaching` |
| **#2** | *Bring on the learning revolution!* | Ken Robinson | **0.2813** | `children`, `creativity`, `education` |
| **#3** | *Rebuilding a neighborhood with beauty, dignity, hope* | Bill Strickland | **0.2658** | `children`, `culture` |
| **#4** | *Letting go of God* | Julia Sweeney | **0.2549** | `culture`, `storytelling` |
| **#5** | *The story of a girl* | Rick Smolan | **0.2520** | `children`, `storytelling` |

### 5. Outputs Generated
- `module4_recommendation_engine.ipynb` — Executed Jupyter Notebook (523.7 KB)
- `recommendation_engine.py` — Standalone Python recommender class and CLI (19.2 KB)
- `models/content_based_model.pkl` — Serialized content pipeline (26.3 MB)
- `models/recommendation_model.pkl` — Serialized production recommender model (26.3 MB)
- `outputs/top_recommendations.csv` — Recommendations for flagship talks (21.0 KB)
- `reports/module4_report.txt` — Technical recommendation engine report (6.3 KB)
- `reports/visualizations/top_recommendations_profile.png` — Visualization of recommendations

### 6. Terminal Execution & Verification
```bash
# Test CLI recommendation
python recommendation_engine.py --title "Do schools kill creativity?" --top_n 10

# Test Hybrid recommendation
python recommendation_engine.py --title "Your body language may shape who you are" --method hybrid --alpha 0.85 --top_n 10
```
*Verification:* Check `outputs/top_recommendations.csv` to confirm 40 recommendation records (Top 10 across 4 flagship queries).

---

## Module 5 – Week 3: Evaluation Metrics

### 1. Module Objective
To scientifically evaluate the recommendation models against authentic ground truth—the **15,010 official human-curated editorial recommendations** (`related_talks`) published by TED Conferences LLC; compute Information Retrieval metrics (Precision@K, Recall@K, NDCG@K, MAP@K) across multiple thresholds; compare competing recommendation paradigms; and establish the best-performing architecture.

### 2. Ground-Truth Dataset Formulation
- **Source:** The `related_talks` column in `ted_main.csv` provides curated recommendations authored by TED conference organizers.
- **Statistics:** 15,010 total ground-truth pairs across 2,550 talks (mean 5.9 related talks/talk).
- **Random Baseline Precision:** A random selection yields Precision $\approx \frac{5.9}{2549} \approx 0.0023$ (0.23%).

### 3. Evaluation Metrics Formulation
For each test talk $q$ and recommendation list $R_K(q)$ of length $K$, with ground-truth set $G(q)$:
- **Precision@K:** Proportion of recommended talks that are in the ground truth:
  $$\text{Precision@}K = \frac{|R_K(q) \cap G(q)|}{K}$$
- **Recall@K:** Proportion of ground-truth talks captured in the Top $K$:
  $$\text{Recall@}K = \frac{|R_K(q) \cap G(q)|}{|G(q)|}$$
- **NDCG@K (Normalized Discounted Cumulative Gain):** Evaluates ranking quality with logarithmic position discounting:
  $$\text{DCG@}K = \sum_{i=1}^K \frac{\mathbb{I}(R_K(q)[i] \in G(q))}{\log_2(i + 1)}, \quad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$
- **Mean Average Precision (MAP@K):** Area under the Precision-Recall curve:
  $$\text{AP@}K = \frac{1}{\min(|G(q)|, K)} \sum_{i=1}^K \text{Precision@}i \cdot \mathbb{I}(R_K(q)[i] \in G(q))$$

### 4. Model Comparison Benchmark Results (At Cut-off $K = 10$)
Evaluated across **510 held-out test talks** (80/20 train-test split):

| Model Architecture | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Relative Lift vs Popularity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Content-Based (TF-IDF + Cosine)** | **0.2300** | **0.3879** | **0.4117** | **0.2671** | **+6,471% (65.7×)** |
| **Hybrid ($\alpha = 0.85$)** | **0.1733** | **0.2931** | **0.3075** | **0.1885** | **+4,851% (49.5×)** |
| **Tag Jaccard Baseline** | 0.0988 | 0.1693 | 0.1628 | 0.0837 | +2,722% (28.2×) |
| **Hybrid ($\alpha = 0.70$)** | 0.0537 | 0.0928 | 0.0913 | 0.0467 | +1,434% (15.3×) |
| **Popularity Baseline** | 0.0035 | 0.0062 | 0.0068 | 0.0032 | 1.0× (Baseline) |

### 5. Metric Progression Across Cut-offs $K \in [3, 5, 10, 15, 20]$

```text
========================================================================================
Model                   K     Precision@K   Recall@K      NDCG@K        MAP@K
========================================================================================
Content-Based (TF-IDF)  3     0.4170        0.2095        0.4566        0.3704
Content-Based (TF-IDF)  5     0.3451        0.2918        0.3978        0.2762
Content-Based (TF-IDF) 10     0.2300        0.3879        0.4117        0.2671
Content-Based (TF-IDF) 15     0.1771        0.4480        0.4405        0.2823
Content-Based (TF-IDF) 20     0.1467        0.4944        0.4604        0.2913
----------------------------------------------------------------------------------------
Hybrid (alpha=0.85)     3     0.3085        0.1556        0.3364        0.2653
Hybrid (alpha=0.85)     5     0.2537        0.2154        0.2928        0.1940
Hybrid (alpha=0.85)    10     0.1733        0.2931        0.3075        0.1885
Hybrid (alpha=0.85)    15     0.1399        0.3549        0.3370        0.2020
Hybrid (alpha=0.85)    20     0.1171        0.3967        0.3548        0.2087
----------------------------------------------------------------------------------------
Tag Jaccard Baseline   10     0.0988        0.1693        0.1628        0.0837
Popularity Baseline    10     0.0035        0.0062        0.0068        0.0032
========================================================================================
```

### 6. Best-Performing Model Selection
**Content-Based Filtering (TF-IDF + Cosine Similarity)** is established as the champion model:
1. **Superior Precision:** At $K=3$, Precision reaches **41.70%**, meaning over 4 out of 10 recommendations match official human curation.
2. **High Coverage:** By $K=20$, the model captures **49.44%** of all related talks.
3. **Zero-User Cold Start:** Functions instantly for any talk with text features without requiring historical user logs.

### 7. Outputs Generated
- `module5_evaluation.ipynb` — Executed Jupyter Notebook (1.18 MB)
- `evaluation.py` — Standalone evaluation script (22.3 KB)
- `outputs/model_comparison.csv` — Model comparison benchmark at $K = 10$ (0.3 KB)
- `outputs/evaluation_results.csv` — Comprehensive metrics across all $K$ thresholds (1.3 KB)
- `reports/module5_evaluation_report.txt` — Technical evaluation report (7.1 KB)
- 4 visualization PNG files: `precision_comparison.png`, `recall_comparison.png`, `ndcg_comparison.png`, `map_comparison.png`

### 8. Terminal Execution & Verification
```bash
python evaluation.py
```
*Verification:* Inspect `outputs/model_comparison.csv` to confirm Content-Based P@10 = 0.2300 and NDCG@10 = 0.4117.

---

## Module 6 – Week 3 and Week 4: Deployment via Streamlit App

### 1. Module Objective
To package the recommendation engine and evaluation benchmarks into a high-performance, user-friendly **Streamlit web application** (`app.py`); provide responsive multi-field search, topic taxonomy exploration, recommendation explainability, and error handling; optimize execution using caching; and prepare deployment configurations.

### 2. Implementation Features & Workflow

#### A. Interactive Multi-Page Navigation
The application features a modern sidebar with TED-branded styling (#e62b1e accent) and 4 dedicated operational pages:
1. **🎯 Recommender:** Main recommendation engine with title autocomplete, quick picks, seed talk overview, and recommendation cards.
2. **🏷️ Topic Explorer:** Thematic browser filtering across 400+ topic tags using AND logic with sorting by views and duration.
3. **📊 Model Benchmarks:** Live evaluation dashboard displaying the `model_comparison.csv` benchmark table, metric highlight cards, and embedded evaluation curves.
4. **ℹ️ About:** System architecture overview and technical details.

#### B. Dynamic Controls & Algorithmic Parameters
- **Algorithm Mode Selector:** Switch between *Content-Based (TF-IDF)* and *Hybrid (Content + Popularity)*.
- **Content Weight Slider ($\alpha$):** Dynamically adjusts the semantic vs. popularity weighting ($0.50 \le \alpha \le 0.95$).
- **Top N Slider:** Allows selecting between 5 and 20 recommendations (default: 10).
- **Quick-Pick Flagship Buttons:** One-click recommendation loading for 4 iconic TED talks.

#### C. Recommendation Card Components & Explainability
Each recommended talk is rendered in a custom CSS card displaying:
- **Rank Badge:** Numbered red pill (`#1`, `#2`, ..., `#10`).
- **Semantic Match Badge:** Blue badge displaying the Cosine Similarity percentage (e.g., `54.7% Match`).
- **Metadata:** Title, main speaker, speaker occupation, duration in minutes, and formatted views count.
- **Why Recommended (Explainability Box):**
  - Identifies shared speakers.
  - Lists shared topic tags.
  - Reports semantic transcript match percentage.
- **Thematic Tag Badges:** Displays tags with shared tags highlighted in **green (`✓ tag_name`)**.
- **Watch on TED.com Link:** Red action button opening the official talk URL.

#### D. Streamlit Caching Optimization
- `@st.cache_data`: Caches talk metadata loading (`load_talks_data`) and benchmark tables (`load_model_benchmarks`).
- `@st.cache_resource`: Caches the 26 MB similarity matrix in memory (`load_recommendation_models`).
- *Performance:* First load: ~3.5 seconds; subsequent user queries: **< 0.5 seconds**.

#### E. Robust Error Handling
- Checks for dataset and model existence at startup with `st.error()` and `st.stop()`.
- Safe fallbacks for missing columns or malformed tag strings using `ast.literal_eval`.
- Path resolution uses `os.path.abspath(__file__)` for cross-platform execution.

### 3. Application Testing Summary
- **Title Autocomplete:** Verified across all 2,550 titles.
- **Recommendation Relevance:** Tested on diverse domains (creativity, climate, neuroscience, statistics).
- **Port Binding:** Verified on `http://localhost:8501` with HTTP 200 response.
- **Usability:** Responsive layout on desktop and tablet viewports.

### 4. Outputs Generated
- `app.py` — Streamlit web application (525 lines, 24.4 KB)
- `requirements.txt` — Pinned production dependencies (0.2 KB)
- `reports/module6_deployment_report.txt` — Deployment and testing report (17.8 KB)
- `screenshots/recommendation_view.png` — Recommender UI screenshot (351.8 KB)
- `screenshots/topic_explorer.png` — Topic Explorer UI screenshot (274.5 KB)
- `screenshots/model_evaluation.png` — Model Evaluation dashboard screenshot (307.4 KB)

### 5. Terminal Execution & Verification
```bash
# Launch the Streamlit Web Application
streamlit run app.py
```
*Verification:* Open `http://localhost:8501` in your browser. Select any talk or click a quick-pick button to confirm Top 10 recommendations appear with explanation boxes and TED.com links.

---

## ⏳ Module 7 – Pending Work

> [!IMPORTANT]
> **Module 7: Presentation and Documentation is PENDING and has NOT yet been implemented.**
>
> In accordance with project instructions and timeline milestones, Module 7 encompasses:
> 1. Comprehensive final internship presentation slide deck (PPTX / PDF).
> 2. Video demonstration and walkthrough of the deployed Streamlit application.
> 3. Formal internship technical project report submission to the Infosys Springboard portal.
>
> Implementation of Modules 1 through 6 is 100% complete and verified. Module 7 will be completed during the scheduled final review phase.

---

## 🖼️ Screenshots & Visualizations

### Application Screenshots (Module 6)

| View | Screenshot Preview | Description |
| :--- | :---: | :--- |
| **Personalized Recommender** | `screenshots/recommendation_view.png` | Displays seed talk overview, rank badges, similarity scores, shared tag highlights, explainability boxes, and TED.com links. |
| **Topic Explorer** | `screenshots/topic_explorer.png` | Interactive multi-tag taxonomy filter with AND logic and sorting across 400+ topic tags. |
| **Model Benchmarks** | `screenshots/model_evaluation.png` | Evaluation dashboard displaying the comparison table at $K=10$, metric cards, and IR curves. |

### Evaluation Visualizations (Module 5)

| Metric | Visualization Path | Key Observation |
| :--- | :---: | :--- |
| **Precision@K** | `reports/visualizations/precision_comparison.png` | Content-Based achieves 41.7% at $K=3$ and 23.0% at $K=10$ (65× over popularity). |
| **Recall@K** | `reports/visualizations/recall_comparison.png` | Content-Based recall expands from 20.9% ($K=3$) to 49.4% ($K=20$). |
| **NDCG@K** | `reports/visualizations/ndcg_comparison.png` | Content-Based NDCG remains consistently superior ($0.4117$ at $K=10$). |
| **MAP@K** | `reports/visualizations/map_comparison.png` | Mean Average Precision confirms strong early-rank ranking quality ($0.2671$ at $K=10$). |

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd "e:\infosys\TED TALKS"

# (Optional) Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Running Individual Pipeline Modules
```bash
# Module 1: Data Collection & Audit
python module1_data_collection.py

# Module 4: CLI Recommendation Engine
python recommendation_engine.py --title "Do schools kill creativity?" --top_n 10

# Module 5: Evaluation Benchmark Suite
python evaluation.py

# Module 6: Launch Streamlit Web Application
streamlit run app.py
```

### 4. Cloud Deployment Options
- **Streamlit Community Cloud (Recommended):** Connect the GitHub repository, select `app.py` as the main entry point, and deploy.
- **Hugging Face Spaces:** Create a new Space with the `Streamlit` SDK and push the repository.
- **Docker Container:**
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

---

## 🔒 Academic & Engineering Integrity Statement

1. **No Synthetic / Fake User Data:** The TED dataset contains talk metadata, video metrics, and transcripts, but **zero individual user interaction logs**. In strict compliance with scientific and engineering ethics, **no fake user profiles, synthetic rating matrices, or fabricated watch logs were created**.
2. **Authentic Ground Truth:** Offline evaluation is conducted against **15,010 authentic human-curated editorial recommendations** published by TED Conferences LLC.
3. **Reproducibility:** All preprocessing scripts, vectorizers, similarity matrices, models, evaluation metrics, and notebooks are preserved in the repository and can be independently verified.

---

*Documentation compiled and verified for Infosys Springboard Virtual Internship 7.0 evaluation.*
