# MR - Movie Analytics Dashboard (Cinematics Dashboard)

## 1. Project Overview

The **MR - Movie Analytics Dashboard (Cinematics Dashboard)** is a comprehensive web-based data analytics platform that provides deep insights into movie financials, audience sentiment, statistical distributions, and machine learning clustering. It leverages a dual-dataset approach combining **TMDB Movies** (financial & metadata) and **IMDB Reviews** (sentiment & textual analysis) to deliver an interactive, production-grade dashboard.

---

## 2. Technology Stack

| Component                | Technology                                                           |
| ------------------------ | -------------------------------------------------------------------- |
| **Framework**            | Streamlit (Python Web Framework)                                     |
| **Data Processing**      | Pandas, NumPy                                                        |
| **Visualization**        | Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn, WordCloud |
| **Machine Learning**     | Scikit-learn (RandomForest, KMeans, PCA, StandardScaler)             |
| **Statistical Analysis** | SciPy (stats module)                                                 |
| **Data Storage**         | Flat CSV Files (Pre-cleaned & Raw)                                   |

---

## 3. System Architecture & File Structure

### Directory Layout

```
project/
├── app.py                 (~450 lines) Core Streamlit application: UI, data processing, ML, stats
├── styles.css             Custom CSS for metric cards & UI components
├── movies_cleaned.csv     (~45,000+ rows) Pre-cleaned TMDB dataset
├── archive/
│   ├── (6)/movies.csv     (~45,000+ rows) Raw TMDB fallback dataset
│   └── (4)/IMDB Dataset.csv (~50,000 rows) IMDB reviews with binary sentiment
└── [Generated Plots].png  EDA visualizations saved during pipeline execution
```

### Architecture Layers

1. **Presentation**: Streamlit web UI (6 interactive tabs, sidebar filters, responsive layout)
2. **Application Logic**: Data loaders, feature engineering, caching, ML training, statistical testing
3. **Visualization Engine**: Plotly (interactive), Matplotlib/Seaborn (static/statistical), WordCloud (NLP)
4. **Data Layer**: CSV-based flat file storage with fallback loading logic

---

## 4. Data Sources & Schema

### 4.1 TMDB Movies Dataset (`movies_cleaned.csv`)

| Column                           | Type          | Description                                                |
| -------------------------------- | ------------- | ---------------------------------------------------------- |
| `title`                          | string        | Movie title                                                |
| `budget`                         | float         | Production budget (USD, 0s imputed with median)            |
| `revenue`                        | float         | Box office revenue (USD, 0s imputed with median)           |
| `rating` / `vote_average`        | float         | Average user rating (0-10)                                 |
| `vote_count`                     | int           | Number of votes received                                   |
| `runtime`                        | float         | Duration in minutes                                        |
| `release_date`                   | datetime      | Original release date                                      |
| `genre_list`                     | object (list) | Engineered: List of all associated genres                  |
| `primary_genre`                  | string        | Engineered: First extracted genre                          |
| `release_year` / `release_month` | int           | Engineered: Date components                                |
| `decade`                         | int           | Engineered: Release decade (e.g., 1980)                    |
| `roi`                            | float         | Engineered: Return on Investment `(revenue-budget)/budget` |
| `is_profitable`                  | int           | Engineered: Binary profitability flag (1/0)                |
| `runtime_category`               | category      | Engineered: Binned runtime (Short/Medium/Long/Very Long)   |
| `release_season`                 | string        | Engineered: Release season (Winter/Spring/Summer/Fall)     |
| `log_budget` / `log_revenue`     | float         | Engineered: Log-transformed financials                     |
| `genre_*`                        | uint8         | Engineered: One-hot encoded genre columns                  |

### 4.2 IMDB Reviews Dataset (`archive(4)/IMDB Dataset.csv`)

| Column      | Type   | Description                                     |
| ----------- | ------ | ----------------------------------------------- |
| `review`    | string | Full text of the movie review                   |
| `sentiment` | string | Binary classification: `positive` or `negative` |

---

## 5. Core Functions & Logic

### 5.1 `load_css(file_name: str) -> None`

- **Purpose**: Injects external CSS into Streamlit via `st.markdown()` with `unsafe_allow_html=True`
- **Fallback**: Provides inline metric-card styling if file is missing

### 5.2 `load_tmdb_data() -> pd.DataFrame`

- **Purpose**: Loads, cleans, and feature-engineers the TMDB dataset
- **Key Operations**:
  - Fallback loading (`movies_cleaned.csv` → `archive(6)/movies.csv`)
  - Budget/Revenue cleaning: `pd.to_numeric(errors='coerce')`, median imputation for 0-values
  - Genre parsing via `ast.literal_eval()`
  - Date extraction, ROI calculation, decade/season mapping, runtime binning
  - Profitability flag generation
- **Caching**: `@st.cache_data` (commented in dev, recommended for production)

### 5.3 `load_imdb_data() -> pd.DataFrame`

- **Purpose**: Loads IMDB reviews for sentiment analysis
- **Caching**: `@st.cache_data`
- **Structure**: Simple load without additional cleaning (pre-validated)

### 5.4 `train_prediction_model(df: pd.DataFrame) -> tuple(model, mae, feature_cols)`

- **Purpose**: Trains a Random Forest Regressor to predict movie ratings
- **Pipeline**:
  1. Drop missing values in critical columns
  2. One-hot encode `primary_genre`
  3. Select features: `budget`, `runtime`, `release_year`, + encoded genres
  4. 80/20 train-test split (`random_state=42`)
  5. Train `RandomForestRegressor(n_estimators=100)`
  6. Evaluate with `mean_absolute_error`
- **Caching**: `@st.cache_resource` (persists trained model across reruns)

### 5.5 Nested Helpers

- `extract_genre(x)`: Safely parses JSON-like genre strings, returns first genre or `"Unknown"`
- `get_season(month: int)`: Maps month number (1-12) to meteorological season

---

## 6. Application Modules (Tabs)

| Tab                         | Purpose               | Key Components                                                                                                                                                                |
| --------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Movie Financials**     | Core analytics        | Sidebar filters (year range, genres), scatter plot (budget vs revenue, log scale), histogram (rating distribution), metric cards (selected movies, avg rating/budget/revenue) |
| **2. Statistical Insights** | Hypothesis testing    | T-tests (Action vs Drama ratings, Summer vs non-Summer revenue), Chi-square (Genre vs Profitability), ANOVA (Rating differences across top genres)                            |
| **3. Advanced ML**          | Unsupervised learning | Correlation heatmap, PCA scree plot & 3D visualization, K-Means clustering (elbow method), genre similarity analysis                                                          |
| **4. Prediction**           | Supervised ML         | Random Forest training display, feature importance chart, user input form (budget/runtime/year/genre), gauge chart for predicted rating                                       |
| **5. Sentiment Analysis**   | NLP & Text Mining     | Pie chart (positive/negative ratio), WordCloud generation, random review explorer                                                                                             |
| **6. Top Charts**           | Rankings              | Top rated movies (filtered `vote_count > 500`), highest grossing films, genre market share                                                                                    |

---

## 7. Machine Learning & Statistical Analysis

### 7.1 Machine Learning Models

| Model                   | Purpose                    | Library                     |
| ----------------------- | -------------------------- | --------------------------- |
| `RandomForestRegressor` | Rating prediction          | `sklearn.ensemble`          |
| `KMeans`                | Movie clustering           | `sklearn.cluster`           |
| `PCA`                   | Dimensionality reduction   | `sklearn.decomposition`     |
| `Lasso`                 | Embedded feature selection | `sklearn.linear_model`      |
| `RFE`                   | Wrapper feature selection  | `sklearn.feature_selection` |

### 7.2 Statistical Tests

| Test               | Variables                          | Purpose                              |
| ------------------ | ---------------------------------- | ------------------------------------ |
| Independent T-Test | Action vs Drama `rating`           | Compare mean ratings                 |
| One-Tailed T-Test  | Summer vs Other `log_revenue`      | Test seasonal revenue impact         |
| Chi-Square         | `primary_genre` vs `is_profitable` | Test genre-profitability association |
| ANOVA              | Top 5 genres vs `rating`           | Multi-group rating difference        |

### 7.3 Feature Engineering Pipeline

- **Financial**: `ROI`, `log_budget`, `log_revenue`, `is_profitable`
- **Temporal**: `release_year`, `release_month`, `decade`, `release_season`
- **Categorical**: `primary_genre`, `runtime_category`, one-hot encoded genres
- **Scaling**: `StandardScaler` applied before PCA, KMeans, and Lasso

---

## 8. UI/UX & Custom Styling

- **Framework**: Streamlit (`layout="wide"`)
- **Custom CSS**: `styles.css` injects premium UI styling
  - Metric cards with hover lift effects
  - Gradient title styling
  - Custom tab styling (rounded corners, active states)
  - Chart container cards with shadows
- **Font**: Google Fonts `Inter` imported for modern UI typography
- **Responsive Design**: Multi-column layouts (`st.columns()`), sidebar controls, collapsible expanders



---

## 2. EXECUTIVE SUMMARY

This project, **"Cinematics Analytics Suite"**, represents a complete end-to-end data science pipeline designed to ingest raw movie metadata, perform rigorous data cleaning, execute advanced statistical analysis, and deploy machine learning models for rating prediction.

The system is built upon a dual-dataset architecture combining financial metrics from **TMDB** and sentiment data from **IMDB**. The primary objective is to uncover the hidden correlations between production budgets, runtime, genres, and audience reception, culminating in an interactive dashboard for stakeholders.

The project is realised as a **Streamlit web application** that offers a rich, multi-tab analytic dashboard, backed by a standalone analysis script that produces cleaned data and exploratory plots.

---

## 3. PROJECT INFO TABLE

| Property                | Details                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| Project Name            | MR – Movie Analytics Dashboard (Cinematics Dashboard)                                      |
| Type                    | Web Application – Data Analytics Dashboard                                                 |
| Technologies            | Python 3.x, Streamlit, Pandas, Plotly, Scikit-learn, SciPy, Seaborn, Matplotlib, WordCloud |
| Primary Framework       | Streamlit (Python Web Framework)                                                           |
| Data Sources            | TMDB Movies Dataset, IMDB Reviews Dataset                                                  |
| Visualization Libraries | Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn, WordCloud                       |
| ML Libraries            | Scikit-learn (RandomForest, KMeans, PCA, StandardScaler)                                   |
| Statistical Library     | SciPy (stats module)                                                                       |

---

## 4. FILE STRUCTURE

### 4.1 Dashboard Application Files

| #   | File Path                      | Lines         | Purpose                                              |
| --- | ------------------------------ | ------------- | ---------------------------------------------------- |
| 1   | `app.py`                       | ~450          | Streamlit application containing all dashboard logic |
| 2   | `styles.css`                   | ~70           | Custom CSS styling for the dashboard UI              |
| 3   | `movies_cleaned.csv`           | ~45,000+ rows | Pre-processed TMDB movie dataset                     |
| 4   | `archive (6)/movies.csv`       | ~45,000+ rows | Raw TMDB movie dataset (fallback)                    |
| 5   | `archive (4)/IMDB Dataset.csv` | ~50,000 rows  | IMDB reviews with binary sentiment classification    |

### 4.2 Analysis Script Files

| #   | File Name                                              | Format | Role   | Description                                                                           |
| --- | ------------------------------------------------------ | ------ | ------ | ------------------------------------------------------------------------------------- |
| 1   | `archive (6)/movies.csv`                               | CSV    | Input  | Raw movie data from TMDB                                                              |
| 2   | `movies_cleaned.csv`                                   | CSV    | Output | Final cleaned dataset ready for ML/visualisation                                      |
| 3   | `run_analysis_fixed.py` / `movie_analysis_pipeline.py` | Python | Core   | End-to-end data analysis pipeline                                                     |
| 4   | `[plot_name].png`                                      | Image  | Output | Generated visualisations (e.g., `rating_distribution.png`, `correlation_heatmap.png`) |

---

## 5. TECHNOLOGY STACK (Detailed Imports)

### 5.1 Dashboard (`app.py`) Imports

| #   | Import                        | Source                  | Purpose                                                       |
| --- | ----------------------------- | ----------------------- | ------------------------------------------------------------- |
| 1   | `streamlit as st`             | streamlit               | Building interactive dashboard UI                             |
| 2   | `pandas as pd`                | pandas                  | Data manipulation, CSV reading, DataFrames                    |
| 3   | `seaborn as sns`              | seaborn                 | Statistical visualisation (used alongside Plotly)             |
| 4   | `matplotlib.pyplot as plt`    | matplotlib              | WordCloud and static plot generation                          |
| 5   | `plotly.express as px`        | plotly                  | High-level interactive charts (scatter, bar, histograms)      |
| 6   | `plotly.figure_factory as ff` | plotly                  | Specialised Plotly figures (e.g., distplots)                  |
| 7   | `plotly.graph_objects as go`  | plotly                  | Custom Gauge charts, indicators                               |
| 8   | `WordCloud`                   | wordcloud               | Generating word cloud visualisations                          |
| 9   | `ast`                         | Python Standard Library | Parsing JSON-like genre strings                               |
| 10  | `scipy.stats`                 | scipy                   | Hypothesis testing (t-tests, chi-square, normal distribution) |
| 11  | `StandardScaler`              | sklearn.preprocessing   | Feature scaling for ML                                        |
| 12  | `PCA`                         | sklearn.decomposition   | Dimensionality reduction                                      |
| 13  | `KMeans`                      | sklearn.cluster         | K-Means clustering                                            |
| 14  | `RandomForestRegressor`       | sklearn.ensemble        | Rating prediction model                                       |
| 15  | `train_test_split`            | sklearn.model_selection | Splitting data for training/testing                           |
| 16  | `mean_absolute_error`         | sklearn.metrics         | Model evaluation                                              |
| 17  | `numpy as np`                 | numpy                   | Numerical operations, array manipulation                      |

### 5.2 Analysis Script Imports

| #   | Import                     | Source                    | Purpose                                      |
| --- | -------------------------- | ------------------------- | -------------------------------------------- |
| 1   | `matplotlib`               | matplotlib                | Core plotting library                        |
| 2   | `matplotlib.pyplot as plt` | matplotlib                | Interface for creating figures               |
| 3   | `os`                       | Python Standard Library   | Changing working directory                   |
| 4   | `pandas as pd`             | pandas                    | DataFrames and data processing               |
| 5   | `numpy as np`              | numpy                     | Numerical computing                          |
| 6   | `seaborn as sns`           | seaborn                   | Statistical visualisations                   |
| 7   | `ast`                      | Python Standard Library   | Safe evaluation of string literals           |
| 8   | `datetime`                 | Python Standard Library   | Date/time manipulation                       |
| 9   | `scipy.stats`              | scipy                     | Statistical functions and hypothesis testing |
| 10  | `warnings`                 | Python Standard Library   | Suppressing warning messages                 |
| 11  | `LabelEncoder`             | sklearn.preprocessing     | Encoding categorical labels                  |
| 12  | `StandardScaler`           | sklearn.preprocessing     | Feature standardisation                      |
| 13  | `Lasso`                    | sklearn.linear_model      | Embedded feature selection                   |
| 14  | `RFE`                      | sklearn.feature_selection | Recursive feature elimination                |
| 15  | `LinearRegression`         | sklearn.linear_model      | Base estimator for RFE                       |
| 16  | `norm`                     | scipy.stats               | Normal distribution modelling                |
| 17  | `PCA`                      | sklearn.decomposition     | Dimensionality reduction                     |
| 18  | `KMeans`                   | sklearn.cluster           | Clustering                                   |
| 19  | `Axes3D`                   | mpl_toolkits.mplot3d      | 3D plotting                                  |
| 20  | `pdist, squareform`        | scipy.spatial.distance    | Pairwise distance computations               |

---

## 6. FUNCTIONS

### 6.1 Dashboard Functions (`app.py`)

| #        | Function Name            | Parameters         | Returns                           | Purpose                                              |
| -------- | ------------------------ | ------------------ | --------------------------------- | ---------------------------------------------------- |
| 1        | `load_css`               | `file_name: str`   | None                              | Loads and injects custom CSS into the Streamlit app  |
| 2        | `load_tmdb_data`         | None               | `pd.DataFrame`                    | Loads, cleans, and feature-engineers TMDB movie data |
| 3        | `load_imdb_data`         | None               | `pd.DataFrame`                    | Loads IMDB reviews dataset                           |
| 4        | `train_prediction_model` | `df: pd.DataFrame` | `tuple(model, mae, feature_cols)` | Trains Random Forest model for rating prediction     |
| (nested) | `extract_genre`          | `x: any`           | `str`                             | Extracts primary genre from JSON-like string         |
| (nested) | `get_season`             | `month: int`       | `str`                             | Converts month number to season name                 |

#### Function Details – `load_tmdb_data`

- **Returns:** Cleaned DataFrame with engineered features
- **Feature Engineering Operations:**
  - `primary_genre`: Extracted from JSON array
  - `release_year`, `release_month`: Extracted from `release_date`
  - `roi`: (revenue – budget) / budget
  - `decade`: (year // 10) * 10
  - `release_season`: Month to season mapping
  - `runtime_category`: Binned (Short/Medium/Long/Very Long)
  - `is_profitable`: Binary flag (revenue > budget)
- **Data Cleaning:**
  - Budget & revenue: 0‑values imputed with median
  - Rows missing `release_date` or `title` dropped
  - Duplicates removed
  - Genres parsed via `ast.literal_eval`

#### Function Details – `train_prediction_model`

- **Configuration:**
  - `n_estimators=100`, `random_state=42`
  - Test size: 20%
- **Returns:**
  - Trained `RandomForestRegressor`
  - Mean Absolute Error on test set
  - List of feature column names
- **Uses:** One‑hot encoding for genres, exclusion of non‑predictive columns.

### 6.2 Analysis Script Functions

| #   | Function Name       | Parameters        | Returns | Purpose                                               |
| --- | ------------------- | ----------------- | ------- | ----------------------------------------------------- |
| 1   | `display`           | `*args`           | None    | Print multiple arguments (mimics Jupyter `display`)   |
| 2   | `parse_json_column` | `x`, `key="name"` | `list`  | Safely parses JSON‑like string to list of genre names |
| 3   | `get_season`        | `month: int`      | `str`   | Converts month number to season name                  |

---

## 7. CLASSES

Both the dashboard and the analysis script use a **procedural programming approach** and do not define custom Python classes. They extensively use classes from imported libraries:

- `pandas.DataFrame`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.decomposition.PCA`
- `sklearn.cluster.KMeans`
- `sklearn.ensemble.RandomForestRegressor`

---

## 8. API ENDPOINTS

This project is a **Streamlit standalone web application** and a **command‑line analysis script**. It does not expose traditional REST API endpoints. The Streamlit app runs on a local server (default port `8501`).

---

## 9. DATABASE

The project uses **flat‑file storage (CSV files)**. No traditional database system (SQL/NoSQL) is employed.

### Data Sources

| File Name                      | Format | Records  | Description                        |
| ------------------------------ | ------ | -------- | ---------------------------------- |
| `movies_cleaned.csv`           | CSV    | ~45,000+ | Pre‑cleaned TMDB movie dataset     |
| `archive (6)/movies.csv`       | CSV    | ~45,000+ | Raw TMDB movie dataset (fallback)  |
| `archive (4)/IMDB Dataset.csv` | CSV    | ~50,000  | IMDB reviews with sentiment labels |

### TMDB Movies Dataset Schema (final cleaned output)

| #   | Column               | Type       | Description                                     |
| --- | -------------------- | ---------- | ----------------------------------------------- |
| 1   | `id`                 | int64      | Unique movie identifier                         |
| 2   | `title`              | object     | Movie title                                     |
| 3   | `budget`             | float64    | Production budget (0s imputed with median)      |
| 4   | `revenue`            | float64    | Worldwide revenue (0s imputed with median)      |
| 5   | `release_date`       | datetime64 | Release date                                    |
| 6   | `runtime`            | float64    | Duration in minutes                             |
| 7   | `rating`             | float64    | Average user rating (originally `vote_average`) |
| 8   | `vote_count`         | int64      | Number of votes                                 |
| 9   | `genre_list`         | object     | [Engineered] Python list of associated genres   |
| 10  | `primary_genre`      | object     | [Engineered] First genre from `genre_list`      |
| 11  | `release_year`       | int64      | [Engineered] Year of release                    |
| 12  | `release_month`      | int64      | [Engineered] Month of release                   |
| 13  | `decade`             | int64      | [Engineered] Decade (e.g., 1980, 1990)          |
| 14  | `roi`                | float64    | [Engineered] Return on Investment               |
| 15  | `is_profitable`      | int64      | [Engineered] 1 if revenue > budget, else 0      |
| 16  | `runtime_category`   | category   | [Engineered] Short / Medium / Long / Very Long  |
| 17  | `release_season`     | object     | [Engineered] Season of release                  |
| 18  | `log_budget`         | float64    | [Engineered] log(1+budget)                      |
| 19  | `log_revenue`        | float64    | [Engineered] log(1+revenue)                     |
| 20  | `genre_*` (multiple) | uint8      | [Engineered] One‑hot encoded genre flags        |

---

## 10. STREAMLIT DASHBOARD TABS

The dashboard is organised into **six interactive tabs**:

1. **Movie Financials** – Budget vs revenue scatter, rating distribution, key metrics, sidebar filters (year, genre).
2. **Statistical Insights** – Hypothesis tests (t‑tests, chi‑square, ANOVA) on financial/rating hypotheses.
3. **Advanced ML** – Correlation heatmap, PCA, K‑Means clustering, 3D cluster visualisation.
4. **Prediction** – Random Forest model training, feature importance, user input form, Gauge chart display.
5. **Sentiment Analysis** – IMDB sentiment pie chart, word clouds, random review explorer.
6. **Top Charts** – Rankings (top rated, highest grossing), genre market share.

**Sidebar widgets:** Year range slider, genre multiselect, data health report, dataset statistics.

---

## 11. DATA PIPELINE (Analysis Script Phases)

### Phase 1 – Data Acquisition & Cleaning

- Load `movies.csv`, rename `vote_average` → `rating`
- Report shape, missing values, duplicates
- Drop rows with missing critical info (`release_date`, `title`)
- Convert budget/revenue to numeric; replace 0s with median
- Parse `release_date` → extract year, month
- Parse `genres` → `genre_list` and `primary_genre`
- Save cleaned data to `movies_cleaned.csv`

### Phase 2 – Exploratory Data Analysis (EDA)

- **Univariate:** Rating histogram, revenue (log‑scale) histogram, genre frequency bar chart, movies per year line plot
- **Bivariate & Correlation:** Budget vs revenue scatter, ratings by genre box plot, numerical correlation heatmap, ratings over time

### Phase 3 – Feature Engineering & Encoding

- Create `decade`, `roi`, `is_profitable`, `runtime_category`, `log_budget`, `log_revenue`
- Encode genres (one‑hot, label), runtime category (ordinal), season (binary flag)
- Scale features with `StandardScaler`

### Phase 4 – Feature Selection

- **Filter method:** Correlation with target (`rating`) > 0.1
- **Embedded method:** Lasso regression (alpha=0.01)
- **Wrapper method:** Recursive Feature Elimination (RFE) with LinearRegression (n=10)
- Summary table comparing selections from all three methods

### Phase 5 – Statistical Analysis & Hypothesis Testing

- **Probability analysis:** Fit normal distribution to ratings; calculate probability of rating > 7, top 10% threshold
- **Hypothesis tests** (α = 0.05):
  - Independent t‑test: Action vs Drama ratings
  - One‑tailed t‑test: Summer vs non‑summer revenue
  - Chi‑square: Genre vs profitability
  - ANOVA: Ratings across top 5 genres
- Results summarised in a DataFrame

### Phase 6 – Unsupervised Machine Learning

- **PCA:** Scree plot, 3‑component loadings heatmap, 2D and 3D visualisations colour‑coded by genre
- **K‑Means clustering:** Elbow method to find optimal k (k=4 used), cluster visualisation on PCA space
- **Genre similarity analysis:** Pairwise Euclidean distances between genre centroids, heatmap

---

## 12. STYLES.CSS

| CSS Selector                 | Purpose                                                |
| ---------------------------- | ------------------------------------------------------ |
| `@import url(...)`           | Imports "Inter" font from Google Fonts                 |
| `html, body, [class*="css"]` | Applies Inter font globally                            |
| `.metric-card`               | KPI card styling (background, padding, border, shadow) |
| `.metric-card:hover`         | Lift‑up effect on hover                                |
| `h1`                         | Gradient text effect for title                         |
| `h1, h2, h3`                 | Default heading styles                                 |
| `.stTabs`                    | Custom tab styling (rounded corners, selected colour)  |
| `.chart-container`           | Card‑like container for charts with padding and shadow |

---

## 13. PROJECT STATISTICS

| Metric                               | Count                         |
| ------------------------------------ | ----------------------------- |
| Total Dashboard Files                | 5 (1 Python, 1 CSS, 3 CSV)    |
| Total Python Lines (dashboard)       | ~450                          |
| Total Python Lines (analysis script) | ~460                          |
| Functions (dashboard)                | 6 (4 main + 2 nested)         |
| Functions (analysis script)          | 3                             |
| Custom Classes                       | 0                             |
| Data Tables                          | 2 (Movies, Reviews)           |
| Visualisations (app)                 | 15+                           |
| Statistical Tests                    | 4                             |
| ML Models                            | 3 (RandomForest, KMeans, PCA) |
| UI Tabs                              | 6                             |
| Sidebar Widgets                      | 5                             |

---

## 

## 10. Project Statistics

| Metric                      | Count                                    |
| --------------------------- | ---------------------------------------- |
| **Total Files**             | 5 (1 Python, 1 CSS, 3 CSV)               |
| **Total Lines (Python)**    | ~450                                     |
| **Total Functions**         | 6 (4 main + 2 nested)                    |
| **Total Classes**           | 0 (Procedural approach)                  |
| **Total Data Tables**       | 2 (Movies, Reviews)                      |
| **Total Visualizations**    | 15+                                      |
| **Total Statistical Tests** | 3-4                                      |
| **ML Models Used**          | 4 (RandomForest, KMeans, PCA, Lasso/RFE) |
| **UI Tabs**                 | 6                                        |
| **Sidebar Widgets**         | 5                                        |
| **Engineered Features**     | 10+                                      |

---


