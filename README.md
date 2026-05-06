# Cinematics Analytics Dashboard

## Overview

The Cinematics Analytics Dashboard repository contains a production-ready Python data analytics pipeline for movie performance and audience sentiment analysis. The project uses TMDB movie metadata and IMDB review data to generate cleaned datasets, exploratory visualizations, and a reproducible analysis workflow.

## One-line Summary

A complete movie analytics pipeline for cleaning dataset inputs, exploring movie finance and genre patterns, and generating calibrated dataset outputs for downstream analysis.

## Problem Solved

This repository helps analytics practitioners understand movie success drivers by providing a structured data cleaning pipeline, exploratory data analysis, and reproducible outputs for finance, rating, and genre evaluation.

## Features

- Clean and impute TMDB movie financial data
- Extract release year, genre, and revenue distribution insights
- Generate EDA visualizations for budget, revenue, ratings, and genre frequency
- Output a cleaned CSV dataset for downstream modeling
- Support reproducible execution with a single installation flow

## Tech Stack

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Jupyter
- Streamlit
- Plotly
- WordCloud

## Installation

```bash
git clone https://github.com/ali-ezz/cinematics-analytics-dashboard.git
cd cinematics-analytics-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run the analysis pipeline to clean the TMDB dataset and generate plots:

```bash
python run_analysis_fixed.py
```

Alternative exploratory options:

```bash
python run_analysis.py
jupyter notebook movie_analysis_pipeline.ipynb
```

## Project Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── .env.example
├── run_analysis_fixed.py
├── run_analysis.py
├── extract_script.py
├── header.py
├── movie_analysis_pipeline.ipynb
├── movies_cleaned.csv
├── archive (6)/
│   └── movies.csv
├── archive (4)/
│   └── IMDB Dataset.csv
├── assets/
│   └── screenshots/
├── src/
│   └── README.md
├── tests/
│   └── test_data_loading.py
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
└── *.png
```

## Environment Variables

This project supports external dataset configuration through a `.env` file. Copy and customize the example file before running the pipeline.

```bash
cp .env.example .env
```

Example settings:

- `DATA_ROOT`: Root dataset folder
- `TMDB_MOVIES`: Path to the raw TMDB dataset
- `IMDB_REVIEWS`: Path to the raw IMDB sentiment dataset
- `OUTPUT_FILE`: Path for the cleaned output CSV

## Results

The main output is a cleaned dataset saved as `movies_cleaned.csv`. The repository also includes analysis artifacts such as:

- `rating_distribution.png`
- `budget_revenue_scatter.png`
- `revenue_distribution.png`
- `genre_frequency.png`

## Future Improvements

- Convert the analysis pipeline into a reusable package under `src/`
- Add end-to-end tests for data integrity and output validation
- Create a Streamlit dashboard entrypoint from the cleaned dataset
- Improve missing-value handling and revenue/budget validation logic
- Add automated release notes and versioning

## Credits

This repository was created for movie analytics and dataset exploration. It uses publicly available TMDB and IMDB CSV data sources.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
