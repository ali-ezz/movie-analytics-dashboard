# Movie Analytics Dashboard

Movie Analytics Dashboard is a production-ready Python pipeline for analyzing movie financial performance, audience sentiment, and genre-level trends using TMDB and IMDB datasets.

## Overview

This project delivers a reproducible movie analytics workflow for cleaning raw movie and review data, generating exploratory analysis, and producing a cleaned dataset ready for downstream modeling.

## Problem Solved

Many movie analytics projects contain fragmented data, inconsistent cleaning steps, and unclear execution paths. This repository centralizes the data ingestion, preprocessing, and exploratory analysis steps into a single reproducible pipeline.

## Features

- Load TMDB movie metadata and IMDB review data
- Clean and merge movie financial, ratings, and genre information
- Extract release year, revenue, budget, and sentiment distributions
- Generate exploratory visualizations for key movie performance metrics
- Produce a cleaned CSV dataset for downstream analysis
- Support reproducible execution with a well-defined installation flow

## Tech Stack

- Python 3.x
- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- Jupyter Notebook

## Installation

```bash
git clone https://github.com/ali-ezz/movie-analytics-dashboard.git
cd movie-analytics-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run the main analysis pipeline to clean data and generate outputs:

```bash
python run_analysis_fixed.py
```

Alternative exploratory options:

```bash
python run_analysis.py
jupyter notebook movie_analysis_pipeline.ipynb
```

## Project Structure

```text
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
└── .github/
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

## Environment Variables

Use the `.env.example` file as a template and create a local `.env` file before running the analysis.

```bash
cp .env.example .env
```

Required variables:

- `DATA_ROOT`: Folder containing raw dataset files
- `TMDB_MOVIES`: Path to the TMDB movie dataset CSV
- `IMDB_REVIEWS`: Path to the IMDB reviews dataset CSV
- `OUTPUT_FILE`: Output path for the cleaned dataset CSV

## Results

The primary result is the cleaned dataset stored in `movies_cleaned.csv`. The pipeline also produces analysis artifacts such as:

- `rating_distribution.png`
- `budget_revenue_scatter.png`
- `revenue_distribution.png`
- `genre_frequency.png`

## Future Improvements

- Refactor analysis code into a reusable Python package under `src/`
- Add unit tests for data validation and output integrity
- Standardize raw data under a dedicated `data/` directory
- Add a lightweight dashboard entrypoint for interactive exploration
- Document dataset preparation and expected input formats

## License

This project is licensed under the MIT License. See `LICENSE` for details.
