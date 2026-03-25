# NBA Team Scoring Prediction Project

## Overview

This project builds a modular pipeline for predicting NBA team scoring outcomes using historical team game logs.  
The workflow has been refactored from a single notebook into standalone scripts and Python modules for:

- raw data generation
- data cleaning
- feature engineering
- baseline statistical modeling
- tree-based modeling
- predictive evaluation using negative log-likelihood (NLL)

## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── README.md
├── data
│   ├── raw
│   │   └── nba_team_logs_raw.csv
│   └── processed
│       └── nba_matchups_features.csv
├── outputs
│   └── predictions
│       ├── glm_predictions.csv
│       └── tree_model_predictions.csv
├── scripts
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── distributions.py
│   ├── evaluation.py
│   └── models
│       ├── __init__.py
│       ├── glm_model.py
│       ├── tree_models.py
│       ├── mlp_model.py
│       └── regularized_poisson.py
└── Data_Science_Project.ipynb