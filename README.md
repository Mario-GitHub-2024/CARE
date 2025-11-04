# CARE - Customer Churn Prediction

A machine learning pipeline for predicting customer churn using multiple algorithms.

## Project Overview

This project implements a comprehensive customer churn prediction system using four different machine learning models:
- XGBoost with hyperparameter optimization
- CatBoost with categorical feature handling
- Random Forest
- Logistic Regression

## Project Structure

```
churn_project/
├── data/                 # Data files
│   ├── train.csv        # Training data
│   ├── test.csv         # Test data
│   └── predictions.csv  # Model predictions
├── model/               # ML pipeline modules
│   ├── preprocessing.py # Data preprocessing
│   ├── train_model.py   # Model training
│   ├── predict.py       # Prediction & evaluation
│   └── utils.py         # Utility functions
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mario-GitHub-2024/CARE.git
cd CARE/churn_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

Or run individual modules:
```bash
python model/preprocessing.py
python model/train_model.py
python model/predict.py
```

## Features

- **Data Preprocessing**: Automatic handling of numeric and categorical features
- **Multiple Models**: Comparison of 4 different ML algorithms
- **Hyperparameter Optimization**: Automated tuning for XGBoost
- **Performance Evaluation**: ROC curves, AUC scores, precision, recall, accuracy
- **Categorical Feature Support**: Proper handling of non-numeric data

## Requirements

See `requirements.txt` for complete dependency list.

## Data

- Training data: `data/train.csv`
- Test data: `data/test.csv`
- Predictions: `data/predictions.csv`

## Model Performance

The system evaluates models using:
- ROC AUC scores
- Precision, Recall, Accuracy
- ROC curves for visual comparison

## License

This project is for educational and research purposes.