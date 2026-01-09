# Aadhaar Data TabFoundry 

A comprehensive machine learning project for analyzing and predicting patterns in Aadhaar enrollment data across Indian states and districts. This project implements multiple prediction tasks including anomaly detection, time-series forecasting, and spatial inequality analysis using AutoGluon and various deep learning approaches.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tasks](#tasks)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)

## Overview

This project analyzes monthly Aadhaar enrollment data to:

1. **Detect Anomalies**: Identify unusual enrollment patterns using statistical methods and machine learning
2. **Forecast Future Enrollments**: Predict enrollment numbers 7 days ahead using time-series analysis
3. **Analyze Spatial Inequality**: Detect regions with disproportionate enrollment patterns

The analysis covers enrollment data across all Indian states and union territories, broken down by districts and pincodes, with demographic segmentation (age groups: 0-5, 5-17, 18+).

## Dataset

The dataset consists of monthly Aadhaar enrollment records from various states and union territories of India. Each CSV file contains:

- **Geographic Information**: State, District, Pincode
- **Temporal Information**: Date of enrollment
- **Demographic Breakdowns**: 
  - Age 0-5 years
  - Age 5-17 years
  - Age 18+ years

The dataset includes 40+ CSV files covering states such as Andhra Pradesh (AP), Karnataka (KA), Maharashtra (MH), Delhi (DL), Tamil Nadu (TN), and many others.


## Tasks

### Task 1: Anomaly Detection

Identifies unusual enrollment patterns based on:
- Z-score analysis (rolling and state-level)
- Enrollment volatility detection
- Statistical outlier identification

**Target Variable**: Binary classification (`is_anomaly`)

**Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Task 2: Time-Series Forecasting

Predicts total enrollments 7 days ahead using:
- Temporal features (day, month, year, cyclical encodings)
- Lag features (1, 7, 30 days)
- Rolling statistics (mean, std, trends)
- Demographic ratios

**Target Variable**: Continuous regression (`target_7d`)

**Metrics**: MAE, RMSE, MAPE, R²

### Task 3: Spatial Inequality Analysis

Detects regions with disproportionately high enrollment relative to state/district averages.

**Target Variable**: Binary classification (`high_inequality`)

**Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 8GB RAM (16GB recommended for AutoGluon)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ADA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For baseline AutoGluon model:
```bash
cd Baseline
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

1. **Merge all state CSV files**:
```bash
python data_merging.py
```

2. **Clean and standardize the data**:
```bash
python data_cleaning.py
```

This creates `cleaned_aadhaar_dataset.csv` with standardized state/district names and proper date formatting.

### Training Models

#### Baseline (AutoGluon)

Run the complete pipeline in Jupyter:
```bash
cd Baseline
jupyter notebook pipeline.ipynb
```

The pipeline includes:
- Feature engineering (temporal, spatial, demographic)
- Lag feature creation
- Train/test splitting (80/20 temporal split)
- AutoGluon model training for all three tasks
- Evaluation and prediction export

#### Alternative Models

Explore other implementations:
```bash
cd TabTransformer
# Follow instructions in the respective directory
```

### Making Predictions

Trained models automatically generate prediction CSV files:
- `task1_anomaly_predictions.csv`
- `task2_forecasting_predictions.csv`
- `task3_inequality_predictions.csv`

## Models

### AutoGluon (Baseline)

AutoGluon automatically trains and ensembles multiple models:
- LightGBM
- CatBoost
- XGBoost
- Neural Networks
- Random Forest

**Configuration**:
- Time limit: 3600 seconds per task
- Evaluation metric: Task-specific (accuracy, RMSE, etc.)
- Cross-validation: Time-series aware splitting

### CustomTabTransformer

Deep learning approach using transformer architecture for tabular data.

### TabPFN

Prior-fitted network comparison with AutoGluon baseline.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request