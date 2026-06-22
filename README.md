# USA Housing Price Prediction

A machine learning regression project that predicts house prices using:

- Linear Regression
- Random Forest Regressor
- CatBoost Regressor

Optuna is used for hyperparameter tuning and the best performing model is printed at the end.

---

## Dataset

**File:** `USA_Housing.csv`

### Features
- Avg. Area Income
- Avg. Area House Age
- Avg. Area Number of Rooms
- Avg. Area Number of Bedrooms
- Area Population

### Target
- Price

The `Address` column is removed before training as it is not useful for numerical regression.

---

## Dataset Source

Dataset used in this project is sourced from Kaggle:
- https://www.kaggle.com/code/dibkb9/usa-housing

All credit goes to the original dataset creator.

---

## Project Structure

```text
USA-Housing-Prediction/
│
├── USA_Housing.csv
├── USA_Housing_predictions.py
├── README.md
└── outlier_analysis.png
```

---

## Workflow

```text
Data Audit
(df.info, null check)
    ↓
Outlier Analysis
(Boxplot → saved as outlier_analysis.png)
    ↓
Splitting of Data
(Train 80% / Test 20%)
    ↓
Baseline Model Training
(Linear Regression, Random Forest, CatBoost)
    ↓
Baseline Validation
(KFold Cross Validation on Linear Regression)
    ↓
Optuna Tuning → Random Forest
(100 trials, KFold-5)
    ↓
Optuna Tuning → CatBoost
(100 trials, KFold-5)
    ↓
Final Performance Analysis
(R² comparison across all 3 models)
    ↓
Best Model Announced
```

---

## Results

| Model | Test R² |
|---|---|
| Linear Regression | ~0.918 |
| Random Forest | ~0.883 |
| CatBoost | ~0.907 |

Linear Regression achieved the best performance on this dataset.

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python USA_Housing_predictions.py
```

---

## Skills Demonstrated

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Selection
- Regression Modeling
- Hyperparameter Tuning with Optuna
- Cross Validation
- Model Performance Comparison