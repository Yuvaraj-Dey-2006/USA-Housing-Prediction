print(".================================================================.")
print("|            ******** USA HOUSING PREDICTION ********            |")
print("'================================================================'")


# ================================= #
#       IMPORTING LIBRARIES         #
# ================================= #

# data loading and feature engineering
import numpy as np
import pandas as pd

# EDA
import seaborn as sns
import matplotlib.pyplot as plt

# Spliting and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Hyperparameter tuning and validation
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import KFold, cross_val_score
import time

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# Performance testing
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# remove any non essential warnings
from warnings import filterwarnings
filterwarnings('ignore')


# ======================== #
#       DATA AUDIT         #
# ======================== #

df = pd.read_csv(r"C:\VS CODES\vs code\PYTHON\CORE ML\Resources and CSvs\USA_Housing.csv") # directory of the dataset
Address = df['Address']
df.drop(columns='Address', inplace=True)

print("============== INFORMATION OF DATASET ==============")
print(df.info()) 
print("====================================================\n")
print("============ CHECKING FOR MISSING VALUES ============")
print(df.isnull().sum()) 
print("=====================================================\n")
# ================= No missing value found =================


# ============================= #
#       OUTLIER ANALYSIS        #
# ============================= #

#### taking only the features and not the target. 
num_cols = df.drop(columns='Price').select_dtypes(include='number').columns
num_cols = np.array(num_cols)

# using boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])

plt.xticks(rotation=30)
plt.savefig("outlier_analysis.png", bbox_inches='tight')
plt.close()
# ============== These are no bad outleirs because there can be rich and poor in income and city and town for population ===============


# ============================= #
#       SPLITING OF DATA        #
# ============================= #

# seperating the features and target
X = df.drop(columns='Price')
y = df['Price']

# splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Model initialization with default parameters
pipe = Pipeline([('scaler', StandardScaler()), ('LR_R', LinearRegression())])
RF_R = RandomForestRegressor()
CB_R = CatBoostRegressor(verbose=0)

# fitting of training data to baseline models
pipe.fit(X_train, y_train)
RF_R.fit(X_train, y_train)
CB_R.fit(X_train, y_train)

print("\n.========================= BASELINE MODEL TEST ACCURACY =========================")

# prediction for baseline linear regression model
y_pred_lr = pipe.predict(X_test)
print(f"|   LINEAR REGRESSION           R² : {r2_score(y_test, y_pred_lr):.5f}")
print(f"|                              MAE : {mean_absolute_error(y_test, y_pred_lr):}")
print(f"|                             RMSE : {root_mean_squared_error(y_test, y_pred_lr):}")
print("---------------------------------------------------------------------------")

# prediction for baseline random forest regression model
y_pred_rf = RF_R.predict(X_test)
print(f"|   RANDOM FOREST REGRESSION    R² : {r2_score(y_test, y_pred_rf):.5f}")
print(f"|                              MAE : {mean_absolute_error(y_test, y_pred_rf)}")
print(f"|                             RMSE : {root_mean_squared_error(y_test, y_pred_rf)}")
print("---------------------------------------------------------------------------")

# predition for baseline catboost regression model
y_pred_cb = CB_R.predict(X_test)
print(f"|   CATBOOST REGRESSION         R² : {r2_score(y_test, y_pred_cb):.5f}")
print(f"|                              MAE : {mean_absolute_error(y_test, y_pred_cb)}")
print(f"|                             RMSE : {root_mean_squared_error(y_test, y_pred_cb)}")
print("'================================================================================\n\n")


cv=KFold(n_splits=5, shuffle=True, random_state=42)

score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2').mean()
print(".========== VALIDATION FOR LINEAR REGRESSION ==========")
print(f"|   VALIDATION SCORE: {score:.5f}")
print("'======================================================\n\n")


# ======================================================= #
#       OPTUNA TUNING FOR RANDOM FOREST REGRESSOR         #
# ======================================================= #

print("⚡ Tuning started for Random Forest....")
def objective_rf(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
              "criterion": trial.suggest_categorical("criterion", ['squared_error', 'absolute_error']),
              "max_depth": trial.suggest_int("max_depth", 3, 20),
              "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
              "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
              "max_features": trial.suggest_float("max_features", 0.3, 1.0),
              "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
              "random_state": 42,
              "n_jobs": -1,
              "verbose": 0
              }
    
    rf_reg_model = RandomForestRegressor(**params)

    score = cross_val_score(rf_reg_model, X_train, y_train, cv=cv, scoring='r2').mean()
    return score

rf_start = time.time()

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=100)

print("Tuning Completed ✅\n")
print(".========== HYPERPARAMETER TUNING FOR RANDOM FOREST ==========")
print(f"|   TIME TAKEN              : {(time.time() - rf_start)/60:.2f} min")
print(f"|   VALIDATION SCORE     R² : {study_rf.best_value:.5f}")
print(f"|   BEST PARAMETER          : {study_rf.best_params}")
print("'=============================================================\n\n")


# ====================================================== #
#        PREDICTIONS FOR RANDOM FOREST REGRESSOR         #
# ====================================================== #

best_rf_reg_model = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)
best_rf_reg_model.fit(X_train, y_train)
y_pred_rf = best_rf_reg_model.predict(X_test)


# ================================================== #
#       OPTUNA TUNING FOR CATBOOST REGRESSION        #
# ================================================== #

print("⚡ Tuning started for CatBoost....")
def objective_cb(trial):
    params = {
              "iterations": trial.suggest_int("iterations", 300, 2000),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
              "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),
              "depth": trial.suggest_int("depth", 4, 10),
              "random_strength": trial.suggest_float("random_strength", 0.1, 10),
              "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
              "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
              "loss_function": "RMSE",
              "eval_metric": "RMSE",
              "verbose": 0,
              "random_state": 42,
              "thread_count": -1,
              "allow_writing_files": False
             }
    
    cb_reg_model = CatBoostRegressor(**params)

    score = cross_val_score(cb_reg_model, X_train, y_train, cv=cv, scoring='r2').mean()
    return score

cb_start = time.time()

optuna.logging.set_verbosity(optuna.logging.WARNING)
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(objective_cb, n_trials=100, show_progress_bar=True)

print("Tuning Completed ✅\n")
print(".======= HYPERPARAMETER TUNING FOR CATBOOST REGRESSION =======")
print(f"|   TIME TAKEN           : {(time.time() - cb_start)/60:.2f}")
print(f"|   VALIDATION SCORE  R² : {study_cb.best_value:.5f}")
print(f"|   BEST PARAMETER       : {study_cb.best_params}")
print("'=============================================================\n\n")


# ================================================= #
#        PREDICTIONS FOR CATBOOST REGRESSOR         #
# ================================================= #

best_cb_reg_model = CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=0, thread_count=-1, allow_writing_files=False)
best_cb_reg_model.fit(X_train, y_train)
y_pred_cb = best_cb_reg_model.predict(X_test)


# ================================= #
#       PERFORMANCE ANALYSIS        #
# ================================= #

print(".=================== FINAL PERFORMANCE ANALYSIS ===================")
print(f"|    LINEAR REGRESSION ACCURACY: {r2_score(y_test, y_pred_lr):.5f}")
print(f"|    RANDOM FOREST REGRESSION ACCURACY: {r2_score(y_test, y_pred_rf):.5f}")
print(f"|    CATBOOST REGRESSION ACCURACY: {r2_score(y_test, y_pred_cb):.5f}")
print("'================================================================\n")

scores = {
    "LINEAR REGRESSION":  r2_score(y_test, y_pred_lr),
    "RANDOM FOREST REGRESSION":  r2_score(y_test, y_pred_rf),
    "CAT BOOST REGRESSION":  r2_score(y_test, y_pred_cb),
}
best_name = max(scores, key=scores.get)

print(".====================================================================================.")
print(f"|   Best model on test set: {best_name.upper()} (R²: {scores[best_name]:.5f})")
print("'===================================================================================='")



