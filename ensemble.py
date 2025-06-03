import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
from pathlib import Path
import os
from utils import get_classifier_results, show_results
import optuna

project_dir = os.path.dirname(os.path.abspath(__file__))
visualization_path = Path(project_dir) / "visualizations" 

## reading processed data
df_train = pd.read_csv(Path(project_dir) / "dataset/processed_train.csv")
df_test = pd.read_csv(Path(project_dir) / "dataset/processed_test.csv")

## features and target
X_train = df_train.drop(columns="churn")
y_train = df_train["churn"]
X_test = df_test.drop(columns="churn")
y_test = df_test["churn"]

rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

base_model_resulst = {
    "Random Forest": get_classifier_results(y_test, X_test, rf),
    "XGBoost Classifier": get_classifier_results(y_test, X_test, xgb)
}

show_results(base_model_resulst, save_fig=True, save_dir=visualization_path)

search_param = {
    "n_estimators": [10, 100, 500, 1000],
    "max_depth": [None, 2, 5, 10, 20, 50],
    "min_samples_split": [2, 5, 10, 20, 50],
    "max_features": ['sqrt', 'log2'],
}

randsearch_cv_rf = RandomizedSearchCV(rf, search_param, n_iter=50, scoring="accuracy", cv=5, random_state=42)
randsearch_cv_rf.fit(X_train, y_train)
print("Best parameters for Random Forest using random search method:\n", randsearch_cv_rf.best_params_)
# {'n_estimators': 500, 'min_samples_split': 10, 'max_features': 'log2', 'max_depth': 50}

gridsearch_cv_rf = GridSearchCV(rf, search_param, scoring="accuracy", cv=5)
gridsearch_cv_rf.fit(X_train, y_train)
print("Best parameters for Random Forest using grid search method:\n", gridsearch_cv_rf.best_params_)

def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_int("n_estimator", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 100),
        "max_features": trial.suggest_categorical("max_features",["sqrt", "log2"]),
        "min_samples_split": trial.suggest_int( "min_samples_split", 2, 50)
    }
    
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best parameters for Random Forest using random search method:\n", study.best_params)
print(study.best_value)
bayesoptimizd_rf = RandomForestClassifier(**study.best_params)
bayesoptimizd_rf.fit(X_train, y_train)

results = {
    "Random Search Tuned Random Forest": get_classifier_results(y_test, X_test, randsearch_cv_rf.best_estimator_),
    "Grid Search Tuned Random Forest": get_classifier_results(y_test, X_test, gridsearch_cv_rf.best_estimator_),
    "Bayesian Optimized Random Forest": get_classifier_results(y_test, X_test, bayesoptimizd_rf)
}

show_results(results, save_fig=True, save_dir=visualization_path)



  