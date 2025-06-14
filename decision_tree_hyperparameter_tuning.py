import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

## Decision Tree classifier 
dt = DecisionTreeClassifier(random_state=42)

## Grid Search hyperparameter tuning
search_params = {
    "max_depth" : [None, 3, 5, 7, 10],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10]
}

gridsearch_dt = GridSearchCV(dt, search_params, scoring="accuracy", cv=5)
gridsearch_dt.fit(X_train, y_train)
print(gridsearch_dt.best_params_)

## Random Search hyperparameter tuning
search_params = {
    "max_depth" : [None, 3, 5, 7, 10],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10, 15, 20]
}

randsearch_dt = RandomizedSearchCV(dt, search_params, scoring="accuracy", cv=5)
randsearch_dt.fit(X_train, y_train)
print(randsearch_dt.best_params_)

## Bayesian Optimizaion hyperparameter tuning  
def objective(trial):
    
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 100),
        "max_features": trial.suggest_categorical("max_features",["sqrt", "log2"]),
        "min_samples_split": trial.suggest_int( "min_samples_split", 2, 50)
    }
    
    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
print(study.best_value)
bayesoptimizd_dt = DecisionTreeClassifier(**study.best_params)
bayesoptimizd_dt.fit(X_train, y_train)

## Results after hyperparameter tuning
results = {
    "Grid Search": get_classifier_results(y_test, X_test, gridsearch_dt),
    "Random Search": get_classifier_results(y_test, X_test, randsearch_dt),
    "Bayesian Optimization": get_classifier_results(y_test, X_test, bayesoptimizd_dt)
}

show_results(results, save_fig=True, save_dir=visualization_path)