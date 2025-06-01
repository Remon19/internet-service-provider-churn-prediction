import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
import os
from utils import get_classifier_results

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

dt = DecisionTreeClassifier(random_state=42)

search_params = {
    "criterion" : ["gini", "log_loss"],
    "max_depth" : [3, 5, 7, 10],
    "max_features": ["sqrt", "log2"] 
}

cv_dt = GridSearchCV(dt, search_params, scoring=("accuracy"), cv=7)
cv_dt.fit(X_train, y_train)
print(cv_dt.best_params_)
result = get_classifier_results(y_test, X_test, cv_dt)

print("Results of Decison Tree with the best parametrs found by Gridsearch")
cm_disp = ConfusionMatrixDisplay(result["confusion matrix"], display_labels=["Not Churn","Churn"])
print(f"""  ROC-AUC Score: {result["roc-auc score"]:0.4f}
            Accuracy: {result["accuracy"]:0.4f}
            Classification Report: {result["classification_report"]}
        """)
cm_disp.plot(cmap="coolwarm")
plt.title(f"Confusion Matrix of Decision Tree trained by GridSearch Cross Validation method")
plt.savefig(Path(visualization_path) / "cm_plot_dt_cv_grid.png")
# plt.show()