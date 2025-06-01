import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from utils import get_classifier_results
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
visualization_path = Path(project_dir) / "visualizations" 

## reading processed data
df_train = pd.read_csv(Path(project_dir) / "dataset/processed_train.csv")
df_test = pd.read_csv(Path(project_dir) / "dataset/processed_test.csv")

X_train = df_train.drop(columns="churn")
y_train = df_train["churn"]
X_test = df_test.drop(columns="churn")
y_test = df_test["churn"]

logr= LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier(random_state=42)

logr.fit(X_train, y_train)
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)


results = {
    "logistic regression": get_classifier_results(y_test, X_test, logr),
    "KNN(n=3)" : get_classifier_results(y_test, X_test, knn),
    "Decision Tree": get_classifier_results(y_test, X_test, dt)
}


for model, result in results.items():
    cm_disp = ConfusionMatrixDisplay(result["confusion matrix"], display_labels=["Not Churn","Churn"])
    print(f"""{model} results:
          ROC-AUC Score: {result["roc-auc score"]:0.4f}
          Accuracy: {result["accuracy"]:0.4f}
          Classification Report: {result["classification_report"]}
          """)
    cm_disp.plot(cmap="coolwarm")
    plt.title(f"Confusion Matrix for {model}")
    plt.savefig(Path(visualization_path) / f"cm_plot_{model}")
    # plt.show()

