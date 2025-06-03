import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from pathlib import Path
from utils import get_classifier_results, show_results
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
svc = SVC(random_state=42)

logr.fit(X_train, y_train)
knn.fit(X_train, y_train)
dt.fit(X_train, y_train)
svc.fit(X_train, y_train)


results = {
    "logistic regression": get_classifier_results(y_test, X_test, logr),
    "KNN(n=3)" : get_classifier_results(y_test, X_test, knn),
    "SVM Classifier": get_classifier_results(y_test, X_test, svc),
    "Decision Tree": get_classifier_results(y_test, X_test, dt),
}

show_results(results, save_fig=True, save_dir=visualization_path)

