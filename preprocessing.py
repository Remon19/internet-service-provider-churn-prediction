import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = Path(project_dir) / "dataset/internet_service_churn.csv"
df = pd.read_csv(dataset_path)
visualization_path = Path(project_dir) / "visualizations" 

## Read dataset
df = pd.read_csv(dataset_path)

df.drop(columns=["id"], inplace=True)
X = df.drop(columns=["churn"])
y = df["churn"]

## Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)
# print(X_train.shape)

## Imputting missing values and Preprocessing dataset
median_imputer = SimpleImputer(strategy="median")
mode_imputer = SimpleImputer(strategy="most_frequent")

fill_median_scale = Pipeline(
    steps=[
        ("median imputer", median_imputer),
        ("scaler", StandardScaler()), 
    ]
)

fill_mode_scale = Pipeline(
    steps=[
        ("mode imputer", mode_imputer),
        ("scaler", StandardScaler())
    ]
)

preprocessr = ColumnTransformer(
    transformers=[
        ("fill median and scale", fill_median_scale, ["reamining_contract"]),
        ("fill mode and scale", fill_mode_scale, ["upload_avg", "download_avg"]),
        ("scaler", StandardScaler(), ["subscription_age"]),
        ("one hot encoder", OneHotEncoder(drop="if_binary"), ["is_tv_subscriber","is_movie_package_subscriber"])
    ],
    remainder="passthrough"
)


X_train = preprocessr.fit_transform(X_train)
X_test = preprocessr.transform(X_test)
# print(X_train.shape)

## Concatenating features and the target
train_xy = np.c_[X_train, y_train]
test_xy = np.c_[X_test, y_test]
# print(train_data.shape)

new_columns = preprocessr.get_feature_names_out().tolist()
new_columns.append("churn")
## Saving processed data as a parquet file
train_data = pd.DataFrame(train_xy, columns=new_columns)
test_data = pd.DataFrame(test_xy, columns=new_columns)
train_data.to_csv(Path(project_dir) / "dataset/processed_train.csv", header=True, index=False)
test_data.to_csv(Path(project_dir) / "dataset/processed_test.csv", header=True, index=False)
# print(processed_data.head())









