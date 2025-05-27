import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
from scipy.stats import probplot
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = Path(project_dir) / "dataset/internet_service_churn.csv"
df = pd.read_csv(dataset_path)
visualization_path = Path(project_dir) / "visualizations"

## Overview of dataset
print(df.head(5))
print(df.describe())
print(df.info())

## Quantifying & Visualizing the missing values
missing_values = pd.DataFrame({"feature":df.columns, "missing_values_ratio":df.isnull().mean()})
print(missing_values)
plt.bar(missing_values["feature"], missing_values["missing_values_ratio"])
plt.ylabel("Percentage")
plt.xlabel("Feature")
plt.title("Percentage of Missing Values")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(Path(visualization_path) / "missing_values.png")
# plt.show()


## Quantifying & Visualizing the unique values  
df.drop(columns="id", inplace=True)
df_values = pd.DataFrame({"feature":df.columns, "unique_values_count":df.nunique()})
# print(df_values)
plt.bar(df_values["feature"], df_values["unique_values_count"])
plt.ylabel("Number of Unique Categories")
plt.xlabel("Feature")
plt.title("Cardinality")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(Path(visualization_path) / "unique_values_count.png")
# plt.show()

## Seperating categorical & numerical fetures
categorical_features = df_values[(df_values['unique_values_count'] > 0) & (df_values['unique_values_count'] <= 10)]['feature'].to_list()
# print("Categorical Features: ", ", ".join(categorical_features))

numerical_features = df_values['feature'].drop(categorical_features).to_list()
# print("Numerial Features: ", ", ".join(numerical_features))

## Visualizing Q-Q plot for numerical features to check for its distribution
for num_feat in numerical_features:
    plt.figure()
    probplot(df[num_feat], plot=plt)
    plt.title(f"Q-Q Plot of {num_feat}")
    plt.savefig(Path(visualization_path) / f"q-q_plot_feature_{num_feat}.png")
    # plt.show()

real_numerical_features = list(df.select_dtypes("float64"))
# print("Real Numerial Features: ", ", ".join(numerical_features))

discrete_numerical_features = [feature for feature in numerical_features if feature in list(df.select_dtypes("int64"))]
# print("Discrete Numerial Features: ", ", ".join(discrete_numerical_features))

df[numerical_features].hist(bins=30, figsize=(12, 6))
plt.tight_layout()
plt.savefig(Path(visualization_path) / "histogram_plot_num_features.png")
# plt.show()

## Imputing Mising Values
fill_values = df[real_numerical_features].mean(axis=0).to_dict()
other_fill_values = {k:v[0] for k, v in df[discrete_numerical_features].mode(axis=0).to_dict().items()}
fill_values.update(other_fill_values)
# print(fill_values)
df.fillna(fill_values,  inplace=True)
print(df.info())


