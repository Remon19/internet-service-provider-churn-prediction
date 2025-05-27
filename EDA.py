import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
import os

dataset_path = Path("F:\MyProjects\internet service provider churn prediction\internet_service_churn.csv")
df = pd.read_csv(dataset_path)

# print(df.head(5))
# print(df.describe())
# print(df.info())

df.drop(columns="id", inplace=True)
df_values = pd.DataFrame({"feature":df.columns, "unique_values_count":df.nunique()})
# print(df_values)

## Seperating categorical & numerical fetures
categorical_features = df_values[(df_values['unique_values_count'] > 0) & (df_values['unique_values_count'] <= 10)]['feature'].to_list()
# print("Categorical Features: ", ", ".join(categorical_features))

numerical_features = df_values['feature'].drop(categorical_features).to_list()
# print("Numerial Features: ", ", ".join(numerical_features))

real_numerical_features = list(df.select_dtypes("float64"))
# print("Real Numerial Features: ", ", ".join(numerical_features))

discrete_numerical_features = [feature for feature in numerical_features if feature in list(df.select_dtypes("int64"))]
# print("Discrete Numerial Features: ", ", ".join(discrete_numerical_features))

# sns.pairplot(df, vars=numerical_features, diag_kind="hist")
# plt.title("Pair Plot")
# plt.show()

## Imputing Mising Values
fill_values = df[real_numerical_features].mean(axis=0).to_dict()
other_fill_values = {k:v[0] for k, v in df[discrete_numerical_features].mode(axis=0).to_dict().items()}
fill_values.update(other_fill_values)
# print(fill_values)
df.fillna(fill_values,  inplace=True)
print(df.info())


