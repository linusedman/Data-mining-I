import pandas as pd
import numpy as np
from seaborn import pairplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import scipy

data = pd.read_csv("Data-mining-I/iris_data.csv", delimiter=";")
labels = pd.read_csv("Data-mining-I/iris_labels-1.csv", delimiter=";")

# data = pd.read_csv("iris_data.csv", delimiter=";")
# labels = pd.read_csv("iris_labels-1.csv", delimiter=";")

# ===================================================================
# Part 1

# print("Shape of data: ", data.shape)
# print("Head of data", data.head)

# print("Shape of labels: ", labels.shape)
# print("Head of labels: ", labels.head)

dataframe = pd.merge(data, labels, on = 'id', how = "inner")

# print("Shape of labels: ", dataframe.shape)
# print("Head of labels: ", dataframe.head)

dataframe.drop(["examiner"], axis = 1, inplace = True)
dataframe.drop(['id'], axis=1, inplace=True)

dataframe.sort_values(by="species")

# pairplot(dataframe, hue= "species")
# plt.savefig("Data-mining-I/pairplot.png")
# plt.savefig("pairplot.png")

# print(f"Mean of sl: {dataframe['sl'].mean()}")
# print(f"Satandard deviation of sl: {dataframe['sl'].std()}")

# ===================================================================
# Part 2

missing_values = dataframe[dataframe["sl"] == -9999]

dataframe = dataframe.drop(index=missing_values.index)


# print(f"Mean of sl: {dataframe['sl'].mean()}")
# print(f"Satandard deviation of sl: {dataframe['sl'].std()}")
# pairplot(dataframe, hue= "species")
# plt.savefig("Data-mining-I/pairplot1.png")

num_virginica = (dataframe["species"] == "Iris-virginica").sum()
num_setosa = (dataframe["species"] == "Iris-setosa").sum()
num_versicolor = (dataframe["species"] == "Iris-versicolor").sum()

print(f"Number of Virginica: {num_virginica}")
print(f"Number of Setosa: {num_setosa}")
print(f"Number of versicolor: {num_versicolor}")

# ===================================================================
# Part 3

columns = ['sl', 'pl', 'pw', 'sw']

zscores = scipy.stats.zscore(dataframe[columns])

# 3 standard deviations 
threshold = 3


mask = (abs(zscores) < threshold).all(axis=1)

# Filter the DataFrame based on the mask
dataframe = dataframe[mask]

# print(f"Mean of sl: {dataframe['sl'].mean()}")
# print(f"Satandard deviation of sl: {dataframe['sl'].std()}")
# pairplot(dataframe, hue= "species")
# plt.savefig("Data-mining-I/pairplot2.png")

# ===================================================================
# Part 4

# Step 1: Scale numeric columns
scaler = MinMaxScaler()
numeric_columns = dataframe.select_dtypes(include=[np.number])  # Select numeric columns
scaled_data = scaler.fit_transform(numeric_columns)

# Step 2: Convert the scaled data back to a DataFrame
scaled_dataframe = pd.DataFrame(scaled_data, columns=numeric_columns.columns)

# Step 3: Concatenate the scaled numeric DataFrame with the non-numeric columns
non_numeric_columns = dataframe.select_dtypes(exclude=[np.number]).reset_index(drop=True)
MM_dataframe = pd.concat([scaled_dataframe, non_numeric_columns], axis=1)

print(f"Mean of sl: {MM_dataframe['sl'].mean()}")
print(f"Satandard deviation of sl: {MM_dataframe['sl'].std()}")

# Step 1: Scale numeric columns
scaler = StandardScaler()
numeric_columns = dataframe.select_dtypes(include=[np.number])  # Select numeric columns
scaled_data = scaler.fit_transform(numeric_columns)

# Step 2: Convert the scaled data back to a DataFrame
scaled_dataframe = pd.DataFrame(scaled_data, columns=numeric_columns.columns)

# Step 3: Concatenate the scaled numeric DataFrame with the non-numeric columns
non_numeric_columns = dataframe.select_dtypes(exclude=[np.number]).reset_index(drop=True)
SS_dataframe = pd.concat([scaled_dataframe, non_numeric_columns], axis=1)

print(f"Mean of sl: {SS_dataframe['sl'].mean()}")
print(f"Satandard deviation of sl: {SS_dataframe['sl'].std()}")

pca = PCA ()

numeric_columns = dataframe.select_dtypes(include=[np.number])  # Select numeric columns
scaled_data = pca.fit_transform(numeric_columns)
print(scaled_data)



# Step 2: Convert the scaled data back to a DataFrame
scaled_dataframe = pd.DataFrame(scaled_data, columns=numeric_columns.columns)

# Step 3: Concatenate the scaled numeric DataFrame with the non-numeric columns
non_numeric_columns = dataframe.select_dtypes(exclude=[np.number]).reset_index(drop=True)
PCA_dataframe = pd.concat([scaled_dataframe, non_numeric_columns], axis=1)

# print(f"Mean of sl: {PCA_dataframe['sl'].mean()}")
# print(f"Satandard deviation of sl: {PCA_dataframe['sl'].std()}")
# pairplot(PCA_dataframe, hue= "species")
# plt.savefig("Data-mining-I/pairplot_pca.png")

print(pca.explained_variance_ratio_)

pca_df = pd.DataFrame(pca.components_, columns=['sl', 'sw', 'pl', 'pw'], index=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

print(pca_df)



# Step 1: Scale numeric columns
scaler = MinMaxScaler(feature_range=(0,100))

# Scale the 'pl' column only
dataframe['pl'] = scaler.fit_transform(dataframe[['pl']])

pca = PCA ()

numeric_columns = dataframe.select_dtypes(include=[np.number])  # Select numeric columns
scaled_data = pca.fit_transform(numeric_columns)
print(scaled_data)

# Step 2: Convert the scaled data back to a DataFrame
scaled_dataframe = pd.DataFrame(scaled_data, columns=numeric_columns.columns)

# Step 3: Concatenate the scaled numeric DataFrame with the non-numeric columns
non_numeric_columns = dataframe.select_dtypes(exclude=[np.number]).reset_index(drop=True)
PCA_dataframe = pd.concat([scaled_dataframe, non_numeric_columns], axis=1)

# print(f"Mean of sl: {PCA_dataframe['sl'].mean()}")
# print(f"Satandard deviation of sl: {PCA_dataframe['sl'].std()}")
# pairplot(PCA_dataframe, hue= "species")
# plt.savefig("Data-mining-I/pairplot_pca.png")

print(pca.explained_variance_ratio_)

pca_df = pd.DataFrame(pca.components_, columns=['sl', 'sw', 'pl', 'pw'], index=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

print(pca_df)

# ===================================================================
# Part 5