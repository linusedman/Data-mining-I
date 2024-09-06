import pandas as pd
import numpy as np
from seaborn import pairplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

