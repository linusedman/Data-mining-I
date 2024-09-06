import pandas as pd
import numpy as np

data = pd.read_csv("Data-mining-I/iris_data.csv", delimiter=";")
labels = pd.read_csv("Data-mining-I/iris_labels-1.csv", delimiter=";")

# data = pd.read_csv("iris_data.csv", delimiter=";")
# labels = pd.read_csv("iris_labels-1.csv", delimiter=";")

print("Shape of data: ", data.shape)
print("Head of data", data.head)

print("Shape of labels: ", labels.shape)
print("Head of labels: ", labels.head)

dataframe = pd.merge(data, labels, on = 'id', how = "inner")

print("Shape of labels: ", dataframe.shape)
print("Head of labels: ", dataframe.head)

# Hej