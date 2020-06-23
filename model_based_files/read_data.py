import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import csv
import pandas
data = pandas.read_csv("predict_data_5_30sec_100cars.csv")
#data = pandas.read_csv("predict_data/predict_data_4_10sec_50cars.csv")
#data = pandas.read_csv("model_based_data/model_based_data_5_10sec_50cars.csv")
#print("Mean of all data:")
#print(data.mean())
print("Mean Distance and Time:")
print(data[['Distance', 'Time']].mean())
print("Standard deviation Distance and Time")
print(data[['Distance', 'Time']].std())
print("Max Distance and Time")
print(data[['Distance', 'Time']].max())
print("Collision percentage:")
print(data['Collision'].mean())
print("Count every type:")
print(data['Collision'].value_counts())
print(data['Type'].value_counts())
