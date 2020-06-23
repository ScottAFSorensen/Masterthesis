import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import csv
import pandas
from scipy.stats import mannwhitneyu
#from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes, hold
#import pandas.DataFrame as df
#data = pandas.read_csv("data_5_30sec_100cars.csv")
#data = pandas.read_csv("model_free_data/data_1_10sec_50cars.csv")


data1 = pandas.read_csv("model_free_data/data_1_10sec_50cars.csv")
data2 = pandas.read_csv("model_free_data/data_2_10sec_50cars.csv")
data3 = pandas.read_csv("model_free_data/data_3_10sec_50cars.csv")
data4 = pandas.read_csv("model_free_data/data_4_10sec_50cars.csv")
data5 = pandas.read_csv("model_free_data/data_5_10sec_50cars.csv")

data11 = pandas.read_csv("model_based_data/model_based_data_1_10sec_50cars.csv")
data22 = pandas.read_csv("model_based_data/model_based_data_2_10sec_50cars.csv")
data33 = pandas.read_csv("model_based_data/model_based_data_3_10sec_50cars.csv")
data44 = pandas.read_csv("model_based_data/model_based_data_4_10sec_50cars.csv")
data55 = pandas.read_csv("model_based_data/model_based_data_5_10sec_50cars.csv")


data1_2 = pandas.read_csv("model_free_data/data_1_20sec_75cars.csv")
data2_2 = pandas.read_csv("model_free_data/data_2_20sec_75cars.csv")
data3_2 = pandas.read_csv("model_free_data/data_3_20sec_75cars.csv")
data4_2 = pandas.read_csv("model_free_data/data_4_20sec_75cars.csv")
data5_2 = pandas.read_csv("model_free_data/data_5_20sec_75cars.csv")

data11_2 = pandas.read_csv("model_based_data/model_based_data_1_20sec_75cars.csv")
data22_2 = pandas.read_csv("model_based_data/model_based_data_2_20sec_75cars.csv")
data33_2 = pandas.read_csv("model_based_data/model_based_data_3_20sec_75cars.csv")
data44_2 = pandas.read_csv("model_based_data/model_based_data_4_20sec_75cars.csv")
data55_2 = pandas.read_csv("model_based_data/model_based_data_5_20sec_75cars.csv")


data1_3 = pandas.read_csv("model_free_data/data_1_30sec_100cars.csv")
data2_3 = pandas.read_csv("model_free_data/data_2_30sec_100cars.csv")
data3_3 = pandas.read_csv("model_free_data/data_3_30sec_100cars.csv")
data4_3 = pandas.read_csv("model_free_data/data_4_30sec_100cars.csv")
data5_3 = pandas.read_csv("model_free_data/data_5_30sec_100cars.csv")

data11_3 = pandas.read_csv("model_based_data/model_based_data_1_30sec_100cars.csv")
data22_3 = pandas.read_csv("model_based_data/model_based_data_2_30sec_100cars.csv")
data33_3 = pandas.read_csv("model_based_data/model_based_data_3_30sec_100cars.csv")
data44_3 = pandas.read_csv("model_based_data/model_based_data_4_30sec_100cars.csv")
data55_3 = pandas.read_csv("model_based_data/model_based_data_5_30sec_100cars.csv")


p_data1 = pandas.read_csv("predict_data/predict_data_1_10sec_50cars.csv")
p_data2 = pandas.read_csv("predict_data/predict_data_2_10sec_50cars.csv")
p_data3 = pandas.read_csv("predict_data/predict_data_3_10sec_50cars.csv")
p_data4 = pandas.read_csv("predict_data/predict_data_4_10sec_50cars.csv")
p_data5 = pandas.read_csv("predict_data/predict_data_5_10sec_50cars.csv")

p_data1_2 = pandas.read_csv("predict_data/predict_data_1_20sec_75cars.csv")
p_data2_2 = pandas.read_csv("predict_data/predict_data_2_20sec_75cars.csv")
p_data3_2 = pandas.read_csv("predict_data/predict_data_3_20sec_75cars.csv")
p_data4_2 = pandas.read_csv("predict_data/predict_data_4_20sec_75cars.csv")
p_data5_2 = pandas.read_csv("predict_data/predict_data_5_20sec_75cars.csv")

p_data1_3 = pandas.read_csv("predict_data/predict_data_1_30sec_100cars.csv")
p_data2_3 = pandas.read_csv("predict_data/predict_data_2_30sec_100cars.csv")
p_data3_3 = pandas.read_csv("predict_data/predict_data_3_30sec_100cars.csv")
p_data4_3 = pandas.read_csv("predict_data/predict_data_4_30sec_100cars.csv")
p_data5_3 = pandas.read_csv("predict_data/predict_data_5_30sec_100cars.csv")


'''
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
'''
'''
list = np.array([data1['Distance'], data2['Distance'], data3['Distance'], data4['Distance'], data5['Distance']])
df = pandas.DataFrame(list.transpose(), columns=['predicted_model-free 1', 'predicted_model-free 2', 'predicted_model-free 3', 'predicted_model-free 4', 'predicted_model-free 5'])
boxplot = df.boxplot(column=['predicted_model-free 1', 'predicted_model-free 2', 'predicted_model-free 3', 'predicted_model-free 4', 'predicted_model-free 5'], )
'''

'''
list = np.array([data11['Distance'], data22['Distance'], data33['Distance'], data44['Distance'], data55['Distance']])
df = pandas.DataFrame(list.transpose(), columns=['model-based 1', 'model-based 2', 'model-based 3', 'model-based 4', 'model-based 5'])
boxplot = df.boxplot(column=['model-based 1', 'model-based 2', 'model-based 3', 'model-based 4', 'model-based 5'])
'''
'''
list2 = np.array([data1['Time'], data2['Time'], data3['Time'], data4['Time'], data5['Time']])
df2 = pandas.DataFrame(list2.transpose(), columns=['model-free 1', 'model-free 2', 'model-free 3', 'model-free 4', 'model-free 5'])
boxplot = df2.boxplot(column=['model-free 1', 'model-free 2', 'model-free 3', 'model-free 4', 'model-free 5'])
'''

'''
list2 = np.array([data11['Time'], data22['Time'], data33['Time'], data44['Time'], data55['Time']])
df2 = pandas.DataFrame(list2.transpose(), columns=['model-based 1', 'model-based 2', 'model-based 3', 'model-based 4', 'model-based 5'])
boxplot = df2.boxplot(column=['model-based 1', 'model-based 2', 'model-based 3', 'model-based 4', 'model-based 5'])
'''

list3 = np.array([data1['Collision'].mean(), data2['Collision'].mean(), data3['Collision'].mean(), data4['Collision'].mean(), data5['Collision'].mean()])
list4 = np.array([data11['Collision'].mean(), data22['Collision'].mean(), data33['Collision'].mean(), data44['Collision'].mean(), data55['Collision'].mean()])
list5 = np.array([p_data1['Collision'].mean(), p_data2['Collision'].mean(), p_data3['Collision'].mean(), p_data4['Collision'].mean(), p_data5['Collision'].mean()])

list3_2 = np.array([data1_2['Collision'].mean(), data2_2['Collision'].mean(), data3_2['Collision'].mean(), data4_2['Collision'].mean(), data5_2['Collision'].mean()])
list4_2 = np.array([data11_2['Collision'].mean(), data22_2['Collision'].mean(), data33_2['Collision'].mean(), data44_2['Collision'].mean(), data55_2['Collision'].mean()])
list5_2 = np.array([p_data1_2['Collision'].mean(), p_data2_2['Collision'].mean(), p_data3_2['Collision'].mean(), p_data4_2['Collision'].mean(), p_data5_2['Collision'].mean()])

list3_3 = np.array([data1_3['Collision'].mean(), data2_3['Collision'].mean(), data3_3['Collision'].mean(), data4_3['Collision'].mean(), data5_3['Collision'].mean()])
list4_3 = np.array([data11_3['Collision'].mean(), data22_3['Collision'].mean(), data33_3['Collision'].mean(), data44_3['Collision'].mean(), data55_3['Collision'].mean()])
list5_3 = np.array([p_data1_3['Collision'].mean(), p_data2_3['Collision'].mean(), p_data3_3['Collision'].mean(), p_data4_3['Collision'].mean(), p_data5_3['Collision'].mean()])

'''
list3 = np.array([data1['Collision'].mean(), data2['Collision'].mean(), data3['Collision'].mean(), data4['Collision'].mean(), data5['Collision'].mean()])
list4 = np.array([data11['Collision'].mean(), data22['Collision'].mean(), data33['Collision'].mean(), data44['Collision'].mean(), data55['Collision'].mean()])
list5 = np.array([p_data1['Collision'].mean(), p_data2['Collision'].mean(), p_data3['Collision'].mean(), p_data4['Collision'].mean(), p_data5['Collision'].mean()])

list3_2 = np.array([data1_2['Collision'].mean(), data2_2['Collision'].mean(), data3_2['Collision'].mean(), data4_2['Collision'].mean(), data5_2['Collision'].mean()])
list4_2 = np.array([data11_2['Collision'].mean(), data22_2['Collision'].mean(), data33_2['Collision'].mean(), data44_2['Collision'].mean(), data55_2['Collision'].mean()])
list5_2 = np.array([p_data1_2['Collision'].mean(), p_data2_2['Collision'].mean(), p_data3_2['Collision'].mean(), p_data4_2['Collision'].mean(), p_data5_2['Collision'].mean()])

list3_3 = np.array([data1_3['Collision'].mean(), data2_3['Collision'].mean(), data3_3['Collision'].mean(), data4_3['Collision'].mean(), data5_3['Collision'].mean()])
list4_3 = np.array([data11_3['Collision'].mean(), data22_3['Collision'].mean(), data33_3['Collision'].mean(), data44_3['Collision'].mean(), data55_3['Collision'].mean()])
list5_3 = np.array([p_data1_3['Collision'].mean(), p_data2_3['Collision'].mean(), p_data3_3['Collision'].mean(), p_data4_3['Collision'].mean(), p_data5_3['Collision'].mean()])
#list5 = np.array([list3, list4])
'''


list55 = np.array([list3, list4, list3_2, list4_2, list3_3, list4_3])
list6 = np.array([list3, list3_2, list3_3])
list7 = np.array([list4, list4_2, list4_2])
#list8 = np.array([list3_3, list4_3, list5_3])
#df3 = pandas.DataFrame(list55.transpose(), columns=['model-free systems\neasy', 'model-based systems\neasy', 'model-free systems\nmedium', 'model-based systems\nmedium', 'model-free systems\nhard', 'model-based systems\nhard' ])
#bp = df3.boxplot(column=['model-free systems\neasy', 'model-based systems\neasy', 'model-free systems\nmedium', 'model-based systems\nmedium', 'model-free systems\nhard', 'model-based systems\nhard' ])
df_free = pandas.DataFrame(list6.transpose(), columns=['model-free systems\neasy', 'model-free systems\nmedium', 'model-free systems\nhard'])
df_based = pandas.DataFrame(list7.transpose(), columns=['model-based systems\neasy', 'model-based systems\nmedium', 'model-based systems\nhard'])
#bp1 = df3.boxplot(column=['model-free systems\neasy', 'model-free systems\nmedium', 'model-free systems\nhard', ])
#bp2 = df3.boxplot(column=['model-based systems\neasy', 'model-based systems\nmedium', 'model-based systems\nhard' ])
#fig, (ax1, ax2) = plt.subplots(ncolumns=2)

bp1 = df_free.plot(kind='box',
             color=dict(boxes='b', whiskers='b', medians='b', caps='b'),
             boxprops=dict(linestyle='-', linewidth=1.5),
             flierprops=dict(linestyle='-', linewidth=1.5),
             medianprops=dict(linestyle='-', linewidth=1.5),
             whiskerprops=dict(linestyle='-', linewidth=1.5),
             capprops=dict(linestyle='-', linewidth=1.5),
             showfliers=False, return_type='axes', grid=True, rot=0)

bp2 = df_based.plot(kind='box',
             color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
             boxprops=dict(linestyle='-', linewidth=1.5),
             flierprops=dict(linestyle='-', linewidth=1.5),
             medianprops=dict(linestyle='-', linewidth=1.5),
             whiskerprops=dict(linestyle='-', linewidth=1.5),
             capprops=dict(linestyle='-', linewidth=1.5),
             showfliers=False, ax=bp1, grid=True, rot=0)

#df3 = pandas.DataFrame(list5.transpose(), columns=['model-free systems\n10 seconds, 50 cars', 'model-based systems\n10 seconds 50 cars', 'model-free systems\n20 seconds, 75 cars', 'model-based systems\n20 seconds 75 cars', 'model-free systems\n30 seconds, 100 cars', 'model-based systems\n30 seconds 100 cars' ])
#boxplot = df3.boxplot(column=['model-free systems\n10 seconds, 50 cars', 'model-free systems\n20 seconds, 75 cars', 'model-free systems\n30 seconds, 100 cars', 'model-based systems\n10 seconds 50 cars','model-based systems\n20 seconds 75 cars', 'model-based systems\n30 seconds 100 cars'])

#df3 = pandas.DataFrame(list8.transpose(), columns=['model-free systems', 'model-based systems', 'predicted-model-free\n systems'])
#boxplot = df3.boxplot(column=['predicted-model-free\n systems', 'model-free systems', 'model-based systems'])

#plt.xlabel('xlabel')
#plt.xticks(rotation=10)

#fig, (ax1, ax2) = plt.subplots(nrows=2)
#plt.ylim(0, 35)
plt.ylabel('Collision rate')







plt.show()

'''
lists1 = np.array([data1_3['Collision'], data2_3['Collision'], data3_3['Collision'], data4_3['Collision'], data5_3['Collision']])
list1 = lists1.flatten()
#print(list1)

lists2 = np.array([data11_3['Collision'], data22_3['Collision'], data33_3['Collision'], data44_3['Collision'], data55_3['Collision']])
list2 =lists2.flatten()
#print(list2)

lists3 = np.array([p_data1_3['Collision'], p_data2_3['Collision'], p_data3_3['Collision'], p_data4_3['Collision'], p_data5_3['Collision']])
list3 = lists3.flatten()

stat, p = mannwhitneyu(list1, list3)
print('Statistics=%.3f, p=%.3f' % (stat, p))
'''
