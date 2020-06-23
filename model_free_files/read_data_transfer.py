import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import csv
import pandas
from scipy.stats import mannwhitneyu
#import seaborn as sns
#from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes, hold
#import pandas.DataFrame as df
#data = pandas.read_csv("data_5_30sec_100cars.csv")
#data = pandas.read_csv("model_free_data/data_1_10sec_50cars.csv")


data1 = pandas.read_csv("final_follow_data/model_free_transfer_test_reward.csv")
data2 = pandas.read_csv("final_follow_data/model_free_scratch_test_reward.csv")

data11 = pandas.read_csv("final_follow_data/model_based_transfer_test_reward.csv")
data22 = pandas.read_csv("final_follow_data/model_based_scratch_test_reward.csv")

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
#Boxplot median distance of mean distance from pilot for each episode model free
list = np.array([data1['Mean_Dist_Pilot'], data2['Mean_Dist_Pilot']])
df = pandas.DataFrame(list.transpose(), columns=['Transfer learning', 'From scratch'])
boxplot = df.boxplot(column=['Transfer learning', 'From scratch'], )
'''
'''
#Boxplot median distance of mean distance from pilot for each episode model based
list = np.array([data11['Mean_Dist_Pilot'], data22['Mean_Dist_Pilot']])
df = pandas.DataFrame(list.transpose(), columns=['Transfer learning', 'From scratch'])
boxplot = df.boxplot(column=['Transfer learning', 'From scratch'], )
'''
'''
#Boxplot mean time spent within range of pilot for model free
list = np.array([data1['Mean_Time']*data1['Time'], data2['Mean_Time']*data2['Time']])
df = pandas.DataFrame(list.transpose(), columns=['Transfer learning', 'From scratch'])
boxplot = df.boxplot(column=['Transfer learning', 'From scratch'], )
'''
'''
#Boxplot mean time spent within range of pilot for model based
list = np.array([data11['Mean_Time']*data11['Time'], data22['Mean_Time']*data22['Time']])
df = pandas.DataFrame(list.transpose(), columns=['Transfer learning', 'From scratch'])
boxplot = df.boxplot(column=['Transfer learning', 'From scratch'], )
'''
'''
list = np.array([data1['Mean_Time']*data1['Time'], data2['Mean_Time']*data2['Time'], data11['Mean_Time']*data11['Time'], data22['Mean_Time']*data22['Time']])
df = pandas.DataFrame(list.transpose(), columns=['Transfer learning\nmodel free', 'From scratch\nmodel free', 'Transfer learning\nmodel based', 'From scratch\nmodel based'])
boxplot = df.boxplot(column=['Transfer learning\nmodel free', 'From scratch\nmodel free', 'Transfer learning\nmodel based', 'From scratch\nmodel based'], )
'''
'''
list = np.array([data1['Mean_Dist_Pilot'], data2['Mean_Dist_Pilot'], data11['Mean_Dist_Pilot'], data22['Mean_Dist_Pilot']])
df = pandas.DataFrame(list.transpose(), columns=['Transfer learning\nmodel free', 'From scratch\nmodel free', 'Transfer learning\nmodel based', 'From scratch\nmodel based'])
boxplot = df.boxplot(column=['Transfer learning\nmodel free', 'From scratch\nmodel free', 'Transfer learning\nmodel based', 'From scratch\nmodel based'], )
'''
#list1 = np.array([data1['Mean_Time']*data1['Time'], data2['Mean_Time']*data2['Time']])
#list2 = np.array([data11['Mean_Time']*data11['Time'], data22['Mean_Time']*data22['Time']])

list1 = [data1['Episode_Reward'], data11['Episode_Reward']]
list2 = [data2['Episode_Reward'], data22['Episode_Reward']]

print("Free, trans:", data1['Episode_Reward'].mean())
print("Free, scratch", data11['Episode_Reward'].mean())

print("Based, trans:", data2['Episode_Reward'].mean())
print("Based, scratch", data22['Episode_Reward'].mean())
#transfer_list1 = np.array(data1['Episode_Reward'])
#scratch_list2 = np.array(data2['Episode_Reward'])

#transfer_list1 = np.array(data11['Mean_Time']*data11['Time'])
#scratch_list2 = np.array(data22['Mean_Time']*data22['Time'])
#stat, p = mannwhitneyu(transfer_list1, scratch_list2)
#print('Statistics=%.3f, p=%.3f' % (stat, p))

#list1 = [data1['Mean_Time']*data1['Time'], data11['Mean_Time']*data11['Time']]
#list2 = [data2['Mean_Time']*data2['Time'], data22['Mean_Time']*data22['Time']]

#list3 = [list1, list2]

ticks = ['Model Free', 'Model Based']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

#bpl = plt.boxplot(list1, positions=np.array(range(len(list1)))*2.0-0.4, sym='', widths=0.6)
#bpr = plt.boxplot(list2, positions=np.array(range(len(list2)))*2.0+0.4, sym='', widths=0.6)

bpl = plt.boxplot(list1, positions=np.array(range(len(list1)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(list2, positions=np.array(range(len(list2)))*2.0+0.4, sym='', widths=0.6)

set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='Transfer learning')
plt.plot([], c='#2C7BB6', label='From scratch')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
#plt.ylim(0, 1)
plt.ylabel('Reward')
plt.tight_layout()
plt.show()


#plt.ylim(0,20)
#plt.ylabel('Distance')
#plt.show()
