#from tensorflow.python.summary import event_accumulator as ea
from tensorboard.backend.event_processing import event_accumulator as ea
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.signal import savgol_filter
#Load log 1
'''
acc1 = ea.EventAccumulator("model_free_logs/model_free_logs_1/")
acc1.Reload()
#Load log 2
acc2 = ea.EventAccumulator("model_free_logs/model_free_logs_2/")
acc2.Reload()
#Load log 3
acc3 = ea.EventAccumulator("model_free_logs/model_free_logs_3/")
acc3.Reload()
#Load log 4
acc4 = ea.EventAccumulator("model_free_logs/model_free_logs_4/")
acc4.Reload()
#Load log 5
acc5 = ea.EventAccumulator("model_free_logs/model_free_logs_5/")
acc5.Reload()
'''
acc1 = ea.EventAccumulator("model_based_logs/model_based_log_1/")
acc1.Reload()
#Load log 2
acc2 = ea.EventAccumulator("model_based_logs/model_based_log_2/")
acc2.Reload()
#Load log 3
acc3 = ea.EventAccumulator("model_based_logs/model_based_log_3/")
acc3.Reload()
#Load log 4
acc4 = ea.EventAccumulator("model_based_logs/model_based_log_4/")
acc4.Reload()
#Load log 5
acc5 = ea.EventAccumulator("model_based_logs/model_based_log_5/")
acc5.Reload()

# Print tags of contained entities, use these names to retrieve entities as below
#print(acc1.Tags())
"""
# E. g. get all values and steps of a scalar called 'l2_loss'
#loss = [(s.step, s.value) for s in acc.Scalars('loss')]
loss_step = [(s.step) for s in acc.Scalars('loss')]
loss_value = [(s.value) for s in acc.Scalars('loss')]
#print(loss)
#print(loss[0])

plt.plot(loss_step, loss_value)
plt.ylabel('loss')
plt.xlabel('episodes')
plt.show()
"""
'''
reward_avg_step_1 = [(s.step) for s in acc1.Scalars('reward_avg')]
reward_avg_value_1 = [(s.value) for s in acc1.Scalars('reward_avg')]
yhat1 = savgol_filter(reward_avg_value_1, 51, 3)

reward_avg_step_2 = [(s.step) for s in acc2.Scalars('reward_avg')]
reward_avg_value_2 = [(s.value) for s in acc2.Scalars('reward_avg')]
yhat2 = savgol_filter(reward_avg_value_2, 51, 3)

reward_avg_step_3 = [(s.step) for s in acc3.Scalars('reward_avg')]
reward_avg_value_3 = [(s.value) for s in acc3.Scalars('reward_avg')]
yhat3 = savgol_filter(reward_avg_value_3, 51, 3)

reward_avg_step_4 = [(s.step) for s in acc4.Scalars('reward_avg')]
reward_avg_value_4 = [(s.value) for s in acc4.Scalars('reward_avg')]
yhat4 = savgol_filter(reward_avg_value_4, 51, 3)

reward_avg_step_5 = [(s.step) for s in acc5.Scalars('reward_avg')]
reward_avg_value_5 = [(s.value) for s in acc5.Scalars('reward_avg')]
yhat5 = savgol_filter(reward_avg_value_5, 51, 3)
'''

reward_avg_step_1 = [(s.step) for s in acc1.Scalars('collision_percent')]
reward_avg_value_1 = [(s.value) for s in acc1.Scalars('collision_percent')]
yhat1 = savgol_filter(reward_avg_value_1, 51, 3)

reward_avg_step_2 = [(s.step) for s in acc2.Scalars('collision_percent')]
reward_avg_value_2 = [(s.value) for s in acc2.Scalars('collision_percent')]
yhat2 = savgol_filter(reward_avg_value_2, 51, 3)

reward_avg_step_3 = [(s.step) for s in acc3.Scalars('collision_percent')]
reward_avg_value_3 = [(s.value) for s in acc3.Scalars('collision_percent')]
yhat3 = savgol_filter(reward_avg_value_3, 51, 3)

reward_avg_step_4 = [(s.step) for s in acc4.Scalars('collision_percent')]
reward_avg_value_4 = [(s.value) for s in acc4.Scalars('collision_percent')]
yhat4 = savgol_filter(reward_avg_value_4, 51, 3)

reward_avg_step_5 = [(s.step) for s in acc5.Scalars('collision_percent')]
reward_avg_value_5 = [(s.value) for s in acc5.Scalars('collision_percent')]
yhat5 = savgol_filter(reward_avg_value_5, 51, 3)
#Average

avg_1 = np.asarray(reward_avg_value_1)
avg_2 = np.asarray(reward_avg_value_2)
avg_3 = np.asarray(reward_avg_value_3)
avg_4 = np.asarray(reward_avg_value_4)
avg_5 = np.asarray(reward_avg_value_5)

avg_tot= (avg_1+avg_2+avg_3+avg_4+avg_5) / 5.0
yhat_avg = savgol_filter(avg_tot, 51, 3)

stack = np.stack((avg_1, avg_2, avg_3, avg_4, avg_5))
#print(stack.shape)
median = np.median(stack, axis=0)
yhat_median = savgol_filter(median, 51, 3)

max = stack.max(axis=0)#np.maximum(avg_1, avg_2, avg_3, avg_4, avg_5)
yhat_max = savgol_filter(max, 51, 3)

min = stack.min(axis=0)
yhat_min = savgol_filter(min, 51, 3)

#plt.plot(reward_avg_step_5, median)
#plt.plot(reward_avg_step_5, avg_tot)
#plt.plot(reward_avg_step_5, max)
#plt.plot(reward_avg_step_5, min)
#plt.plot(reward_avg_step_5, yhat_avg)
plt.plot(reward_avg_step_5, yhat_median)
plt.plot(reward_avg_step_5, yhat_max)
plt.plot(reward_avg_step_5, yhat_min)
"""
plt.plot(reward_avg_step_1, yhat1)
plt.plot(reward_avg_step_2, yhat2)
plt.plot(reward_avg_step_3, yhat3)
plt.plot(reward_avg_step_4, yhat4)
plt.plot(reward_avg_step_5, yhat5)
"""
#plt.legend(["Model-Free 1", "Model-Free 2", "Model-Free 3", "Model-Free 4", "Model-Free 5", "Average"])
#plt.legend(["Median", "Average", "Max", "Min", "Savgol Average", "Savgol Median", "Savgol Max", "Savgol Min"])
#plt.xlabel('model-based')
#plt.xticks(rotation=10)

plt.ylim(0, 1)
plt.legend(["Median", "Max", "Min"])
plt.ylabel('Collision rate')
plt.xlabel('Episodes')
plt.show()






# Retrieve images, e. g. first labeled as 'generator'
#img = acc.Images('generator/image/0')
#with open('img_{}.png'.format(img.step), 'wb') as f:
#  f.write(img.encoded_image_string)
