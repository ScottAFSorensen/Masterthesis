#from tensorflow.python.summary import event_accumulator as ea
from tensorboard.backend.event_processing import event_accumulator as ea
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.signal import savgol_filter

#Load log 1 transfer learning
acc1 = ea.EventAccumulator("final_follow_models/transfer_model_based_follower_logs/")
acc1.Reload()

#Load log 2 from scratch
acc2 = ea.EventAccumulator("final_follow_models/scratch_model_based_follower_logs/")
acc2.Reload()

'''
#Load log 1 transfer learning
acc1 = ea.EventAccumulator("final_follow_models/transfer_model_free_follower_logs/")
acc1.Reload()

#Load log 2 from scratch
acc2 = ea.EventAccumulator("final_follow_models/scratch_model_free_follower_logs/")
acc2.Reload()
'''
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

reward_avg_step_1 = [(s.step) for s in acc1.Scalars('reward_avg')]
reward_avg_value_1 = [(s.value) for s in acc1.Scalars('reward_avg')]
yhat1 = savgol_filter(reward_avg_value_1, 51, 3)

reward_avg_step_2 = [(s.step) for s in acc2.Scalars('reward_avg')]
reward_avg_value_2 = [(s.value) for s in acc2.Scalars('reward_avg')]
yhat2 = savgol_filter(reward_avg_value_2, 51, 3)



#plt.legend(["Model-Free 1", "Model-Free 2", "Model-Free 3", "Model-Free 4", "Model-Free 5", "Average"])
#plt.legend(["Median", "Average", "Max", "Min", "Savgol Average", "Savgol Median", "Savgol Max", "Savgol Min"])
#plt.xlabel('model-based')
#plt.xticks(rotation=10)

#plt.plot(reward_avg_step_1, yhat1)
#plt.plot(reward_avg_step_2, yhat2)


plt.plot(reward_avg_step_1, reward_avg_value_1)
plt.plot(reward_avg_step_2, reward_avg_value_2)

plt.ylim(-0.5, 3)
plt.legend(["Transfer learning", "From scratch"])
plt.ylabel('Reward')
plt.xlabel('Episodes')
plt.show()






# Retrieve images, e. g. first labeled as 'generator'
#img = acc.Images('generator/image/0')
#with open('img_{}.png'.format(img.step), 'wb') as f:
#  f.write(img.encoded_image_string)
