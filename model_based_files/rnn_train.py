'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time

from vae.vae import ConvVAE, reset_graph
from rnn.rnn import HyperParams, MDNRNN

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

DATA_DIR = "series"
model_save_path = "tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  return z, action

def default_hps():
  return HyperParams(num_steps=40000,
                     max_seq_len=199,#999, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=65,#33,#35,    # width of our data (32 + 3 actions)
                     output_seq_width=64,#32,    # width of our data is 32
                     rnn_size=512,#256,    # number of rnn cells
                     batch_size=200,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=8,#5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=1, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
max_seq_len = hps_model.max_seq_len

#N_data = len(data_mu) # should be 10k
batch_size = hps_model.batch_size

# save 1000 initial mu and logvars:
#initial_mu = np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist()
#initial_logvar = np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist()
#tmp = np.array([data_mu[i] for i in range(data_mu.shape[0])])
#print(tmp.shape)
"""
for i in range(data_mu.shape[0]):
    tmp = []
    for j in range(data_mu[0].shape[0]):
        tmp.append(data_mu[i][j,:])
    data_mu_tmp.append(tmp)
data_mu_tmp = np.array(data_mu_tmp)
"""
#data_mu_tmp = np.array([[data_mu[i][j,:] for j in range(data_mu[0].shape[0])] for i in range(data_mu.shape[0])])
#data_logvar_tmp = np.array([[data_logvar[i][j,:] for j in range(data_logvar[0].shape[0])] for i in range(data_logvar.shape[0])])
#data_action_tmp = np.array([[data_action[i][j] for j in range(data_action[0].shape[0])] for i in range(data_action.shape[0])])

#index = []
#for i in range(2144):
#    if data_mu[i].shape[0] != 200:
#        index.append(i)
#data_mu = np.delete(data_mu, index)
#data_logvar = np.delete(data_logvar, index)
#data_action = np.delete(data_action, index)
N_data = len(data_mu)
#print(data_mu_tmp.shape)
#print(data_logvar_tmp.shape)
#print(data_action_tmp.shape)
#print(data_mu.shape, data_mu[0][:].shape)
#print(data_mu[0].shape)
#print(data_logvar.shape)
#print(data_logvar[0].shape)
#print(data_action.shape)
#print(data_action[0].shape)
#print(data_mu[0].shape)


data_mu = np.array([[data_mu[i][j,:] for j in range(data_mu[0].shape[0])] for i in range(data_mu.shape[0])])
data_logvar = np.array([[data_logvar[i][j,:] for j in range(data_logvar[0].shape[0])] for i in range(data_logvar.shape[0])])
data_action = np.array([[data_action[i][j] for j in range(data_action[0].shape[0])] for i in range(data_action.shape[0])])


print(data_mu.shape, data_mu[0][:].shape)

initial_mu = np.copy(data_mu[:200, 0,:]*2400).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:200, 0,:]*2400).astype(np.int).tolist()


with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

reset_graph()
rnn = MDNRNN(hps_model)

# train loop:
hps = hps_model
start = time.time()
for local_step in range(hps.num_steps):

  step = rnn.sess.run(rnn.global_step)
  curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

  raw_z, raw_a = random_batch()
  #print(raw_z.shape, raw_a.shape)

  #inputs = np.concatenate((raw_z[:,:-1,:], raw_a[:, :-1,:]), axis=2)
  raw_a = np.expand_dims(raw_a,2)
  #print(raw_a.shape)
  #print(raw_z[:,:,0].shape)
  inputs = np.concatenate((raw_z[:,:-1,:], raw_a[:, :-1, :]), axis=2)
  outputs = raw_z[:, 1:, :] # teacher forcing (shift by one predictions)

  feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
  (train_cost, state, train_step, _) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
  if (step%20==0 and step > 0):
    end = time.time()
    time_taken = end-start
    start = time.time()
    output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
    print(output_log)

# save the model (don't bother with tf checkpoints json all the way ...)
rnn.save_json(os.path.join(model_save_path, "rnn_new.json"))
