import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cv2
import time
from collections import namedtuple
from rnn.rnn import HyperParams, MDNRNN, sample_sequence, rnn_next_state, get_pi_idx

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

DATA_DIR = "record"
model_path_name = "tf_vae"
model_path_name2 = "tf_rnn"
z_size=64
filelist = os.listdir(DATA_DIR)
#obs = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["obs"]
#obs = obs.astype(np.float32)/255.0
#print(obs.shape)
#frame = random.choice(obs).reshape(1, 128, 128, 3)
#act = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["action"]
sequence = np.load(os.path.join(DATA_DIR, random.choice(filelist)))
obs = sequence["obs"]
obs = obs.astype(np.float32)/255.0
frame = obs[0].reshape(1, 128, 128, 3)
act = sequence["action"]
print(act.shape)
vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)

HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])
def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=1,#999, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=65,#33,#35,    # width of our data (32 + 3 actions)
                     output_seq_width=64,#32,    # width of our data is 32
                     rnn_size=512,#256,    # number of rnn cells
                     batch_size=1,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=8,   # number of mixtures in MDN
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
                     is_training=0)
hps_model = default_hps()
rnn = MDNRNN(hps_model, gpu_mode=False, reuse=True)

vae.load_json(os.path.join(model_path_name, 'vae.json'))
rnn.load_json(os.path.join(model_path_name2, 'rnn.json'))

sess = rnn.sess
#initial_state = rnn.initial_state
initial_state = rnn_init_state(rnn)
init_z = vae.encode(frame)
#print(np.random.randn(64))
#print(np.random.randn(64)*np.sqrt(1.15))
rnn_state = rnn_next_state(rnn, init_z, np.asarray(1), initial_state)


"""
strokes = sample_sequence(sess, rnn, hps_model, init_z, act, temperature=0.1, seq_len=200)
#print(strokes.shape)
frame = strokes[0]
#sess = rnn.sess
#initial_state = rnn.initial_state
#print(z)
#print(z.shape)
#reconstruct = vae.decode(z)
#print(reconstruct.shape)



for i in range(0,99):
    if not os.path.isdir('images/' + str(i)):
        os.makedirs('images/' + str(i))
        folder = str(i)
        break
"""
"""
#os.makedirs('images/' + folder)
for i in range(0,200):
    frame = strokes[i]
    enc = vae.encode(obs[i].reshape(1, 128, 128, 3))
    dec = vae.decode(enc)
    z = np.expand_dims(frame, axis=0)
    reconstruct = vae.decode(z)
    #resized_reconstruct = cv2.resize(reconstruct[0], (300, 300))
    #cv2.imshow("VAE reconstruction resized", resized_reconstruct)
    #cv2.imshow("Original sequence", obs[i])
    #cv2.imshow("RNN prediction", reconstruct[0])
    #print(dec[0].shape)
    #print(reconstruct[0].shape)
    vis = np.concatenate((obs[i], dec[0], reconstruct[0]), axis=1)
    image = (vis*255).astype('uint8')
    path = 'images/' + folder + '/' + str(i) + '.png'
    cv2.imwrite(path, image)
    cv2.imshow("Original sequence and RNN predicted sequence", vis)
    cv2.waitKey(1)
    time.sleep(0.02)

"""
"""
for i in range(0,200):
    frame = obs[i].reshape(1, 128, 128, 3)
    init_z = vae.encode(frame)
    strokes = sample_sequence(sess, rnn, hps_model, init_z, act, temperature=0.1, seq_len=50)
    pred_frame = strokes[15]
    z = np.expand_dims(pred_frame, axis=0)
    reconstruct = vae.decode(z)
    #print(reconstruct[0].shape)
    vis = np.concatenate((obs[i], reconstruct[0]), axis=1)
    image = (vis*255).astype('uint8')
    path = 'images/' + folder + '/' + str(i) + '.png'
    cv2.imwrite(path, image)
    cv2.imshow("Original sequence and RNN predicted sequence", vis)
    cv2.waitKey(1)
    #time.sleep(0.02)
"""