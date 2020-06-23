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
from skimage.measure import compare_ssim
import argparse
import pandas as pd
#import imutils

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

DATA_DIR = "record"
model_path_name = "tf_vae"
z_size=64
filelist = os.listdir(DATA_DIR)
vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)


vae.load_json(os.path.join(model_path_name, 'vae_1k.json'))

#obs = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["obs"]
print(len(filelist))
obs = np.load(os.path.join(DATA_DIR, filelist[100]))["obs"]
obs2 = np.load(os.path.join(DATA_DIR, filelist[400]))["obs"]
obs3 = np.load(os.path.join(DATA_DIR, filelist[500]))["obs"]

obs = obs.astype(np.float32)/255.0
obs2 = obs2.astype(np.float32)/255.0
obs3 = obs3.astype(np.float32)/255.0
#frame = random.choice(obs).reshape(1, 128, 128, 3)
frame = obs[70].reshape(1, 128, 128, 3)
frame2 = obs2[30].reshape(1, 128, 128, 3)
frame3 = obs3[20].reshape(1, 128, 128, 3)
#plt.imshow(frame2[0])
#plt.show()

'''Perform interpolation between two classes a and b for any sample x_c.
model: a trained generative model
X: data in the original space with shape: (n_samples, n_features)
labels: array of class labels (n_samples, )
a, b: class labels a and b
x_c: input sample to manipulate (1, n_features)
alpha: scalar for the magnitude and direction of the interpolation
'''
# Encode samples to the latent space
Z_a, Z_b = vae.encode(frame), vae.encode(frame2)
# Find the centroids of the classes a, b in the latent space
z_a_centoid = Z_a.mean(axis=0)
z_b_centoid = Z_b.mean(axis=0)
# The interpolation vector pointing from b -> a
z_b2a = z_a_centoid - z_b_centoid
# Manipulate x_c
z_c = vae.encode(frame3)
alpha = 0.0
z_c_interp = z_c + alpha * z_b2a
interpolation = vae.decode(z_c_interp)
vis = np.hstack((frame[0], interpolation[0], frame2[0]))
#plt.imshow(interpolation[0])
plt.imshow(vis)
plt.show()

'''
z = vae.encode(frame)

#z[0][61] += -3 #Makes a car appear on theleft
#z[0][61] += -3
z1 = z
z2= z
z3= z
z4= z
z5= z
z1[0][0] += 0
reconstruct1 = vae.decode(z1)
z2[0][0] += -2.5
reconstruct2 = vae.decode(z2)
z3[0][0] += -3
reconstruct3 = vae.decode(z3)
z4[0][0] += -3.5
reconstruct4 = vae.decode(z4)
z5[0][0] += -4
reconstruct5 = vae.decode(z5)

vis = np.hstack((reconstruct1[0], reconstruct2[0], reconstruct3[0],reconstruct4[0],reconstruct5[0]))
'''
'''
vis = frame[0]
for i in range(0, 20):
    zz = z
    print(i)
    zz[0][i] += -5
    #print(z[0][0])
    reconstruct = vae.decode(zz)
    vis = np.hstack((vis, reconstruct[0]))
'''
'''
plt.imshow(vis)
plt.show()
'''
#image = (vis*255).astype('uint8')
#path = 'SSIM_examples' + '/mse_' + str(mse_score) + '_ssim' + str(ssim_score) + '.png'
#cv2.imwrite(path, image)
