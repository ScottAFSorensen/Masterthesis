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
#obs = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["obs"]
#obs = obs.astype(np.float32)/255.0
#print(obs.shape)
#frame = random.choice(obs).reshape(1, 128, 128, 3)
#act = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["action"]
#sequence = np.load(os.path.join(DATA_DIR, random.choice(filelist)))
#obs = sequence["obs"]
#obs = obs.astype(np.float32)/255.0
#frame = obs[0].reshape(1, 128, 128, 3)
#act = sequence["action"]
#print(act.shape)
vae = ConvVAE(z_size=z_size,
              batch_size=1,
              is_training=False,
              reuse=False,
              gpu_mode=False)


vae.load_json(os.path.join(model_path_name, 'vae_1k.json'))

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score

mse_list = []
ssim_list = []
for i in range(0,100):
    obs = np.load(os.path.join(DATA_DIR, random.choice(filelist)))["obs"]
    obs = obs.astype(np.float32)/255.0
    frame = random.choice(obs).reshape(1, 128, 128, 3)
    #plt.imshow(frame[0])
    #plt.show()
    z = vae.encode(frame)
    reconstruct = vae.decode(z)
    vis = np.vstack((frame[0], reconstruct[0]))
    #plt.imshow(vis)
    #plt.show()

    mse_score = mse(frame[0], reconstruct[0])
    #print("MSE: ", mse_score)
    mse_list.append(mse_score)
    ssim_score = ssim(frame[0], reconstruct[0])
    #print("SSIM: ", ssim_score)
    ssim_list.append(ssim_score)
    image = (vis*255).astype('uint8')
    path = 'SSIM_examples' + '/mse_' + str(mse_score) + '_ssim' + str(ssim_score) + '.png'
    cv2.imwrite(path, image)


print(len(ssim_list))
df = pd.DataFrame(np.array([ssim_list]).T, columns=['SSIM'])
#df.columns =['SSIM']
df['MSE'] = mse_list
print('Mean:')
print(df[['SSIM', 'MSE']].mean())
print('STD:')
print(df[['SSIM', 'MSE']].std())
print('Median:')
print(df[['SSIM', 'MSE']].median())
