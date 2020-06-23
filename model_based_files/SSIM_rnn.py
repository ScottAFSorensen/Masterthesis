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
import matplotlib.pyplot as plt
from math import log10, sqrt
from scipy.spatial import distance
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

#print(act.shape)
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

vae.load_json(os.path.join(model_path_name, 'vae_1k.json'))
rnn.load_json(os.path.join(model_path_name2, 'rnn.json'))

sess = rnn.sess
#initial_state = rnn.initial_state
initial_state = rnn_init_state(rnn)
#init_z = vae.encode(frame)
#rnn_state = rnn_next_state(rnn, init_z, np.asarray(1), initial_state)

#sess = rnn.sess
#initial_state = rnn.initial_state
#print(z)
#print(z.shape)
#reconstruct = vae.decode(z)
#print(reconstruct.shape)

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

def PSNR(original, compressed):

    mse = np.mean((original - compressed) ** 2)
    #if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
    #    return 100
    max_pixel = 1#255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
'''
mse_list = []
ssim_list = []
for i in range(0,100):
    sequence = np.load(os.path.join(DATA_DIR, random.choice(filelist)))
    obs = sequence["obs"]
    obs = obs.astype(np.float32)/255.0
    ground_truth = obs[20].reshape(1, 128, 128, 3)
    frame = obs[0].reshape(1, 128, 128, 3)
    act = sequence["action"]
    z = vae.encode(frame)
    rnn_state = rnn_next_state(rnn, z, np.asarray(1), initial_state)
    strokes = sample_sequence(sess, rnn, hps_model, z, act, temperature=1.0, seq_len=50)
    pred_z = strokes[20]
    pred_zz = np.expand_dims(pred_z, axis=0)
    pred_frame = vae.decode(pred_zz)
    mse_score = mse(ground_truth[0], pred_frame[0])
    #print("MSE: ", mse_score)
    mse_list.append(mse_score)
    ssim_score = ssim(ground_truth[0], pred_frame[0])
    #print("SSIM: ", ssim_score)
    ssim_list.append(ssim_score)
    #vis = np.vstack((frame[0], ground_truth[0], pred_frame[0]))
    #plt.imshow(vis)
    #plt.show()
    #image = (vis*255).astype('uint8')
    #path = 'SSIM_examples_rnn/' + 'mse_' + str(mse_score) + '_ssim' + str(ssim_score) + '.png'
    #cv2.imwrite(path, image)
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
#cv2.imshow("frame", frame[0])
#cv2.imshow("ground_truth", ground_truth[0])
#cv2.imshow("pred_frame", pred_frame[0])
#cv2.waitKey(1)
'''
'''
sequence2 = np.load(os.path.join(DATA_DIR, random.choice(filelist)))
img1 = sequence2["obs"]
sequence3 = np.load(os.path.join(DATA_DIR, random.choice(filelist)))
img2 = sequence3["obs"]
print(ssim(img1[0], img2[0]))
'''
'''
ssim_list = []
mse_list = []
ssim_start_list = []
mse_start_list = []
#psnr_list = []
df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
#df3 = pd.DataFrame()
for j in range(1,4):
    for i in range(0,100):
        sequence = np.load(os.path.join(DATA_DIR, random.choice(filelist)))
        obs = sequence["obs"]
        obs = obs.astype(np.float32)/255.0
        ground_truth = obs[j*10].reshape(1, 128, 128, 3)
        frame = obs[0].reshape(1, 128, 128, 3)
        act = sequence["action"]
        z = vae.encode(frame)
        recon_start = vae.decode(z)
        rnn_state = rnn_next_state(rnn, z, np.asarray(1), initial_state)
        strokes = sample_sequence(sess, rnn, hps_model, z, act, temperature=1.0, seq_len=200)
        pred_z = strokes[j*10]
        pred_zz = np.expand_dims(pred_z, axis=0)
        pred_frame = vae.decode(pred_zz)

        mse_score = mse(ground_truth[0], pred_frame[0])
        mse_start_score = mse(ground_truth[0], recon_start[0])
        mse_list.append(mse_score)
        mse_start_list.append(mse_start_score)

        ssim_score = ssim(ground_truth[0], pred_frame[0])
        ssim_start_score = ssim(ground_truth[0], recon_start[0])
        ssim_list.append(ssim_score)
        ssim_start_list.append(ssim_start_score)
        #psnr_score = PSNR(ground_truth[0], pred_frame[0])
        #psnr_list.append(psnr_score)
    name = str(j*10)
    #print(len(ssim_list))

    df[name] = ssim_list#np.array([ssim_list])
    df2[name] = mse_list
    df3[name+'start'] = ssim_start_list
    df4[name+'start'] = mse_start_list
    #df = pd.DataFrame(np.array([ssim_list]).T, columns=[name])
    ssim_list = []
    mse_list = []
    ssim_start_list = []
    mse_start_list = []
    #psnr_list = []
print("SSIM mean:")
print(df.mean())
print(df3.mean())
print("MSE mean:")
print(df2.mean())
print(df4.mean())
'''
'''
df.mean().plot( kind = 'line')
plt.show()
df2.mean().plot( kind = 'line')
#df2.mean().plot(kind = 'bar')
#plt.ylim(0.85, 0.95)
plt.show()
'''




df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
euc_distance_list = []
for j in range(1,2):
    for i in range(0,100):
        sequence = np.load(os.path.join(DATA_DIR, random.choice(filelist)))
        obs = sequence["obs"]
        obs = obs.astype(np.float32)/255.0
        ground_truth = obs[10].reshape(1, 128, 128, 3)
        frame = obs[0].reshape(1, 128, 128, 3)
        act = sequence["action"]
        z = vae.encode(frame)
        ground_z = vae.encode(ground_truth)
        ground_zz = np.expand_dims(ground_z, axis=0)
        rnn_state = rnn_next_state(rnn, z, np.asarray(1), initial_state)
        strokes = sample_sequence(sess, rnn, hps_model, z, act, temperature=1.0, seq_len=200)
        pred_z = strokes[10]
        pred_zz = np.expand_dims(pred_z, axis=0)
        dist = np.linalg.norm(pred_zz-ground_zz)
        euc_distance_list.append(dist)



    df['euc'] = euc_distance_list#np.array([ssim_list])
    #df2[name] = mse_list
    #df3[name+'start'] = ssim_start_list
    #df4[name+'start'] = mse_start_list
    #df = pd.DataFrame(np.array([ssim_list]).T, columns=[name])
    #ssim_list = []
    #mse_list = []
    #ssim_start_list = []
    #mse_start_list = []
    #psnr_list = []
print("euc distance mean:")
print(df.mean())



'''
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
'''
