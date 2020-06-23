'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems
import tensorflow as tf
import random
import numpy as np


np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
import keras.backend.tensorflow_backend as backend
from vae_seg import ConvVAE, reset_graph

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Hyperparameters for ConvVAE
z_size=64#128#32
batch_size = 50
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 100
DATA_DIR = "record_seg2"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    #print(filename) #Use to find corrupted files.
    raw_data = np.load(os.path.join("record_seg2", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1200 == 0):
      print("loading file", i)
  return  total_length

#def create_dataset(filelist, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
def create_dataset(filelist, N=1000, M=200):
  #data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8) #For 64x64 images
  data = np.zeros((M*N, 128, 128, 3), dtype=np.uint8) #For 128x128 images
  segm = np.zeros((M*N, 128, 128), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record_seg2", filename))['obs']
    raw_seg = np.load(os.path.join("record_seg2", filename))['seg']
    #raw_data.close()
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      segm = segm[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    segm[idx:idx+l] = raw_seg
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  return data, segm

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:5000]
print("check total number of images:", count_length_of_filelist(filelist))

obs_dataset, seg_dataset = create_dataset(filelist)
#print(obs_dataset.shape)
#dataset = np.column_stack((obs_dataset, seg_dataset))
#print(dataset.shape)
# split into batches:
total_length = len(obs_dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

reset_graph()
#print("test0")
vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

vae.load_json()
#print("test0.0")
# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):

  #np.random.shuffle(dataset)
  #np.random.shuffle(np.transpose(dataset))
  #print(dataset.shape)
  c = list(zip(obs_dataset, seg_dataset))
  random.shuffle(c)
  obs_dataset, seg_dataset = zip(*c)
  obs_dataset = np.asarray(obs_dataset)
  seg_dataset = np.asarray(seg_dataset)
  for idx in range(num_batches):

    obs_batch = obs_dataset[idx*batch_size:(idx+1)*batch_size]
    seg_batch = seg_dataset[idx*batch_size:(idx+1)*batch_size]
    #print(batch.shape)

    obs = obs_batch.astype(np.float)/255.0
    seg = seg_batch
    #dummy = np.ones((9,128,128,3)) * (np.expand_dims(seg, 3) == 1)*9999
    dummy = np.zeros((batch_size,128,128,3))

    #seg = np.expand_dims(seg, 3)

    dummy[(seg==0)] = dummy[(seg == 0)]+3.0#0.02 #unlabelled
    dummy[(seg==1)] = dummy[(seg == 1)]+1.0#0.02 #building
    dummy[(seg==2)] = dummy[(seg == 2)]+1.0#0.02 #fence
    dummy[(seg==3)] = dummy[(seg == 3)]+3.0#0.02 #other
    dummy[(seg==4)] = dummy[(seg == 4)]+10.0#0.0 #pedestrian
    dummy[(seg==5)] = dummy[(seg == 5)]+7.0#0.02 #pole
    dummy[(seg==6)] = dummy[(seg == 6)]+1.0#0.02 #road line
    dummy[(seg==7)] = dummy[(seg == 7)]+1.0#0.02 #road
    dummy[(seg==8)] = dummy[(seg == 8)]+1.0#0.02 #sidewalk
    dummy[(seg==9)] = dummy[(seg == 9)]+3.0#0.02 #vegetation
    dummy[(seg==10)] = dummy[(seg == 10)]+10.0#0.8 #car
    dummy[(seg==11)] = dummy[(seg == 11)]+3.0#0.02 #wall
    dummy[(seg==12)] = dummy[(seg == 12)]+3.0#0.02 #traffic sign
    #print(dummy.shape)
    #print("hei")
    #print(dummy[0,:16,:16])



    feed = {vae.x: obs, vae.s: dummy}
    #feed_seg = {vae.s: seg,}
    #print(vae._get_x())
    #feed_seg = {vae.s: seg,}

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)


    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss, kl_loss)
    if ((train_step+1) % 5000 == 0):
      vae.save_json("tf_vae/vae_2k.json")

# finished, final model:
print("training finished, saving model")
vae.save_json("tf_vaeg/vae_2k.json")
