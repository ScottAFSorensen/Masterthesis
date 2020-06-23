import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from model_based_follow import CarEnv, MEMORY_FRACTION
import spawn_npc
#import pandas
import csv
from vae.vae import ConvVAE, reset_graph
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
from collections import namedtuple
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import keras
#MODEL_PATH = 'model_based_models/model_based_5.model'

#MODEL_PATH = 'final_follow_models/transfer_model_based_follower.model'
MODEL_PATH = 'final_follow_models/scratch_model_based_follower.model'
SPAWN_NPC = False
HD_CAM = True
csv_name = "model_based_scratch_test_reward.csv"
with open(csv_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Collision", "Type", "Distance", "Time", "Mean_Dist_Pilot", "Mean_Time", "Episode_Reward"])

if __name__ == '__main__':

    model_path_vae = "tf_vae"
    model_path_rnn = "tf_rnn"
    z_size = 64#32

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
                         rnn_size=512,    # number of rnn cells
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

    vae = ConvVAE(z_size=z_size, batch_size=1, is_training=False, reuse=False, gpu_mode=False)
    vae.load_json(os.path.join(model_path_vae, 'vae_1k.json'))
    rnn = MDNRNN(hps_model, gpu_mode=False, reuse=True)
    rnn.load_json(os.path.join(model_path_rnn, 'rnn.json'))


    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Load the model
    model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    if SPAWN_NPC:
        client2 = spawn_npc.main()
    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    #model.predict((np.ones((1, env.im_height, env.im_width, 3)))
    #print(np.ones(576))
    model.predict(np.ones((1, 576)))


    c = 0
    episode_count = 0
    # Loop over episodes
    while True:
        c += 1
        print('Restarting episode')
        episode_count += 1
        if SPAWN_NPC:
            if episode_count % 5 == 0:
                episode_count = 0
                spawn_npc.destroyAll(client2)
                client2 = spawn_npc.main()
                # Error check:
                if client2 == None:
                    print("SOMETHINH IS WONRTB")

        # Reset environment and get initial state
        current_state = env.reset()
        #print(current_state)
        sess = rnn.sess
        rnn_state = rnn_init_state(rnn)
        current_state_z = current_state.astype(np.float32)/255.0
        current_state_z = current_state_z.reshape(1, 128, 128, 3)
        z = vae.encode(current_state_z)
        #Concatenates hidden states from rnn with encoded z vector
        input_hz = rnn_output(rnn_state, z, 4)
        #Normalize values between -1 and 1
        #input_hz = (2* ((input_hz - np.min(input_hz)) / (np.max(input_hz) - np.min(input_hz)))) - 1
        #Normalize values between 0 and 1
        input_hz = (input_hz-min(input_hz))/(max(input_hz)-min(input_hz))


        env.collision_hist = []
        episode_reward = 0
        done = False
        start_time = time.time()
        collision = 0
        step = 0
        collision_type = "no collision"
        pilot_dist_list = []
        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            if HD_CAM == False:
                #cv2.imshow(f'Agent - preview', current_state)
                resized_current_state = cv2.resize(current_state, (640, 480))
                cv2.imshow("Agent - resized network input", resized_current_state)
                #cv2.imshow("Agent - preview", current_state)
                cv2.waitKey(1)

            # Predict an action based on current observation space
            #qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            #print(input_hz.shape)
            qs = model.predict(np.expand_dims(input_hz, axis=0))
            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            #new_state, reward, done, _ = env.step(action)
            if HD_CAM:
                new_state, reward, done, _, third_person = env.step(action)
                cv2.imshow("HD CAM", third_person)
                cv2.waitKey(1)
            else:
                new_state, reward, done, _ = env.step(action)
            # Set current step for next loop iteration
            episode_reward += reward
            new_state_z = new_state.astype(np.float32)/255.0
            new_state_z = new_state_z.reshape(1, 128, 128, 3)
            z = vae.encode(new_state_z)
            rnn_state = rnn_next_state(rnn, z, np.asarray(action), rnn_state)
            new_input_hz = rnn_output(rnn_state, z, 4)
            #Normalize between -1 and 1
            #new_input_hz = (2 * ((new_input_hz - np.min(new_input_hz)) / (np.max(new_input_hz) - np.min(new_input_hz)))) - 1
            #Normalize between 0 and 1
            new_input_hz = (new_input_hz-min(new_input_hz))/(max(new_input_hz)-min(new_input_hz))
            #agent.update_replay_memory((input_hz, action, reward, new_input_hz, done))
            current_state = new_state
            input_hz = new_input_hz
            step+= 1
            #current_state = new_state

            #print(env.distance)
            pilot_dist_list.append(env.pilot_distance)

            # If done - agent crashed, break an episode
            if done:
                print(env.distance2)
                print(round((time.time() - start_time) % 60))
                range_list = np.asarray(pilot_dist_list)
                mean_range = range_list.mean()
                percentage_in_range = np.count_nonzero((range_list >= 0) & (range_list < 20))/range_list.size
                print("% in range: ",percentage_in_range)
                if (len(env.collision_hist) > 0):
                    collision = 1

                    coll_object = env.collision_hist[0].other_actor
                    coll_id = coll_object.type_id
                    #print(coll_id)
                    if(coll_id[0:7] == "vehicle"):
                        #print("You crashed in a car, 1")
                        collision_type = 'vehicle'#1
                    elif(coll_id[0:15]== "static.building" or coll_id[0:11] == "static.wall"):
                        #print("You crashed in a wall or building, 2")
                        collision_type = 'building_wall'#2
                    elif(coll_id[0:7] == "traffic" or coll_id == "static.pole" or coll_id == "static.vegetation"):
                        #print("You crashed in a traffic light, a pole or a tree, 3")
                        collision_type = 'pole'#3
                    #elif(coll_id[0:6] == "walker"):
                    #    print("You crashed in a pedestrian. R.I.P.")
                    else:
                        #print("You crashed in an unknown object, 4")#Unkown is tunnel entrance, construction markers, pots et
                        collision_type = 'unknown'#4

                print(collision)
                print(collision_type)
                if (round((time.time() - start_time)) % 60) > 1:
                    with open(csv_name, 'a+', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([collision, collision_type, env.distance2, round((time.time() - start_time) % 60), mean_range, percentage_in_range, episode_reward])

                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            #print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')
            #print("Agent: {} FPS | Action: [{}, {} , {}] {}".format(len(fps_counter)/sum(fps_counter), qs[0], qs[1], qs[2], action))

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()

        with open(csv_name) as f:
            rows = sum(1 for line in f)
        if rows == 101:
            print("100 tests complete")
            break
        #print(c)
        #if c == 101:
        #    break
