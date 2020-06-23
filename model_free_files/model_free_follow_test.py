import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from followDQN import CarEnv, MEMORY_FRACTION
import spawn_npc
import pandas
import csv

#MODEL_PATH = 'model_free_models/model_free_1.model'
#MODEL_PATH = 'model_free_models/model_free_2.model'
#MODEL_PATH = 'model_free_models/model_free_3.model'
#MODEL_PATH = 'model_free_models/model_free_4.model'
#MODEL_PATH = 'model_free_models/model_free_5.model'
#MODEL_PATH = 'follow_models/follow_model_standard_8.3max_4.4avg_1.9min_1000eps_final.model'
#MODEL_PATH = 'follow_models/car_follower_4__49.58max_18.733000000000004avg_6.720000000000013min__1589895751.model'
#MODEL_PATH = 'follow_models/car_follower_3__50.56max_13.379999999999999avg_-8.370000000000005min__1589658590.model'
#MODEL_PATH = 'final_follow_data/transfer_follow_model.model'
MODEL_PATH = 'final_follow_data/scratch_follow_model.model'
SPAWN_NPC = False
HD_CAM = True

csv_name = "model_free_scratch_test_reward_15.csv"
with open(csv_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Collision", "Type", "Distance", "Time", "Mean_Dist_Pilot", "Mean_Time", "Episode_Reward"])

if __name__ == '__main__':

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
    model.predict(np.ones((1, env.im_height, env.im_width, 3)))



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
        env.collision_hist = []
        episode_reward = 0
        done = False
        start_time = time.time()
        collision = 0
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
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
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
            current_state = new_state
            pilot_dist_list.append(env.pilot_distance)
            episode_reward += reward
            #print(env.distance)


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
