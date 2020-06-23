import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from DQNAgent import CarEnv, MEMORY_FRACTION
import spawn_npc


MODEL_PATH = 'model_free_models/model_free_1.model'
#MODEL_PATH = 'model_free_models/model_free_2.model'
#MODEL_PATH = 'model_free_models/model_free_3.model'
#MODEL_PATH = 'model_free_models/model_free_4.model'
#MODEL_PATH = 'model_free_models/model_free_5.model'


SPAWN_NPC = True
HD_CAM = True
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


    episode_count = 0
    #step = 0
    # Loop over episodes
    while True:

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

        done = False
        step = 0
        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()

            # Show current frame
            if HD_CAM == True:
                #cv2.imshow(f'Agent - preview', current_state)
                resized_current_state = cv2.resize(current_state, (640, 480))
                #cv2.imshow("Agent - resized network input", resized_current_state)
                #cv2.imshow("Agent - preview", current_state)
                #cv2.waitKey(1)

            # Predict an action based on current observation space
            qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
            action = np.argmax(qs)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            #new_state, reward, done, _ = env.step(action)
            if HD_CAM:
                new_state, reward, done, _, third_person = env.step(action)
                #cv2.imshow("HD CAM", third_person)
                #cv2.waitKey(1)
                #image = (third_person*255).astype('uint8')
                #if(step % 10):
                name = str(step).zfill(4)#"00"+str(step)
                path = 'film/' + name + '.png'
                cv2.imwrite(path, third_person)
                step += 1
            else:
                new_state, reward, done, _ = env.step(action)
            # Set current step for next loop iteration
            current_state = new_state


            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            #print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')
            #print("Agent: {} FPS | Action: [{}, {} , {}] {}".format(len(fps_counter)/sum(fps_counter), qs[0], qs[1], qs[2], action))

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()
