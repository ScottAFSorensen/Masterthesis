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


#MODEL_PATH = 'models/64x3_CNN_Radical_UTE_10_MS_32__47.0max_28.0avg_6.0min__1573485896.model'
#MODEL_PATH = 'models/640x480_Radical_CNN_UTE_10_MS_32_cont__45.0max_22.8avg_6.0min__1573561129.model' #Current best
#MODEL_PATH = 'models/640x480_Radical_CNN_UTE_10_MS_32_cont_2500ep__54.0max_32.7avg_10.0min__1573602969.model' #Attempt at introducing braking
#MODEL_PATH = 'models/640x480_Radical_CNN_UTE_10_MS_32_Stochastic_brake__34.0max_16.0avg_0min__1573737867.model' #Another attempt at introducing braking
#MODEL_PATH = 'models/640x480_Radical_CNN_UTE_10_MS_32_Stochastic_brake_Testtestestt__42.0max_31.9avg_8.0min__1573798435.model'
#MODEL_PATH = 'models/128x128_good_CNN_1kep__107.0max_27.6avg_4.0min__1574171303.model' #128x128 1.5k ep speedup 0.5
#MODEL_PATH = 'models/640x480_Radical_CNN_UTE_10_MS_32_cont_2500ep__54.0max_32.7avg_10.0min__1573602969.model' #Possibly the best
#MODEL_PATH = 'models/64x64_resnet__53.0max_33.7avg_0min__1574738371.model' #resnet 9k episodes
#MODEL_PATH = 'models/64x64_world_models_net_5k_fast__0.23000000000000007max_-0.573avg_-0.95min__1574885978.model' #World models net
#MODEL_PATH = 'models/128x128_world_models_net_500ep_plswork_final__0.5500000000000003max_-0.3689999999999999avg_-0.95min__1575380152.model' #world models net, 128x128, 2k ep
#MODEL_PATH = 'models/128x128_world_models_net_1k_ep_plswork__0.3900000000000002max_-0.546avg_-1.7400000000000013min__1575460154.model'
#MODEL_PATH = 'models/128x128_world_models_net_1k_ep_plswork_anotherone__0.6100000000000003max_-0.962999999999998avg_-3.039999999999979min__1575466232.model'
#MODEL_PATH = 'models/128x128_world_models_net_1k_ep_plswork_3__0.5700000000000003max_-0.22899999999999993avg_-0.9299999999999999min__1575472889.model'
#Load model from selected models:
#MODEL_PATH = 'selected_models/982_ep/DQN_checkpoint.model'

#Models to be tested in the Results chapter
#M1: THis one is bad, only drives straight
#MODEL_PATH = 'model_free_models/world_models_net_M1__0.8800000000000006max_-0.5579999999999997avg_-0.95min__1586549986.model'
#M2: This one is good, 5000 replay memory size
#MODEL_PATH = 'model_free_models/world_models_net_M2__0.5400000000000003max_-0.591avg_-0.9299999999999999min__1586572850.model'
#M3: 3400 eps not good
#MODEL_PATH = 'model_free_models/DQN_checkpoint_M3.model'
#M3: +1000 eps not good
#MODEL_PATH = 'model_free_models/world_models_net_M3__0.10999999999999999max_-0.673avg_-1min__1586649195.model'
#M4: not good, 1500 replay memory size
#MODEL_PATH = 'model_free_models/world_models_net_M4__0.09999999999999999max_-0.6940000000000001avg_-1.1199999999999999min__1586887628.model'
#M5: This one is good, 5000 replay memory size
#MODEL_PATH= 'model_free_models/world_models_net_M5__0.4000000000000002max_-0.40099999999999997avg_-0.97min__1586927743.model'
#M6: This one is good, 5000 replay memory size
#MODEL_PATH = 'model_free_models/world_models_net_M6__0.7100000000000004max_-0.32999999999999985avg_-0.94min__1586964998.model'
#M7: This one is weird but okay
#MODEL_PATH = 'model_free_models/world_models_net_M7__0.5600000000000003max_-0.33199999999999996avg_-0.92min__1586990182.model'
#M8: This one is good
#MODEL_PATH = 'model_free_models/world_models_net_M8__0.49000000000000027max_-1.1519999999999864avg_-8.399999999999865min__1587011772.model'

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
            #image = (current_state*255).astype('uint8')
            #if(step % 150):
            #    path = 'model_free_sequence/' + str(step) + '.png'
            #    cv2.imwrite(path, current_state)

            #step += 1
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
