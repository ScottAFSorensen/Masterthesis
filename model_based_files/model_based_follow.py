import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import keras
from multiprocessing import Process
import subprocess
from collections import namedtuple
import matplotlib.pyplot as plt
from collections import deque
from keras.applications.xception import Xception
#from keras.applications.resnet import ResNet50
from keras.optimizers import Adam, Nadam, SGD, RMSprop
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten, LeakyReLU, MaxPooling2D
from keras import regularizers
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from tqdm import tqdm
import spawn_npc
from vae.vae import ConvVAE, reset_graph
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size
import online_VAE
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


SHOW_PREVIEW = False
SHOW_THIRD_PERSON = True
SPEED_UP = False
IM_WIDTH = 128#64#640#128
IM_HEIGHT = 128#64#480#128
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 1000
MIN_REPLAY_MEMORY_SIZE = 500#0
MINIBATCH_SIZE = 32#16 #32 seem to work better
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 10#10
MODEL_NAME = "7sec_scratch_2_noboost_model_based_follower"#"128x128_world_models_net"#"Xception"
NR_OF_ACTIONS = 6
MEMORY_FRACTION = 0.3
MIN_REWARD = -200
SPAWN_NPC = False
EPISODES = 1000
SEGMENTATION = False

DISCOUNT = 0.99
epsilon = 0
EPSILON_DECAY = 0.99# 0.9975 #0.996## 0.9975 99975
MIN_EPSILON = 0.001#0.001

AGGREGATE_STATS_EVERY = 10


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
        pass

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 0.5#0.75
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None
    seg_camera = None

    def __init__(self):
        #self.SHOW_CAM = SHOW_PREVIEW
        self.client = carla.Client("localhost", 2002)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()

        if SPEED_UP:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.5
            self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.police = self.blueprint_library.filter("police")[0]
        self.prev_speed = 0
        self.start_location = 0
        self.temp_reward = 0

        #self.prev_distance = 0
        #List of predefined weather settings, will be randomly selected.
        #Because there are so many rainy presets, a few extra instances of clear weather presets
        #have been added to avoid almost every episode being rainy.
        self.weather_presets = [carla.WeatherParameters.Default,
                            carla.WeatherParameters.Default,
                            carla.WeatherParameters.Default,
                            carla.WeatherParameters.ClearNoon,
                            carla.WeatherParameters.ClearNoon,
                            carla.WeatherParameters.ClearNoon,
                            carla.WeatherParameters.CloudyNoon,
                            carla.WeatherParameters.WetNoon,
                            carla.WeatherParameters.WetCloudyNoon,
                            carla.WeatherParameters.MidRainyNoon,
                            carla.WeatherParameters.HardRainNoon,
                            carla.WeatherParameters.SoftRainNoon,
                            carla.WeatherParameters.ClearSunset,
                            carla.WeatherParameters.ClearSunset,
                            carla.WeatherParameters.ClearSunset,
                            carla.WeatherParameters.CloudySunset,
                            carla.WeatherParameters.WetSunset,
                            carla.WeatherParameters.WetCloudySunset,
                            carla.WeatherParameters.MidRainSunset,
                            carla.WeatherParameters.HardRainSunset,
                            carla.WeatherParameters.SoftRainSunset]



    def reset(self):

        #Set random distance that obstacle sensor can see:
        dist = str(random.uniform(4,8))
        self.prev_distance = 0
        self.distance = 0
        self.distance2 = 0
        #print(dist)
        #Set random weather
        #self.world.set_weather(self.weather_presets[np.random.randint(0, 21)])
        self.world.set_weather(self.weather_presets[5])

        #If a collision is detected, add it to this list
        self.collision_hist = []
        #If an obstacle ahead is detected, add it to this list
        self.obstacle_hist = []
        #Add actors to this list
        self.actor_list = []
        #Get a random spawn point from a list of reccommended transforms:
        self.transform = random.choice(self.world.get_map().get_spawn_points())

        #Get spawn point from agent car for a random car that will be spawned ahead of the agent
        x = self.transform.location.x
        y = self.transform.location.y
        z = self.transform.location.z
        forward_vector = self.transform.get_forward_vector()
        xx = forward_vector.x
        yy = forward_vector.y
        zz = forward_vector.z
        #rotation for agent car, the random car must be rotated in relation to this car
        pitch = self.transform.rotation.pitch #Rotation about Y-axis.
        yaw = self.transform.rotation.yaw #Rotation about Z-axis. Mainly yaw that will be altered.
        roll = self.transform.rotation.roll #Rotation about X-axis.

        self.point1 = carla.Transform(carla.Location(x+xx*15,y+yy*15,z+zz), self.transform.rotation)
        self.point2 = carla.Transform(carla.Location(x+xx*27,y+yy*27,z+zz), carla.Rotation(pitch=pitch, yaw=yaw+90, roll=roll))
        self.point3 = carla.Transform(carla.Location(x+xx*25,y+yy*25,z+zz), carla.Rotation(pitch=pitch, yaw=yaw-90, roll=roll))
        self.point4 = carla.Transform(carla.Location(x+xx*25,y+yy*25,z+zz), carla.Rotation(pitch=pitch, yaw=yaw+180, roll=roll))
        self.point5 = carla.Transform(carla.Location(x+xx*25,y+yy*25,z+zz), carla.Rotation(pitch=pitch, yaw=yaw+45, roll=roll))
        self.point6 = carla.Transform(carla.Location(x+xx*25,y+yy*25,z+zz), carla.Rotation(pitch=pitch, yaw=yaw-45, roll=roll))
        self.scenario_list = [self.point1, self.point2, self.point3, self.point4, self.point5, self.point6]



        while True:
            self.random_car = random.choice(self.blueprint_library.filter('vehicle'))
            if (self.random_car.id == "vehicle.yamaha.yzf" or self.random_car.id == "vehicle.kawasaki.ninja"
                    or self.random_car.id == "vehicle.harley-davidson.low-rider" or self.random_car.id == "vehicle.gazelle.omafiets"
                    or self.random_car.id == "vehicle.bh.crossbike" or self.random_car.id == "vehicle.diamondback.century"):
                continue
            else:
                break

        #Spawn car:
        #When spawn point is already occupied, spawning a car throws an exception
        #It is allowed to try for 3 seconds then the spawning is reset
        spawn_start = time.time()
        while True:

            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                #self.vehicle.set_autopilot(True)
                self.actor_list.append(self.vehicle)
                #self.start_location = self.vehicle.get_location()
                #Spawn random car ahead of the agent car:
                #if random.randint(0,1) == 1:
                #    self.vehicle_ahead = self.world.spawn_actor(self.random_car, self.scenario_list[0])
                    #self.vehicle_ahead = self.world.spawn_actor(self.random_car, self.scenario_list[random.randint(0,6)])
                #self.vehicle_ahead.set_simulate_physics(False)
                #    self.actor_list.append(self.vehicle_ahead)
                self.vehicle_ahead = self.world.spawn_actor(self.police, self.scenario_list[0])
                self.actor_list.append(self.vehicle_ahead)
                break
            except:
                #print("sleeping")
                time.sleep(0.01)
            # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                #raise Exception('Can\'t spawn a car')
                print("Cant spawn car - resetting")
                for actor in self.actor_list:
                    actor.destroy()
                self.reset()

        #Camera:
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        #self.rgb_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.rgb_cam.set_attribute("image_size_x", "{}".format(self.im_width))
        self.rgb_cam.set_attribute("image_size_y", "{}".format(self.im_height))
        #Sets field of view:
        #self.rgb_cam.set_attribute("fov", "110")
        self.rgb_cam.set_attribute("fov", "130")
        #Set camera location at hood of car. Camera location is relative to the car type.
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        #Spawn camera and attach to car
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        #self.sensor.listen(lambda data: self.process_segmentation(data))

        #Spawn third person camera and attach to car
        if SHOW_THIRD_PERSON:
            self.rgb_cam2 = self.blueprint_library.find('sensor.camera.rgb')
            #self.rgb_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            #self.rgb_cam2.set_attribute("image_size_x", "{}".format(640))
            #self.rgb_cam2.set_attribute("image_size_y", "{}".format(480))

            self.rgb_cam2.set_attribute("image_size_x", "600")
            self.rgb_cam2.set_attribute("image_size_y", "512")

            self.rgb_cam2.set_attribute("fov", "130")
            #transform = carla.Transform(carla.Location(x=-7.0, z=5.0))
            transform2 = carla.Transform(carla.Location(x=2.6, z=0.8))
            self.sensor2 = self.world.spawn_actor(self.rgb_cam2, transform2, attach_to=self.vehicle)
            self.actor_list.append(self.sensor2)
            self.sensor2.listen(lambda data2: self.process_img_3(data2))

        if SEGMENTATION:

            self.rgb_cam2 = self.blueprint_library.find('sensor.camera.rgb')
            #self.rgb_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            #self.rgb_cam2.set_attribute("image_size_x", "{}".format(640))
            #self.rgb_cam2.set_attribute("image_size_y", "{}".format(480))

            self.rgb_cam2.set_attribute("image_size_x", "600")
            self.rgb_cam2.set_attribute("image_size_y", "512")

            self.rgb_cam2.set_attribute("fov", "130")
            #transform = carla.Transform(carla.Location(x=-7.0, z=5.0))
            transform2 = carla.Transform(carla.Location(x=2.6, z=0.8))
            self.sensor2 = self.world.spawn_actor(self.rgb_cam2, transform2, attach_to=self.vehicle)
            self.actor_list.append(self.sensor2)
            self.sensor2.listen(lambda data2: self.process_img_3(data2))

            self.seg_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            self.seg_cam.set_attribute("image_size_x", "{}".format(self.im_width))
            self.seg_cam.set_attribute("image_size_y", "{}".format(self.im_height))
            self.seg_cam.set_attribute("fov", "130")
            transform = carla.Transform(carla.Location(x=2.5, z=0.7))
            self.sensor3 = self.world.spawn_actor(self.seg_cam, transform, attach_to=self.vehicle)
            self.actor_list.append(self.sensor3)
            self.sensor3.listen(lambda data: self.process_segmentation(data))


        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        #Give camera 2 seconds to get ready as carla sometimes takes time to start the camera
        time.sleep(1)
        #Collision sensor:
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        #Obstacle sensor:
        """obsensor = self.blueprint_library.find("sensor.other.obstacle")
        obsensor.set_attribute('distance', '5')
        obsensor.set_attribute('only_dynamics', 'true')
        obsensor.set_attribute('debug_linetrace', 'true')
        self.obsensor = self.world.spawn_actor(obsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.obsensor)
        self.obsensor.listen(lambda event2: self.obstacle_data(event2))"""

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        #self.vehicle_ahead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        self.start_location = self.vehicle.get_location()
        self.old_location = self.vehicle.get_location()
        #print(self.start_location)
        self.pilot_location = self.vehicle.get_location()
        self.pilot_distance = self.pilot_location.distance(self.start_location)
        self.vehicle_ahead.set_autopilot(True)

        if SEGMENTATION:
            return self.front_camera, self.seg_camera
        else:
            return self.front_camera
    def collision_data(self, event):
        coll_object = event.other_actor
        if coll_object.type_id == 'static.sidewalk' or coll_object.type_id == 'static.road':
            #print("Not Appending {}".format(coll_object.type_id))
            pass

        else:
            #print("Appending {}".format(coll_object.type_id))
            self.collision_hist.append(event)


    #def obstacle_data(self, event2):
    #    self.obstacle_hist.append(event2)

    #Processing images from sensor
    def process_img(self, image):

        i = np.array(image.raw_data)
        #ii = np.array(image.raw_data)
        #cv2.imshow("Agent - camera", i)
        #cv2.waitKey(1)
        #Reshaping image array to image, by 4 as image is rgba:
        i2 = i.reshape((self.im_height, self.im_width, 4))
        #i2_hd = i.reshape((256, 256, 4))
        #Converting image from rgba to rgb
        i3 = i2[:, :, :3]
        #i3_hd = i2_hd[:, :, :3]
        #if self.SHOW_CAM:
        #    cv2.imshow("display", i3/255.0)
        #    cv2.waitKey(1)
        self.front_camera = i3
        #self.hd_cam = i3_hd

    def process_img_3(self, image):
        #image. = labels_to_cityscapes_palette(image)
        x = np.array(image.raw_data)
        #cv2.imshow("Agent - camera", i)
        #cv2.waitKey(1)
        #Reshaping image array to image, by 4 as image is rgba:
        #x2 = x.reshape((self.im_height, self.im_width, 4))
        x2 = x.reshape((512, 600, 4))
        #Converting image from rgba to rgb
        x3 = x2[:, :, :3]

        self.third_person_view = x3

    def process_segmentation(self, image):
        # Converts a batch of one-hot encoded segmentation
        # maps to RGB images within the scale [0,1]

        # CARLA label to RGB conversion table
        #img2 = keras.one_hot(img, 13)
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        #Converting image from rgba to rgb
        i3 = i2[:, :, 2]
        #print(i3)
        #img = i3
        self.seg_camera = i3


    def step(self, action):
        if action == 0: #Go left
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=-1*self.STEER_AMT))
            #print("0")
        elif action == 1: #Go straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.90, steer= 0))
            #print("1")
        elif action == 2: # Go right
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.75, steer=1*self.STEER_AMT))
            #print("2")
        elif action == 3: #Brake straight
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.75))
            #print("3")
        elif action == 4: #Go left and brake
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5, steer=-1*self.STEER_AMT))
            #print("4")
        elif action == 5: # Go right and brake
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5, steer=1*self.STEER_AMT))
            #print("5")


        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        #print(kmh)
        new_location = self.vehicle.get_location()
        new_pilot_location = self.vehicle_ahead.get_location()
        self.distance = new_location.distance(self.start_location)
        #print(self.distance)
        #print("distance={}".format(distance))
        self.pilot_distance = new_pilot_location.distance(new_location)
        #print(self.pilot_distance)
        reward = 0.0
        done = False
        '''
        #If a collision happens give punishment:
        if len(self.collision_hist) != 0:
            done = True
            reward += -1
        elif self.pilot_distance >= 1 and self.pilot_distance <= 15:
            done = False
            reward += 0.5
            if self.distance - self.prev_distance >= 1:
                reward += 0.01
                self.prev_distance = self.distance
        else:
            done = False
            reward += -0.5
        '''
        '''
        if len(self.collision_hist) != 0:
            done = True
            reward += -1
        elif self.pilot_distance >= 15 and self.pilot_distance <= 20:
            done = False
            reward += 0.01
        elif self.pilot_distance >= 10 and self.pilot_distance <= 15:
            done = False
            reward += 0.02
        elif self.pilot_distance >= 2 and self.pilot_distance <= 10:
            done = False
            reward += 0.03
        else:
            done = False
            reward += -0.05
        '''
        if len(self.collision_hist) != 0:
            done = True
            reward += -1
        elif self.pilot_distance <= 2:
            done = False
            #reward += -0.05
        elif self.pilot_distance > 2 and self.pilot_distance <= 30:
            done = False
            reward +=  (1 / self.pilot_distance)
        elif self.pilot_distance > 30:
            done = False
        else:
            done = False
            #reward += -0.01


        location_1 = self.vehicle.get_location()
        self.distance2 += location_1.distance(self.old_location)
        self.old_location = location_1

        #Limit time of episodes
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            #reward += (self.distance-13)*0.001
            #print("Distance reward {}".format((self.distance-13)*0.001))
            #print("No collision, end episode {}".format(reward))
            done = True
            #self.temp_reward = 0
        if SHOW_THIRD_PERSON:
            return self.front_camera, reward, done, None, self.third_person_view
        if SEGMENTATION:
            return self.front_camera, reward, done, None, self.third_person_view, self.seg_camera
        else:
            #print(reward)
            #print(self.temp_reward)
            return self.front_camera, reward, done, None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir= "logs/{}-{}".format(MODEL_NAME, int(time.time())))
        #self.tensorboard = keras.callbacks.TensorBoard(log_dir= "logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False



    def model_base_fullyconnected_576(self, input_shape):
        model = Sequential()
        model.add(Dense(576, input_dim=576, activation='relu'))
        #model.add(LeakyReLU(alpha=0.1))
        #model.add(Dense(512, kernel_initializer=keras.initializers.he_uniform(), activation='relu'))
        #model.add(Dense(512, activation='relu'))
        #model.add(Dense(256, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        #model.add(Dense(32, activation='relu'))
        return model.input, model.output

    def create_model(self):
        #Load model_base_64x3_CNN
        '''
        base_model = self.model_base_fullyconnected_576(input_shape=(1, 576))
        #base_model = self.model_base_64x3_CNN(input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        #base_model = self.model_base_radical_CNN(input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        #base_model = self.model_base_world_models(input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        x = base_model[1]
        #6 possible predictions: left, right, straight, braking, left and braking, right and braking.
        #predictions = Dense(NR_OF_ACTIONS, activation="linear")(x)
        predictions = Dense(NR_OF_ACTIONS, activation="linear")(x)
        model = Model(inputs=base_model[0], outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.0001, clipnorm = 1.0, amsgrad = True), metrics=['accuracy'])
        '''

        #model = keras.models.load_model('model_based_models/model_based_1.model')
        #model = keras.models.load_model('model_based_models/model_based_2.model')
        #model = keras.models.load_model('model_based_models/model_based_3.model')
        #model = keras.models.load_model('model_based_models/model_based_4.model')
        #model.layers[0].trainable = False
        model = keras.models.load_model('model_based_models/model_based_5.model')
        model.compile(loss="mse", optimizer=Adam(lr=0.0001, clipnorm = 1.0, amsgrad = True), metrics=['accuracy'])
        model.summary()

        #model = keras.models.load_model('follow_models/model_based_follower_5_norm__1.9200000000000013max_-0.4979999999999999avg_-2.0min__1590265297.model')
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            #print("tensortest")
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        #Calculate current states
        current_states = np.array([transition[0] for transition in minibatch])/255

        with self.graph.as_default():
            #print("tensortrollmann3")
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        #Calculate future states
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            #print("tensortrollmann4")
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)
        #Current state
        X = []
        #Current q-values
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step


        with self.graph.as_default():
            #print("tensortrollmann")
            #self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        #X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        #X = np.random.uniform(size=(1, 288)).astype(np.float32)
        X = np.random.uniform(size=(1, 576)).astype(np.float32)
        y = np.random.uniform(size=(1, NR_OF_ACTIONS)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 100#110
    # For stats
    ep_rewards = [0]
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

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    #Set memory fraction, used mostly when training multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    #Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    #Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    coll_count = 0 #Counts number of collisions experienced
    car_count = 0 #Counts number of collisions experienced with cars
    ped_count = 0 #Counts number of collisions experienced with pedestrians
    wall_count = 0 #Counts number of collisions with walls and buildings experienced
    ob_count = 0 #Counts number of collisions with objects, lightpoles and vegetation experienced
    dist_reach = False
    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    #agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
    #agent.get_qs(np.ones((288,)))
    agent.get_qs(np.ones((576,)))
    if SPAWN_NPC:
        client2 = spawn_npc.main()
    # Iterate over episodes
    file_int = 0

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
            if SPAWN_NPC:
                if episode % 5 == 0:
                    spawn_npc.destroyAll(client2)
                    client2 = spawn_npc.main()
                    # Error check:
                    if client2 == None:
                        print("SOMETHINH IS WONRTB")


            env.collision_hist = []
            #env.obstacle_hist = []
            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0

            step = 1
            action = 1
            # Reset environment and get initial state
            if SEGMENTATION:
                current_state, current_seg_camera = env.reset()
            else:
                current_state = env.reset()
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
            #input_hz = rnn_output(rnn_state, z, 4)
            #print(input_hz.shape)
            #print(input_hz)
            #h = rnn.initial_state
            #print(h)
            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()
            #print(env.distance)
            # Play for given number of seconds only
            """
            DIR_NAME = "online_record"
            filename = DIR_NAME+"/"+str(file_int)+".npz"
            recording_obs = []
            recording_action = []
            recording_seg = []
            obs = current_state
            pixel_obs = current_state
            seg_obs = current_seg_camera
            """

            while True:
                if SHOW_PREVIEW:
                    #resized_current_state = cv2.resize(current_state, (640, 480))
                    #cv2.imshow("Agent - resized network input", resized_current_state)
                    cv2.imshow("Agent - network input", current_state)
                    cv2.waitKey(1)


                #Concatenates hidden states from rnn with encoded z vector
                #input_hz = rnn_output(rnn_state, z, 4)
                #Normalize values between -1 and 1
                #input_hz = (2* ((input_hz - np.min(input_hz)) / (np.max(input_hz) - np.min(input_hz)))) - 1
                #Normalize values between 0 and 1
                #input_hz = (input_hz-min(input_hz))/(max(input_hz)-min(input_hz))
                #print(input_hz)
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon and env.distance >= 0:# and len(env.obstacle_hist) != 0:
                    # Get action from Q table
                    #action = np.argmax(agent.get_qs(current_state))
                    action = np.argmax(agent.get_qs(input_hz))
                    #print("q-action: {}".format(action))
                elif env.distance >= 0: # len(env.obstacle_hist) != 0:
                    # Get random action
                    action = np.random.randint(0, NR_OF_ACTIONS)

                    #action = 3
                    #print("random-action: {}".format(action))
                    # This takes no time, so we add a delay matching 10 FPS (prediction above takes longer)
                    time.sleep(1/FPS)
                else:
                    action = 1


                #print(action)
                if SHOW_THIRD_PERSON:
                    new_state, reward, done, _, third_person = env.step(action)
                    cv2.imshow("hd view", third_person)
                    cv2.waitKey(1)
                elif SEGMENTATION:
                    new_state, reward, done, _, third_person, seg_camera = env.step(action)
                    #cv2.imshow("hd view", third_person)
                    #cv2.waitKey(1)
                else:
                    new_state, reward, done, _ = env.step(action)
                    #print("test")
                    #print(action)
                #print(reward)
                # Transform new continous state to new discrete state and count reward
                #if env.distance >=10:# or dist_reach == True:
                    #dist_reach = True
                episode_reward += reward
                    #print("episode_reward {}".format(episode_reward))
                #print(episode_reward)
                # Every step we update replay memory
                    #agent.update_replay_memory((current_state, action, reward, new_state, done))




                new_state_z = new_state.astype(np.float32)/255.0
                new_state_z = new_state_z.reshape(1, 128, 128, 3)
                z = vae.encode(new_state_z)
                rnn_state = rnn_next_state(rnn, z, np.asarray(action), rnn_state)
                new_input_hz = rnn_output(rnn_state, z, 4)
                    #Normalize between -1 and 1
                    #new_input_hz = (2 * ((new_input_hz - np.min(new_input_hz)) / (np.max(new_input_hz) - np.min(new_input_hz)))) - 1
                    #Normalize between 0 and 1
                new_input_hz = (new_input_hz-min(new_input_hz))/(max(new_input_hz)-min(new_input_hz))
                agent.update_replay_memory((input_hz, action, reward, new_input_hz, done))
                current_state = new_state
                input_hz = new_input_hz

                    #rnn_test = np.asarray(rnn_state)
                    #print(rnn_test[0][0][-1])
                step += 1

                    #z = np.expand_dims(z, axis=0)
                    #reconstruct = vae.decode(z)
                    #cv2.imshow("reconstruction", reconstruct[0])
                    #cv2.waitKey(1)

                if done:
                    print("episode_reward {}".format(episode_reward))
                    break
            """
            print(len(recording_obs))
            print(len(recording_seg))
            print(len(recording_action))
            recording_obs = np.array(recording_obs, dtype=np.uint8)
            recording_seg = np.array(recording_seg, dtype=np.uint8)
            recording_action = np.array(recording_action, dtype=np.float16)

            if (len(recording_obs) > 1):
                print("Episode saved")
                file_int += 1
                np.savez_compressed(filename, obs=recording_obs, seg = recording_seg, action=recording_action)
            """
            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            if len(env.collision_hist) > 0:
                #print("collision happened")
                coll_count += 1
                coll_object = env.collision_hist[0].other_actor
                coll_id = coll_object.type_id
                #print(coll_id)
                if(coll_id[0:7] == "vehicle"):
                    #print("You crashed in a car")
                    car_count += 1
                elif(coll_id[0:15]== "static.building" or coll_id[0:11] == "static.wall"):
                    #print("You crashed in a wall or building")
                    wall_count += 1
                elif(coll_id[0:7] == "traffic" or coll_id == "static.pole" or coll_id == "static.vegetation"): #Unkown is tunnel entrance, construction markers, pots etc, probably shouldnt include this here
                    #print("You crashed in a traffic light, a pole or a tree")
                    ob_count += 1
                elif(coll_id[0:6] == "walker"):
                    #print("You crashed in a pedestrian. R.I.P.")
                    ped_count += 1
                else:
                    #print("You crashed in an unknown object")
                    pass
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            #print(ep_rewards)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                collision_percent = coll_count/10#episode
                car_percent = car_count/10
                wall_percent = wall_count/10
                object_percent = ob_count/10
                pedestrian_percent = ped_count/10
                coll_count = 0
                car_count = 0
                wall_count = 0
                ob_count = 0
                ped_count = 0
                #print(coll_count)
                #print(collision_percent)
                #print(coll_count)
                #print(episode)
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, collision_percent = collision_percent,
                car_percent = car_percent, wall_percent = wall_percent, object_percent = object_percent, pedestrian_percent = pedestrian_percent)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                #    agent.model.save("models/{}__{}max_{}avg_{}min__{}.model".format(MODEL_NAME, max_reward, average_reward, min_reward, int(time.time())))
                    agent.model.save("follow_models/DQN_checkpoint.model")
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            """    #Train vae every 100 episode
            if file_int >= 200:
                #print("hei")
                #print(file_int)
                file_int = 0
                online_VAE.main()
                #p = Process(target=online_VAE.main(), args=())
                #p.start()
                #p.join()
                #subprocess.Popen('python online_VAE.py', shell=False)
            """

    # Set termination flag for training thread and wait for it to finish
    if SPAWN_NPC:
        spawn_npc.destroyAll(client2)
    agent.terminate = True
    trainer_thread.join()
    agent.model.save("follow_models/{}__{}max_{}avg_{}min__{}.model".format(MODEL_NAME, max_reward, average_reward, min_reward, int(time.time())))
