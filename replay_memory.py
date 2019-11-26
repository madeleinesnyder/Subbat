import numpy as np
import h5py
import os
from utils import *
import time


class ReplayMemory:

    def __init__(self, name, obs_shape, buffer_capacity = 512, storage_capacity = 2048):
        self.buffer_capacity = buffer_capacity
        self.storage_capacity = storage_capacity
        self.buffer_counter = 0
        self.name = name
        self.obs_shape = obs_shape
        self.memories = [np.empty([buffer_capacity] + list(self.obs_shape)), np.empty([buffer_capacity]),
            np.empty([buffer_capacity]), np.empty([buffer_capacity] + list(self.obs_shape))]
        

        if not os.path.exists("replay_buffer"):
            os.makedirs("replay_buffer")

        if os.path.exists("replay_buffer/" + self.name + ".hdf5"):
            with h5py.File("replay_buffer/" + self.name + ".hdf5", "r") as f:
                load_num = min(len(f['reward']), self.buffer_capacity)
                if self.name == "controller":
                    self.memories[0][:load_num] = f['obs_goals_t'][-load_num:]
                    self.memories[1][:load_num] = np.array(f['action'][-load_num:], dtype = np.uint8)
                    self.memories[2][:load_num] = f['reward'][-load_num:]
                    self.memories[3][:load_num] = f['obs_goals_tp1'][-load_num:]
                elif self.name == "metacontroller":
                    self.memories[0][:load_num] = f['obs_t'][-load_num:]
                    self.memories[1][:load_num] = np.array(f['goal_idx'][-load_num:], dtype = np.uint8)
                    self.memories[2][:load_num] = f['reward'][-load_num:]
                    self.memories[3][:load_num] = f['obs_tp1'][-load_num:]
            self.buffer_counter = load_num
        else:
            with h5py.File("replay_buffer/" + self.name + ".hdf5", "w") as f:
                
                f.create_dataset("reward", (0,), maxshape = [None,])
                if self.name == "controller":
                    f.create_dataset("obs_goals_t", [0] + list(self.obs_shape), maxshape = [None] + list(self.obs_shape))
                    f.create_dataset("action", (0,), maxshape = [None,])
                    f.create_dataset("obs_goals_tp1", [0] + list(self.obs_shape), maxshape = [None] + list(self.obs_shape))
                elif self.name == "metacontroller":
                    f.create_dataset("obs_t", [0] + list(self.obs_shape), maxshape = [None] + list(self.obs_shape))
                    f.create_dataset("goal_idx", (0,), maxshape = [None,])
                    f.create_dataset("obs_tp1", [0] + list(self.obs_shape), maxshape = [None] + list(self.obs_shape))
                
    def store(self, args):
        # Input:
        #   args: list of items to store; [observation, goal_xy, action, intrinsic reward, next_observation] for controller 
        #         and [observation, goal_idx, external reward, next_observation] for metacontroller

        if self.buffer_counter == self.buffer_capacity:
            self.write_to_file()
            self.buffer_counter = 0

        self.memories[0][self.buffer_counter] = args[0][np.newaxis, :, :, :]
        self.memories[1][self.buffer_counter] = args[1]
        self.memories[2][self.buffer_counter] = args[2]
        self.memories[3][self.buffer_counter] = args[3][np.newaxis, :, :, :]
        self.buffer_counter += 1

    def write_to_file(self):
        with h5py.File("replay_buffer/" + self.name + ".hdf5", 'a') as f:

            if self.name == "controller":

                obs_t = f['obs_goals_t']
                a_or_g = f['action']
                reward = f['reward']
                obs_tp1 = f['obs_goals_tp1']

            elif self.name == "metacontroller":

                obs_t = f['obs_t']
                a_or_g = f['action']
                reward = f['reward']
                obs_tp1 = f['obs_tp1']

            if obs_t.shape[0] + len(self.memories[0]) > self.storage_capacity:
                addSize = len(self.memories[0])
                keepSize = self.storage_capacity - addSize

                # shift old data to keep to the top
                obs_t[:keepSize] = obs_t[-keepSize:]
                a_or_g[:keepSize] = a_or_g[-keepSize:]
                reward[:keepSize] = reward[-keepSize:]
                obs_tp1[:keepSize] = obs_tp1[-keepSize:]

                if obs_t.shape[0] < self.storage_capacity:
                    obs_t.resize(tuple([self.storage_capacity] + list(self.obs_shape)))
                    a_or_g.resize((self.storage_capacity,))
                    reward.resize((self.storage_capacity,))
                    obs_tp1.resize(tuple([self.storage_capacity] + list(self.obs_shape)))

            else:

                obs_t.resize(tuple([obs_t.shape[0] + len(self.memories[0])] + list(self.obs_shape)))
                a_or_g.resize((a_or_g.shape[0] + len(self.memories[1]),))
                reward.resize((reward.shape[0] + len(self.memories[2]),))
                obs_tp1.resize(tuple([obs_tp1.shape[0] + len(self.memories[3])] + list(self.obs_shape)))

            obs_t[-len(self.memories[0]):] = self.memories[0]
            a_or_g[-len(self.memories[1]):] = self.memories[1]
            reward[-len(self.memories[2]):] = self.memories[2]
            obs_tp1[-len(self.memories[3]):] = self.memories[3]

    def sample(self, size):
        # if size > current: return everything
        if size >= len(self.memories[0]):
            return [self.memories[0], self.memories[1], self.memories[2], self.memories[3]]

        # randomly sample
        batch_idx = np.random.choice(len(self.memories[0]), size = size, replace = False)
        a = time.perf_counter()
        el_0 = self.memories[0][batch_idx]
        el_3 = self.memories[3][batch_idx]
        b = time.perf_counter()
        print("time creating return value in sample func: ", b - a)
        el_1 = self.memories[1][batch_idx]
        el_2 = self.memories[2][batch_idx]
        test = [el_0, el_1, el_2, el_3]
        return test

            
        
