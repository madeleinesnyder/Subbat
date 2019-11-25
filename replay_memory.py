import numpy as np
import h5py
import os
from utils import *
import time


class ReplayMemory:

    def __init__(self, name, buffer_capacity = 512, storage_capacity = 2048):
        self.buffer_capacity = buffer_capacity
        self.storage_capacity = storage_capacity
        self.buffer_counter = 0
        self.name = name
        self.memories = []
        self.obs_shape = [210, 160, 3]

        if not os.path.exists("replay_buffer"):
            os.makedirs("replay_buffer")

        if os.path.exists("replay_buffer/" + self.name + ".hdf5"):
            with h5py.File("replay_buffer/" + self.name + ".hdf5", "r") as f:
                if self.name == "controller":
                    self.memories.append(f['obs_t'][-self.buffer_capacity:])
                    self.memories.append(np.array(f['goal_xy'][-self.buffer_capacity:], dtype = np.uint8))
                    self.memories.append(np.array(f['action'][-self.buffer_capacity:], dtype = np.uint8))
                    self.memories.append(f['reward'][-self.buffer_capacity:])
                    self.memories.append(f['obs_tp1'][-self.buffer_capacity:])
                elif self.name == "metacontroller":
                    self.memories.append(f['obs_t'][-self.buffer_capacity:])
                    self.memories.append(np.array(f['goal_idx'][-self.buffer_capacity:], dtype = np.uint8))
                    self.memories.append(f['reward'][-self.buffer_capacity:])
                    self.memories.append(f['obs_tp1'][-self.buffer_capacity:])

        else:
            with h5py.File("replay_buffer/" + self.name + ".hdf5", "w") as f:
                f.create_dataset("obs_t", [0] + self.obs_shape, maxshape = [None] + self.obs_shape)
                f.create_dataset("reward", (0,), maxshape = [None,])
                f.create_dataset("obs_tp1", [0] + self.obs_shape, maxshape = [None] + self.obs_shape)
                if self.name == "controller":
                    f.create_dataset("goal_xy", [0, 2], maxshape = [None, 2])
                    f.create_dataset("action", (0,), maxshape = [None,])
                elif self.name == "metacontroller":
                    f.create_dataset("goal_idx", (0,), maxshape = [None,])
                
    def store(self, args):
        # Input:
        #   args: list of items to store; [observation, goal_xy, action, intrinsic reward, next_observation] for controller 
        #         and [observation, goal_idx, external reward, next_observation] for metacontroller

        if self.buffer_counter == self.buffer_capacity:
            self.write_to_file()
            self.buffer_counter = 0

        # store memories to buffer
        if len(self.memories) == 0:
            self.memories += [args[0][np.newaxis, :, :, :]]
            for i in range(1, len(args) - 1, 1):
                self.memories += [args[i]]
            self.memories += [args[-1][np.newaxis, :, :, :]]
        else:
            args[0] = args[0][np.newaxis, :, :, :]
            args[-1] = args[-1][np.newaxis, :, :, :]
            for i in range(len(args)):
                if type(args[i]) == np.ndarray:
                    self.memories[i] = np.concatenate([self.memories[i], args[i]])
                else:
                    self.memories[i] = np.append(self.memories[i], args[i])
                
        self.buffer_counter += 1

        # if number of memories exceed buffer_capacity, cycle through buffer of memories
        if len(self.memories[0]) > self.buffer_capacity:
            for i in range(len(self.memories)):
                self.memories[i] = self.memories[i][1:]

    def write_to_file(self):
        with h5py.File("replay_buffer/" + self.name + ".hdf5", 'a') as f:

            if self.name == "controller":

                obs_t = f['obs_t']
                goal_xy = f['goal_xy']
                action = f['action']
                reward = f['reward']
                obs_tp1 = f['obs_tp1']

                if obs_t.shape[0] + len(self.memories[0]) > self.storage_capacity:
                    addSize = len(self.memories[0])
                    keepSize = self.storage_capacity - addSize

                    # shift old data to keep to the top
                    obs_t[:keepSize] = obs_t[-keepSize:]
                    goal_xy[:keepSize] = goal_xy[-keepSize:]
                    action[:keepSize] = action[-keepSize:]
                    reward[:keepSize] = reward[-keepSize:]
                    obs_tp1[:keepSize] = obs_tp1[-keepSize:]

                    if obs_t.shape[0] < self.storage_capacity:
                        obs_t.resize(tuple([self.storage_capacity] + self.obs_shape))
                        goal_xy.resize((self.storage_capacity, 2))
                        action.resize((self.storage_capacity,))
                        reward.resize((self.storage_capacity,))
                        obs_tp1.resize(tuple([self.storage_capacity] + self.obs_shape))

                else:

                    obs_t.resize(tuple([obs_t.shape[0] + len(self.memories[0])] + self.obs_shape))
                    goal_xy.resize(tuple([goal_xy.shape[0] + len(self.memories[1]), 2]))
                    action.resize((action.shape[0] + len(self.memories[2]),))
                    reward.resize((reward.shape[0] + len(self.memories[3]),))
                    obs_tp1.resize(tuple([obs_tp1.shape[0] + len(self.memories[4])] + self.obs_shape))

                obs_t[-len(self.memories[0]):] = self.memories[0]
                goal_xy[-len(self.memories[1]):] = self.memories[1]
                action[-len(self.memories[2]):] = self.memories[2]
                reward[-len(self.memories[3]):] = self.memories[3]
                obs_tp1[-len(self.memories[4]):] = self.memories[4]

            elif self.name == "metacontroller":

                obs_t = f['obs_t']
                goal_idx = f['goal_idx']
                reward = f['reward']
                obs_tp1 = f['obs_tp1']

                if obs_t.shape[0] + len(self.memories[0]) > self.storage_capacity:
                    addSize = len(self.memories[0])
                    keepSize = self.storage_capacity - addSize

                    # shift old data to keep to the top
                    obs_t[:keepSize] = obs_t[-keepSize:]
                    goal_idx[:keepSize] = goal_idx[-keepSize:]
                    reward[:keepSize] = reward[-keepSize:]
                    obs_tp1[:keepSize] = obs_tp1[-keepSize:]

                    if obs_t.shape[0] < self.storage_capacity:
                        obs_t.resize(tuple([self.storage_capacity] + self.obs_shape))
                        goal_idx.resize((self.storage_capacity,))
                        reward.resize((self.storage_capacity,))
                        obs_tp1.resize(tuple([self.storage_capacity] + self.obs_shape))

                else:

                    obs_t.resize(tuple([obs_t.shape[0] + len(self.memories[0])] + self.obs_shape))
                    goal_idx.resize((goal_idx.shape[0] + len(self.memories[1]),))
                    reward.resize((reward.shape[0] + len(self.memories[2]),))
                    obs_tp1.resize(tuple([obs_tp1.shape[0] + len(self.memories[3])] + self.obs_shape))

                obs_t[-len(self.memories[0]):] = self.memories[0]
                goal_idx[-len(self.memories[1]):] = self.memories[1]
                reward[-len(self.memories[2]):] = self.memories[2]
                obs_tp1[-len(self.memories[3]):] = self.memories[3]

    def sample(self, size):
        # if size > current: return everything
        if size >= len(self.memories[0]):
            if self.name == "controller":

                goal_xy = self.memories[1]
                goal_mask = np.concatenate([convertToBinaryMask([(xy[0] - 5, xy[1] - 5), (xy[0] + 5, xy[1] + 5)])[np.newaxis, :, :, :] for xy in goal_xy], axis = 0)

                obs_t = self.memories[0]
                obs_tp1 = self.memories[-1]
                
                obs_t_goal = np.concatenate([obs_t, goal_mask], axis = 1)
                obs_tp1_goal = np.concatenate([obs_tp1, goal_mask], axis = 1)

                actions = self.memories[2]
                rewards = self.memories[3]

                return [obs_t_goal, actions, rewards, obs_tp1_goal]

            elif self.name == "metacontroller":

                obs_t = self.memories[0]
                goals = self.memories[1]
                rewards = self.memories[2]
                obs_tp1 = self.memories[3]

                return [obs_t, goals, rewards, obs_tp1]

        # randomly sample
        batch_idx = np.random.choice(len(self.memories[0]), size = size, replace = False)

        if self.name == "controller":
            goal_xy_sample = self.memories[1][batch_idx]
            goal_mask_sample = np.concatenate([convertToBinaryMask([(xy[0] - 5, xy[1] - 5), (xy[0] + 5, xy[1] + 5)])[np.newaxis, :, :, :] for xy in goal_xy_sample], axis = 0)

            obs_t = self.memories[0][batch_idx]
            obs_tp1 = self.memories[-1][batch_idx]

            obs_t_goal_sample = np.concatenate([obs_t, goal_mask_sample], axis = 1)
            obs_tp1_goal_sample = np.concatenate([obs_tp1, goal_mask_sample], axis = 1)

            action_sample = self.memories[2][batch_idx]
            reward_sample = self.memories[3][batch_idx]
            
            return [obs_t_goal_sample, action_sample, reward_sample, obs_tp1_goal_sample]

        elif self.name == "metacontroller":

            return [self.memories[0][batch_idx], self.memories[1][batch_idx], self.memories[2][batch_idx], self.memories[3][batch_idx]]
        
