import numpy as np
import h5py
import os


class ReplayMemory:

    def __init__(self, name, write_capacity = 20, capacity = 1000):
        self.capacity = capacity
        self.write_capacity = write_capacity
        self.name = name
        self.memories = []
        # TEST
        self.ret = []
        if not os.path.exists("replay_buffer"):
            os.makedirs("replay_buffer")

        if os.path.exists("replay_buffer/" + self.name + ".hdf5"):
            with h5py.File("replay_buffer/" + self.name + ".hdf5", 'r') as f:
                self.memories = [f['obs_t_goal'][:], f['goal_or_action'][:], f['reward'][:], f['obs_tp1_goal'][:]]
        else:
            with h5py.File("replay_buffer/" + self.name + ".hdf5", "w") as f:
                if self.name == "controller":
                    self.obs_goal_shape = [0, 420, 160, 3]
                elif self.name == "metacontroller":
                    self.obs_goal_shape = [0, 210, 160, 3]
                f.create_dataset("obs_t_goal", self.obs_goal_shape, maxshape = [None] + self.obs_goal_shape[1:])
                f.create_dataset("goal_or_action", (0,), maxshape = [None,])
                f.create_dataset("reward", (0,), maxshape = [None,])
                f.create_dataset("obs_tp1_goal", self.obs_goal_shape, maxshape = [None] + self.obs_goal_shape[1:])


    def store(self, arg1, arg2, arg3, arg4):
        if len(self.memories) == 0:
            self.memories = [arg1[np.newaxis, :, :, :], np.array([arg2]), np.array([arg3]), arg4[np.newaxis, :, :, :]]
        else:
            self.memories[0] = np.concatenate((self.memories[0], arg1[np.newaxis, :, :, :]), axis = 0)
            self.memories[1] = np.append(self.memories[1], arg2)
            self.memories[2] = np.append(self.memories[2], arg2)
            self.memories[3] = np.concatenate((self.memories[3], arg4[np.newaxis, :, :, :]), axis = 0)

        #if self.memories[0].shape[0] > self.write_capacity:
            #self.memories[0] = self.memories[0][1:]
            #self.memories[1] = self.memories[1][1:]
            #self.memories[2] = self.memories[2][1:]
            #self.memories[3] = self.memories[3][1:]
        if self.memories[0].shape[0] == self.write_capacity and len(self.ret) == 0:
            self.ret = self.memories
            
        #return

        if self.memories[0].shape[0] >= self.write_capacity:
            with h5py.File("replay_buffer/" + self.name + ".hdf5", 'a') as f:
                obs_t_goal = f['obs_t_goal']
                goal_or_action = f['goal_or_action']
                reward = f['reward']
                obs_tp1_goal = f['obs_tp1_goal']

                if obs_t_goal.shape[0] >= self.capacity:
                    obs_t_goal[:-self.write_capacity] = obs_t_goal[self.write_capacity:]
                    obs_t_goal[-self.write_capacity:] = self.memories[0]

                    goal_or_action[:-self.write_capacity] = goal_or_action[self.write_capacity:]
                    goal_or_action[-self.write_capacity:] = self.memories[1]

                    reward[:-self.write_capacity] = reward[self.write_capacity:]
                    reward[-self.write_capacity:] = self.memories[2]

                    obs_tp1_goal[:-self.write_capacity] = obs_tp1_goal[self.write_capacity:]
                    obs_tp1_goal[-self.write_capacity:] = self.memories[3]

                else:
                    obs_t_goal.resize(tuple([obs_t_goal.shape[0] + self.write_capacity] + self.obs_goal_shape[1:]))
                    obs_t_goal[-self.write_capacity:] = self.memories[0]

                    goal_or_action.resize((goal_or_action.shape[0] + self.write_capacity, ))
                    goal_or_action[-self.write_capacity:] = self.memories[1]

                    reward.resize((reward.shape[0] + self.write_capacity, ))
                    reward[-self.write_capacity:] = self.memories[2]
                
                    obs_tp1_goal.resize(tuple([obs_tp1_goal.shape[0] + self.write_capacity] + self.obs_goal_shape[1:]))
                    obs_tp1_goal[-self.write_capacity:] = self.memories[3]
                print("Memories stored: {0}".format(obs_tp1_goal.shape[0]))

            self.memories = []

    def sample(self, size):
        # TEST
        batch_idx = np.random.choice(self.memories[0].shape[0], size = size, replace = False)
        return [self.memories[0][batch_idx], self.memories[1][batch_idx], self.memories[2][batch_idx], self.memories[3][batch_idx]]

        with h5py.File("replay_buffer/" + self.name + ".hdf5", 'r') as f:
            obs_t_goal = f['obs_t_goal']
            goal_or_action = f['goal_or_action']
            reward = f['reward']
            obs_tp1_goal = f['obs_tp1_goal']

            if obs_t_goal.shape[0] == 0:
                memory_sample = self.memories
            else:

                if size >= obs_t_goal.shape[0]:
                    memory_sample = [obs_t_goal[:], goal_or_action[:], reward[:], obs_tp1_goal[:]]
                else:
                    batch_idx = sorted(np.random.choice(obs_t_goal.shape[0], size = size, replace = False))
                    memory_sample = [obs_t_goal[batch_idx], goal_or_action[batch_idx], reward[batch_idx], obs_tp1_goal[batch_idx]]

        return memory_sample
        
