import numpy as np
from collections import defaultdict

class ActionReplayBuffer:

    '''
    TODO: Find location of subgoal generator that observation starting from where the subgoal is and take 1 action, subtract the 2 and that's the location of the subgoals
    This gives 3d matrix of subgoal, go into 2d.
    Radius calculation
    '''

    def __init__(self,capacity = np.float("inf")):
        self.capacity = capacity
        self.memories = set()
        self.Goals = []
        self.ARP_dict = defaultdict(list)
        self.subgoal_locations = [] # TODO Write this function

    def store(self,arg1,arg2,arg3):
        '''
        Stores each state-action-reward sample from the environment
        If the capacity of the buffer is exceeded, delete the first sample
        '''
        # arg1 - state, arg2 - action, arg3 - reward.
        # Convert to a tuple to store in memories
        memory = [arg1, arg2, arg3]
        memory = tuple(memory)
        # If this new memory is already in the set of the memories, don't put it in.
        self.memories.add(memory)
        if len(self.memories) > self.capacity:
            self.memories = self.memories[1:]

    def find_Goals(self):
        '''
        Find the states that corrospond to goals.
        '''
        if len(self.memories) < 1:
            return self.Goals
        for memory in self.memories:
            self.ARP_dict[memory[0]].append(memory[1:])

        for key, value in self.ARP_dict.items():
            unique_arps = set(value)
            if len(unique_arps) > 1:
                self.Goals.append(key)
        return self.Goals

    def get_Goal_xy(self,env,observation):
        '''
        How to find key?? TODO
        Get the xy location of the goal from the state
        Fix this unwrapping shit.
        '''
        env = env.unwrapped
        clone_state = env.clone_full_state()
        obs = observation
        action = 4
        next_obs,reward,done,info = env.step(action)
        rgb_coords = next_obs-obs
        nonzero_coords = np.where(rgb_coords[:,:,0] != 0)
        [mean_x, mean_y] = [np.mean(nonzero_coords[0]),np.mean(nonzero_coords[1])]
        coord_tuple = (int(np.ceil(mean_x)),int(np.ceil(mean_y)))
        self.subgoal_locations.append(coord_tuple)
        env.restore_full_state(clone_state)

    def at_subgoal(self,observation):
        '''
        Is ALE near enough to a subgoal?
        '''
        k = 10
        for subgoal_loc in self.subgoal_locations:
            if np.linalg.norm(observation - subgoal_loc) < k:
                return True
            else:
                return False

    def load_temp_subgoals(self,sg_file):
        '''
        For testing purposes only. Loading in Calvin's list of subgoals npy array
        '''
        sgs = np.load(sg_file)
        for i in range(len(sgs)):
            self.subgoal_locations.append(sgs[i])

    def random_Goal(self):
        '''
        Get a random subgoal to initialize the exploration run in env.
        '''
        idx = np.random.randint(len(self.subgoal_locations))
        return self.subgoal_locations[idx]
