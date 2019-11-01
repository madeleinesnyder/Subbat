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
        if len(self.memories) < 1:
            return self.Goals
        for memory in self.memories:
            self.ARP_dict[memory[0]].append(memory[1:])

        for key, value in self.ARP_dict.items():
            unique_arps = set(value)
            if len(unique_arps) > 1:
                self.Goals.append(key)
        return self.Goals

    def find_Goal_xy(self):
        '''
        How to find key?? TODO
        '''
        for goal in Goals:
            action = 4 #
            obs,reward,done,info = env.step(action)
            goal_xy = (goal-obs)
            subgoal_locations.append(goal_xy)


    def random_Goal(self):
        random_state = random.getstate()
        return random_state

    #def at_subgoal(self,env,)
        # TODO Write this function
