import numpy as np

class ActionReplayBuffer:

    def __init__(self,capacity = np.float("inf")):
        self.capacity = capacity
        self.memories = set()
        self.Goals = []
        self.ARP_dict = {}

    def store(self,arg1,arg2,arg3):
        '''
        Stores each state-action-reward sample from the environment
        If the capacity of the buffer is exceeded, delete the first sample
        '''
        # arg1 - state, arg2 - action, arg3 - reward
        memory = [arg1, arg2, arg3]
        # If this new memory is already in the set of the memories, don't put it in.
        if memory in set(self.memory):
            continue
        else:
            self.memory.append(memory)
            if len(self.memories) > self.capacity:
                self.memories = self.memories[1:]

    def find_Goals():
        for memory in self.memories:
            ARP_dict[memory[0]].append(memory[1:])

        for key, value in self.ARP_dict.items():
            unique_arps = set(value)
            if len(unique_arps) > 1:
                self.Goals.append(key)
        return self.Goals
