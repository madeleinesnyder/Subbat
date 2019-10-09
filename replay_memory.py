import numpy as np

class ReplayMemory:

    def __init__(self, capacity = np.float("inf")):
        self.capacity = capacity
        self.memories = []

    def store(self, arg1, arg2, arg3, arg4):
        memory = [arg1, arg2, arg3, arg4]
        self.memory.append(memory)
        if len(self.memories) > self.capacity:
            self.memories = self.memories[1:]

    def sample(self, size):
        pass
