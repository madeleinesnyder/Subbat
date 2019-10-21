import numpy as np

class ReplayMemory:

    def __init__(self, capacity = np.float("inf")):
        self.capacity = capacity
        self.memories = np.empty(shape = (0, 4))

    def store(self, arg1, arg2, arg3, arg4):
        memory = np.array([arg1, arg2, arg3, arg4])
        self.memories = np.vstack((self.memories, memory))
        if len(self.memories) > self.capacity:
            self.memories = self.memories[1:]

    def sample(self, size):
        if size >= len(self.memories):
            return self.memories
        batch_idx = np.random.choice(len(self.memories), size = size, replace = False)
        return self.memories[batch_idx]
        
