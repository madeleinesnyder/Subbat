import numpy as np

class ReplayMemory:

    def __init__(self, capacity = np.float("inf")):
        self.capacity = capacity
        #self.memories = np.empty(shape = (0, 4))
        self.memories = []

    def store(self, arg1, arg2, arg3, arg4):
        if len(self.memories) == 0:
            self.memories = [arg1[np.newaxis, :, :, :], np.array(arg2), np.array(arg3), arg4[np.newaxis, :, :, :]]
        self.memories[0] = np.concatenate((self.memories[0], arg1[np.newaxis, :, :, :]), axis = 0)
        self.memories[1] = np.append(self.memories[1], arg2)
        self.memories[2] = np.append(self.memories[2], arg2)
        self.memories[3] = np.concatenate((self.memories[3], arg4[np.newaxis, :, :, :]), axis = 0)

        if self.memories[0].shape[0] > self.capacity:
            self.memories[0] = self.memories[0][1:]
            self.memories[1] = self.memories[1][1:]
            self.memories[2] = self.memories[2][1:]
            self.memories[3] = self.memories[3][1:]

    def sample(self, size):
        if size >= self.memories[0].shape[0]:
            return self.memories
        batch_idx = np.random.choice(self.memories[0].shape[0], size = size, replace = False)
        memory_sample = [self.memories[0][batch_idx], self.memories[1][batch_idx],
            self.memories[2][batch_idx], self.memories[3][batch_idx]]
        return memory_sample
        
