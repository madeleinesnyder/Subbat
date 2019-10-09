import tensorflow as tf

class Controller:

    def __init__(self, learning_rate, epsilon = 1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def anneal(self):
        if self.epsilon > 0.1:
            self.epsilon -= (1 - 0.1)/50000

    def epsGreedy(self, x, Actions):
        if random.randint(0,1) <= self.eps:
            return Actions.sample()
        else:
            return self.get_action(x)

    def build_graph(self, input_placeholder, goal):
        #input should be images and goal
        layer1 = tf.layers.conv2d()

    def loss(self):
        pass

    def update(self, memory):
        pass

    def get_action(self):
        pass
