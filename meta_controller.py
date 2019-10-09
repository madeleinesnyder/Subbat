import tensorflow as tf

class MetaController:

    def __init__(self, learning_rate, epsilon = 1):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def anneal(self):
        if self.epsilon > 0.1:
            self.epsilon -= (1 - 0.1)/50000

    def epsGreedy(self, x, Goals):
        if random.randint(0,1) <= self.eps:
            indx = random.randint(0, len(B) - 1)
            return Goals[indx]
        else:
            return self.get_goal(x)

    def build_graph(self, input_placeholder):
        #input should be images and binary goal mask
        layer1 = tf.kersas.layers.Conv2D(input_shape = (210,160,3), filters = 8, kernel_size = (32,32), strides = 4, activation = 'relu')
        layer2 = tf.kersas.layers.Conv2D(filters = 4, kernel_size = (64,64), strides = 2, activation = 'relu')
        layer3 = tf.kersas.layers.Conv2D(filters = 3, kernel_size = (64,64), strides = 1, activation = 'relu')
        #layer4 =


    def loss(self):
        pass

    def update(self, memory):

        #should look like self.sess.run(self.train_op, observations, actions)
        pass

    def get_goal(self):
        pass
