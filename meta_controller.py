import tensorflow as tf

class MetaController:

    def __init__(self, sess, hparams, epsilon = 1):
        self.sess = sess
        self.learning_rate = hparams["learning_rate"]
        self.epsilon = hparams["epsilon"]
        self.goal_dim = hparams['goal_dim']
        self.ob_dim = hparams['ob_dim']
        self.input_shape = hparams['input_shape']
        self.define_placeholders()
        self.build_graph(self.obs_ph, self.goal_dim)

    def define_placeholders(self):
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape = [None] + list(self.input_shape), name = "observations")
        self.targets = tf.placeholder(dtype=tf.float32, shape = [None], name = "targets")
        self.goals = tf.placeholder(dtype=tf.float32, shape = [None], name = "goals")

    def anneal(self):
        if self.epsilon > 0.1:
            self.epsilon -= (1 - 0.1)/50000

    def epsGreedy(self, goals, observations):
        if random.randint(0,1) <= self.eps:
            indx = random.randint(0, len(goals) - 1)
            return goals[indx]
        else:
            return self.get_goal(observations)

    def build_graph(self, input_placeholder, output_size):
        #input should be observations (images)
        # output_size should be dimension of goal space
        layer1 = tf.layers.Conv2D(input_placeholder, filters = 8, kernel_size = (32,32), strides = 4, activation = 'relu')
        layer2 = tf.layers.Conv2D(layer1, filters = 4, kernel_size = (64,64), strides = 2, activation = 'relu')
        layer3 = tf.layers.Conv2D(layer2, filters = 3, kernel_size = (64,64), strides = 1, activation = 'relu')
        layer4 = tf.layers.Dense(layer3, units = 512, activation = 'relu')
        self.q_t_values = tf.layers.Dense(layer4, units = output_size)

    def define_train_op(self):
        q_predictions = tf.reduce_sum(self.q_t_values * tf.one_hot(self.goals, self.goal_dim), axis=1)
        self.loss = tf.losses.mean_squared_error(self.targets, q_predictions)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def update(self, observations, goals):
        self.sess.run(self.trian_op, feed_dict = {self.obs_ph: observations, self.goals: goals})

    def get_goal(self, observations):
        q_t_values = self.sess.run(self.q_t_values, feed_dict = {self.obs_ph: observations})
        return tf.math.argmax(q_t_values, axis = 1)



