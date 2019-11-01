import tensorflow as tf

class MetaController:

    def __init__(self, sess, hparams):
        self.sess = sess
        self.learning_rate = hparams["learning_rate"]
        self.epsilon = hparams["epsilon"]
        self.goal_dim = hparams['goal_dim']
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

    def epsGreedy(self, goals, observation):
        if random.randint(0,1) <= self.eps:
            indx = random.randint(0, len(goals) - 1)
            return goals[indx]
        else:
            return self.get_goal(observation)

    def build_graph(self, input_placeholder, output_size):
        # input should be observations (images)
        # output_size should be dimension of goal space
        layer1 = tf.layers.conv2d(input_placeholder, filters = 32, kernel_size = (8,8), strides = 4, activation = 'relu')
        layer2 = tf.layers.conv2d(layer1, filters = 64, kernel_size = (4,4), strides = 2, activation = 'relu')
        layer3 = tf.layers.conv2d(layer2, filters = 64, kernel_size = (3,3), strides = 1, activation = 'relu')
        layer4 = tf.layers.dense(layer3, units = 512, activation = 'relu')

        self.q_t_values = tf.layers.dense(layer4, units = output_size)

    def define_train_op(self):
        q_predictions = tf.reduce_sum(self.q_t_values * tf.one_hot(self.goals, self.goal_dim), axis=1)
        self.loss = tf.losses.mean_squared_error(self.targets, q_predictions)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def update(self, observations, goals, targets):
        self.sess.run(self.train_op, feed_dict = {self.obs_ph: observations, self.goals: goals, self.targets: targets})

    def get_goal(self, observations):
        goals = tf.math.argmax(self.q_t_values, axis = 1)
        return self.sess.run(goals, feed_dict = {self.obs_ph: observations})

    def get_q_vals(self, x):
        return self.sess.run(self.q_t_values, feed_dict = {self.obs_ph: x})
