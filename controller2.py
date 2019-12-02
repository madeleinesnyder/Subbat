import tensorflow as tf
import numpy as np
import sys

class Controller:

    def __init__(self, sess, hparams):
        self.sess = sess
        self.learning_rate = hparams["learning_rate"]
        self.epsilon = hparams["epsilon"]
        self.action_dim = hparams['action_dim']
        self.input_shape = hparams['input_shape']
        self.define_placeholders()
        self.build_graph(self.obs_g_ph, self.action_dim)
        self.define_train_op()
        tf.global_variables_initializer().run(session=sess)

    def define_placeholders(self):
        self.obs_g_ph = tf.placeholder(dtype=tf.float32, shape = [None] + list(self.input_shape), name = "observations")
        self.targets = tf.placeholder(dtype=tf.float32, shape = [None], name = "targets")
        self.actions = tf.placeholder(dtype=tf.int32, shape = [None], name = "actions")

    def anneal(self):
        if self.epsilon > 0.1:
            #self.epsilon -= (1 - 0.1)/50000
            self.epsilon -= (1 - 0.1) / 100

    def epsGreedy(self, x, Actions):
        if np.random.random() <= self.epsilon:
            return Actions.sample()
        else:
            return np.squeeze(self.get_action(x))

    def build_graph(self, input_placeholder, output_size):
        #input should be observations (images)
        # output_size should be dimension of action space
        layer1 = tf.layers.dense(input_placeholder, units = 32, activation = 'relu')
        self.q_t_values = tf.layers.dense(layer1, units = output_size)

    def define_train_op(self):
        q_predictions = tf.reduce_sum(self.q_t_values * tf.one_hot(self.actions, self.action_dim), axis=1)
        self.loss = tf.losses.mean_squared_error(self.targets, q_predictions)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def update(self, observations_goals, actions, targets):
        #observations_goals = np.concatenate(tuple(observations_goals), axis = 0)
        self.sess.run(self.train_op, feed_dict = {self.obs_g_ph: observations_goals, self.actions: actions, self.targets: targets})

    def get_action(self, x):
        # x = [observations, goals]
        #observations_goals = np.concatenate((x[0], x[1]), axis = 0)
        actions = tf.math.argmax(self.q_t_values, axis = 1)
        #return self.sess.run(actions, feed_dict = {self.obs_g_ph: observations_goals})
        return self.sess.run(actions, feed_dict = {self.obs_g_ph: x})

    def get_q_vals(self, x):
        # x = [observations, goals]
        # observations_goals = np.concatenate((x[0], x[1]), axis = 0)
        # return self.sess.run(self.q_t_values, feed_dict = {self.obs_g_ph: observations_goals})
        return self.sess.run(self.q_t_values, feed_dict = {self.obs_g_ph: x})


