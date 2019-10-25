import gym
import random
from replay_memory import ReplayMemory
from meta_controller import MetaController
from controller import Controller
from utils import *
import time

random.seed(42)

learning_rate = 2.5e-4
num_episodes = 10000
# adjust later
num_pre_training_episodes = 5000
discount = 0.99
batch_size = 256

# Goals = [] #TODO

env = gym.make('MontezumaRevenge-v0')
sess = tf.Session()

d1 = ReplayMemory()
d2 = ReplayMemory()

meta_controller_input_shape = env.observation_space.shape
controller_input_shape = env.observation_space.shape
controller_input_shape[0] = controller_input_shape[0] * 2

meta_controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "goal_dim": len(Goals), "input_shape": meta_controller_input_shape}
controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "action_dim": env.action_space.n, "input_shape": controller_input_shape}

meta_controller = MetaController(sess, meta_controller_hparams)
controller = Controller(sess, controller_hparams)

# pretraining step
for i in range(num_pre_training_episodes):
    observation = env.reset()
    goal = random_goal(Goals)
    done = False
    while not done:
        F = 0
        initial_observation = observation
        while not (done or observation == goal):
            #action space is discrete on set {0,1,...,17}
            action = controller.epsGreedy([observation, goal], env.action_space)
            next_observation, f, done, info = env.step(action)
            r = intrinsic_reward(next_observation, goal)

            d1.store([initial_observation, goal], action, r, [next_observation, goal])
            controller_batch = d1.sample(batch_size)
            c_targets = controller_targets(controller_batch[:, 2], controller_batch[:, 3], controller, discount)
            controller.update(controller_batch[:, 0], controller_batch[:, 1], c_targets)

            F += f
            observation = next_observation
        d2.store(initial_observation, goal, F, next_observation)
        if not done:
            goal = random_goal(Goals)
    controller.anneal()

# main h-DQN algorithm
for i in range(num_episodes):
    observation = env.reset()
    goal = meta_controller.epsGreedy(observation, Goals)
    done = False
        while not done:
            F = 0
            initial_observation = observation
            while not (done or observation == goal):
                #action space is discrete on set {0,1,...,17}
                action = controller.epsGreedy([observation, goal], env.action_space)
                next_observation, f, done, info = env.step(action)
                r = intrinsic_reward(next_observation, goal)

                d1.store([initial_observation, goal], action, r, [next_observation, goal])
                controller_batch = d1.sample(batch_size)
                c_targets = controller_targets(controller_batch[:, 2], controller_batch[:, 3], controller, discount)
                controller.update(controller_batch[:, 0], controller_batch[:, 1], c_targets)

                meta_controller_batch = d2.sample(batch_size)
                m_targets = meta_controller_targets(meta_controller_batch[:, 2], meta_controller_batch[:, 3], meta_controller, discount)
                meta_controller.update(controller_batch[:, 0], controller_batch[:, 1], m_targets)
                F += f
                observation = next_observation
            d2.store(initial_observation, goal, F, next_observation)
            if not done:
                goal = meta_controller.epsGreedy(Goals, obervation)
        meta_controller.anneal()
        controller.anneal()
env.close()


# 0: do nothing
# 1: jump vertically up
# 2: nothing??? (may be an action not able to be used in current position, up maybe)
# 3: move one step right
# 4: move one step left
# 5: move one step down
# 6: looks like another move one step right
# 7: looks like another move one step left
# 8: looks like another move one step right


env = gym.make('MontezumaRevenge-v0')
env.reset()
for i_episode in range(4):
    observation = env.reset()
    for t in range(4):
    	# env.render() opens window of MontezumaRevenge game
        env.render()
        action = env.action_space
        observation, reward, done, info = env.step(8)
        time.sleep(2)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
