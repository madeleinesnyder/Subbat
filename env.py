import gym
import random
from replay_memory import ReplayMemory
from meta_controller import MetaController
from controller import Controller
import time

random.seed(42)

d1 = ReplayMemory()
d2 = ReplayMemory()
#
# Goals = [] #TODO
# meta_controller = MetaController(0.0025)
# controller = Controller(0.00025)

# num_episodes = 10000
# env = gym.make('MontezumaRevenge-v0')
# for i in range(num_episodes):
#     #observation space is 210x160x3
#     observation = env.reset()
#     goal = meta_controller.epsGreedy(observation, Goals)
#     done = False
#     while not done:
#         F = 0
#         initial_observation = observation
#         while not (done or observation == goal):
#             #action space is discrete on set {0,1,...,17}
#             Actions = env.action_space
#             action = controller.epsGreedy([observation, goal], Actions)
#             next_observation, f, done, info = env.step(action)
#             r = 1 if next_observation == goal else 0
#             d1.store([observation, goal], action, r, [next_observation, goal])
#             controller.update(d1)
#             meta_controller.update(d2)
#             F += f
#             observation = next_observation
#         d2.store(initial_observation, goal, F, next_observation)
#         if not done:
#             goal = meta_controller.epsGreedy(observation, Goals)
#     meta_controller.anneal()
#     controller.anneal()
# env.close()


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
