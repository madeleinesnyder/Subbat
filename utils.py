import numpy as np

def intrinsic_reward(obs, goal):
	if obs == goal:
		return 1
	return 0

def controller_targets(rewards, next_observations, controller, discount):
	next_observations = np.array([obs[0] for obs in next_observations])
	q_values = controller.get_q_vals(next_observations)
	targets = rewards + discount * np.max(q_values, axis = 1)
	return targets

def meta_controller_targets(rewards, next_observations, meta_controller, discount):
	q_values = meta_controller.get_q_vals(next_observations)
	targets = rewards + discount * np.max(q_values, axis = 1)
	return targets

