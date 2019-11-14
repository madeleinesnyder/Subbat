import numpy as np
import sys
from Action_Replay_Buffer import *

def intrinsic_reward(obs, goal, ARP):
	if ARP.at_subgoal == True:
		return 1
	return 0

def controller_targets(rewards, next_observations, controller, discount):
	#next_observations = np.array([obs[0] for obs in next_observations])
	q_values = controller.get_q_vals(next_observations)
	targets = rewards + discount * np.max(q_values, axis = 1)
	return targets

def meta_controller_targets(rewards, next_observations, meta_controller, discount):
	q_values = meta_controller.get_q_vals(next_observations)
	targets = rewards + discount * np.max(q_values, axis = 1)
	return targets

def random_goal(Goals):
	return Goals[np.random.randint(low = 0, high = len(Goals))]

def convertToBinaryMask(subgoal_coordinates):
	# Input:
	#	subgoal_coordinates: list of two tuples. First tuple (r, c) coordinates of top left corner of box
	#	Second tuple (r, c) coordinates of bottom right corner of box.
	# Output:
	#	3D numpy array of shape (210, 160, 3), with entries in region specified by subgoal_coordinates set to 1
	#	and 0 elsewhere.
	gameSize = (210,160,3)
	topLeft = subgoal_coordinates[0]
	bottomRight = subgoal_coordinates[1]
	upper_y_bound = topLeft[0]
	lower_y_bound = bottomRight[0]
	left_x_bound = topLeft[1]
	right_x_bound = bottomRight[1]
	mask = np.zeros(gameSize)
	mask[upper_y_bound:lower_y_bound, left_x_bound:right_x_bound, :] = 1
	return mask

def convertToSubgoalCoordinates(mask):
	# Input:
	#	mask: 3D numpy array of shape (210, 160, 3) for binary mask of subgoals
	# Output:
	#	subgoal_coordinates: tuple (r, c) coordinates, the centroid of region of 1's
	one_idx = np.where(mask[:, :, 0] == 1)
	row_coord = int((min(one_idx[0]) + max(one_idx[0])) / 2)
	col_coord = int((min(one_idx[1]) + max(one_idx[1])) / 2)
	return (row_coord, col_coord)
