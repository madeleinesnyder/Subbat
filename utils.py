import numpy as np
import sys
from Action_Replay_Buffer import *

def intrinsic_reward(obs, goal, ARP):
    if ARP.at_subgoal == True:
        return 1
    return 0

def achieved_subgoal(env, observation, goal_xy):
    coords = get_ALE_coord(env, observation)
    k = 10 # Because Calvin says so.
    if np.linalg.norm(coords - goal_xy) < k:
        return True
    else:
        return False

def get_ALE_coord(env, observation):
    obs = observation
    for action in [4,5,11]:
        rgb_coords = action_difference_of_frames(env,action,obs)
        if np.sum(rgb_coords) > 0:
            action_used = action
            rgb_coords = rgb_coords
            break
    if (action == 11 and np.sum(rgb_coords) == 0): #note to change to 11 later
        return (-1,-1)
    nonzero_coords = np.where(rgb_coords[:,:,0] != 0)
    [mean_y, mean_x] = [np.mean(nonzero_coords[0]),np.mean(nonzero_coords[1])]
    coords = (float(mean_x),float(mean_y))
    return coords

def action_difference_of_frames(env, action, obs):
    observation = obs
    env = env.unwrapped
    clone_state = env.clone_full_state()
    test_action = action
    for _ in range(3):
        next_observation, reward, done, info = env.step(test_action)

        # Black box things that move
        #replace treadmill in picture
        observation[135:142,59:101,:] = np.zeros((7,42,3))
        next_observation[135:142,59:101,:] = np.zeros((7,42,3))

        #replace header in picture
        observation[:20,:,:] = np.zeros((20,160,3))
        next_observation[:20,:,:] = np.zeros((20,160,3))

        #replace skull in picture
        observation[165:180,52:114,:] = np.zeros((15, 62, 3))
        next_observation[165:180,52:114,:] = np.zeros((15, 62, 3))

        #replace key in picture
        observation[98:116,10:23,:] = np.zeros((18, 13, 3))
        next_observation[98:116,10:23,:] = np.zeros((18, 13, 3))

        rgb_coords = next_observation-observation
        if np.sum(rgb_coords) != 0:
            env.restore_full_state(clone_state)
            return rgb_coords

        observation = next_observation

    env.restore_full_state(clone_state)
    return np.zeros(1)

def controller_targets(rewards, next_observations, controller, discount):
    #next_observations = np.array([obs[0] for obs in next_observations])
    q_values = controller.get_q_vals(next_observations)
    targets = rewards + discount * np.max(q_values, axis = 1)
    return targets

def meta_controller_targets(rewards, next_observations, meta_controller, discount):
    q_values = meta_controller.get_q_vals(next_observations)
    targets = rewards + discount * np.max(q_values, axis = 1)
    return targets

def random_goal_idx(goal_dim):
    return np.random.randint(low = 0, high = goal_dim)

def convertToBinaryMask(subgoal_coordinates):
    # Input:
    #   subgoal_coordinates: list of two tuples. First tuple (r, c) coordinates of top left corner of box
    #   Second tuple (r, c) coordinates of bottom right corner of box.
    # Output:
    #   3D numpy array of shape (210, 160, 3), with entries in region specified by subgoal_coordinates set to 1
    #   and 0 elsewhere.
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
    #   mask: 3D numpy array of shape (210, 160, 3) for binary mask of subgoals
    # Output:
    #   subgoal_coordinates: tuple (r, c) coordinates, the centroid of region of 1's
    one_idx = np.where(mask[:, :, 0] == 1)
    row_coord = int((min(one_idx[0]) + max(one_idx[0])) / 2)
    col_coord = int((min(one_idx[1]) + max(one_idx[1])) / 2)
    return (row_coord, col_coord)

