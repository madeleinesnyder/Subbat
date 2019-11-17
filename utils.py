import numpy as np

def isInAir(env, original_observation, action, last_action):

    test_action = 0
    clone_state = env.clone_full_state()
    for _ in range(2):
        observation, reward, done, info = env.step(test_action)
        test_observation, reward, done, info = env.step(test_action)
    env.restore_full_state(clone_state)

    #replace treadmill in picture
    observation[135:142,59:101,:] = np.zeros((7,42,3))
    test_observation[135:142,59:101,:] = np.zeros((7,42,3))

    #replace header in picture
    observation[:20,:,:] = np.zeros((20,160,3))
    test_observation[:20,:,:] = np.zeros((20,160,3))

    #replace skull in picture
    observation[165:180,52:114,:] = np.zeros((15, 62, 3))
    test_observation[165:180,52:114,:] = np.zeros((15, 62, 3))

    #replace key in picture
    observation[98:116,10:23,:] = np.zeros((18, 13, 3))
    test_observation[98:116,10:23,:] = np.zeros((18, 13, 3))

    #add in key logic
    key_test_observation = test_observation[23:45,53:67,:]
    key_original_observation = original_observation[23:45,53:67,:]

    if np.any(key_original_observation - key_test_observation):
        return True

    if not np.any(test_observation - observation):
        return False

    treadmill_observation = original_observation[135:136,60:100,:]
    valid_jumps = [1,10,11,12,14,15,16,17]
    if np.any(treadmill_observation) and action not in valid_jumps and last_action not in valid_jumps:
        return False

    return True


def getJumpOutcome(env, original_lives):
    #outcomes: death (-1), no death (0), reward (1)

    action = 0
    clone_state = env.clone_full_state()
    while True:
        obs, reward, done, info = env.step(action)
        lives = info['ale.lives']
        if reward != 0:
            env.restore_full_state(clone_state)
            return 1
        if lives < original_lives:
            env.restore_full_state(clone_state)
            return -1
        if not isInAir(env, obs, action, action):
            obs, reward, done, info = env.step(action)
            lives = info['ale.lives']
            if lives < original_lives:
                env.restore_full_state(clone_state)
                return -1
            env.restore_full_state(clone_state)
            return 0
