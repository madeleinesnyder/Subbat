import numpy as np

def isInAir(env, original_observation):

    #steady state it? Seems to work empirically quite well. I'll add in the treadmill
    #logic and give it some more testing

    test_action = 0
    env = env.unwrapped
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

    if not np.any(test_observation - observation):
        return False

    treadmill_window = original_observation[100:136,50:110,:]
    # img = Image.fromarray(treadmill_window, 'RGB')
    # img.show()
    #if str(treadmill_window) in treadmill_spaces:
    #    return False

    # diff = test_observation - observation
    # img = Image.fromarray(diff, 'RGB')
    # img.show()

    return True
