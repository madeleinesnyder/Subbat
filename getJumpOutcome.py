from isInAir import *
import numpy as np

def getJumpOutcome(env, original_lives):
    #outcomes: death (-1), no death (0), reward (1)

    action = 0
    env = env.unwrapped
    clone_state = env.clone_full_state()
    while True:
        obs, reward, done, info = env.step(action)
        lives = info['ale.lives']
        if reward != 0:
            env.restore_full_state(clone_state)
            return 1
        if lives < original_lives:
            env.restore_full_state(clone_state)
            return -100
        if not isInAir(env, obs):
            obs, reward, done, info = env.step(action)
            lives = info['ale.lives']
            if lives < original_lives:
                env.restore_full_state(clone_state)
                return -100
            env.restore_full_state(clone_state)
            return 0
    #Key reward is gained while isInAir is True. For skull case, self.lives
    #decremented at same time isInAir is false. For falling to death case,
    #isInAir if False one frame before self.lives is decremented

    return
