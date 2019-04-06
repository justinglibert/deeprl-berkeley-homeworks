#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
from __future__ import absolute_import
import os
import pickle
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import gym
import load_policy_eager
def main():
    print('loading and building expert policy')
    policy_fn = load_policy_eager.load_policy("experts/Ant-v2.pkl")
    print('loaded and built')
    env = gym.make("Ant-v2")
    max_steps =  env.spec.timestep_limit
    returns = []
    observations = []
    actions = []
    for i in range(1):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:].astype(np.float32))
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            #env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
if __name__ == '__main__':
    main()
