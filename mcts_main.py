import os
import copy
import random
import platform, multiprocessing
import numpy as np
from fire import Fire
import gym
import gym2048
from mcts_agent import MCTSAgent 

def main(seed=0, gamma=0.99, c=100, iter_time=1, d=10, render=True):
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')

    env = gym.make("Env2048-v0")
    env = env.unwrapped
    # you can fix the seed for debugging, but your agent SHOULD NOT overfit to the env of a certain seed
    env.seed(seed)
    # render is automatically set to False for copied envs
    # remember to call reset() before calling step()
    obs = env.reset()
    obs, rew, done, info = env.step(0)
    done = False

    while not done:
        # copy the observation for simulate    
        agent = MCTSAgent(env, gamma=gamma, c=c, iter_time=iter_time)
        obs = copy.deepcopy(obs)

        # select an action
        action = agent.select_action(obs, d)
        obs_new, rew, done, info = env.step(action)

        # if freeze at the state the select a random action
        if (obs_new == obs).all():
            action = random.randint(0, 3)
            obs_new, rew, done, info = env.step(action)
            print('select a random action: {}'.format(action))

        obs2board = (2**obs.reshape((4,4)))
        obs2board = np.where(obs2board==1, 0, obs2board).astype(int).tolist()
        print('state: \n{}'.format(str(obs2board).replace('], [', '\n').replace(', ', '\t').replace('[', '').replace(']', '')))
        print('do action: {}'.format(['up', 'down', 'left', 'right'][action]))
        print('obtain reward: {}'.format(rew))
        print('done: {}'.format(done))
        print('info: {}\n'.format(info))

        # reset the observation
        obs = obs_new
    
    print('#' * 70)
    print('game done, final score: {}\n\n'.format(env.get_score()))

    # remember to close the env, but you can always let resources leak on your own computer :|
    env.close()

if __name__ == '__main__':
    Fire(main)