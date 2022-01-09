import os
import copy
import random
import platform, multiprocessing
import numpy as np
from fire import Fire
import gym
import gym2048
import time
from mcts_agent import MCTSAgent 
from ntuple.tuple_util import getValue
from mcts_agent import Once_env
from mcts_agent import Q_value
from mcts_agent import raw_obs

def stat(results,workers,trial):
    final_results = np.zeros(6)
    for res in results:
        final_results += res
    final_results /= (workers*trial)
    return final_results


def worker(rank=None, results=None, trial=1, gamma=1, c=0, iter_time=0.01, d=10, ntuple=True, render=True):
    # if platform.system() == 'Darwin':
    #     multiprocessing.set_start_method('spawn')

    env = gym.make("Env2048-v0")
    env = env.unwrapped
    if results is None:
        results = [[]]
        rank = 0
    results[rank] = np.zeros(6) # count the time reaching 512,1024,2048,4096,8192,16384

    # you can fix the seed for debugging, but your agent SHOULD NOT overfit to the env of a certain seed
    #env.seed(seed)
    # render is automatically set to False for copied envs
    # remember to call reset() before calling step()
    for _ in range(trial):
        obs = env.reset()
        obs, rew, done, info = env.step(0)
        done = False
        step_cnt = 0

        while not done:
            # copy the observation for simulate    
            agent = MCTSAgent(env, gamma=gamma, c=c, iter_time=iter_time, ntuple = ntuple)
            obs = copy.deepcopy(obs)

            # select an action
            action = agent.select_action(obs, d)
            obs_new, rew, done, info = env.step(action)
            # if freeze at the state the select a random action
            if (obs_new == obs).all():
                '''
                print(obs.reshape(4,4))
                print(action)
                Env = Once_env()
                Qs = [Q_value(obs,i,Env) for i in range(4)]
                print(Qs)
                time.sleep(5)
                '''
                action = random.randint(0, 3)
                obs_new, rew, done, info = env.step(action)
                print('select a random action: {}'.format(action))

            obs2board = (2**obs.reshape((4,4)))
            obs2board = np.where(obs2board==1, 0, obs2board).astype(int).tolist()
            if render:
                print('state: \n{}'.format(str(obs2board).replace('], [', '\n').replace(', ', '\t').replace('[', '').replace(']', '')))
                print('do action: {}'.format(['up', 'down', 'left', 'right'][action]))
                print('obtain reward: {}'.format(rew))
            #print('done: {}'.format(done))
            #print('info: {}\n'.format(info))

            # reset the observation
            obs = obs_new
        
        pre_res = results[rank]
        for i in range(int(np.max(obs)-8)):
            pre_res[i] += 1
        results[rank] = pre_res

        if render:
            print(obs)
            print('#' * 70)
            print('game done, final score: {}\n\n'.format(env.get_score()))
        print("worker",rank,"has finished trial",_, results[rank])
        print("worker",rank,"final board is:\n",np.power(2,obs).reshape(4,4))
        print("current stat:",stat(results,50,2))
        print('#'*70)

    # remember to close the env, but you can always let resources leak on your own computer :|
    env.close()

if __name__ == '__main__':
    Fire(worker)