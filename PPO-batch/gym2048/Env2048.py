import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym2048.game_2048 import Game2048

class Env2048(gym.Env):

    def __init__(self):
        # self.init_state = init_state
        # self.size = size 
        self.game = Game2048()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0,high=20.0,shape=(16,),dtype=np.float32)
        # self.observation_space = spaces.MultiDiscrete([18]*16)
        # self.observation_space = spaces.Discrete(18,shape=(4,4))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        # costs = np.sum(u**2) + np.sum(self.state**2)
        # self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        # return self._get_obs(), -costs, False, {}
        next_board, reward, done = self.game.step(u)
        return self._get_obs(), reward, done, {}

    def reset(self):
        # high = self.init_state*np.ones((self.size,))
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.last_u = None
        self.game.reset()
        return self._get_obs()

    def _get_obs(self):
        # return self.state
        self.board = self.game.get_state()
        return np.log2(np.where(self.board>0, self.board, 1)).flatten()

    def get_state(self):
        return self._get_obs()

    def get_score(self):
        return self.game.score

    def get_board(self):
        return np.array(self.game.get_state())

    def render(self, mode='human'):
        return str(np.array(self.game.get_state()).reshape((4,4))) + '\n' + str(self.game.score)

    def seed(self,seed):
        return 

class Env2048soft(gym.Env):

    def __init__(self):
        # self.init_state = init_state
        # self.size = size 
        self.game = Game2048()
        self.isTraining = False
        self.action_space = spaces.Box(low=0, high=1, shape=(4,))
        self.observation_space = spaces.MultiDiscrete([18]*16)
        # self.observation_space = spaces.Discrete(18,shape=(4,4))

    def set_training(self, flag):
        self.isTraining = flag

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, p):
        if self.isTraining:
            u = np.random.choice([0,1,2,3], p=p)
        else:
            u = np.argmax(p)
        next_board, reward, done = self.game.step(u)
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.game.reset()
        return self._get_obs()

    def _get_obs(self):
        self.board = self.game.get_state()
        return np.log2(np.where(self.board>0, self.board, 1)).flatten()

    def get_state(self):
        return self._get_obs()

    def get_board(self):
        return np.array(self.game.get_state())

    def get_score(self):
        return self.game.score

    def render(self, mode='human'):
        return str(np.array(self.game.get_state()).reshape((4,4))) + '\n' + str(self.game.score)

def make_onehot(a):
    return (np.arange(18)==a[:,None]).astype(np.int).astype(np.float)

class Env2048onehot(gym.Env):

    def __init__(self):
        # self.init_state = init_state
        # self.size = size 
        self.game = Game2048()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(18*16,))
        # self.observation_space = spaces.Discrete(18,shape=(4,4))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        # costs = np.sum(u**2) + np.sum(self.state**2)
        # self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        # return self._get_obs(), -costs, False, {}
        next_board, reward, done = self.game.step(u)
        return self._get_obs(), reward, done, {}

    def reset(self):
        # high = self.init_state*np.ones((self.size,))
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.last_u = None
        self.game.reset()
        return self._get_obs()

    def _get_obs(self):
        # return self.state
        self.board = self.game.get_state()
        chess_labels = np.log2(np.where(self.board>0, self.board, 1)).astype(np.int).flatten()
        return make_onehot(chess_labels).flatten()

    def get_state(self):
        return self._get_obs()

    def get_board(self):
        return np.array(self.game.get_state())

    def get_score(self):
        return self.game.score

    def render(self, mode='human'):
        return str(np.array(self.game.get_state()).reshape((4,4))) + '\n' + str(self.game.score)

# gym.envs.register(
#      id='Env2048',
#      entry_point='gym.envs.classic_control:Env2048',
#      max_episode_steps=5000,
# )

# if __name__ == '__main__':
#     env = gym.make('Env2048-v0')