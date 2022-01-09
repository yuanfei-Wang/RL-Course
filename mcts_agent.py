import numpy as np
import random
import copy
import time
from ntuple.tuple_util import getValue

# for mcts_agent, all obs are 16*1 vector, which is logged
# for Once_env, all obs should be 4*4 matrix, which is not logged
# for getValue, all state should be 4*4 matrix, which is logged

class Once_env:
    def __init__(self,board=None):
        self.board = board
        self.clear_score = 0

    def _line_squeeze(self, line, inv):
        # push box
        i, j = 3 if inv else 0, 3 if inv else 0
        d = -1 if inv else 1
        while ((j >= 0) if inv else (j < 4)):
            if line[j] != 0:
                t = line[j]
                line[j] = 0
                line[i] = t
                i += d   
            j += d
        return line
    
    def _line_combine(self, line, inv):
        # combine box with same value
        i = 3 if inv else 0
        d = -1 if inv else 1
        while ((i >= 1) if inv else (i < 3)):
            if line[i] != 0 and line[i] == line[i + d]:
                line[i] += line[i + d]
                self.clear_score += line[i]
                line[i + d] = 0
                i += d
            i += d
        return line

    def next_state(self, op_num):
        board = self.board.copy()
        self.clear_score = 0
        inv = (op_num % 2 == 1) # inverse direction
        f = lambda line: self._line_squeeze(self._line_combine(self._line_squeeze(line, inv), inv), inv)
        for i in range(4):
            if op_num < 2:
                board[:,i] = f(board[:,i])
            else:
                board[i,:] = f(board[i,:])
        if np.sum(np.abs(board-self.board))==0:
            reward = -1e8
        else:
            reward = self.clear_score
        return board, reward

    def reset(self,board):
        self.board = board
        self.clear_score = 0

def Q_value(s,a,env):
    gamma = 1
    env.reset(raw_obs(s))
    next_board,r = env.next_state(a)
    next_board = np.log2(np.where(next_board>0, next_board, 1))
    return r+ gamma*getValue(next_board)

def raw_obs(obs):
    raw_obs = np.power(2,obs).reshape(4,4)
    mask = (raw_obs==1)
    raw_obs[mask] = 0
    return raw_obs

class MCTSAgent:
    def __init__(self, env, gamma=0.99, c=100, iter_time=1, ntuple=False):
        '''
        Q and N are saved by list in which
        Q[index] is a list where [0, 1, 2, 3] stands for [Q(s, 0), Q(s, 1), Q(s, 2), Q(s, 3)]
        And index are saved by dict I
        N is saved similarly
        I: state -> index where I[str(state)] is the index saved in Q and N
        '''
        self.env_backup = copy.deepcopy(env)
        self.Q = []
        self.N = []
        self.I = {}
        self.gamma = gamma
        self.c = c
        self.iter_time = iter_time
        self.once_env = Once_env()
        self.eps = 0.1
        self.ntuple = ntuple

    def reset_env(self):
        '''
        reset the env by applying the backup
        '''
        self.env = copy.deepcopy(self.env_backup)

    def rollout(self, s, d):
        '''
        random rollout
        '''
        if d == 0:
            if self.ntuple:
                return getValue(s.reshape(4,4))
            else:
                return 0
        if self.ntuple:
            # epsilong greedy expansion
            if str(s) not in self.I:
                Q_values = [Q_value(s,a,self.once_env) for a in range(4)]
            else:
                index = self.I[str(s)]
                Q_values = self.Q[index]

            if np.random.rand() > self.eps:
                a = np.argmax(Q_values)
            else:
                a = random.randint(0, 3)
        else:
            a = random.randint(0, 3)

        obs, reward, done, info = self.env.step(a)
        if done:
            return reward
        return reward + self.gamma * self.rollout(obs, d - 1)

    def select_action(self, s, d):
        # record the start time
        t0 = time.time()

        self.reset_env()
        '''
        # test if no search
        Qs = [Q_value(s,a,self.once_env) for a in range(4)]
        return np.argmax(Qs)
        '''
        while True:
            self.simulate(s, d)
            # simulate only for 1 second
            if time.time() - t0 >= self.iter_time:
                break

        # search the list for action
        index = self.I[str(s)]
        Qs = [self.Q[index][i] for i in range(4)]

        action = np.argmax(Qs)
        return action

    def valid_action(self, s, a):
        s = raw_obs(s)
        self.once_env.reset(s)
        next_board,r = self.once_env.next_state(a)
        return np.sum(np.abs(next_board - s)) != 0
        # s = np.array(s)
        # s = s.reshape((4, 4))

        # # up or down
        # if a == 0 or a == 1:
        #     return ((s != 0) * (s - np.vstack((np.zeros((1, 4), dtype='int'), s[: 3, :])) == 0)).sum().sum() > 0
        # if a == 2 or a == 3:
        #     return ((s != 0) * (s - np.hstack((np.zeros((4, 1), dtype='int'), s[:, : 3])) == 0)).sum().sum() > 0

    def simulate(self, s, d):
        if d == 0:
            if self.ntuple:
                return getValue(s.reshape(4,4))
            else:
                return 0
        # use str to save the historical node
        ss = str(s)

        # unseen state
        if ss not in self.I:
            self.I[ss] = len(self.N)
            if self.ntuple:
                Q_values = [Q_value(s,a,self.once_env) for a in range(4)]
            else:
                Q_values = [0,0,0,0]
            self.Q.append(Q_values)
            self.N.append([0, 0, 0, 0])
            q = self.rollout(s, d)
            self.reset_env()
            return q
        
        # select a UCB best action
        index = self.I[ss]
        Ns = np.array(self.N[index])
        Qs = np.array(self.Q[index])
        cost = self.c * np.sqrt(np.log(sum(Ns) + 1) / (Ns + 1e-9))
        if self.ntuple:
            punish = -1e9
        else:
            punish = -1e5
        for i in range(4):
            Qs[i] += 0 if self.valid_action(s, i) else punish
        a = np.argmax(Qs + cost)
        # do the action
        obs, reward, done, info = self.env.step(a)
        if done:
            self.reset_env()
            return reward

        # update
        q = reward + self.gamma * self.simulate(obs, d - 1)
        self.N[index][a] += 1
        self.Q[index][a] += (q - self.Q[index][a]) / self.N[index][a]

        # reset the env for next time
        self.reset_env()
        return q