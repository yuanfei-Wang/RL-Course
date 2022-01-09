import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym2048
import gym
import matplotlib.pyplot as plt
import copy
from tensorboardX import SummaryWriter
from data_structure.P_replay import Prioritised_Replay_Buffer
import os

# hyper-parameters
BATCH_SIZE = 1024
LR = 0.0005
GAMMA = 0.99
EPISILO_MAX = 1.0
EPISILO = 0.0
EPISILO_INC = 1e-6
MEMORY_CAPACITY = 100000
Q_NETWORK_ITERATION = 100
hyperparameters = {
    "buffer_size": MEMORY_CAPACITY,
    "batch_size": BATCH_SIZE,
    "alpha_prioritised_replay": 0.6,
    "beta_prioritised_replay": 0.1,
    "incremental_td_error": 1e-8
}

use_cuda = torch.cuda.is_available()
# use_cuda = False

# env = gym.make("CartPole-v0")
# setting_name = ''
setting_name = 'CNNNet-epsilon1.0'
env = gym.make("Env2048onehot-v0")
env = env.unwrapped
writer = SummaryWriter('DQN_log/epsilon/')
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

print(setting_name, os.getpid())
print('bsz', BATCH_SIZE)
print('lr', LR)
print('gamma', GAMMA)
print('eps', EPISILO_MAX)
print('eps inc', EPISILO_INC)
print('mem cap', MEMORY_CAPACITY)
print('q net iter', Q_NETWORK_ITERATION)

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 128)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(128,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob.softmax(dim=-1)

class EmbNet(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(EmbNet, self).__init__()
        self.fc1 = nn.Linear(18,128)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(16*128,64)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(64,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = x.view(-1, 4, 4, 18)
        x = self.fc1(x)
        x = F.relu(x)
        x = x.view(-1, 16*128)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob.softmax(dim=-1)

class CNNNet(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(CNNNet, self).__init__()
        self.fc1 = nn.Linear(18, 64)
        self.fc1.weight.data.normal_(0,0.1)
        self.conv = nn.Conv2d(64,64,3,padding=(1,1))
        self.fc2 = nn.Linear(16*64, 64)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(64,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = x.view(-1, 4, 4, 18)
        x = self.fc1(x)
        x = F.relu(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x).permute(0,2,3,1).view(-1, 16*64)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob.softmax(dim=-1)

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        # self.eval_net, self.target_net = Net(), Net()
        self.eval_net, self.target_net = CNNNet(), CNNNet()
        self.EPISILO = EPISILO
        self.EPISILO_MAX = EPISILO_MAX
        self.EPISILO_INC = EPISILO_INC
        for n, p in self.eval_net.named_parameters():
            print(n, p.size())
        print('Net built')
        if use_cuda:
            self.eval_net.cuda()
            self.target_net.cuda()
            print('Net cuda')

        print('memory start to built')
        self.learn_step_counter = 0
        self.memory_counter = 0
        #self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.memory = Prioritised_Replay_Buffer(hyperparameters)
        #if use_cuda:
        #    self.memory = torch.zeros(MEMORY_CAPACITY, NUM_STATES * 2 + 2).cuda()
        print('memory built')
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        #self.loss_func = F.mse_loss()
        print('optim built')

    def save(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load(self, path):
        maploc = lambda storage, loc: storage
        if use_cuda:
            maploc = lambda storage, loc: storage.cuda()
        obj = torch.load(path, map_location=maploc)
        self.eval_net.load_state_dict(obj)
        self.target_net.load_state_dict(obj)

    def choose_action(self, state, greedy=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if use_cuda:
            state = state.cuda()
        if greedy or np.random.randn() <= self.EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            if use_cuda:
                action_value = action_value.cpu()
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        if self.EPISILO < self.EPISILO_MAX:
            self.EPISILO += self.EPISILO_INC
        return action

    def store_transition(self, state, action, reward, next_state, done):
        '''
        transition = np.hstack((state, [action, reward], next_state))
        if use_cuda:
            transition = torch.FloatTensor(transition).cuda()
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        '''
        max_td_error_in_experiences = self.memory.give_max_td_error() + 1e-9
        self.memory.add_experience(max_td_error_in_experiences, state, action, reward, next_state, done)

    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        '''
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        if use_cuda:
            batch_state = batch_memory[:, :NUM_STATES]
            batch_action = batch_memory[:, NUM_STATES:NUM_STATES+1].long()
            batch_reward = batch_memory[:, NUM_STATES+1:NUM_STATES+2]
            batch_next_state = batch_memory[:,-NUM_STATES:]
        else:
            batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
            batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
            batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
            batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])
        '''
        sampled_experiences, importance_sampling_weights = self.memory.sample()
        batch_state, batch_action, batch_reward, batch_next_state, dones = sampled_experiences
        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action.long())
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = F.mse_loss(q_eval, q_target)
        loss = loss * importance_sampling_weights
        loss = torch.mean(loss)
        td_errors = q_target.data.cpu().numpy() - q_eval.data.cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_td_errors(td_errors.squeeze(1))

    def greedy_eval(self, env):
        state = env.reset()
        while True:
            env.render()
            action = self.choose_action(state, greedy=True)
            # print(action, end='')
            next_state, reward, done, info = env.step(action)
            # np.sum(np.abs(next_board - self.board)) == 0
            if done or np.sum(np.abs(next_state - state)) == 0:
                # print('\n')
                return env.get_score(), np.log2(env.get_board().max())
            state = next_state

# def reward_func(env, x, x_dot, theta, theta_dot):
#     r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
#     r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#     reward = r1 + r2
#     return reward

def main():
    dqn = DQN()
    episodes = 400000
    print("Collecting Experience....")
    reward_list = []
    # plt.ion()
    # fig, ax = plt.subplots()
    for i in range(episodes):
        if (i+1) % (episodes // 10) == 0:
            dqn.save(setting_name+'_'+str(i)+'.pkl')
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            # x, x_dot, theta, theta_dot = next_state
            # reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state, done)
            ep_reward += reward

            if i >= 30:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)), env.get_board().flatten())
                    writer.add_scalar(setting_name+'Reward/Episodes reward', ep_reward, global_step=i)
                    #writer.add_scalar(setting_name+'Reward/Score', env.get_score(), global_step=i)
                    writer.add_scalar(setting_name+'Reward/Max tile', np.log2(env.get_board().max()), global_step=i)
                    if i % 400 == 0:
                        scores = 0
                        maxtile = 0
                        for _ in range(3):
                            score, board = dqn.greedy_eval(env)
                            scores += score
                            maxtile = max(maxtile, board)
                        writer.add_scalar(setting_name+'Reward/Greedy Score', scores/3, global_step=i)
                        writer.add_scalar(setting_name+'Reward/Greedy Max tile', maxtile, global_step=i)

            if done:
                break
            state = next_state
        r = copy.copy(reward)
        # reward_list.append(r)
        # ax.set_xlim(0,300)
        # #ax.cla()
        # ax.plot(reward_list, 'g-', label='rewards')
        # plt.pause(0.001)
        
        
if __name__ == '__main__':
    main()
