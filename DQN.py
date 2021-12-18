import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym2048
import gym
import matplotlib.pyplot as plt
import copy
from tensorboardX import SummaryWriter
import os

# hyper-parameters
BATCH_SIZE = 1024 # 128 originally
LR = 0.001 # 0.001 originally
GAMMA = 1 # 0.9 originally
EPISILO = 0.5 # 0.9 originally
MEMORY_CAPACITY = 200000 # 2e+5 originally
Q_NETWORK_ITERATION = 100
EPISODES = 150000 # 4e+4 originally
# LOAD_PATH = 'CNNNet-gamma1_39999.pkl' # None originally
LOAD_PATH = None

use_cuda = torch.cuda.is_available()
# use_cuda = False

torch.cuda.set_device(2)

# env = gym.make("CartPole-v0")
# setting_name = ''
setting_name = 'CNNNet-bsz1024-gamma1-epsd15w-epsl0.5to1after3w'
env = gym.make("Env2048onehot-v0")
env = env.unwrapped
writer = SummaryWriter('DQN_log/onehot/')
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

print(setting_name, os.getpid())
print('bsz', BATCH_SIZE)
print('lr', LR)
print('gamma', GAMMA)
print('eps', EPISILO)
print('mem cap', MEMORY_CAPACITY)
print('q net iter', Q_NETWORK_ITERATION)
print('LOAD_PATH', LOAD_PATH)

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

class CNNNet_NoFirstRelu(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(CNNNet_NoFirstRelu, self).__init__()
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
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x).permute(0,2,3,1).view(-1, 16*64)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob.softmax(dim=-1)

class RowColNet(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(RowColNet, self).__init__()
        # self.fc1 = nn.Linear(18, 64)
        # self.fc1.weight.data.normal_(0,0.1)
        self.rowemb = nn.Linear(18*4, 64)
        self.rowemb.weight.data.normal_(0,0.1)
        self.colemb = nn.Linear(18*4, 64)
        self.colemb.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(8*64, 64)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(64,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = x.view(-1, 4, 4, 18)
        x1 = self.rowemb(x.view(-1, 4, 4*18))
        x2 = self.colemb(x.permute(0,2,1,3).reshape(-1, 4, 4*18))
        x = torch.cat([x1, x2], dim=-1).reshape(-1, 8*64)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob.softmax(dim=-1)

class RCCNet(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(RCCNet, self).__init__()
        # self.fc1 = nn.Linear(18, 64)
        # self.fc1.weight.data.normal_(0,0.1)
        self.rowemb = nn.Linear(18*4, 64)
        self.rowemb.weight.data.normal_(0,0.1)
        self.colemb = nn.Linear(18*4, 64)
        self.colemb.weight.data.normal_(0,0.1)
        self.fc1 = nn.Linear(18, 64)
        self.fc1.weight.data.normal_(0,0.1)
        self.conv = nn.Conv2d(64,64,3,padding=(1,1))
        self.fc2 = nn.Linear(24*64, 128)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = x.view(-1, 4, 4, 18)
        # == RowCol
        x1 = self.rowemb(x.view(-1, 4, 4*18))
        x2 = self.colemb(x.permute(0,2,1,3).reshape(-1, 4, 4*18))
        # === CNN
        x = self.fc1(x)
        x = F.relu(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x).permute(0,2,3,1).view(-1, 16*64)

        x1 = torch.cat([x1, x2], dim=-1).reshape(-1, 8*64)
        x = torch.cat([x1, x], dim=-1)
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
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        if use_cuda:
            self.memory = torch.zeros(MEMORY_CAPACITY, NUM_STATES * 2 + 2).cuda()
        print('memory built')
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
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
        if greedy or np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            if use_cuda:
                action_value = action_value.cpu()
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        if use_cuda:
            transition = torch.FloatTensor(transition).cuda()
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
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

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
                print('greedy', env.get_score(), env.get_board())
                return env.get_score(), np.log2(env.get_board().max())
            state = next_state

# def reward_func(env, x, x_dot, theta, theta_dot):
#     r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
#     r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#     reward = r1 + r2
#     return reward

def main():
    dqn = DQN()
    if LOAD_PATH != None:
        dqn.load(LOAD_PATH)
    episodes = EPISODES
    print("Collecting Experience....")
    reward_list = []
    # plt.ion()
    # fig, ax = plt.subplots()
    for i in range(episodes):
        if i == 30000:
            global EPISILO
            EPISILO = 1
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

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)), env.get_board().flatten())
                    writer.add_scalar(setting_name+'Reward/Episodes reward', ep_reward, global_step=i)
                    # writer.add_scalar(setting_name+'Reward/Score', env.get_score(), global_step=i)
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
