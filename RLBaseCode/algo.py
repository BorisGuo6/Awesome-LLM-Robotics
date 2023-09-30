import time
import copy
import torch
import numpy as np
import torch.nn as nn
import time
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import soft_update,hard_update
from copy import deepcopy

class Algo():
    def __init__(self,state_dim,action_dim,buffer,args,logger,device):
        self.gamma=args['gamma']
        self.learning_rate=args['learning_rate']
        self.epslion=args['epslion']
        self.update_interval=args['update_interval']
        self.seed=args['seed']
        self.logger=logger
        self.device=device

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.num_update=0
        self.buffer=buffer
        self.action_dim=action_dim
        self.critic=Critic(state_dim,action_dim).to(self.device)
        self.critic_target=deepcopy(self.critic).to(self.device)
        self.optimizer = optim.Adam(self.critic.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()

    def train(self):
        batch=self.buffer.sample_batch()
        batch_obs,batch_act,batch_reward,batch_obs_,batch_done=batch['obs'],batch['act'],batch['rew'],batch['obs2'],batch['done']

        q_eval = self.critic(batch_obs).gather(1, batch_act.type(torch.int64))
        q_next = self.critic_target(batch_obs_).detach()
        q_target = batch_reward + (1 - batch_done)*self.gamma * q_next.max(1)[0].view(-1, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.num_update%self.update_interval==0:
            hard_update(self.critic_target, self.critic)

        self.num_update+=1

    @torch.no_grad()
    def choose_action(self,state):
        if np.random.rand(1) >= self.epslion: # epslion greedy           
            return np.random.choice(range(self.action_dim), 1).item()
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            return torch.argmax(self.critic(state)).item()

    def save(self, filename, directory):
        # torch.save(self.actor.state_dict(), '%s/%s_actor.pth' %
        #            (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' %
                   (directory, filename))

    def load(self, filename, directory):
        # self.actor.load_state_dict(torch.load(
        #     '%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load(
            '%s/%s_critic.pth' % (directory, filename)))
        

class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        #put your network here
        self.fc1 = nn.Linear(state_dim, 50)
        self.fc2 = nn.Linear(50, action_dim)

    def forward(self,sate):
        #put your code here
        x = F.relu(self.fc1(sate))
        action_value = self.fc2(x)
        return action_value


class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,args):
        super().__init__()
        # put your network here 
        pass

    def forward(state):
        # put your code here
        pass

    @torch.no_grad()
    def act(self,state):
        # put your code here
        pass

