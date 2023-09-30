import torch
import gym
from algo import Algo
from buffer import ReplayBuffer
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt
from utils.logger import Logger,setup_logger
from utils.wrappers import make_atari
from utils.mario_env import create_train_env

# set param
args={
'algo_name':'DQN',
'env_name':'CartPole-v0',
'seed':0,
'total_step':10000,
'buffer_size':1000,
'learning_rate' : 1e-3,
'update_interval':100,
'batch_size' : 64,
'gamma' : 0.99,
'epslion':0.9
}

render=True
plot=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''init'''

# env=make_atari(args['env_name'])  # Only for atari games
# env, STATE_DIM, ACTION_DIM = create_train_env('1','1', "complex") # for super-mario

env = gym.make(args['env_name'])
env=env.unwrapped      # Remove the maximum step limit
env.seed(args['seed'])
ACTION_DIM = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
logger=Logger()
time_strip=str(datetime.datetime.now().strftime("%m-%d_%H_%M"))
log_dir='log/'+args['env_name']+'-'+args['algo_name']+'-'+time_strip
setup_logger(logger, variant=args, log_dir=log_dir)

buffer=ReplayBuffer(STATE_DIM,1,args,device)
policy=Algo(STATE_DIM,ACTION_DIM,buffer,args,logger,device)

# policy.load('model',log_dir)

print('env: ',args['env_name'])
print('state_dim: ',STATE_DIM)
print('action_dim: ',ACTION_DIM)

'''simulate'''
ep_reward = 0
done = False
state = env.reset()
episode=0
episode_list=[]

for step in range(args['total_step']):
    if render:
        env.render()
    action=policy.choose_action(state)
    
    # print(env.step(action))
    state_, reward, done, _, info = env.step(action)

    '''modify reward only for cart-pole'''
    fake_reward=0
    if done:
        fake_reward=-1

    buffer.store(state, action, fake_reward, state_, done)
    if buffer.size==buffer.max_size:
        policy.train()

    state = state_
    ep_reward += reward

    if done:
        logger.record_tabular('Episode reward', ep_reward)
        episode_list.append(ep_reward)
        state=env.reset()
        episode+=1
        ep_reward=0
        logger.dump_tabular()
        if plot:
            plt.plot(episode_list)
            plt.pause(0.0001)

policy.save('model',log_dir)

print('======= Press Enter to Finish ========')
input()
