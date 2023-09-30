import gym
from utils.wrappers import make_atari
from utils.mario_env import create_train_env

#设置环境
env=gym.make('MountainCar-v0') #CartPole-v0  MountainCar-v0  Pendulum-v0  
# env=make_atari('PongNoFrameskip-v4') #Atari
# env, STATE_DIM, ACTION_DIM=create_train_env(world=2, stage=1, action_type="complex") #super-mario

#结束标志位
done=False
#刷新环境
env.reset()
ep_r=0
while not done:
		#渲染画面（较为耗时）
    env.render()
    #随机采样动作
    action=env.action_space.sample()
    #执行动作，并返回数据
    # print(env.step(action))
    obs_,reward,done,info=env.step(action)
    ep_r+=reward

print(ep_r)

