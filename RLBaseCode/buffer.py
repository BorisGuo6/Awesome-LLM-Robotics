import random
import numpy as np
import torch

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer 
    """

    def __init__(self, obs_dim, act_dim, args,device):
        self.ptr=0
        self.size=0
        self.device=device
        self.batch_size=args['batch_size']
        self.max_size = args['buffer_size']
        np.random.seed(args['seed'])

        self.obs_buf = np.zeros(combined_shape(self.max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(self.max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(self.max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=np.expand_dims(self.rew_buf[idxs],axis=1),
                     obs2=self.obs2_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)