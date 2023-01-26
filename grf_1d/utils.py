import jax.numpy as jnp
from jax import random, pmap, tree_map, process_index, device_get, local_device_count
from flax.training import checkpoints
from flax import jax_utils

from torch.utils import data
from functools import partial

def save_checkpoint(state, workdir):
    if process_index() == 0:
        state = device_get(tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3, overwrite=True)

def restore_checkpoint(model, workdir):
    state = checkpoints.restore_checkpoint(workdir, model.state)
    model.state = jax_utils.replicate(state) 
    return model

class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, w, 
                 eps_dim = 16, 
                 num_mc_samples = 32, batch_size=64, 
                 rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        self.w = w
        self.N = u.shape[0]
        self.eps_dim = eps_dim
        self.num_mc_samples = num_mc_samples
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        inputs, targets, weights = self.__data_generation(keys)
        return inputs, targets, weights

    @partial(pmap, static_broadcasted_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        eps = random.normal(key, (self.num_mc_samples, self.batch_size, self.eps_dim))  
        u = self.u[idx,...]
        y = self.y[idx,...]
        s = self.s[idx,...]
        w = self.w[idx,...]
        # Construct batch
        inputs = (u, y, eps)
        targets = s
        weights = w
        return inputs, targets, weights