import jax.numpy as jnp
from jax import vmap, pmap, random, local_devices

import wandb
import ml_collections
from tqdm.auto import trange

import numpy as np
from jax import image

from models import DiscreteVAE
from utils import DataGeneratorDiscrete, save_checkpoint, restore_checkpoint

N = 4096
m = 256
num_channels = 1

resize = lambda x: image.resize(x, shape=(m,m,num_channels), method='bilinear')

data_buf = np.load('../cahn_hilliard_patterns.npy')
num_examples = data_buf.shape[0]

key = random.PRNGKey(0)
idx = random.choice(key, num_examples, (N,), replace=False)
train_X = jnp.array(data_buf[idx,...])
train_X = vmap(resize)(train_X)

key = random.PRNGKey(1)
idx = random.choice(key, num_examples, (N,), replace=False)
test_X = jnp.array(data_buf[idx,...])
test_X = vmap(resize)(test_X)

def get_example(u, n):
    u = jnp.flipud(u)
    return u, u, jnp.ones_like(u)

N, m, _, num_channels = train_X.shape

gen_fn = lambda u: get_example(u, m)
u_train, s_train, w_train = vmap(gen_fn)(train_X)
print('Training data')
print('u: {}'.format(u_train.shape))
print('s: {}'.format(s_train.shape))
print('w: {}'.format(w_train.shape))

# Generate testing samples
gen_fn = lambda u: get_example(u, m)
u_test, s_test, w_test = vmap(gen_fn)(test_X)
print('Testing data')
print('u: {}'.format(u_test.shape))
print('s: {}'.format(s_test.shape))
print('w: {}'.format(w_test.shape))

def compute_mmd(config, model, num_samples=256):
    # Generate samples from model
    key = random.PRNGKey(123)
    eps_test = random.normal(key, (num_samples, config.eps_dim))
    sample_fn = lambda params: model.state.decode_fn(params, eps_test)
    samples = pmap(sample_fn)(model.state.params)[0,...]

    def MMD(X, Y, k):
        n = X.shape[0]
        m = Y.shape[0]
        kXX = k(X,X)
        kYY = k(Y,Y)
        kXY = k(X,Y)
        Xterm = 1./(n*(n-1))*jnp.sum(kXX)
        Yterm = 1./(m*(m-1))*jnp.sum(kYY)
        XYterm = 1./(n*m)*jnp.sum(kXY)
        return Xterm + Yterm - 2*XYterm

    dim = samples.shape[1]**2
    kernelfunc = lambda x, y, sigmasq: jnp.exp(- 1./(2*sigmasq*dim) * jnp.dot(x-y, x-y))

    # Define sweep range and containers
    sigmasqs = jnp.linspace(1e-2,10,100)
    def get_mmds(X, Y, sigmasq):
        kernel = lambda x, y: kernelfunc(x, y, sigmasq)
        k = vmap(vmap(kernel, in_axes=(None, 0)), in_axes=(0, None))
        mmd = MMD(X, Y, k)
        return mmd

    mmds = vmap(get_mmds, in_axes=(None,None,0))(samples.reshape(num_samples,-1), 
                                                 s_train[:num_samples,...].reshape(num_samples,-1),
                                                 sigmasqs)
    return mmds    
    

def eval_step(config, model, batch):

    params = model.state.params
    log_dict = {}

    if config.logging.log_losses:
        kl_loss, recon_loss = model.eval_losses(params, batch)
        kl_loss = kl_loss.mean()
        recon_loss = recon_loss.mean()
        log_dict['kl_loss'] = kl_loss
        log_dict['recon_loss'] = recon_loss

    return log_dict


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):

    wandb_config = config.wandb
    wandb.init(project=wandb_config.project,
               name=wandb_config.name,
               config=dict(config),
               entity='team_nomad')

    print(local_devices())
    
    model = DiscreteVAE(config)
    if config.training.restart_checkpoint is not None:
        model = restore_checkpoint(model, config.training.restart_checkpoint)

    dataset = DataGeneratorDiscrete(u_train, s_train, w_train,
                                    config.eps_dim, 
                                    config.training.num_mc_samples, 
                                    config.training.batch_size)
    data = iter(dataset)
    batch = next(data)
    inputs, targets, weights = batch
    u, eps = inputs
    s = targets
    w = weights
    print('Batch dimensions')
    print('u: {}'.format(u.shape))
    print('eps: {}'.format(eps.shape))
    print('s: {}'.format(s.shape))
    print('w: {}'.format(w.shape))

    pbar = trange(config.training.max_steps)
    for step in pbar:
        batch = next(data)        
        model.state = model.step(model.state, batch)
        # logging
        if step % config.logging.log_every_steps == 0:
            log_dict = eval_step(config, model, batch)
            wandb.log(log_dict, step)
            pbar.set_postfix({'kl_loss': log_dict['kl_loss'], 
                              'recon_loss': log_dict['recon_loss']})

        # Saving
        if config.training.save_every_steps is not None:
            if (step + 1) % config.training.save_every_steps == 0 or (step + 1) == config.training.max_steps:
                save_checkpoint(model.state, workdir)

    # Save last parameter configuration
    if config.training.save_every_steps is None:
        save_checkpoint(model.state, workdir)

    if config.logging.log_mmds:
        mmds = compute_mmd(config, model)
        log_dict['mmds'] = mmds
        log_dict['max_mmd'] = mmds.max()
        wandb.log(log_dict, step)

    wandb.finish()

    return None
    


