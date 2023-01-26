import jax.numpy as jnp
from jax import vmap, pmap, random, local_devices

import wandb
import ml_collections
from tqdm.auto import trange
import numpy as np

from models import VAE
from utils import DataGenerator, save_checkpoint, restore_checkpoint

volcano_data = np.load('../datasets/volcano.npy')
volcano_data = jnp.array(volcano_data)
N, m, _, num_channels = volcano_data.shape

def get_example(u, X, n):
    s = u.reshape(-1,num_channels)
    # w = 1.0/jnp.linalg.norm(s, 2, axis=0)**2
    return u, X, s, jnp.tile(1.0, (X.shape[0],1))

x = jnp.linspace(0,1,m)
y = jnp.linspace(0,1,m)
grid = jnp.meshgrid(x,y)
X = jnp.array(grid).T.reshape(-1,2)

gen_fn = lambda u: get_example(u, X, m)

# Generate training samples
gen_fn = lambda u: get_example(u, X, m)
u_train, y_train, s_train, w_train = vmap(gen_fn)(volcano_data)
print('Training data')
print('u: {}'.format(u_train.shape))
print('y: {}'.format(y_train.shape))
print('s: {}'.format(s_train.shape))
print('w: {}'.format(w_train.shape))


def compute_mmd(config, model, num_samples=256):
    # Generate samples from model
    key = random.PRNGKey(123)
    eps_test = random.normal(key, (num_samples, config.eps_dim))
    sample_fn = lambda params: model.state.decode_fn(params, eps_test, y_train[:num_samples,...])
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

    dim = samples.shape[1]
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
               config=dict(config))

    print(local_devices())

    model = VAE(config)
    if config.training.restart_checkpoint is not None:
        model = restore_checkpoint(model, config.training.restart_checkpoint)

    dataset = DataGenerator(u_train, y_train, s_train, w_train,
                            config.eps_dim, 
                            config.training.num_mc_samples, 
                            config.training.batch_size)
    data = iter(dataset)
    batch = next(data)
    inputs, targets, weights = batch
    u, y, eps = inputs
    s = targets
    w = weights
    print('Batch dimensions')
    print('u: {}'.format(u.shape))
    print('y: {}'.format(y.shape))
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
    

