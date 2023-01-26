import jax.numpy as jnp
from jax import vmap, pmap, random, local_devices
from numpy.polynomial.legendre import leggauss

import wandb
import ml_collections
from tqdm.auto import trange

from archs import MLP, periodic_encoding
from models import VAE
from utils import DataGenerator, save_checkpoint, restore_checkpoint

def legendre_quadrature_1d(n_quad, bounds=(-1.0,1.0)):
    lb, ub = bounds
    # GLL nodes and weights in [-1,1]        
    x, w = leggauss(n_quad)
    # Rescale nodes to [lb,ub]
    x = 0.5*(ub - lb)*(x + 1.0) + lb
    x = jnp.array(x[:,None])
    # Determinant of Jacobian of mapping [lb,ub]-->[-1,1]
    jac_det = 0.5*(ub-lb)
    w = jnp.array(w*jac_det)
    return x, w

def exact_eigenpairs(x, n, alpha=2.0, tau=0.1):
    idx = jnp.arange(n)+1
    evals = jnp.power((2.0 * jnp.pi * idx)**2 + tau**2, -alpha)
    efuns = jnp.sqrt(2.0) * jnp.sin(2.0 * jnp.pi * idx * x)
    return evals, efuns

def sample_u(rng_key, x, n):
    evals, efuns = exact_eigenpairs(x, n)
    xi = random.normal(rng_key, (n,))
    u = jnp.einsum('ij,j->i', efuns, xi*jnp.sqrt(evals))
    w = 1.0/jnp.linalg.norm(u, 2)**2
    return u, x, u[:,None], jnp.tile(w, (x.shape[0],1))


N = 2048
m = 128
neig = 32
bounds = (0.0, 1.0)

x, _ = legendre_quadrature_1d(m, bounds)

evals, efuns = exact_eigenpairs(x, neig)

# Generate training samples
key = random.PRNGKey(0)
keys = random.split(key, N)
gen_fn = lambda key: sample_u(key, x, neig)
u_train, y_train, s_train, w_train = vmap(gen_fn)(keys)
print('Training data')
print('u: {}'.format(u_train.shape))
print('y: {}'.format(y_train.shape))
print('s: {}'.format(s_train.shape))
print('w: {}'.format(w_train.shape))

# Generate testing samples
key = random.PRNGKey(1)
keys = random.split(key, N)
gen_fn = lambda key: sample_u(key, x, neig)
u_test, y_test, s_test, w_test = vmap(gen_fn)(keys)
print('Testing data')
print('u: {}'.format(u_test.shape))
print('y: {}'.format(y_test.shape))
print('s: {}'.format(s_test.shape))
print('w: {}'.format(w_test.shape))

def compute_covariance_operator(efuns, evals):
    return evals*1.0/efuns.shape[0]*jnp.tensordot(efuns, efuns, axes=0)

def relative_covariance_norm(config, params, x, evals, efuns):
    # Evaluate the trunk functions
    trunk = MLP(config.decoder_arch.num_layers,
                config.decoder_arch.hidden_dim,
                config.decoder_arch.output_dim)
    params = {'params': params['params']['decoder']['MLP_0']}
    inputs = periodic_encoding(x, 1.0)
    pred_fn = lambda params: trunk.apply(params, inputs)
    tau = pmap(pred_fn)(params)[0,...]
    # Compute covariances
    C = vmap(compute_covariance_operator, in_axes=(1,0))(efuns, evals)
    C_hat = vmap(compute_covariance_operator, in_axes=(1,0))(tau, jnp.ones(config.eps_dim))
    # Compute metric
    C = jnp.sum(C[:config.eps_dim,...], axis=0)
    C_hat = jnp.sum(C_hat,axis=0)
    diff = C - C_hat
    diff_norm = jnp.linalg.norm(diff,ord='fro') / jnp.linalg.norm(C, ord='fro')
    return diff_norm

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

    if config.logging.cov_norm and config.decoder_arch.name == 'LinearDecoder':
        cov_norm = relative_covariance_norm(config, model.state.params, x, evals, efuns)
        log_dict['cov_norm'] = cov_norm
        wandb.log(log_dict, step)

    wandb.finish()

    return None


