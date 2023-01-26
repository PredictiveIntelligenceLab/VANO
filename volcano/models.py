import jax.numpy as jnp
from jax import random, grad, vmap, pmap
from jax.lax import pmean
from flax import linen as nn
from flax import jax_utils
from flax import struct
from flax.training import train_state
import optax

from functools import partial
from typing import Callable

import archs

class TrainState(train_state.TrainState):
    encode_fn: Callable = struct.field(pytree_node=False)
    decode_fn: Callable = struct.field(pytree_node=False)

def _create_encoder_arch(config):
    if config.name == 'MlpEncoder':
        arch = archs.MlpEncoder(**config)

    elif config.name == 'ConvEncoder':
        arch = archs.ConvEncoder(**config)

    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')

    return arch

def _create_decoder_arch(config):
    if config.name == 'LinearDecoder':
        arch = archs.LinearDecoder(**config)

    elif config.name == 'ConcatDecoder':
        arch = archs.ConcatDecoder(**config)

    elif config.name == 'SplitDecoder':
        arch = archs.SplitDecoder(**config)

    elif config.name == 'MrhDecoder':
        arch = archs.MrhDecoder(**config)

    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')

    return arch


def _create_optimizer(config):
    if config.optimizer == 'Adam':
        lr = optax.exponential_decay(init_value=config.learning_rate,
                                     transition_steps=config.decay_steps,
                                     decay_rate=config.decay_rate)
        optimizer = optax.adam(learning_rate=lr, b1=config.beta1, b2=config.beta2,
                               eps=config.eps)

    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def _create_train_state(config):
    # Build architecture
    encoder = _create_encoder_arch(config.encoder_arch)
    decoder = _create_decoder_arch(config.decoder_arch)
    arch = NeuralOperator(encoder, decoder)
    # Initialize parameters
    u = jnp.ones(config.input_dim)
    y = jnp.ones((1,config.query_dim))
    eps = jnp.ones(config.eps_dim)
    key = random.PRNGKey(config.seed)
    params = arch.init(key, u, y, eps)
    print(arch.tabulate(key, u, y, eps))
    # Vectorized functions across a mini-batch
    apply_fn = vmap(arch.apply, in_axes=(None,0,0,0))
    encode_fn = vmap(lambda params, u, eps: arch.apply(params, u, eps, method=arch._encode), in_axes=(None,0,0))
    decode_fn = vmap(lambda params, beta, y: arch.apply(params, beta, y, method=arch._decode), in_axes=(None,0,0))
    # Optimizaer
    tx = _create_optimizer(config.optim)
    # Create state
    state = TrainState.create(apply_fn=apply_fn,
                                params=params,
                                tx=tx,
                                encode_fn=encode_fn,
                                decode_fn=decode_fn)
    # Replicate state across devices
    state = jax_utils.replicate(state) 
    return state


class NeuralOperator(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, u, y, eps):
        beta, _ = self.encoder(u, eps)
        outputs = vmap(self.decoder, in_axes=(None,0))(beta, y)
        return outputs

    def _encode(self, u, eps):
        beta, kl_loss = self.encoder(u, eps)
        return beta, kl_loss

    def _decode(self, beta, y):
        outputs = vmap(self.decoder, in_axes=(None,0))(beta, y)
        return outputs


# Define the model
class VAE:
    def __init__(self, config): 
        self.config = config
        self.state = _create_train_state(config)
        self.beta = config.beta

    # Computes KL loss across a mini-batch
    def kl_loss(self, params, u, eps):
        _, loss = self.state.encode_fn(params, u, eps)
        return jnp.mean(loss)

    # Computes reconstruction loss across a mini-batch for a single MC sample
    def recon_loss(self, params, u, y, s, w, eps):
        outputs = self.state.apply_fn(params, u, y, eps)
        loss = jnp.mean(0.5*w*(s-outputs)**2)
        return loss

    # Computes total loss across a mini-batch for multiple MC samples
    def loss(self, params, batch):
        inputs, targets, weights = batch
        u, y, eps = inputs
        s = targets
        w = weights
        kl_loss = vmap(self.kl_loss, in_axes=(None,None,0))(params, u, eps)
        recon_loss = vmap(self.recon_loss, in_axes=(None,None,None,None,None,0))(params, u, y, s, w, eps)
        kl_loss = jnp.mean(kl_loss)
        recon_loss = jnp.mean(recon_loss)
        loss = self.beta*kl_loss + recon_loss
        return loss

    @partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(0,))
    def eval_losses(self, params, batch):
        inputs, targets, weights = batch
        u, y, eps = inputs
        s = targets
        w = weights
        kl_loss = vmap(self.kl_loss, in_axes=(None,None,0))(params, u, eps)
        recon_loss = vmap(self.recon_loss, in_axes=(None,None,None,None,None,0))(params, u, y, s, w, eps)
        kl_loss = jnp.mean(kl_loss)
        recon_loss = jnp.mean(recon_loss)
        return kl_loss, recon_loss

    # Define a compiled update step
    @partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(0,))
    def step(self, state, batch):
        grads = grad(self.loss)(state.params, batch)
        grads = pmean(grads, 'num_devices')
        state = state.apply_gradients(grads=grads)
        return state
