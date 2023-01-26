import jax.numpy as jnp
from flax import linen as nn
from jax import vmap

from typing import Any, Callable, Sequence, Optional, Union, Dict

from layers import Dense, Conv, FourierEnc, MultiresEnc, get_voxel_vertices

Dense = nn.Dense
Conv = nn.Conv

identity = lambda x: x

def periodic_encoding(x, L=1.0):
    x = jnp.hstack([jnp.cos(2.0*jnp.pi*x/L), jnp.sin(2.0*jnp.pi*x/L)])
    return x 

class MLP(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(self.hidden_dim)(x)
            x = self.activation(x)
        x = Dense(self.output_dim)(x)
        return x

class SIREN(nn.Module):
    num_layers: int=2
    hidden_dim: int=32
    output_dim: int=2

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = Dense(self.hidden_dim)(x)
            x = jnp.sin(x)
        x = Dense(self.output_dim)(x)
        return x
        
class GaussianMLP(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(self.hidden_dim)(x)
            x = self.activation(x)
        mu = Dense(self.output_dim)(x)
        logsigma = Dense(self.output_dim)(x)
        return mu, logsigma

class GaussianCNN(nn.Module):
    out_channels: Sequence[int]
    out_dim: int=8
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.out_channels)):
            x = Conv(self.out_channels[i],  kernel_size=(2,2), strides=(2,2), padding="SAME")(x)
            x = self.activation(x)
        x = x.flatten()
        mu = Dense(self.out_dim)(x)
        logsigma = Dense(self.out_dim)(x)
        return mu, logsigma

class ConvEncoder(nn.Module):
    latent_dim: int
    out_channels: Sequence[int]
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x, eps):
        mu, logsigma = GaussianCNN(self.out_channels, 
                                   self.latent_dim,
                                   self.activation)(x)
        beta = mu + eps*jnp.sqrt(jnp.exp(logsigma))
        kl_loss = 0.5*jnp.sum(jnp.exp(logsigma) + mu**2 - 1.0 - logsigma, axis=-1)
        return beta, kl_loss

class MlpEncoder(nn.Module):
    latent_dim: int=8
    num_layers: int=2
    hidden_dim: int=64
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x, eps):
        mu, logsigma = GaussianMLP(self.num_layers, 
                                   self.hidden_dim,
                                   self.latent_dim,
                                   self.activation)(x)
        beta = mu + eps*jnp.sqrt(jnp.exp(logsigma))
        kl_loss = 0.5*jnp.sum(jnp.exp(logsigma) + mu**2 - 1.0 - logsigma, axis=-1)
        return beta, kl_loss

class LinearDecoder(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    pos_enc: Union[None, Dict] = None
    activation: Callable=nn.gelu
    output_activation: Callable=identity

    @nn.compact
    def __call__(self, beta, y):
        if self.pos_enc['type'] == 'periodic':
            y = periodic_encoding(y, self.pos_enc['L'])
        elif self.pos_enc['type'] == 'fourier':
            y = FourierEnc(self.pos_enc['freq'], beta.shape[-1])(y)
        y = MLP(self.num_layers,
                self.hidden_dim,
                self.output_dim,
                self.activation)(y)
        outputs = jnp.sum(beta*y, axis=-1, keepdims=True)
        return self.output_activation(outputs)

class ConcatDecoder(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    pos_enc: Union[None, Dict] = None
    activation: Callable=nn.gelu
    output_activation: Callable=identity

    @nn.compact
    def __call__(self, beta, y):
        if self.pos_enc['type'] == 'periodic':
            y = jnp.tile(y, (beta.shape[-1]//2,))
            y = periodic_encoding(y, self.pos_enc['L'])
        elif self.pos_enc['type'] == 'fourier':
            y = FourierEnc(self.pos_enc['freq'], beta.shape[-1])(y)
        elif self.pos_enc['type'] == 'multires':
            y = MultiresEnc(self.pos_enc['num_levels'],
                            self.pos_enc['min_res'],
                            self.pos_enc['max_res'],
                            self.pos_enc['hash_size'],
                            self.pos_enc['num_features'])(y)
        elif self.pos_enc['type'] == 'none':
            y = jnp.tile(y, (beta.shape[-1]//y.shape[0],))
        outputs = MLP(self.num_layers,
                      self.hidden_dim,
                      self.output_dim,
                      self.activation)(jnp.concatenate([beta, y], axis=-1))
        return self.output_activation(outputs)

class SplitDecoder(nn.Module):
    num_layers: int=8
    hidden_dim: int=256
    output_dim: int=2
    activation: Callable=nn.gelu
    output_activation: Callable=identity

    @nn.compact
    def __call__(self, beta, y):
        beta = jnp.split(beta, self.num_layers)
        if self.pos_enc['type'] == 'fourier':
            y = FourierEnc(self.pos_enc['freq'], beta[0].shape[-1])(y)
        elif self.pos_enc['type'] == 'multires':
            y = MultiresEnc(self.pos_enc['num_levels'],
                            self.pos_enc['min_res'],
                            self.pos_enc['max_res'],
                            self.pos_enc['hash_size'],
                            self.pos_enc['num_features'])(y)
        elif self.pos_enc['type'] == 'none':
            y = jnp.tile(y, (beta[0].shape[-1]//y.shape[0],))
        for i in range(self.num_layers):
            y = Dense(self.hidden_dim)(jnp.concatenate([y, beta[i]]))
            y = self.activation(y)
        outputs = Dense(self.output_dim)(y)
        return self.output_activation(outputs)

class MrhDecoder(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    pos_enc: Union[None, Dict] = None
    activation: Callable=nn.gelu
    output_activation: Callable=identity
        
    @nn.compact
    def __call__(self, beta, y):
        # Determine grid resolutions
        b = jnp.exp((jnp.log(self.pos_enc['max_res'])-jnp.log(self.pos_enc['min_res']))/(self.pos_enc['num_levels']-1))
        N = jnp.int32(jnp.floor(self.pos_enc['min_res']*b**jnp.arange(1,self.pos_enc['num_levels']+1)))        
        # Get nearby vertices
        voxel_indices, \
        voxel_min_vertex, \
        voxel_max_vertex = vmap(get_voxel_vertices, in_axes=(None,0))(y, N)
        # Inputs to MLP
        y = vmap(lambda a, b: a*b, in_axes=(None,0))(y, N)
        inputs = jnp.concatenate([voxel_min_vertex, 
                                  voxel_max_vertex,
                                  jnp.tile(beta, (self.pos_enc['num_levels'],1)),
                                  y], axis=-1)     
        # Compute features at y
        f_fn = nn.vmap(SIREN,
                       in_axes=0, out_axes=0,
                       variable_axes={'params': 0},
                       split_rngs={'params': True})
        features = f_fn(num_layers=2,
                        hidden_dim=64,
                        output_dim=self.pos_enc['num_features'])(inputs)
        # Decode 
        features = features.flatten()
        outputs = MLP(self.num_layers,
                      self.hidden_dim,
                      self.output_dim,
                      self.activation)(features)
        return self.output_activation(outputs)



