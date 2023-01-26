import ml_collections
import jax.numpy as jnp
from flax import linen as nn


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.job_type = 'train'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "grf1d_vae"
    wandb.name = 'latent_dim_sweep'
    wandb.tag = None

    # Simulation settings
    config.input_dim = (128,)
    config.query_dim = 1
    config.eps_dim = 64
    config.beta = 5e-6
    config.seed = 2

    # Encoder Architecture
    config.encoder_arch = encoder_arch = ml_collections.ConfigDict()
    encoder_arch.name = 'MlpEncoder'
    encoder_arch.num_layers = 3
    encoder_arch.hidden_dim = 128
    encoder_arch.latent_dim = config.eps_dim
    encoder_arch.activation = nn.gelu

    # Decoder Architecture
    config.decoder_arch = decoder_arch = ml_collections.ConfigDict()
    decoder_arch.name = 'LinearDecoder'
    decoder_arch.num_layers = 3
    decoder_arch.hidden_dim = 128
    decoder_arch.output_dim = config.eps_dim
    decoder_arch.pos_enc = ml_collections.ConfigDict({'type': 'periodic', 'L': 1.0})
    decoder_arch.activation: nn.gelu
    decoder_arch.output_activation: lambda x: x

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.num_mc_samples = 16
    training.max_steps = 40000
    training.save_every_steps = None
    training.restart_checkpoint = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 1000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_losses = True
    logging.cov_norm = True

    return config

