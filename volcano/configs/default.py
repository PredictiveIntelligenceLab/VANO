import ml_collections
import jax.numpy as jnp
from flax import linen as nn


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.job_type = 'train'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = 'volcano_vae'
    wandb.name = None
    wandb.tag = None

    # Simulation settings
    config.input_dim = (1,128,128,2)
    config.query_dim = 2
    config.eps_dim = 256
    config.beta = 1e-4
    config.seed = 0

    # Encoder Architecture
    config.encoder_arch = encoder_arch = ml_collections.ConfigDict()
    encoder_arch.name = 'ConvEncoder'
    encoder_arch.latent_dim = config.eps_dim
    encoder_arch.out_channels = (8, 16, 32, 64, 128, 256)
    encoder_arch.activation = nn.gelu

    # Decoder Architecture
    config.decoder_arch = decoder_arch = ml_collections.ConfigDict()
    decoder_arch.name = 'SplitDecoder'
    decoder_arch.num_layers = 8
    decoder_arch.hidden_dim = 512
    decoder_arch.output_dim = config.input_dim[-1]
    decoder_arch.pos_enc = ml_collections.ConfigDict({'type': 'multires', 
                                                      'num_levels': 16,
                                                      'min_res': 16,
                                                      'max_res': 1024,
                                                      'hash_size': 2**16,
                                                      'num_features': 8})
    decoder_arch.activation: nn.gelu
    decoder_arch.output_activation: lambda x: x

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 16
    training.num_mc_samples = 4
    training.max_steps = 25000
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
    logging.log_mmds = True

    return config