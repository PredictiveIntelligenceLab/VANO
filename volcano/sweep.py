# import os
# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
# os.environ['TF_CUDNN_DETERMINISTIC'] ='1'  # For better reproducible!  ~35% slower !
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] ='0.8'

# import tensorflow as tf
# tf.config.experimental.set_visible_devices([], "GPU")
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] ='platform'


from functools import partial

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import wandb

import train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir


    sweep_config = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'max_mmd'
            },
        }

    parameters_dict = { 'beta': {'values': [1e-5, 1e-4, 1e-3]},
                        'eps_dim': {'values': [128, 256, 512]},
                        'hidden_dim': {'values': [256, 512]}
                      }
    sweep_config['parameters'] = parameters_dict


    def train_sweep():
        config = FLAGS.config

        wandb.init(project=config.wandb.project,
                   name=config.wandb.name)

        sweep_config = wandb.config

        config.beta = sweep_config.beta
        config.eps_dim = sweep_config.eps_dim
        config.encoder_arch.latent_dim = sweep_config.eps_dim
        config.decoder_arch.hidden_dim = sweep_config.hidden_dim
        config.wandb.name = 'h%d_eps%d_beta%.1e' % (config.decoder_arch.hidden_dim, config.eps_dim, config.beta)

        train.train_and_evaluate(config, workdir)

        wandb.finish()

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb.project,
        )

    wandb.agent(sweep_id, function=train_sweep)

if __name__ == "__main__":
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
