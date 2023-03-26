
import jax.numpy as jnp
from flax.training import checkpoints
from flax.training import train_state
from ml_collections import config_flags

import inference
import os
from absl import logging
from absl import flags
from absl import app

FLAGS = flags.FLAGS
# 載入保存的checkpoint模型
flags.DEFINE_string('workdir', None, 'Directory to load model checkpoints.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    inference.inference(FLAGS.config, FLAGS.workdir)
    
    
if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)