from absl import logging
from absl import flags
from absl import app
from ml_collections import config_flags
import jax
import train
import tensorflow as tf
import os


FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('datadir', None, 'Directory to the tfRecord file.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.set_visible_devices([], 'GPU')
    
    
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())
    
    train.train_and_evalutation(FLAGS.config, FLAGS.workdir, FLAGS.datadir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir', 'datadir'])
    app.run(main)
