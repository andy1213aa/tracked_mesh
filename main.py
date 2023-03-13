from absl import logging
from absl import flags
from absl import app
from ml_collections import config_flags
import train

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True
)
    
def main(argv):
    train()

if __name__ == '__main__':
    flags.mark_flag_as_required(['config'])
    app.run(main)
