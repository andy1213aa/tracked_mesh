"""Default Hyperparameter configuration"""

import ml_collections

def get_config():
    """Get the default Hypermarameter configuration"""
    config = ml_collections.ConfigDict()
    config.model = 'Classic_CNN'
    config.dataset = 'multiface'
    config.image_size = 28
    config.num_epochs = 100
    config.warmup_epochs = 10
    config.batch_size = 1
    config.learning_rate = 0.01
    config.log_every_steps = 100
    config.num_epochs = 100.0
    return config