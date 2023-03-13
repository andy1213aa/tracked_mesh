"""Default Hyperparameter configuration"""

import ml_collections

def get_config():
    """Get the default Hypermarameter configuration"""
    config = ml_collections.ConfigDict()
    config.model = 'Classic_CNN'
    config.dataset = 'multiface'

    return config