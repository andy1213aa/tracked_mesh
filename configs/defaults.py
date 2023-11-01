"""Default Hyperparameter configuration"""

import ml_collections


def get_config():
    """Get the default Hypermarameter configuration"""
    config = ml_collections.ConfigDict()
    config.model = 'Classic_CNN'
    config.dataset = 'multiface'
    
    config.data_size = 15250
    config.image_size = (2048, 1334)
    config.render_size = (512, 334)
    config.texture_size = (1024, 1024)
    
    config.num_epochs = 2000
    config.warmup_epochs = 1
    config.batch_size = 16
    config.steps_per_epoch = config.data_size // config.batch_size
    config.learning_rate = 1e-3
    config.log_every_steps = 10
    config.vertex_num = 7306
    config.num_train_steps = -1
    config.kpt_num = 478
    config.pca = '../training_data/pca_for_jax.pickle'
    config.mean_mesh = '../training_data/mean_mesh_for_jax.pickle'
    return config