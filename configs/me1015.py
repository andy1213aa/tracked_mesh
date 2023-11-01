"""Default Hyperparameter configuration"""

import ml_collections


def get_config():
    """Get the default Hypermarameter configuration"""
    config = ml_collections.ConfigDict()
    config.model = 'Classic_CNN'
    config.dataset = 'me1015'
    
    config.data_size = 664
    config.image_size = (3024, 4032) # cv2 style
    config.render_size = (756, 1008) # cv2 style
    config.texture_size = (1024, 1024)
    
    config.num_epochs = 10000
    config.warmup_epochs = 1
    config.batch_size = 4
    config.steps_per_epoch = config.data_size // config.batch_size
    config.learning_rate = 1e-3
    config.log_every_steps = 10
    config.vertex_num = 7306
    config.num_train_steps = -1
    config.kpt_num = 478
    config.pca = '../training_data/pca1015.pickle'
    config.mean_mesh = '../training_data/1015_mean.pickle'
    return config