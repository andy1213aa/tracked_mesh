import tensorflow as tf
import numpy as np
import ml_collections
from absl import logging
from pathlib import Path
# Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.set_visible_devices([], 'GPU')
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 320

# IMAGE_WIDTH_RESIZE = 240
# IMAGE_HEIGHT_RESIZE = 320

NUM_VERTEX = 7306

# 7306*3 array
vert_mean = np.load('../training_data/6674443_vert_mean.npy')
# 1 float32
vert_var = np.load('../training_data/6674443_vert_var.npy')


def load_obj(pth):

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    obj_pth = Path(pth)

    if not obj_pth.exists:
        logging.info(f'file {obj_pth} does not exist')
        return None, False
    else:
        with open(obj_pth, 'r') as f:
            vertices = []
            for line in f:
                if 'v' in line and 'vn' not in line and 'vt' not in line:

                    v = [
                        float(x) for x in list(
                            filter(lambda x: is_number(x), line.split()))
                    ]
                    vertices.append(v)
                if 'f' in line:
                    break
            return True, vertices



def readTFRECORD(tfrecord_pth: str,
                 config: ml_collections.ConfigDict) -> tf.data:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_set = tf.data.TFRecordDataset(tfrecord_pth)
    data_set = data_set.repeat()

    data_set = data_set.map(parse, num_parallel_calls=AUTOTUNE)
    data_set = data_set.shuffle(config.data_size//2,
                                reshuffle_each_iteration=True)
    data_batch = data_set.batch(config.batch_size, drop_remainder=True)
    data_batch = data_batch.prefetch(buffer_size=AUTOTUNE)
    return data_batch


def parse(example_proto):
    features = tf.io.parse_single_example(example_proto,
                                          features={
                                              'img':
                                              tf.io.FixedLenFeature([],
                                                                    tf.string),
                                              'vtx':
                                              tf.io.FixedLenFeature([],
                                                                    tf.string)
                                          })

    img = features['img']
    vtx = features['vtx']

    img = tf.io.decode_raw(img, np.float32)
    vtx = tf.io.decode_raw(vtx, np.float32)

    # standardize
    # vtx = (vtx - vert_mean) / tf.sqrt(vert_var)
    # neutral face model
    neutral_mesh = tf.convert_to_tensor(np.array(load_obj('../training_data/000220.obj')[1]).flatten(), dtype=tf.float32)
    # neutral_mesh = (neutral_mesh-vert_mean) / tf.sqrt(vert_var)
    
    # reshape
    img = tf.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    # img = ((img / 255)) * 2 -1
    vtx = tf.reshape(vtx, [NUM_VERTEX*3])
    
    # img = tf.image.resize(img, [IMAGE_HEIGHT_RESIZE, IMAGE_WIDTH_RESIZE])

    return {'img': img, 'neutral_vtx': neutral_mesh, 'vtx': vtx}
