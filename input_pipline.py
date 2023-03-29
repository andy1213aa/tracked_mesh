import tensorflow as tf
import numpy as np
import ml_collections

IMAGE_WIDTH  = 240
IMAGE_HEIGHT = 320

# IMAGE_WIDTH_RESIZE = 240
# IMAGE_HEIGHT_RESIZE = 320

NUM_VERTEX = 7306


def readTFRECORD(tfrecord_pth: str,
                 config: ml_collections.ConfigDict) -> tf.data:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_set = tf.data.TFRecordDataset(tfrecord_pth)
    data_set = data_set.repeat()
    data_set = data_set.map(parse, num_parallel_calls=AUTOTUNE)

    data_set = data_set.shuffle(config.batch_size * 16,
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

    img = tf.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    vtx = tf.reshape(vtx, [NUM_VERTEX * 3])
    # img = tf.image.resize(img, [IMAGE_HEIGHT_RESIZE, IMAGE_WIDTH_RESIZE])

    return {'img': img, 'vtx': vtx}
