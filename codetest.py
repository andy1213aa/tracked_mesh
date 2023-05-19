import jax
import tensorflow as tf
import ml_collections
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch

from pytorch3d.renderer import (PointLights, MeshRenderer,
                                RasterizationSettings, MeshRasterizer,
                                SoftPhongShader, Textures, PerspectiveCameras)

from pytorch3d.structures import Meshes
from PIL import Image
from pytorch3d.io import load_objs_as_meshes, load_obj
import time
from tqdm import tqdm
# tf.config.set_visible_devices([], 'GPU')

IMAGE_WIDTH = 334
IMAGE_HEIGHT = 512

# IMAGE_WIDTH_RESIZE = 240
# IMAGE_HEIGHT_RESIZE = 320

NUM_VERTEX = 7306

config = ml_collections.ConfigDict()
config.model = 'Classic_CNN'
config.dataset = 'multiface'
config.image_size = (334, 512)
config.num_epochs = 200
config.warmup_epochs = 10
config.batch_size = 32
config.learning_rate = 0.0001
config.log_every_steps = 1
config.vertex = 7306
config.num_train_steps = -1


def readTFRECORD(tfrecord_pth: str,
                 config: ml_collections.ConfigDict) -> tf.data:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_set = tf.data.TFRecordDataset(tfrecord_pth)
    # data_set = data_set.repeat()
    data_set = data_set.map(parse, num_parallel_calls=AUTOTUNE)

    # data_set = data_set.shuffle(config.batch_size * 16,
    #                             reshuffle_each_iteration=True)
    data_batch = data_set.batch(config.batch_size, drop_remainder=True)
    data_batch = data_batch.prefetch(buffer_size=AUTOTUNE)
    return data_batch


feature_description = {
    'camID': tf.io.FixedLenFeature([], tf.int64),
    # 'vtx': tf.io.FixedLenFeature([], tf.string),
}


def parse(example_proto):

    features = tf.io.parse_single_example(example_proto,
                                          features=feature_description)

    camID = features['camID']
    # print(img)
    # vtx = features['vtx']

    # img = tf.io.decode_image(img,channels=3, detype=tf.float32)
    # camID = tf.io.decode_raw(camID, tf.int32)
    # # # vtx = tf.io.decode_raw(vtx, np.float32)

    # img = tf.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # vtx = tf.reshape(vtx, [NUM_VERTEX, 3])

    return {
        'camID': camID,
        # 'vtx': vtx
    }


ds = readTFRECORD('../training_data/test.tfrecord', config)
# 開始計時
start_time = time.time()
# 初始化計數器
step_count = 0
for i, batch in enumerate(ds):
    step_count += 1
    print(batch['camID'])
    
    if time.time() - start_time > 1.:
        print("即時每秒執行的步驟數：", step_count)
        step_count = 0
        start_time = time.time()
