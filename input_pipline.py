import tensorflow as tf
import numpy as np
import ml_collections
from absl import logging
from pathlib import Path

image_mean = 47.727367
image_std = 27.568207

mean_mesh_pth = '/home/aaron/Desktop/multiface/6674443_GHS/geom/vert_mean.bin'
with open(mean_mesh_pth, 'rb') as f:
    data = f.read()
    mesh_mean = np.frombuffer(data, dtype=np.float32).reshape((7306, 3))
    center = mesh_mean.mean(0)

SCALE = np.max(np.abs(mesh_mean - center))


def readTFRECORD(tfrecord_pth: str,
                 config: ml_collections.ConfigDict) -> tf.data:

    global IMAGE_WIDTH, IMAGE_HEIGHT, NUM_VERTEX

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_set = tf.data.TFRecordDataset(tfrecord_pth)


    IMAGE_WIDTH = config.render_size[1]
    IMAGE_HEIGHT = config.render_size[0]
    NUM_VERTEX = config.vertex_num

    data_set = (data_set
            .shuffle(config.data_size, reshuffle_each_iteration=True, seed=42)
            .map(parse, num_parallel_calls=AUTOTUNE)
            .repeat()
            .batch(config.batch_size, drop_remainder=True)
            # .cache()
            .map(augment_using_ops,  num_parallel_calls=AUTOTUNE)
            .prefetch(buffer_size=AUTOTUNE)
        )


    return data_set


def augment_using_ops(batch):
    # images = tf.image.random_flip_left_right(images)
    # images = tf.image.random_flip_up_down(images)
    batch['img'] = tf.image.random_brightness(batch['img'], 0.2)
    batch['img'] = tf.keras.layers.RandomZoom(0.2)(batch['img'])
    batch['img'] = tf.keras.layers.RandomRotation(0.1)(batch['img'])
    
    # images = tf.image.rot90(images)
    return batch


def parse(example_proto):
    features = tf.io.parse_single_example(
        example_proto,
        features={
            'img': tf.io.FixedLenFeature([], tf.string),
            'vtx': tf.io.FixedLenFeature([], tf.string),
            # 'vtx_mean': tf.io.FixedLenFeature([], tf.string),
            # 'tex': tf.io.FixedLenFeature([], tf.string),
            # 'verts_uvs': tf.io.FixedLenFeature([], tf.string),
            # 'faces_uvs': tf.io.FixedLenFeature([], tf.string),
            # 'verts_idx': tf.io.FixedLenFeature([], tf.string),
            # 'head_pose': tf.io.FixedLenFeature([], tf.string),
            # 'intricsic_camera': tf.io.FixedLenFeature([], tf.string),
            # 'extrinsic_camera': tf.io.FixedLenFeature([], tf.string),
        })

    img = features['img']
    vtx = features['vtx']
    # vtx_mean = features['vtx_mean']
    # tex = features['tex']
    # verts_uvs = features['verts_uvs']
    # faces_uvs = features['faces_uvs']
    # verts_idx = features['verts_idx']
    # head_pose = features['head_pose']
    # intricsic_camera = features['intricsic_camera']
    # extrinsic_camera = features['extrinsic_camera']

    img = tf.io.decode_raw(img, np.uint8)
    img = tf.cast(img, tf.float32)
    vtx = tf.io.decode_raw(vtx, np.float32)
    # vtx_mean = tf.io.decode_raw(vtx_mean, np.float32)
    # tex = tf.io.decode_raw(tex, np.float32)
    # verts_uvs = tf.io.decode_raw(verts_uvs, np.float32)
    # faces_uvs = tf.io.decode_raw(faces_uvs, np.float32)
    # verts_idx = tf.io.decode_raw(verts_idx, np.float32)
    # head_pose = tf.io.decode_raw(head_pose, np.float32)
    # intricsic_camera = tf.io.decode_raw(intricsic_camera, np.float32)
    # extrinsic_camera = tf.io.decode_raw(extrinsic_camera, np.float32)

    img = tf.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    img = tf.image.rgb_to_grayscale(img)
    img = (img - image_mean) / image_std
    vtx = tf.reshape(vtx, [NUM_VERTEX, 3])
    vtx /= SCALE
    # tex = tf.reshape(tex, [1024, 1024, 3])
    # vtx_mean = tf.reshape(vtx_mean, [NUM_VERTEX, 3])
    # verts_uvs = tf.reshape(verts_uvs, [-1, 2])
    # faces_uvs = tf.reshape(faces_uvs, [-1, 3])
    # verts_idx = tf.reshape(verts_idx, [-1, 3])
    # head_pose = tf.reshape(head_pose, [3, 4])
    # intricsic_camera = tf.reshape(intricsic_camera, [3, 3])
    # extrinsic_camera = tf.reshape(extrinsic_camera, [3, 4])

    return {
        'img': img,
        'vtx': vtx,
        # 'vtx_mean': vtx_mean,
        # 'texture_image': tex,
        # 'verts_uvs': verts_uvs,
        # 'faces_uvs': faces_uvs,
        # 'verts_idx': verts_idx,
        # 'head_pose': head_pose,
        # 'in_cam': intricsic_camera,
        # 'ex_cam': extrinsic_camera
    }
