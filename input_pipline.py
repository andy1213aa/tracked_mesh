import tensorflow as tf
import numpy as np
import ml_collections
from absl import logging
from pathlib import Path

IMAGE_WIDTH = 240
IMAGE_HEIGHT = 320

# IMAGE_WIDTH_RESIZE = 240
# IMAGE_HEIGHT_RESIZE = 320

NUM_VERTEX = 7306

# # 7306*3 array
# vert_mean = np.load('../training_data/6674443_vert_mean.npy')
# # 1 float32
# vert_var = np.load('../training_data/6674443_vert_var.npy')


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
    data_set = data_set.shuffle(config.data_size // 2,
                                reshuffle_each_iteration=True)
    data_batch = data_set.batch(config.batch_size, drop_remainder=True)
    data_batch = data_batch.prefetch(buffer_size=AUTOTUNE)
    return data_batch


def parse(example_proto):
    features = tf.io.parse_single_example(
        example_proto,
        features={
            'img': tf.io.FixedLenFeature([], tf.string),
            'vtx': tf.io.FixedLenFeature([], tf.string),
            'vtx_mean': tf.io.FixedLenFeature([], tf.string),
            'tex': tf.io.FixedLenFeature([], tf.string),
            'verts_uvs': tf.io.FixedLenFeature([], tf.string),
            'faces_uvs': tf.io.FixedLenFeature([], tf.string),
            'verts_idx': tf.io.FixedLenFeature([], tf.string),
            'head_pose': tf.io.FixedLenFeature([], tf.string),
            'intricsic_camera': tf.io.FixedLenFeature([], tf.string),
            'extrinsic_camera': tf.io.FixedLenFeature([], tf.string),
        })

    img = features['img']
    vtx = features['vtx']
    vtx_mean = features['vtx_mean']
    tex = features['tex']
    verts_uvs = features['verts_uvs']
    faces_uvs = features['faces_uvs']
    verts_idx = features['verts_idx']
    head_pose = features['head_pose']
    intricsic_camera = features['intricsic_camera']
    extrinsic_camera = features['extrinsic_camera']

    img = tf.io.decode_raw(img, np.float32)
    vtx = tf.io.decode_raw(vtx, np.float32)
    vtx_mean = tf.io.decode_raw(vtx_mean, np.float32)
    tex = tf.io.decode_raw(tex, np.float32)
    verts_uvs = tf.io.decode_raw(verts_uvs, np.float32)
    faces_uvs = tf.io.decode_raw(faces_uvs, np.float32)
    verts_idx = tf.io.decode_raw(verts_idx, np.float32)
    head_pose = tf.io.decode_raw(head_pose, np.float32)
    intricsic_camera = tf.io.decode_raw(intricsic_camera, np.float32)
    extrinsic_camera = tf.io.decode_raw(extrinsic_camera, np.float32)

    img = tf.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    tex = tf.reshape(tex, [1024, 1024, 3])
    vtx = tf.reshape(vtx, [NUM_VERTEX, 3])
    vtx_mean = tf.reshape(vtx_mean, [NUM_VERTEX, 3])
    verts_uvs = tf.reshape(verts_uvs, [-1, 2])
    faces_uvs = tf.reshape(faces_uvs, [-1, 3])
    verts_idx = tf.reshape(verts_idx, [-1, 3])
    head_pose = tf.reshape(head_pose, [3, 4])
    intricsic_camera = tf.reshape(intricsic_camera, [3, 3])
    extrinsic_camera = tf.reshape(extrinsic_camera, [3, 4])

    # focal = np.array([intricsic_camera.numpy()[0, 0], intricsic_camera.numpy()[1, 1]])

    return {
        'img': img,
        'vtx': vtx,
        'vtx_mean': vtx_mean,
        'tex': tex,
        'verts_uvs': verts_uvs,
        'faces_uvs': faces_uvs,
        'verts_idx': verts_idx,
        'head_pose': head_pose,
        'in_cam': intricsic_camera,
        'ex_cam': extrinsic_camera
    }
