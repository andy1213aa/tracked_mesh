from absl import logging
from absl import flags
from absl import app
from pathlib import Path
import tensorflow as tf
import cv2
import os
import numpy as np
import einops
from sklearn.decomposition import PCA
import pickle
from pytorch3d.io import load_obj
from PIL import Image
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'record_pth',
    default=None,
    help='The directory where TFRecord file save.',
)

flags.DEFINE_string(
    'data_pth',
    default=None,
    help='The directory of data root.',
)
RESIZE_WIDTH = 756
RESIZE_HEIGHT = 1008


class ImageMesh2TFRecord_Converter():

    def __init__(self, records_path: str, data_pth: str,
                 tfrecord_writer: object):
        self._records_path = Path(records_path)
        self.writer = tfrecord_writer
        self._data_pth = Path(data_pth)

    # The following functions can be used to convert a value to a type compatible
    # with tf.train.Example.

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _load_image(self, pth):
        if pth.exists():
            img = cv2.imread(str(pth))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return True, img
        return False, ''

    def serialize_example(
        self,
        img,
        vtx,
    ):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            'img': self._bytes_feature(img),
            'vtx': self._bytes_feature(vtx),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(
            feature=feature))
        return example_proto

    def _load_obj_vertex_data(self, obj_file_path):
        # 使用 PyTorch3D 来加载 .obj 文件
        meshes = load_objs_as_meshes([obj_file_path])
        vertices = meshes.verts_packed().detach().cpu().numpy()
        return vertices

    def create_tfrecord(self):
        n = 0
        for category_folder in tqdm(os.listdir(self._data_pth)):
            category_path = self._data_pth / category_folder
            obj_file = category_path / 'model_align.obj'
            image_folder = category_path / 'images_rvm'
            vertices = self._load_obj_vertex_data(obj_file).flatten()
            for img_path in os.listdir(image_folder):

                res, img = self._load_image(image_folder / img_path)
                if res:
                    img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))
                tf_example = self.serialize_example(
                    img.tobytes(),
                    vertices.tobytes(),
                )

                self.writer.write(tf_example.SerializeToString())
                n += 1
        print(f'total size: {n}')


def main(argv):

    assert not Path(FLAGS.record_pth).exists(
    ), f'There exist a TFRecord file at "{FLAGS.record_pth}".'

    assert Path(FLAGS.data_pth).exists(), 'The data_pth is not exist.'

    writer = tf.io.TFRecordWriter(str(FLAGS.record_pth))

    conveter = ImageMesh2TFRecord_Converter(
        records_path=FLAGS.record_pth,
        data_pth=FLAGS.data_pth,
        tfrecord_writer=writer,
    )

    conveter.create_tfrecord()

    writer.close()


if __name__ == '__main__':
    flags.mark_flags_as_required(['record_pth', 'data_pth'])
    app.run(main)

#f'/home/aaron/Desktop/multiface/{sub}'