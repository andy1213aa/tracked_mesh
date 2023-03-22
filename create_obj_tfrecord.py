from absl import logging
from absl import flags
from absl import app
from pathlib import Path
import tensorflow as tf
import cv2
import os
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('record_pth',
                    default=None,
                    help='The directory where TFRecord file save.')

flags.DEFINE_string('data_pth',
                    default=None,
                    help='The directory of data root.')

types = ['E001_Neutral_Eyes_Open', 'E057_Cheeks_Puffed', 'E061_Lips_Puffed']


class ImageMesh2TFRecord_Converter():

    def __init__(self, records_path: str, data_pth: str):
        self._records_path = Path(records_path)
        assert not self._records_path.exists(
        ), f'There exist a TFRecord file at "{self._records_path}".'

        self._data_pth = Path(data_pth)
        assert self._data_pth.exists(), 'The data_pth is not exist.'

        self._images_pth = Path(f'{data_pth}/images')
        self._geom_pth = Path(f'{data_pth}/geom/')

    def _load_obj(self, pth):

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

    def _load_image(self, pth):
        if pth.exists():
            img = cv2.imread(str(pth))
            return True, img
        return False, ''

    def create_tfrecord(self):
        """Create the TFRecord
        
        Folder Structure:
        
        data/
        |--geom
        |  |--tracked_mesh
        |     |--type1
        |     |--type2
        |     |   ...
        |     |__typeN
        |  
        |--images
        |  |--angle1
        |  |  |--type1
        |  |  |--type2
        |  |  |  ...
        |  |  |__typeN
        |  |  
        |  |--angle2
        |  |   ...
        |  |__angleN
        |     |--type1
        |     |--type2
        |     |  ...
        |     |__typeN
        
        
        """
        writer = tf.io.TFRecordWriter(str(self._records_path))
        cnt = 0
        for typ in types:  # list
            
            # Read Mesh
            mesh_pth = Path(f'{self._geom_pth}/tracked_mesh/{typ}')

            type_pth = Path(f'{self._images_pth}/{typ}')
            camera_angle_pths = [
                pth for pth in type_pth.glob('*') if pth.is_dir()
            ]

            for ang_pth in camera_angle_pths:
                images_pth = [pth for pth in ang_pth.glob('*.png')]

                for img_pth in images_pth:
                    idx = img_pth.stem
                    res_mesh, mesh = self._load_obj(
                        Path(f'{mesh_pth}/{idx}.obj'))
                    res_img, img = self._load_image(img_pth)
                    if res_mesh and res_img:

                        img = img.astype(np.float32)
                        mesh = np.array(mesh).astype(np.float32)

                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                "img":
                                tf.train.Feature(bytes_list=tf.train.BytesList(
                                    value=[img.tobytes()])),
                                "vtx":
                                tf.train.Feature(bytes_list=tf.train.BytesList(
                                    value=[mesh.tobytes()]))
                            }))
                        cnt += 1

                        if cnt % 10 == 0:
                            logging.info(f'Finished {cnt} data.')
                        writer.write(example.SerializeToString())

        writer.close()


def main(argv):
    conveter = ImageMesh2TFRecord_Converter(FLAGS.record_pth, FLAGS.data_pth)
    conveter.create_tfrecord()
    # # res, vertexs = load_obj(FLAGS.data_dir)
    # if res:
    #     logging.info(vertexs)


if __name__ == '__main__':
    flags.mark_flags_as_required(['record_pth', 'data_pth'])
    app.run(main)