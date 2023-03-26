from absl import logging
from absl import flags
from absl import app
from pathlib import Path
import tensorflow as tf
import cv2
import os
import numpy as np

IMAGE_WIDTH_RESIZE = 240
IMAGE_HEIGHT_RESIZE = 320

FLAGS = flags.FLAGS
flags.DEFINE_string('record_pth',
                    default=None,
                    help='The directory where TFRecord file save.')

flags.DEFINE_string('data_pth',
                    default=None,
                    help='The directory of data root.')

types= [
        'E001_Neutral_Eyes_Open',
        "E006_Jaw_Drop_Brows_Up",
        # "E007_Neck_Stretch_Brows_Up",
        # "E008_Smile_Mouth_Closed",
        # "E009_Smile_Mouth_Open",
        # "E010_Smile_Stretched",
        # "E011_Jaw_Open_Sharp_Corner_Lip_Stretch",
        # "E012_Jaw_Open_Huge_Smile",
        # "E013_Open_Lips_Mouth_Stretch_Nose_Wrinkled",
        # "E014_Open_Mouth_Stretch_Nose_Wrinkled",
        # "E015_Jaw_Open_Upper_Lip_Raised",
        # "E016_Raise_Upper_Lip_Scrunch_Nose",
        # "E017_Jaw_Open_Mouth_Corners_Down_Nose_Wrinkled",
        # "E018_Raise_Cheeks",
        # "E019_Frown",
        # "E020_Lower_Eyebrows",
        # "E021_Pressed_Lips_Brows_Down",
        # "E022_Raise_Inner_Eyebrows",
        # "E023_Hide_Lips_Look_Up",
        # "E024_Kiss_Lips_Look_Down",
        # "E025_Shh",
        # "E026_Oooo",
        # "E027_Scrunch_Face_Squeeze_Eyes",
        # "E028_Scream_Eyebrows_Up",
        # "E029_Show_All_Teeth",
        # "E030_Open_Mouth_Wide_Tongue_Up_And_Back",
        # "E031_Jaw_Open_Lips_Together",
        # "E032_Jaw_Open_Pull_Lips_In",
        # "E033_Jaw_Clench",
        # "E034_Jaw_Open_Lips_Pushed_Out",
        # "E035_Lips_Together_Pushed_Forward",
        # "E036_Stick_Lower_Lip_Out",
        # "E037_Bite_Lower_Lip",
        # "E038_Bite_Upper_Lip",
        # "E039_Lips_Open_Right",
        # "E040_Lips_Open_Left",
        # "E041_Mouth_Nose_Right",
        # "E042_Mouth_Nose_Left",
        # "E043_Mouth_Open_Jaw_Right_Show_Teeth",
        # "E044_Mouth_Open_Jaw_Left_Show_Teeth",
        # "E045_Jaw_Back",
        # "E046_Jaw_Forward",
        # "E047_Tongue_Over_Upper_Lip",
        # "E048_Tongue_Out_Lips_Closed",
        # "E049_Mouth_Open_Tongue_Out",
        # "E050_Bite_Tongue",
        # "E051_Tongue_Out_Flat",
        # "E052_Tongue_Out_Thick",
        # "E053_Tongue_Out_Rolled",
        # "E054_Tongue_Out_Right_Teeth_Showing",
        # "E055_Tongue_Out_Left_Teeth_Showing",
        # "E056_Suck_Cheeks_In",
        # "E057_Cheeks_Puffed",
        # "E058_Right_Cheek_Puffed",
        # "E059_Left_Cheek_Puffed",
        # "E060_Blow_Cheeks_Full_Of_Air",
        # "E061_Lips_Puffed",
        # "E062_Nostrils_Dilated",
        # "E063_Nostrils_Sucked_In",
        # "E064_Raise_Right_Eyebrow",
        # "E065_Raise_Left_Eyebrow",
        # "E074_Blink"

    ]

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
                        img = cv2.resize(img, [IMAGE_HEIGHT_RESIZE, IMAGE_WIDTH_RESIZE])
                        mesh = np.array(mesh).astype(np.float32)
                        print('--------------------------')
                        print(f'Obj: {mesh_pth}/{idx}.obj')
                        print(f'Img: {img_pth}')
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

                        if cnt % 100 == 0:
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