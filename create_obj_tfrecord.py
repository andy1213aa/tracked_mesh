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

IMAGE_WIDTH_RESIZE = 334  #240
IMAGE_HEIGHT_RESIZE = 512  #320
PCA_NUM = 160
VERTEX_NUM = 7306

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

subjects = [
    '6674443_GHS'
    # '2183941_GHS', '002539136_GHS', '002643814_GHS', '002757580_GHS',
    # '002914589_GHS', '5372021_GHS', '6674443_GHS', '6795937_GHS',
    # '7889059_GHS', '8870559_GHS',
]

types = [
    'E001_Neutral_Eyes_Open',
    'E003_Neutral_Eyes_Closed',
    "E006_Jaw_Drop_Brows_Up",
    "E007_Neck_Stretch_Brows_Up",
    "E008_Smile_Mouth_Closed",
    "E009_Smile_Mouth_Open",
    "E010_Smile_Stretched",
    "E011_Jaw_Open_Sharp_Corner_Lip_Stretch",
    "E012_Jaw_Open_Huge_Smile",
    "E013_Open_Lips_Mouth_Stretch_Nose_Wrinkled",  # don't use with mediapipe
    "E014_Open_Mouth_Stretch_Nose_Wrinkled",
    "E015_Jaw_Open_Upper_Lip_Raised",
    "E016_Raise_Upper_Lip_Scrunch_Nose",
    "E017_Jaw_Open_Mouth_Corners_Down_Nose_Wrinkled",
    "E018_Raise_Cheeks",
    "E019_Frown",
    "E020_Lower_Eyebrows",
    "E021_Pressed_Lips_Brows_Down",
    "E022_Raise_Inner_Eyebrows",
    "E023_Hide_Lips_Look_Up",
    "E024_Kiss_Lips_Look_Down",
    "E025_Shh",
    "E026_Oooo",
    "E027_Scrunch_Face_Squeeze_Eyes",
    "E028_Scream_Eyebrows_Up",
    "E029_Show_All_Teeth",
    "E030_Open_Mouth_Wide_Tongue_Up_And_Back",
    "E031_Jaw_Open_Lips_Together",
    "E032_Jaw_Open_Pull_Lips_In",
    "E033_Jaw_Clench",
    "E034_Jaw_Open_Lips_Pushed_Out",
    "E035_Lips_Together_Pushed_Forward",
    "E036_Stick_Lower_Lip_Out",
    "E037_Bite_Lower_Lip",
    "E038_Bite_Upper_Lip",
    "E039_Lips_Open_Right",
    "E040_Lips_Open_Left",
    "E041_Mouth_Nose_Right",
    "E042_Mouth_Nose_Left",
    "E043_Mouth_Open_Jaw_Right_Show_Teeth",
    "E044_Mouth_Open_Jaw_Left_Show_Teeth",
    "E045_Jaw_Back",
    "E046_Jaw_Forward",
    "E047_Tongue_Over_Upper_Lip",
    "E048_Tongue_Out_Lips_Closed",
    "E049_Mouth_Open_Tongue_Out",
    "E050_Bite_Tongue",
    "E051_Tongue_Out_Flat",
    "E052_Tongue_Out_Thick",
    "E053_Tongue_Out_Rolled",
    "E054_Tongue_Out_Right_Teeth_Showing",
    "E055_Tongue_Out_Left_Teeth_Showing",
    "E056_Suck_Cheeks_In",
    "E057_Cheeks_Puffed",
    "E058_Right_Cheek_Puffed",
    "E059_Left_Cheek_Puffed",
    "E060_Blow_Cheeks_Full_Of_Air",
    "E061_Lips_Puffed",
    "E062_Nostrils_Dilated",
    "E063_Nostrils_Sucked_In",
    "E064_Raise_Right_Eyebrow",
    "E065_Raise_Left_Eyebrow",
    "E074_Blink",
    # "SEN_a_good_morrow_to_you_my_boy",
    # "SEN_a_voice_spoke_near-at-hand",
    # "SEN_alfalfa_is_healthy_for_you",
    # "SEN_all_your_wishful_thinking_wont_change_that",
    # "SEN_allow_each_child_to_have_an_ice_pop",
    # "SEN_and_you_think_you_have_language_problems",
    # "SEN_approach_your_interview_with_statuesque_composure",
    # "SEN_are_you_looking_for_employment",
    # "SEN_as_she_drove_she_thought_about_her_plan",
    # "SEN_both_figures_would_go_higher_in_later_years",
    # "SEN_boy_youre_stirrin_early_a_sleepy_voice_said",
    # "SEN_by_eating_yogurt_you_may_live_longer",
    # "SEN_cliff_was_soothed_by_the_luxurious_massage",
    # "SEN_did_Shawn_catch_that_big_goose_without_help",
    # "SEN_do_they_make_class_biased_decisions",
    # "SEN_drop_five_forms_in_the_box_before_you_go_out",
    # "SEN_george_is_paranoid_about_a_future_gas_shortage",
    # "SEN_go_change_your_shoes_before_you_turn_around",
    # "SEN_greg_buys_fresh_milk_each_weekday_morning",
    # "SEN_have_you_got_our_keys_handy",
    # "SEN_how_do_oysters_make_pearls",
    # "SEN_how_long_would_it_be_occupied",
    # "SEN_how_ya_gonna_keep_em_down_on_the_farm",
    # "SEN_however_a_boys_lively_eyes_might_rove",
    # "SEN_id_rather_not_buy_these_shoes_than_be_overcharged",
    # "SEN_if_dark_came_they_would_lose_her",
    # "SEN_im_going_to_search_this_house",
    # "SEN_its_healthier_to_cook_without_sugar",
    # "SEN_jeffs_toy_go_cart_never_worked",
    # "SEN_more_he_could_take_at_leisure",
    # "SEN_nobody_else_showed_pleasure",
    # "SEN_oh_we_managed_she_said",
    # "SEN_she_always_jokes_about_too_much_garlic_in_his_food",
    # "SEN_take_charge_of_choosing_her_bridesmaids_gowns",
    # "SEN_thank_you_she_said_dusting_herself_off",
    # "SEN_the_small_boy_put_the_worm_on_the_hook",
    # "SEN_then_he_thought_me_more_perverse_than_ever",
    # "SEN_they_all_like_long_hot_showers",
    # "SEN_they_are_both_trend_following_methods",
    # "SEN_they_enjoy_it_when_I_audition",
    # "SEN_they_had_slapped_their_thighs",
    # "SEN_they_werent_as_well_paid_as_they_should_have_been",
    # "SEN_theyre_going_to_louse_me_up_good",
    # # "SEN_theyve_never_met_you_know", # don't use
    # "SEN_when_she_awoke_she_was_the_ship",
    # "SEN_why_buy_oil_when_you_always_use_mine",
    # "SEN_why_charge_money_for_such_garbage",
    # "SEN_why_put_such_a_high_value_on_being_top_dog",
    # "SEN_with_each_song_he_gave_verbal_footnotes",
    # "SEN_youre_boiling_milk_aint_you",
]
'''
data with right mp face landmark detection.
'''
views = [
    '400002'
    # '400009',
    # '400013',
    # '400015',
    # '400037',
    # '400041',
]
# views = [
#     "400002", "400007", "40009", "400012", "400013", "400015", "400016",
#     , "400019", "400023", "400029", "400030", "400031", "400037",
#     "400039", "400041", "400048", "400049", "400051", "400060", "400061",
#     "400063", "400064", "400069",
# ]


class ImageMesh2TFRecord_Converter():

    def __init__(self, records_path: str, data_pth: str,
                 tfrecord_writer: object):
        self._records_path = Path(records_path)
        # assert not self._records_path.exists(
        # ), f'There exist a TFRecord file at "{self._records_path}".'

        self._data_pth = Path(data_pth)
        assert self._data_pth.exists(), 'The data_pth is not exist.'

        self._images_pth = Path(f'{data_pth}/images')
        self._geom_pth = Path(f'{data_pth}/geom')
        self._tex_pth = Path(f'{self._geom_pth}/unwrapped_uv_1024')
        self.writer = tfrecord_writer
        self._camera_info = Path(f'{data_pth}/KRT')
        self._load_KRT()

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

    def _load_head_transform(self, pth):

        with open(pth, 'r') as f:
            lines = f.readlines()
        head_transform = []

        for i, _ in enumerate(lines):
            head_transform.append([float(x) for x in lines[i].strip().split()])
        return np.array(head_transform, dtype=np.float32).reshape((3, 4))

    def _load_KRT(self):
        # 定義相機參數列表
        self.camera_params = {}
        with open(self._camera_info, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            # 讀取相機ID
            camera_id = int(lines[i].strip())
            i += 1

            # 讀取相機內部參數
            intrinsics = []
            for _ in range(3):
                intrinsics.append([float(x) for x in lines[i].strip().split()])
                i += 1
            intrinsics = np.array(intrinsics).reshape((3, 3))

            #跳過一行
            i += 1

            # 讀取相機外部參數
            extrinsics = []
            for _ in range(3):
                extrinsics.append([float(x) for x in lines[i].strip().split()])
                i += 1
            extrinsics = np.array(extrinsics).reshape((3, 4))
            # 添加相機參數到dict
            self.camera_params[str(camera_id)] = {
                'K': intrinsics,
                'RT': extrinsics
            }

            #跳過一行
            i += 1

    def _load_image(self, pth):
        if pth.exists():
            img = cv2.imread(str(pth))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return True, img
        return False, ''

    def serialize_example(
        self,
        # expression,
        # camID,
        img,
        vtx,
        # texture,
        # verts_uvs,
        # faces_uvs,
        # verts_idx,
        # head_pose,
    ):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            # 'expression': self._bytes_feature(expression),
            # 'camID': self._int64_feature(camID),
            'img': self._bytes_feature(img),
            'vtx': self._bytes_feature(vtx),
            # 'texture': self._bytes_feature(texture),
            # 'verts_uvs': self._bytes_feature(verts_uvs),
            # 'faces_uvs': self._bytes_feature(faces_uvs),
            # 'verts_idx': self._bytes_feature(verts_idx),
            # 'head_pose': self._bytes_feature(head_pose),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(
            feature=feature))
        return example_proto

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
        |     |  ...160
        |     |__typeN
        
        
        """

        cnt = 0

        # # read mean vertexes from .bin file.
        # with open(Path(f'{self._geom_pth}/vert_mean.bin'), 'rb') as f:
        #     data = f.read()
        # mesh_mean = np.frombuffer(data, dtype=np.float32)

        for typ in tqdm(types):  # list

            mesh_pth = Path(f'{self._geom_pth}/tracked_mesh/{typ}')
            type_pth = Path(f'{self._images_pth}/{typ}')
            type_tex_pth = Path(f'{self._tex_pth}/{typ}')

            camera_angle_pths = [
                pth for pth in type_pth.glob('*')
                if pth.is_dir() and pth.stem in views
            ]

            tex_angle_pths = [
                pth for pth in type_tex_pth.glob('*')
                if pth.is_dir() and pth.stem in views
            ]

            for i, ang_pth in enumerate(camera_angle_pths):
                camID = int(ang_pth.stem)
                # # read intrinsic
                # intricsic_camera = self.camera_params[ang_pth.stem]['K']
                # # read extrinsic
                # extrinsic_camera = self.camera_params[ang_pth.stem]['RT']

                images_pth = [pth for pth in ang_pth.glob('*.png')]
                tex_pth = [pth
                           for pth in tex_angle_pths[i].glob('*.png')]
                
                images_pth.sort()
                tex_pth.sort()
                #只拿前25個 因為後面表情都統一為沒表情 造成資料不平均
                images_pth = images_pth[:25]
                tex_pth = tex_pth[:25]

                for j, img_pth in tqdm(enumerate(images_pth)):
                    idx = img_pth.stem

                    # read vertexes from .bin file.
                    with open(Path(f'{mesh_pth}/{idx}.bin'), 'rb') as f:
                        data = f.read()
                    mesh = np.frombuffer(data, dtype=np.float32)

                    # read uv mapping from obj file.
                    _, faces, aux = load_obj(Path(f'{mesh_pth}/{idx}.obj'))

                    verts_uvs = aux.verts_uvs.numpy().astype(np.float32)
                    faces_uvs = faces.textures_idx.numpy().astype(np.float32)
                    verts_idx = faces.verts_idx.numpy().astype(np.float32)

                    # read 2D image
                    res_img, img = self._load_image(img_pth)

                    # read texture map

                    res_tex_img, tex_img = self._load_image(tex_pth[j])
                    # cv2.imwrite('teset.png', tex_img)
                    # read head transform
                    head_pose = self._load_head_transform(
                        Path(f'{mesh_pth}/{idx}_transform.txt'))

                    if res_img:
                        img = cv2.resize(
                            img, [IMAGE_WIDTH_RESIZE, IMAGE_HEIGHT_RESIZE])

                        tf_example = self.serialize_example(
                            # typ.encode('utf-8'),
                            # camID,
                            img.tobytes(),
                            mesh.tobytes(),
                            # tex_img.tobytes(),
                            # verts_uvs.tobytes(),
                            # faces_uvs.tobytes(),
                            # verts_idx.tobytes(),
                            # head_pose.tobytes(),
                        )
                        cnt += 1
                        self.writer.write(tf_example.SerializeToString())

        print(f'total_number: {cnt}')


def main(argv):

    assert not Path(FLAGS.record_pth).exists(
    ), f'There exist a TFRecord file at "{FLAGS.record_pth}".'

    writer = tf.io.TFRecordWriter(str(FLAGS.record_pth))

    for sub in subjects:
        conveter = ImageMesh2TFRecord_Converter(
            FLAGS.record_pth,
            FLAGS.data_pth + f'/{sub}',
            writer,
        )

        conveter.create_tfrecord()

    writer.close()


if __name__ == '__main__':
    flags.mark_flags_as_required(['record_pth', 'data_pth'])
    app.run(main)

#f'/home/aaron/Desktop/multiface/{sub}'