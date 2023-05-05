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



IMAGE_WIDTH_RESIZE = 240  #240
IMAGE_HEIGHT_RESIZE = 320  #320
PCA_NUM = 160
VERTEX_NUM = 7306

FLAGS = flags.FLAGS
flags.DEFINE_string('record_pth',
                    default=None,
                    help='The directory where TFRecord file save.')

flags.DEFINE_string('data_pth',
                    default=None,
                    help='The directory of data root.')
flags.DEFINE_bool('pca', default=False, help='Do pca modeling.')
flags.DEFINE_bool('tfrecord', default=False, help='Build tfrecord data.')

subjects = [
    '6674443_GHS'
    # '2183941_GHS', '002539136_GHS', '002643814_GHS', '002757580_GHS',
    # '002914589_GHS', '5372021_GHS', '6674443_GHS', '6795937_GHS',
    # '7889059_GHS', '8870559_GHS'
]

types = [
    'E001_Neutral_Eyes_Open',
    'E003_Neutral_Eyes_Closed',  "E006_Jaw_Drop_Brows_Up",
    "E007_Neck_Stretch_Brows_Up", "E008_Smile_Mouth_Closed",
    "E009_Smile_Mouth_Open", "E010_Smile_Stretched",
    "E011_Jaw_Open_Sharp_Corner_Lip_Stretch", "E012_Jaw_Open_Huge_Smile",
    "E013_Open_Lips_Mouth_Stretch_Nose_Wrinkled",
    "E014_Open_Mouth_Stretch_Nose_Wrinkled", "E015_Jaw_Open_Upper_Lip_Raised",
    "E016_Raise_Upper_Lip_Scrunch_Nose",
    "E017_Jaw_Open_Mouth_Corners_Down_Nose_Wrinkled", "E018_Raise_Cheeks",
    "E019_Frown", "E020_Lower_Eyebrows", "E021_Pressed_Lips_Brows_Down",
    "E022_Raise_Inner_Eyebrows", "E023_Hide_Lips_Look_Up",
    "E024_Kiss_Lips_Look_Down", "E025_Shh", "E026_Oooo",
    "E027_Scrunch_Face_Squeeze_Eyes", "E028_Scream_Eyebrows_Up",
    "E029_Show_All_Teeth", "E030_Open_Mouth_Wide_Tongue_Up_And_Back",
    "E031_Jaw_Open_Lips_Together", "E032_Jaw_Open_Pull_Lips_In",
    "E033_Jaw_Clench", "E034_Jaw_Open_Lips_Pushed_Out",
    "E035_Lips_Together_Pushed_Forward", "E036_Stick_Lower_Lip_Out",
    "E037_Bite_Lower_Lip", "E038_Bite_Upper_Lip", "E039_Lips_Open_Right",
    "E040_Lips_Open_Left", "E041_Mouth_Nose_Right", "E042_Mouth_Nose_Left",
    "E043_Mouth_Open_Jaw_Right_Show_Teeth",
    "E044_Mouth_Open_Jaw_Left_Show_Teeth", "E045_Jaw_Back", "E046_Jaw_Forward",
    "E047_Tongue_Over_Upper_Lip", "E048_Tongue_Out_Lips_Closed",
    "E049_Mouth_Open_Tongue_Out", "E050_Bite_Tongue", "E051_Tongue_Out_Flat",
    "E052_Tongue_Out_Thick", "E053_Tongue_Out_Rolled",
    "E054_Tongue_Out_Right_Teeth_Showing",
    "E055_Tongue_Out_Left_Teeth_Showing", "E056_Suck_Cheeks_In",
    "E057_Cheeks_Puffed", "E058_Right_Cheek_Puffed", "E059_Left_Cheek_Puffed",
    "E060_Blow_Cheeks_Full_Of_Air", "E061_Lips_Puffed",
    "E062_Nostrils_Dilated", "E063_Nostrils_Sucked_In",
    "E064_Raise_Right_Eyebrow", "E065_Raise_Left_Eyebrow", "E074_Blink",
    "SEN_a_good_morrow_to_you_my_boy", "SEN_a_voice_spoke_near-at-hand",
    "SEN_alfalfa_is_healthy_for_you",
    "SEN_all_your_wishful_thinking_wont_change_that",
    "SEN_allow_each_child_to_have_an_ice_pop",
    "SEN_and_you_think_you_have_language_problems",
    "SEN_approach_your_interview_with_statuesque_composure",
    "SEN_are_you_looking_for_employment",
    "SEN_as_she_drove_she_thought_about_her_plan",
    "SEN_both_figures_would_go_higher_in_later_years",
    "SEN_boy_youre_stirrin_early_a_sleepy_voice_said",
    "SEN_by_eating_yogurt_you_may_live_longer",
    "SEN_cliff_was_soothed_by_the_luxurious_massage",
    "SEN_did_Shawn_catch_that_big_goose_without_help",
    "SEN_do_they_make_class_biased_decisions",
    "SEN_drop_five_forms_in_the_box_before_you_go_out",
    "SEN_george_is_paranoid_about_a_future_gas_shortage",
    "SEN_go_change_your_shoes_before_you_turn_around",
    "SEN_greg_buys_fresh_milk_each_weekday_morning",
    "SEN_have_you_got_our_keys_handy", "SEN_how_do_oysters_make_pearls",
    "SEN_how_long_would_it_be_occupied",
    "SEN_how_ya_gonna_keep_em_down_on_the_farm",
    "SEN_however_a_boys_lively_eyes_might_rove",
    "SEN_id_rather_not_buy_these_shoes_than_be_overcharged",
    "SEN_if_dark_came_they_would_lose_her",
    "SEN_im_going_to_search_this_house",
    "SEN_its_healthier_to_cook_without_sugar",
    "SEN_jeffs_toy_go_cart_never_worked", "SEN_more_he_could_take_at_leisure",
    "SEN_nobody_else_showed_pleasure", "SEN_oh_we_managed_she_said",
    "SEN_she_always_jokes_about_too_much_garlic_in_his_food",
    "SEN_take_charge_of_choosing_her_bridesmaids_gowns",
    "SEN_thank_you_she_said_dusting_herself_off",
    "SEN_the_small_boy_put_the_worm_on_the_hook",
    "SEN_then_he_thought_me_more_perverse_than_ever",
    "SEN_they_all_like_long_hot_showers",
    "SEN_they_are_both_trend_following_methods",
    "SEN_they_enjoy_it_when_I_audition", "SEN_they_had_slapped_their_thighs",
    "SEN_they_werent_as_well_paid_as_they_should_have_been",
    "SEN_theyre_going_to_louse_me_up_good", "SEN_theyve_never_met_you_know",
    "SEN_when_she_awoke_she_was_the_ship",
    "SEN_why_buy_oil_when_you_always_use_mine",
    "SEN_why_charge_money_for_such_garbage",
    "SEN_why_put_such_a_high_value_on_being_top_dog",
    "SEN_with_each_song_he_gave_verbal_footnotes",
    "SEN_youre_boiling_milk_aint_you"
]

'''
data with right mp face landmark detection.
'''
views=['400009','400013', '400015','400037', '400041']  
# views = [
#     "400002", "400007", "40009", "400012", "400013", "400015", "400016",
#     , "400019", "400023", "400029", "400030", "400031", "400037",
#     "400039", "400041", "400048", "400049", "400051", "400060", "400061",
#     "400063", "400064", "400069"
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
        self._geom_pth = Path(f'{data_pth}/geom/')
        self.writer = tfrecord_writer

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
        |     |  ...160
        |     |__typeN
        
        
        """

        cnt = 0
        for typ in types:  # list

            # Read Mesh
            mesh_pth = Path(f'{self._geom_pth}/tracked_mesh/{typ}')

            type_pth = Path(f'{self._images_pth}/{typ}')
            camera_angle_pths = [
                pth for pth in type_pth.glob('*')
                if pth.is_dir() and pth.stem in views
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
                        img = cv2.resize(
                            img, [IMAGE_HEIGHT_RESIZE, IMAGE_WIDTH_RESIZE])
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

                        # if cnt % 100 == 0:
                        #     logging.info(f'Finished {cnt} data.')
                        self.writer.write(example.SerializeToString())

        # print(f'total_number: {cnt}')

    def create_pca(self):

        total_mesh = []
        for typ in types:
            # Read Mesh
            type_pth = Path(f'{self._geom_pth}/tracked_mesh/{typ}')
            mesh_pth = [pth for pth in type_pth.glob('*.obj')]

            for pth in mesh_pth:
                logging.info(pth)
                res_mesh, mesh = self._load_obj(Path(f'{pth}'))
                if res_mesh:
                    total_mesh.append(mesh)

        total_mesh = np.array(total_mesh)
        total_mesh = einops.rearrange(total_mesh, 'b v c -> b (v c)')
        
        return total_mesh


def main(argv):
    if FLAGS.tfrecord:
        assert not Path(FLAGS.record_pth).exists(
        ), f'There exist a TFRecord file at "{FLAGS.record_pth}".'

        writer = tf.io.TFRecordWriter(str(FLAGS.record_pth))

        for sub in subjects:
            conveter = ImageMesh2TFRecord_Converter(
                FLAGS.record_pth, f'/home/aaron/Desktop/multiface/{sub}',
                writer)
            conveter.create_tfrecord()

        writer.close()
    # elif FLAGS.pca:
    #     pca = PCA(PCA_NUM)
    #     sample = np.zeros((1, 7306))
    #     for sub in subjects:
    #         conveter = ImageMesh2TFRecord_Converter(
    #             FLAGS.record_pth, f'/home/aaron/Desktop/multiface/{sub}', pca)

    #         pca = conveter.create_pca()
    #     pca = self.writer.fit(total_mesh)
        
        # with open('pca.pickle', 'wb') as f:
        #     pickle.dump(pca, f)
        # logging.info(f"Finish PCA; n_com = {PCA_NUM}")
    else:
        logging.info('Nothing to do.')
        logging.info('Use --pca or --tfrecord to assign the job.')


if __name__ == '__main__':
    flags.mark_flags_as_required(['record_pth'])
    app.run(main)