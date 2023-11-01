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
from pytorch3d.io import load_objs_as_meshes
# from pytorch3d.io import load_obj

FLAGS = flags.FLAGS
flags.DEFINE_string('record_pth',
                    default=None,
                    help='The directory where TFRecord file save.')

flags.DEFINE_string('data_pth',
                    default=None,
                    help='The directory of data root.')

flags.DEFINE_integer('n_com',
                     default=160,
                     help='The components of linear PCA model.')


class Mesh2PCA():

    def __init__(self, records_path: str, data_pth: str):
        self._records_path = Path(records_path)
        assert not self._records_path.exists(
        ), f'There exists a file at "{self._records_path}".'

        self._data_pth = Path(data_pth)
        assert self._data_pth.exists(), 'The data_pth is not exist.'

    def _load_obj_vertex_data(self, obj_file_path):
        # 使用 PyTorch3D 来加载 .obj 文件
        meshes = load_objs_as_meshes([obj_file_path])
        vertices = meshes.verts_packed().detach().cpu().numpy()
        return vertices
    
    def create_pca(self, n_com: int):
        pca = PCA(n_com)
        cnt = 0
        total_verts = []
        
        for category_folder in os.listdir(self._data_pth):
            category_path = self._data_pth / category_folder
            obj_file = category_path / 'model_align.obj'
            vertices = self._load_obj_vertex_data(obj_file)
            total_verts.append(vertices)       
        
        total_verts = np.array(total_verts)
        total_verts = einops.rearrange(total_verts, 'b v c -> b (v c)')
        print(total_verts.shape)
        pca.fit(total_verts)
        with open(self._records_path, 'wb') as f:
            pickle.dump(pca, f)

        logging.info(f"Finish PCA at {self._records_path}; n_com = {n_com}")


def main(argv):
    
    conveter = Mesh2PCA(FLAGS.record_pth,FLAGS.data_pth)
    conveter.create_pca(FLAGS.n_com)


if __name__ == '__main__':
    flags.mark_flags_as_required(['data_pth', 'record_pth', 'n_com'])
    app.run(main)