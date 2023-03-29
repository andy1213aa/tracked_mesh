from absl import logging
from absl import flags
from absl import app
from pathlib import Path
import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
import argparse
import pickle



def main(argv):
    
     
    # # res, vertexs = load_obj(FLAGS.data_dir)
    # if res:
    #     logging.info(vertexs)


if __name__ == '__main__':
    flags.mark_flags_as_required(['record_pth', 'data_pth'])
    app.run(main)