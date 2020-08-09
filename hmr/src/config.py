"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import os.path as osp
from os import makedirs
import os
from glob import glob
from datetime import datetime
import json

import numpy as np

curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
SMPL_MODEL_PATH = osp.join(model_dir, 'neutral_smpl_with_cocoplus_reg.pkl')
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')
DATASET_SIZES = {'vlog': {'all': 4122, 'cropped_keypoint': 1807, 'uncropped_keypoint': 1807}, \
                'cross_task': {'all': 3908, 'cropped_keypoint': 1951, 'uncropped_keypoint': 1951}, \
                'instructions': {'all': 2258, 'cropped_keypoint': 829, 'uncropped_keypoint': 829}, \
                'youcook': {'all': 3370, 'cropped_keypoint': 1583, 'uncropped_keypoint': 1583}}

# Default pre-trained model path for the demo.
PRETRAINED_MODEL = osp.join(model_dir, 'ours/model.ckpt-694216')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')
flags.DEFINE_string('load_path', None,
                    'run from this checkpoint')
flags.DEFINE_integer('batch_size', 1,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('gpu', '0',
                     'GPU to use for training / testing')

# Don't change if testing:
flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate regressor')
flags.DEFINE_string('model_type', 'resnet_fc3_dropout',
                    'Specifies which network to use')
flags.DEFINE_string(
    'joint_type', 'cocoplus',
    'cocoplus (19 keypoints) or lsp 14 keypoints, returned by SMPL. We use cocoplus for evaluation.')

# visualization settings
flags.DEFINE_string('write_imgs', 'False',
    'whether to save images')
flags.DEFINE_string('white_background', 'True',
    'if True, overlays mesh on white background. If false, overlays with image')

# evaluation settings
flags.DEFINE_string('base_dir', None, 'data directory')
flags.DEFINE_string('eval_pck', 'False', 'if True, evaluate PCK')
flags.DEFINE_float('pck_eval_threshold', 0.5, 'PCK Eval threshold')
flags.DEFINE_string('dataset', None, 'which dataset to evaluate')
flags.DEFINE_string('crop_setting', None, 'evaluate on all, uncropped_keypoint, or cropped_keypoint')

# additional custom run settings
flags.DEFINE_string('custom_dir', None, 'data directory for custom run')
flags.DEFINE_integer('num_imgs', None,
                     'Number of images in dataset')
flags.DEFINE_string('logdir','hmr_pck_results.txt', 'directory to save test results')

# demo settings
flags.DEFINE_string('img_path', 'vlog_d_R_8_v_DWeCvOFcdR8_012_frame000931.jpg', 'Image to run')

# Hyperparameters:
flags.DEFINE_float('e_wd', 0.0001, 'Encoder weight decay')

def get_num_imgs(config):
    num_imgs = config.num_imgs
    if config.dataset in DATASET_SIZES:
        num_imgs = DATASET_SIZES[config.dataset][config.crop_setting]
    return num_imgs

def get_config():
    config = flags.FLAGS
    config(sys.argv)
    config.num_imgs = get_num_imgs(config)
    return config