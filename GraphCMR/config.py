"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

BASE_DATA_DIR = '/x/cnris/github/data/'#'/z/cnris/github/data/'
CUSTOM_ROOT = '/x/cnris/demo2/'

'''
# Output folder to save test/train npz files
DATASET_NPZ_PATH = CUSTOM_ROOT


# Define paths to each dataset
H36M_ROOT = ''
LSP_ROOT = ''
LSP_ORIGINAL_ROOT = ''
UPI_S1H_ROOT = ''
MPII_ROOT = '/z/cnris/MPII/'
VLOG_ROOT_CROPPED = '/z/cnris/VLOG/frame_cache_cropped_testset/'

VLOG_ROOT = '/z/cnris/github/data/vlog/uncropped_keypoint/images/'
CT_ROOT = '/z/cnris/github/data/cross_task/uncropped_keypoint/images/'
IV_ROOT = '/z/cnris/github/data/instructions/uncropped_keypoint/images/'
YC_ROOT = '/z/cnris/github/data/youcook/uncropped_keypoint/images/'

VLOG_CROPPED_TEST = 'vlog_cropped_test.npz'
VLOG_TRAIN = 'vlog_train.npz'
VLOG_VAL = 'vlog_val.npz'
VLOG_TEST = 'vlog_test.npz'
COCO_ROOT = ''
UP_3D_ROOT = ''
VLOG_TEST = 'vlog/uncropped_keypoint/images.npz'
IV_TEST = 'instructions/uncropped_keypoint/images.npz'
YC_TEST = 'youcook/uncropped_keypoint/images.npz'
CT_TEST = 'cross_task/uncropped_keypoint/images.npz'
'''

'''
# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'extras', 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH,'extras', 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'extras', 'lsp_dataset_test.npz'),
                   'up-3d': join(DATASET_NPZ_PATH, 'extras', 'up_3d_lsp_test.npz'),
                   'vlog': join(BASE_DATA_DIR, VLOG_TEST),
                   'cross_task': join(BASE_DATA_DIR, CT_TEST),
                   'instructions': join(BASE_DATA_DIR, IV_TEST),
                   'youcook': join(BASE_DATA_DIR, YC_TEST),
                  },

                  {'lsp-orig': join(DATASET_NPZ_PATH, 'extras', 'lsp_dataset_original_train.npz'),
                   'up-3d': join(DATASET_NPZ_PATH, 'extras', 'up_3d_trainval.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'extras', 'mpii.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'extras', 'coco_2014_train.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m-p1': H36M_ROOT,
                 'h36m-p2': H36M_ROOT,
                 'lsp-orig': LSP_ORIGINAL_ROOT,
                 'lsp': LSP_ROOT,
                 'upi-s1h': UPI_S1H_ROOT,
                 'up-3d': UP_3D_ROOT,
                 'mpii': MPII_ROOT,
                 'coco': COCO_ROOT,
                 'vlog': VLOG_ROOT,
                 'cross_task': CT_ROOT,
                 'instructions': IV_ROOT,
                 'youcook': YC_ROOT
                }
'''
DATASET_SIZES = {'vlog': {'all': 4122, 'cropped_keypoint': 1807, 'uncropped_keypoint': 1807}, \
                'cross_task': {'all': 3908, 'cropped_keypoint': 1951, 'uncropped_keypoint': 1951}, \
                'instructions': {'all': 2258, 'cropped_keypoint': 829, 'uncropped_keypoint': 829}, \
                'youcook': {'all': 3370, 'cropped_keypoint': 1583, 'uncropped_keypoint': 1583}}

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
SMPL_FILE = 'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'

"""
Each dataset uses different sets of joints.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are:
0 - Right Ankle
1 - Right Knee
2 - Right Hip
3 - Left Hip
4 - Left Knee
5 - Left Ankle
6 - Right Wrist
7 - Right Elbow
8 - Right Shoulder
9 - Left Shoulder
10 - Left Elbow
11 - Left Wrist
12 - Neck (LSP definition)
13 - Top of Head (LSP definition)
14 - Pelvis (MPII definition)
15 - Thorax (MPII definition)
16 - Spine (Human3.6M definition)
17 - Jaw (Human3.6M definition)
18 - Head (Human3.6M definition)
19 - Nose
20 - Left Eye
21 - Right Eye
22 - Left Ear
23 - Right Ear
"""
JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18, 20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]

FOCAL_LENGTH = 5000.
INPUT_RES = 224

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
