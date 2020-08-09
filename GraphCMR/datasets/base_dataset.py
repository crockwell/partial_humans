from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

import config as cfg
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa, preprocess_generic

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        '''
        self.is_train = is_train
        '''
        self.options = options
        if options.dataset == 'custom':
            self.img_dir = join(cfg.CUSTOM_ROOT, 'images')
            self.data_dir = join(cfg.CUSTOM_ROOT, 'images.npz')
        else:
            self.img_dir = join(cfg.BASE_DATA_DIR, options.dataset, options.crop_setting, 'images') 
            self.data_dir = join(cfg.BASE_DATA_DIR, options.dataset, options.crop_setting, 'images.npz')
        print('imgdir:',self.img_dir)
        print('dataset:',self.data_dir) 
        self.normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
        self.data = np.load(self.data_dir) 
        self.imgname = self.data['imgname']
        
        self.length = self.imgname.shape[0]
        print('len',self.length)

    def rgb_processing(self, rgb_img_in):
        """Process rgb image and do augmentation."""
        rgb_img = preprocess_generic(rgb_img_in,
                      self.options.img_res)

        disp_img = preprocess_generic(rgb_img_in,
                      self.options.img_res, display=True)
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        disp_img = np.transpose(disp_img.astype('float32'),(2,0,1))/255.0
        return rgb_img, disp_img

    def __getitem__(self, index):
        item = {}
        
        # Load image
        imgname = join(self.img_dir, str(self.imgname[index]))
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Process image
        img, disp_img = self.rgb_processing(img)
        img = torch.from_numpy(img).float()
        disp_img = torch.from_numpy(disp_img).float()
        # Store image before normalization to use it in visualization
        try:
            item['master_index'] = self.mater_index[index]
        except AttributeError:
            item['master_index'] = ''
        item['disp_img'] = disp_img
        item['img'] = self.normalize_img(img)
        item['imgname'] = str(self.imgname[index])
        item['orig_shape'] = orig_shape
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''
        return item

    def __len__(self):
        return len(self.imgname)
