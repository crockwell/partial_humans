import os
from os.path import join
import numpy as np
import config as cfg

def extract(dataset, crop_setting):
    LABEL_FILE = join(cfg.BASE_DATA_DIR, dataset, crop_setting, 'images.txt')
    file = open(LABEL_FILE,'r')
    img_names = file.read().split('\n')
    file.close()
    
    out_file = os.path.join(cfg.BASE_DATA_DIR, dataset, crop_setting, 'images.npz')
    np.savez(out_file, imgname=img_names)

def extract_custom():
    LABEL_FILE = join(cfg.CUSTOM_ROOT, 'images.txt')
    file = open(LABEL_FILE,'r')
    img_names = file.read().split('\n')
    file.close()
    
    out_file = os.path.join(cfg.CUSTOM_ROOT, 'images.npz')
    np.savez(out_file, imgname=img_names)