#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
from datasets.preprocess.datasets import extract

if __name__ == '__main__':   
    for dataset in ['vlog', 'cross_task', 'instructions', 'youcook']:
        for crop_setting in ['all', 'uncropped_keypoint', 'cropped_keypoint']:
            extract(dataset, crop_setting)