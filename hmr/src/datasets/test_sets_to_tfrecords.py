"""
Convert ALL Vlog images to TFRecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import makedirs
from os.path import join, exists
from time import time

import numpy as np
import tensorflow as tf
import cPickle as pkl
from PIL import Image

from common import ImageCoder
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
BASE_DIR = '/x/cnris/github/data/'
CATEGORIES = ['vlog','cross_task', 'youcook', 'instructions']
TYPES = ['all', 'uncropped_keypoint', 'cropped_keypoint']
NUM_SHARDS = 500

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert_data(image_data, image_path, height, width, total_index):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      image_path: string, path to this image file
      height, width: integers, image shapes in pixels.
      total_index: int, number in the dataset
    Returns:
      Example proto
    """
    from os.path import basename

    image_format = 'JPEG'

    feat_dict = {
        'image/filename': bytes_feature(
            tf.compat.as_bytes(image_path)),
        'image/encoded': bytes_feature(tf.compat.as_bytes(image_data)),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/ll_idx': int64_feature(total_index),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feat_dict))

    return example

def add_to_tfrecord(writer, total_index, labels, cat_dir, coder):
    """
    Add each "single person" in this image.
    """
    
    # Add each img to tf record
    path = labels[total_index]
    image_path = join(cat_dir, 'images', path)
        
    with open(image_path, 'rb') as f:
        im = Image.open(f)
        im = im.convert("RGB")
        image_data = coder.encode_jpeg(np.array(im))
        
    height, width, __ = np.shape(im)

    example = convert_data(image_data, path, height, width, total_index)
    writer.write(example.SerializeToString())

def process(out_dir, labels, cat_dir, num_shards, coder):    
    total_index = 0
    i = 0   
    fidx = 0
    num_imgs = len(labels)
    while i < num_imgs:
        out_path = join(out_dir, '%03d.tfrecord')
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            # Count on total ppl in each shard
            num_ppl = 0
            while i < num_imgs and num_ppl < num_shards:
                if i % 100 == 0:
                    print('Reading img %d/%d' % (i, num_imgs))
                add_to_tfrecord(writer, total_index, labels, cat_dir, coder)
                i += 1
                num_ppl += 1
                total_index += 1

        fidx += 1   

def main(unused_argv):
    coder = ImageCoder()
    
    for cat in CATEGORIES:
        for t in TYPES:
            cat_dir = join(BASE_DIR, cat, t)
            out_dir = join(cat_dir,'tf_records/')
            label_file = join(cat_dir,'images.txt')
            file = open(label_file,'r')
            labels = file.read().split('\n')
            file.close()

            print('Saving results to %s' % out_dir)

            if not exists(out_dir):
                makedirs(out_dir)

            process(out_dir, labels, cat_dir, NUM_SHARDS, coder)


if __name__ == '__main__':
    tf.app.run()