"""
Data loader with data augmentation.
Only used for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob

import tensorflow as tf

from .tf_smpl.batch_lbs import batch_rodrigues
from .util import data_utils
from cv2 import warpAffine
import numpy as np

class DataLoader(object):
    def __init__(self, config):
        self.config = config
        if config.dataset is None:
            self.dataset_dir = join(config.custom_dir, 'tf_records')
        else:
            self.dataset_dir = join(config.base_dir, config.dataset, config.crop_setting, 'tf_records')
        self.batch_size = config.batch_size
        self.data_format = config.data_format
        self.output_size = config.img_size

        self.image_normalizing_fn = data_utils.rescale_image

    def load(self):
        return self.get_loader()

    def get_loader(self):
        """
        Outputs:
          image_batch: batched images as per data_format
          label_batch: now batched smpl labels N x 85
        """
        files = data_utils.get_all_files(self.dataset_dir)

        fqueue = tf.train.string_input_producer(
            files, shuffle=False, name="input")
        image, display_image, idx, filename, size = self.read_data(fqueue)
        min_after_dequeue = 5000
        num_threads = 8
        capacity = min_after_dequeue + 3 * self.batch_size

        pack_these = [image, idx, filename, display_image, size]
        pack_name = ['image', 'idx', 'filename', 'display_image', 'size']

        all_batched = tf.train.batch(
            pack_these,
            batch_size=self.batch_size,
            num_threads=num_threads,
            capacity=capacity,
            enqueue_many=False,
            name='input_batch_train')
        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict
    
    def read_data(self, filename_queue):
        with tf.name_scope(None, 'read_data', [filename_queue]):
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)
            
            image, image_size, idx, filename = data_utils.parse_data(
                example_serialized) 
            image, display_image = self.image_preprocessing(
                image, image_size)

            return image, display_image, idx, filename, image_size
    
    def image_preprocessing(self,
                            image,
                            image_size,
                            pose=None,
                            gt3d=None):
        with tf.name_scope(None, 'image_preprocessing',
                           [image, image_size]):
            '''
            At inference, we adopt preprocessing from hmr's demo.py.
            tf.image.resize_images does not match demo (cv2), so we use
            the version from tf2, image.resize
            ''' 

            center = tf.cast(image_size / 2, tf.int32)
            scale = 224. / tf.cast(tf.reduce_max(image_size), tf.float64)
            new_size = tf.cast(tf.floor(tf.cast(image_size, tf.float64)*scale), tf.int32)
            #image_pad = tf.image.resize_images(image, new_size)
            image_pad = tf.compat.v2.image.resize(image, new_size)
            scale_factors = [tf.cast(new_size[0], tf.float32) / tf.cast(image_size[0], tf.float32), tf.cast(new_size[1], tf.float32) / tf.cast(image_size[1], tf.float32)]
            center_new = tf.cast(tf.round(tf.cast(center, tf.float32) * scale_factors), tf.int32)
            margin = tf.to_int32(self.output_size / 2)
            
            # Crop image pad.
            display_image = data_utils.pad_image_edge(image_pad, margin, display=True) 
            image_pad = data_utils.pad_image_edge(image_pad, margin)
            center_pad = center_new + margin
            start_pt = center_pad - margin
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[0], start_pt[1], 0])
            bbox_size = tf.stack([self.output_size, self.output_size, 3])

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            display_image_crop = tf.slice(display_image, bbox_begin, bbox_size)

            # rescale image from [0, 1] to [-1, 1]
            crop = self.image_normalizing_fn(crop)
            display_image_crop = self.image_normalizing_fn(display_image_crop)
            
            return crop, display_image_crop
    
    