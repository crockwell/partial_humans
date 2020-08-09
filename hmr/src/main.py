""" Driver for train """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .config import get_config
from .data_loader import DataLoader
from .inference import HMRInference
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main(config):  
    # Load data on CPU
    with tf.device("/cpu:0"):
        data_loader = DataLoader(config)
        image_loader = data_loader.load() 

    runner = HMRInference(config, image_loader)
    
    runner.run()


if __name__ == '__main__':
    config = get_config()
    main(config)
