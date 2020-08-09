#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
from datasets.preprocess.datasets import extract_custom

if __name__ == '__main__':   
    extract_custom()