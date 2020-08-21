### Requirements
- Python 2.7
- [Pytorch](https://pytorch.org/) (tested on version 1.0)

### Installation

#### Linux setup with conda
```
conda create --name cmr python=2.7
conda activate cmr
pip install numpy==1.15.4
pip install torch==1.0.0 torchvision==0.2.1
pip install -r requirements.txt
```
- While installing requirements.txt, we sometimes noticed an error message print "failed building wheel for opendr", but later in the same installation, opendr was able to install properly. 

#### Download required models
1. Download SMPL model. 
```
wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl --directory-prefix=data
```
*This model falls under the SMPLify license. If you find this model useful for your research, please follow the [corresponding citing instructions](https://github.com/classner/up/blob/master/3dfit/README.md)*
2. Download our pretrained models
```
cd data && wget http://fouheylab.eecs.umich.edu:/~cnris/partial_humans/cmr/models.tar.gz && tar -xf models.tar.gz
```

### Running the Demo
```
python demo.py --checkpoint=data/models/ours/2020_02_29-18_30_01.pt --img demo/instructions_coffee_0004_00001634.jpg
```
- For a small number of images, it might be easier to adapt the demo code for running. For a large number, our script below should be much faster. In either case, it can take some time for the model to get setup before running.

### Running with Custom Images
- Place custom images in `CUSTOM_ROOT/images`, and image paths in `CUSTOM_ROOT/images.txt`. In `config.py`, set `CUSTOM_ROOT`.
- Run `preprocess_custom.py`

To run, use `run_custom.sh`, changing `NUM_IMGS` to number of images on which you want to run model. To avoid confusing off-by-one bugs, make sure images.txt does not have a blank line at the end.

### Evaluation
This requires downloading the annotated test set, detailed in the parent [README](https://github.com/crockwell/partial_humans/blob/master/README.md).

- In `config.py`, set `BASE_DATA_DIR` to the parent directory in which you downloaded the datasets. 
- Run `preprocess_test_datasets.py`

To run, use `test.sh` for single dataset, `test_all.sh` for multiple datasets. Within these files, make the following change:

- Change LP variable to your extracted model path

For comparison, you may want to download the original CMR model of Kolotouros et al. We compare to the model trained on extra 2d in-the-wild annotation `model_checkpoint_h36m_up3d_extra2d.pt`.
```
cd data/models && wget https://seas.upenn.edu/~nkolot/data/cmr/models.tar &&
mkdir tmp && tar -xvf models.tar -C tmp && rm models.tar && mv tmp/models kolotouros && rmdir tmp
```

### Other Notes
- We observed predictions can be slightly sensitive to image compression and preprocessing (e.g. saving input image as jpg vs. png). Keypoint accuracies may differ by ~1% if using a different method than available here, so we recommend consistency across future method ablations.

### Base Model
This repo builds from the project page for [Convolutional Mesh Regression for Single-Image Human Shape Reconstruction](https://github.com/nkolot/GraphCMR). You may find this useful for additional training and for reference. We attempt to keep our code similar for convenience; we remove some files that are not used or supported by our code. We leave some in such as in the `docker` folder, as they may be useful (even though we do not explicitly support them).
If you find this code useful for your research, please consider citing the following paper:

	@Inproceedings{kolotouros2019cmr,
	  Title          = {Convolutional Mesh Regression for Single-Image Human Shape Reconstruction},
	  Author         = {Kolotouros, Nikos and Pavlakos, Georgios and Daniilidis, Kostas},
	  Booktitle      = {CVPR},
	  Year           = {2019}
	}
