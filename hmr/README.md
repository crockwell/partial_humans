### Requirements
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.14

### Installation

#### Linux setup with conda
```
conda create --name hmr python=2.7
conda activate hmr
pip install numpy
pip install -r requirements.txt
```
#### Install TensorFlow
With GPU:
```
pip install tensorflow-gpu==1.14.0
```
Without GPU:
```
pip install tensorflow==1.14.0
```

#### Download required models
1. Download our pretrained models, extract into `models`
```
wget http://megaflop.eecs.umich.edu:8888/~cnris/partial_humans_website/crockwell.github.io/hmr/models.tar.gz && tar -xf models.tar.gz
```
2. In the models folder, download mean SMPL parameters used for initialization (we use same as original work). These are compressed along with the original HMR model, which is therefore available for ablation. 
```
cd models
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/neutral_smpl_mean_params.h5
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz && mv models/neutral_smpl_with_cocoplus_reg.pkl . && mv models kanazawa
```
*The SMPL model falls under the SMPLify license. If you find the model useful for your research, please follow the [corresponding citing instructions](https://github.com/classner/up/blob/master/3dfit/README.md)*

### Running the Demo
```
python -m demo --img_path demo/vlog_q_u_Q_v_qNSfZz0HquQ_017_frame000151.jpg
```
- For a small number of images, it might be easier to adapt the demo code for running. For a large number,
we have built a script to make tf_records (below)

### Running with Custom Images
- Place custom images in `CUSTOM_DIR/images`, and image paths in `CUSTOM_DIR/images.txt`
- Use `make_custom_dataset.sh` to make tf_records. Update `CUSTOM_DIR` in script.
- Run `run_custom.sh`. Update `CUSTOM_DIR` and `NUM_IMGS` in script. 

### Evaluation
This requires downloading the annotated test set, detailed in the parent [README](https://github.com/crockwell/partial_humans/blob/master/README.md).
Use `test.sh` for single dataset, `test_all.sh` for multiple datasets. Within these files, make the following changes:
- Change `LP` variable to your extracted model path
- Change `BASE_DIR` to the folder in which your data exists

For comparison, you have already download the original HMR model of Kanazawa et al. We also provide the `MPII_CROP` model from the paper ablations.

### Other Notes
- We observed predictions can be slightly sensitive to image compression (e.g. saving input image as jpg vs. png, different ways of loading images in tf). Keypoint accuracies may differ by ~1% if using a different method than available here, so we recommend consistency across future method ablations.

### Base Model
This repo builds from the project page for [End-to-end Recovery of Human Shape and Pose](https://github.com/akanazawa/hmr). You may find this useful for additional training and for reference. We attempt to keep our code similar for convenience; we remove some files that are not used or supported by our code. `src/inference.py` is most similar to `src/train.py` in the original work; `demo.py` is similar to the original `demo.py`. 

If you use this code for your research, please also cite the original work.
```
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}
```


