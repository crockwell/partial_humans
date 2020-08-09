# load path of pretrained model
LP=models/ours/model.ckpt-694216

# If on a multi-GPU machine, can set os.env['']
GPU=4

# Directory in which data is located
BASE_DIR=data/

# If true, saves predicted images
SAVE_IMGS=True

# If true and SAVE_IMGS=True, predictions are overlayed on white background.
# If false and SAVE_IMGS=True, predictions are overlayed on image.
WHITE_BACKGROUND=False

# Set to true if want to run PCK evaluation. 
# Only valid for CROP_SETTING "uncropped_keypoint" and "cropped_keypoint"
EVAL_PCK=True

# Dataset on which to run: vlog, cross_task, instructions, or youcook
DATASET=vlog

# selection of images on which to run: all, uncropped_keypoint, or cropped_keypoint
CROP_SETTING=all

#if evaluating keypoints, file name to save results
LOG_DIR=hmr_pck_results_ours.txt

CMD="python -m src.main
--gpu ${GPU} --load_path=${LP}
--base_dir=${BASE_DIR} --dataset=${DATASET} --crop_setting=${CROP_SETTING}
--write_imgs=${SAVE_IMGS} --white_background=${WHITE_BACKGROUND}
--eval_pck=${EVAL_PCK} --logdir=${LOG_DIR}"

echo $CMD
$CMD