# load path of pretrained model
LP=data/models/ours/2020_02_29-18_30_01.pt

# If true, saves predicted images
SAVE_IMGS=False

# If true and SAVE_IMGS=True, predictions are overlayed on white background.
# If false and SAVE_IMGS=True, predictions are overlayed on image.
WHITE_BACKGROUND=True

# Set to true if want to run PCK evaluation. 
# Only valid for CROP_SETTING "uncropped_keypoint" and "cropped_keypoint"
EVAL_PCK=True

for DATASET in vlog cross_task instructions youcook
do
    for CROP_SETTING in uncropped_keypoint cropped_keypoint
    do
        CMD="python eval.py --pretrained_checkpoint=${LP}
        --eval_pck=${EVAL_PCK} --crop_setting=${CROP_SETTING} --dataset=${DATASET}
        --write_imgs=${SAVE_IMGS} --white_background=${WHITE_BACKGROUND}"
        echo $CMD
        $CMD
    done
done