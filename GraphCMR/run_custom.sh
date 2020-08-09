# load path of pretrained model
LP=data/models/ours/2020_02_29-18_30_01.pt

# If true, saves predicted images
SAVE_IMGS=True

# Don't change this
DATASET=custom

# set the number of images on which you want to run
NUM_IMGS=

# If true and SAVE_IMGS=True, predictions are overlayed on white background.
# If false and SAVE_IMGS=True, predictions are overlayed on image.
WHITE_BACKGROUND=False

CMD="python eval.py --pretrained_checkpoint=${LP} 
--dataset=${DATASET} --num_imgs=${NUM_IMGS}
--write_imgs=${SAVE_IMGS} --white_background=${WHITE_BACKGROUND}"
echo $CMD
$CMD