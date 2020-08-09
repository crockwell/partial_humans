# load path of pretrained model
LP=models/ours/model.ckpt-694216

# If on a multi-GPU machine, can set os.env['']
GPU=5

# Directory in which data is located
CUSTOM_DIR=

# Number of images to evaluate on
NUM_IMGS=

# If true, saves predicted images
SAVE_IMGS=True

# If true and SAVE_IMGS=True, predictions are overlayed on white background.
# If false and SAVE_IMGS=True, predictions are overlayed on image.
WHITE_BACKGROUND=False

CMD="python -m src.main
--gpu ${GPU} --load_path=${LP} --num_imgs=${NUM_IMGS}
--custom_dir=${CUSTOM_DIR} --write_imgs=${SAVE_IMGS} --white_background=${WHITE_BACKGROUND}"

echo $CMD
$CMD