CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

DATA_ROOT = "/data/ephemeral/home/dataset/"
TRAIN_ANN = "new_train_split_15.json"
VAL_ANN   = "new_val_split_15.json"
TEST_ANN  = "test.json"

MODEL_CONFIG_PATH = "/data/ephemeral/home/code/cascade_rcnn/configs/custom_swin_large.py"

WORK_DIR = "./work_dirs/cascade_rcnn_swin_5_trash"
SEED = 2025