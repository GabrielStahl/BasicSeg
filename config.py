# import the necessary packages
import torch
import os

# base path of the dataset
base_path = "/Users/Gabriel/MResMedicalImaging/RESEARCH_PROJECT/BasicSeg/"

DATASET_PATH = os.path.join(base_path,"dataset","oxford-iiit-pet")

# define the path to the images and masks dataset
#IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images_mini")

MASK_DATASET_PATH = os.path.join(DATASET_PATH, "annotations","trimaps")
# define the path to the checkpoint
MODEL_CHECKPOINT_PATH = "model_weights"

# define the validation percentage
VAL_PERCENT = 0.2
# batch size for training
BATCH_SIZE = 40
# learning rate for the optimizer
LEARNING_RATE = 1e-4
# momentum for the optimizer
MOMENTUM = 0.999
# gradient clipping value (for stability while training)
GRADIENT_CLIPPING = 1.0
# weight decay (L2 regularization) for the optimizer
WEIGHT_DECAY = 1e-8
# number of epochs for training
EPOCHS = 2

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128 # CHANGE
INPUT_IMAGE_HEIGHT = 128

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(base_path, BASE_OUTPUT, "unet_catsAndDogs.pth")
PLOT_PATH = os.path.join(base_path, BASE_OUTPUT, "plot.png")
TEST_PATHS = os.path.join(base_path, BASE_OUTPUT, "test_paths.txt")
