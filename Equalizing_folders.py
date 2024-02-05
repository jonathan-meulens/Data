import torch
import os
from glob import glob
from model import UNET
from tqdm import tqdm
from dataset import MyOPSDataset
from torch.utils.data import Dataset
from monai.data import DataLoader
from monai.utils import first
import matplotlib.pyplot as plt
from monai.transforms import (
Compose,
LoadImaged,
ToTensord,
AddChanneld,
Orientationd,
Spacingd,
RandRotated,
ScaleIntensityRanged,
RandGaussianNoised,
RandAffined,
Flipd,
CropForegroundd,
)

LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 482
IMAGE_WIDTH = 478
DEPTH= 5
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25\\'
TRAIN_DIR_2 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\'
TRAIN_MASKDIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25_myops_gd\\'
TEST_DIR_1 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\test_model1\\'
TEST_DIR_2 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\test20\\'

# The different TRAINING-SET folds for five-fold cross validation: #
FOLD_1 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\train_fold_1\\'
FOLD_2 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\train_fold_2\\'
FOLD_3 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\train_fold_3\\'
FOLD_4 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\train_fold_4\\'
VAL_5 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\val_fold_5\\'

# The different MASK-SET folds for five-fold cross validation: #
MASK_FOLD_1 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\mask_model1\\mask_fold_1\\'
MASK_FOLD_2 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\mask_model1\\mask_fold_2\\'
MASK_FOLD_3 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\mask_model1\\mask_fold_3\\'
MASK_FOLD_4 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\mask_model1\\mask_fold_4\\'
MASK_VAL_5 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\mask_model1\\mask_val_fold_5\\'

# The different MASK-SET folds for five-fold cross validation: #
TEST_FOLD_1 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\test_model1\\test_fold_1\\'
TEST_FOLD_2 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\test_model1\\test_fold_1\\'

SAVED_PREDS = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\predictions_model_1\\'
MODEL1_CHEKCPOINTS = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\model1_checkpoints\\my_checkpoint.pt'

IMAGES = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\validation\\'
MASKS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_model1\\'
#-------------------------LOADERS------------------------------------#


folder_1 = MASKS
folder_2 = IMAGES

# Get the filenames in folder_1 and folder_2
filenames_1 = os.listdir(folder_1)
filenames_2 = os.listdir(folder_2)

# Sort the filenames to ensure consistent ordering
filenames_1.sort()
filenames_2.sort()

# Iterate over the filenames and rename the files in folder_1
for filename_1, filename_2 in zip(filenames_1, filenames_2):
    file_path_1 = os.path.join(folder_1, filename_1)
    file_path_2 = os.path.join(folder_2, filename_2)
    if file_path_2 != file_path_1:
        os.rename(filename_2, filename_1)
print("Filenames in folder_1 renamed to match folder_2.")




