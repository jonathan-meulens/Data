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
import torch.nn.functional as F
from keras.utils import to_categorical
import SimpleITK as sitk
import numpy as np
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

MODEL2_MASKS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\C0_mask_modality\\'
model2_masks = os.listdir(MODEL2_MASKS)
#-------------------------LOADERS------------------------------------#







