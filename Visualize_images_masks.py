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

#-------------------------LOADERS------------------------------------#
def main():
    DATA_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\'

    training_images = glob(os.path.join(DATA_DIR, 'train25'))
    mask_images = glob(os.path.join(DATA_DIR, 'train25_myops_gd'))

    train_files = [{'images':images, 'masks':masks} for images,masks in zip(training_images, mask_images)]

    original_transform = Compose(
        [
            LoadImaged(keys=["image", "label"])
        ])

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
            RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
            RandGaussianNoised(keys='image', prob=0.5),
        ])
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
        ])

    train_ds = MyOPSDataset(image_dir=TRAIN_DIR,
                            mask_dir=TRAIN_MASKDIR,
                            transform= val_transform
                            )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        timeout=30)

    original_ds = MyOPSDataset(image_dir=TRAIN_DIR,
                            mask_dir=TRAIN_MASKDIR,
                            transform= original_transform
                            )

    original_loader = DataLoader(
        original_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        timeout=20)

    test_patient = first(train_loader)
    test_image = test_patient['image']
    #print(test_image.shape)

    original_patient = first(original_loader)
    original_image = original_patient['image']
    #print(original_image.shape)

    plt.figure('test', (12,6))

    #Original Image
    plt.subplot(1,3,1)
    plt.title('orig patient')
    plt.imshow(original_patient['image'][0, 0, 0, : , :], cmap='gray')

    #Augmented Image
    plt.subplot(1,3,2)
    plt.title('test patient')
    plt.imshow(test_patient['image'][0, 0, 0, : , :], cmap='gray')

    #Augmented Mask
    plt.subplot(1,3,3)
    plt.title('mask patient')
    plt.imshow(test_patient['label'][0, 0, 0, : , :], cmap='gray')

    plt.show()

#if __name__ == '__main__':
#    main()






