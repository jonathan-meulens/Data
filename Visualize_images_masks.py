#Import Libraries
import os
from glob import glob
from dataset import MyOPSDataset
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


#Training images folder
TRAIN_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25\\'
#Mask images folder 
TRAIN_MASKDIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25_myops_gd\\'


#-------------------------If you want to see how the augmentations look like------------------------------------#
#Main function
def main():
    #Folder where the images lie
    DATA_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\'

    #Training images (NOTE: This FOLDER seperation is the same as the one we do below by accessing the 2 seperate folders)
    3##(Its the same because the other 2 folders, are in DATA_DIR)
    
    training_images = glob(os.path.join(DATA_DIR, 'train25'))

    #Mask images
    mask_images = glob(os.path.join(DATA_DIR, 'train25_myops_gd'))

    #Transform the the data in the dict in list format
    train_files = [{'images':images, 'masks':masks} for images,masks in zip(training_images, mask_images)]

    #Add the augmentations
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

    #Create the Dataset and loader objects
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

    #Grab the first augmented image
    test_patient = first(train_loader)
    test_image = test_patient['image']
    
    #Grab the first normal (unchanged) image
    original_patient = first(original_loader)
    original_image = original_patient['image']

    #-----------------------Visualize the Augmentations-----------------------------#
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

    #Show images
    plt.show()








