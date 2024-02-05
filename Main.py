# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:10:38 2023

@author: Jonathan M
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from model import UNET
from BiFPN_model import Myops_UNET
from Model_checkpoints import save_checkpoint, save_predictions_as_imgs, load_checkpoint
from dataset import MyOPSDataset
from Dice_score import val
from train import train_fn_1
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

#---------------------dataset.py-------------------------------------#
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
LEARNING_RATE = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
BATCH_SIZE = 1
NUM_EPOCHS = 25
NUM_WORKERS = 2
IMAGE_HEIGHT = 482
IMAGE_WIDTH = 478
DEPTH= 5
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25\\'
TRAIN_DIR_2 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\'
TRAIN_MASKDIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25_myops_gd\\'
TEST_DIR_1 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\test_model1\\'
TEST_DIR_2 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\test20\\'

# The different TRAINING-SET folds for five-fold cross validation: #
FOLD_1 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\train_fold_1\\'
FOLD_2 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\train_fold_2\\'
FOLD_3 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\train_fold_3\\'
FOLD_4 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\train_fold_4\\'
VAL_5 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\val_fold_5\\'

# The different MASK-SET folds for five-fold cross validation: #
MASK_FOLD_1 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_model1\\mask_fold_1\\'
MASK_FOLD_2 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_model1\\mask_fold_2\\'
MASK_FOLD_3 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_model1\\mask_fold_3\\'
MASK_FOLD_4 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_model1\\mask_fold_4\\'
MASK_VAL_5 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_model1\\mask_val_fold_5\\'

# The different MASK-SET folds for five-fold cross validation: #
TEST_FOLD_1 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\test_model1\\test_fold_1\\'
TEST_FOLD_2 = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\test_model1\\test_fold_1\\'

SAVED_PREDS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\predictions_model_1\\'
MODEL1_CHEKCPOINTS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\model1_checkpoints\\my_checkpoint.pt'

OUTPUT_IMG_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D\\'
OUTPUT_MSK_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_model1\\'
VALIDATION = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\validation\\'



def main():
    model_1 = UNET().to(DEVICE)
    model_2 = Myops_UNET()

    # optimizer for model1
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=LEARNING_RATE[3])

    # optimizer for model2
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=LEARNING_RATE[2])

    #Augementations Pipeline
    train_transform = A.Compose(
        [
          A.Rotate(limit=35, p=0.5),
          A.HorizontalFlip(p=0.5),
          #A.RandomBrightnessContrast(p=0.5),
          #A.ElasticTransform(p=0.9),
          #A. ShiftScaleRotate(p=0.5),
         # A. GridDistortion(p=0.8),
          A.VerticalFlip(p=0.5),

        ],
    )
    val_transform = A.Compose(
        [
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
            #A.ElasticTransform(p=0.7),
            # A.ShiftScaleRotate(p=0.5),
             #A.GridDistortion(p=0.8),
            # A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

        ])

    train_ds_model1 = MyOPSDataset(
        transform=train_transform)

    train_loader_model1 = DataLoader(
        train_ds_model1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,

    )


    val_ds_model1 = MyOPSDataset(
        transform=val_transform)

    val_loader_model1 = DataLoader(
        val_ds_model1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,

    )


    # schduler = torch.optim.lr_scheduler.LambdaLR()
    for epoch in range(NUM_EPOCHS):
        if LOAD_MODEL:
            load_checkpoint(MODEL1_CHEKCPOINTS)
        #train model
        train_fn_1(train_loader_model1, model_1, optimizer_1, epoch)

        #validation_round
        initial_dice, accuracy, initial_loss= val(val_loader_model1, model_1, epoch)
        print(f'Val_Dice:{initial_dice}, Val_Accuracy:{accuracy}, Val_Loss:{initial_loss}')

        # save model
        checkpoint = {
            'state_dict': model_1.state_dict(),
            'optimizer': optimizer_1.state_dict(),
        }
        save_checkpoint(checkpoint, MODEL1_CHEKCPOINTS)

        save_predictions_as_imgs(
        train_loader_model1, model_1, folder=SAVED_PREDS, device=DEVICE)


if __name__ == '__main__':
    main()