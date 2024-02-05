# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:10:38 2023

@author: Jonathan M
"""

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dice_score import accuracy

#---------------------dataset.py-------------------------------------#
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
LEARNING_RATE = 1e-4
BATCH_SIZE = 5
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 482
IMAGE_WIDTH = 478
DEPTH= 5
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train25\\'
TRAIN_DIR_2 = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\'
TRAIN_MASKDIR = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train25_myops_gd\\'
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



#----------------------Training Function for Binary Model----------------------------#

def dice_score(input, prediction):
    smooth = 1e-6
    prediction = (prediction>=0.5).float() * 1

    intersection = (input * prediction).sum()
    dice = (2. * intersection + smooth) / (input.sum() + prediction.sum() + smooth)
    return dice

def train_fn_1(loader, model, optimizer, x_axis, scale_factors = [0.1 , 0.01, 0.001, 0.0001]):  # WHEN YOU HAVE ACCESS TO GPU, PUT SCALER
    loss_fn_1 = torch.nn.BCELoss()
    writer = SummaryWriter('runs/model_1/writer_70')
    initial_loss = 0
    initial_dice = 0
    step = 0


    loop = tqdm(loader)
    for batch_idx, batch in enumerate(loop):
        data, target = batch
        data = data.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        optimizer.zero_grad()

        # forward PASS
        # with torch.cuda.amp.autocast():
        predictions = model(data)
        predictions = torch.sigmoid(predictions)
        target = torch.sigmoid(target)

        loss = loss_fn_1(predictions, target)
        initial_loss += loss.item()

        dice = dice_score(target, predictions)
        initial_dice += dice
        acc = accuracy(target, predictions)

        #Scale the loss if needed, to prevent "EXPLODING GRADIENT"
        scaled_loss = torch.mul(loss, scale_factors[0])

        # backward PASS
        loss.backward()

        #Scale all the parameters that change with gradient, to prevent "EXPLODING GRADIENT"
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul((1.0 / scale_factors[0]))

        optimizer.step()

        # Update  tqdm loop
        step += data.shape[0]
        loop.set_postfix(loss=initial_loss, dice_score = dice)
        print('epoch:{}, average_dice:{}'.format(x_axis, initial_dice / step))
        writer.add_scalar('Accuracy', acc, x_axis)
        writer.add_scalar('Training loss', loss, x_axis)
        writer.add_scalar('Dice score', dice, x_axis)
        writer.flush()



