import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from model import UNET
from Model_checkpoints import save_checkpoint, save_predictions_as_imgs, load_checkpoint
from dataset import MyOPSDataset
from Dice_score import dice_loss, accuracy


def dice_score(input, prediction):
    smooth = 1e-6
    prediction = (prediction>=0.5).float() * 1

    intersection = (input * prediction).sum()
    dice = (2. * intersection + smooth) / (input.sum() + prediction.sum() + smooth)
    return dice

def train_fn_dense(loader, model, optimizer, x_axis):  # WHEN YOU HAVE ACCESS TO GPU, PUT SCALER
    loss_fn_1 = torch.nn.BCELoss()
    writer = SummaryWriter('runs/model_dense/writer_2')
    initial_loss = 0
    initial_dice = 0
    step = 0
    max_loss = 1

    loop = tqdm(loader)
    for batch_idx, batch in enumerate(loop):
        data, target = batch
        data = data.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        #new_target = target.expand(data.shape[0], (data.shape[1]), data.shape[2], data.shape[3])
        optimizer.zero_grad()
        # forward
        # with torch.cuda.amp.autocast():
        predictions = model(data)
        predictions = torch.sigmoid(predictions)
        #new_target = target.expand(predictions.shape[0], (predictions.shape[1]), predictions.shape[2], predictions.shape[3])
        target = torch.sigmoid(target)

        loss = loss_fn_1(predictions, target)
        initial_loss += loss.item()

        dice = dice_score(target, predictions)
        initial_dice += dice
        acc = accuracy(target, predictions)
        # print(epoch, dice_coe)
        # backward
        #scaled_loss = torch.mul(loss, scale_factors[0])
        loss.backward()

        #for param in model.parameters():
         #   if param.grad is not None:
          #      param.grad.mul((1.0 / scale_factors[0]))

        optimizer.step()
        # Update  tqdm loop
        step += data.shape[0]
        loop.set_postfix(loss=initial_loss, dice_score = dice)
        print('epoch:{}, average_dice:{}'.format(x_axis, initial_dice / step))
        writer.add_scalar('Accuracy', acc, x_axis)
        writer.add_scalar('Training loss', loss, x_axis)
        writer.add_scalar('Dice score', dice, x_axis)
        writer.flush()