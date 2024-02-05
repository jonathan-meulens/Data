# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:10:14 2023

@author: victo
"""
import os 
from PIL import Image
import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tf
from dataset import MyOPSDataset
import SimpleITK as sitk
import numpy as np
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from dataset import MyOPSDataset



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL1_CHEKCPOINTS = 'C:\\Users\\victo\\anaconda3\\envs\\JayM\\Jconda\\Data\\MyoSeg\\train_model1\\model1_checkpoints\\my_checkpoint.pt'
def save_checkpoint(checkpoint, folder= MODEL1_CHEKCPOINTS):
    print('=> Saving checkpoint')
    torch.save(checkpoint['state_dict'], folder, _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint_folder):
    print('=> Loading checkpoint')
    torch.load(checkpoint_folder)


def save_predictions_as_imgs(loader, model, folder, device=DEVICE):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        with torch.no_grad():
            preds = model(x)
            preds = F.interpolate(preds, size=y.shape[2:], mode='trilinear', align_corners=False)
            #print(x.shape, preds.shape)

        for i in range(preds.shape[0]):
            pred_per_bacth = preds[i]
            for j in range (pred_per_bacth.shape[0]):
                #print(pred.shape)
                pred_per_channel = pred_per_bacth[j]
                for p in range(pred_per_channel.shape[0]):
                    pred_per_layer = pred_per_channel[p]
                    pred_image = tf.to_pil_image(pred_per_layer)
                    pred_image.save(f'{folder}/pred{idx}_channel{j}.png')
    model.train()



