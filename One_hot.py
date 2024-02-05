import torch.nn as nn
import torch
from torch.autograd import Variable
import os
from PIL import Image
from monai.data import Dataset, DataLoader
from monai.utils import first
import tarfile
import torchvision.transforms.functional as tf
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image
import albumentations as A



IMAGE_C0_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\C0_modality\\'
MASK_C0_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\C0_mask_modality\\'


IMAGE_DE_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\DE_modality\\'
MASK_DE_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\DE_mask_modality\\'

IMAGE_T2_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\T2_modality\\'
MASK_T2_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\T2_mask_modality\\'

image_C0_modality = os.listdir(IMAGE_C0_MODALITY)
image_DE_modality = os.listdir(IMAGE_DE_MODALITY)
image_T2_modality = os.listdir(IMAGE_T2_MODALITY)

mask_C0_modality = os.listdir(MASK_C0_MODALITY)
mask_DE_modality = os.listdir(MASK_DE_MODALITY)
mask_T2_modality = os.listdir(MASK_T2_MODALITY)

# valid_values = [200, 1220, 2221]
# for i in range(len(mask_C0_modality)):
#     mask_C0_path = os.path.join(MASK_C0_MODALITY, mask_C0_modality[i])
#     mask_C0 = sitk.ReadImage(mask_C0_path)
#     mask_C0_array = sitk.GetArrayFromImage(mask_C0)
#
#     mask_C0_array[~np.isin(mask_C0_array, valid_values)] = 0.0
#     mask_C0_array[(mask_C0_array == 200)] = 1.0
#     mask_C0_array[(mask_C0_array == 1220)] = 2.0
#     mask_C0_array[(mask_C0_array == 2221)] = 3.0
#
#     mask_C0_tensor = torch.from_numpy(mask_C0_array).long()
#     one_hot_mask = F.one_hot(mask_C0_tensor)
#     print(f'filename: {mask_C0_path} , One_hot_mask_tensor: {one_hot_mask.shape}')

# valid_values = [200, 1220, 2221]
# for i in range(len(mask_C0_modality)):
#     mask_DE_path = os.path.join(MASK_DE_MODALITY, mask_DE_modality[i])
#     mask_C0 = sitk.ReadImage(mask_DE_path)
#     mask_C0_array = sitk.GetArrayFromImage(mask_C0)
#
#     mask_C0_array[~np.isin(mask_C0_array, valid_values)] = 0.0
#     mask_C0_array[(mask_C0_array == 200)] = 1.0
#     mask_C0_array[(mask_C0_array == 1220)] = 2.0
#     mask_C0_array[(mask_C0_array == 2221)] = 3.0
#
#     mask_C0_tensor = torch.from_numpy(mask_C0_array).long()
#     one_hot_mask = F.one_hot(mask_C0_tensor)
#     print(f'filename: {mask_DE_path} , One_hot_mask_tensor: {one_hot_mask.shape}')

# valid_values = [200, 1220, 2221]
# for i in range(len(mask_C0_modality)):
#     mask_T2_path = os.path.join(MASK_T2_MODALITY, mask_T2_modality[i])
#     mask_C0 = sitk.ReadImage(mask_T2_path)
#     mask_C0_array = sitk.GetArrayFromImage(mask_C0)
#
#     mask_C0_array[~np.isin(mask_C0_array, valid_values)] = 0.0
#     mask_C0_array[(mask_C0_array == 200)] = 1.0
#     mask_C0_array[(mask_C0_array == 1220)] = 2.0
#     mask_C0_array[(mask_C0_array == 2221)] = 3.0
#
#     mask_C0_tensor = torch.from_numpy(mask_C0_array).long()
#     one_hot_mask = F.one_hot(mask_C0_tensor)
#     print(i, f'filename: {mask_T2_path} , One_hot_mask_tensor: {one_hot_mask.shape}')