import os
import shutil
import sys
from collections import OrderedDict
import numpy as np
from glob import glob
import SimpleITK as sitk


def get_ND_bounding_box(volume, margin = None):
    """
    get the bounding box of nonzero region in an ND volume
    """
    input_shape = volume.shape
    if(margin is None):
        margin = [0] * len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(volume)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max() + 1)

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i])
    return idx_min, idx_max # REMEMBER TO STORE THIS VALUE IN OUTPUT_VARAIBLE


def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

def crop_align():
    OUTPUT_IMG_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D\\'
    OUTPUT_MSK_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_model1\\'

    training_images = os.listdir(OUTPUT_IMG_DIR)
    mask_images = os.listdir(OUTPUT_MSK_DIR)

    image_paths = []
    mask_paths = []
    for i in range(len(training_images)):
        img_path = os.path.join(OUTPUT_IMG_DIR, training_images[i])
        image_paths.append(img_path)

        mask_path = os.path.join(OUTPUT_MSK_DIR, mask_images[i])
        mask_paths.append(mask_path)

    for image, mask  in zip (training_images, mask_images):
        image = sitk.ReadImage(image)
        mask = sitk.ReadImage(mask)
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array[(mask_array == 500) | (mask_array == 600)] = 0.0
        mask_array[(mask_array >= 200) | (mask_array == 1220) | (mask_array == 2221)] = 1.0



DATA_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25_myops_gd - Copy\\'
masks = os.listdir('C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train25_myops_gd - Copy\\')
path = glob(os.path.join(DATA_DIR, 'train25_myops_gd'))

for i in range(len(masks)):
    mask_images = os.path.join(DATA_DIR, masks[i])
    image = sitk.ReadImage(mask_images)
    mask_array = sitk.GetArrayFromImage(image)
    print(mask_array.shape)
