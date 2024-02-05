import SimpleITK as sitk
import random
import os
import numpy as np

OUTPUT_IMG_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D\\'
OUTPUT_MSK_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_model1\\'
OUTPUT_OPT_IMG_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\optimized_images_model1\\'
VALIDATION = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\validation\\'

# Load the image and mask using ITK-SNAP's Python interface
image_path = os.listdir(OUTPUT_IMG_DIR)
mask_path = os.listdir(OUTPUT_MSK_DIR)


def image_optimization(mask_paths):
    t = 0
    for i in range(len(image_path)):
        img_path = os.path.join(OUTPUT_IMG_DIR, image_path[i])
        msk_path = os.path.join(OUTPUT_MSK_DIR, mask_paths[i])

        image = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(msk_path)

        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        optimized_array = image_array * mask_array  # Element-wise multiplication

        optimized_image = sitk.GetImageFromArray(optimized_array)

        output_filename = f'myops_optimized_image_{t + 1}_.nii.gz'
        output_path = os.path.join(OUTPUT_OPT_IMG_DIR, output_filename)

        sitk.WriteImage(optimized_image, output_path)

        t += 1


mask_paths = [mask_path[i] for i in range(len(mask_path))]  # Use mask_path instead of mask_paths

image_optimization(mask_paths)

