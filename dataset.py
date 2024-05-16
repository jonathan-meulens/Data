#Import libraries 
import torch
import os
from monai.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import torch.nn.functional as F
import SimpleITK as sitk
from Normalization import normalization
from bounding_box import get_ND_bounding_box, set_ND_volume_roi_with_bounding_box_range



#-----------------------------Augmentation Pipeline----------------------------------#
OUTPUT_IMG_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D\\'
OUTPUT_MSK_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_model1\\'

IMAGE_C0_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\C0_modality\\'
MASK_C0_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\C0_mask_modality\\'

IMAGE_DE_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\DE_modality\\'
MASK_DE_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\DE_mask_modality\\'

IMAGE_T2_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\T2_modality\\'
MASK_T2_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\T2_mask_modality\\'


class MyOPSDataset(Dataset):
    def __init__(self, image_dir_C0 = IMAGE_C0_MODALITY, image_dir_DE = IMAGE_DE_MODALITY, image_dir_T2 = IMAGE_T2_MODALITY,

                 mask_dir_C0 = MASK_C0_MODALITY, mask_dir_DE = MASK_DE_MODALITY,

                 mask_dir_T2 = MASK_T2_MODALITY, transform = None):

        self.image_dir_C0 = image_dir_C0
        self.image_dir_DE = image_dir_DE
        self.image_dir_T2 = image_dir_T2

        self.mask_dir_C0 = mask_dir_C0
        self.mask_dir_DE = mask_dir_DE
        self.mask_dir_T2 = mask_dir_T2

        self.images_C0 = os.listdir(image_dir_C0)
        self.images_DE = os.listdir(image_dir_DE)
        self.images_T2 = os.listdir(image_dir_T2)

        self.masks_CO= os.listdir(mask_dir_C0)
        self.masks_DE = os.listdir(mask_dir_DE)
        self.masks_T2 = os.listdir(mask_dir_T2)

        self.transform = transform

    def __len__(self):
        return len(self.images_C0)

    # THIS FUNCTION IS GOOD FOR MAKING IMAGES READILY ACCESIBLE
    def __getitem__(self, index):
        img_C0_path = os.path.join(self.image_dir_C0, self.images_C0[index])
        mask_C0_path = os.path.join(self.mask_dir_C0, self.images_C0[index])
        img_DE_path = os.path.join(self.image_dir_DE, self.images_DE[index])
        mask_DE_path = os.path.join(self.mask_dir_DE, self.images_DE[index])
        img_T2_path = os.path.join(self.image_dir_T2, self.images_T2[index])
        mask_T2_path = os.path.join(self.mask_dir_T2, self.images_T2[index])

        img_C0 = sitk.ReadImage(img_C0_path)
        msk_C0= sitk.ReadImage(mask_C0_path)
        image_array_C0 = sitk.GetArrayFromImage(img_C0)
        mask_array_C0 = sitk.GetArrayFromImage(msk_C0)

        img_DE = sitk.ReadImage(img_DE_path)
        msk_DE = sitk.ReadImage(mask_DE_path)
        image_array_DE = sitk.GetArrayFromImage(img_DE)
        mask_array_DE = sitk.GetArrayFromImage(msk_DE)

        img_T2 = sitk.ReadImage(img_T2_path)
        msk_T2 = sitk.ReadImage(mask_T2_path)
        image_array_T2 = sitk.GetArrayFromImage(img_T2)
        mask_array_T2 = sitk.GetArrayFromImage(msk_T2)






#--------------------Resizing tensors to have the correct dimensions---------------------------------#
        desired_size = (470, 470)

        image_tensor_C0 = torch.from_numpy(image_array_C0).unsqueeze(0).float()
        img_tensor_C0 = image_tensor_C0.unsqueeze(0)

        mask_tensor_C0 = torch.from_numpy(mask_array_C0).unsqueeze(0).float()
        msk_tensor_C0 = mask_tensor_C0.unsqueeze(0)

        image_tensor_DE = torch.from_numpy(image_array_DE).unsqueeze(0).float()
        img_tensor_DE = image_tensor_DE.unsqueeze(0)

        mask_tensor_DE = torch.from_numpy(mask_array_DE).unsqueeze(0).float()
        msk_tensor_DE = mask_tensor_DE.unsqueeze(0)

        image_tensor_T2 = torch.from_numpy(image_array_T2).unsqueeze(0).float()
        img_tensor_T2 = image_tensor_T2.unsqueeze(0)

        mask_tensor_T2 = torch.from_numpy(mask_array_T2).unsqueeze(0).float()
        msk_tensor_T2 = mask_tensor_T2.unsqueeze(0)


        img_resized_tensor_C0 = F.interpolate(img_tensor_C0, size=desired_size, mode='bilinear', align_corners=False)
        msk_resized_tensor_C0 = F.interpolate(msk_tensor_C0, size=desired_size, mode='bilinear', align_corners=False)

        img_resized_tensor_DE = F.interpolate(img_tensor_DE, size=desired_size, mode='bilinear', align_corners=False)
        msk_resized_tensor_DE = F.interpolate(msk_tensor_DE, size=desired_size, mode='bilinear', align_corners=False)

        img_resized_tensor_T2 = F.interpolate(img_tensor_T2, size=desired_size, mode='bilinear', align_corners=False)
        msk_resized_tensor_T2 = F.interpolate(msk_tensor_T2, size=desired_size, mode='bilinear', align_corners=False)

        msk_resized_tensor_C0_copy = msk_resized_tensor_C0.clone()
        img_resized_array_C0 = np.array(img_resized_tensor_C0).squeeze(0)

        msk_resized_array_C0_model_2 = np.array(msk_resized_tensor_C0_copy)
        msk_resized_array_C0 = np.array(msk_resized_tensor_C0).squeeze(0)

        img_resized_array_DE = np.array(img_resized_tensor_DE).squeeze(0)
        msk_resized_array_DE= np.array(msk_resized_tensor_DE).squeeze(0)

        img_resized_array_T2 = np.array(img_resized_tensor_T2).squeeze(0)
        msk_resized_array_T2 = np.array(msk_resized_tensor_T2).squeeze(0)

        #Mean , Std of pixel distribution for both image and mask
        mean_img = 277.6305
        std_img = 390.7119
        mean_msk = 0.0167
        std_mask = 0.1219

        #Apply normilization to the pixel distribution of the images
        image_C0 = normalization(img_resized_array_C0)
        image_DE = normalization(img_resized_array_DE)
        image_T2 = normalization(img_resized_array_T2)

        if self.transform is not None:
            valid_values = [200, 1220, 2221]
            msk_resized_array_C0[~np.isin(msk_resized_array_C0, valid_values)] = 0.0
            msk_resized_array_C0[(msk_resized_array_C0 == 200)] = 1.0
            msk_resized_array_C0[(msk_resized_array_C0 == 1220)] = 1.0
            msk_resized_array_C0[(msk_resized_array_C0 == 2221)] = 1.0

            #This section is simply to get the bounding box , variable mask_C0_loc is a throw_away
            mask_C0_loc_tensor = torch.from_numpy(msk_resized_array_C0).squeeze(0)
            msk_C0_loc_array = np.array(mask_C0_loc_tensor)

            image_C0_loc_tensor = torch.from_numpy(image_C0).squeeze(0)
            img_C0_loc_array = np.array(image_C0_loc_tensor)

            #Get the bounding box of the image
            bb_min_loc, bb_max_loc = get_ND_bounding_box(msk_C0_loc_array, margin = 100)

            #Set all areas outside of the area of interest to a pixel value of 0. 
            image_C0_loc = set_ND_volume_roi_with_bounding_box_range(img_C0_loc_array, bb_min_loc,bb_max_loc)
            msk_C0_loc = set_ND_volume_roi_with_bounding_box_range(msk_C0_loc_array, bb_min_loc, bb_max_loc)


            img_loc_C0_tensor = torch.from_numpy(image_C0_loc)

            msk_loc_C0_tensor = torch.from_numpy(msk_C0_loc)

            #Transform the image of by centering it. 
            image_mask_transform = T.CenterCrop(256)
            augmented_mask_tensor = image_mask_transform(msk_loc_C0_tensor)
            augmented_image_tensor = image_mask_transform(img_loc_C0_tensor)
            
            aug_mask_tensor = augmented_mask_tensor.unsqueeze(0)
            aug_msk_tensor = augmented_image_tensor.unsqueeze(0)

            img_loc_C0_array = np.array(aug_msk_tensor)
            msk_loc_C0_array = np.array(aug_mask_tensor)

            #Apply augmentations
            augmentations = self.transform(image=img_loc_C0_array, mask=msk_loc_C0_array)
            augmented_image = augmentations['image']
            augmented_mask = augmentations['mask']

            augmented_mask = np.copy(augmented_mask)
            augmented_image = np.copy(augmented_image)

            #Return images 
            return augmented_image, augmented_mask
















