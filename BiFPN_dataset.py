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

#-------------------------------------Image-Files----------------------------------#
OUTPUT_IMG_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D\\'
OUTPUT_MSK_DIR = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_model1\\'

##Image directory for the BssfP modality
IMAGE_C0_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\C0_modality\\'
MASK_C0_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\C0_mask_modality\\'

##Image directory for the DE modality
IMAGE_DE_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\DE_modality\\'
MASK_DE_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\DE_mask_modality\\'

##Image directory for the T2 modality
IMAGE_T2_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\training_2D_all_modalities\\T2_modality\\'
MASK_T2_MODALITY = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\mask_2D_all_modalities\\T2_mask_modality\\'

#This is dataset object for that is fed to the BiFPN-model.
class BiFPNDataset(Dataset):
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
        img_C02_path = os.path.join(self.image_dir_C0, self.images_C0[index])
        mask_C02_path = os.path.join(self.mask_dir_C0, self.images_C0[index])
        img_DE2_path = os.path.join(self.image_dir_DE, self.images_DE[index])
        mask_DE2_path = os.path.join(self.mask_dir_DE, self.images_DE[index])
        img_T2_2_path = os.path.join(self.image_dir_T2, self.images_T2[index])
        mask_T2_2_path = os.path.join(self.mask_dir_T2, self.images_T2[index])

        img_C02 = sitk.ReadImage(img_C02_path)
        msk_C02= sitk.ReadImage(mask_C02_path)
        image_array_C02 = sitk.GetArrayFromImage(img_C02)
        mask_array_C02 = sitk.GetArrayFromImage(msk_C02)

        img_DE2 = sitk.ReadImage(img_DE2_path)
        msk_DE2 = sitk.ReadImage(mask_DE2_path)
        image_array_DE2 = sitk.GetArrayFromImage(img_DE2)
        mask_array_DE2 = sitk.GetArrayFromImage(msk_DE2)

        img_T2_2 = sitk.ReadImage(img_T2_2_path)
        msk_T2_2 = sitk.ReadImage(mask_T2_2_path)
        image_array_T2_2 = sitk.GetArrayFromImage(img_T2_2)
        mask_array_T2_2 = sitk.GetArrayFromImage(msk_T2_2)






#--------------------Resizing tensors to have the correct dimensions---------------------------------#
        desired_size = (470, 470)

        image_tensor_C02 = torch.from_numpy(image_array_C02).unsqueeze(0).float()
        img_tensor_C02 = image_tensor_C02.unsqueeze(0)

        mask_tensor_C02 = torch.from_numpy(mask_array_C02).unsqueeze(0).float()
        msk_tensor_C02 = mask_tensor_C02.unsqueeze(0)

        image_tensor_DE2 = torch.from_numpy(image_array_DE2).unsqueeze(0).float()
        img_tensor_DE2 = image_tensor_DE2.unsqueeze(0)

        mask_tensor_DE2 = torch.from_numpy(mask_array_DE2).unsqueeze(0).float()
        msk_tensor_DE2 = mask_tensor_DE2.unsqueeze(0)

        image_tensor_T2_2 = torch.from_numpy(image_array_T2_2).unsqueeze(0).float()
        img_tensor_T2_2 = image_tensor_T2_2.unsqueeze(0)

        mask_tensor_T2_2 = torch.from_numpy(mask_array_T2_2).unsqueeze(0).float()
        msk_tensor_T2_2 = mask_tensor_T2_2.unsqueeze(0)


        img_resized_tensor_C02 = F.interpolate(img_tensor_C02, size=desired_size, mode='bilinear', align_corners=False)
        msk_resized_tensor_C02 = F.interpolate(msk_tensor_C02, size=desired_size, mode='bilinear', align_corners=False)

        img_resized_tensor_DE2 = F.interpolate(img_tensor_DE2, size=desired_size, mode='bilinear', align_corners=False)
        msk_resized_tensor_DE2 = F.interpolate(msk_tensor_DE2, size=desired_size, mode='bilinear', align_corners=False)

        img_resized_tensor_T2_2 = F.interpolate(img_tensor_T2_2, size=desired_size, mode='bilinear', align_corners=False)
        msk_resized_tensor_T2_2 = F.interpolate(msk_tensor_T2_2, size=desired_size, mode='bilinear', align_corners=False)

        img_resized_array_C02 = np.array(img_resized_tensor_C02).squeeze(0)
        msk_resized_array_C02 = np.array(msk_resized_tensor_C02).squeeze(0)

        img_resized_array_DE2 = np.array(img_resized_tensor_DE2).squeeze(0)
        msk_resized_array_DE2= np.array(msk_resized_tensor_DE2).squeeze(0)

        img_resized_array_T2_2 = np.array(img_resized_tensor_T2_2).squeeze(0)
        msk_resized_array_T2_2 = np.array(msk_resized_tensor_T2_2).squeeze(0)

        #optimized_image_array_C0 = img_resized_array_C0 * msk_resized_array_C0
        #optimized_image_array_DE = img_resized_array_DE * msk_resized_array_DE
        #optimized_image_array_T2 = img_resized_array_T2 * msk_resized_array_T2


        mean_img = 277.6305
        std_img = 390.7119
        mean_msk = 0.0167
        std_mask = 0.1219

        #We apply normilization , so that the model doesn't train itself on attributes that are merely artifacts of the pixel distribution,
        ## rather than the nature of the tissue. 
        image_C0_seg = normalization(img_resized_array_C02)
        image_DE_seg = normalization(img_resized_array_DE2)
        image_T2_seg = normalization(img_resized_array_T2_2)

        binary_mask_C0_array = np.copy(msk_resized_array_C02)

        binary_mask_C0_tensor = torch.from_numpy(binary_mask_C0_array).squeeze(0)
        mask_seg_C0_seg_tensor = torch.from_numpy(msk_resized_array_C02).squeeze(0)
        image_C0_seg_tensor = torch.from_numpy(image_C0_seg).squeeze(0)
        image_DE_seg_tensor = torch.from_numpy(image_DE_seg).squeeze(0)
        image_T2_seg_tensor = torch.from_numpy(image_T2_seg).squeeze(0)

        #Get the area of interest
        bb_min, bb_max = get_ND_bounding_box(mask_seg_C0_seg_tensor, margin=100)

       #Here we select the area of interest , by setting all areas that are not of interest , to a pixel intensity to 0. 
        binary_mask_C0_seg_bound = set_ND_volume_roi_with_bounding_box_range(binary_mask_C0_tensor, bb_min, bb_max)
        mask_C0_seg_bound = set_ND_volume_roi_with_bounding_box_range(mask_seg_C0_seg_tensor, bb_min, bb_max)
        image_C0_seg_bound = set_ND_volume_roi_with_bounding_box_range(image_C0_seg_tensor, bb_min, bb_max)
        image_DE_seg_bound = set_ND_volume_roi_with_bounding_box_range(image_DE_seg_tensor, bb_min, bb_max)
        image_T2_seg_bound = set_ND_volume_roi_with_bounding_box_range(image_T2_seg_tensor, bb_min, bb_max)
#----------------------------------BINARY MASK---------------------------------------#
        #This mask is binary, and used for the concatenation
        binary_mask_C0_seg_array = np.array(binary_mask_C0_seg_bound)

        valid_values = [200, 1220, 2221]
        binary_mask_C0_seg_array[~np.isin(binary_mask_C0_seg_array, valid_values)] = 0.0
        binary_mask_C0_seg_array[(binary_mask_C0_seg_array == 200)] = 1.0
        binary_mask_C0_seg_array[(binary_mask_C0_seg_array == 1220)] = 1.0
        binary_mask_C0_seg_array[(binary_mask_C0_seg_array == 2221)] = 1.0

        #This one will be plugged in to the center function and after in the concatenation below!
        binary_mask_C0_seg_tensor = torch.from_numpy(binary_mask_C0_seg_array).unsqueeze(0)
#-------------------------------MULTI-MASK---------------------------------------------#

        # This mask is multi-label and used for one_hot encoding
        mask_seg_C0_seg_array = np.array(mask_C0_seg_bound)
        
        # Concatenating all modalities
        valid_values = [200, 1220, 2221]
        mask_seg_C0_seg_array[~np.isin(mask_seg_C0_seg_array, valid_values)] = 0.0
        mask_seg_C0_seg_array[(mask_seg_C0_seg_array == 200)] = 1.0
        mask_seg_C0_seg_array[(mask_seg_C0_seg_array == 1220)] = 2.0
        mask_seg_C0_seg_array[(mask_seg_C0_seg_array == 2221)] = 3.0

        #This is the one_hot mask as well
        mask_C0_seg_bound_classified = torch.from_numpy(mask_seg_C0_seg_array)

        image_seg_C0_seg_tensor_dim = image_C0_seg_bound.unsqueeze(0)
        image_seg_DE_seg_tensor_dim = image_DE_seg_bound.unsqueeze(0)
        image_seg_T2_seg_tensor_dim = image_T2_seg_bound.unsqueeze(0)

        tensor_center = T.CenterCrop(256)
        binary_mask_C0_seg_tensor_center = tensor_center(binary_mask_C0_seg_tensor)
        mask_seg_C0_seg_tensor_one_hot = tensor_center(mask_C0_seg_bound_classified)
        image_seg_C0_seg_tensor_center = tensor_center(image_seg_C0_seg_tensor_dim)
        image_seg_DE_seg_tensor_center = tensor_center(image_seg_DE_seg_tensor_dim)
        image_seg_T2_seg_tensor_center = tensor_center(image_seg_T2_seg_tensor_dim)

        augmented_image_concat = torch.cat((image_seg_DE_seg_tensor_center, image_seg_T2_seg_tensor_center, image_seg_C0_seg_tensor_center, binary_mask_C0_seg_tensor_center), dim=0)

        augmented_mask_tensor = mask_seg_C0_seg_tensor_one_hot.long()

        one_hot_mask_tensor = F.one_hot(augmented_mask_tensor)
        one_hot_mask_tensor_resized = one_hot_mask_tensor.permute(2, 0, 1).float()


        augmented_image = np.array(augmented_image_concat)
        augmented_mask = np.array(one_hot_mask_tensor_resized)

        #Apply augmentations
        augmentations = self.transform(image=augmented_image, mask=augmented_mask)
        augmented_image = augmentations['image']
        augmented_mask = augmentations['mask']
    
        augmented_mask = np.copy(augmented_mask)
        augmented_image = np.copy(augmented_image)

        #Return the processed multichannel-images
        return augmented_image, augmented_mask
    


