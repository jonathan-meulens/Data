#Import libraries
import os
import torch
from torch.utils.data import DataLoader
from model import UNET
from Dense_UNET_model import Dense_UNET
from Dense_dataset import DenseDataset
from BiFPN_model import Myops_UNET
from Model_checkpoints import save_checkpoint, save_predictions_as_imgs, load_checkpoint
from Dense_validation import val_dense
from train_dense import train_fn_dense
import albumentations as A

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
LOAD_MODEL = True

#Folder saved DENSE UNET model states  
MODEL_DENSE_CHECKPOINTS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\model_dense\\model_dense_checkpoints\\my_checkpoint.pt'


#Folder saved DENSE UNET model predictions  
DENSE_SAVED_PREDICTIONS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\model_dense\\model_dense_predictions\\'



#Main function
def main():
    model_1 = UNET()
    model_2 = Myops_UNET()
    model_dense = Dense_UNET()

    # optimizer for model1
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=LEARNING_RATE[3])

    # optimizer for model2
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=LEARNING_RATE[2])

    # optimizer for model_dense
    optimizer_dense = torch.optim.Adam(model_1.parameters(), lr=LEARNING_RATE[3])

    #Augementations Pipeline
    train_transform = A.Compose(
        [
          A.Rotate(limit=35, p=0.2),
          A.HorizontalFlip(p=0.1),
          #A.RandomBrightnessContrast(p=0.5),
          #A.ElasticTransform(p=0.9),
          #A. ShiftScaleRotate(p=0.5),
         # A. GridDistortion(p=0.8),
          A.VerticalFlip(p=0.1),

        ],
    )
    val_transform = A.Compose(
        [
            A.Rotate(limit=35, p=0.2),
            A.HorizontalFlip(p=0.1),
            # A.RandomBrightnessContrast(p=0.5),
            #A.ElasticTransform(p=0.7),
            # A.ShiftScaleRotate(p=0.5),
             #A.GridDistortion(p=0.8),
            # A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

        ])
    
    #Dataset object and loader for training Dense UNET model
    train_ds_model_dense = DenseDataset(
        transform=train_transform)

    train_loader_model_dense = DataLoader(
        train_ds_model_dense,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        # timeout=20
    )

    #Dataset object and loader for validating Dense UNET model
    val_ds_model_dense = DenseDataset(
        transform=val_transform)

    val_loader_model_dense = DataLoader(
        val_ds_model_dense,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        #timeout=20
    )

    #Start training
    for epoch in range(NUM_EPOCHS):
        if epoch >= 1:
            load_checkpoint(MODEL_DENSE_CHECKPOINTS)
        #train model
        train_fn_dense(train_loader_model_dense, model_dense, optimizer_dense, epoch)

        #validation_round
        initial_dice, accuracy, initial_loss= val_dense(val_loader_model_dense, model_dense, epoch)
        print(f'Val_Dice:{initial_dice}, Val_Accuracy:{accuracy}, Val_Loss:{initial_loss}')

        # save model
        checkpoint = {
            'state_dict': model_dense.state_dict(),
            'optimizer': optimizer_dense.state_dict(),
        }
        save_checkpoint(checkpoint, MODEL_DENSE_CHECKPOINTS)

        save_predictions_as_imgs(
        train_loader_model_dense, model_dense, folder=DENSE_SAVED_PREDICTIONS, device=DEVICE)


if __name__ == '__main__':
    main()
