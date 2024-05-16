#Import libraries
import os
import torch
from torch.utils.data import DataLoader
from model import UNET
from BiFPN_model import Myops_UNET
from Model_checkpoints import save_checkpoint, save_predictions_as_imgs, load_checkpoint
from dataset import MyOPSDataset
from Dice_score import val
from train import train_fn_1
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
LOAD_MODEL = False

#Folder with model predictions
SAVED_PREDS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\predictions_model_1\\'

#Folder with saved model states
MODEL1_CHEKCPOINTS = 'C:\\Users\\victo\\anaconda3\\envs\\Jay\\Data\\MyoSeg\\train_model1\\model1_checkpoints\\my_checkpoint.pt'


#Main function
def main():
    #Call the both the basic UNET model (model_1) and the specialized Myops_UNET model (model_2)
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
    
    #Dataset and loader for training model 1
    train_ds_model1 = MyOPSDataset(
        transform=train_transform)

    train_loader_model1 = DataLoader(
        train_ds_model1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,

    )

    #Dataset and loader for validating model 1
    val_ds_model1 = MyOPSDataset(
        transform=val_transform)

    val_loader_model1 = DataLoader(
        val_ds_model1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,

    )


    #Start training
    for epoch in range(NUM_EPOCHS):
        #Load previous weights , if have them. 
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
        
        #Save the images that the model produces (model predictions) 
        save_predictions_as_imgs(
        train_loader_model1, model_1, folder=SAVED_PREDS, device=DEVICE)


if __name__ == '__main__':
    main()
