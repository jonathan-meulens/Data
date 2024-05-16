#Import libraries
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dice_score import accuracy

#---------------------dataset.py-------------------------------------#
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        #Loss function
        loss = loss_fn_1(predictions, target)
        initial_loss += loss.item()

        #Dice_score 
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



