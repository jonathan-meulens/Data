#Import libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dice_score import dice_loss, accuracy


#Calculate Dice score
def dice_score(input, prediction):
    smooth = 1e-6
    prediction = (prediction>=0.5).float() * 1

    intersection = (input * prediction).sum()
    dice = (2. * intersection + smooth) / (input.sum() + prediction.sum() + smooth)
    return dice
    
# Training function for Dense_UNET()
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

        optimizer.zero_grad()
        # forward
        predictions = model(data)
        predictions = torch.sigmoid(predictions)
        #new_target = target.expand(predictions.shape[0], (predictions.shape[1]), predictions.shape[2], predictions.shape[3])
        target = torch.sigmoid(target)

        #Loss function 
        loss = loss_fn_1(predictions, target)
        initial_loss += loss.item()

        #Dice score calculation
        dice = dice_score(target, predictions)
        initial_dice += dice
        acc = accuracy(target, predictions)
       
        # backward
        #scaled_loss = torch.mul(loss, scale_factors[0])
        loss.backward()


        optimizer.step()
        # Update  tqdm loop
        step += data.shape[0]
        loop.set_postfix(loss=initial_loss, dice_score = dice)
        print('epoch:{}, average_dice:{}'.format(x_axis, initial_dice / step))
        writer.add_scalar('Accuracy', acc, x_axis)
        writer.add_scalar('Training loss', loss, x_axis)
        writer.add_scalar('Dice score', dice, x_axis)
        writer.flush()
