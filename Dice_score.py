import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10

def dice_score(input, prediction):
    smooth = 1e-6
    prediction = (prediction >= 0.5).float() * 1
    intersection = (input * prediction).sum()
    dice = (2. * intersection + smooth) / (input.sum() + prediction.sum() + smooth)
    return dice

def dice_loss(input, prediction):
    smooth = 1e-6
    prediction = (prediction >= 0.5).float() * 1
    intersection = (input * prediction).sum()
    dice = (2. * intersection + smooth) / (input.sum() + prediction.sum() + smooth)
    return 1 - dice

def accuracy(target, prediction):
    num_correct = 0
    num_pixels = 0
    prediction = (prediction >= 0.5).float() * 1
    num_correct += (prediction == target).sum().item()
    num_pixels += torch.numel(prediction)
    acc = num_correct/num_pixels
    return acc

def val(loader, model, x_axis, device=DEVICE):
    loss_fn = torch.nn.BCELoss()
    writer = SummaryWriter(f'runs/model_1/writer_70')
    initial_loss = 0
    initial_dice = 0
    step = 0

    model.eval()
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x = x.to(device=DEVICE, dtype=torch.float32)
        y = y.to(device=DEVICE, dtype=torch.float32)
        #print(y.shape)
        with torch.no_grad():
            preds = model(x)
            #print(preds.shape)
            acy = accuracy(y,preds)
            # forward
            # with torch.cuda.amp.autocast():
            preds = torch.sigmoid(preds)
            #print(y[(y>1)])
            y = torch.sigmoid(y)
            loss = loss_fn(preds, y)
            initial_loss += loss.item()

            dice = dice_score(y, preds)
            initial_dice += dice

            step += x.shape[0]
            writer.add_scalar('Accuracy', acy, x_axis)
            writer.add_scalar('Training loss', loss, x_axis)
            writer.add_scalar('Val Dice score', initial_dice/step, x_axis)
            writer.flush()
            if x_axis == NUM_EPOCHS:
                writer.close()
            return initial_dice, accuracy, initial_loss
    model.train()

def val_2(loader, model, x_axis, device=DEVICE):
    writer = SummaryWriter(f'runs/model_2/writer_71_test') # Post optimization
    initial_loss = 0
    initial_dice = 0
    step = 0

    model.eval()
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x = x.to(device=DEVICE, dtype=torch.float32)
        y = y.to(device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            preds = model(x)

            acy = accuracy(y,preds)
            # forward PASS
            # with torch.cuda.amp.autocast():
            preds_sig = torch.sigmoid(preds)
            y_sig = torch.sigmoid(y)
            dice_loss_val = dice_loss(y_sig, preds_sig)
            cross_entropy_val = F.cross_entropy(y_sig, preds_sig)
            loss = dice_loss_val + cross_entropy_val
            initial_loss += loss.item()

            dice = dice_score(y_sig, preds_sig)
            initial_dice += dice

            step += x.shape[0]
            writer.add_scalar('Accuracy', acy, x_axis)
            writer.add_scalar('Training loss', loss, x_axis)
            writer.add_scalar('Val Dice score', initial_dice/step, x_axis)
            writer.flush()
            if x_axis == NUM_EPOCHS:
                writer.close()
            return initial_dice, accuracy, initial_loss
    model.train()