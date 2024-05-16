import torch
from torch.utils.tensorboard import SummaryWriter
from Dice_score import accuracy, dice_score

#Validation for the Dense_UNET_model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10
def val_dense(loader, model, x_axis):
    loss_fn = torch.nn.BCELoss()
    writer = SummaryWriter(f'runs/model_dense/writer_2')
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
            # forward
            # with torch.cuda.amp.autocast():
            preds = torch.sigmoid(preds)
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
