import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from Dice_score import dice_loss, dice_score,  accuracy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_fn_2(loader, model, optimizer, x_axis):
    torch.autograd.set_detect_anomaly(True)    # WHEN YOu HAVE ACCESS TO GPU, PUT SCALER
    writer = SummaryWriter('runs/model_2/writer_71_test')
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
        predictions = model(data)
        predictions_sigmoid = torch.sigmoid(predictions)
        target_sigmoid = torch.sigmoid(target)

        dice_loss_val = dice_loss(target_sigmoid, predictions_sigmoid)
        cross_entropy_val = F.cross_entropy(target_sigmoid, predictions_sigmoid)
        loss = dice_loss_val + cross_entropy_val
        initial_loss = initial_loss + loss.item()

        dice = dice_score(target_sigmoid, predictions_sigmoid)
        initial_dice = initial_dice + dice
        acc = accuracy(target_sigmoid, predictions_sigmoid)

        #backward PASS
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        # Update  tqdm loop
        step = step + data.shape[0]
        loop.set_postfix(loss=initial_loss, dice_score = dice)
        print('epoch:{}, average_dice:{}'.format(x_axis, initial_dice / step))

        #Use Summary Writer to visualize training process
        writer.add_scalar('Accuracy', acc, x_axis)
        writer.add_scalar('Training loss', loss, x_axis)
        writer.add_scalar('Dice score', dice, x_axis)
        writer.flush()