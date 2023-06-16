import torch
from utils.reshape import reshape
from utils.dice_metric import DiceMetric
from utils.one_hot import one_hot
from utils.Segloss import SegLoss

def evaluate(model, data_loader, device, epoch,num_classes):
    model.eval()

    loss_func = SegLoss()
    total_dice = torch.zeros(1,num_classes).to(device)
    total_loss = 0
    number = 0
    for _, batch_data in enumerate(data_loader):
        images, labels = batch_data['img'], batch_data['seg']
        one_hot_labels = one_hot(labels,num_classes)  #[n,1,d,w]-->[n,num_class,d,w]
        
        images = images.to(device)
        labels = labels.to(device)
        one_hot_labels = one_hot_labels.to(device)


        #predict
        pred = model(images)
        pred = reshape(pred,images.size())
        loss = loss_func(pred, one_hot_labels, is_average=True)

        #dice
        Dice = DiceMetric()
        dice_metric = Dice(pred,one_hot_labels,num_classes)
        
        total_dice = torch.add(total_dice, dice_metric)
        total_loss = total_loss + loss.item()
        number = number + 1
        
    epoch_mean_dice = total_dice/number
    epoch_mean_loss = total_loss/number
    
    print('evaluate---------epoch:{} , loss: {} , dice:  {}  {}  {}  {}'\
        .format(epoch,epoch_mean_loss,epoch_mean_dice[0][1].item(),epoch_mean_dice[0][2].item(),epoch_mean_dice[0][3].item(),epoch_mean_dice[0][4].item()))
    # print('evaluate---------epoch:{} , loss: {} , dice: {}  {} '\
    #     .format(epoch,epoch_mean_loss,epoch_mean_dice[0][0].item(),epoch_mean_dice[0][1].item()))
    
    return epoch_mean_loss, epoch_mean_dice