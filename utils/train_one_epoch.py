import torch
from utils.reshape import reshape
from utils.dice_metric import DiceMetric
from utils.one_hot import one_hot
from utils.Segloss import SegLoss
def train_one_epoch(model, optimizer, data_loader, device, epoch,num_classes,step):
    model.train()
    loss_func = SegLoss()
    total_dice = torch.zeros(1,num_classes).to(device)
    total_loss = 0
    number = 0
    for _, batch_data in enumerate(data_loader):
        step = step + 1
        images, labels = batch_data['img'], batch_data['seg']  #label : [0,1,2,3,4][0.0000, 0.0039, 0.0078]),
        one_hot_labels = one_hot(labels,num_classes)  #[n,1,d,w]-->[n,num_class,d,w]

        
        images = images.to(device)
        labels = labels.to(device)
        one_hot_labels = one_hot_labels.to(device)

        #梯度清0
        optimizer.zero_grad()
        
        #predict and loss
        pred = model(images)
        pred = reshape(pred,images.size())
        loss = loss_func(pred, one_hot_labels, is_average=True)

        #dice
        Dice = DiceMetric()
        dice_metric = Dice(pred,one_hot_labels,num_classes)

        #反向传播
        loss.backward()
        optimizer.step()

        #show
        total_dice = torch.add(total_dice, dice_metric)
        total_loss = total_loss + loss.item()
        number = number + 1

        if step % 10 == 0 :
            print("---------step:{} ,  dice:{}  {}   {} {}------ ".format(step,dice_metric[1].item(),dice_metric[2].item(),dice_metric[3].item(),dice_metric[4].item()))
            # print("---------step:{} ,  dice:{}  {}  ------ ".format(step,dice_metric[0].item()\
            #           ,dice_metric[1].item()))
    epoch_mean_dice = total_dice/number
    epoch_mean_loss = total_loss/number
    
    print('train---------epoch:{} , loss: {} , dice:  {}  {}  {} {}'\
        .format(epoch,epoch_mean_loss,epoch_mean_dice[0][1].item(),epoch_mean_dice[0][2].item(),epoch_mean_dice[0][3].item(),epoch_mean_dice[0][4].item()))
    # print('train---------epoch:{} , loss: {} , dice: {}  {} '\
    #     .format(epoch,epoch_mean_loss,epoch_mean_dice[0][0].item(),epoch_mean_dice[0][1].item()))
    
    return epoch_mean_loss, epoch_mean_dice