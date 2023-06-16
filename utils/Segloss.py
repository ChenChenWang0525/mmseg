import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import DiceLoss


class SegLoss(nn.Module):
    def __init__(self, loss_func='dice', activation='softmax'):
        super(SegLoss, self).__init__()
        assert loss_func in {'dice', 'diceAndBce', 'diceAndFocal', 'diceAndTopK', 'diceAndHausdorff'}
        assert activation in {'sigmoid', 'softmax'}
        self.loss_func = loss_func
        self.activation = activation

    def forward(self, predict, gt, is_average=True):
        '''
        predict:[n,c,d,w]
        gt:[n,c,d,w]   经过one-hot
        '''

        predict = predict.float()
        gt = gt.float()
        if self.activation == 'softmax':
            predict = F.softmax(predict, dim=1)
        elif self.activation == 'sigmoid':
            predict = F.sigmoid(predict)

        dice_loss_func = DiceLoss()
        loss = dice_loss_func(predict, gt, is_average)

        return loss
