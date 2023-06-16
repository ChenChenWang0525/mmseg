import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceMetric(nn.Module):
    '''
        predict:[n,c,d,w] 
        gt:[n,c,d,w]  one-hot
    '''

    def __init__(self, dims=(2, 3)):
        super(DiceMetric, self).__init__()
        self.dims = dims

    def forward(self, predict, gt,num_classes, activation='softmax', is_average=True):
        predict = predict.float()
        gt = gt.float()

        if activation == 'sigmoid':  #不算背景，求单独一张特征图，划分0，1
            pred = F.sigmoid(predict)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
        elif activation == 'softmax': #算背景，经过softmax和argmax,求
            pred = F.softmax(predict, dim=1)
            pred = pred.argmax(dim=1)  #最大的索引值
            pred = F.one_hot(pred,num_classes)
            pred =  torch.transpose(torch.transpose(pred,1,3),2,3) #one-hot的预测[n,c,w,h]

        intersection = torch.sum(pred * gt, dim=self.dims)
        union = torch.sum(pred, dim=self.dims) + torch.sum(gt, dim=self.dims)
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        # dice.mean(0) 得到 0 维（行）均值
        dice = dice.mean(0) if is_average else dice.sum(0)
        return dice


