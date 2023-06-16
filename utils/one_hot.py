import torch
import torch.nn.functional as F
def one_hot(label,num_classes):
    '''
        label:[n,1,d,w] --->[n,num_classes,d,w]
    '''
    label = label[:,0,:,:].long()  #[n,d,w]
    label = F.one_hot(label,num_classes)
    label =  torch.transpose(torch.transpose(label,1,3),2,3)
    return label
