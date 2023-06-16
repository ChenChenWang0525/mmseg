from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch
from torchvision import  transforms
import numpy as np
class MyDataSet(Dataset):
    def __init__(self, images_path: list, label_path: list, transform=None,transform_label=None):
        self.images_path = images_path
        self.label_path = label_path
        self.transform = transform
        self.trainform_label = transform_label

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        
        img = Image.open(self.images_path[item])
        label = Image.open(self.label_path[item])
        #读取标签是3通道，但batch输出是单通道
        if self.transform is not None:
            img = self.transform(img)
        #label = transforms.ToTensor()(label)   #[512,512]这里都不需要进行扩充维度，在dataloader读取时会自动补充
        label = torch.Tensor(np.array(label))
        label = torch.unsqueeze(label,dim=0)
        # lable = label /255.0
        return {'img':img, 'seg':label}
