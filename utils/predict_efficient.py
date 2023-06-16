import cv2
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import torch
import os
from utils.reshape import reshape
#from unet_model.unet_model import UNet_monai
from utils.dice_metric import DiceMetric
from utils.one_hot import one_hot
from model.upernet_convnext_tiny import  upernet_convnext_tiny

def main(img_path, label_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    img_path = img_path
    label_path = label_path
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485], std=[0.229])])

    img = Image.open(img_path)
    img = train_transform(img)
    label = Image.open(label_path)
    label = torch.Tensor(np.array(label))
    label = torch.unsqueeze(label,dim=0)

    img = torch.unsqueeze(img, dim=0)
    label = torch.unsqueeze(label, dim=0)

    #model = UNet_monai(n_channels=1, n_classes=5).to(device)
    model = upernet_convnext_tiny(in_chans=1,out_chans=5).to(device)

    # load model weights
    model_weight_path = "../weights/convnext_5.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        one_hot_labels = one_hot(label, 5)  # [n,1,d,w]-->[n,num_class,d,w]
        data = torch.randn(2,3,512,512)
        output = model(img.to(device))
        output = reshape(output,data.size())

        Dice = DiceMetric()
        dice_metric = Dice(output.cpu(), one_hot_labels, 5)
        print(dice_metric)

        pred = F.softmax(output, dim=1)
        pred = pred.argmax(dim=1)  # 最大的索引值
        pred = pred.cpu()
        pred = np.array(pred)
        cv2.imshow('s', pred[0] * 255.0)
        cv2.waitKey()


def path_get(dir):
    path = dir
    list = []
    pre = []
    new_pre = []
    for filename in os.listdir(path):
        name = dir + "\\" + filename
        list.append(name)
    return list


if __name__ == '__main__':
    path_train = "../data/images"
    path_label = "../data/labels"
    path_trains = path_get(path_train)
    path_labels = path_get(path_label)
    for i in range(len(path_labels)):
        main(path_trains[i], path_labels[i])
