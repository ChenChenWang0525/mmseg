from pyexpat import model
import torch 
import argparse
import glob
import os
from dataset.dataset import MyDataSet
from torchvision import transforms
import torch.optim as optim
from utils.evaluate_one_epoch import evaluate
from utils.train_one_epoch import train_one_epoch
#from unet_model.unet_model import UNet_monai
from model.upernet_convnext_tiny import  upernet_convnext_tiny
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #数据集路径
    data = sorted(glob.glob(os.path.join(args.data_path, "images", "*.png")))
    label = sorted(glob.glob(os.path.join(args.data_path, "labels", "*.png")))
    train_image_path = data; train_label_path = label
    val_image_path = data; val_label_path = label
    
    # 定义transform
    train_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485],std=[0.229])])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485],std=[0.229])])

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_image_path,
                              label_path=train_label_path,
                              transform=train_transform)
    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_image_path,
                            label_path=val_label_path,
                            transform=val_transform)

   
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1)
    
    #定义模型，损失函数，优化器
    model = upernet_convnext_tiny(in_chans=args.in_chans,out_chans=args.out_chans).to(device)
    #model = UNet_monai(n_channels=args.in_chans, n_classes=args.out_chans).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    #epoch训练
    best_dice = torch.zeros(1,args.out_chans).to(device)
    step = 0
    for epoch in range(args.epochs):
        #train
        train_loss, train_dice = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                num_classes=args.out_chans,
                                                step = step)

        #validate
        val_loss, val_dice = evaluate(model=model,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch,
                                    num_classes=args.out_chans)


        if best_dice[0].sum(dim=0) < val_dice[0].sum(dim=0):
            torch.save(model.state_dict(), "./weights/convnext_5.pth")
            best_dice = val_dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_chans', type=int,default=1)
    parser.add_argument('--out_chans', type=int,default=5)
    parser.add_argument('--epochs', type=int,default=10)
    parser.add_argument('--batch_size', type=int,default=4)
    parser.add_argument('--lr', type=float,default=1e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--data_path', type=str,default="./data/")
    args = parser.parse_args()
    main(args)