from mimetypes import init
import torch
import torch.nn as nn
from model.backbone_ConvNeXt import ConvNeXt
from model.decoder_UPerhead import UPerHead

class upernet_convnext_tiny(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.backbone = ConvNeXt(in_chans=in_chans)
        self.decoder = UPerHead(in_channels=[16, 32, 64, 128], #tiny的参数
                        in_index=[0, 1, 2, 3],
                        pool_scales=(1, 2, 3, 6),
                        channels=out_chans,
                        dropout_ratio=0.1,
                        num_classes=out_chans,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        align_corners=False,
                        loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        
    def forward(self, x):
        backbone_out = self.backbone(x)
        out = self.decoder(backbone_out) 
        return out

if __name__ == '__main__':
    from torchsummary import summary
    data = torch.randn(2,1,256,256).cuda()
    a = upernet_convnext_tiny(1,4).cuda()
    s = a(data)
    print(s.shape)