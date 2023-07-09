import torch
from torch import nn
depths = [3,4,6,3]
drop_path_rate = 0.7
pr = [x.item() for x in torch.linspace(0, drop_path_rate,sum(depths))]

#for i in pr:
    #print(i)

"""


print(len(pr))
layer_scale_init_value = 1e-2
layer_scale_1 = nn.Parameter(
    layer_scale_init_value * torch.ones((64)), requires_grad=True)


dd = layer_scale_1.unsqueeze(-1).unsqueeze(-1)
print(dd.shape)
print(layer_scale_1.unsqueeze(-1).unsqueeze(-1))

a = torch.ones(2, 2, 3, 3)
b = 2*torch.ones(2, 3, 3, 3)
c = torch.cat((a, b), 1)
print(c.shape)
"""
#a = torch.randn(2,3,4).permute(0,2,1)
#print(a)

a = torch.ones(2, 2, 128, 128)
cc = nn.Conv2d(2, 2, 13, 1, 78, 13)

x = cc(a)

print(x.shape)




# 创建输入张量
input_tensor = torch.randn(2, 48, 256, 256)

# 定义全局平均池化层
global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

# 进行全局平均池化
output_tensor = global_avg_pool(input_tensor)

# 输出结果
print('输出',output_tensor.shape)  # 打印压缩后的特征图大小
