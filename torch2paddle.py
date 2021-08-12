import torch
import paddle
import numpy as np
from deeplabpytorch.libs.models import DeepLabV2_ResNet101_MSC
# 构建输入
input_data = np.random.rand(1, 3, 321, 321).astype("float32")
# 获取PyTorch Module
torch_module = DeepLabV2_ResNet101_MSC(n_classes=21)
torch_dict = torch.load('deeplabpytorch/data/models/voc12/deeplabv2_resnet101_msc/caffemodel/deeplabv2_resnet101_msc-vocaug.pth')
paddle_dict = {}
for key in torch_dict:
    weight = torch_dict[key].cpu().numpy().astype(np.float32)
    key = key.replace('running_mean','_mean').replace('running_var','_variance')
    paddle_dict[key]=weight

paddle.save(paddle_dict,'deeplabpaddle/data/models/voc12/deeplabv2_resnet101_msc/train/deeplabv2_resnet101_msc-vocaug.pdparams')
