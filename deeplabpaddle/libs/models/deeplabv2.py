#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer, _Stem


class _ASPP(nn.Layer):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_sublayer(
                "c{}".format(i),
                nn.Conv2D(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias_attr=True),
            )

        #for m in self.children():
        #    nn.initializer.Normal(m.weight, mean=0, std=0.01)
        #    nn.initializer.Constant(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_sublayer("layer1", _Stem(ch[0]))
        self.add_sublayer("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_sublayer("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_sublayer("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_sublayer("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_sublayer("aspp", _ASPP(ch[5], n_classes, atrous_rates))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

'''
if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = paddle.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
'''