from typing import Any

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    """
    SE（Squeeze-Excitation） Block实现
    参数结构
    |- in_channels: int, 输入张量的通道数
    |- reduction_ratio: float, Excitation中通道缩小的倍数
    """

    def __init__(self, in_channels: int, reduction_ratio: float):
        super(SqueezeExcitation, self).__init__()
        self.in_channels = in_channels
        self.add_module('squeeze', nn.AdaptiveMaxPool1d(1))
        self.temp_channels = int(in_channels // reduction_ratio)
        if self.temp_channels <= 0:
            self.temp_channels = 1
        self.add_module('excitation', nn.ModuleDict({
            'linear0': nn.Linear(
                in_features=in_channels,
                out_features=self.temp_channels,
            ),
            'relu': nn.ReLU(),
            'linear1': nn.Linear(
                in_features=self.temp_channels,
                out_features=in_channels,
            ),
            'sigmoid': nn.Sigmoid()
        }))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        :param input: 模型输入，单张量
        :return: 模型输出，单张量
        """
        childrens = dict(self.named_children())
        output: torch.Tensor = childrens['squeeze'](input[0]).view(*input[0].size()[:2])
        output = childrens['excitation']['linear0'](output)
        output = childrens['excitation']['relu'](output)
        output = childrens['excitation']['linear1'](output)
        output = childrens['excitation']['sigmoid'](output)
        return output.view(*output.size(), 1)


class SqueezeExcitationSoftmax(SqueezeExcitation):
    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return super(SqueezeExcitationSoftmax, self).forward(*input, **kwargs).softmax(dim=1)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.ones(*input[0].size()[:2], 1).float().to(input[0].device)
