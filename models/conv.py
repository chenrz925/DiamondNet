from math import floor
from typing import Any, Dict, Text, List

import torch
from torch import nn


class Conv1LayerCoSeBlock(nn.Module):
    @staticmethod
    def _padding_size(kernel_size):
        return int(floor((kernel_size - 1) / 2))

    def __init__(self, profile):
        super(Conv1LayerCoSeBlock, self).__init__()
        self.vertexes = profile['vertexes']
        self.add_module('conv', nn.ModuleList((
            nn.Conv1d(
                in_channels=profile['conv'][vertex]['in_channels'],
                out_channels=profile['conv'][vertex]['growth_ratio'],
                kernel_size=profile['conv'][vertex]['kernel_size'],
                padding=self._padding_size(profile['conv'][vertex]['kernel_size']),
            ) for vertex in range(self.vertexes)
        )))
        self.add_module('bn', nn.ModuleList((
            nn.BatchNorm1d(profile['conv'][vertex]['in_channels'])
            for vertex in range(self.vertexes)
        )))
        self.add_module('selu', nn.ModuleList((
            nn.SELU() for vertex in range(self.vertexes)
        )))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> List[torch.Tensor]:
        children = dict(self.named_children())
        return list(map(
            lambda vertex:
            children['selu'][vertex](
                children['conv'][vertex](
                    children['bn'][vertex](input[0][vertex])
                )
            ),
            range(self.vertexes)
        ))
