from typing import Dict, Text, Any, Tuple, Union

import torch
from torch import nn


class DenoiseL(nn.Module):
    def __init__(self, in_features: int, ratio: float):
        super(DenoiseL, self).__init__()
        assert in_features > 0
        assert 0.0 <= ratio < 1.0
        self.permutation = nn.Parameter(torch.randperm(in_features), requires_grad=False)
        self.ratio = ratio
        self.in_features = in_features

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return input[0].index_fill(-1, self.permutation[:int(self.ratio * self.in_features)], 0.0)

    def __repr__(self):
        return f'DenoiseL({self.in_features}, ratio={self.ratio})'


class ConvAutoEncoder1LayerDeCoBnCotSi(nn.Module):
    def __init__(self, **kwargs):
        super(ConvAutoEncoder1LayerDeCoBnCotSi, self).__init__()
        self.add_module('encoder', nn.ModuleDict({
            'denoise': DenoiseL(kwargs['in_features'],
                                kwargs['denoise']['ratio'] if 'denoise' in kwargs and 'ratio' in kwargs[
                                    'denoise'] else 0.2),
            'conv': nn.Conv1d(
                in_channels=kwargs['conv1d']['in_channels'],
                out_channels=kwargs['conv1d']['out_channels'],
                kernel_size=kwargs['conv1d']['kernel_size'],
                stride=kwargs['conv1d']['stride'] if 'conv1d' in kwargs and 'stride' in kwargs['conv1d'] else 1,
                padding=kwargs['conv1d']['padding'] if 'conv1d' in kwargs and 'padding' in kwargs['conv1d'] else 0,
                dilation=kwargs['conv1d']['dilation'] if 'conv1d' in kwargs and 'dilation' in kwargs['conv1d'] else 1,
                groups=kwargs['conv1d']['groups'] if 'conv1d' in kwargs and 'groups' in kwargs['conv1d'] else 1,
                bias=kwargs['conv1d']['bias'] if 'conv1d' in kwargs and 'bias' in kwargs['conv1d'] else True,
                padding_mode=kwargs['padding_mode'] if 'conv1d' in kwargs and 'padding_mode' in kwargs[
                    'conv1d'] else 'zeros',
            ),
            'batchnorm': nn.BatchNorm1d(
                num_features=kwargs['conv1d']['out_channels']
            ),
        }))
        self.add_module('decoder', nn.ModuleDict({
            'convtranspose': nn.ConvTranspose1d(
                in_channels=kwargs['conv1d']['out_channels'],
                out_channels=kwargs['conv1d']['in_channels'],
                kernel_size=kwargs['conv1d']['kernel_size'],
                stride=kwargs['conv1d']['stride'] if 'stride' in kwargs else 1,
                padding=kwargs['conv1d']['padding'] if 'padding' in kwargs else 0,
                dilation=kwargs['conv1d']['dilation'] if 'dilation' in kwargs else 1,
                groups=kwargs['conv1d']['groups'] if 'groups' in kwargs else 1,
                bias=kwargs['conv1d']['bias'] if 'bias' in kwargs else True,
                padding_mode=kwargs['padding_mode'] if 'padding_mode' in kwargs else 'zeros',
            ),
            'sigmoid': nn.Sigmoid()
        }))

    def forward(self, *input: torch.Tensor, **kwargs: Dict[Text, torch.Tensor]) -> Union[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        return_features = kwargs['return_features'] if 'return_features' in kwargs else False
        childrens = dict(self.named_children())
        features = childrens['encoder']['denoise'](input[0])
        features = childrens['encoder']['conv'](features)
        features = childrens['encoder']['batchnorm'](features)
        output_features = features
        features = childrens['decoder']['convtranspose'](features)
        features = childrens['decoder']['sigmoid'](features)
        if return_features:
            return features, output_features
        else:
            return features


class ConvAutoEncoder1LayerDeCoSeCotSi(nn.Module):
    def __init__(self, **kwargs: Dict[Text, Any]):
        super(ConvAutoEncoder1LayerDeCoSeCotSi, self).__init__()
        self.add_module('encoder', nn.ModuleDict({
            'denoise': DenoiseL(kwargs['in_features'],
                                kwargs['denoise']['ratio'] if 'denoise' in kwargs and 'ratio' in kwargs[
                                    'denoise'] else 0.2),
            'conv': nn.Conv1d(
                in_channels=kwargs['conv1d']['in_channels'],
                out_channels=kwargs['conv1d']['out_channels'],
                kernel_size=kwargs['conv1d']['kernel_size'],
                stride=kwargs['conv1d']['stride'] if 'conv1d' in kwargs and 'stride' in kwargs['conv1d'] else 1,
                padding=kwargs['conv1d']['padding'] if 'conv1d' in kwargs and 'padding' in kwargs['conv1d'] else 0,
                dilation=kwargs['conv1d']['dilation'] if 'conv1d' in kwargs and 'dilation' in kwargs['conv1d'] else 1,
                groups=kwargs['conv1d']['groups'] if 'conv1d' in kwargs and 'groups' in kwargs['conv1d'] else 1,
                bias=kwargs['conv1d']['bias'] if 'conv1d' in kwargs and 'bias' in kwargs['conv1d'] else True,
                padding_mode=kwargs['padding_mode'] if 'conv1d' in kwargs and 'padding_mode' in kwargs[
                    'conv1d'] else 'zeros',
            ),
            'selu': nn.SELU(),
        }))
        self.add_module('decoder', nn.ModuleDict({
            'convtranspose': nn.ConvTranspose1d(
                in_channels=kwargs['conv1d']['out_channels'],
                out_channels=kwargs['conv1d']['in_channels'],
                kernel_size=kwargs['conv1d']['kernel_size'],
                stride=kwargs['conv1d']['stride'] if 'stride' in kwargs else 1,
                padding=kwargs['conv1d']['padding'] if 'padding' in kwargs else 0,
                dilation=kwargs['conv1d']['dilation'] if 'dilation' in kwargs else 1,
                groups=kwargs['conv1d']['groups'] if 'groups' in kwargs else 1,
                bias=kwargs['conv1d']['bias'] if 'bias' in kwargs else True,
                padding_mode=kwargs['padding_mode'] if 'padding_mode' in kwargs else 'zeros',
            ),
            'sigmoid': nn.Sigmoid()
        }))

    def forward(self, *input: torch.Tensor, **kwargs: Dict[Text, torch.Tensor]) -> Union[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        return_features = kwargs['return_features'] if 'return_features' in kwargs else False
        childrens = dict(self.named_children())
        features = childrens['encoder']['denoise'](input[0])
        features = childrens['encoder']['conv'](features)
        features = childrens['encoder']['selu'](features)
        output_features = features
        features = childrens['decoder']['convtranspose'](features)
        features = childrens['decoder']['sigmoid'](features)
        if return_features:
            return features, output_features
        else:
            return features


class ConvAutoEncoder2LayerDeCoSeCoSeCotSeCotSi(nn.Module):
    def __init__(self, **kwargs):
        super(ConvAutoEncoder2LayerDeCoSeCoSeCotSeCotSi, self).__init__()
        self.add_module('encoder', nn.ModuleDict({
            'denoise': DenoiseL(kwargs['in_features'],
                                kwargs['denoise']['ratio'] if 'denoise' in kwargs and 'ratio' in kwargs[
                                    'denoise'] else 0.2),
            'conv1': nn.Conv1d(
                in_channels=kwargs['conv1d'][0]['in_channels'],
                out_channels=kwargs['conv1d'][0]['out_channels'],
                kernel_size=kwargs['conv1d'][0]['kernel_size'],
                stride=kwargs['conv1d'][0]['stride'] if 'conv1d' in kwargs and 'stride' in kwargs['conv1d'][0] else 1,
                padding=kwargs['conv1d'][0]['padding'] if 'conv1d' in kwargs and 'padding' in kwargs['conv1d'][
                    0] else 0,
                dilation=kwargs['conv1d'][0]['dilation'] if 'conv1d' in kwargs and 'dilation' in kwargs[
                    'conv1d'] else 1,
                groups=kwargs['conv1d'][0]['groups'] if 'conv1d' in kwargs and 'groups' in kwargs['conv1d'][0] else 1,
                bias=kwargs['conv1d'][0]['bias'] if 'conv1d' in kwargs and 'bias' in kwargs['conv1d'][0] else True,
                padding_mode=kwargs['padding_mode'] if 'conv1d' in kwargs and 'padding_mode' in kwargs[
                    'conv1d'] else 'zeros',
            ),
            'bn1': nn.BatchNorm1d(kwargs['conv1d'][0]['in_channels']),
            'selu1': nn.SELU(),
            'conv2': nn.Conv1d(
                in_channels=kwargs['conv1d'][1]['in_channels'],
                out_channels=kwargs['conv1d'][1]['out_channels'],
                kernel_size=kwargs['conv1d'][1]['kernel_size'],
                stride=kwargs['conv1d'][1]['stride'] if 'conv1d' in kwargs and 'stride' in kwargs['conv1d'][1] else 1,
                padding=kwargs['conv1d'][1]['padding'] if 'conv1d' in kwargs and 'padding' in kwargs['conv1d'][
                    1] else 0,
                dilation=kwargs['conv1d'][1]['dilation'] if 'conv1d' in kwargs and 'dilation' in kwargs['conv1d'][
                    1] else 1,
                groups=kwargs['conv1d'][1]['groups'] if 'conv1d' in kwargs and 'groups' in kwargs['conv1d'][1] else 1,
                bias=kwargs['conv1d'][1]['bias'] if 'conv1d' in kwargs and 'bias' in kwargs['conv1d'][1] else True,
                padding_mode=kwargs['padding_mode'] if 'conv1d' in kwargs and 'padding_mode' in kwargs[
                    'conv1d'] else 'zeros',
            ),
            'bn2': nn.BatchNorm1d(kwargs['conv1d'][1]['in_channels']),
            'selu2': nn.SELU(),
        }))
        self.add_module('decoder', nn.ModuleDict({
            'convtranspose1': nn.ConvTranspose1d(
                in_channels=kwargs['conv1d'][1]['out_channels'],
                out_channels=kwargs['conv1d'][1]['in_channels'],
                kernel_size=kwargs['conv1d'][1]['kernel_size'],
                stride=kwargs['conv1d'][1]['stride'] if 'stride' in kwargs['conv1d'][1] else 1,
                padding=kwargs['conv1d'][1]['padding'] if 'padding' in kwargs['conv1d'][1] else 0,
                dilation=kwargs['conv1d'][1]['dilation'] if 'dilation' in kwargs['conv1d'][1] else 1,
                groups=kwargs['conv1d'][1]['groups'] if 'groups' in kwargs['conv1d'][1] else 1,
                bias=kwargs['conv1d'][1]['bias'] if 'bias' in kwargs['conv1d'][1] else True,
                padding_mode=kwargs['padding_mode'] if 'padding_mode' in kwargs['conv1d'][1] else 'zeros',
            ),
            'selu': nn.SELU(),
            'bn1': nn.BatchNorm1d(kwargs['conv1d'][1]['out_channels']),
            'convtranspose2': nn.ConvTranspose1d(
                in_channels=kwargs['conv1d'][0]['out_channels'],
                out_channels=kwargs['conv1d'][0]['in_channels'],
                kernel_size=kwargs['conv1d'][0]['kernel_size'],
                stride=kwargs['conv1d'][0]['stride'] if 'stride' in kwargs['conv1d'][0] else 1,
                padding=kwargs['conv1d'][0]['padding'] if 'padding' in kwargs['conv1d'][0] else 0,
                dilation=kwargs['conv1d'][0]['dilation'] if 'dilation' in kwargs['conv1d'][0] else 1,
                groups=kwargs['conv1d'][0]['groups'] if 'groups' in kwargs['conv1d'][0] else 1,
                bias=kwargs['conv1d'][0]['bias'] if 'bias' in kwargs['conv1d'][0] else True,
                padding_mode=kwargs['padding_mode'] if 'padding_mode' in kwargs['conv1d'][0] else 'zeros',
            ),
            'bn2': nn.BatchNorm1d(kwargs['conv1d'][0]['out_channels']),
            'sigmoid': nn.Sigmoid()
        }))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> Union[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        return_features = kwargs['return_features'] if 'return_features' in kwargs else False
        childrens = dict(self.named_children())
        features = childrens['encoder']['denoise'](input[0])
        # features = childrens['encoder']['bn1'](features)
        features = childrens['encoder']['conv1'](features)
        # print(features.shape)
        features = childrens['encoder']['selu1'](features)
        # features = childrens['encoder']['bn2'](features)
        features = childrens['encoder']['conv2'](features)
        features = childrens['encoder']['selu2'](features)
        output_features = features
        # features = childrens['decoder']['bn1'](features)
        features = childrens['decoder']['convtranspose1'](features)
        features = childrens['decoder']['selu'](features)
        # features = childrens['decoder']['bn2'](features)
        features = childrens['decoder']['convtranspose2'](features)
        features = childrens['decoder']['sigmoid'](features)
        if return_features:
            return features, output_features
        else:
            return features


class ConvAutoEncoder2LayerLiDeCoSeCoSeCotSeCotSi(nn.Module):
    def __init__(self, **kwargs):
        super(ConvAutoEncoder2LayerLiDeCoSeCoSeCotSeCotSi, self).__init__()
        self.add_module('encoder', nn.ModuleDict({
            'linear': nn.Linear(
                in_features=kwargs['in_features'],
                out_features=kwargs['linear_out_features']
            ),
            'selu0': nn.SELU(),
            'denoise': DenoiseL(kwargs['linear_out_features'],
                                kwargs['denoise']['ratio'] if 'denoise' in kwargs and 'ratio' in kwargs[
                                    'denoise'] else 0.2),
            'conv1': nn.Conv1d(
                in_channels=kwargs['conv1d'][0]['in_channels'],
                out_channels=kwargs['conv1d'][0]['out_channels'],
                kernel_size=kwargs['conv1d'][0]['kernel_size'],
                stride=kwargs['conv1d'][0]['stride'] if 'conv1d' in kwargs and 'stride' in kwargs['conv1d'][0] else 1,
                padding=kwargs['conv1d'][0]['padding'] if 'conv1d' in kwargs and 'padding' in kwargs['conv1d'][
                    0] else 0,
                dilation=kwargs['conv1d'][0]['dilation'] if 'conv1d' in kwargs and 'dilation' in kwargs[
                    'conv1d'] else 1,
                groups=kwargs['conv1d'][0]['groups'] if 'conv1d' in kwargs and 'groups' in kwargs['conv1d'][0] else 1,
                bias=kwargs['conv1d'][0]['bias'] if 'conv1d' in kwargs and 'bias' in kwargs['conv1d'][0] else True,
                padding_mode=kwargs['padding_mode'] if 'conv1d' in kwargs and 'padding_mode' in kwargs[
                    'conv1d'] else 'zeros',
            ),
            # 'bn1': nn.BatchNorm1d(config['conv1d'][0]['in_channels']),
            'selu1': nn.SELU(),
            'conv2': nn.Conv1d(
                in_channels=kwargs['conv1d'][1]['in_channels'],
                out_channels=kwargs['conv1d'][1]['out_channels'],
                kernel_size=kwargs['conv1d'][1]['kernel_size'],
                stride=kwargs['conv1d'][1]['stride'] if 'conv1d' in kwargs and 'stride' in kwargs['conv1d'][1] else 1,
                padding=kwargs['conv1d'][1]['padding'] if 'conv1d' in kwargs and 'padding' in kwargs['conv1d'][
                    1] else 0,
                dilation=kwargs['conv1d'][1]['dilation'] if 'conv1d' in kwargs and 'dilation' in kwargs['conv1d'][
                    1] else 1,
                groups=kwargs['conv1d'][1]['groups'] if 'conv1d' in kwargs and 'groups' in kwargs['conv1d'][1] else 1,
                bias=kwargs['conv1d'][1]['bias'] if 'conv1d' in kwargs and 'bias' in kwargs['conv1d'][1] else True,
                padding_mode=kwargs['padding_mode'] if 'conv1d' in kwargs and 'padding_mode' in kwargs[
                    'conv1d'] else 'zeros',
            ),
            # 'bn2': nn.BatchNorm1d(config['conv1d'][1]['in_channels']),
            'selu2': nn.SELU(),
        }))
        self.add_module('decoder', nn.ModuleDict({
            'convtranspose1': nn.ConvTranspose1d(
                in_channels=kwargs['conv1d'][1]['out_channels'],
                out_channels=kwargs['conv1d'][1]['in_channels'],
                kernel_size=kwargs['conv1d'][1]['kernel_size'],
                stride=kwargs['conv1d'][1]['stride'] if 'stride' in kwargs['conv1d'][1] else 1,
                padding=kwargs['conv1d'][1]['padding'] if 'padding' in kwargs['conv1d'][1] else 0,
                dilation=kwargs['conv1d'][1]['dilation'] if 'dilation' in kwargs['conv1d'][1] else 1,
                groups=kwargs['conv1d'][1]['groups'] if 'groups' in kwargs['conv1d'][1] else 1,
                bias=kwargs['conv1d'][1]['bias'] if 'bias' in kwargs['conv1d'][1] else True,
                padding_mode=kwargs['padding_mode'] if 'padding_mode' in kwargs['conv1d'][1] else 'zeros',
            ),
            'selu1': nn.SELU(),
            # 'bn1': nn.BatchNorm1d(config['conv1d'][1]['out_channels']),
            'convtranspose2': nn.ConvTranspose1d(
                in_channels=kwargs['conv1d'][0]['out_channels'],
                out_channels=kwargs['conv1d'][0]['in_channels'],
                kernel_size=kwargs['conv1d'][0]['kernel_size'],
                stride=kwargs['conv1d'][0]['stride'] if 'stride' in kwargs['conv1d'][0] else 1,
                padding=kwargs['conv1d'][0]['padding'] if 'padding' in kwargs['conv1d'][0] else 0,
                dilation=kwargs['conv1d'][0]['dilation'] if 'dilation' in kwargs['conv1d'][0] else 1,
                groups=kwargs['conv1d'][0]['groups'] if 'groups' in kwargs['conv1d'][0] else 1,
                bias=kwargs['conv1d'][0]['bias'] if 'bias' in kwargs['conv1d'][0] else True,
                padding_mode=kwargs['padding_mode'] if 'padding_mode' in kwargs['conv1d'][0] else 'zeros',
            ),
            # 'bn2': nn.BatchNorm1d(config['conv1d'][0]['out_channels']),
            'selu2': nn.SELU(),
            'linear': nn.Linear(
                in_features=kwargs['linear_out_features'],
                out_features=kwargs['in_features'],
            ),
            'sigmoid': nn.Sigmoid()
        }))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> Union[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor
    ]:
        return_features = kwargs['return_features'] if 'return_features' in kwargs else False
        childrens = dict(self.named_children())
        features = childrens['encoder']['linear'](input[0])
        features = childrens['encoder']['selu0'](features)
        features = childrens['encoder']['denoise'](features)
        # features = childrens['encoder']['bn1'](features)
        features = childrens['encoder']['conv1'](features)
        # print(features.shape)
        features = childrens['encoder']['selu1'](features)
        # features = childrens['encoder']['bn2'](features)
        features = childrens['encoder']['conv2'](features)
        features = childrens['encoder']['selu2'](features)
        output_features = features
        # features = childrens['decoder']['bn1'](features)
        features = childrens['decoder']['convtranspose1'](features)
        features = childrens['decoder']['selu1'](features)
        # features = childrens['decoder']['bn2'](features)
        features = childrens['decoder']['convtranspose2'](features)
        features = childrens['decoder']['selu2'](features)
        features = childrens['decoder']['linear'](features)
        features = childrens['decoder']['sigmoid'](features)
        if return_features:
            return features, output_features
        else:
            return features
