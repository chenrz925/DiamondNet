from collections import OrderedDict
from functools import reduce
from itertools import product
from math import floor
from typing import Dict, Text, Any, List

import torch
from torch import nn
from torch.nn import functional as F

from .attention import SqueezeExcitation


class _CommonConv1d(nn.Module):
    @staticmethod
    def _padding_size(kernel_size):
        return int(floor((kernel_size - 1) / 2))

    def __init__(self, out_channels: int, kernel_size: int, bias: bool = True):
        super(_CommonConv1d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.register_parameter('weight', nn.Parameter(torch.rand(out_channels, 1, kernel_size)))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.rand(out_channels)))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        parameters = dict(self.named_parameters())
        output = torch.cat(list(map(
            lambda t: t.sum(dim=1).reshape(t.size(0), 1, t.size(2)),
            map(
                lambda o: torch.cat(list(map(
                    lambda c: F.conv1d(
                        input=input[0][:, c, :].reshape(input[0].size(0), 1, input[0].size(2)),
                        weight=parameters['weight'][o, :, :].reshape(1, *parameters['weight'].shape[1:]),
                        bias=None,
                        padding=self._padding_size(self.kernel_size)
                    ),
                    range(input[0].size(1)),
                )), dim=1),
                range(self.out_channels)
            ),
        )), dim=1)
        if 'bias' in parameters:
            output += parameters['bias'].reshape(1, parameters['bias'].size(0), 1).expand_as(output)
        return output


class AdjacencyMatrix(nn.Module):
    @staticmethod
    def _padding_size(kernel_size):
        return int(floor((kernel_size - 1) / 2))

    def __init__(self, profile):
        super(AdjacencyMatrix, self).__init__()
        self.vertexes = profile['vertexes']
        self.add_module('conv', _CommonConv1d(
            out_channels=profile['conv']['out_channels'],
            kernel_size=profile['conv']['kernel_size'],
        ))
        se_class = profile.squeeze_excitation.reference  # import_class(profile['squeeze_excitation']['module'], profile['squeeze_excitation']['class'],
        # nn.Module)
        self.add_module('squeeze_excitation', nn.ModuleList(map(
            lambda v: se_class(**profile['squeeze_excitation']['kwargs'][v]),
            range(self.vertexes),
        )))
        self.add_module('compress', nn.AdaptiveAvgPool1d(1))
        self.add_module('self_attention', nn.Linear(
            bias=False,
            in_features=2 * profile['conv']['out_channels'],
            out_features=1,
        ) if (
            profile['enable_self_attention'] if 'enable_self_attention' in profile else True) else Identity())
        self.add_module('activation', nn.ModuleDict(OrderedDict(
            conv=nn.SELU(),
            self_attention=nn.SELU(),
        )))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> List[torch.Tensor]:
        children = OrderedDict(self.named_children())
        conv_outputs = list(map(
            lambda v: children['activation']['conv'](children['conv'](input[0][v])),  # (N, C, L)
            range(self.vertexes),
        ))
        compressed_outputs = list(map(
            lambda s_e: children['compress'](s_e),  # (N, C, 1)
            map(
                lambda v: children['squeeze_excitation'][v](conv_outputs[v]) * conv_outputs[v],  # (N, C, L)
                range(self.vertexes),
            )
        ))
        attention_weights = dict(map(
            lambda e_d: (e_d[0], children['activation']['self_attention'](children['self_attention'](e_d[1]))),
            # (N, 1)
            map(
                lambda e_d: (e_d[0], e_d[1].reshape(*e_d[1].shape[:-1])),
                map(
                    lambda e: (e, torch.cat((compressed_outputs[e[0]], compressed_outputs[e[1]]), dim=1)),  # (N, 2C, 1)
                    product(range(self.vertexes), range(self.vertexes)),
                )
            )
        ))
        softmax_attention_weights = list(map(
            lambda a: F.softmax(a, 1),  # (N, V_f)
            map(
                lambda v_t: torch.cat(list(map(
                    lambda v_f: attention_weights[v_f, v_t],
                    range(self.vertexes),
                )), dim=1),
                range(self.vertexes),
            )
        ))
        outputs = list(map(
            lambda v_t: sum(map(
                lambda v_f: softmax_attention_weights[v_t][:, v_f]
                                .reshape(softmax_attention_weights[v_t].size(0), 1)
                                .repeat((1, conv_outputs[v_f].size(1)))
                                .reshape(*conv_outputs[v_f].shape[:-1], 1) * conv_outputs[v_f],
                # (N, C, 1) * (N, C, L)
                range(self.vertexes),
            )),
            range(self.vertexes),
        ))
        return outputs


class OutputAdjacencyMatrix(nn.ModuleList):
    """
    SpiderLayer 输出虚拟节点邻接矩阵实现
    参数结构
    |- vertexes: int, 图结点个数
    |- model_module: Text, Attention模型所在包路径
    |- model_class: Text, Attention模型包名
    |-+kwargs: List[Dict[Text, Any]], 每个Attention模型的构造参数
      |- [from_vertex]: Dict[Text, Any], Attention模型构造参数
    """

    def __init__(self, profile):
        """
        :param profile: 所有输入参数
        """
        self.vertexes: int = profile['vertexes']
        self.softmax: bool = profile['softmax']
        # self._counter = TensorCollector.counter()
        # self._collector = TensorCollector(f'{self._counter}_outadjmat')
        model_class: nn.Module = profile.model_reference
        model_array = []
        for from_vertex in range(self.vertexes):
            model = model_class(**profile['kwargs'][from_vertex] if 'kwargs' in profile else {})
            model_array.append(model)
        super(OutputAdjacencyMatrix, self).__init__(model_array)

    def __delitem__(self, key):
        raise RuntimeError(f'It\'s not permitted to delete item in {self.__class__.__name__}.')

    def __setitem__(self, key, value):
        raise RuntimeError(f'It\'s not permitted to modify item in {self.__class__.__name__}.')

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> List[torch.Tensor]:
        """
        :param input: 模型输入，多传感器的张量序列
        :return: 模型输出，多传感器张量序列
        """
        se_outputs = list(map(
            lambda vertex: self[vertex](input[0][vertex]),
            range(self.vertexes),
        ))
        return [
            se_outputs[vertex] * input[0][vertex]
            for vertex in range(self.vertexes)
        ]


class SpiderLayer(nn.Module):
    """
    参数结构
    |- update_module: Text, 更新函数类所在包名
    |- update_class: Text, 更新函数类类名
    |- adjacency: Dict, 邻接矩阵参数字典
    |- update: Dict, 更新函数参数字典
    """

    def __init__(self, profile):
        """
        :param profile: 所有输入参数
        """
        super(SpiderLayer, self).__init__()
        update_class = profile.update_reference  # import_class(profile['update_module'], profile['update_class'])
        adjacency_class = profile.adjacency_reference  # import_class(
        # profile['adjacency_module'] if 'adjacency_module' in profile else 'models.graph',
        # profile['adjacency_class'] if 'adjacency_class' in profile else 'AdjacencyMatrix')
        self.add_module('adjacency', adjacency_class(profile['adjacency']))
        self.add_module('update', update_class(profile['update']))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        children = dict(self.named_children())
        return children['update'](children['adjacency'](*input))


class SpiderOutputLayer(nn.Module):
    def __init__(self, profile):
        """
        :param profile: 所有输入参数
        """
        super(SpiderOutputLayer, self).__init__()
        update_class = profile.update_reference  # import_class(profile['update_module'], profile['update_class'])

        self.add_module('adjacency', OutputAdjacencyMatrix(profile['adjacency']))
        self.add_module('update', update_class(profile['update']))

    def forward(self, *input: Any, **kwargs: Any) -> torch.Tensor:
        children = dict(self.named_children())
        return children['update'](children['adjacency'](*input))


class DiamondNet(nn.Module):
    """
    参数结构
    |-+model: List[Dict[Text, Any]], 分层模型参数字典
      |- [index]: Dict[Text, Any], 参数字典
    """

    def _out_features(self, profile) -> int:
        out_channels = None
        for index in range(len(profile['layer'])):
            layer = profile['layer'][index]
            in_channels = list(map(lambda kw: kw['in_channels'], layer['adjacency']['kwargs']))
            if out_channels is not None:
                if not reduce(
                        lambda v, s: v and s,
                        map(
                            lambda calc, input: calc == input,
                            out_channels,
                            in_channels
                        ),
                        True
                ):
                    raise RuntimeError(
                        'Calculated adjacency model\'s in_channels mismatch with configured value in TOML.'
                    )
            adjacency_channels = [sum(map(lambda kw: kw['in_channels'], layer['adjacency']['kwargs']))] * \
                                 layer['adjacency']['vertexes']
            update_in_channels = list(map(lambda conv: conv['in_channels'], layer['update']['conv']))
            if index != len(profile['layer']) - 1 and not reduce(
                    lambda v, s: v and s,
                    map(
                        lambda calc, input: calc == input,
                        adjacency_channels,
                        update_in_channels
                    ),
                    True
            ):
                raise RuntimeError('Calculated update model\'s in_channels mismatch with configured value in TOML.')
            elif index == len(profile['layer']) - 1 and not reduce(
                    lambda v, s: v and s,
                    map(
                        lambda calc, input: calc == input,
                        in_channels,
                        update_in_channels
                    ),
                    True
            ):
                raise RuntimeError('Calculated update model\'s in_channels mismatch with configured value in TOML.')

            if out_channels is None and not self.dense:
                out_channels = list(map(lambda conv: conv['growth_ratio'], layer['update']['conv']))
            elif self.dense:
                out_channels = list(map(
                    lambda i, o: i + o,
                    in_channels,
                    map(lambda conv: conv['growth_ratio'], layer['update']['conv'])
                ))
            else:
                out_channels = list(map(lambda conv: conv['growth_ratio'], layer['update']['conv']))
        return sum(out_channels) * self.in_features

    def __init__(self, profile):
        super(DiamondNet, self).__init__()
        index = 0
        self.dense = profile['dense']
        self.in_features = profile['in_features']
        self.output_attention = profile['output_attention'] if 'output_attention' in profile else False

        model_configs = profile['layer']
        layer_list = []
        for model_config in model_configs:
            if index == len(model_configs) - 1:
                model_class = SpiderOutputLayer
                layer_list.append(model_class(model_config))
            else:
                model_class = SpiderLayer
                layer_list.append(model_class(model_config))
            index += 1
        self.add_module('layers', nn.ModuleList(layer_list))
        if self.output_attention:
            self.add_module('attention', SqueezeExcitation(
                in_channels=profile['attention_in_channels'],
                reduction_ratio=profile['attention_reduction_ratio']
            ))
        self.add_module('pool', nn.AdaptiveMaxPool1d(
            profile['adaptive_linear_features'] if 'adaptive_linear_features' in profile else 2048
        ))
        # self.add_module('pool1', nn.AdaptiveMaxPool1d(
        #     512
        # ))
        self.add_module('linear', nn.Sequential(
            nn.Linear(
                in_features=profile['adaptive_linear_features'] if 'adaptive_linear_features' in profile else 2048,
                out_features=profile['linear'][0]['out_features'],
                # out_features=2048,
            ),
            # nn.SELU(),
            # nn.Linear(
            # in_features=2048,
            # out_features=1024,
            # ),
            nn.SELU(),
        ))
        self.add_module('output', nn.Linear(
            in_features=profile['linear'][0]['out_features'],
            # in_features=2048,
            out_features=profile['class_number'],
        ))

    def forward(self, *input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        childrens: Dict[Text, nn.Module] = dict(self.named_children())
        features = input[0]
        first = True
        for child in childrens['layers'].children():
            if self.dense and not first:
                dense_features = list(map(
                    lambda raw, feature: torch.cat((raw, feature), dim=1),
                    dense_features,
                    features
                ))
                features = child(dense_features)
            else:
                if self.dense:
                    dense_features = features
                features = child(features)
            first = False
        if self.dense:
            if self.output_attention:
                output_features = torch.cat(
                    list(map(
                        lambda raw, feature: torch.cat((raw, feature), dim=1),
                        dense_features,
                        features
                    )), dim=1
                )
                output_features_weight = childrens['attention'](output_features)
                # TODO: dump this attention weight. Different layers' outputs concatenated by channels should be calculated to mean value, for example, 3 layers to 3 points. 3 points concantented by 3 vertexes = 9 points.
                output_features = output_features * output_features_weight
            else:
                output_features = torch.cat(
                    list(map(
                        lambda t: t.reshape(t.size(0), -1),
                        map(
                            lambda raw, feature: torch.cat((raw, feature), dim=1),
                            dense_features,
                            features
                        )
                    )), dim=1
                )

            y = childrens['pool'](output_features.reshape(output_features.size(0), 1, -1)).reshape(
                output_features.size(0), -1)
            y = childrens['linear'](y)
            # fuck = childrens['pool1'](fuck.reshape(fuck.size(0), 1, -1)).reshape(fuck.size(0), -1)
            y = childrens['output'](y)
            return y
        else:
            features = torch.cat(
                list(map(
                    lambda t: t.reshape(t.size(0), -1),
                    features
                )), dim=1
            )
            if self.output_attention:
                features_weight = childrens['attention'](features)
                features_weight = features_weight.reshape(*features_weight.size(), 1)
                features = features * features_weight

            y = childrens['pool'](features.reshape(features.size(0), 1, -1)).reshape(features.size(0), -1)
            y = childrens['linear'](y)
            y = childrens['output'](y)
            return y
