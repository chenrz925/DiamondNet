from logging import Logger
from typing import Tuple, Text, List

import torch
from tasker import Profile
from tasker.contrib.torch import SimpleTrainTask
from tasker.storages.basic import Storage
from torch import nn

from models.graph import DiamondNet


def extend_l2(tensor: torch.Tensor):
    batch_size, channels, feature_size = tensor.shape
    assert channels == 3

    return torch.cat((
        torch.sqrt(tensor[:, 0, :] ** 2 + tensor[:, 1, :] ** 2 + tensor[:, 2, :] ** 2).view(batch_size, -1,
                                                                                            feature_size),
        tensor
    ), dim=1)


class DiamondNetCDAETrainTask(SimpleTrainTask):
    def prepare_train_batch(
            self, profile: Profile, shared: Storage, logger: Logger,
            batch: Tuple[torch.Tensor], device: Text, non_blocking: bool = False
    ):
        process_features = profile.process_features if 'process_features' in profile else (lambda it: it)
        features, labels = batch

        return super(DiamondNetCDAETrainTask, self).prepare_train_batch(
            profile, shared, logger,
            (extend_l2(process_features(features)),) * 2, device,
            non_blocking
        )

    def prepare_validate_batch(
            self, profile: Profile, shared: Storage, logger: Logger,
            batch: Tuple[torch.Tensor], device: Text, non_blocking: bool = False
    ):
        process_features = profile.process_features if 'process_features' in profile else (lambda it: it)
        features, labels = batch

        return super(DiamondNetCDAETrainTask, self).prepare_validate_batch(
            profile, shared, logger,
            (extend_l2(process_features(features)),) * 2, device,
            non_blocking
        )


class DiamondNetCDAEGraphWrapper(nn.Module):
    def __init__(self, auto_encoders, process_features, model_profile):
        self.process_features = dict(process_features)

        super(DiamondNetCDAEGraphWrapper, self).__init__()

        self.add_module('auto_encoders', nn.ModuleDict(auto_encoders))
        self.add_module('graph', DiamondNet(model_profile))

    def forward(self, input: List[torch.Tensor]):
        return self.graph(tuple(map(
            lambda it: (it[1](it[2])),
            map(
                lambda it: (it[0], it[1], extend_l2(self.process_features[it[0]](input))),
                self.auto_encoders.items(),
            )
        )))


class DiamondNetGraphTrainTask(SimpleTrainTask):
    def __init__(self, *args, **kwargs):
        self.prefix = 'diamond_net'
        self.auto_encoders = args

    def require(self) -> List[Text]:
        required = super(DiamondNetGraphTrainTask, self).require()
        required.extend(map(
            lambda it: f'{it}_model',
            self.auto_encoders
        ))
        return required

    def create_model(self, profile: Profile, shared: Storage, logger: Logger, **kwargs) -> nn.Module:
        return DiamondNetCDAEGraphWrapper(dict(map(
            lambda it: (it, shared[f'{it}_model']),
            self.auto_encoders
        )), profile.process_features, profile.diamond)
