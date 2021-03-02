import pickle
from functools import reduce
from logging import Logger, getLogger
from multiprocessing.dummy import Pool
from os import path, listdir, makedirs
from random import shuffle
from typing import Text, List, Dict

import numpy as np
import pandas as pd
import torch
from tasker import Definition, Profile, Return
from tasker.contrib.torch import SimpleDataLoaderTask
from tasker.mixin import value, ProfileMixin
from tasker.storages.basic import Storage
from tasker.tasks import Task
from torch.utils import data
from torch.utils.data import Dataset, BatchSampler, RandomSampler


class HTCTMDPrepareTask(Task):
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        cache_root = path.join('.tasker', 'cache', 'htc_tmd', f'{profile.sample_period}_sec')
        labels = [
            'still', 'walk', 'run', 'bicycle',
            'vehicle/motorcycle', 'vehicle/car', 'vehicle/bus',
            'vehicle/MRT', 'vehicle/train', 'vehicle/HSR'
        ]
        pool = Pool(profile.num_workers)
        records = {}

        class IndexGenerator:
            _index = 0

            @classmethod
            def next(cls):
                index = cls._index
                cls._index += 1
                return index

        def dump_record(index, label, arrays):
            for sensor, array in arrays.items():
                target_key = f'{index}_{label.replace("/", "~")}_{sensor}'
                records[target_key] = array
                logger.info(f'Recorded to {target_key}')

        def load_file(label, folder, filename):
            raw_frame = pd.read_csv(path.join(folder, filename), delimiter='\t', header=None)
            raw_frame[-1] = raw_frame[0] // (1_000_000_000 * profile.sample_period)
            sensor_limit = profile.sample_period * 47
            for group_index, period_frame in raw_frame.groupby(-1):
                if period_frame.shape[0] <= sensor_limit * 3:
                    continue
                sensor_map: Dict[Text, pd.DataFrame] = dict(tuple(period_frame.groupby(1)))
                try:
                    target_array = dict(map(
                        lambda it: (
                            it[0][0],
                            it[1].sample(sensor_limit, axis=0).sort_values(0, 0).loc[:,
                            (2, 3, 4)].to_numpy().astype(np.float32).transpose()
                        ),
                        sorted(sensor_map.items(), key=lambda it: it[0])
                    ))
                except ValueError:
                    continue

                dump_record(IndexGenerator.next(), label, target_array)

        def dump_stats(raw_arrays):
            stats_path = path.join(cache_root, 'stats.pkl')
            raw_keys = tuple(raw_arrays.keys())

            raw_arrays = dict(map(
                lambda it: (it, np.concatenate(tuple(map(
                    lambda idx: raw_arrays[idx],
                    sorted(
                        filter(lambda idx: idx.endswith(it), raw_keys),
                        key=lambda idx: int(idx.split('_')[0])
                    )
                )), axis=1)),
                'AMG'
            ))
            stats_arrays = {}
            for sensor, array in raw_arrays.items():
                stats_arrays[f'percentile_{sensor}'] = np.percentile(raw_arrays[sensor], [0, 25, 50, 75, 100], axis=1)
                stats_arrays[f'std_{sensor}'] = array.std(axis=1)
                stats_arrays[f'mean_{sensor}'] = array.mean(axis=1)
                logger.info(f'Stats info of {sensor} computed')

            with open(stats_path, 'wb') as fp:
                pickle.dump(stats_arrays, fp)
            logger.info(f'Stats data dumped to {stats_path}')

        def load_single_label(label: Text):
            data_folder = path.join(profile.root_dir, label)
            logger.info(f'Processing {data_folder}')
            pool.map(
                lambda filename: load_file(label, data_folder, filename),
                sorted(listdir(data_folder))
            )

        tuple(map(load_single_label, labels))
        try:
            makedirs(cache_root)
        except FileExistsError:
            pass

        with open(path.join(cache_root, 'raw.pkl'), 'wb') as fp:
            pickle.dump(records, fp)
        logger.info(f'Dumped to {path.join(cache_root, "raw.pkl")}')
        dump_stats(records)
        index = list(set(map(lambda it: it[:-2], records.keys())))
        index = list(filter(
            lambda idx: reduce(lambda bit, sensor: bit and f'{idx}_{sensor}' in records.keys(), 'AMG', True),
            index
        ))
        shuffle(index)
        np.save(path.join(cache_root, 'index.npy'), np.array(index))

        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return []

    def provide(self) -> List[Text]:
        return []

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('root_dir', str),
            value('num_workers', int),
            value('sample_period', int)
        ]


class HTCTMD(Dataset, ProfileMixin):
    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('cache_dir', str),
            value('num_workers', int),
            value('label_mapping', list, [
                [
                    value('name', list, [str]),
                    value('index', int)
                ]
            ]),
            value('slice', str),
            value('preprocess', str),
        ]

    def __init__(self, cache_dir: Text, num_workers: int, label_mapping, slice, preprocess, **kwargs):
        self.logger = getLogger('datasets.htc_tmd.HTCTMD')
        cache_path = path.join(cache_dir, 'raw.pkl')
        stats_path = path.join(cache_dir, 'stats.pkl')
        index_path = path.join(cache_dir, 'index.npy')

        self.mapped_labels = {}
        for mapping in label_mapping:
            for name in mapping.name:
                self.mapped_labels[name] = mapping.index

        with open(cache_path, 'rb') as fp:
            self.raw_arrays = pickle.load(fp)
        with open(stats_path, 'rb') as fp:
            self.stats_arrays = pickle.load(fp)
        index = np.load(index_path, allow_pickle=True)
        index = np.array(tuple(filter(
            lambda idx: reduce(lambda bit, sensor: bit and f'{idx}_{sensor}' in self.raw_arrays, 'AMG', True),
            index
        )))
        total_length = index.shape[0]
        if slice[0] == '+':
            slice_position = int((1 - float(slice[1:])) * total_length)
            self.index = index[:slice_position]
        elif slice[0] == '-':
            slice_position = int((1 - float(slice[1:])) * total_length)
            self.index = index[slice_position:]
        elif slice == 'none':
            self.index = index
        else:
            raise ValueError('Field "slice" should be started with "+", "-" or "none"')
        self.preprocessor = getattr(self, f'{preprocess}_preprocess', self.robust_preprocess)
        self.tensor = True

    def robust_preprocess(self, sensor, array: np.ndarray):
        per25 = self.stats_arrays[f'percentile_{sensor}'][1].reshape(-1, 1)
        per75 = self.stats_arrays[f'percentile_{sensor}'][3].reshape(-1, 1)
        per_range = per75 - per25
        lower = per25 - 1.5 * per_range
        upper = per75 + 1.5 * per_range
        clipped = np.clip(array, lower, upper)
        return (clipped - lower) / (upper - lower)

    def minmax_preprocess(self, sensor, array: np.ndarray):
        lower = self.stats_arrays[f'percentile_{sensor}'][0].reshape(-1, 1)
        upper = self.stats_arrays[f'percentile_{sensor}'][4].reshape(-1, 1)
        clipped = np.clip(array, lower, upper)
        return (clipped - lower) / (upper - lower)

    def raw_preprocess(self, sensor, array: np.ndarray):
        return array

    def __getitem__(self, item):
        if self.tensor:
            label = torch.tensor([self.mapped_labels[self.index[item].split('_')[-1]]], dtype=torch.int64)
        else:
            label = np.array([self.mapped_labels[self.index[item].split('_')[-1]]], dtype=np.int64)

        if self.tensor:
            def convert(it):
                return torch.from_numpy(it).type(torch.float32)

        else:
            def convert(it):
                return it

        features = tuple(map(
            lambda it: convert(self.preprocessor(it, self.raw_arrays[f'{self.index[item]}_{it}'])),
            'AMG'
        ))
        sequence = [label, *features]
        return dict(map(
            lambda idx: (idx, sequence[idx]),
            range(len(sequence))
        ))

    def __len__(self):
        return self.index.shape[0]


class HTCTMDConcatenated(HTCTMD):
    def __getitem__(self, item):
        sequence_dict = super().__getitem__(item)
        output = torch.cat(tuple(map(lambda idx: sequence_dict[idx], range(1, len(sequence_dict))))), sequence_dict[
            0].item()
        return output


class HTCTMDDataLoaderTask(SimpleDataLoaderTask):
    def create_sampler(
            self, dataset: data.Dataset, batch_sampler: bool, profile: Profile, shared: Storage, logger: Logger
    ):
        assert batch_sampler

        return BatchSampler(
            RandomSampler(dataset),
            batch_size=profile.batch_size,
            drop_last=profile.drop_last
        )
