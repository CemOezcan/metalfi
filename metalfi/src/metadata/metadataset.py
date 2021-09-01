from typing import List, Tuple

import pandas as pd

from metalfi.src.metadata.dataset import Dataset


class MetaDataset:

    def __init__(self, dataset: Tuple[pd.DataFrame, str], train: bool = False):
        self.__name = dataset[1]
        self.__meta_data, self.__target_names, self.__times, self.nr_feat, self.nr_inst = \
            self.calculate_training_data(dataset[0]) if train else self.calculate_test_data(dataset[0])

    def get_name(self) -> str:
        return self.__name

    def get_meta_data(self) -> pd.DataFrame:
        return self.__meta_data

    def get_target_names(self) -> List[str]:
        return self.__target_names

    def get_times(self) -> Tuple[float, float]:
        return self.__times

    def get_nrs(self) -> Tuple[int, int]:
        return self.nr_feat, self.nr_inst

    @staticmethod
    def calculate_training_data(dataset: Dataset) -> Tuple[pd.DataFrame, List[str], Tuple[float, float], int, int]:
        data_frames = list()
        nr_feat = 0
        nr_inst = 0

        data, targets, (data_time, target_time), x, y = dataset.train_meta_data()
        nr_feat += x
        nr_inst += y
        times = (data_time, target_time)
        data_frames.append(data)

        return pd.concat(data_frames), targets, times, nr_feat, nr_inst

    @staticmethod
    def calculate_test_data(datasets: List[Dataset]) -> Tuple[pd.DataFrame, None, None, None, None]:
        data_frames = list()
        for dataset in datasets:
            data_frames.append(dataset.test_meta_data())

        return pd.concat(data_frames), None, None, None, None
