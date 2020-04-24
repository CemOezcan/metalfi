import pandas as pd


class MetaDataset:

    def __init__(self, datasets, train=False):
        data_frames = list()
        for dataset in datasets:
            data_frames.append(dataset.testMetaData())

        self.__meta_data, self.__target_names, self.__times = \
            self.calculateTrainingData(datasets) if train else self.calculateTestData(datasets)

    def getMetaData(self):
        return self.__meta_data

    def getTargetNames(self):
        return self.__target_names

    def getTimes(self):
        return self.__times

    @staticmethod
    def calculateTrainingData(datasets):
        data_frames = list()
        times = None
        targets = None
        for dataset in datasets:
            data, targets, (data_time, target_time) = dataset.trainMetaData()
            times = (data_time, target_time)
            data_frames.append(data)

        return pd.concat(data_frames), targets, times

    @staticmethod
    def calculateTestData(datasets):
        data_frames = list()
        for dataset in datasets:
            data_frames.append(dataset.testMetaData())

        return pd.concat(data_frames), None, None
