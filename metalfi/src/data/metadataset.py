import pandas as pd


class MetaDataset:

    def __init__(self, datasets, train=False):
        self.__meta_data, self.__target_names, self.__times, self.nr_feat, self.nr_inst = \
            self.calculateTrainingData(datasets) if train else self.calculateTestData(datasets)

    def getMetaData(self):
        return self.__meta_data

    def getTargetNames(self):
        return self.__target_names

    def getTimes(self):
        return self.__times

    def getNrs(self):
        return self.nr_feat, self.nr_inst

    @staticmethod
    def calculateTrainingData(datasets):
        data_frames = list()
        times = None
        targets = None
        nr_feat = 0
        nr_inst = 0

        for dataset in datasets:
            data, targets, (data_time, target_time), x, y = dataset.trainMetaData()
            nr_feat += x
            nr_inst += y
            times = (data_time, target_time)
            data_frames.append(data)

        return pd.concat(data_frames), targets, times, nr_feat, nr_inst

    @staticmethod
    def calculateTestData(datasets):
        data_frames = list()
        for dataset in datasets:
            data_frames.append(dataset.testMetaData())

        return pd.concat(data_frames), None, None, None, None
