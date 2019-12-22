import pandas as pd


class MetaDataset:

    def __init__(self, datasets, train=False):
        data_frames = list()
        for dataset in datasets:
            data_frames.append(dataset.testMetaFeatureVectors())

        self.__meta_data, self.__target_names = \
            self.calculateTrainingData(datasets) if train else self.calculateTestData(datasets)

    def getMetaData(self):
        return self.__meta_data

    def getTargetNames(self):
        return self.__target_names

    @staticmethod
    def calculateTrainingData(datasets):
        data_frames = list()
        targets = None
        for dataset in datasets:
            data, targets = dataset.trainingMetaFeatureVectors()
            data_frames.append(data)

        return pd.concat(data_frames), targets

    @staticmethod
    def calculateTestData(datasets):
        data_frames = list()
        for dataset in datasets:
            data_frames.append(dataset.testMetaFeatureVectors())

        return pd.concat(data_frames), None
