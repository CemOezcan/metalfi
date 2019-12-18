import pandas as pd


class MetaDataset:

    def __init__(self, datasets):
        self.__datasets = datasets

    def calculateTrainingData(self):
        data_frames = list()
        for dataset in self.__datasets:
            dataset.trainingMetaFeatureVectors()
            data_frames.append(dataset.getMetaData())

        return pd.concat(data_frames)
