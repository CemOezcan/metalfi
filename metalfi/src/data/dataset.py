import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

from metalfi.src.data.meta.importance.permutation import PermutationImportance


class Dataset:

    def __init__(self, data_frame, target):
        self.__data_frame = data_frame
        self.__target = target
        self.scale()
        self.__meta_feature_vectors = list()
        self.__correlation_matrix = self.__data_frame.corr()
        plt.matshow(self.__correlation_matrix)
        plt.show()

    def getCorrelationMatrix(self):
        return self.__correlation_matrix

    def getDataFrame(self):
        return self.__data_frame

    def getTarget(self):
        return self.__target

    def splitDataRandom(self):
        return

    def scale(self):
        values = self.__data_frame.values
        values_scaled = preprocessing.MinMaxScaler().fit_transform(values)

        self.__data_frame = pd.DataFrame(values_scaled, columns=self.__data_frame.columns)

    def calculateMetaFeatureVectors(self):
        perm = PermutationImportance(self)
        perm.calculateScores()
