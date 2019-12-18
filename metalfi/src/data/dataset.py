import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from sklearn import preprocessing

from metalfi.src.data.meta.importance.dropcolumn import DropColumnImportance
from metalfi.src.data.meta.importance.permutation import PermutationImportance
from metalfi.src.data.meta.importance.shap import ShapImportance
from metalfi.src.data.meta.metafeatures import MetaFeatures


class Dataset:

    def __init__(self, data_frame, target):
        self.__data_frame = data_frame
        self.__target = target
        #self.scale()
        self.__meta_data = DataFrame()
        self.__correlation_matrix = self.__data_frame.corr()

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

    def trainingMetaFeatureVectors(self):
        mf = MetaFeatures(self)
        mf.calculateMetaFeatures()
        mf.createTarget()
        self.__meta_data = mf.getMetaData()

    def testMetaFeatureVectors(self):
        mf = MetaFeatures(self)
        mf.calculateMetaFeatures()
        self.__meta_data = mf.getMetaData()

