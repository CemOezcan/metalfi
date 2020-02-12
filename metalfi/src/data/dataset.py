import pandas as pd

from sklearn import preprocessing

from metalfi.src.data.meta.metafeatures import MetaFeatures


class Dataset:

    def __init__(self, data_frame, target):
        self.__data_frame = data_frame
        self.__target = target

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

    # TODO: Rename
    def trainingMetaFeatureVectors(self):
        mf = MetaFeatures(self)
        mf.calculateMetaFeatures()
        targets = mf.createTarget()
        data = mf.getMetaData()

        return data, targets

    def testMetaFeatureVectors(self):
        mf = MetaFeatures(self)
        mf.calculateMetaFeatures()

        return mf.getMetaData()
