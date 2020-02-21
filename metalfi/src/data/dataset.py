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

    def trainMetaData(self):
        mf = MetaFeatures(self)
        mf.calculateMetaFeatures()
        targets = mf.createTarget()
        data = mf.getMetaData()

        return data, targets

    def testMetaData(self):
        mf = MetaFeatures(self)
        mf.calculateMetaFeatures()

        return mf.getMetaData()
