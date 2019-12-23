import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import pandas as pd

from metalfi.src.data.metadataset import MetaDataset


class MetaModel:

    def __init__(self, train, test, name):
        # TODO: Parameter optimization?
        self.__test_data = MetaDataset(test)
        self.__train_data = MetaDataset(train, True)
        self.file_name = name

    def save(self):
        # TODO: Implement in Memory class
        model = LinearRegression()
        pickle.dump(model, open(Path(__file__).parents[2] / ("data/model/" + self.file_name), 'wb'))

    def run(self):
        targets = self.__train_data.getTargetNames()
        print(targets)
        y = self.__train_data.getMetaData()[targets]
        X = self.__train_data.getMetaData().drop(targets, axis=1)
        model = RandomForestRegressor()
        model.fit(X, y)

        pd.set_option('display.max_columns', 220)

        for i in range(len(X.columns)):
            print(X.columns[i])
            print(model.feature_importances_[i])

        print(y)
        print(self.__test_data.getMetaData().index)
        print(model.predict(self.__test_data.getMetaData()))
