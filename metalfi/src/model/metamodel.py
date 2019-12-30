import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import pandas as pd

from metalfi.src.data.metadataset import MetaDataset


class MetaModel:

    def __init__(self, model, train, name):
        # TODO: Parameter optimization
        self.__train_data = MetaDataset(train, True)
        self.__model = model
        self.file_name = name

    def save(self):
        # TODO: Implement in Memory class
        model = LinearRegression()
        pickle.dump(model, open(Path(__file__).parents[2] / ("data/model/" + self.file_name), 'wb'))

    def train(self):
        targets = self.__train_data.getTargetNames()
        X = self.__train_data.getMetaData().drop(targets, axis=1)

        self.__model = RandomForestRegressor()
        for target in targets:
            y = self.__train_data.getMetaData()[target]
            mean = sum(y) / len(y)

            s_1 = cross_val_score(self.__model, X, y, cv=5, scoring='r2')
            s_2 = cross_val_score(self.__model, X, y, cv=5, scoring='neg_mean_absolute_error')
            s_3 = cross_val_score(self.__model, X, y, cv=5, scoring='neg_mean_squared_error')
            s_4 = cross_val_score(self.__model, X, y, cv=5)

            print(target)
            print(s_1)
            print(set(map(lambda x: x / mean, s_2)))
            print(set(map(lambda x: x / mean, s_3)))
            print(s_4)

    def test(self, test):
        test_data = MetaDataset(test, True)
        targets = test_data.getTargetNames()
        X = test_data.getMetaData().drop(targets, axis=1)

        for target in targets:
            y = test_data.getMetaData()[target]
            self.__model.fit(self.__train_data.getMetaData().drop(targets, axis=1),
                             self.__train_data.getMetaData()[target])

            print(target)
            print(self.__model.score(X, y))
