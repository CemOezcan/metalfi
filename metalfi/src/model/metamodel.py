import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from metalfi.src.data.metadataset import MetaDataset


class MetaModel:

    def __init__(self, train, name):
        # TODO: Parameter optimization
        data = MetaDataset(train, True)
        self.__train_data = data.getMetaData()
        self.__targets = data.getTargetNames()
        self.__model = RandomForestRegressor(n_estimators=100)
        self.file_name = name

    def save(self):
        # TODO: Implement in Memory class
        model = LinearRegression()
        pickle.dump(model, open(Path(__file__).parents[2] / ("data/model/" + self.file_name), 'wb'))

    def train(self, scale):
        X = self.__train_data.drop(self.__targets, axis=1)

        if scale:
            sc = StandardScaler()
            X = sc.fit_transform(X)

        for target in self.__targets:
            y = self.__train_data[target]
            mean = sum(y) / len(y)

            s_1 = cross_val_score(self.__model, X, y, cv=6, scoring='r2')
            s_2 = cross_val_score(self.__model, X, y, cv=6, scoring='neg_mean_absolute_error')
            s_3 = cross_val_score(self.__model, X, y, cv=6, scoring='neg_mean_squared_error')

            print(target)
            print(s_1)
            print(set(map(lambda x: x / mean, s_2)))
            print(set(map(lambda x: x / mean, s_3)))

    def test(self, test, scale):
        test_data = MetaDataset(test, True).getMetaData()
        X_test = test_data.drop(self.__targets, axis=1)
        X_train = self.__train_data.drop(self.__targets, axis=1)

        if scale:
            sc_X = StandardScaler()
            X_test = sc_X.fit_transform(X_test)
            X_train = sc_X.fit_transform(X_train)

        for target in self.__targets:
            y_test = test_data[target]
            y_train = self.__train_data[target]
            self.__model.fit(X_train, y_train)

            model = SelectFromModel(self.__model, prefit=True)
            X_train_new = model.transform(X_train)
            X_test_new = model.transform(X_test)

            self.__model.fit(X_train_new, y_train)
            print(target)
            print(self.__model.score(X_test_new, y_test))
            print(self.__model.predict(X_test_new))
