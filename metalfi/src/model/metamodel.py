import pickle
from statistics import mean

import matplotlib.pyplot as plt

from pathlib import Path
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from metalfi.src.data.metadataset import MetaDataset


class MetaModel:

    def __init__(self, train, name):
        # TODO: Parameter optimization,
        #  more models (change implementation of feature selection for non-linear and non-tree-based models)
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

            # print(target)
            # print(s_1)
            # print(set(map(lambda x: x / mean, s_2)))
            # print(set(map(lambda x: x / mean, s_3)))

    def test(self, test, scale):
        test_data = MetaDataset(test, True).getMetaData()
        X_og = test[0].getDataFrame().drop(test[0].getTarget(), axis=1)
        y_og = test[0].getDataFrame()[test[0].getTarget()]
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
            act, pred = self.compareRankings(test_data.index, self.__model.predict(X_test_new), y_test)
            print("%s%s%s%s" % ("optimal \n", act, "\n predicted \n", pred))

            if target.startswith("rf"):
                model = RandomForestClassifier()
            elif target.startswith("svc"):
                model = SVC()
            elif target.startswith("log"):
                model = LogisticRegression()
            elif target.startswith("lin"):
                model = LinearSVC()
            else:
                model = LinearRegression()

            self.calculatePerformance(model, X_og, y_og, pred, act, 3)

    def compareRankings(self, columns, prediction, actual, depth=None):
        pred_data = {"target": prediction, "names": columns}
        act_data = {"target": actual, "names": columns}
        pred = DataFrame(pred_data).sort_values(by=["target"], ascending=False)["names"].values
        act = DataFrame(act_data).sort_values(by=["target"], ascending=False)["names"].values

        depth = len(act) if depth is None else depth
        sum = 0

        for i in range(1, depth + 1):
            set_1, set_2 = set(pred[:i]), set(act[:i])
            sum += len(set_1.intersection(set_2)) / i

        print(sum / depth)
        return act, pred

    def calculatePerformance(self, model, X, y, predicted, actual, k):
        # TODO: Parameter optimization
        X_chi_2 = SelectKBest(chi2, k=k).fit_transform(X, y)
        X_anova_f = SelectKBest(f_classif, k=k).fit_transform(X, y)
        X_mutual_info = SelectKBest(mutual_info_classif, k=k).fit_transform(X, y)
        X_fi = X[actual[:k]]
        X_meta_lfi = X[predicted[:k]]

        print("%s%s" % ("Chi² Stats \n", mean(cross_val_score(model, X_chi_2, y, cv=5))))
        print("%s%s" % ("ANOVA F-Value \n", mean(cross_val_score(model, X_anova_f, y, cv=5))))
        print("%s%s" % ("Mutual Information \n", mean(cross_val_score(model, X_mutual_info, y, cv=5))))
        print("%s%s" % ("Feature Importance \n", mean(cross_val_score(model, X_fi, y, cv=5))))
        print("%s%s" % ("Meta-Learning Feature Importance \n", mean(cross_val_score(model, X_meta_lfi, y, cv=5))))
