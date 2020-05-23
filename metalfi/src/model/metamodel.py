import numpy as np
import pandas as pd

from copy import deepcopy
from statistics import mean
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectPercentile
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


class MetaModel:

    def __init__(self, train, name, test, og_data, selected, base_models, target_names):
        self.__og_y = og_data.getDataFrame()[og_data.getTarget()]
        self.__og_X = og_data.getDataFrame().drop(og_data.getTarget(), axis=1)
        self.__base_models = base_models
        self.__target_names = target_names
        self.__selected = selected
        self.__file_name = name
        self.__meta_models = list()
        self.__stats = list()
        self.__results = list()
        self.__result_configurations = list()

        sc1 = StandardScaler()
        sc1.fit(pd.concat([train, test]))
        self.__train_data = DataFrame(data=sc1.transform(train), columns=train.columns)
        self.__test_data = DataFrame(data=sc1.transform(test), columns=test.columns)

        # TODO: get from FeatureImportance class
        train = train.drop(self.__target_names, axis=1)
        fmf = \
            [x for x in train.columns if "." not in x]
        lm = \
            [x for x in train.columns if x.startswith("target_")]
        multi = \
            [x for x in train.columns if x.startswith("multi_")]
        uni = \
            [x for x in train.columns if (x in fmf) and (x not in multi) and (x not in lm)]

        self.__feature_sets = [["Auto"], train.columns, fmf, lm, multi, uni]
        self.__enum = {0: "Auto", 1: "All", 2: "FMF", 3: "LM", 4: "Multi", 5: "Uni"}

    def getName(self):
        return self.__file_name

    def getTargets(self):
        return self.__target_names

    def getStats(self):
        return self.__stats

    def getBaseModels(self):
        return self.__base_models

    def getMetaModels(self):
        return self.__meta_models

    def getEnum(self):
        return self.__enum

    def getResults(self):
        return self.__results

    def getResultConfig(self):
        return self.__result_configurations

    def fit(self):
        X = self.__train_data.drop(self.__target_names, axis=1)

        for base_model, base_model_name in self.__base_models:
            for target in self.__target_names:
                i = 0
                for feature_set in self.__feature_sets:
                    y = self.__train_data[target]
                    X_train, selected_features = self.featureSelection(base_model_name, X, target) \
                        if feature_set[0] == "Auto" else (X[feature_set], feature_set)

                    model = self.hyperparameterOptimization(base_model, X_train, y)
                    feature_set_name = self.__enum.get(i)

                    self.__meta_models.append((deepcopy(model), selected_features,
                                               [base_model_name, target, feature_set_name]))

                    i -= -1

    def featureSelection(self, name, X_train, target_name):
        features = self.__selected[name][target_name]

        return X_train[features], features

    def hyperparameterOptimization(self, model, X, y):
        # TODO: Implement
        model.fit(X, y)
        return model

    def test(self, k, renew=False):
        if renew:
            self.__stats = list()

        if len(self.__stats) == len(self.__meta_models):
            return

        X = self.__test_data.drop(self.__target_names, axis=1)
        for (model, features, config) in self.__meta_models:
            X_test = X[features]
            y_test = self.__test_data[config[1]]
            y_train = self.__train_data[config[1]]
            y_pred = model.predict(X_test)

            r_2 = 1 - (sum([(y_pred[i] - y_test[i]) ** 2 for i in range(len(y_pred))]) /
                       sum([(np.mean(y_train) - y_test[i]) ** 2 for i in range(len(y_pred))]))
            rmse = np.sqrt(np.mean(([(y_pred[i] - y_test[i]) ** 2 for i in range(len(y_pred))])))
            base = np.sqrt(np.mean(([(np.mean(y_train) - y_test[i]) ** 2 for i in range(len(y_pred))])))
            r = np.corrcoef(y_pred, y_test)[0][1]

            if np.math.isnan(r):
                r_2, rmse, base, r = -2, rmse, 1, 0

            self.__stats.append([r_2, rmse / base, r])

    def getRankings(self, columns, prediction, actual):
        pred_data = {"target": prediction, "names": columns}
        act_data = {"target": actual, "names": columns}
        pred = DataFrame(pred_data).sort_values(by=["target"], ascending=False)["names"].values
        act = DataFrame(act_data).sort_values(by=["target"], ascending=False)["names"].values

        return act, pred

    def getOriginalModel(self, name):
        og_model = None

        if name.startswith("RF"):
            og_model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=4)
        elif name.startswith("SVC"):
            og_model = SVC(kernel="rbf", gamma="scale", random_state=0)
        elif name.startswith("LOG"):
            og_model = LogisticRegression(dual=False, solver="lbfgs", multi_class="auto", max_iter=1000, random_state=0,
                                          n_jobs=4)
        elif name.startswith("linSVC"):
            og_model = LinearSVC(max_iter=10000, dual=False, random_state=0)
        elif name.startswith("NB"):
            og_model = GaussianNB()

        return og_model

    def compare(self, models, targets, subsets, k, renew=False):
        if renew:
            self.__results = list()
            self.__result_configurations = list()

        if len(self.__results) == len(models) * len(targets) * len(subsets):
            return

        sc = StandardScaler()
        sc.fit(self.__og_X)
        self.__og_X = DataFrame(data=sc.transform(self.__og_X), columns=self.__og_X.columns)
        X = self.__test_data.drop(self.__target_names, axis=1)

        X_anova_f = SelectPercentile(f_classif, percentile=k).fit_transform(self.__og_X, self.__og_y)
        X_mutual_info = SelectPercentile(mutual_info_classif, percentile=k).fit_transform(self.__og_X, self.__og_y)
        k_number = len(X_anova_f[0])

        cache = {}

        for (model, features, config) in self.__meta_models:
            if config[0] in models and config[1] in targets and config[2] in subsets:
                X_test = X[features]
                y_test = self.__test_data[config[1]]
                y_pred = model.predict(X_test)
                og_model = self.getOriginalModel(config[1])

                a, p = self.getRankings(self.__test_data.index, y_pred, y_test)

                columns = self.__og_X.columns
                predicted = [columns[i] for i in p]
                actual = [columns[i] for i in a]

                X_fi = self.__og_X[actual[:k_number]]
                X_meta_lfi = self.__og_X[predicted[:k_number]]

                key = config[0] + " " + config[1]
                if key in cache:
                    anova_f, mutual_info, fi = cache[key]
                else:
                    anova_f = mean(cross_val_score(og_model, X_anova_f, self.__og_y, cv=5))
                    mutual_info = mean(cross_val_score(og_model, X_mutual_info, self.__og_y, cv=5))
                    fi = mean(cross_val_score(og_model, X_fi, self.__og_y, cv=5))
                    cache[key] = (anova_f, mutual_info, fi)

                meta_lfi = mean(cross_val_score(og_model, X_meta_lfi, self.__og_y, cv=5))

                self.__result_configurations.append(config)
                self.__results.append([anova_f, mutual_info, fi, meta_lfi])

        return ["ANOVA", "MI", "FI", "MetaLFI"]
