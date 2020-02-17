import numpy as np

from copy import deepcopy
from statistics import mean
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class MetaModel:

    def __init__(self, train, name, test):
        # TODO: Parameter optimization,
        #  more models (change implementation of feature selection for non-linear and non-tree-based models) and
        #  do not store datasets (ids should be sufficient)

        self.__base_models = [(RandomForestRegressor(n_estimators=100), "Rf", "RMSE")]
        self.__train_data = train
        self.__test_data = test
        # TODO: get from FeatureImportance class
        self.__target_names = ["rf_perm", "linSVC_perm", "svc_perm", "log_perm", "rf_dropCol", "linSVC_dropCol",
                               "svc_dropCol", "log_dropCol", "rf_shap", "linSVC_shap", "log_shap"]
        train = train.drop(self.__target_names, axis=1)
        self.__feature_sets = [["Auto"], train.columns,
                               [x for x in train.columns if "." not in x],
                               [x for x in train.columns if x.startswith("target")],
                               [x for x in train.columns if not x.startswith("target")]]

        # Name of the test dataset + information about whether features are independent or not
        self.__file_name = name
        self.__meta_models = list()

    def fit(self):
        X = self.__train_data.drop(self.__target_names, axis=1)

        for base_model, base_model_name, metric in self.__base_models:
            for target in self.__target_names:
                i = 0
                for feature_set in self.__feature_sets:
                    y = self.__train_data[target]
                    # TODO: a little bit of hyperparameter optimization before selecting features
                    X_train, selected_features = self.featureSelection(base_model, X, y) \
                        if feature_set[0] == "Auto" else (X[feature_set], feature_set)

                    model, scale = self.hyperparameterOptimization(base_model, metric, X_train, y)
                    enum = {0: "Auto", 1: "All", 2: "FMF", 3: "LM", 4: "NoLM"}
                    feature_set_name = enum.get(i)

                    self.__meta_models.append((deepcopy(model), selected_features,
                                               [base_model_name, metric, target, feature_set_name], scale))

                    i -= -1

    def featureSelection(self, base_model, X_train, y_train):
        base_model.fit(X_train, y_train)

        selector = SelectFromModel(base_model, prefit=True)
        support = selector.get_support(indices=True)

        features = [x for x in list(X_train.columns) if list(X_train.columns).index(x) in support]
        X_selected = selector.transform(X_train)

        return X_selected, features

    def hyperparameterOptimization(self, model, metric, X, y):
        # TODO: Implement
        model.fit(X, y)
        scale = False
        return model, scale

    def test(self):
        X = self.__test_data.drop(self.__target_names, axis=1)
        stats = list()

        for (model, features, config, scale) in self.__meta_models:
            sc = StandardScaler()
            X_test = sc.fit_transform(X[features]) if scale else X[features]

            y_test = self.__test_data[config[2]]
            y_train = self.__train_data[config[2]]
            y_pred = model.predict(X_test)

            name = config
            r_2 = model.score(X_test, y_test)
            rmse = np.sqrt(np.mean(([(y_pred[i] - y_test[i]) ** 2 for i in range(len(y_pred))])))
            base = np.sqrt(np.mean(([(np.mean(y_train) - y_test[i]) ** 2 for i in range(len(y_pred))])))
            r = np.corrcoef(y_pred, y_test)[0][1]
            rho = spearmanr(y_pred, y_test)[0]

            stats.append((name, r_2, rmse, base, r, rho))

        return stats

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

        # print(sum / depth)
        return act, pred

    def calculatePerformance(self, model, X, y, predicted, actual, k, svc):
        # X_chi_2 = SelectKBest(chi2, k=k).fit_transform(X, y)
        X_anova_f = SelectKBest(f_classif, k=k).fit_transform(X, y)
        X_mutual_info = SelectKBest(mutual_info_classif, k=k).fit_transform(X, y)
        X_fi = X[actual[:k]]
        X_meta_lfi = X[predicted[:k]]

        if svc:
            sc_X = StandardScaler()
            X_anova_f = sc_X.fit_transform(X_anova_f)
            X_mutual_info = sc_X.fit_transform(X_mutual_info)
            X_fi = sc_X.fit_transform(X_fi)
            X_meta_lfi = sc_X.fit_transform(X_meta_lfi)

        # print("%s%s" % ("Chi² Stats \n", mean(cross_val_score(model, X_chi_2, y, cv=5))))
        print("%s%s" % ("ANOVA F-Value \n", mean(cross_val_score(model, X_anova_f, y, cv=5))))
        print("%s%s" % ("Mutual Information \n", mean(cross_val_score(model, X_mutual_info, y, cv=5))))
        print("%s%s" % ("Feature Importance \n", mean(cross_val_score(model, X_fi, y, cv=5))))
        print("%s%s" % ("Meta-Learning Feature Importance \n", mean(cross_val_score(model, X_meta_lfi, y, cv=5))))
