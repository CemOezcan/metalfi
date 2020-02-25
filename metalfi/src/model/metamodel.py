import numpy as np
import sklearn.metrics
import pandas as pd

import sklearn.preprocessing as preprocessing
from copy import deepcopy
from statistics import mean
from pandas import DataFrame
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR


class MetaModel:

    def __init__(self, train, name, test, og_data):
        self.__og_y = og_data.getDataFrame()[og_data.getTarget()]
        self.__og_X = og_data.getDataFrame().drop(og_data.getTarget(), axis=1)

        self.__base_models = [(RandomForestRegressor(n_estimators=100, n_jobs=4), "Rf", "RMSE"),
                              (SVR(), "Svr", "RMSE"),
                              (LinearRegression(n_jobs=4), "lin", "RMSE"),
                              (LinearSVR(dual=True, max_iter=10000), "linSVR", "RMSE")]

        sc1 = StandardScaler()
        sc1.fit(pd.concat([train, test]))
        self.__train_data = DataFrame(data=sc1.transform(train), columns=train.columns)
        self.__test_data = DataFrame(data=sc1.transform(test), columns=test.columns)

        # TODO: get from FeatureImportance class
        self.__target_names = ["lda_shap", "linSVC_shap", "log_shap", "rf_shap", "nb_shap", "svc_shap",
                               "lda_lime", "linSVC_lime", "log_lime", "rf_lime", "nb_lime", "svc_lime",
                               "lda_perm", "linSVC_perm", "log_perm", "rf_perm", "nb_perm", "svc_perm",
                               "lda_dCol", "linSVC_dCol", "log_dCol", "rf_dCol", "nb_dCol", "svc_dCol"]

        train = train.drop(self.__target_names, axis=1)
        fmf = [x for x in train.columns if "." not in x]
        lm = [x for x in train.columns
              if (x.startswith("target") or x == "joint_ent" or x == "mut_inf" or x == "var_importance")]
        no_lm = [x for x in train.columns if (x not in lm)]

        self.__feature_sets = [["Auto"], train.columns, fmf, lm, no_lm]
        self.__enum = {0: "Auto", 1: "All", 2: "FMF", 3: "LM", 4: "NoLM"}

        # Name of the test dataset + information about whether features are independent or not
        self.__file_name = name
        self.__meta_models = list()
        self.__stats = list()

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

    def fit(self):
        X = self.__train_data.drop(self.__target_names, axis=1)

        for base_model, base_model_name, metric in self.__base_models:
            for target in self.__target_names:
                i = 0
                for feature_set in self.__feature_sets:
                    y = self.__train_data[target]
                    X_train, selected_features = self.featureSelection(base_model, base_model_name, X, y) \
                        if feature_set[0] == "Auto" else (X[feature_set], feature_set)

                    model, scale = self.hyperparameterOptimization(base_model, metric, X_train, y)
                    feature_set_name = self.__enum.get(i)

                    self.__meta_models.append((deepcopy(model), selected_features,
                                               [base_model_name, metric, target, feature_set_name], scale))

                    i -= -1

    def featureSelection(self, base_model, name, X_train, y_train):
        # TODO: Implement for SVR
        if name == "Svr":
            return X_train, self.__feature_sets[1]

        base_model.fit(X_train, y_train)
        """selector = RFECV(base_model, step=1, cv=5, scoring=make_scorer(sklearn.metrics.mean_squared_error,
                                                                       greater_is_better=False))
        selector = selector.fit(X_train, y_train)"""

        selector = SelectFromModel(base_model, prefit=True)
        support = selector.get_support(indices=True)

        features = [x for x in list(X_train.columns) if list(X_train.columns).index(x) in support]
        X_selected = selector.transform(X_train)

        return X_selected, features

    def hyperparameterOptimization(self, model, metric, X, y):
        # TODO: Implement
        model.fit(X, y)
        scale = True
        return model, scale

    def test(self, k):
        X = self.__test_data.drop(self.__target_names, axis=1)

        for (model, features, config, scale) in self.__meta_models:
            X_test = X[features]
            y_test = self.__test_data[config[2]]
            y_train = self.__train_data[config[2]]
            y_pred = model.predict(X_test)

            if config[2].startswith("rf"):
                og_model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=4)
            elif config[2].startswith("svc"):
                og_model = SVC(kernel="rbf", gamma="scale", random_state=0)
            elif config[2].startswith("log"):
                og_model = LogisticRegression(dual=False, solver="lbfgs", multi_class="auto", max_iter=1000, random_state=0, n_jobs=4)
            elif config[2].startswith("lin"):
                og_model = LinearSVC(max_iter=10000, dual=False, random_state=0)
            else:
                og_model = LinearRegression()

            r_2 = 1 - (sum([(y_pred[i] - y_test[i]) ** 2 for i in range(len(y_pred))]) /
                       sum([(np.mean(y_train) - y_test[i]) ** 2 for i in range(len(y_pred))]))
            rmse = np.sqrt(np.mean(([(y_pred[i] - y_test[i]) ** 2 for i in range(len(y_pred))])))
            base = np.sqrt(np.mean(([(np.mean(y_train) - y_test[i]) ** 2 for i in range(len(y_pred))])))
            r = np.corrcoef(y_pred, y_test)[0][1]
            rho = spearmanr(y_pred, y_test)[0]

            a, p = self.getRankings(self.__test_data.index, y_pred, y_test)
            anova_f, mutual_info, fi, meta_lfi = self.compare(og_model, self.__og_X, self.__og_y, p, a, k)

            self.__stats.append([anova_f, mutual_info, fi, meta_lfi, r_2, rmse / base, r, rho])

    def getRankings(self, columns, prediction, actual):
        pred_data = {"target": prediction, "names": columns}
        act_data = {"target": actual, "names": columns}
        pred = DataFrame(pred_data).sort_values(by=["target"], ascending=False)["names"].values
        act = DataFrame(act_data).sort_values(by=["target"], ascending=False)["names"].values

        return act, pred

    @staticmethod
    def compare(model, X, y, predicted, actual, k):
        columns = X.columns
        predicted = [columns[i] for i in predicted]
        actual = [columns[i] for i in actual]

        X_anova_f = SelectKBest(f_classif, k=k).fit_transform(X, y)
        X_mutual_info = SelectKBest(mutual_info_classif, k=k).fit_transform(X, y)
        X_fi = X[actual[:k]]
        X_meta_lfi = X[predicted[:k]]

        sc_X = StandardScaler()
        X_anova_f = sc_X.fit_transform(X_anova_f)
        X_mutual_info = sc_X.fit_transform(X_mutual_info)
        X_fi = sc_X.fit_transform(X_fi)
        X_meta_lfi = sc_X.fit_transform(X_meta_lfi)

        anova_f = mean(cross_val_score(model, X_anova_f, y, cv=5))
        mutual_info = mean(cross_val_score(model, X_mutual_info, y, cv=5))
        fi = mean(cross_val_score(model, X_fi, y, cv=5))
        meta_lfi = mean(cross_val_score(model, X_meta_lfi, y, cv=5))

        return anova_f, mutual_info, fi, meta_lfi
