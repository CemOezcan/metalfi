import numpy as np

from copy import deepcopy
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from metalfi.src.metadata.dataset import Dataset
from metalfi.src.metadata.dropcolumn import DropColumnImportance
from metalfi.src.metadata.lime import LimeImportance
from metalfi.src.metadata.permutation import PermutationImportance
from metalfi.src.metadata.shap import ShapImportance
from metalfi.src.metadata.metafeatures import MetaFeatures
from metalfi.src.model.evaluation import Evaluation


class MetaModel:

    def __init__(self, train, name, test, og_data, selected, untrained_meta_models, target_names):
        self.__og_y = og_data.getDataFrame()[og_data.getTarget()]
        self.__og_X = og_data.getDataFrame().drop(og_data.getTarget(), axis=1)
        self.__untrained_meta_models = untrained_meta_models
        self.__target_names = target_names
        self.__selected = selected
        self.__file_name = name
        self.__meta_models = list()
        self.__stats = list()
        self.__results = list()
        self.__result_configurations = list()
        self.__was_compared = False

        sc1 = StandardScaler()
        sc1.fit(train)
        self.__train_data = DataFrame(data=sc1.transform(train), columns=train.columns)
        self.__test_data = DataFrame(data=sc1.transform(test), columns=test.columns)

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
        self.__meta_feature_groups = {0: "Auto", 1: "All", 2: "FMF", 3: "LM", 4: "Multi", 5: "Uni"}

    def wasCompared(self):
        return self.__was_compared

    def getName(self):
        return self.__file_name

    def getTargets(self):
        return self.__target_names

    def getStats(self):
        return self.__stats

    def getBaseModels(self):
        return self.__untrained_meta_models

    def getMetaModels(self):
        return self.__meta_models

    def getEnum(self):
        return self.__meta_feature_groups

    def getResults(self):
        return self.__results

    def getResultConfig(self):
        return self.__result_configurations

    def fit(self):
        X = self.__train_data.drop(self.__target_names, axis=1)

        for base_model, base_model_name in self.__untrained_meta_models:
            for target in self.__target_names:
                i = 0
                for feature_set in self.__feature_sets:
                    y = self.__train_data[target]
                    X_train, selected_features = self.featureSelection(base_model_name, X, target) \
                        if feature_set[0] == "Auto" else (X[feature_set], feature_set)

                    model = self.trainModel(base_model, X_train, y)
                    feature_set_name = self.__meta_feature_groups.get(i)

                    self.__meta_models.append((deepcopy(model), selected_features,
                                               [base_model_name, target, feature_set_name]))

                    i -= -1

    def featureSelection(self, name, X_train, target_name):
        features = self.__selected[name][target_name]

        return X_train[features], features

    def trainModel(self, model, X, y):
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
                r_2, rmse, base, r = 0, 1, 1, 0

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
            og_model = RandomForestClassifier(n_estimators=10, random_state=115, n_jobs=-1)
        elif name.startswith("SVC"):
            og_model = SVC(random_state=115)
        elif name.startswith("LOG"):
            og_model = LogisticRegression(dual=False, max_iter=1000, random_state=115, n_jobs=-1)
        elif name.startswith("linSVC"):
            og_model = LinearSVC(dual=False, max_iter=10000, random_state=115)
        elif name.startswith("NB"):
            og_model = GaussianNB()

        return og_model

    def compare(self, models, targets, subsets, k, renew=False):
        if renew:
            self.__results = list()
            self.__result_configurations = list()
            self.__was_compared = False

        if self.__was_compared:
            return ["ANOVA", "MI", "FI", "MetaLFI"]

        meta_models = [(model, features, config[1], config) for (model, features, config) in self.__meta_models
                       if ((config[0] in models) and (config[1] in targets) and (config[2] in subsets))]
        self.__result_configurations += [config for (_, _, _, config) in meta_models]
        results = {0: list(), 1: list(), 2: list(), 3: list(), 4: list()}

        folds = self.get_cross_validation_folds(self.__og_X, self.__og_y, k=5)
        i = 0
        for X_train, X_test, y_train, y_test in folds:
            sc = StandardScaler()

            X_train = DataFrame(data=sc.fit_transform(X_train), columns=self.__og_X.columns)
            X_test = DataFrame(data=sc.fit_transform(X_test), columns=self.__og_X.columns)
            whole_train = DataFrame(data=sc.transform(X_train), columns=self.__og_X.columns)
            whole_train["target"] = y_train
            X_meta, y_meta = self.get_meta(whole_train, "target", targets)

            selector_anova = SelectPercentile(f_classif, percentile=k)
            selector_anova.fit(X_train, y_train)
            selector_mi = SelectPercentile(mutual_info_classif, percentile=k)
            selector_mi.fit(X_train, y_train)

            X_anova_train = selector_anova.transform(X_train)
            X_mi_train = selector_mi.transform(X_train)

            X_anova_test = selector_anova.transform(X_test)
            X_mi_test = selector_mi.transform(X_test)
            k_number = len(X_anova_test[0])

            for model, features, target, config in meta_models:
                X_temp = X_meta[features]
                y_temp = y_meta[target]
                y_pred = model.predict(X_temp)
                og_model = self.getOriginalModel(target)
                columns = self.__og_X.columns

                a, p = self.getRankings(self.__test_data.index, y_pred, y_temp)
                predicted = [columns[i] for i in p]
                actual = [columns[i] for i in a]

                X_fi_train = X_train[actual[:k_number]]
                X_metalfi_train = X_train[predicted[:k_number]]

                X_fi_test = X_test[actual[:k_number]]
                X_metalfi_test = X_test[predicted[:k_number]]

                og_model.fit(X_fi_train, y_train)
                fi = og_model.score(X_fi_test, y_test)

                og_model.fit(X_metalfi_train, y_train)
                metalfi = og_model.score(X_metalfi_test, y_test)

                og_model.fit(X_anova_train, y_train)
                anova = og_model.score(X_anova_test, y_test)

                og_model.fit(X_mi_train, y_train)
                mi = og_model.score(X_mi_test, y_test)

                results[i].append([anova, mi, fi, metalfi])

            i += 1

        for i in results:
            self.__results = Evaluation.vectorAddition(self.__results, results[i])

        self.__results = [list(map(lambda x: x / 5, result)) for result in self.__results]
        self.__was_compared = True

        return ["ANOVA", "MI", "FI", "MetaLFI"]

    def get_meta(self, data_frame, target, targets):
        new_targets = [x[-4:] for x in targets]
        dataset = Dataset(data_frame, target)
        meta_features = MetaFeatures(dataset)
        meta_features.calculateMetaFeatures()

        if "SHAP" in new_targets:
            meta_features.addTarget(ShapImportance(dataset))

        if "PIMP" in new_targets:
            meta_features.addTarget(PermutationImportance(dataset))

        if "LOFO" in new_targets:
            meta_features.addTarget(DropColumnImportance(dataset))

        if "LIME" in new_targets:
            meta_features.addTarget(LimeImportance(dataset))

        meta_data = meta_features.getMetaData()

        return meta_data.drop(targets, axis=1), meta_data[targets]

    def get_cross_validation_folds(self, X, y, k=5):
        X_temp = X.values
        y_temp = y.values

        kf = KFold(n_splits=k, shuffle=True, random_state=115)
        kf.get_n_splits(X)

        folds = list()
        for train_index, test_index in kf.split(X):
            folds.append((X_temp[train_index], X_temp[test_index], y_temp[train_index], y_temp[test_index]))

        return folds
