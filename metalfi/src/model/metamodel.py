import math
from typing import List, Tuple

import numpy
import numpy as np

from copy import deepcopy

import sklearn
from pandas import DataFrame
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectPercentile
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.dataset import Dataset
from metalfi.src.metadata.dropcolumn import DropColumnImportance
from metalfi.src.metadata.lime import LimeImportance
from metalfi.src.metadata.permutation import PermutationImportance
from metalfi.src.metadata.shap import ShapImportance
from metalfi.src.metadata.metafeatures import MetaFeatures
from metalfi.src.model.evaluation import Evaluation
from metalfi.src.parameters import Parameters


class MetaModel:
    """
    Contains all meta-models, that are supposed to be tested on the same cross validation split.

    Attributes
    ----------
        __og_y : (DataFrame)
            Target variable of the base-data set.
        __og_X : (DataFrame)
            Base-data set without the target variable.
        __selected : (Dict[str, Dict[str, List[str]]])
            Selected meta-features for all meta-models.
        __file_name : (str)
            Name of the file, this instance is going to be saved as.
        __meta_models : (List[Tuple[sklearn estimator, List[str], List[str]]])
            Trained meta-models, their meta-features and configurations.
        __stats : (List[List[float]])
            Performance estimates for trained meta-models.
        __results : (List[List[float]])
            Performance estimates for base-models, trained on different feature subsets.
        __result_configurations : (List[List[str]])
            Configurations for feature selection approach comparisons:
            meta-model name, meta-target name and feature selection approach.
        __was_compared : (bool)
            Whether the feature selection approaches have been compared or not.
        __train_data : (DataFrame)
            Meta-data set, on which the meta-models are trained.
        __test_data : (DataFrame)
            Meta-data set, on which the meta-models are tested.
        __feature_sets : (List[List[str]])
            Meta-feature subsets.
        __meta_feature_groups : (Dict[int, str])
            Maps indices to meta-feature subset names.

    """
    def __init__(self, iterable):
        train, name, test, og_data, selected = iterable
        self.__og_y = og_data.getDataFrame()[og_data.getTarget()]
        self.__og_X = og_data.getDataFrame().drop(og_data.getTarget(), axis=1)
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

        train = train.drop(Parameters.targets, axis=1)
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

    def was_compared(self):
        return self.__was_compared

    def get_name(self):
        return self.__file_name

    def get_stats(self):
        return self.__stats

    def get_meta_models(self):
        return self.__meta_models

    def get_enum(self):
        return self.__meta_feature_groups

    def get_results(self):
        return self.__results

    def get_result_config(self):
        return self.__result_configurations

    def fit(self):
        """
        Fit all meta-models to the train split in `__train_data`. Save trained meta-models at `__meta_models`.
        """
        X = self.__train_data.drop(Parameters.targets, axis=1)

        for base_model, base_model_name, _ in Parameters.meta_models:
            for target in Parameters.targets:
                i = 0
                for feature_set in self.__feature_sets:
                    y = self.__train_data[target]
                    X_train, selected_features = self.__feature_selection(base_model_name, X, target) \
                        if feature_set[0] == "Auto" else (X[feature_set], feature_set)

                    model = base_model.fit(X_train, y)
                    feature_set_name = self.__meta_feature_groups.get(i)

                    self.__meta_models.append((deepcopy(model), selected_features,
                                               [base_model_name, target, feature_set_name]))

                    i -= -1

    def __feature_selection(self, name, X_train, target_name):
        features = self.__selected[name][target_name]
        return X_train[features], features

    def test(self, renew=False):
        """
        Estimate meta-model performances by evaluating predictions on the test split `__test_data`.
        Save test results at `__stats`.

        Parameters
        ----------
            renew : (bool) Whether to re-calculate model performances.

        """
        if renew:
            self.__stats = list()

        if len(self.__stats) == len(self.__meta_models):
            return

        X = self.__test_data.drop(Parameters.targets, axis=1)
        for (model, features, config) in self.__meta_models:
            X_test = X[features]
            y_test = self.__test_data[config[1]]
            y_train = self.__train_data[config[1]]
            y_pred = model.predict(X_test)

            self.__stats.append(Parameters.calculate_metrics(y_train, y_test, y_pred))

    def __get_rankings(self, columns, prediction, actual):
        pred_data = {"target": prediction, "names": columns}
        act_data = {"target": actual, "names": columns}
        pred = DataFrame(pred_data).sort_values(by=["target"], ascending=False)["names"].values
        act = DataFrame(act_data).sort_values(by=["target"], ascending=False)["names"].values

        return act, pred

    @staticmethod
    def __get_original_model(name: str) -> 'sklearn estimator':
        """
        Identifies and returns the base-model, given the its name.

        Parameters
        ----------
            name : Base-model name.

        Returns
        -------
            The base-model.
        """
        for model, model_name, _ in Parameters.base_models:
            if name.startswith(model_name):
                return model

    def compare(self, models: List[str], targets: List[str], subsets: List[str], k: int, renew=False) -> List[str]:
        """
        Apply different feature selection approaches to the base-data set `__og_X`, `__og_y`.
        Estimate the performances of different base-models in combination with all feature selection approaches.
        Save the results at `__results` and set `__was_compared` to True.

        Parameters
        ----------
            models : Meta-model names.
            targets : Meta-targets to predict.
            subsets : Meta-feature subset names.
            k : Parameter for k-fold cross-validation.
            renew : (bool) Whether to recalculate the results.

        Returns
        -------
            List of feature selection approaches.
        """
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

        folds = self.__get_cross_validation_folds(self.__og_X, self.__og_y, k=5)
        i = 0
        for X_train, X_test, y_train, y_test in folds:
            sc = StandardScaler()

            X_train = DataFrame(data=sc.fit_transform(X_train), columns=self.__og_X.columns)
            X_test = DataFrame(data=sc.fit_transform(X_test), columns=self.__og_X.columns)
            whole_train = DataFrame(data=sc.transform(X_train), columns=self.__og_X.columns)
            whole_train["target"] = y_train
            X_meta, y_meta = self.__get_meta(whole_train, "target", targets)

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
                og_model = self.__get_original_model(target)
                columns = self.__og_X.columns

                a, p = self.__get_rankings(self.__test_data.index, y_pred, y_temp)
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
            self.__results = Evaluation.matrix_addition(self.__results, results[i])

        self.__results = [list(map(lambda x: x / 5, result)) for result in self.__results]
        self.__was_compared = True

        return ["ANOVA", "MI", "FI", "MetaLFI"]

    def __get_meta(self, data_frame: DataFrame, target: str, targets: List[str]) -> Tuple[DataFrame, DataFrame]:
        new_targets = [x[-4:] for x in targets]
        dataset = Dataset(data_frame, target)
        meta_features = MetaFeatures(dataset)
        meta_features.calculate_meta_features()

        if "SHAP" in new_targets:
            meta_features.add_target(ShapImportance(dataset))

        if "PIMP" in new_targets:
            meta_features.add_target(PermutationImportance(dataset))

        if "LOFO" in new_targets:
            meta_features.add_target(DropColumnImportance(dataset))

        if "LIME" in new_targets:
            meta_features.add_target(LimeImportance(dataset))

        meta_data = meta_features.get_meta_data()

        return meta_data.drop(targets, axis=1), meta_data[targets]

    @staticmethod
    def __get_cross_validation_folds(X, y, k=5):
        X_temp = X.values
        y_temp = y.values

        kf = KFold(n_splits=k, shuffle=True, random_state=115)
        kf.get_n_splits(X)

        folds = list()
        for train_index, test_index in kf.split(X):
            folds.append((X_temp[train_index], X_temp[test_index], y_temp[train_index], y_temp[test_index]))

        return folds
