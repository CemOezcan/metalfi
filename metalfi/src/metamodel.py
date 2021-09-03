from copy import deepcopy
from functools import partial
import multiprocessing as mp
import time
from typing import List, Tuple
import warnings

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectPercentile
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
import tqdm

from metalfi.src.metadata.dataset import Dataset
from metalfi.src.metadata.dropcolumn import DropColumnImportance
from metalfi.src.metadata.lime import LimeImportance
from metalfi.src.metadata.metafeatures import MetaFeatures
from metalfi.src.metadata.permutation import PermutationImportance
from metalfi.src.metadata.shap import ShapImportance
from metalfi.src.evaluation import Evaluation
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
        __meta_models : (List[Tuple[Estimator, List[str], List[str]]])
            Trained meta-models, their meta-features and configurations.
        __stats : (List[List[float]])
            Performance estimates for trained meta-models.
        __results : (List[List[float]])
            Performance estimates for base-models, trained on different feature subsets.
        __result_configurations : (List[List[str]])
            Configurations for feature selection approach comparisons:
            meta-model name, meta-target name and feature selection approach.
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
        self.__og_y = og_data.get_data_frame()[og_data.get_target()]
        self.__og_X = og_data.get_data_frame().drop(og_data.get_target(), axis=1)
        self.__selected = selected
        self.__file_name = name
        self.__meta_models = []
        self.__stats = []
        self.__results = []
        self.__times = []
        self.__result_configurations = []

        self.__sc1 = StandardScaler()
        self.__sc1.fit(train)
        temp = train.drop(Parameters.targets, axis=1)
        self.__train_data = DataFrame(data=self.__sc1.transform(train), columns=train.columns)
        self.__test_data = DataFrame(data=self.__sc1.transform(test), columns=test.columns)

        self.__sc1.fit(temp)

        train = train.drop(Parameters.targets, axis=1)
        fmf = [x for x in train.columns if "." not in x]
        lm = [x for x in train.columns if x.startswith("target_")]
        multi = [x for x in train.columns if x.startswith("multi_")]
        uni = [x for x in train.columns if (x in fmf) and (x not in multi) and (x not in lm)]
        lm_uni = lm + uni
        lm_multi = lm + multi
        lm_multi_ft = lm + [x for x in multi if x.startswith("multi_cb")]
        uni_multi_ff = uni + [x for x in multi if not x.startswith("multi_cb")]

        self.__feature_sets = [["Auto"], train.columns, fmf, lm, multi, uni, lm_uni, lm_multi,
                               lm_multi_ft, uni_multi_ff]
        self.__meta_feature_groups = {0: "Auto", 1: "All", 2: "FMF", 3: "LM", 4: "Multi", 5: "Uni",
                                      6: "LMUni", 7: "LMMulti", 8: "LMMultiFT", 9: "UniMultiFF"}

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

    def get_times(self):
        return self.__times

    def get_result_config(self):
        return self.__result_configurations

    def fit(self):
        """
        Fit all meta-models to the train split in `__train_data`.
        Save trained meta-models at `__meta_models`.
        """
        X_1 = self.__train_data.drop(Parameters.targets, axis=1)
        X_2 = self.__test_data.drop(Parameters.targets, axis=1)

        for base_model, base_model_name, _ in Parameters.meta_models:
            for target in Parameters.targets:
                i = 0
                for feature_set in self.__feature_sets:
                    y = self.__train_data[target]
                    X_train, selected_features = self.__feature_selection(base_model_name, X_1, target) \
                        if feature_set[0] == "Auto" else (X_1[feature_set], feature_set)

                    warnings.filterwarnings("ignore", message="Liblinear failed to converge, increase the number of iterations.")
                    model = base_model.fit(X_train, y)
                    warnings.filterwarnings("default")
                    feature_set_name = self.__meta_feature_groups.get(i)

                    self.__meta_models.append((deepcopy(model), selected_features,
                                               [base_model_name, target, feature_set_name]))
                    X_test = X_2[selected_features]
                    y_test = self.__test_data[target]
                    y_pred = model.predict(X_test)

                    self.__stats.append(Parameters.calculate_metrics(y_test=y_test, y_pred=y_pred))

                    i -= -1

        results, times = self.parallel_comparisons((self.__og_X, self.__og_y, self.__file_name))

        all_res = {list(results.keys())[0]: results[list(results.keys())[0]]}
        all_times = {list(times.keys())[0]: times[list(times.keys())[0]]}
        sum_up = lambda x: x[0] if len(x) == 1 else Evaluation.matrix_addition(x[0], sum_up(x[1:]))
        self.__results = {key: [[x / 5 for x in result] for result in sum_up(all_res[key])]
                          for key in all_res.keys()}
        self.__times = {key: [[x / 5 for x in result] for result in sum_up(all_times[key])]
                        for key in all_times.keys()}
        self.__meta_models = [(None, feat, config) for model, feat, config in self.__meta_models]

    def __feature_selection(self, name: str, X_train: DataFrame, target_name: str) -> Tuple[DataFrame, List[str]]:
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
            self.__stats = []

        if len(self.__stats) == len(self.__meta_models):
            return

        X = self.__test_data.drop(Parameters.targets, axis=1)
        for (model, features, config) in self.__meta_models:
            X_test = X[features]
            y_test = self.__test_data[config[1]]
            y_pred = model.predict(X_test)

            self.__stats.append(Parameters.calculate_metrics(y_test=y_test, y_pred=y_pred))

    @staticmethod
    def __get_original_model(name: str) -> BaseEstimator:
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
        return None

    def parallel_comparisons(self, args):
        """
        Apply different feature selection approaches to the base-data set `__og_X`, `__og_y`.
        Estimate the performances of different base-models in combination with all
        feature-selection approaches.
        Save the results at `__results` and set `__was_compared` to True.

        Parameters
        ----------
            args :

        Returns
        -------
            Accuracy scores and computation times.
        """
        k = 33
        meta_target_names = [x for x in Parameters.targets if "SHAP" in x]
        used_models = [(model, features, config) for model, features, config in self.__meta_models
                       if config[1] in meta_target_names]
        self.__result_configurations = [config for (_, _, config) in used_models]
        X_test, y_test, name = args

        anova_scores = {name: dict()}
        mi_scores = {name: dict()}
        pimp_scores = {name: dict()}
        bagging_scores = {name: dict()}
        baseline_scores = {name: dict()}
        all_res = {name: dict()}

        anova_times = {name: dict()}
        mi_times = {name: dict()}
        pimp_times = {name: dict()}
        all_times = {name: dict()}

        results = []
        times = []
        for X_tr, X_te, y_tr, y_te in self.__get_cross_validation_folds(X_test, y_test):
            warnings.filterwarnings("ignore", category=DeprecationWarning, message="Using.*")
            results.append([])
            times.append([])
            dataset = Dataset(DataFrame(data=X_tr).assign(target=y_tr), "target")
            mf = MetaFeatures(dataset)
            metalfi_time_tuple = mf.calculate_meta_features()
            metalfi_time = {"Auto": sum(metalfi_time_tuple[2:]), "All": sum(metalfi_time_tuple),
                            "FMF": sum(metalfi_time_tuple[1:]), "LM": metalfi_time_tuple[4],
                            "Multi": sum(metalfi_time_tuple[2:4]), "Uni": metalfi_time_tuple[1],
                            "LMUni": metalfi_time_tuple[1] + metalfi_time_tuple[4],
                            "LMMulti": sum(metalfi_time_tuple[2:]), "LMMultiFT": sum(metalfi_time_tuple[3:]),
                            "UniMultiFF": metalfi_time_tuple[1] + metalfi_time_tuple[3]}
            X_m = DataFrame(data=self.__sc1.transform(mf.get_meta_data()), columns=mf.get_meta_data().columns)
            mf.add_target(PermutationImportance(dataset))
            warnings.filterwarnings("ignore", category=DeprecationWarning, message="tostring.*")
            warnings.filterwarnings("ignore", message="(invalid value.*|"
                                                      "Features.*|"
                                                      "Input data for shapiro has range zero.*)")

            normalizer = Normalizer()
            anova_time = self.measure_time(f_classif, X_tr, y_tr)
            mi_time = self.measure_time(partial(mutual_info_classif, random_state=115), X_tr, y_tr)
            bagging_time = self.measure_time(lambda x, y: [x.mean() for x in normalizer.fit_transform(X_m[self.__feature_sets[3]])], X_tr, y_tr)

            for og_model, n in {(self.__get_original_model(target), target[:-5])
                                for target in meta_target_names}:
                pipeline_anova = make_pipeline(
                    StandardScaler(), SelectPercentile(f_classif, percentile=k), og_model)
                pipeline_mi = make_pipeline(
                    StandardScaler(),
                    SelectPercentile(partial(mutual_info_classif, random_state=115), percentile=k),
                    og_model)
                pipeline_pimp = make_pipeline(
                    StandardScaler(),
                    SelectPercentile(lambda x, y: np.asarray(mf.get_meta_data()[n + "_PIMP"]), percentile=k),
                    og_model)
                pipeline_bagging = make_pipeline(
                    StandardScaler(),
                    SelectPercentile(lambda x, y: np.asarray(
                        [x.mean() for x in normalizer.fit_transform(X_m[self.__feature_sets[3]])]), percentile=k),
                    og_model)
                pipeline_baseline = make_pipeline(StandardScaler(), og_model)

                anova_scores[name][n] = pipeline_anova.fit(X_tr, y_tr).score(X_te, y_te)
                mi_scores[name][n] = pipeline_mi.fit(X_tr, y_tr).score(X_te, y_te)
                pimp_scores[name][n] = pipeline_pimp.fit(X_tr, y_tr).score(X_te, y_te)
                bagging_scores[name][n] = pipeline_bagging.fit(X_tr, y_tr).score(X_te, y_te)
                baseline_scores[name][n] = pipeline_baseline.fit(X_tr, y_tr).score(X_te, y_te)

                anova_times[name][n] = anova_time
                mi_times[name][n] = mi_time
                pimp_times[name][n] = self.measure_time(
                    partial(PermutationImportance.data_permutation_importance, og_model), X_tr, y_tr)

            for model, features, config in used_models:
                metalfi_prediction_time = self.measure_time(lambda x, y: model.predict(x), X_m[features], None)
                pipeline_metalfi = make_pipeline(
                    StandardScaler(),
                    SelectPercentile(lambda x, y: model.predict(X_m[features]), percentile=k),
                    self.__get_original_model(config[1]))

                metalfi = pipeline_metalfi.fit(X_tr, y_tr).score(X_te, y_te)
                results[-1].append([anova_scores[name][config[1][:-5]],
                                    mi_scores[name][config[1][:-5]],
                                    bagging_scores[name][config[1][:-5]],
                                    pimp_scores[name][config[1][:-5]],
                                    metalfi,
                                    baseline_scores[name][config[1][:-5]]])

                times[-1].append([anova_times[name][config[1][:-5]],
                                  mi_times[name][config[1][:-5]],
                                  metalfi_time["LM"] + bagging_time,
                                  pimp_times[name][config[1][:-5]],
                                  metalfi_prediction_time + metalfi_time[config[2]]])

            warnings.filterwarnings("default")

        all_res[name] = results
        all_times[name] = times
        return all_res, all_times

    @staticmethod
    def measure_time(func, X, y):
        start = time.time()
        func(X, y)
        end = time.time()
        return end - start

    def compare_all(self, test_data: List[Tuple[Dataset, str]]):
        # Deprecated
        X_1 = self.__train_data.drop(Parameters.targets, axis=1)
        meta_model_names, meta_target_names, _ = Parameters.question_5_parameters()
        test_data_sets = [(d.get_data_frame().drop("base-target_variable", axis=1),
                           d.get_data_frame()["base-target_variable"], name) for d, name in test_data]

        for meta_model, model_name, _ in [x for x in Parameters.meta_models if x[1] in meta_model_names]:
            for target in meta_target_names:
                y = self.__train_data[target]
                j = 0
                for feature_set in self.__feature_sets[:4]:
                    X_train, selected_features = self.__feature_selection(model_name, X_1, target) \
                        if feature_set[0] == "Auto" else (X_1[feature_set], feature_set)
                    warnings.filterwarnings("ignore", message="Liblinear failed to converge, increase the number of iterations.")
                    model = meta_model.fit(X_train, y)
                    warnings.filterwarnings("default")
                    feature_set_name = self.__meta_feature_groups.get(j)
                    self.__meta_models.append((deepcopy(model), selected_features,
                                               [model_name, target, feature_set_name]))
                    j += 1

        with mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1) as pool:
            progress_bar = tqdm.tqdm(total=len(test_data_sets), desc="Comparing feature-selection approaches")
            results = [pool.map_async(self.parallel_comparisons, (arg,),
                                      callback=(lambda x: progress_bar.update()))
                       for arg in test_data_sets]

            results = [x.get()[0] for x in results]
            pool.close()
            pool.join()

        progress_bar.close()

        all_res = {list(result.keys())[0]: result[list(result.keys())[0]] for result in results}
        sum_up = lambda x: x[0] if len(x) == 1 else Evaluation.matrix_addition(x[0], sum_up(x[1:]))
        self.__result_configurations += [config for (_, _, config) in self.__meta_models]
        self.__results = {key: [[x / 5 for x in result] for result in sum_up(all_res[key])]
                          for key in all_res.keys()}

    @staticmethod
    def __get_meta(data_frame: DataFrame, target: str, targets: List[str]) -> Tuple[DataFrame, DataFrame]:
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
    def __get_cross_validation_folds(X: DataFrame, y: DataFrame, k=5) \
            -> List[Tuple[DataFrame, DataFrame, DataFrame, DataFrame]]:
        X_temp = X.values
        y_temp = y.values

        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=115)
        kf.get_n_splits(X)

        folds = []
        for train_index, test_index in kf.split(X, y):
            folds.append((DataFrame(X_temp[train_index], columns=X.columns),
                          DataFrame(X_temp[test_index], columns=X.columns),
                          y_temp[train_index], y_temp[test_index]))

        return folds
