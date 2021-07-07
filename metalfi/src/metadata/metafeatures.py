import math
import time
from typing import List, Dict, Union, Sequence
import warnings

import numpy as np
from pandas import DataFrame
from pymfe.mfe import MFE
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import MinMaxScaler

from metalfi.src.metadata.dropcolumn import DropColumnImportance
from metalfi.src.metadata.featureimportance import FeatureImportance
from metalfi.src.metadata.lime import LimeImportance
from metalfi.src.metadata.permutation import PermutationImportance
from metalfi.src.metadata.shap import ShapImportance


class MetaFeatures:
    """
    A meta-data set.
    Computes a meta-data set, given a base-data set.

    Attributes
    ----------
    __dataset : (Dataset)
        The underlying base-data set.
    __meta_data : (DataFrame)
        The extracted meta-data set.
    __targets : (List[str])
        The names of all meta-targets.
    __feature_meta_features : (List[List[float]])
        The values of all feature-meta-features.
    __data_meta_features : (List[float])
        The values of all data-meta-features.
    __data_meta_feature_names : (List[str])
        The names of all data-meta-features.
    __feature_meta_feature_names : (List[str])
        The names of all feature-meta-features.
    """

    def __init__(self, dataset):
        self.__dataset = dataset
        self.__meta_data = DataFrame()
        self.__targets = list()
        self.__feature_meta_features = list()
        self.__data_meta_features = list()
        self.__data_meta_feature_names = list()
        self.__feature_meta_feature_names = list()

    def get_meta_data(self):
        return self.__meta_data

    def calculate_meta_features(self) -> (float, float, float, float):
        """

        Returns
        -------

        """
        uni_time, multi_time, lm_time = self.__compute_feature_meta_features()
        data_time = self.__compute_data_meta_features()
        self.__create_meta_data()

        return data_time, uni_time, multi_time, lm_time

    @staticmethod
    def __run_pymfe(X: Union[DataFrame, np.ndarray], y: Union[DataFrame, np.ndarray], summary: Union[List[str], None],
                    features: List[str]) -> (List[str], List[str]):
        warnings.filterwarnings("ignore", message="(It is not possible make equal discretization|"
                                                  "(divide by zero|invalid value) encountered in .*|"
                                                  "Can't summarize feature 'cor' with .*|"
                                                  "Features.*|"
                                                  "Input data for shapiro has range zero.*)")
        mfe = MFE(summary=summary, features=features)
        mfe.fit(X, y)
        vector = mfe.extract()

        return vector

    def __compute_data_meta_features(self) -> float:
        start = time.time()
        data_frame = self.__dataset.get_data_frame()
        target = self.__dataset.get_target()

        X = data_frame.drop(target, axis=1)
        y = data_frame[target]

        names_1, dmf_1 = self.__run_pymfe(X.values, y.values,
                                          ["min", "median", "max", "sd", "kurtosis", "skewness", "mean"],
                                          ["attr_to_inst", "freq_class", "inst_to_attr", "nr_attr", "nr_class",
                                           "nr_inst", "gravity", "cor", "cov", "eigenvalues", "nr_cor_attr",
                                           "class_ent", "eq_num_attr", "ns_ratio", "iq_range", "kurtosis", "mad", "max",
                                           "mean", "median", "min", "range", "sd", "skewness", "sparsity", "t_mean",
                                           "var", "attr_ent", "joint_ent", "mut_inf", "nr_norm", "nr_outliers",
                                           "nr_cat", "nr_bin", "nr_num"])

        self.__data_meta_feature_names = names_1
        self.__data_meta_features = dmf_1
        end = time.time()

        return end - start

    def __compute_feature_meta_features(self) -> (float, float, float):
        start_uni = time.time()
        data_frame = self.__dataset.get_data_frame()
        target = self.__dataset.get_target()

        y = data_frame[target]
        X = data_frame.drop(target, axis=1)
        columns = list()

        for feature in X.columns:
            X_temp = data_frame[feature]

            columns, values = self.__run_pymfe(X_temp.values, y.values, None,
                                               ["iq_range", "kurtosis", "mad", "max", "mean", "median", "min", "range",
                                                "sd", "skewness", "sparsity", "t_mean", "var", "attr_ent", "nr_norm",
                                                "nr_outliers", "nr_cat", "nr_bin", "nr_num"])

            self.__feature_meta_features.append(self.__to_feature_vector(values))

        self.__feature_meta_feature_names = columns
        end_uni = time.time()
        total_uni = end_uni - start_uni

        start_multi = time.time()
        cov = X.cov()
        p_cor = X.corr("pearson")
        s_cor = X.corr("spearman")
        k_cor = X.corr("kendall")
        su_cor = self.__symmetrical_uncertainty(X=X, y=y, matrix=True)

        self.__correlation_feature_meta_features(cov, "_cov")
        mean_p = self.__correlation_feature_meta_features(p_cor, "_p_corr")
        mean_s = self.__correlation_feature_meta_features(s_cor, "_s_corr")
        mean_k = self.__correlation_feature_meta_features(k_cor, "_k_corr")
        mean_su = self.__correlation_feature_meta_features(su_cor, "_su_corr")

        end_multi = time.time()
        total_multi = end_multi - start_multi

        start_lm = time.time()
        columns, values = self.__run_pymfe(X.values, y.values, None, ["joint_ent", "mut_inf"])
        for feature in X.columns:
            loc = X.columns.get_loc(feature)
            self.__feature_meta_features[loc].append(values[0][loc])
            self.__feature_meta_features[loc].append(values[1][loc])

        for column in columns:
            self.__feature_meta_feature_names.append("target_" + column)

        self.__filter_scores(X, y, mean_p, mean_s, mean_k, mean_su)
        end_lm = time.time()
        total_lm = end_lm - start_lm

        return total_uni, total_multi, total_lm

    def __correlation_feature_meta_features(self, matrix: DataFrame, name: str, threshold=0.5) -> Dict[str, np.ndarray]:
        mean_correlation = {}
        for i in range(0, len(matrix.columns)):
            values = list()
            for j in range(0, len(matrix.columns)):
                if i != j:
                    values.append(abs(matrix.iloc[i].iloc[j]))

            mean_correlation[matrix.columns[i]] = np.mean(values)
            percentile = np.percentile(values, 75)
            th = map(lambda x: x >= threshold, values)
            th_list = list(map(lambda x: x if (x >= threshold) else 0, values))
            percentile_list = list(map(lambda x: x if (x >= percentile) else 0, values))

            self.__feature_meta_features[i] += [np.mean(values), np.median(values),
                                                np.std(values), np.var(values),
                                                max(values), min(values), percentile, np.mean(percentile_list),
                                                sum(th), sum(th) / len(values), np.mean(th_list)]

        self.__feature_meta_feature_names += ["multi_mean" + name, "multi_median" + name, "multi_sd" + name,
                                              "multi_var" + name, "multi_max" + name, "multi_min" + name,
                                              "multi_percentile_0,75" + name, "multi_mean_percentile_0,75" + name,
                                              "multi_high_corr" + name, "multi_high_corr_ratio" + name,
                                              "multi_mean_high_corr" + name]
        return mean_correlation

    def __filter_scores(self, X: DataFrame, y: DataFrame, p_cor: Dict[str, np.ndarray], s_cor: Dict[str, np.ndarray],
                        k_cor: Dict[str, np.ndarray], su_cor: Dict[str, np.ndarray]):
        sc = MinMaxScaler()
        X_sc = sc.fit_transform(X)

        chi2_values, chi2_p_values = chi2(X_sc, y)
        warnings.filterwarnings("ignore", message="divide by zero encountered in .*")
        f_values, anova_p_values = f_classif(X, y)
        # TODO: Better solution
        f_values = list(map((lambda x: 500 if x == float("inf") else x), f_values))
        log_anova_p = list(map((lambda x: -500 if x == float("-inf") else x), [np.log(x) for x in anova_p_values]))
        log_anova_p = list(map((lambda x: 1 if math.isnan(x) else x), log_anova_p))
        log_chi2_p = list(map((lambda x: -500 if x == float("-inf") else x), [np.log(x) for x in chi2_p_values]))

        for feature in X.columns:
            loc = X.columns.get_loc(feature)
            p = X[feature].corr(y, method="pearson")
            s = X[feature].corr(y, method="kendall")
            k = X[feature].corr(y, method="spearman")
            su = self.__symmetrical_uncertainty(X[feature], y, matrix=False)

            self.__feature_meta_features[loc].append(abs(p))
            self.__feature_meta_features[loc].append(abs(s))
            self.__feature_meta_features[loc].append(abs(k))
            self.__feature_meta_features[loc].append(su)
            self.__feature_meta_features[loc].append(f_values[loc])
            self.__feature_meta_features[loc].append(log_anova_p[loc])
            self.__feature_meta_features[loc].append(chi2_values[loc])
            self.__feature_meta_features[loc].append(log_chi2_p[loc])
            self.__feature_meta_features[loc].append(abs(p) / np.sqrt(1 + 2 * p_cor[feature]))
            self.__feature_meta_features[loc].append(abs(s) / np.sqrt(1 + 2 * s_cor[feature]))
            self.__feature_meta_features[loc].append(abs(k) / np.sqrt(1 + 2 * k_cor[feature]))
            self.__feature_meta_features[loc].append(su / np.sqrt(1 + 2 * su_cor[feature]))

        self.__feature_meta_feature_names.append("target_p_corr")
        self.__feature_meta_feature_names.append("target_s_corr")
        self.__feature_meta_feature_names.append("target_k_corr")
        self.__feature_meta_feature_names.append("target_su_corr")
        self.__feature_meta_feature_names.append("target_F_value")
        self.__feature_meta_feature_names.append("target_anova_p_value")
        self.__feature_meta_feature_names.append("target_chi2")
        self.__feature_meta_feature_names.append("target_chi2_p_value")
        self.__feature_meta_feature_names.append("multi_cb_pearson")
        self.__feature_meta_feature_names.append("multi_cb_spearman")
        self.__feature_meta_feature_names.append("multi_cb_kendall")
        self.__feature_meta_feature_names.append("multi_cb_SU")

    @staticmethod
    def __to_feature_vector(double_list: Sequence[np.array]) -> List[float]:
        vector = list()
        for x in double_list:
            try:
                vector.append(x[0])
            except TypeError:
                vector.append(x)
            except IndexError:
                vector.append(x)

        return vector

    def __create_meta_data(self):
        self.__meta_data = DataFrame(
            columns=self.__feature_meta_feature_names,
            data=self.__feature_meta_features,
            index=self.__dataset.get_data_frame().drop(self.__dataset.get_target(), axis=1).columns)

        for i in range(0, len(self.__data_meta_feature_names)):
            self.__meta_data[self.__data_meta_feature_names[i]] = self.__data_meta_features[i]

        self.__meta_data.fillna(0, inplace=True)

    def create_target(self) -> (List[str], float, float, float, float):
        """
        Compute meta-target variables and append results to meta-data set.

        Returns
        -------
            Names and computation times of all meta-target variables.

        """
        perm = PermutationImportance(self.__dataset)
        dCol = DropColumnImportance(self.__dataset)
        shap = ShapImportance(self.__dataset)
        lime = LimeImportance(self.__dataset)

        total_perm = self.add_target(perm)
        total_dCol = self.add_target(dCol)
        total_shap = self.add_target(shap)
        total_lime = self.add_target(lime)

        return self.__targets, total_dCol, total_perm, total_lime, total_shap

    def add_target(self, target: FeatureImportance):
        """
        Compute meta-target values and add them to the meta-data set.

        Parameters
        ----------
        target : The underlying feature importance measure.
        """
        start = time.time()
        target.calculate_scores()
        imp = target.get_feature_importances()
        name = target.get_name()
        target_names = list()

        for i in range(0, len(imp)):
            self.__meta_data.insert(len(self.__meta_data.columns), target.get_model_names()[i] + name, 0.0, True)
            target_names.append(target.get_model_names()[i] + name)
            for x in imp[i].index:
                self.__meta_data.at[x, target.get_model_names()[i] + name] = imp[i].loc[x].iat[0]

        self.__targets = self.__targets + target_names
        end = time.time()
        total = end - start
        return total

    def __symmetrical_uncertainty(self, X: DataFrame, y: DataFrame, matrix=False) -> Union[float, DataFrame]:
        if matrix:
            data = {}
            for feature_1 in X.columns:
                data[feature_1] = {}
                for feature_2 in X.columns:
                    _, values_1 = \
                        self.__run_pymfe(X[feature_1].values, X[feature_2].values, None, ["attr_ent", "mut_inf"])
                    _, values_2 = \
                        self.__run_pymfe(X[feature_2].values, X[feature_1].values, None, ["attr_ent", "mut_inf"])

                    mut_inf = np.mean([values_1[1][0], values_2[1][0]])
                    h_1 = values_1[0][0]
                    h_2 = values_2[0][0]
                    if h_1 == 0.0 and h_2 == 0.0:
                        data[feature_1][feature_2] = 1
                    else:
                        data[feature_1][feature_2] = (2 * mut_inf) / (h_1 + h_2)

            return DataFrame(data=data, columns=X.columns, index=X.columns)

        _, values = self.__run_pymfe(X.values, y.values, None, ["attr_ent", "class_ent", "mut_inf"])
        su = (2 * values[2][0]) / (values[0][0] + values[1])

        return su
