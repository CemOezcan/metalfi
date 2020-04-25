import statistics
import time

import numpy as np

from pandas import DataFrame
from pymfe.mfe import MFE
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

from metalfi.src.data.meta.importance.dropcolumn import DropColumnImportance
from metalfi.src.data.meta.importance.lime import LimeImportance
from metalfi.src.data.meta.importance.permutation import PermutationImportance
from metalfi.src.data.meta.importance.shap import ShapImportance


class MetaFeatures:

    def __init__(self, dataset):
        self.__dataset = dataset
        self.__meta_data = DataFrame()
        self.__targets = list()
        self.__feature_meta_features = list()
        self.__data_meta_features = list()
        self.__data_meta_feature_names = list()
        self.__feature_meta_feature_names = list()

    def getMetaData(self):
        return self.__meta_data

    def calculateMetaFeatures(self):
        uni_time, multi_time, lm_time = self.featureMetaFeatures()
        data_time = self.dataMetaFeatures()
        self.createMetaData()

        return data_time, uni_time, multi_time, lm_time

    def run(self, X, y, summary, features):
        mfe = MFE(summary=summary,
                  groups=["general", "statistical", "info-theory", "model-based"],
                  features=features)
        mfe.fit(X, y)
        vector = mfe.extract()

        return vector

    def dataMetaFeatures(self):
        start = time.time()
        data_frame = self.__dataset.getDataFrame()
        target = self.__dataset.getTarget()

        X = data_frame.drop(target, axis=1)
        y = data_frame[target]

        names_1, dmf_1 = self.run(X.values, y.values,
                                  ["min", "median", "max", "sd", "kurtosis", "skewness", "mean"],
                                  ["attr_to_inst", "freq_class", "inst_to_attr", "nr_attr", "nr_class", "nr_inst",
                                   "gravity", "cor", "cov", "nr_disc", "eigenvalues", "nr_cor_attr", "w_lambda",
                                   "class_ent", "eq_num_attr", "ns_ratio", "iq_range", "kurtosis", "mad",
                                   "max", "mean", "median", "min", "range", "sd", "skewness", "sparsity", "t_mean",
                                   "var", "attr_ent", "class_conc", "joint_ent", "mut_inf", "nr_norm", "nr_outliers",
                                   "nr_cat", "nr_bin", "nr_num"])

        self.__data_meta_feature_names = names_1
        self.__data_meta_features = dmf_1
        end = time.time()

        return end - start

    def featureMetaFeatures(self):
        start_uni = time.time()
        data_frame = self.__dataset.getDataFrame()
        target = self.__dataset.getTarget()

        y = data_frame[target]
        X = data_frame.drop(target, axis=1)
        columns = list()

        for feature in X.columns:
            X_temp = data_frame[feature]

            columns, values = self.run(X_temp.values, y.values, None,
                                       ["iq_range", "kurtosis", "mad", "max", "mean", "median", "min",
                                        "range", "sd", "skewness", "sparsity", "t_mean", "var", "attr_ent", "nr_norm",
                                        "nr_outliers", "nr_cat", "nr_bin", "nr_num"])

            self.__feature_meta_features.append(self.toFeatureVector(values))

        self.__feature_meta_feature_names = columns
        end_uni = time.time()
        total_uni = end_uni - start_uni

        start_multi = time.time()
        cov = X.cov()
        p_cor = X.corr("pearson")
        s_cor = X.corr("spearman")
        k_cor = X.corr("kendall")
        su_cor = None

        self.correlationFeatureMetaFeatures(cov, "_cov")
        self.correlationFeatureMetaFeatures(p_cor, "_p_corr")
        self.correlationFeatureMetaFeatures(s_cor, "_s_corr")
        self.correlationFeatureMetaFeatures(k_cor, "_k_corr")
        end_multi = time.time()
        total_multi = end_multi - start_multi

        start_lm = time.time()
        columns, values = self.run(X.values, y.values, None, ["joint_ent", "mut_inf", "var_importance"])
        for feature in X.columns:
            loc = X.columns.get_loc(feature)
            self.__feature_meta_features[loc].append(values[0][loc])
            self.__feature_meta_features[loc].append(values[1][loc])
            self.__feature_meta_features[loc].append(values[2][loc])

        self.__feature_meta_feature_names += columns

        self.filterScores(X, y, p_cor, s_cor, k_cor, su_cor)
        end_lm = time.time()
        total_lm = end_lm - start_lm

        return total_uni, total_multi, total_lm

    def correlationFeatureMetaFeatures(self, matrix, name, threshold=0.8):
        for i in range(0, len(matrix.columns)):
            values = list()
            for j in range(0, len(matrix.columns)):
                if not (i == j):
                    values.append(abs(matrix.iloc[i].iloc[j]))

            percentile = np.percentile(values, 75)
            th = map(lambda x: x > threshold, values)
            th_list = list(map(lambda x: x if (x > threshold) else 0, values))
            percentile_list = list(map(lambda x: x if (x >= percentile) else 0, values))

            self.__feature_meta_features[i] += [statistics.mean(values), statistics.median(values),
                                                statistics.stdev(values), statistics.variance(values),
                                                max(values), min(values), percentile, statistics.mean(percentile_list),
                                                sum(th), sum(th) / len(values), statistics.mean(th_list)]

        self.__feature_meta_feature_names += ["mean" + name, "median" + name, "sd" + name, "var" + name, "max" + name,
                                              "min" + name, "percentile_0,75" + name, "mean_percentile_0,75" + name,
                                              "high_corr" + name, "high_corr_norm" + name, "mean_high_corr" + name]

    def filterScores(self, X, y, p_cor, s_cor, k_cor, su_cor):
        sc = MinMaxScaler()
        X_sc = sc.fit_transform(X)

        chi2_values, chi2_p_values = chi2(X_sc, y)
        f_values, anova_p_values = f_classif(X, y)
        log_anova_p = [np.log(x) for x in anova_p_values]
        log_chi2_p = [np.log(x) for x in chi2_p_values]
        mut_info = mutual_info_classif(X, y)

        for feature in X.columns:
            loc = X.columns.get_loc(feature)

            self.__feature_meta_features[loc].append(X[feature].corr(y, method="pearson"))
            self.__feature_meta_features[loc].append(X[feature].corr(y, method="kendall"))
            self.__feature_meta_features[loc].append(X[feature].corr(y, method="spearman"))
            self.__feature_meta_features[loc].append(f_values[loc])
            self.__feature_meta_features[loc].append(log_anova_p[loc])
            self.__feature_meta_features[loc].append(mut_info[loc])
            self.__feature_meta_features[loc].append(chi2_values[loc])
            self.__feature_meta_features[loc].append(log_chi2_p[loc])

        self.__feature_meta_feature_names.append("target_p_corr")
        self.__feature_meta_feature_names.append("target_k_corr")
        self.__feature_meta_feature_names.append("target_s_corr")
        self.__feature_meta_feature_names.append("target_F_value")
        self.__feature_meta_feature_names.append("target_anova_p_value")
        self.__feature_meta_feature_names.append("target_mut_info")
        self.__feature_meta_feature_names.append("target_chi2")
        self.__feature_meta_feature_names.append("target_chi2_p_value")

    def toFeatureVector(self, double_list):
        vector = list()
        for x in double_list:
            try:
                vector.append(x[0])
            except TypeError:
                vector.append(x)
            except IndexError:
                vector.append(x)

        return vector

    def createMetaData(self):
        self.__meta_data = DataFrame(columns=self.__feature_meta_feature_names,
                                     data=self.__feature_meta_features,
                                     index=self.__dataset.getDataFrame().drop(self.__dataset.getTarget(), axis=1).columns)

        for i in range(0, len(self.__data_meta_feature_names)):
            self.__meta_data[self.__data_meta_feature_names[i]] = self.__data_meta_features[i]

    def createTarget(self):
        perm = PermutationImportance(self.__dataset)
        dCol = DropColumnImportance(self.__dataset)
        shap = ShapImportance(self.__dataset)
        lime = LimeImportance(self.__dataset)

        start_perm = time.time()
        self.addTarget(perm)
        end_perm = time.time()
        total_perm = end_perm - start_perm

        start_dCol = time.time()
        self.addTarget(dCol)
        end_dCol = time.time()
        total_dCol = end_dCol - start_dCol

        start_shap = time.time()
        self.addTarget(shap)
        end_shap = time.time()
        total_shap = end_shap - start_shap

        start_lime = time.time()
        self.addTarget(lime)
        end_lime = time.time()
        total_lime = end_lime - start_lime

        return self.__targets, total_dCol, total_perm, total_lime, total_shap

    def addTarget(self, target):
        target.calculateScores()
        imp = target.getFeatureImportances()
        name = target.getName()
        target_names = list()

        for i in range(0, len(imp)):
            self.__meta_data.insert(len(self.__meta_data.columns), target.getModelNames()[i] + name, 0.0, True)
            target_names.append(target.getModelNames()[i] + name)
            for x in imp[i].index:
                self.__meta_data.at[x, target.getModelNames()[i] + name] = imp[i].loc[x].iat[0]

        self.__targets = self.__targets + target_names
