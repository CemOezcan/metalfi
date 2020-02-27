import statistics

import numpy as np

from pandas import DataFrame
from pymfe.mfe import MFE
from sklearn.feature_selection import f_classif, mutual_info_classif

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
        self.featureMetaFeatures()
        self.dataMetaFeatures()
        self.createMetaData()

    def run(self, X, y, summary, features):
        mfe = MFE(summary=summary,
                  groups=["general", "statistical", "info-theory", "model-based"],
                  features=features)
        mfe.fit(X, y)
        vector = mfe.extract()

        return vector

    def dataMetaFeatures(self):
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

    def featureMetaFeatures(self):
        data_frame = self.__dataset.getDataFrame()
        target = self.__dataset.getTarget()

        y = data_frame[target]
        data_frame.drop(target, axis=1)

        columns = list()

        for feature in data_frame.columns:
            X = data_frame[feature]

            columns, values = self.run(X.values, y.values, None,
                                       ["iq_range", "kurtosis", "mad", "max", "mean", "median", "min",
                                        "range", "sd", "skewness", "sparsity", "t_mean", "var", "attr_ent",
                                        "joint_ent", "mut_inf", "nr_norm", "nr_outliers", "nr_cat", "nr_bin", "nr_num",
                                        "var_importance"])

            self.__feature_meta_features.append(self.toFeatureVector(values))

        self.__feature_meta_feature_names = columns
        self.filterScores(data_frame, target)

        cov = data_frame.cov()
        p_cor = data_frame.corr("pearson")
        s_cor = data_frame.corr("spearman")
        k_cor = data_frame.corr("kendall")

        self.correlationFeatureMetaFeatures(cov, "_cov")
        self.correlationFeatureMetaFeatures(p_cor, "_p_corr")
        self.correlationFeatureMetaFeatures(s_cor, "_s_corr")
        self.correlationFeatureMetaFeatures(k_cor, "_k_corr")

    def correlationFeatureMetaFeatures(self, matrix, name):
        for i in range(0, len(matrix.columns)):
            values = list()
            for j in range(0, len(matrix.columns)):
                if not (i == j):
                    values.append(abs(matrix.iloc[i].iloc[j]))

            self.__feature_meta_features[i] += [statistics.mean(values), statistics.median(values),
                                                statistics.stdev(values), statistics.variance(values),
                                                max(values), min(values), np.percentile(values, 75),
                                                sum(map(lambda x: x > 0.8, values)),
                                                sum(map(lambda x: x > 0.8, values)) / len(values)]

        self.__feature_meta_feature_names += ["mean" + name, "median" + name, "sd" + name, "var" + name, "max" + name,
                                              "min" + name, "quantile_0,75" + name, "high_corr" + name,
                                              "high_corr_norm" + name]

    def filterScores(self, data, target):
        f_values, anova_p_values = f_classif(data.drop(target, axis=1), data[target])
        mut_info = mutual_info_classif(data.drop(target, axis=1), data[target])

        for feature in data.drop(target, axis=1).columns:
            loc = data.columns.get_loc(feature)

            self.__feature_meta_features[loc].append(data[feature].corr(data[target], method="pearson"))
            self.__feature_meta_features[loc].append(data[feature].corr(data[target], method="kendall"))
            self.__feature_meta_features[loc].append(data[feature].corr(data[target], method="spearman"))
            self.__feature_meta_features[loc].append(f_values[loc])
            self.__feature_meta_features[loc].append(mut_info[loc])

        self.__feature_meta_feature_names.append("target_p_corr")
        self.__feature_meta_feature_names.append("target_k_corr")
        self.__feature_meta_feature_names.append("target_s_corr")
        self.__feature_meta_feature_names.append("target_F_value")
        self.__feature_meta_feature_names.append("target_mut_info")

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
                                     index=self.__dataset.getDataFrame().columns)

        for i in range(0, len(self.__data_meta_feature_names)):
            self.__meta_data[self.__data_meta_feature_names[i]] = self.__data_meta_features[i]

        self.__meta_data = self.__meta_data.drop(self.__dataset.getTarget())

    def createTarget(self):
        perm = PermutationImportance(self.__dataset)
        dCol = DropColumnImportance(self.__dataset)
        shap = ShapImportance(self.__dataset)
        lime = LimeImportance(self.__dataset)

        self.addTarget(perm)
        self.addTarget(dCol)
        self.addTarget(shap)
        self.addTarget(lime)

        return self.__targets

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
