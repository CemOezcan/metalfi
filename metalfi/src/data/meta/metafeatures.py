import statistics
import sys
import pandas as pd
from pandas import DataFrame, Series
from pymfe.mfe import MFE

from metalfi.src.data.meta.importance.dropcolumn import DropColumnImportance
from metalfi.src.data.meta.importance.permutation import PermutationImportance
from metalfi.src.data.meta.importance.shap import ShapImportance


class MetaFeatures:

    def __init__(self, dataset):
        self.__dataset = dataset
        self.__meta_data = DataFrame()
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
                  groups=["general", "statistical", "info-theory"],
                  features=features)
        mfe.fit(X, y)
        vector = mfe.extract()

        return vector

    def dataMetaFeatures(self):
        # TODO: Drop target
        data_frame = self.__dataset.getDataFrame()
        target = self.__dataset.getTarget()

        X = data_frame.drop(target, axis=1)
        y = data_frame[target]

        names_1, dmf_1 = self.run(X.values, y.values,
                                  ["min", "median", "max", "sd", "kurtosis", "skewness", "mean"],
                                  ["attr_to_inst", "freq_class", "inst_to_attr", "nr_attr", "nr_class", "nr_inst",
                                   "gravity", "cor", "cov", "nr_disc", "eigenvalues", "nr_cor_attr", "w_lambda",
                                   "class_ent", "eq_num_attr", "ns_ratio", "h_mean", "iq_range", "kurtosis", "mad",
                                   "max", "mean", "median", "min", "range", "sd", "skewness", "sparsity", "t_mean",
                                   "var", "attr_ent", "class_conc", "joint_ent", "mut_inf", "nr_norm", "nr_outliers"])

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
                                       ["h_mean", "iq_range", "kurtosis", "mad", "max", "mean", "median", "min",
                                        "range", "sd", "skewness", "sparsity", "t_mean", "var", "attr_ent",
                                        "class_conc", "joint_ent", "mut_inf"])
            # TODO: implement is_norm (or use nr_norm as is_norm), nr_outliers and accumulate for dmf
            # TODO: calc. landmarking and other mfs, that depend on other features
            self.__feature_meta_features.append(self.toFeatureVector(values))

        self.__feature_meta_feature_names = columns

        cov = data_frame.cov()
        p_cor = data_frame.corr("pearson")
        s_cor = data_frame.corr("spearman")

        self.correlationFeatureMetaFeatures(cov, "_cov")
        self.correlationFeatureMetaFeatures(p_cor, "_p_cor")
        self.correlationFeatureMetaFeatures(s_cor, "_s_cor")

    def correlationFeatureMetaFeatures(self, data, name):
        for i in range(0, len(data.columns)):
            values = list()
            for j in range(0, len(data.columns)):
                if not (i == j):
                    values.append(abs(data.iloc[i].iloc[j]))

            self.__feature_meta_features[i] += [statistics.mean(values), statistics.median(values),
                                                statistics.stdev(values), statistics.variance(values),
                                                max(values), min(values)]

        self.__feature_meta_feature_names += ["mean" + name, "median" + name, "sd" + name, "var" + name,
                                              "max" + name, "min" + name]

    def toFeatureVector(self, double_list):
        vector = list()

        for x in double_list:
            vector.append(x[0])

        return vector

    def createMetaData(self):
        # TODO: Implement other target variables
        dropCol = DropColumnImportance(self.__dataset)
        shap = ShapImportance(self.__dataset)
        perm = PermutationImportance(self.__dataset)

        self.__meta_data = DataFrame(columns=self.__feature_meta_feature_names,
                                     data=self.__feature_meta_features,
                                     index=self.__dataset.getDataFrame().columns)

        for i in range(0, len(self.__data_meta_feature_names)):
            self.__meta_data[self.__data_meta_feature_names[i]] = self.__data_meta_features[i]

        self.__meta_data = self.__meta_data.drop(self.__dataset.getTarget())

        self.addTarget(dropCol)
        self.addTarget(shap)
        self.addTarget(perm)
        # Kpd.set_option('display.max_columns', 210)
        # print(self.__meta_data)

    def addTarget(self, target):
        target.calculateScores()
        imp = target.getFeatureImportances()
        name = target.getName()

        for i in range(0, len(imp)):
            self.__meta_data.insert(len(self.__meta_data.columns), target.getModelNames()[i] + name, 0.0, True)
            for x in imp[i].index:
                self.__meta_data.at[x, target.getModelNames()[i] + name] = imp[i].loc[x].iat[0]
