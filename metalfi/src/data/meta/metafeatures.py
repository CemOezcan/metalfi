from pymfe.mfe import MFE


class MetaFeatures:

    def __init__(self, dataset):
        self.__dataset = dataset
        self.__meta_feature_vector = list()
        self.__feature_meta_features = list()
        self.__data_meta_features = list()
        self.__data_meta_feature_names = list()
        self.__feature_meta_feature_names = list()

    def calculateMetaFeatures(self):
        self.featureMetaFeatures()
        self.dataMetaFeatures()

    def run(self, X, y, summary, features):
        mfe = MFE(summary=summary,
                  groups=["general", "statistical", "info-theory"],
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
                                  ["min", "median", "max", "sd", "kurtosis", "skewness"],
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

    def toFeatureVector(self, double_list):
        vector = list()

        for x in double_list:
            vector.append(x[0])

        return vector
