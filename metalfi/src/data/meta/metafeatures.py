from pymfe.mfe import MFE


class MetaFeatures:

    def __init__(self, dataset):
        self.__dataset = dataset
        self.__meta_feature_vector = list()
        self.__feature_meta_features = list()
        self.__data_meta_features = list()

    def calculateMetaFeatures(self):
        data = self.__dataset.getDataFrame()
        target = self.__dataset.getTarget()

        fmf = self.featureMetaFeatures(data, target)
        #dmf = self.dataMetaFeatures(data, target)

    def run(self, X, y, summary, features):
        mfe = MFE(summary=summary,
                  groups=["general", "statistical", "info-theory"],
                  features=features)

        mfe.fit(X.values, y.values)
        vector = mfe.extract()

        return vector

    def dataMetaFeatures(self, data_frame, target):
        # TODO: apply on raw data?
        pass

    def featureMetaFeatures(self, data_frame, target):
        y = data_frame[target]

        for feature in data_frame.columns:
            X = data_frame[feature]

            vector = self.run(X, y, None,
                              ["h_mean", "iq_range", "kurtosis", "mad", "max", "mean", "median", "min", "nr_norm",
                               "nr_outliers", "range", "sd", "skewness", "sparsity", "t_mean", "var", "attr_ent",
                               "class_conc", "joint_ent", "mut_inf"])
            # TODO: implement is_norm (or use nr_norm as is_norm), nr_outliers and accumulate for dmf
            # TODO: calc. landmarking and append
            print(target)
            print(vector)
            self.__feature_meta_features.append(vector)
            # TODO: return dataframe

    def toDataFrame(self, vector):
        return 5
