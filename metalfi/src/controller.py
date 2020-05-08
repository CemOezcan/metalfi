import pandas as pd

from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory
from metalfi.src.data.metadataset import MetaDataset
from metalfi.src.model.evaluation import Evaluation
from metalfi.src.model.featureselection import MetaFeatureSelection
from metalfi.src.model.metamodel import MetaModel


class Controller:

    def __init__(self):
        self.__train_data = None
        self.__enum = None
        self.__meta_data = list()
        self.downloadData()
        self.storeMetaData()

        self.__targets = ["lda_shap", "linSVC_shap", "log_shap", "rf_shap", "nb_shap", "svc_shap",
                          "lda_lime", "linSVC_lime", "log_lime", "rf_lime", "nb_lime", "svc_lime",
                          "lda_perm", "linSVC_perm", "log_perm", "rf_perm", "nb_perm", "svc_perm",
                          "lda_dCol", "linSVC_dCol", "log_dCol", "rf_dCol", "nb_dCol", "svc_dCol"]

        self.__meta_models = [(RandomForestRegressor(n_estimators=100, n_jobs=4), "Rf"),
                              (SVR(), "Svr"),
                              (LinearRegression(n_jobs=4), "lin"),
                              (LinearSVR(dual=True, max_iter=10000), "linSVR")]

    def getTrainData(self):
        return self.__train_data

    def downloadData(self):
        data_frame, target = Memory.loadTitanic()
        data_1 = Dataset(data_frame, target)

        data_frame_2, target_2 = Memory.loadCancer()
        data_2 = Dataset(data_frame_2, target_2)

        data_frame_3, target_3 = Memory.loadIris()
        data_3 = Dataset(data_frame_3, target_3)

        data_frame_4, target_4 = Memory.loadWine()
        data_4 = Dataset(data_frame_4, target_4)

        data_frame_5, target_5 = Memory.loadBoston()
        data_5 = Dataset(data_frame_5, target_5)

        open_ml = [(Dataset(data_frame, target), name) for data_frame, name, target in Memory.loadOpenML()]

        self.__train_data = [(data_1, "Titanic"), (data_2, "Cancer"), (data_3, "Iris"), (data_4, "Wine"),
                             (data_5, "Boston")] + open_ml

        self.__enum = dict({})
        i = 0
        for data, name in self.__train_data:
            self.__enum[name] = i
            i += 1

    def storeMetaData(self):
        for dataset, name in self.__train_data:
            if not (Memory.getPath() / ("input/" + name + "meta.csv")).is_file():
                print("meta-data calc.: " + name)
                meta = MetaDataset([dataset], True)
                data = meta.getMetaData()
                d_times, t_times = meta.getTimes()
                nr_feat, nr_inst = meta.getNrs()
                Memory.storeInput(data, name)
                Memory.storeDataFrame(DataFrame(data=d_times, index=["Time"], columns=[x for x in d_times]),
                                      name + "XmetaX" + str(nr_feat) + "X" + str(nr_inst), "runtime")
                Memory.storeDataFrame(DataFrame(data=t_times, index=["Time"], columns=[x for x in t_times]),
                                      name + "XtargetX" + str(nr_feat) + "X" + str(nr_inst), "runtime")

    def loadMetaData(self):
        for dataset, name in self.__train_data:
            sc = StandardScaler()
            data = Memory.load(name + "meta.csv", "input")
            fmf = [x for x in data.columns if "." not in x]
            dmf = [x for x in data.columns if "." in x]

            X_f = DataFrame(data=sc.fit_transform(data[fmf]), columns=fmf)
            X_d = DataFrame(data=data[dmf], columns=dmf)

            data_frame = pd.concat([X_d, X_f], axis=1)

            self.__meta_data.append((data_frame, name))

    def selectMetaFeatures(self, meta_model_name="", memory=False):
        sets = None
        if memory:
            sets = Memory.loadMetaFeatures()

        if sets is None:
            data = [d for d, n in self.__meta_data if n != meta_model_name]
            fs = MetaFeatureSelection(pd.concat(data), self.__targets)
            sets = {}

            for meta_model, name in self.__meta_models:
                print("Select meta-features: " + name)
                tree = (name == "Rf")
                percentiles = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30) if (name == "Svr") \
                    else (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)
                if memory:
                    percentiles = (15, 15)

                sets[name] = fs.select(meta_model, f_regression, percentiles, k=10, tree=tree)

        if memory:
            Memory.storeMetaFeatures(sets)

        return sets

    def trainMetaModel(self):
        self.loadMetaData()

        for i in range(0, len(self.__meta_data)):
            test_data, test_name = self.__meta_data[i]
            train_data = list()

            for j in range(0, len(self.__meta_data)):
                if not (i == j):
                    train_data.append(self.__meta_data[j][0])

            path = Memory.getPath() / ("model/" + test_name)
            if not path.is_file():
                print("Train meta-model: " + test_name)
                og_data, name = self.__train_data[self.__enum[test_name]]
                model = MetaModel(pd.concat(train_data), test_name + "meta",
                                  test_data, og_data, self.selectMetaFeatures(test_name),
                                  self.__meta_models, self.__targets)
                model.fit()
                Memory.storeModel(model, test_name, None)

    def evaluate(self, names):
        evaluation = Evaluation(names)
        evaluation.predictions()

    def compare(self, names):
        evaluation = Evaluation(names)
        evaluation.comparisons(["lin", "Svr", "Rf", "linSVR"],
                               ["linSVC_shap", "log_shap", "rf_shap", "nb_shap", "svc_shap"], ["Auto"], True)

    def metaFeatureImportances(self):
        data = [d for d, _ in self.__meta_data]
        models = [(RandomForestRegressor(n_estimators=100, n_jobs=4), "Rf", "tree"),
                  (SVR(), "Svr", "kernel"),
                  (LinearRegression(n_jobs=4), "lin", "linear"),
                  (LinearSVR(dual=True, max_iter=10000), "linSVR", "linear")]
        targets = ["lda_shap", "linSVC_shap", "log_shap", "rf_shap", "nb_shap", "svc_shap"]
        importance = MetaFeatureSelection.metaFeatureImportance(pd.concat(data), self.__targets, models, targets,
                                                                self.selectMetaFeatures(memory=True))

        meta_features = {}
        for target in targets:
            meta_features[target] = set()
            for data_frame in importance[target]:
                meta_features[target].update(data_frame.index)

        for target in targets:
            this_target = {}
            index = []
            imp = []
            for meta_feature in meta_features[target]:
                this_meta_feature = 0
                index.append(meta_feature)
                for data_frame in importance[target]:
                    if meta_feature in data_frame.index:
                        this_meta_feature += data_frame.loc[meta_feature].iloc[0]

                this_meta_feature /= len(importance[target])
                imp.append(this_meta_feature)

            this_target["mean absolute SHAP"] = imp
            this_target["meta-features"] = index
            # or index=index
            Memory.storeDataFrame(DataFrame(data=this_target, columns=["meta-features", "mean absolute SHAP"]),
                                  target, "importance")

    def loadModel(self, names):
        return Memory.loadModel(names)
