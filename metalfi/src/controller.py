import os
import sys

from multiprocessing import Pool
import tqdm
import pandas as pd

from pandas import DataFrame
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from metalfi.src.metadata.dataset import Dataset
from metalfi.src.memory import Memory
from metalfi.src.metadata.metadataset import MetaDataset
from metalfi.src.model.evaluation import Evaluation
from metalfi.src.model.featureselection import MetaFeatureSelection
from metalfi.src.model.metamodel import MetaModel
from metalfi.src.parameters import Parameters


class Controller:

    def __init__(self):
        self.__train_data = None
        self.__data_names = None
        self.__meta_data = list()
        self.fetchData()
        self.storeMetaData()

    def getTrainData(self):
        return self.__train_data

    def fetchData(self):
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

        self.__data_names = dict({})
        i = 0
        for data, name in self.__train_data:
            self.__data_names[name] = i
            i += 1

    def storeMetaData(self):
        data = [(dataset, name) for dataset, name in self.__train_data
                if not (Memory.getPath() / ("input/" + name + "meta.csv")).is_file()]

        with Pool(processes=4) as pool:
            progress_bar = tqdm.tqdm(total=len(data), desc="Computing meta-data")
            [pool.apply_async(self.parallel_meta_computation, (args, ), callback=(lambda x: progress_bar.update(n=1)))
             for args in data]

            pool.close()
            pool.join()

        progress_bar.close()

    @staticmethod
    def parallel_meta_computation(data):
        result = MetaDataset(data, train=True)
        meta_data = result.getMetaData()
        name = result.getName()
        d_times, t_times = result.getTimes()
        nr_feat, nr_inst = result.getNrs()

        Memory.storeInput(meta_data, name)
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
            fs = MetaFeatureSelection(pd.concat(data))
            sets = {}

            for meta_model, name, _ in Parameters.meta_models:
                tree = (name == "RF")
                percentiles = [10]
                if memory:
                    tree = False
                    percentiles = [25]
                sets[name] = fs.select(meta_model, f_regression, percentiles, k=5, tree=tree)

        if memory:
            Memory.storeMetaFeatures(sets)

        return sets

    def trainMetaModel(self):
        self.loadMetaData()

        parameters = [(
            pd.concat([x[0] for x in self.__meta_data if x[1] != test_name]),
            test_name + "meta",
            test_data,
            self.__train_data[self.__data_names[test_name]][0])
            for test_data, test_name in self.__meta_data
            if not (Memory.getPath() / ("model/" + test_name)).is_file()]

        selection_results = [self.selectMetaFeatures(parameter[1][:-4]) for parameter in parameters]
        args = list(map(lambda x: (*x[0], x[1]), zip(parameters, selection_results)))

        with Pool(processes=4) as pool:
            progress_bar = tqdm.tqdm(total=len(args), desc="Training meta-models")
            [pool.apply_async(self.parallel_training, (arg, ), callback=(lambda x: progress_bar.update(n=1)))
             for arg in args]

            pool.close()
            pool.join()

        progress_bar.close()

    def parallel_training(self, iterable):
        model = MetaModel(iterable)
        sys.stderr = open(os.devnull, 'w')
        sys.stdout.close()
        model.fit()
        sys.stderr = sys.__stderr__
        Memory.storeModel(model, iterable[1][:-4], None)

    def evaluate(self, names):
        evaluation = Evaluation(names)
        evaluation.predictions()

    def questions(self, names):
        evaluation = Evaluation(names)
        evaluation.questions()

    def compare(self, names):
        evaluation = Evaluation(names)
        meta_model, meta_targets, meta_features = Parameters.question_5_parameters()
        evaluation.comparisons(meta_model, meta_targets, meta_features, False)

    def metaFeatureImportances(self):
        data = [d for d, _ in self.__meta_data]
        models = Parameters.meta_models
        computed = list(map(lambda x: x[:-4], filter(lambda x: x.endswith(".csv"),
                                                     Memory.getContents("output/importance"))))

        targets = list(filter(lambda x: x.endswith("_SHAP") and x not in computed, Parameters.targets))
        importance = MetaFeatureSelection.metaFeatureImportance(pd.concat(data), models, targets,
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
            data = DataFrame(data=this_target, index=index, columns=["mean absolute SHAP"])
            data.index.name = "meta-features"
            Memory.storeDataFrame(data, target, "importance")

    def loadModel(self, names):
        return Memory.loadModel(names)
