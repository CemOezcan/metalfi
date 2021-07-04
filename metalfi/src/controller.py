import multiprocessing as mp
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
import tqdm

from metalfi.src.metadata.dataset import Dataset
from metalfi.src.metadata.metadataset import MetaDataset
from metalfi.src.memory import Memory
from metalfi.src.model.evaluation import Evaluation
from metalfi.src.model.featureselection import MetaFeatureSelection
from metalfi.src.model.metamodel import MetaModel
from metalfi.src.parameters import Parameters


class Controller:
    """
    Determine parameters for major computations and handle their results.

    Attributes
    ----------
        # TODO: Adjust
        __train_data : (List[Tuple[Dataset, str]])
            Base-data sets and their names
        __data_names : (Dict[str, int])
            Names of all base-data sets
        __meta_data : (List[DataFrame, name])
            Meta-data sets and their names
    """

    def __init__(self):
        self.__train_data = None
        self.__test_data = None
        self.__data_names = None
        self.__meta_data = list()
        self.fetch_data()
        self.store_meta_data()

    def fetch_data(self):
        """
        Fetch base-data sets from openMl and scikit-learn
        """
        self.__train_data = [(Dataset(data_frame, target), name) for data_frame, name, target in Memory.load_open_ml()]
        self.__test_data = [(Dataset(data_frame, target), name) for data_frame, name, target in Memory.load_open_ml(large=True)]
        self.__data_names = dict({})
        i = 0
        for _, name in self.__train_data:
            self.__data_names[name] = i
            i += 1

    def store_meta_data(self):
        """
        Parallel computation of meta-data sets from base-data sets.
        """
        data = [(dataset, name) for dataset, name in self.__train_data
                if not (Memory.get_path() / ("input/" + name + "meta.csv")).is_file()]

        with mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1) as pool:
            progress_bar = tqdm.tqdm(total=len(data), desc="Computing meta-data")
            results = [pool.map_async(self.parallel_meta_computation, (args,),
                                      callback=(lambda x: progress_bar.update())) for args in data]

            results = [x.get()[0] for x in results]
            pool.close()
            pool.join()

        progress_bar.close()
        d_times = list()
        t_times = list()
        for d_time, t_time, name in results:
            d_times.append(pd.DataFrame(data=d_time, index=[name], columns=[x for x in d_times]))
            t_times.append(pd.DataFrame(data=d_time, index=[name], columns=[x for x in t_times]))
        Memory.store_data_frame(pd.concat(d_times), "meta-features", "runtime")
        Memory.store_data_frame(pd.concat(t_times), "meta-targets", "runtime")

    @staticmethod
    def parallel_meta_computation(data: Tuple[pd.DataFrame, str]):
        """
        Computes a meta-data set, given a base-data set.

        Parameters
        ----------
            data : Parameters for instantiating :py:class:`MetaDataset`.
        """
        result = MetaDataset(data, train=True)
        meta_data = result.get_meta_data()
        name = result.get_name()
        d_times, t_times = result.get_times()
        nr_feat, nr_inst = result.get_nrs()

        Memory.store_input(meta_data, name)
        return d_times, t_times, name

    def __load_meta_data(self):
        for _, name in self.__train_data:
            sc = StandardScaler()
            data = Memory.load(name + "meta.csv", "input")
            fmf = [x for x in data.columns if "." not in x]
            dmf = [x for x in data.columns if "." in x]

            X_f = pd.DataFrame(data=sc.fit_transform(data[fmf]), columns=fmf)
            X_d = pd.DataFrame(data=data[dmf], columns=dmf)

            data_frame = pd.concat([X_d, X_f], axis=1)
            self.__meta_data.append((data_frame, name))

    def __select_meta_features(self, meta_model_name=""):
        # Selects features for meta-data with `meta_model_name` as test and all other meta-data sets as train
        data = [d for d, n in self.__meta_data if n != meta_model_name]
        fs = MetaFeatureSelection(pd.concat(data))
        sets = {}

        for meta_model, name, _ in Parameters.meta_models:
            sets[name] = fs.select(meta_model, f_regression, [10], k=5)

        return sets

    def train_meta_models(self):
        """
        Partition meta-data into train-test splits based on base-data sets and
        fit instances of class :py:class:`MetaModel` using the train split.
        Save the trained meta-models as instances of class :py:class:`MetaModel` in metalfi/data/model.
        """
        self.__load_meta_data()

        parameters = [(
            pd.concat([x[0] for x in self.__meta_data if x[1] != test_name]),
            test_name,
            test_data,
            self.__train_data[self.__data_names[test_name]][0])
            for test_data, test_name in self.__meta_data
            if not (Memory.get_path() / ("model/" + test_name)).is_file()]

        selection_results = [self.__select_meta_features(parameter[1][:-4]) for parameter in parameters]
        args = list(map(lambda x: (*x[0], x[1]), zip(parameters, selection_results)))

        with mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1) as pool:
            progress_bar = tqdm.tqdm(total=len(args), desc="Training meta-models")
            _ = [pool.apply_async(self.parallel_training, (arg,), callback=(lambda x: progress_bar.update()))
                 for arg in args]

            pool.close()
            pool.join()

        progress_bar.close()

    @staticmethod
    def parallel_training(iterable: Tuple[pd.DataFrame, str, pd.DataFrame, Dataset, Dict[str, Dict[str, List[str]]]]):
        """
        Create an instance of class :py:class:`MetaModel` and call its :py:func:`MetaModel.fit()` function.

        Parameters
        ----------
            iterable : Parameters for the instance of :py:class:`MetaModel`.
        """
        model = MetaModel(iterable)
        model.fit()
        Memory.store_model(model, iterable[1])

    @staticmethod
    def estimate(names: List[str]):
        """
        Estimate meta-model performances. Use file names in `names` to identify these meta-models.

        Parameters
        ----------
        names :
            File names referring to instances of class :py:class:`MetaModel`,
            whose meta-models are supposed to be tested.
        """
        evaluation = Evaluation(names)
        evaluation.predictions()
        evaluation.new_comparisons()

    @staticmethod
    def questions(names: List[str]):
        """
        Create and save .csv files that contain the estimated meta-model performances.
        Use file names in `names` to identify these meta-models.

        Parameters
        ----------
            names : File names referring to instances of class :py:class:`MetaModel`,
                    whose meta-model performances are supposed to be summarized and saved as .csv files.
        """
        evaluation = Evaluation(names)
        evaluation.questions()

    def compare_all(self):
        self.__load_meta_data()

        parameters = [(
            pd.concat([x[0] for x in self.__meta_data]),
            "Comp",
            self.__meta_data[0][0],
            self.__train_data[0][0])]

        selection_results = [self.__select_meta_features("")]
        args = list(map(lambda x: (*x[0], x[1]), zip(parameters, selection_results)))

        m = MetaModel(args[0])
        m.compare_all(self.__test_data)
        evaluation = Evaluation(["all"])
        evaluation.new_comparisons(m)

    @staticmethod
    def compare(names: List[str]):
        """
        Compare MetaLFI to other feature selection approaches on base-data sets.
        Use file names in `names` to identify these base-data sets.

        Parameters
        ----------
            names :
                File names referring to instances of class :py:class:`MetaModel`,
                whose underlying base-data set is supposed to be used to compare feature selection approaches.
        """
        evaluation = Evaluation(names)
        meta_model, meta_targets, meta_features = Parameters.question_5_parameters()
        evaluation.comparisons(meta_model, meta_targets, meta_features, True)

    def meta_feature_importances(self):
        """
        Estimate the importance of each meta-feature. Save the results as .csv files in metalfi/data/output/importance.
        """
        data = [d for d, _ in self.__meta_data]
        models = Parameters.meta_models
        computed = list(map(lambda x: x[:-4], filter(lambda x: x.endswith(".csv"),
                                                     Memory.get_contents("output/importance"))))

        targets = list(filter(lambda x: x.endswith("_SHAP") and x not in computed, Parameters.targets))
        importance = MetaFeatureSelection.meta_feature_importance(pd.concat(data), models, targets,
                                                                  self.__select_meta_features())

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
            data = pd.DataFrame(data=this_target, index=index, columns=["mean absolute SHAP"])
            data.index.name = "meta-features"
            Memory.store_data_frame(data, target, "importance")
