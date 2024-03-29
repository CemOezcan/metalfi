import multiprocessing as mp
import os
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
import tqdm

from metalfi.src.metadata.dataset import Dataset
from metalfi.src.metadata.metadataset import MetaDataset
from metalfi.src import memory
from metalfi.src.evaluation import Evaluation
from metalfi.src.featureselection import MetaFeatureSelection
from metalfi.src.metamodel import MetaModel
from metalfi.src.parameters import Parameters


class Controller:
    """
    Determine parameters for major computations and handle their results.

    Attributes
    ----------
        __train_data : (List[Tuple[Dataset, str]])
            Base-data sets and their names
        __data_names : (Dict[str, int])
            Names of all base-data sets
        __meta_data : (List[DataFrame, name])
            Meta-data sets and their names
    """

    def __init__(self):
        self.__train_data = None
        self.__data_names = None
        self.__meta_data = []
        self.fetch_data()
        self.store_meta_data()

    def fetch_data(self) -> None:
        """
        Fetch base-data sets from openMl and scikit-learn
        """
        self.__train_data = [(Dataset(data_frame, target), name) for data_frame, name, target in memory.load_open_ml()]
        self.__data_names = dict({})
        i = 0
        for _, name in self.__train_data:
            self.__data_names[name] = i
            i += 1

    def store_meta_data(self) -> None:
        """
        Parallel computation of meta-data sets from base-data sets.
        """
        data = [(dataset, name) for dataset, name in self.__train_data
                if not os.path.isfile(Parameters.meta_dataset_dir + name + "meta.csv")]

        with mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1) as pool:
            progress_bar = tqdm.tqdm(total=len(data), desc="Computing meta-data")
            results = [pool.apply_async(self.parallel_meta_computation, (args,),
                                        callback=(lambda x: progress_bar.update())) for args in data]
            _ = [x.get() for x in results]

            pool.close()
            pool.join()

        progress_bar.close()

    @staticmethod
    def parallel_meta_computation(data: Tuple[pd.DataFrame, str]) -> None:
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
        del d_times["total"]
        del t_times["total"]
        d_times.update(t_times)
        meta_data.to_csv(Parameters.meta_dataset_dir + name + "meta.csv", index=False)
        memory.update_runtimes(new_runtime_data=d_times, base_dataset_name=name)

    def __load_meta_data(self) -> None:
        for _, name in self.__train_data:
            sc = StandardScaler()
            data = pd.read_csv(Parameters.meta_dataset_dir + name + "meta.csv")
            fmf = [x for x in data.columns if "." not in x]
            dmf = [x for x in data.columns if "." in x]

            X_f = pd.DataFrame(data=sc.fit_transform(data[fmf]), columns=fmf)
            X_d = pd.DataFrame(data=data[dmf], columns=dmf)

            data_frame = pd.concat([X_d, X_f], axis=1)
            self.__meta_data.append((data_frame, name))

    def __select_meta_features(self, meta_model_name: str = "",
                               percentiles: Optional[Sequence[float]] = None) -> Dict[str, Dict[str, List[str]]]:
        if percentiles is None:
            percentiles = [10]
        # Selects features for meta-data with `meta_model_name` as test and all other meta-data sets as train
        data = [d for d, n in self.__meta_data if n != meta_model_name]
        fs = MetaFeatureSelection(pd.concat(data))
        sets = {}

        for meta_model, name, _ in Parameters.meta_models:
            sets[name] = fs.select(meta_model, f_regression, percentiles, k=5)

        return sets

    def train_meta_models(self) -> None:
        """
        Partition meta-data into train-test splits based on base-data sets and
        fit instances of class :py:class:`MetaModel` using the train split.
        Save the trained meta-models as instances of class :py:class:`MetaModel`.
        """
        self.__load_meta_data()

        parameters = [(
            pd.concat([x[0] for x in self.__meta_data if x[1] != test_name]),
            test_name,
            test_data,
            self.__train_data[self.__data_names[test_name]][0])
            for test_data, test_name in self.__meta_data
            if not os.path.isfile(Parameters.meta_model_dir + test_name)]

        selection_results = [self.__select_meta_features(parameter[1][:-4]) for parameter in parameters]
        args = [(*x[0], x[1]) for x in zip(parameters, selection_results)]

        with mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1) as pool:
            progress_bar = tqdm.tqdm(total=len(args), desc="Training meta-models")
            results = [pool.apply_async(self.parallel_training, (arg,), callback=(lambda x: progress_bar.update()))
                       for arg in args]
            _ = [x.get() for x in results]

            pool.close()
            pool.join()

        progress_bar.close()

    @staticmethod
    def parallel_training(iterable: Tuple[pd.DataFrame, str, pd.DataFrame, Dataset,
                                          Dict[str, Dict[str, List[str]]]]) -> None:
        """
        Create an instance of class :py:class:`MetaModel` and call its :py:func:`MetaModel.fit()` function.

        Parameters
        ----------
            iterable : Parameters for the instance of :py:class:`MetaModel`.
        """
        model = MetaModel(iterable)
        model.fit()
        memory.store_model(model, model_name=iterable[1])

    @staticmethod
    def estimate(names: List[str]) -> None:
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

    def compare_all(self) -> None:
        # Deprecated
        self.__load_meta_data()

        parameters = [(
            pd.concat([x[0] for x in self.__meta_data]),
            "Comp",
            self.__meta_data[0][0],
            self.__train_data[0][0])]

        selection_results = [self.__select_meta_features("")]
        args = [(*x[0], x[1]) for x in zip(parameters, selection_results)]

        m = MetaModel(args[0])
        m.compare_all(self.__train_data)
        evaluation = Evaluation(["all"])
        evaluation.new_comparisons(m)

    def meta_feature_importances(self) -> None:
        """
        Estimate the importance of each meta-feature. Save the results as .csv files.
        """
        data = [d for d, _ in self.__meta_data]
        models = Parameters.meta_models
        targets = Parameters.targets
        importance = MetaFeatureSelection.meta_feature_importance(
            pd.concat(data), models, targets, self.__select_meta_features(percentiles=[100]))

        meta_features = {}
        for target in targets:
            meta_features[target] = set()
            for data_frame in importance[target]:
                meta_features[target].update(data_frame.index)

        data_frames = []
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

            this_target["PIMP"] = imp
            data = pd.DataFrame(data=this_target, index=index, columns=["PIMP"])
            data.index.name = "meta_feature"
            data["base_model"] = [target[:-5]] * len(index)
            data["importance_measure"] = [target[-4:]] * len(index)
            data_frames.append(data.sort_values(["PIMP", "base_model", "importance_measure"]))

        pd.concat(data_frames).to_csv(Parameters.output_dir + "meta_feature_importance.csv")
