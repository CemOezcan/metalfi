import multiprocessing as mp
import os
import pathlib
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder


class Memory:
    """
    Provides methods for saving and loading data and files from the metalfi/data directory.

    Global variables
    ----------
        lock : (Lock)
            Mutex lock.
    """
    lock = mp.Lock()

    @staticmethod
    def load(file_name: str, directory_name: str) -> pd.DataFrame:
        """
        Load a .csv file.

        Parameters
        ----------
            name : Name of the file.
            directory : (str) Directory of the file.

        Returns
        -------
            The .csv file as :py:object:`DataFrame` object.
        """
        path = Memory.get_path()
        if directory_name != "":
            directory_name += "/"
        return pd.read_csv(path / (directory_name + file_name))

    @staticmethod
    def load_open_ml() -> List[Tuple[pd.DataFrame, str, str]]:
        """
        Fetch and preprocess base-data sets from openML:
        Criteria:
            - Number of base-data sets: 60 (20 small, medium sized and large base-data sets)
            - Number of instances in [100, 2000]
            - Number of classes = 2
            - Number of missing values = 0
            - Number of zero-variance features = 0
            - Majority class ratio < 0.67

        We categorize base-data sets based on their complexity.
        We randomly choose 20 base-data sets from each of the following categories:

        Small base-data sets:
            - Number of features in [5, 10]
        Medium base-data sets:
            - Number of features in [11, 20]
        Large base-data sets:
            - Number of features in [21, 50]

        Apply ordinal encoding on categorical features and target variables.
        Binarize target variables if necessary.

        Returns
        -------
            List of tuples containing the preprocessed base-data sets,
            the name of the base-data set and the name of its target variable.
        """
        openml_list = openml.datasets.list_datasets()
        dataset_overview = pd.DataFrame.from_dict(openml_list, orient="index")
        dataset_overview = dataset_overview[dataset_overview['NumberOfClasses'] == 2]
        dataset_overview = dataset_overview[(dataset_overview["MajorityClassSize"] / dataset_overview['NumberOfInstances']) < 0.67]

        target = "base-target_variable"
        Memory.filter_data_frames(dataset_overview=dataset_overview, instances=(99, 2001), features=(4, 11),
                                  max_num_datasets=20, target_name=target)
        Memory.filter_data_frames(dataset_overview=dataset_overview, instances=(99, 2001), features=(10, 21),
                                  max_num_datasets=20, target_name=target)
        Memory.filter_data_frames(dataset_overview=dataset_overview, instances=(99, 2001), features=(20, 51),
                                  max_num_datasets=20, target_name=target)

        return [(Memory.load(file, "preprocessed"), file[:-4], target) for file in Memory.get_contents("preprocessed")]

    @staticmethod
    def filter_data_frames(dataset_overview: pd.DataFrame, instances: Tuple[int, int], features: Tuple[int, int],
                           max_num_datasets: int, target_name: str) -> None:
        dataset_overview = dataset_overview[dataset_overview['NumberOfInstances'] > instances[0]]
        dataset_overview = dataset_overview[dataset_overview['NumberOfInstances'] < instances[1]]
        dataset_overview = dataset_overview[dataset_overview['NumberOfFeatures'] > features[0]]
        dataset_overview = dataset_overview[dataset_overview['NumberOfFeatures'] < features[1]]
        dataset_overview = dataset_overview[dataset_overview['NumberOfMissingValues'] == 0].sort_values(["version"])
        dataset_overview = dataset_overview.drop_duplicates("name", "last")

        ids = [tuple(x) for x in dataset_overview[["name", "version"]].values
               if str(x[0]) + "_" + str(x[1]) + ".csv" not in Memory.get_contents("preprocessed")]  # not pre-processed

        counter = sum(str(x[0]) + "_" + str(x[1]) + ".csv" in Memory.get_contents("preprocessed")
                      for x in dataset_overview[["name", "version"]].values)  # already pre-processed

        for name, version in ids:
            if counter >= max_num_datasets:
                return
            try:
                dataset = fetch_openml(name=name, version=version, as_frame=True)
                categories = fetch_openml(name=name, version=version, as_frame=False)["categories"]
            except Exception:
                ids.remove((name, version))
                continue

            all_features = dataset["feature_names"]
            cat_features = []
            data_frame = dataset["frame"].dropna(axis=0)
            data_frame = data_frame.rename(columns={dataset["target_names"][0]: target_name})

            X = data_frame.drop(target_name, axis=1)
            X_cat = X

            y = data_frame[target_name]

            sorted_categories = []
            for feature in all_features:
                if str(data_frame[feature].dtypes) == "category":
                    cat_features.append(feature)
                    sorted_categories.append(categories[feature])
                else:
                    X_cat = X_cat.drop(feature, axis=1)

            if str(dataset["target"].dtypes) == "category":
                y_enc = LabelEncoder()
                y = y_enc.fit_transform(y)

            else:
                est = KBinsDiscretizer(n_bins=2, encode='ordinal')
                y = est.fit_transform([[x] for x in y])

            X_num = X.drop(X_cat.columns, axis=1)
            num_features = list(set(all_features) - set(cat_features))

            X_enc = OrdinalEncoder(categories=sorted_categories)
            X_cat = X_enc.fit_transform(X_cat)

            data_frame = pd.DataFrame(data=np.c_[np.c_[X_cat, X_num], y],
                                      columns=cat_features + num_features + [target_name])

            if True in [data_frame[d].var() == 0.0 for d in data_frame.columns]:
                ids.remove((name, version))
                continue

            Memory.store_preprocessed(data_frame, name + "_" + str(version))
            counter += 1

    @staticmethod
    def store_input(data: pd.DataFrame, dataset_name: str):
        """
        Store `data` as a .csv file in metalfi/data/input.

        Parameters
        ----------
            data : Contains a meta-data set.
            name : Name of the base-data set, from which the meta-data set in `data` was extracted.
        """
        path = Memory.get_path() / ("input/" + dataset_name + "meta.csv")
        if not path.is_file():
            data.to_csv(path, index=False, header=True)

    @staticmethod
    def store_model(model: 'MetaModel', file_name: str):
        """
        Serialize and save an instance of :py:class:`MetaModel` in metalfi/data/model.

        Parameters
        ----------
            model : Instance of :py:class:`MetaModel`, that is supposed to be saved as a pickle-file.
            name : Name of the pickle-file.
        """
        path = Memory.get_path() / ("model/" + file_name)
        data = (model.get_meta_models(), model.get_stats(), model.get_result_config(), model.get_results(), model.get_times())
        if not path.is_file():
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    @staticmethod
    def load_model(names: List[str]) -> List[Tuple['MetaModel', str]]:
        """
        Load pickle files in metalfi/data/model and return them as instances of :py:class:`MetaModel`.

        Parameters
        ----------
            names : Names of files that are supposed to be loaded.

        Returns
        -------
            ist of Tuples containing meta-models as instances of :py:class:`MetaModel` and their respective names.
        """
        models = []
        for name in names:
            path = Memory.get_path() / ("model/" + name)
            with open(path, 'rb') as file:
                data = pickle.load(file)
            models.append((data, name))

        return models

    @staticmethod
    def store_data_frame(data: pd.DataFrame, file_name: str, directory_name: str,
                         renew: bool = True) -> None:
        """
        Store a :py:obj:`DataFrame` object as .csv file in a given sub directory of metalfi/data/output.

        Parameters
        ----------
            data : Contains the contents of the .csv file.
            name : Name of the file.
            directory : Subdirectory of metalfi/data
            renew : Whether to renew the file, or not, should it already exist.
        """
        path = Memory.get_path() / ("output/" + directory_name + "/" + file_name + ".csv")
        if renew or not path.is_file():
            if data.index.name is None:
                data.index.name = "Index"
            data.to_csv(path, header=True)

    @staticmethod
    def update_runtimes(new_runtime_data: Dict[str, float], name: str) -> None:
        Memory.lock.acquire()
        try:
            try:
                runtimes = Memory.load("runtimes.csv", "output/runtime").set_index("Index")
            except (FileNotFoundError, KeyError):
                Memory.store_data_frame(pd.DataFrame(), "runtimes", "runtime")
                runtimes = pd.DataFrame()

            runtimes.drop([name], inplace=True, errors="ignore")
            Memory.store_data_frame(data=runtimes.append(pd.DataFrame(data=new_runtime_data, index=[name])),
                                    file_name="runtimes", directory_name="runtime")
        finally:
            Memory.lock.release()

    @staticmethod
    def store_visual(file_name: str, directory_name: str) -> None:
        """
        Save `plt` as .pdf file in a subdirectory of metalfi/data.

        Parameters
        ----------
            name : Name of the .pdf file.
            directory : Subdirectory of metalfi/data.
        """
        plt.savefig(Memory.get_path() / ("output/" + directory_name + "/" + file_name + ".pdf"))
        plt.close()

    @staticmethod
    def get_contents(directory_name: str) -> List[str]:
        """
        Fetch and return all file names in the subdirectory `directory` of metalfi/data.

        Parameters
        ----------
            directory : Subdirectory of metalfi/data.

        Returns
        -------
            File names in metalfi/data/`directory`.
        """
        path = Memory.get_path() / directory_name
        file_names = [x for x in os.listdir(path) if not x.endswith(".gitignore")]
        return file_names

    @staticmethod
    def get_path() -> pathlib.Path:
        """

        Returns:
            The metalfi/data directory.
        """
        path = pathlib.Path(__file__).parents[1] / "data"
        return path

    @staticmethod
    def store_preprocessed(data: pd.DataFrame, file_name: str) -> None:
        """
        Store a :py:obj:`DataFrame` object as .csv file in metalfi/data/preprocessed.

        Parameters
        ----------
            data : Contains the contents of the .csv file.
            name : Name of the file.
        """
        path = Memory.get_path() / ("preprocessed/" + file_name + ".csv")
        data.to_csv(path, index=False, header=True)

    @staticmethod
    def clear_directory(directories: List[str]) -> None:
        """
        Delete all files (except .gitignore files) from given directories.

        Parameters
        ----------
        directories :
            The directories, whose contents are supposed to be deleted.
        """
        for directory in directories:
            for file in Memory.get_contents(directory):
                try:
                    os.remove(Memory.get_path() / directory / file)
                except FileNotFoundError:
                    continue
