import multiprocessing as mp
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder

from scr.metalfi.parameters import Parameters


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

        return [(pd.read_csv(Parameters.base_dataset_dir + file), file[:-4], target)
                for file in os.listdir(Parameters.base_dataset_dir) if file.endswith('.csv')]

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
               if str(x[0]) + "_" + str(x[1]) + ".csv" not in os.listdir(Parameters.base_dataset_dir)]  # not pre-processed

        counter = sum(str(x[0]) + "_" + str(x[1]) + ".csv" in os.listdir(Parameters.base_dataset_dir)
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

            data_frame.to_csv(Parameters.base_dataset_dir + name + "_" + str(version) + ".csv", index=False)
            counter += 1

    @staticmethod
    def store_model(model: 'MetaModel', model_name: str):
        """
        Serialize and save an instance of :py:class:`MetaModel`.

        Parameters
        ----------
            model : Instance of :py:class:`MetaModel`, that is supposed to be saved as a pickle file.
            model_name : Name of the pickle file.
        """
        path = Parameters.meta_model_dir + model_name + '.pickle'
        data = (model.get_meta_models(), model.get_stats(), model.get_result_config(), model.get_results(), model.get_times())
        if not path.is_file():
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    @staticmethod
    def load_model(model_name: str) -> Tuple['MetaModel', str]:
        """
        Load pickle file and return an instance of :py:class:`MetaModel`.

        Parameters
        ----------
            file_names : Names of file that is supposed to be loaded.

        Returns
        -------
            Tuples containing meta-model as instances of :py:class:`MetaModel` and its respective name.
        """
        path = Parameters.meta_model_dir + model_name + '.pickle'
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return (data, model_name)

    @staticmethod
    def store_data_frame(data: pd.DataFrame, file_name: str, directory_name: str,
                         renew: bool = True) -> None:
        """
        Store a :py:obj:`DataFrame` object as .csv file in a given sub directory of the output directory.

        Parameters
        ----------
            data : Contains the contents of the .csv file.
            name : Name of the file.
            directory : Subdirectory of metalfi/data
            renew : Whether to renew the file, or not, should it already exist.
        """
        path = Parameters.output_dir + directory_name + "/" + file_name + ".csv"
        if renew or not path.is_file():
            if data.index.name is None:
                data.index.name = "Index"
            data.to_csv(path, header=True)

    @staticmethod
    def update_runtimes(new_runtime_data: Dict[str, float], name: str) -> None:
        Memory.lock.acquire()
        try:
            try:
                runtimes = pd.read_csv(Parameters.output_dir + "runtime/runtimes.csv").set_index("Index")
            except (FileNotFoundError, KeyError):
                Memory.store_data_frame(pd.DataFrame(), "runtimes", "runtime")
                runtimes = pd.DataFrame()

            runtimes.drop([name], inplace=True, errors="ignore")
            Memory.store_data_frame(data=runtimes.append(pd.DataFrame(data=new_runtime_data, index=[name])),
                                    file_name="runtimes", directory_name="runtime")
        finally:
            Memory.lock.release()

    @staticmethod
    def clear_directories(directories: List[str]) -> None:
        """
        Delete all files (except .gitignore files) from given directories.

        Parameters
        ----------
        directories :
            The directories, whose contents are supposed to be deleted.
        """
        for directory in directories:
            for file in os.listdir(directory):
                if file != '.gitignore':
                    try:
                        os.remove(directory + file)
                    except FileNotFoundError:
                        continue
