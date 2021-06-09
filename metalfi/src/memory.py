import os
from pathlib import Path
import pickle
from typing import List, Tuple

import numpy as np
import openml
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder


class Memory:
    """
    Provides methods for saving and loading data and files from the metalfi/data directory.
    """

    @staticmethod
    def load(name: str, directory) -> pd.DataFrame:
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
        return pd.read_csv(path / (directory + "/" + name))

    @staticmethod
    def load_open_ml() -> List[Tuple[pd.DataFrame, str, str]]:
        """
        Fetch and preprocess base-data sets from openML:
        Criteria:
            - Number of instances <= 500
            - Number of features in [5, 10]
            - Number of classes = 2
            - Number of missing values = 0
            - Number of zero-variance features = 0
            - Majority class ratio < 0.67
        Apply ordinal encoding on categorical features and target variables.
        Binarize target variables if necessary.

        Returns
        -------
            List of tuples containing the preprocessed base-data sets,
            the name of the base-data set and the name of its target variable.
        """
        openml_list = openml.datasets.list_datasets()
        data = pd.DataFrame.from_dict(openml_list, orient="index")
        data = data[data['NumberOfInstances'] < 501]
        data = data[data['NumberOfFeatures'] < 11]
        data = data[data['NumberOfFeatures'] > 4]
        data = data[data['NumberOfClasses'] == 2]
        data = data[(data["MajorityClassSize"] / data['NumberOfInstances']) < 0.67]
        data = data[data['NumberOfMissingValues'] == 0].sort_values(["version"])
        data = data.drop_duplicates("name", "last")

        target = "base-target_variable"
        ids = list(filter(lambda x: str(x[0]) + "_" + str(x[1]) + ".csv" not in Memory.get_contents("preprocessed"),
                          [tuple(x) for x in data[["name", "version"]].values]))

        for name, version in ids:
            try:
                dataset = fetch_openml(name=name, version=version, as_frame=True)
                categories = fetch_openml(name=name, version=version, as_frame=False)["categories"]
            except Exception:
                ids.remove((name, version))
                continue

            all_features = dataset["feature_names"]
            cat_features = list()
            data_frame = dataset["frame"].dropna(axis=0)
            data_frame = data_frame.rename(columns={dataset["target_names"][0]: target})

            X = data_frame.drop(target, axis=1)
            X_cat = X

            y = data_frame[target]

            sorted_categories = list()
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
                y = est.fit_transform(list(map(lambda x: [x], y)))

            X_num = X.drop(X_cat.columns, axis=1)
            num_features = list(set(all_features) - set(cat_features))

            X_enc = OrdinalEncoder(categories=sorted_categories)
            X_cat = X_enc.fit_transform(X_cat)

            data_frame = pd.DataFrame(data=np.c_[np.c_[X_cat, X_num], y],
                                      columns=cat_features + num_features + [target])

            if True in [data_frame[d].var() == 0.0 for d in data_frame.columns]:
                ids.remove((name, version))
                continue

            Memory.store_preprocessed(data_frame, name + "_" + str(version))

        return [(Memory.load(file, "preprocessed"), file[:-4], target) for file in Memory.get_contents("preprocessed")]

    @staticmethod
    def store_meta_features(data):
        path = Memory.get_path() / "features/selected"
        if not path.is_file():
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    @staticmethod
    def load_meta_features():
        try:
            path = Memory.get_path() / "features/selected"
            with open(path, 'rb') as file:
                data = pickle.load(file)

        except FileNotFoundError:
            data = None

        return data

    @staticmethod
    def store_input(data: pd.DataFrame, name: str):
        """
        Store `data` as a .csv file in metalfi/data/input.

        Parameters
        ----------
            data : Contains a meta-data set.
            name : Name of the base-data set, from which the meta-data set in `data` was extracted.
        """
        path = Memory.get_path() / ("input/" + name + "meta.csv")
        if not path.is_file():
            data.to_csv(path, index=False, header=True)

    @staticmethod
    def store_model(model: 'MetaModel', name: str):
        """
        Serialize and save an instance of :py:class:`MetaModel` in metalfi/data/model.

        Parameters
        ----------
            model : Instance of :py:class:`MetaModel`, that is supposed to be saved as a pickle-file.
            name : Name of the pickle-file.
        """
        path = Memory.get_path() / ("model/" + name)
        if not path.is_file():
            with open(path, 'wb') as file:
                pickle.dump(model, file)

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
        models = list()
        for name in names:
            path = Memory.get_path() / ("model/" + name)
            with open(path, 'rb') as file:
                data = pickle.load(file)
            models.append((data, name))

        return models

    @staticmethod
    def renew_model(model: 'MetaModel', name: str):
        """
        Replace a meta-model in metalfi/data/model.

        Parameters
        ----------
            model : New instance of :py:class:`MetaModel`, that is supposed to replace the old instance.
            name : Identifies the file that is supposed to be replaced.
        """
        path = Memory.get_path() / ("model/" + name)
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def store_data_frame(data: pd.DataFrame, name: str, directory: str, renew=True):
        """
        Store a :py:obj:`DataFrame` object as .csv file in a given sub directory of metalfi/data/output.

        Parameters
        ----------
            data : Contains the contents of the .csv file.
            name : Name of the file.
            directory : Subdirectory of metalfi/data
            renew : Whether to renew the file, or not, should it already exist.
        """
        path = Memory.get_path() / ("output/" + directory + "/" + name + ".csv")
        if renew or not path.is_file():
            if data.index.name is None:
                data.index.name = "Index"
            data.to_csv(path, header=True)

    @staticmethod
    def store_visual(plt, name, directory):
        """
        Save `plt` as .png file in a subdirectory of metalfi/data.

        Parameters
        ----------
            plt : Matplotlib-plot.
            name : Name of the .png file.
            directory : Subdirectory of metalfi/data.
        """
        plt.savefig(Memory.get_path() / ("output/" + directory + "/" + name + ".png"))
        plt.close()

    @staticmethod
    def get_contents(directory) -> List[str]:
        """
        Fetch and return all file names in the subdirectory `directory` of metalfi/data.

        Parameters
        ----------
            directory : Subdirectory of metalfi/data.

        Returns
        -------
            File names in metalfi/data/`directory`.
        """
        path = Memory.get_path() / directory
        file_names = list(filter(lambda x: not x.endswith(".gitignore"), os.listdir(path)))
        return file_names

    @staticmethod
    def get_path():
        """

        Returns:
            The metalfi/data directory.
        """
        path = Path(__file__).parents[1] / "data"
        return path

    @staticmethod
    def store_preprocessed(data: pd.DataFrame, name: str):
        """
        Store a :py:obj:`DataFrame` object as .csv file in metalfi/data/preprocessed.

        Parameters
        ----------
            data : Contains the contents of the .csv file.
            name : Name of the file.
        """
        path = Memory.get_path() / ("preprocessed/" + name + ".csv")
        data.to_csv(path, index=False, header=True)

    @staticmethod
    def clear_directory(directories: List[str]):
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
