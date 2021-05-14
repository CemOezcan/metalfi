
import os
import pickle
import numpy as np
import pandas as pd

from typing import List, Tuple
from pathlib import Path
from pandas import DataFrame
from sklearn.datasets import load_wine, load_iris, load_boston, fetch_openml
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder


class Memory:
    """
    Provides methods for saving and loading data and files from the metalfi/data directory.
    """

    @staticmethod
    def load(name: str, dir=None):
        """
        Load a .csv file.

        Parameters
        ----------
            name : Name of the file.
            dir : Directory of the file.

        Returns
        -------
            # TODO: after reimplementing base-data set selection.
        """
        path = Memory.get_path()
        if not (dir is None):
            return pd.read_csv(path / (dir + "/" + name))

        try:
            data = pd.read_csv(path / ("preprocessed/pp" + name)), True
        except FileNotFoundError:
            data = pd.read_csv(path / ("raw/" + name)), False

        return data

    @staticmethod
    def loadBoston():
        boston = load_boston()
        data_frame = DataFrame(data=boston.data, columns=boston['feature_names'])

        est = KBinsDiscretizer(n_bins=2, encode='ordinal')
        data_frame["target"] = est.fit_transform(list(map(lambda x: [x], boston.target)))

        return data_frame, "target"

    @staticmethod
    def loadTitanic():
        data_frame, preprocessed = Memory.load("titanic.csv")

        if not preprocessed:
            data_frame = data_frame.drop("Ticket", axis=1)

            data_frame["Sex"].replace({"male": 1, "female": 0}, inplace=True)

            data_frame["Name"] = [x.split(", ")[1] for x in data_frame["Name"]]
            data_frame["Name"] = [x.split(' ')[0] for x in data_frame["Name"]]
            data_frame["Name"] = data_frame["Name"].apply(
                lambda x: 0 if (x == "Mrs." or x == "Ms.") else (1 if x == "Mr." else 2))
            data_frame = data_frame.rename(columns={"Name": "Title"})

            data_frame["Cabin"] = [0 if (str(x) == "nan") else x[0] for x in data_frame["Cabin"]]
            data_frame["Cabin"].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}, inplace=True)

            data_frame["Embarked"] = [0 if (str(x) == "nan") else x for x in data_frame["Embarked"]]
            data_frame["Embarked"].replace({'C': 1, 'Q': 2, 'S': 3}, inplace=True)

            avg_age = int(data_frame["Age"].mean())
            data_frame["Age"] = [avg_age if (str(x) == "nan") else x for x in data_frame["Age"]]

            data = {"Survived": data_frame["Survived"].values}
            data_frame_2 = DataFrame(data, columns=["Survived"])
            data_frame = data_frame.drop("Survived", axis=1)
            data_frame = data_frame.assign(Survived=data_frame_2["Survived"])
            data_frame = data_frame.drop("PassengerId", axis=1)

            data_frame.to_csv(Memory.get_path() / "preprocessed/pptitanic.csv", index=None, header=True)

        return data_frame, "Survived"

    @staticmethod
    def loadCancer():
        data_frame, preprocessed = Memory.load("cancer.csv")

        return data_frame.drop("Unnamed: 0", axis=1), "MEDV"

    @staticmethod
    def loadWine():
        wine = load_wine()
        data_frame = DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])

        return data_frame, "target"

    @staticmethod
    def loadIris():
        iris = load_iris()
        data_frame = DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

        return data_frame, "target"

    @staticmethod
    def load_open_ml() -> List[Tuple[DataFrame, str, str]]:
        """
        Fetch and preprocess base-data sets from openML:
        Apply ordinal encoding on categorical features and target variables.
        Binarize target variables if necessary.

        Returns
        -------
            List of tuples containing the preprocessed base-data sets,
            the name of the base-data set and the name of its target variable.
        """
        datasets = list()
        ids = [("tic-tac-toe", 1), ("banknote-authentication", 1), ("haberman", 1), ("servo", 1), ("cloud", 2),
               ("primary-tumor", 2), ("EgyptianSkulls", 1), ("SPECTF", 2), ("cpu", 2), ("bodyfat", 2), ("Engine1", 1),
               ("ESL", 2), ("ilpd-numeric", 2), ("credit-approval", 1), ("vowel", 3), ("socmob", 2), ("ERA", 1),
               ("LEV", 1), ("credit-g", 1), ("cmc", 2), ("phoneme", 1), ("bank8FM", 2), ("wind", 2)]

        for name, version in ids:
            dataset = fetch_openml(name=name, version=version, as_frame=True)
            categories = fetch_openml(name=name, version=version, as_frame=False)["categories"]
            all_features = dataset["feature_names"]
            cat_features = list()
            data_frame = dataset["frame"].dropna(axis=0)
            target = dataset["target_names"][0]

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

            X_enc = OrdinalEncoder(sorted_categories)
            X_cat = X_enc.fit_transform(X_cat)

            data_frame = DataFrame(data=np.c_[np.c_[X_cat, X_num], y],
                                   columns=cat_features + num_features + [target])

            datasets.append((data_frame, name, target))

        return datasets

    @staticmethod
    def store_meta_features(data):
        path = Memory.get_path() / "features/selected"
        if not path.is_file():
            pickle.dump(data, open(path, 'wb'))

    @staticmethod
    def load_meta_features():
        try:
            path = Memory.get_path() / "features/selected"
            file = open(path, 'rb')
            data = pickle.load(file)
            file.close()

        except FileNotFoundError:
            data = None

        return data

    @staticmethod
    def store_input(data: DataFrame, name: str):
        """
        Store `data` as a .csv file in metalfi/data/input.

        Parameters
        ----------
            data : Contains a meta-data set.
            name : Name of the base-data set, from which the meta-data set in `data` was extracted.
        """
        path = Memory.get_path() / ("input/" + name + "meta.csv")
        if not path.is_file():
            data.to_csv(path, index=None, header=True)

    def storeOutput(self, data):
        return

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
            file = open(path, 'wb')
            pickle.dump(model, file)
            file.close()

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
            file = open(path, 'rb')
            data = pickle.load(file)
            file.close()
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
        file = open(path, 'wb')
        pickle.dump(model, file)
        file.close()

    @staticmethod
    def store_data_frame(data: DataFrame, name: str, directory: str, renew=False):
        """
        Store a :py:obj:`DataFrame` object as .csv file in a given sub directory of metalfi/data.

        Parameters
        ----------
            data : Contains the contents of the .csv file.
            name : Name of the file.
            directory : Subdirectory of metalfi/data
            renew : Whether to renew the file, or not, should it already exist.
        """
        path = Memory.get_path() / ("output/" + directory + "/" + name + ".csv")
        if renew or not path.is_file():
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
