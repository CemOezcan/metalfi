import os
import sys
import warnings

import numpy as np
from pandas import DataFrame
import rfpimp
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class PermutationImportance(FeatureImportance):
    """
    PIMP-importance.
    """

    def __init__(self, dataset: 'Dataset'):
        super().__init__(dataset=dataset)
        self._name = "_PIMP"

    def calculate_scores(self) -> None:
        warnings.simplefilter("ignore")
        with open(os.devnull, 'w') as file:
            sys.stderr = file

        for model in self._all_models:
            self._feature_importances.append(self.__permutation_importance(model, self._target))

        sys.stderr = sys.__stderr__
        warnings.simplefilter("default")

    def __permutation_importance(self, model: BaseEstimator, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        np.random.seed(115)
        return rfpimp.importances(model, X, y)
