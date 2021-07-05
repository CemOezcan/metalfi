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
        with self.ignore_np_deprecation_warning():
            for model in self._all_models:
                self._feature_importances.append(self.__permutation_importance(model, self._target))

    def __permutation_importance(self, model: BaseEstimator, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        np.random.seed(115)
        return rfpimp.importances(model, X, y)

    @staticmethod
    def data_permutation_importance(model: BaseEstimator, X, y) -> DataFrame:
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        model.fit(X_sc, y)
        np.random.seed(115)
        return rfpimp.importances(model, X_sc, y)
