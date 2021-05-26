from pandas import DataFrame
import rfpimp
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class DropColumnImportance(FeatureImportance):
    """
    LOFO-importance.
    """

    def __init__(self, dataset: 'Dataset'):
        super().__init__(dataset=dataset)
        self._name = "_LOFO"

    def calculate_scores(self) -> None:
        for model_type in self._models.keys():
            for name in self._models[model_type].keys():
                model = self._models[model_type][name]
                self._feature_importances.append(self.__dropcol_importance(model, self._target))

    def __dropcol_importance(self, model: BaseEstimator, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        return rfpimp.dropcol_importances(model, X, y)

    def __oob_dropcol_importance(self, model: BaseEstimator, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        return rfpimp.oob_dropcol_importances(model, X, y)
