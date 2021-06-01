from typing import List

import numpy as np
from pandas import DataFrame
import shap
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class ShapImportance(FeatureImportance):
    """
    SHAP-importance.
    """

    def __init__(self, dataset: 'Dataset'):
        super().__init__(dataset=dataset)
        self._name = "_SHAP"

    def calculate_scores(self) -> None:
        with self.ignore_progress_bars():
            sc = StandardScaler()
            X = DataFrame(data=sc.fit_transform(self._data_frame.drop(self._target, axis=1)),
                          columns=self._data_frame.drop(self._target, axis=1).columns)
            y = self._data_frame[self._target]

            for model_type in self._models.keys():
                for model in self._models[model_type].values():
                    if model_type == "tree":
                        self._feature_importances.append(self.tree_shap(model, X, y))
                    elif model_type == "linear":
                        self._feature_importances.append(self.linear_shap(model, X, y))
                    else:
                        self._feature_importances.append(self.kernel_shap(model, X, y))

    def tree_shap(self, model: BaseEstimator, X: DataFrame, y: DataFrame) -> DataFrame:
        model.fit(X, y)
        imp = shap.TreeExplainer(model).shap_values(X)

        return self.__create_data_frame(imp[1], X)

    def linear_shap(self, model: BaseEstimator, X: DataFrame, y: DataFrame) -> DataFrame:
        model.fit(X, y)
        imp = shap.LinearExplainer(model, X).shap_values(X)

        return self.__create_data_frame(imp, X)

    def tree_regression_shap(self, model: BaseEstimator, X: DataFrame, y: DataFrame) -> DataFrame:
        model.fit(X, y)
        imp = shap.TreeExplainer(model).shap_values(X)

        return self.__create_data_frame(imp, X)

    def kernel_shap(self, model: BaseEstimator, X: DataFrame, y: DataFrame, k=10) -> DataFrame:
        model.fit(X, y)
        np.random.seed(115)
        X_summary = shap.kmeans(X, k)
        imp = shap.KernelExplainer(model.predict, X_summary).shap_values(X)

        return self.__create_data_frame(imp, X)

    def __create_data_frame(self, array: List[List[float]], X: DataFrame) -> DataFrame:
        if str(type(array)).endswith("'list'>"):
            importances = list(map(lambda x: x / len(array),
                                   map(sum,
                                       zip(*[self.__calculate_importances(c) for c in array]))))
        else:
            importances = self.__calculate_importances(array)

        return DataFrame(data=importances, index=X.columns, columns=["Importances"])

    @staticmethod
    def __calculate_importances(array: List[List[float]]) -> List[float]:
        importances = list()
        for i in range(len(array[0])):
            importances.append(sum([abs(x[i]) for x in array]) / len(array))

        return importances
