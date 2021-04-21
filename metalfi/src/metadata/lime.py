import os
import sys

import lime
import lime.lime_tabular

from functools import partial
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from multiprocessing.pool import ThreadPool

from metalfi.src.metadata.featureimportance import FeatureImportance


class LimeImportance(FeatureImportance):

    def __init__(self, dataset):
        super(LimeImportance, self).__init__(dataset)
        self._name = "_LIME"

    def calculateScores(self):
        sys.stdout = open(os.devnull, 'w')
        sys.stdout.close()
        models = self._linear_models + self._tree_models + self._kernel_models

        for model in models:
            if str(type(model).__name__) == "LinearSVC":
                self._feature_importances.append(self.limeImportance(SVC(kernel="linear",
                                                                         decision_function_shape="ovr",
                                                                         probability=True),
                                                                     self._target))
            elif str(type(model).__name__) == "SVC":
                self._feature_importances.append(self.limeImportance(SVC(kernel="rbf",
                                                                         gamma="scale",
                                                                         probability=True),
                                                                     self._target))
            else:
                self._feature_importances.append(self.limeImportance(model, self._target))

        sys.stdout = sys.__stdout__

    def limeImportance(self, model, target):
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X.values,
                                                           feature_names=X.columns.values,
                                                           discretize_continuous=True,
                                                           mode="classification",
                                                           verbose=True)
        # TODO: Processes (Look up, how SHAP parallelizes LIME)
        importances = [0] * len(X.columns)
        with ThreadPool(processes=4) as pool:
            results = pool.map(
                partial(explainer.explain_instance, predict_fn=model.predict_proba, num_features=len(X.columns)),
                [X.values[i, :] for i in range(len(X.values))])

        for xp in results:
            for index, importance in xp.as_map()[1]:
                importances[index] += abs(importance)

        importances = list(map(lambda x: x / len(X.values), importances))
        data_frame = DataFrame(data=importances, index=X.columns, columns=["Importances"])

        return data_frame
