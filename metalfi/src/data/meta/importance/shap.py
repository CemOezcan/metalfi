import shap
import numpy as np

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from metalfi.src.data.meta.importance.featureimportance import FeatureImportance


class ShapImportance(FeatureImportance):

    def __init__(self, dataset):
        super(ShapImportance, self).__init__(dataset)
        self._name = "_shap"

    def calculateScores(self):
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(self._target, axis=1)),
                      columns=self._data_frame.drop(self._target, axis=1).columns)
        y = self._data_frame[self._target]

        for model in self._linear_models:
            self._feature_importances.append(self.linearShap(model, X, y))

        for model in self._tree_models:
            self._feature_importances.append(self.treeShap(model, X, y))

        #for model in self._kernel_models:
         #   self._feature_importances.append(self.kernelShap(model, X, y))

    def treeShap(self, model, X, y):
        model.fit(X, y)
        imp = shap.TreeExplainer(model).shap_values(X)
        shap.summary_plot(imp[1], X, plot_type="bar")

        return self.createDataFrame(imp[1], X)

    def linearShap(self, model, X, y):
        model.fit(X, y)
        imp = shap.LinearExplainer(model, X).shap_values(X)
        shap.summary_plot(imp, X, plot_type="bar")

        return self.createDataFrame(imp, X)

    def sampleShap(self, model, X, y):
        model.fit(X, y)
        imp = shap.SamplingExplainer(model.predict, X)
        #shap.summary_plot(imp, X, plot_type="bar")

        return self.createDataFrame(imp, X)

    def kernelShap(self, model, X, y):
        model.fit(X, y)
        X_summary = shap.kmeans(X, 5)
        imp = shap.KernelExplainer(model.predict, X_summary).shap_values(X)
        shap.summary_plot(imp, X, plot_type="bar")

        return self.createDataFrame(imp, X)

    def createDataFrame(self, array, X):
        if str(type(array)).endswith("'list'>"):
            importances = list(map(lambda x: x / len(array),
                                   map(sum,
                                       zip(*[self.calculateImportances(c) for c in array]))))
        else:
            importances = self.calculateImportances(array)

        return DataFrame(data=importances, index=X.columns, columns=["Importances"])

    def calculateImportances(self, array):
        importances = list()
        for i in range(len(array[0])):
            importances.append(sum([abs(x[i]) for x in array]) / len(array))

        return importances
