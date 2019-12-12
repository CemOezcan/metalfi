import sys

import shap
from pandas import DataFrame

from metalfi.src.data.meta.importance.featureimportance import FeatureImportance


class ShapImportance(FeatureImportance):

    def __init__(self, dataset):
        super(ShapImportance, self).__init__(dataset)
        self._name = "_shap"

    def calculateScores(self):
        X = self._data_frame.drop(self._target, axis=1)
        y = self._data_frame[self._target]

        for model in self._linear_models:
            self._feature_importances.append(self.linearShap(model, X, y))

        for model in self._tree_models:
            self._feature_importances.append(self.treeShap(model, X, y))

        for model in self._kernel_models:
            return
            #self.kernelShap(model, X, y)

    def treeShap(self, model, X, y):
        model.fit(X, y)
        imp = shap.TreeExplainer(model).shap_values(X)
        shap.summary_plot(imp[1], X, plot_type="bar")

        return self.createDataframe(imp[1], X)

    def linearShap(self, model, X, y):
        model.fit(X, y)
        imp = shap.LinearExplainer(model, X).shap_values(X)
        shap.summary_plot(imp, X, plot_type="bar")

        return self.createDataframe(imp, X)

    def sampleShap(self, model, X, y):
        model.fit(X, y)
        imp = shap.SamplingExplainer(model.predict, X)
        shap.summary_plot(imp, X, plot_type="bar")

        return self.createDataframe(imp, X)

    def kernelShap(self, model, X, y):
        model.fit(X, y)
        X_summary = shap.kmeans(X, 5)
        imp = shap.KernelExplainer(model.predict, X_summary).shap_values(X)
        shap.summary_plot(imp, X, plot_type="bar")

        return self.createDataframe(imp, X)

    def createDataframe(self, array, X):
        importances = list()
        average = len(array)
        rows = X.columns

        for i in range(0, len(array[0])):
            feature = list()
            for x in array:
                feature.append(abs(x[i]))

            importances.append(sum(feature) / average)

        columns = ["Importances"]
        data = DataFrame(data=importances, index=rows, columns=columns)

        return data
