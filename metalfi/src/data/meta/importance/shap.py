import shap

from metalfi.src.data.meta.importance.featureimportance import FeatureImportance


class ShapImportance(FeatureImportance):

    def __init__(self, dataset):
        super(ShapImportance, self).__init__(dataset)

    def calculateScores(self):
        for model in self._linear_models:
            self.linearShap(model, self._target)

        for model in self._tree_models:
            self.treeShap(model, self._target)

        for model in self._kernel_models:
            self.kernelShap(model, self._target)

    def treeShap(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = shap.TreeExplainer(model).shap_values(X)
        shap.summary_plot(imp[1], X, plot_type="violin")

    def linearShap(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = shap.LinearExplainer(model, X).shap_values(X)
        shap.summary_plot(imp, X, plot_type="violin")

    def sampleShap(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = shap.SamplingExplainer(model.predict, X)
        shap.summary_plot(imp, X)

    def kernelShap(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        X_summary = shap.kmeans(X, 10)
        imp = shap.KernelExplainer(model.predict, X_summary).shap_values(X)
        shap.summary_plot(imp, X, plot_type="violin")