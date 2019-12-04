import shap
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from metalfi.src.data.meta.importance.featureimportance import FeatureImportance


class ShapImportance(FeatureImportance):

    def __init__(self, dataset):
        super(ShapImportance, self).__init__(dataset)

    def calculateScores(self):
        reg = linear_model.LinearRegression()
        rf = RandomForestClassifier(random_state=101)
        svc = SVC(gamma='auto')

        target = self._dataset.getTarget()

        self.linearShap(reg, target)

        self.treeShap(rf, target)

        self.kernelShap(svc, target)


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