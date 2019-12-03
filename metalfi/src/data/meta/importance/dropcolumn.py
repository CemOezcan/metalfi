from metalfi.src.data.meta.importance.featureimportance import FeatureImportance
from rfpimp import *
from sklearn import linear_model
from sklearn.svm import SVC


class DropColumnImportance(FeatureImportance):

    def __init__(self, dataset):
        super(DropColumnImportance, self).__init__(dataset)

    def calculateScores(self):
        reg = linear_model.LinearRegression()
        rf = RandomForestClassifier(random_state=101)
        svc = SVC(gamma='auto')

        target = self._dataset.getTarget()

        self.dropcolImportance(reg, target)

        self.oobDropcolImportance(rf, target)

        self.dropcolImportance(svc, target)

    def dropcolImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = dropcol_importances(model, X, y)
        plot_importances(imp).view()

    def oobDropcolImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = oob_dropcol_importances(model, X, y)
        plot_importances(imp).view()
