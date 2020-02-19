import lime
import lime.lime_tabular

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from metalfi.src.data.meta.importance.featureimportance import FeatureImportance


class LimeImportance(FeatureImportance):

    def __init__(self, dataset):
        super(LimeImportance, self).__init__(dataset)
        self._name = "_lime"
        linSVC = LinearSVC(max_iter=10000, dual=False)
        svc = SVC(kernel="rbf", gamma="scale")

    def calculateScores(self):
        # TODO: Calc. importances for feature subsets that are multicollinear
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

    def limeImportance(self, model, target):
        # TODO: Sampling for efficiency?
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
        importances = [0] * len(X.columns)

        for i in range(len(X.values)):
            xp = explainer.explain_instance(X.values[i, :], model.predict_proba, num_features=len(X.columns))

            for index, importance in xp.as_map()[1]:
                importances[index] += abs(importance)

        importances = list(map(lambda x: x / len(X.values), importances))
        data_frame = DataFrame(data=importances, index=X.columns, columns=["Importances"])

        return data_frame
