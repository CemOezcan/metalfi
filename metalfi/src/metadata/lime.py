
import math
import lime
import lime.lime_tabular
import numpy as np

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class LimeImportance(FeatureImportance):
    """
    LIME-importance.
    """
    def __init__(self, dataset: 'Dataset'):
        super(LimeImportance, self).__init__(dataset)
        self._name = "_LIME"

    def calculate_scores(self):
        for model in self._all_models:
            self._feature_importances.append(self.__lime_importance(model, self._target))

    def __lime_importance(self, model, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X.values,
                                                           feature_names=X.columns.values,
                                                           mode="classification",
                                                           random_state=115)

        num_features = len(X.columns)
        num_entries = len(X.values)
        num_samples = int(max([0.1 * num_entries, min([num_features * 2 * math.log2(num_entries), 0.5 * num_entries])]))
        np.random.seed(115)
        samples = list(np.random.permutation(num_entries))[:num_samples]

        importances = [0] * num_features
        for i in samples:
            xp = explainer.explain_instance(X.values[i, :], model.predict_proba, num_features=num_features)
            for index, importance in xp.as_map()[1]:
                importances[index] += abs(importance)

        importances = list(map(lambda x: x / num_samples, importances))
        data_frame = DataFrame(data=importances, index=X.columns, columns=["Importances"])

        return data_frame
