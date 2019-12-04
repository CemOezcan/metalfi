from abc import ABC, abstractmethod

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class FeatureImportance(ABC):

    #TODO: Score metric & more models
    def __init__(self, dataset):
        self._dataset = dataset
        self._data_frame = dataset.getDataFrame()
        self._target = self._dataset.getTarget()

        reg = linear_model.LinearRegression()
        rf = RandomForestClassifier(random_state=101)
        svc = SVC(gamma='auto')

        self._linear_models = [reg]
        self._tree_models = [rf]
        self._kernel_models = [svc]

        self._vif = list()
        self._featureImportances = list()

    def getFeatureImportances(self):
        return self._featureImportances

    @abstractmethod
    def calculateScores(self):
        #TODO: Parallelize
        pass

