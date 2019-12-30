from abc import ABC, abstractmethod

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
        log = LogisticRegression()

        self._linear_models = [reg]
        self._tree_models = [rf]
        self._kernel_models = [svc, log]

        self._vif = list()
        self._feature_importances = list()

        self.__model_names = ["reg", "rf", "svc", "log"]
        self._name = ""

    def getModelNames(self):
        return self.__model_names

    def getFeatureImportances(self):
        return self._feature_importances

    def getName(self):
        return self._name

    @abstractmethod
    def calculateScores(self):
        #TODO: Parallelize
        pass

