from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC


class FeatureImportance(ABC):

    def __init__(self, dataset):
        if dataset is None:
            return

        self._dataset = dataset
        self._data_frame = dataset.getDataFrame()
        self._target = self._dataset.getTarget()

        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=115)
        linSVC = LinearSVC(dual=False, max_iter=10000, random_state=115)
        svc = SVC(random_state=115)
        log = LogisticRegression(dual=False, max_iter=1000, n_jobs=-1, random_state=115)
        nb = GaussianNB()

        self._linear_models = [linSVC, log]
        self._tree_models = [rf]
        self._kernel_models = [nb, svc]

        self._vif = list()
        self._feature_importances = list()

        self.__model_names = ["linSVC", "LOG", "RF", "NB", "SVC"]
        self._name = ""

    def getModelNames(self):
        return self.__model_names

    def getFeatureImportances(self):
        return self._feature_importances

    def getName(self):
        return self._name

    @abstractmethod
    def calculateScores(self):
        pass

