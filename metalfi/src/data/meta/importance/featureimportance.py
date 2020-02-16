from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC


class FeatureImportance(ABC):

    #TODO: Score metric & more models
    def __init__(self, dataset):
        self._dataset = dataset
        self._data_frame = dataset.getDataFrame()
        self._target = self._dataset.getTarget()

        #reg = linear_model.LinearRegression()
        rf = RandomForestClassifier(n_estimators=100)
        linSVC = LinearSVC(max_iter=10000, dual=False)
        svc = SVC(kernel="rbf", gamma="scale")
        log = LogisticRegression(dual=False, solver="lbfgs", multi_class="auto", max_iter=1000)

        self._linear_models = [linSVC, log]
        self._tree_models = [rf]
        self._kernel_models = [svc]

        self._vif = list()
        self._feature_importances = list()

        self.__model_names = ["linSVC", "log", "rf", "svc"]
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

