from abc import ABC, abstractmethod
from metalfi.src.memory import Memory


class FeatureImportance(ABC):

    def __init__(self, dataset):
        if dataset is None:
            return

        self._dataset = dataset
        self._data_frame = dataset.getDataFrame()
        self._target = self._dataset.getTarget()

        self._linear_models = []
        self._tree_models = []
        self._kernel_models = []

        self.__model_names = []

        for model, name, type in Memory.base_models(True):
            self.__model_names.append(name)
            if type == "tree":
                self._tree_models.append(model)
            elif type == "linear":
                self._linear_models.append(model)
            elif type == "kernel":
                self._kernel_models.append(model)

        self._vif = list()
        self._feature_importances = list()
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

