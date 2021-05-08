from abc import ABC, abstractmethod
from metalfi.src.memory import Memory
from metalfi.src.parameters import Parameters


class FeatureImportance(ABC):

    def __init__(self, dataset):
        if dataset is None:
            return

        self._dataset = dataset
        self._data_frame = dataset.getDataFrame()
        self._target = self._dataset.getTarget()
        self._models = dict()

        for model, name, type in Parameters.base_models:
            try:
                self._models[type][name] = model
            except KeyError:
                self._models[type] = dict()
                self._models[type][name] = model

        self.__model_names = sum([list(d.keys()) for d in self._models.values()], [])
        self._all_models = sum([list(d.values()) for d in self._models.values()], [])

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

