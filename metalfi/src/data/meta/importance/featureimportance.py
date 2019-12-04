from abc import ABC, abstractmethod


class FeatureImportance(ABC):

    #TODO: Score metric
    def __init__(self, dataset):
        self._dataset = dataset
        self._data_frame = dataset.getDataFrame()
        self._featureImportances = None
        self._vif = list()

    def getFeatureImportances(self):
        return self._featureImportances

    @abstractmethod
    def calculateScores(self):
        pass
