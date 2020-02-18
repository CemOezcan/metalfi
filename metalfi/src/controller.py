import pandas as pd

from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory
from metalfi.src.data.metadataset import MetaDataset
from metalfi.src.model.evaluation import Evaluation
from metalfi.src.model.metamodel import MetaModel


class Controller:

    def __init__(self):
        self.__train_data = None
        self.__enum = None
        self.__meta_data = list()
        self.downloadData()
        self.storeMetaData()

    def getTrainData(self):
        return self.__train_data

    def downloadData(self):
        data_frame, target = Memory.loadTitanic()
        data_1 = Dataset(data_frame, target)

        data_frame_2, target_2 = Memory.loadCancer()
        data_2 = Dataset(data_frame_2, target_2)

        data_frame_3, target_3 = Memory.loadIris()
        data_3 = Dataset(data_frame_3, target_3)

        data_frame_4, target_4 = Memory.loadWine()
        data_4 = Dataset(data_frame_4, target_4)

        data_frame_5, target_5 = Memory.loadBoston()
        data_5 = Dataset(data_frame_5, target_5)

        self.__train_data = [(data_1, "Titanic"), (data_2, "Cancer"), (data_3, "Iris"), (data_4, "Wine"),
                             (data_5, "Boston")]
        self.__enum = {"Titanic": 0, "Cancer": 1, "Iris": 2, "Wine": 3, "Boston": 4}

    def storeMetaData(self):
        for dataset, name in self.__train_data:
            if not (Memory.getPath() / ("input/" + name + "meta.csv")).is_file():
                data = MetaDataset([dataset], True).getMetaData()
                Memory.storeInput(data, name)

    def loadMetaData(self):
        for dataset, name in self.__train_data:
            self.__meta_data.append((Memory.load(name + "meta.csv", "input"), name))

    def trainMetaModel(self):
        # TODO: Combine Meta-Datasets + CV + Different Meta-Feature splits <-- In MetaModel
        self.loadMetaData()
        for i in range(0, len(self.__meta_data)):
            test_data, test_name = self.__meta_data[i]
            train_data = list()
            for j in range(0, len(self.__meta_data)):
                if not (i == j):
                    train_data.append(self.__meta_data[j][0])

            path = Memory.getPath() / ("model/" + test_name)
            if not path.is_file():
                og_data, name = self.__train_data[self.__enum[test_name]]
                model = MetaModel(pd.concat(train_data), test_name + "meta", test_data, og_data)
                model.fit()
                Memory.storeModel(model, test_name, None)

    def evaluate(self, names):
        evaluation = Evaluation(Memory.loadModel(names))
        evaluation.run()

    def loadModel(self, names):
        return Memory.loadModel(names)
