from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory
from metalfi.src.model.metamodel import MetaModel


class Controller:

    def __init__(self):
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

        self.__train_data = [data_2, data_3, data_4, data_5]
        self.__test_data = [data_1]

    def train_and_test(self):
        model = MetaModel(None, self.__train_data, "name")
        model.train()

        model.test(self.__test_data)
