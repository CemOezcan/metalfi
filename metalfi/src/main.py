from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory


class Main(object):

    @staticmethod
    def main():
        data_frame, target = Memory.loadTitanic()
        data = Dataset(data_frame, target)

        data_frame_2, target_2 = Memory.loadCancer()
        data_2 = Dataset(data_frame_2, target_2)

        data_frame_3, target_3 = Memory.loadWine()
        data_3 = Dataset(data_frame_3, target_3)

        data.calculateMetaFeatureVectors()
        data_2.calculateMetaFeatureVectors()
        data_3.calculateMetaFeatureVectors()


if __name__ == '__main__':
    Main().main()
