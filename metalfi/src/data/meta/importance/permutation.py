from metalfi.src.data.meta.importance.featureimportance import FeatureImportance


class PermutationImportance(FeatureImportance):

    def __init__(self, dataset):
        super(PermutationImportance, self).__init__(dataset)

    def calculateScores(self):
        return