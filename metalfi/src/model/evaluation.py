from metalfi.src.data.memory import Memory


class Evaluation:

    def __init__(self, meta_models):
        self.__meta_models = meta_models
        self.__tests = list()
        self.__config = list()

        self.__comparisons = list()
        self.__parameters = list()

    @staticmethod
    def vectorAddition(x, y):
        if len(x) == 0:
            return y

        result = [list(map(sum, zip(x[i], y[i]))) for i in range(len(x))]

        return result

    def predictions(self):
        for (model, name) in self.__meta_models:
            # TODO: renew MetaModel object so that calculations do not have to be recalculated
            model.test(4)
            stats = model.getStats()
            Memory.renewModel(model, model.getName()[:-4])
            self.__tests = self.vectorAddition(self.__tests, stats)

        self.__tests = [list(map(lambda x: x / len(self.__meta_models), stat)) for stat in self.__tests]
        self.__config = [c for (a, b, c) in self.__meta_models[0][0].getMetaModels()]

        for i in range(len(self.__tests)):
            print(self.__config[i])
            print(self.__tests[i])

    def comparisons(self, models, targets, subsets):
        for (model, name) in self.__meta_models:
            model.compare(models, targets, subsets, 4)
            results = model.getResults()
            Memory.renewModel(model, model.getName()[:-4])
            self.__comparisons = self.vectorAddition(self.__comparisons, results)

        self.__comparisons = [list(map(lambda x: x / len(self.__meta_models), stat)) for stat in self.__comparisons]
        self.__parameters = self.__meta_models[0][0].getResultConfig()

        for i in range(len(self.__tests)):
            print(self.__parameters[i])
            print(self.__comparisons[i])
