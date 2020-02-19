

class Evaluation:

    def __init__(self, meta_models):
        self.__meta_models = meta_models
        self.__tests = list()

        for (model, name) in self.__meta_models:
            model.test(4)
            stats = model.getStats()
            self.__tests = self.vectorAddition(self.__tests, stats)

        self.__tests = [list(map(lambda x: x / len(self.__meta_models), stat)) for stat in self.__tests]
        self.__config = [c for (a, b, c, d) in self.__meta_models[0][0].getMetaModels()]

    @staticmethod
    def vectorAddition(x, y):
        if len(x) == 0:
            return y

        result = [list(map(sum, zip(x[i], y[i]))) for i in range(len(x))]

        return result

    def run(self):
        for i in range(len(self.__tests)):
            print(self.__config[i])
            print(self.__tests[i])
