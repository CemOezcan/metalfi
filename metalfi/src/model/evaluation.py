import sys

from pandas import DataFrame

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
            model.test(4)
            stats = model.getStats()
            Memory.renewModel(model, model.getName()[:-4])
            self.__tests = self.vectorAddition(self.__tests, stats)

        self.__tests = [list(map(lambda x: x / len(self.__meta_models), stat)) for stat in self.__tests]
        self.__config = [c for (a, b, c) in self.__meta_models[0][0].getMetaModels()]

        targets = self.__meta_models[0][0].getTargets()
        algorithms = [x[:-5] for x in targets]
        metrics = {0: "r2", 1: "rmse", 2: "r"}
        rows = list()

        all_results = {}
        for i in metrics:
            shap = {a: [] for a in algorithms}
            lime = {a: [] for a in algorithms}
            perm = {a: [] for a in algorithms}
            dCol = {a: [] for a in algorithms}
            metric = {"shap": shap, "lime": lime, "perm": perm, "dCol": dCol}

            index = 0
            for a, b, c in self.__config:
                row = a + " x " + c
                if row not in rows:
                    rows.append(row)

                metric[b[-4:]][b[:-5]].append(self.__tests[index][i])
                index -= -1

            all_results[metrics[i]] = metric

        for metric in all_results:
            for importance in all_results[metric]:
                Memory.storeDataFrame(DataFrame(data=all_results[metric][importance], index=rows,
                                                columns=[x for x in all_results[metric][importance]]),
                                      metric + " x " + importance, "predictions")

    def comparisons(self, models, targets, subsets, renew=False):
        for (model, name) in self.__meta_models:
            model.compare(models, targets, subsets, 4, renew)
            results = model.getResults()
            Memory.renewModel(model, model.getName()[:-4])
            self.__comparisons = self.vectorAddition(self.__comparisons, results)

        self.__comparisons = [list(map(lambda x: x / len(self.__meta_models), result)) for result in self.__comparisons]
        self.__parameters = self.__meta_models[0][0].getResultConfig()

        for i in range(len(self.__comparisons)):
            print(self.__parameters[i])
            print(self.__comparisons[i])
