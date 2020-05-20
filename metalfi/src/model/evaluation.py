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

    def q_4(self):
        rows = list()
        model, _ = Memory.loadModel([self.__meta_models[0]])[0]
        config = [c for (a, b, c) in model.getMetaModels()]
        meta_model_names = list(set([c[0] for c in config]))
        data = {"R^2": {key: list() for key in meta_model_names},
                "RMSE": {key: list() for key in meta_model_names},
                "r": {key: list() for key in meta_model_names}}

        for data_set in self.__meta_models:
            rows.append(data_set)
            model, _ = Memory.loadModel([data_set])[0]
            tuples = [t for t in list(zip(config, model.getStats()))
                      if (t[0][1][-4:] != "LOFO") and (t[0][2] == "Auto")]

            for name in meta_model_names:
                numerator = [0, 0, 0]
                denominator = 0
                for t in tuples:
                    if t[0][0] == name:
                        numerator = list(map(sum, zip(numerator, t[1])))
                        denominator += 1

                values = list(map(lambda x: x / denominator, numerator))
                data["R^2"][name].append(values[0])
                data["RMSE"][name].append(values[1])
                data["r"][name].append(values[2])

        for metric in data:
            Memory.storeDataFrame(DataFrame(data=data[metric], index=rows, columns=[x for x in data[metric]]),
                                  metric, "questions/q4")

    def predictions(self):
        # TODO: restructure
        model = None
        for name in self.__meta_models:
            print("Test meta-model: " + name)
            model, _ = Memory.loadModel([name])[0]
            model.test(4)
            stats = model.getStats()
            Memory.renewModel(model, model.getName()[:-4])
            self.__tests = self.vectorAddition(self.__tests, stats)

        self.__tests = [list(map(lambda x: x / len(self.__meta_models), stat)) for stat in self.__tests]
        self.__config = [c for (a, b, c) in model.getMetaModels()]

        targets = model.getTargets()
        algorithms = [x[:-5] for x in targets]
        metrics = {0: "r2", 1: "rmse", 2: "r"}
        rows = list()

        all_results = {}
        for i in metrics:
            shap = {a: [] for a in algorithms}
            lime = {a: [] for a in algorithms}
            perm = {a: [] for a in algorithms}
            dCol = {a: [] for a in algorithms}
            metric = {"SHAP": shap, "LIME": lime, "PIMP": perm, "LOFO": dCol}
            print("Metric: " + metrics[i])

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
        rows = None
        model = None
        for name in self.__meta_models:
            print("Compare meta-model: " + name)
            model, _ = Memory.loadModel([name])[0]
            rows = model.compare(models, targets, subsets, 33, renew)
            results = model.getResults()
            Memory.renewModel(model, model.getName()[:-4])
            self.__comparisons = self.vectorAddition(self.__comparisons, results)

        self.__comparisons = [list(map(lambda x: x / len(self.__meta_models), result)) for result in self.__comparisons]
        self.__parameters = model.getResultConfig()

        all_results = {}
        for model in models:
            this_model = {}
            for subset in subsets:
                index = 0
                for a, b, c in self.__parameters:
                    if a == model and c == subset:
                        this_model[b] = self.__comparisons[index]

                    index += 1

            all_results[model] = this_model

        for model in all_results:
            for subset in subsets:
                Memory.storeDataFrame(DataFrame(data=all_results[model], index=rows,
                                                columns=[x for x in all_results[model]]),
                                      model + " x " + subset, "selection")
