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

    def questions(self, subset_names):
        model, _ = Memory.loadModel([self.__meta_models[0]])[0]
        config = [c for (a, b, c) in model.getMetaModels()]

        # Q_2
        subset_names_lin = list(dict.fromkeys([c[2] for c in config if c[2] != "All"]))
        subset_names_non = list(dict.fromkeys([c[2] for c in config]))
        data_2_lin = {"R^2": {key: list() for key in subset_names_lin},
                      "RMSE": {key: list() for key in subset_names_lin},
                      "r": {key: list() for key in subset_names_lin}}

        data_2_non = {"R^2": {key: list() for key in subset_names_non},
                      "RMSE": {key: list() for key in subset_names_non},
                      "r": {key: list() for key in subset_names_non}}

        # Q_3
        target_names = list(dict.fromkeys([c[1] for c in config]))
        data_3 = {"R^2": {key: list() for key in target_names},
                  "RMSE": {key: list() for key in target_names},
                  "r": {key: list() for key in target_names}}

        # Q_4
        meta_model_names = list(dict.fromkeys([c[0] for c in config]))
        data_4 = {"R^2": {key: list() for key in meta_model_names},
                  "RMSE": {key: list() for key in meta_model_names},
                  "r": {key: list() for key in meta_model_names}}

        # Q_5
        selection_names = ["ANOVA", "MI", "FI", "MetaLFI"]
        data_5 = {"LOG_SHAP": {key: list() for key in selection_names},
                  "linSVC_SHAP": {key: list() for key in selection_names},
                  "NB_SHAP": {key: list() for key in selection_names},
                  "RF_SHAP": {key: list() for key in selection_names},
                  "SVC_SHAP": {key: list() for key in selection_names}}

        rows = list()
        rows_5 = list()
        for data_set in self.__meta_models:
            print("Questions, " + data_set)
            rows.append(data_set)
            model, _ = Memory.loadModel([data_set])[0]

            data_2_lin = self.createQuestionCsv(model, config, subset_names_lin, data_2_lin, 2, question=2, linear=True)
            data_2_non = self.createQuestionCsv(model, config, subset_names_non, data_2_non, 2, question=2, linear=False)
            data_3 = self.createQuestionCsv(model, config, target_names, data_3, 1, question=3)
            data_4 = self.createQuestionCsv(model, config, meta_model_names, data_4, 0, question=4)

            if data_set in subset_names:
                rows_5.append(data_set)
                data_5 = self.createQuestion5Csv(model, data_5, "linSVR", "LM")

        self.q_2(data_2_lin, rows, "LIN")
        self.q_2(data_2_non, rows, "NON")
        self.q_3(data_3, rows)
        self.q_4(data_4, rows)
        self.q_5(data_5, rows_5)

    def q_2(self, data, rows, end):
        for metric in data:
            Memory.storeDataFrame(DataFrame(data=data[metric], index=rows, columns=[x for x in data[metric]]),
                                  metric + end, "questions/q2")

    def q_3(self, data, rows):
        for metric in data:
            data_frame = DataFrame(data=data[metric], index=rows, columns=[x for x in data[metric]])
            Memory.storeDataFrame(data_frame, metric, "questions/q3")

            dictionary = {"SHAP": [0] * len(rows), "PIMP": [0] * len(rows), "LIME": [0] * len(rows),
                          "LOFO": [0] * len(rows)}
            self.helper_q_3(dictionary, data_frame, rows, metric, "targets_", targets=True)

            dictionary = {"linSVC": [0] * len(rows), "LOG": [0] * len(rows), "RF": [0] * len(rows),
                          "NB": [0] * len(rows), "SVC": [0] * len(rows)}
            self.helper_q_3(dictionary, data_frame, rows, metric, "base_")

    def q_4(self, data, rows):
        for metric in data:
            Memory.storeDataFrame(DataFrame(data=data[metric], index=rows, columns=[x for x in data[metric]]),
                                  metric, "questions/q4")

    def q_5(self, data, rows):
        for target in data:
            Memory.storeDataFrame(DataFrame(data=data[target], index=rows, columns=[x for x in data[target]]),
                                  target, "questions/q5")

    @staticmethod
    def helper_q_3(dictionary, data_frame, rows, metric, name, targets=False):
        for key in dictionary:
            if targets:
                subset = [column for column in data_frame.columns if (key == column[-4:])]
            else:
                subset = [column for column in data_frame.columns if (key == column[:-5]) and (column[-4:] != "LOFO")]

            for column in subset:
                dictionary[key] = list(map(sum, zip(dictionary[key], list(data_frame[column].values))))

            dictionary[key] = [element / len(subset) for element in dictionary[key]]

        Memory.storeDataFrame(DataFrame(data=dictionary, index=rows, columns=[x for x in dictionary]),
                              name + metric, "questions/q3")

    def createQuestionCsv(self, model, config, names, data, index, question, linear=False):
        if question == 2:
            if linear:
                tuples = [t for t in list(zip(config, model.getStats())) if (t[0][0].lower().startswith("lin"))]
            else:
                tuples = [t for t in list(zip(config, model.getStats())) if not (t[0][0].lower().startswith("lin"))]
        elif question == 3:
            tuples = [t for t in list(zip(config, model.getStats()))
                      if (t[0][0].lower().startswith("lin") and (t[0][2] == "LM"))
                      or (((t[0][0] == "RF") or (t[0][0] == "SVR")) and (t[0][2] == "Auto"))]
        elif question == 4:
            tuples = [t for t in list(zip(config, model.getStats()))
                      if (t[0][1][:-4] != "LOFO") and ((t[0][0].lower().startswith("lin") and (t[0][2] == "LM"))
                      or (((t[0][0] == "RF") or (t[0][0] == "SVR")) and (t[0][2] == "Auto")))]
        else:
            tuples = list()

        for name in names:
            numerator = [0, 0, 0]
            denominator = 0
            for t in tuples:
                if t[0][index] == name:
                    numerator = list(map(sum, zip(numerator, t[1])))
                    denominator += 1

            values = list(map(lambda x: x / denominator, numerator))
            data["R^2"][name].append(values[0])
            data["RMSE"][name].append(values[1])
            data["r"][name].append(values[2])

        return data

    def createQuestion5Csv(self, model, data, meta_model_name, subset_name):
        tuples = [t for t in list(zip(model.getResultConfig(), model.getResults()))
                  if (t[0][0] == meta_model_name) and (t[0][2] == subset_name)]

        for key in data:
            values = [0, 0, 0, 0]
            avg = 0

            for t in [t for t in tuples if t[0][1] == key]:
                values = list(map(sum, zip(values, t[1])))
                avg += 1

            values = [value / avg for value in values]
            data[key]["ANOVA"].append(values[0])
            data[key]["MI"].append(values[1])
            data[key]["FI"].append(values[2])
            data[key]["MetaLFI"].append(values[3])

        return data

    def predictions(self):
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
                row = "$" + a + "_{" + c + "}$"
                if row not in rows:
                    rows.append(row)

                metric[b[-4:]][b[:-5]].append(self.__tests[index][i])
                index -= -1

            all_results[metrics[i]] = metric

        for metric in all_results:
            for importance in all_results[metric]:
                data_frame = DataFrame(data=all_results[metric][importance], index=rows,
                                       columns=[x for x in all_results[metric][importance]])
                Memory.storeDataFrame(data_frame.round(3), metric + "x" + importance, "predictions")

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
