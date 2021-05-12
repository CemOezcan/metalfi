import re
from functools import partial
from multiprocessing import Pool

import numpy as np
from pandas import DataFrame
import tqdm

from metalfi.src.memory import Memory
from metalfi.src.parameters import Parameters
from metalfi.src.visualization import Visualization


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

    def questions(self):
        directory = "output/predictions"
        file_names = list(filter(lambda x: "x" not in x and x.endswith(".csv"), Memory.get_contents(directory)))

        data = {name[:-4]: Memory.load(name, directory) for name in file_names}
        pattern = re.compile(r"\$(?P<meta>.+)\_\{(?P<features>.+)\}\((?P<target>.+)\)\$")

        config = [[pattern.match(column).group("meta"),
                   pattern.match(column).group("target"),
                   pattern.match(column).group("features")]
                  for column in data[file_names[0][:-4]].columns if "Unnamed" not in column]

        # Q_5
        directory = "output/selection"
        file_name = list(filter(lambda x: x.endswith(".csv") and "_" in x, Memory.get_contents(directory)))[0]
        comparison_data = {file_name[:-4]: Memory.load(file_name, directory)}
        pattern = re.compile(r"\$(?P<meta>.+)\_\{(?P<features>.+) \\times (?P<selection>.+)\}\((?P<target>.+)\)\$")

        config_5 = [[pattern.match(column).group("meta"), pattern.match(column).group("target"),
                     pattern.match(column).group("features"), pattern.match(column).group("selection")]
                    for column in comparison_data[file_name[:-4]].columns if "Unnamed" not in column]

        # Q_2
        subset_names_lin = list(dict.fromkeys([c[2] for c in config if c[2] != "All"]))
        subset_names_non = list(dict.fromkeys([c[2] for c in config]))
        data_2_lin = {metric: {key: list() for key in subset_names_lin} for metric in Parameters.metrics.values()}
        data_2_non = {metric: {key: list() for key in subset_names_non} for metric in Parameters.metrics.values()}

        # Q_3
        target_names = list(dict.fromkeys([c[1] for c in config]))
        data_3 = {metric: {key: list() for key in target_names} for metric in Parameters.metrics.values()}

        # Q_4
        meta_model_names = list(dict.fromkeys([c[0] for c in config]))
        data_4 = {metric: {key: list() for key in meta_model_names} for metric in Parameters.metrics.values()}

        # Q_5
        selection_names = list(set(c[3] for c in config_5))
        data_5 = {meta_target: {key: list() for key in selection_names}
                  for meta_target in Parameters.question_5_parameters()[1]}

        rows = list()
        base_data_sets = list(data.values())[0]["Unnamed: 0"]
        for i in range(len(base_data_sets)):
            stats = list(map(lambda x: list(x), zip(*[data_set.iloc[i][1:] for data_set in data.values()])))
            comps = list(map(lambda x: list(x), zip(*[data_set.iloc[i][1:] for data_set in comparison_data.values()])))

            performances = list(zip(config, stats))
            comparisons = list(zip(config_5, comps))
            rows.append(base_data_sets.iloc[i])

            data_2_lin = self.createQuestionCsv(performances, subset_names_lin, data_2_lin, 2, question=2, linear=True)
            data_2_non = self.createQuestionCsv(performances, subset_names_non, data_2_non, 2, question=2, linear=False)
            data_3 = self.createQuestionCsv(performances, target_names, data_3, 1, question=3)
            data_4 = self.createQuestionCsv(performances, meta_model_names, data_4, 0, question=4)
            data_5 = self.createQuestion5Csv(comparisons, data_5)

        def question_data(data, rows, suffix):
            return [(DataFrame(data=data[metric], index=rows, columns=[x for x in data[metric]]), metric + suffix)
                    for metric in data]

        q_2_lin = question_data(data_2_lin, rows, "_LIN")
        q_2_non = question_data(data_2_non, rows, "_NON")
        q_3 = self.q_3(data_3, rows)
        q_4 = question_data(data_4, rows, "")
        q_5 = question_data(data_5, rows, "")

        Visualization.compare_means(q_2_lin + q_2_non, "groups")
        Visualization.compare_means(q_3, "targets")
        Visualization.compare_means(q_4, "models")
        Visualization.compare_means(q_5, "selection")

    def q_3(self, data, rows):
        data_frames = list()
        for metric in data:
            data_frame = DataFrame(data=data[metric], index=rows, columns=[x for x in data[metric]])
            fi_measures = {fi_measure: [0] * len(rows) for fi_measure in Parameters.fi_measures()}
            base_models = {name: [0] * len(rows) for _, name, _ in Parameters.base_models}

            data_frames.append((data_frame, metric))
            data_frames.append(self.helper_q_3(fi_measures, data_frame, rows, metric, "targets_", targets=True))
            data_frames.append(self.helper_q_3(base_models, data_frame, rows, metric, "base_"))

        return data_frames

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

        return DataFrame(data=dictionary, index=rows, columns=[x for x in dictionary]), name + metric

    def createQuestionCsv(self, performances, names, data, index, question, linear=False):
        linear_meta_models = [name for _, name, _ in filter(lambda x: x[2] == "linear", Parameters.meta_models)]
        non_linear_meta_models = [name for _, name, _ in filter(lambda x: x[2] != "linear", Parameters.meta_models)]

        if question == 2:
            if linear:
                tuples = [t for t in performances if (t[0][0] in linear_meta_models)]
            else:
                tuples = [t for t in performances if (t[0][0] in non_linear_meta_models)]
        elif question == 3:
            tuples = [t for t in performances
                      if ((t[0][0] in linear_meta_models) and (t[0][2] == "LM"))
                      or ((t[0][0] in non_linear_meta_models) and (t[0][2] == "Auto"))]
        elif question == 4:
            tuples = [t for t in performances if (t[0][1][:-4] != "LOFO")
                      and ((t[0][0] in linear_meta_models and (t[0][2] == "LM"))
                           or ((t[0][0] in non_linear_meta_models) and (t[0][2] == "Auto")))]
        else:
            tuples = list()

        for name in names:
            numerator = [0] * len(Parameters.metrics.keys())
            denominator = 0
            for t in tuples:
                if t[0][index] == name:
                    numerator = list(map(sum, zip(numerator, t[1])))
                    denominator += 1

            values = list(map(lambda x: x / denominator, numerator))
            for i in Parameters.metrics.keys():
                data[Parameters.metrics[i]][name].append(values[i])

        return data

    def createQuestion5Csv(self, comparisons, data):
        for key in data:
            tuples = list(filter(lambda x: x[0][1] == key, comparisons))
            selection_methods = {method: list() for method in list(set(map(lambda x: x[0][3], comparisons)))}

            for t in tuples:
                selection_methods[t[0][3]].append(t[1][0])

            for method in selection_methods.keys():
                data[key][method].append(np.mean(selection_methods[method]))

        return data

    @staticmethod
    def parallelize_predictions(name):
        model, _ = Memory.load_model([name])[0]
        model.test()
        stats = model.getStats()
        Memory.renew_model(model, model.getName()[:-4])
        config = [c for (a, b, c) in model.getMetaModels()]
        targets = Parameters.targets
        return stats, config, targets

    def predictions(self):
        with Pool(processes=4) as pool:
            progress_bar = tqdm.tqdm(total=len(self.__meta_models), desc="Evaluating meta-models")
            results = [
                pool.map_async(
                    self.parallelize_predictions,
                    (meta_model, ),
                    callback=(lambda x: progress_bar.update(n=1)))
                for meta_model in self.__meta_models]

            results = [x.get()[0] for x in results]
            pool.close()
            pool.join()

        progress_bar.close()
        for stats, _, _ in results:
            self.__tests = self.vectorAddition(self.__tests, stats)

        self.__tests = [list(map(lambda x: x / len(self.__meta_models), stat)) for stat in self.__tests]
        self.__config = results[0][1]

        targets = results[0][2]
        algorithms = [x[:-5] for x in targets]
        rows = list()

        all_results = {}
        for i in Parameters.metrics:
            metric = {measure: {a: list() for a in algorithms} for measure in Parameters.fi_measures()}
            index = 0
            for a, b, c in self.__config:
                row = "$" + a + "_{" + c + "}$"
                if row not in rows:
                    rows.append(row)

                metric[b[-4:]][b[:-5]].append(self.__tests[index][i])
                index -= -1

            all_results[Parameters.metrics[i]] = metric

        for metric in all_results:
            for importance in all_results[metric]:
                data_frame = DataFrame(data=all_results[metric][importance], index=rows,
                                       columns=[x for x in all_results[metric][importance]])
                Memory.store_data_frame(data_frame.round(3), metric + "x" + importance, "predictions")

        self.storeAllRsults(results)

    @staticmethod
    def parallel_comparisons(name, models, targets, subsets, renew):
        model, _ = Memory.load_model([name])[0]
        model.compare(models, targets, subsets, 33, renew)
        results = model.getResults()
        Memory.renew_model(model, model.getName()[:-4])
        return results

    def storeAllRsults(self, results):
        columns = ["$" + meta + "_{" + features + "}(" + target + ")$" for meta, target, features in self.__config]
        index = self.__meta_models

        for key in Parameters.metrics.keys():
            data = [tuple(map((lambda x: x[key]), results[i][0])) for i in range(len(index))]
            Memory.store_data_frame(DataFrame(data, columns=columns, index=index), Parameters.metrics[key], "predictions")

    def comparisons(self, models, targets, subsets, renew=False):
        with Pool(processes=4) as pool:
            progress_bar = tqdm.tqdm(total=len(self.__meta_models), desc="Comparing feature-selection approaches")

            results = [pool.map_async(
                partial(self.parallel_comparisons, models=models, targets=targets, subsets=subsets, renew=renew),
                (model, ), callback=(lambda x: progress_bar.update(n=1))) for model in self.__meta_models]

            results = [x.get()[0] for x in results]
            pool.close()
            pool.join()

        progress_bar.close()
        model, _ = Memory.load_model([self.__meta_models[0]])[0]
        rows = model.compare(models, targets, subsets, 33, False)
        for result in results:
            self.__comparisons = self.vectorAddition(self.__comparisons, result)

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
                Memory.store_data_frame(DataFrame(data=all_results[model], index=rows,
                                                  columns=[x for x in all_results[model]]),
                                        model + " x " + subset, "selection", True)

        self.store_all_comparisons(results, rows, "all_comparisons")

    def store_all_comparisons(self, results, rows, name):
        data = {"$" + self.__parameters[i][0] + "_{" + self.__parameters[i][2]
                + " \\times " + rows[j] + "}(" + self.__parameters[i][1] + ")$": list(map(lambda x: x[i][j], results))
                for i in range(len(self.__parameters)) for j in range(len(rows))}

        Memory.store_data_frame(DataFrame(data, index=self.__meta_models), name, "selection")
