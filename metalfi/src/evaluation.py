from decimal import Decimal
import os
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import tqdm

from metalfi.src import memory
from metalfi.src.parameters import Parameters
from metalfi.src import visualization


class Evaluation:
    """
    Train and test meta-models. Evaluate

    Attributes
    ----------
        __meta_models : (List[str])
            File names of :py:class:`MetaModel` instances.
        __tests : (List[List[float]])
            Meta-model performance estimates for different metrics.
        __config : (List[List[str]])
            Meta-model configurations.
        __comparisons :
            Base-model performance estimates for different feature selection approaches.
        __parameters :
            Meta-model configurations for MetaLFI feature selection.
    """

    def __init__(self, meta_models: List[str]):
        self.__meta_models = meta_models
        self.__tests = []
        self.__config = []

        self.__comparisons = []
        self.__comparison_times = []
        self.__parameters = []

    @staticmethod
    def matrix_addition(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
        """
        Given two matrices, computes their sum.

        Parameters
        ----------
            x : First matrix.
            y : Second matrix.

        Returns
        -------
            Sum of `x` and `y`.
        """
        if len(x) == 0:
            return y

        result = [[x[i][j] + y[i][j] for j in range(len(x[i]))] for i in range(len(x))]

        return result

    def questions(self):
        """
        Provides answers to our research questions:
        (Q_1, Q_2, Q_3):
            Fetch the performance estimates of all meta-models.
            (Q_1):
                Group performance estimates by meta-feature subsets and differentiate between
                linear and non-linear meta-models. Examine the differences between meta-models,
                trained on different meta-feature subsets, for linear and non-linear meta-models respectively.
            (Q_2):
                Group performance estimates by meta-targets. Examine the differences between meta-models,
                trained to predict different meta-targets, based on their respective performances.
            (Q_3):
                Group performance estimates by meta-models. Examine the differences between
                meta-model performances.
        (Q_4):
            Fetch the performance estimates of all base-models, trained on different base-feature
            subsets. The base-feature subsets are the results of different feature selection
            approaches, including MetaLFI. Group performance estimates by feature selection
            approaches and examine the differences between base-model performances.

        Use statistical hypothesis testing to determine, whether the differences between the groups
        (across cross validation splits) are significant. Visualize the results.
        """
        directory = Parameters.output_dir + "meta_prediction_performance/"
        file_names = [x for x in os.listdir(directory)
                      if "x" not in x and "long" not in x and x.endswith(".csv")]

        data = {file_name[:-4]: pd.read_csv(directory + file_name) for file_name in file_names}
        pattern = re.compile(r"\$(?P<meta>.+)\_\{(?P<features>.+)\}\((?P<target>.+)\)\$")

        config = [[pattern.match(column).group("meta"),
                   pattern.match(column).group("target"),
                   pattern.match(column).group("features")]
                  for column in data[file_names[0][:-4]].columns if "Index" not in column]

        # Q_4
        directory = Parameters.output_dir + "feature_selection_performance/"
        file_name = [x for x in os.listdir(directory) if x.endswith(".csv") and "_" in x][0]
        comparison_data = {file_name[:-4]: pd.read_csv(directory + file_name)}
        pattern = re.compile(r"\$(?P<meta>.+)\_\{(?P<features>.+) \\times (?P<selection>.+)\}\((?P<target>.+)\)\$")

        config_4 = [[pattern.match(column).group("meta"), pattern.match(column).group("target"),
                     pattern.match(column).group("features"), pattern.match(column).group("selection")]
                    for column in comparison_data[file_name[:-4]].columns if "Index" not in column]

        # Q_1
        subset_names_lin = list(dict.fromkeys([c[2] for c in config if c[2] != "All"]))
        subset_names_non = list(dict.fromkeys([c[2] for c in config]))
        data_1_lin = {metric: {key: [] for key in subset_names_lin} for metric in Parameters.metrics}
        data_1_non = {metric: {key: [] for key in subset_names_non} for metric in Parameters.metrics}

        # Q_2
        target_names = list(dict.fromkeys([c[1] for c in config]))
        data_2 = {metric: {key: [] for key in target_names} for metric in Parameters.metrics}

        # Q_3
        meta_model_names = list(dict.fromkeys([c[0] for c in config]))
        data_3 = {metric: {key: [] for key in meta_model_names} for metric in Parameters.metrics}

        # Q_4
        selection_names = list(set(c[3] for c in config_4))
        data_4 = {meta_target: {key: [] for key in selection_names}
                  for meta_target in Parameters.question_5_parameters()[1]}

        rows = []
        base_data_sets = list(data.values())[0]["Index"]
        for i in range(len(base_data_sets)):
            stats = [list(x) for x in zip(*[data_set.iloc[i][1:] for data_set in data.values()])]
            performances = list(zip(config, stats))
            rows.append(base_data_sets.iloc[i])
            data_1_lin = self.__create_question_csv(performances, subset_names_lin, data_1_lin, 2,
                                                    question=1, linear=True)
            data_1_non = self.__create_question_csv(performances, subset_names_non, data_1_non, 2,
                                                    question=1, linear=False)
            data_2 = self.__create_question_csv(performances, target_names, data_2, 1, question=2)
            data_3 = self.__create_question_csv(performances, meta_model_names, data_3, 0, question=3)

        comp_rows = []
        base_data_sets = list(comparison_data.values())[0]["Index"]
        for i in range(len(base_data_sets)):
            comp_rows.append(base_data_sets.iloc[i])
            comps = [list(x) for x in zip(*[data_set.iloc[i][1:] for data_set in comparison_data.values()])]
            comparisons = list(zip(config_4, comps))
            data_4 = self.__create_question_4_csv(comparisons, data_4)

        def __question_data(data: Dict[str, Dict[str, List[float]]], rows: List[str], suffix: str) \
                -> List[Tuple[pd.DataFrame, str]]:
            return [(pd.DataFrame(data=data[metric], index=rows, columns=data[metric].keys()),
                     metric + suffix) for metric in data]

        q_1_lin = __question_data(data_1_lin, rows, "_featureGroups_linearMetaModels")
        q_1_non = __question_data(data_1_non, rows, "_featureGroups_nonlinearMetaModels")
        q_2 = self.__q_2(data_2, rows)
        q_3 = __question_data(data_3, rows, "_metaModels")
        q_4 = __question_data(data_4, comp_rows, "_featureSelection")

        visualization.compare_means(q_1_lin + q_1_non, "predictions/")
        visualization.compare_means(q_2, "predictions/")
        visualization.compare_means(q_3, "predictions/")
        # visualization.compare_means(q_4, "selection")

    def __q_2(self, data: Dict[str, Dict[str, List[float]]], rows: List[str]) -> List[Tuple[pd.DataFrame, str]]:
        data_frames = []
        for metric in data:
            data_frame = pd.DataFrame(data=data[metric], index=rows, columns=data[metric].keys())
            fi_measures = {fi_measure: [0] * len(rows) for fi_measure in Parameters.fi_measures()}
            base_models = {name: [0] * len(rows) for _, name, _ in Parameters.base_models}

            data_frames.append((data_frame, metric + "_allMetaTargets"))
            data_frames.append(self.__helper_q_2(fi_measures, data_frame, rows, metric,
                                                 "_impMeasures", targets=True))
            data_frames.append(self.__helper_q_2(base_models, data_frame, rows, metric,
                                                 "_baseModels"))

        return data_frames

    @staticmethod
    def __helper_q_2(dictionary: Dict[str, List[float]], data_frame: pd.DataFrame, rows: List[str],
                     metric: str, name: str, targets=False) -> Tuple[pd.DataFrame, str]:
        for key in dictionary:
            if targets:
                subset = [column for column in data_frame.columns if key == column[-4:]]
            else:
                subset = [column for column in data_frame.columns
                          if (key == column[:-5]) and (column[-4:] != "LOFO")]

            for column in subset:
                dictionary[key] = [sum(x) for x in zip(dictionary[key], list(data_frame[column].values))]

            dictionary[key] = [element / len(subset) for element in dictionary[key]]

        return pd.DataFrame(data=dictionary, index=rows, columns=dictionary.keys()), metric + name

    @staticmethod
    def __create_question_csv(performances: List[Tuple[List[str], List[float]]], names: List[str],
                              data: Dict[str, Dict[str, List[float]]], index: int, question: int,
                              linear=False) -> Dict[str, Dict[str, List[float]]]:
        linear_meta_models = [name for _, name, category in Parameters.meta_models
                              if category == "linear"]
        non_linear_meta_models = [name for _, name, category in Parameters.meta_models
                                  if category != "linear"]

        if question == 1:
            if linear:
                tuples = [t for t in performances if t[0][0] in linear_meta_models]
            else:
                tuples = [t for t in performances if t[0][0] in non_linear_meta_models and t[0][0] != "DT"]
        elif question == 2:
            tuples = [t for t in performances
                      if ((t[0][0] in linear_meta_models)
                          and (t[0][2] not in ["Uni", "UniMultiFF", "Multi", "All"]))
                      or ((t[0][0] in non_linear_meta_models)
                          and (t[0][2] not in ["Uni", "UniMultiFF", "Multi"]))]
        elif question == 3:
            tuples = [t for t in performances if (t[0][1][:-4] != "LOFO")
                      and ((t[0][0] in linear_meta_models
                            and (t[0][2] not in ["Uni", "UniMultiFF", "Multi", "All"]))
                           or ((t[0][0] in non_linear_meta_models)
                               and (t[0][2] not in ["Uni", "UniMultiFF", "Multi"])))]
        else:
            tuples = []

        for name in names:
            numerator = [0] * len(Parameters.metrics)
            denominator = 0
            for t in tuples:
                if t[0][index] == name:
                    numerator = [sum(x) for x in zip(numerator, t[1])]
                    denominator += 1

            values = [x / denominator for x in numerator]
            for metric_idx, metric_name in enumerate(Parameters.metrics):
                data[metric_name][name].append(values[metric_idx])

        return data

    @staticmethod
    def __create_question_4_csv(comparisons: List[Tuple[List[str], List[float]]],
                                data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
        for key in data:
            tuples = [x for x in comparisons if x[0][1] == key]
            selection_methods = {method: [] for method in list({x[0][3] for x in comparisons})}

            for t in tuples:
                selection_methods[t[0][3]].append(t[1][0])

            for method in selection_methods.keys():
                data[key][method].append(np.mean(selection_methods[method]))

        return data

    @staticmethod
    def parallelize_predictions(model_name: str, progress_bar) -> Tuple[List[List[float]], List[List[str]], List[str]]:
        """
        Compute performance estimates for meta-models.

        Parameters
        ----------
            name : Name of the :py:class:`MetaModel` instance whose meta-models should be tested.

        Returns
        -------
            Performance estimates, configurations and meta-targets of said meta-models.

        """
        model, _ = memory.load_model(model_name=model_name)
        stats = model[1]
        config = [c for (a, b, c) in model[0]]
        targets = Parameters.targets
        progress_bar.update(n=1)
        return stats, config, targets

    def predictions(self):
        """
        Estimate meta-model performances by testing them on their respective cross-validation test
        splits. Save the results as .csv files.
        """
        progress_bar = tqdm.tqdm(total=len(self.__meta_models), desc="Evaluating meta-models")
        results = [self.parallelize_predictions(meta_model, progress_bar)
                   for meta_model in self.__meta_models]
        progress_bar.close()
        for stats, _, _ in results:
            self.__tests = self.matrix_addition(self.__tests, stats)

        self.__tests = [[x / len(self.__meta_models) for x in stat] for stat in self.__tests]
        self.__config = results[0][1]

        targets = results[0][2]
        algorithms = [x[:-5] for x in targets]
        rows = []

        all_results = {}
        for i, metric_name in enumerate(Parameters.metrics):
            metric = {measure: {a: [] for a in algorithms} for measure in Parameters.fi_measures()}
            index = 0
            for a, b, c in self.__config:
                row = "$" + a + "_{" + c + "}$"
                if row not in rows:
                    rows.append(row)

                metric[b[-4:]][b[:-5]].append(self.__tests[index][i])
                index -= -1

            all_results[metric_name] = metric

        self.create_tables(all_results, rows)
        self.__store_all_results(results)

    @staticmethod
    def create_tables(results: Dict, rows):
        for metric in results:
            for importance in results[metric]:
                data_frame = pd.DataFrame(data=results[metric][importance], index=rows,
                                          columns=results[metric][importance].keys()).round(3)
                for i in range(len(data_frame.index)):
                    for j in range(len(data_frame.columns)):
                        string = str(data_frame.iloc[i].iloc[j])
                        if abs(data_frame.iloc[i].iloc[j]) > 10:
                            d = '%.2e' % Decimal(string)
                            data_frame.iloc[i, j] = d
                        elif "e" in string:
                            match = re.split(r'e', string)
                            match[0] = str(round(float(match[0]), 2))
                            data_frame.iloc[i, j] = float(match[0] + "e" + match[1])
                        else:
                            data_frame.iloc[i, j] = round(data_frame.iloc[i].iloc[j], 3)

                data_frame.to_csv(Parameters.output_dir + "meta_prediction_performance/" +
                                  metric + "x" + importance + ".csv")

    def __store_all_results(self, results: List[Tuple[List[List[float]], List[List[str]], List[str]]]) -> None:
        data = {key: [] for key in ["base_dataset", "meta_model", "meta_feature_group",
                                    "base_model", "importance_measure", "r^2"]}
        for i in range(len(self.__meta_models)):
            base_dataset = self.__meta_models[i]
            for j in range(len(self.__config)):
                meta, target, features = self.__config[j]
                data["base_dataset"].append(base_dataset)
                data["meta_model"].append(meta)
                data["meta_feature_group"].append(features)
                data["base_model"].append(target[:-5])
                data["importance_measure"].append(target[-4:])
                data["r^2"].append(results[i][0][j][0])

        pd.DataFrame(data=data).to_csv(Parameters.output_dir + "meta_prediction_performance.csv", index=False)

    @staticmethod
    def new_parallel_comparisons(model_name: str, progress_bar):
        data = memory.load_model(model_name=model_name)
        results = data[0][3]
        times = data[0][4]
        key = list(results.keys())[0]
        progress_bar.update(n=1)
        return results[key], times[key]

    def new_comparisons(self):
        rows = ["ANOVA", "MI", "Bagging", "PIMP", "MetaLFI", "Baseline"]
        meta_models, _, subsets = Parameters.question_5_parameters()

        progress_bar = tqdm.tqdm(total=len(self.__meta_models), desc="Comparing feature-selection approaches")
        results = [self.new_parallel_comparisons(meta_model, progress_bar)
                   for meta_model in self.__meta_models]

        progress_bar.close()

        for result, time in results:
            self.__comparisons = self.matrix_addition(self.__comparisons, result)
            self.__comparison_times = self.matrix_addition(self.__comparison_times, time)

        self.__comparisons = [[x / len(self.__meta_models) for x in result]
                              for result in self.__comparisons]
        self.__comparison_times = [[x / len(self.__meta_models) for x in time]
                                   for time in self.__comparison_times]
        self.__parameters = memory.load_model(model_name=self.__meta_models[0])[0][2]

        all_results = {}
        for _, model, _ in [x for x in Parameters.meta_models if x[1] in meta_models]:
            this_model = {}
            for subset in subsets:
                index = 0
                for a, b, c in self.__parameters:
                    if a == model and c == subset:
                        try:
                            z = this_model[c]
                        except KeyError:
                            this_model[c] = dict()
                        finally:
                            this_model[c][b] = self.__comparisons[index]

                    index += 1

            all_results[model] = this_model

        plotting = {"models": meta_models, "subsets": subsets}
        self.plot_accuracies(all_results, rows, plotting)
        self.__store_all_comparisons([result for result, _ in results], [time for _, time in results], rows)

    @staticmethod
    def plot_accuracies(results, rows, plotting):
        data = []
        for model in results:
            for subset in results[model]:
                if model in plotting["models"] and subset in plotting["subsets"]:
                    data.append((pd.DataFrame(data=results[model][subset], index=rows,
                                              columns=results[model][subset].keys()),
                                 "featureSelectionAcc x " + model + " x " + subset))
        visualization.performance(data)

    def __store_all_comparisons(self, results: List[List[List[float]]], times: List[List[List[float]]],
                                rows: List[str]) -> None:
        data = {key: [] for key in [
            "base_dataset", "meta_model", "meta_feature_group", "feature_selection_approach",
            "base_model", "importance_measure", "accuracy", "runtime"]}
        for i in range(len(self.__parameters)):
            for j in range(len(rows)):
                for k in range(len(self.__meta_models)):
                    data["base_dataset"].append(self.__meta_models[k])
                    data["meta_model"].append(self.__parameters[i][0])
                    data["meta_feature_group"].append(self.__parameters[i][2])
                    data["feature_selection_approach"].append(rows[j])
                    data["base_model"].append(self.__parameters[i][1][:-5])
                    data["importance_measure"].append(self.__parameters[i][1][-4:])
                    data["accuracy"].append(results[k][i][j])
                    if j != 5:
                        data["runtime"].append(times[k][i][j])
                    else:
                        data["runtime"].append(0)

        pd.DataFrame(data=data).to_csv(Parameters.output_dir + "feature_selection_performance.csv", index=False)
