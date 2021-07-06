from decimal import Decimal
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import Orange
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler

from metalfi.src.memory import Memory
from metalfi.src.parameters import Parameters


class Visualization:
    """"
    Postprocessing data and visualization.
    """

    @staticmethod
    def fetch_runtime_data(substring, threshold: int = 1000000):
        # Deprecated
        directory = "output/runtime"
        file_names = list(filter(lambda x: x.endswith('.csv') and substring in x, Memory.get_contents(directory)))
        data = list()

        for name in file_names:
            file = Memory.load(name, directory)
            data.append((file, name))

        summary = {}
        columns = list(data[0][0].columns)
        columns.pop(0)
        for column in columns:
            summary[column] = pd.DataFrame(columns=["size", column])

        for file, name in data:
            for x in summary:
                split = name.split("X")
                temp = int(split[3][:-4]) * int(split[2])
                if temp < threshold:
                    summary[x] = summary[x].append(pd.Series([temp, file[x].values[0]],
                                                             index=summary[x].columns), ignore_index=True)

        for x in summary:
            summary[x] = summary[x].sort_values(by=["size"])

        return summary

    @staticmethod
    def runtime_graph(name):
        # Deprecated
        target_data = Visualization.fetch_runtime_data("XtargetX")
        meta_data = Visualization.fetch_runtime_data("XmetaX")
        for x in target_data:
            if x in ["LOFO", "SHAP", "LIME", "total"]:
                continue
            target_data[x][x] /= 5
            plt.plot(target_data[x].columns[0], x, data=target_data[x], linewidth=2)

        for x in meta_data:
            if x in ["total", "multivariate"]:
                continue
            plt.plot(meta_data[x].columns[0], x, data=meta_data[x], linewidth=2)

        plt.legend()
        Memory.store_visual(plt, name, "runtime")

    @staticmethod
    def runtime_boxplot(threshold, targets, meta, name):
        # Deprecated
        target_data = Visualization.fetch_runtime_data("XtargetX", threshold)
        meta_data = Visualization.fetch_runtime_data("XmetaX", threshold)

        data = list()
        names = list()
        for x in target_data:
            if x not in targets:
                continue
            names.append(x)
            target_data[x][x] /= 5
            data.append(target_data[x][x].values)

        for x in meta_data:
            if x not in meta:
                continue
            names.append(x)
            data.append(meta_data[x][x].values)

        _, ax = plt.subplots()
        ax.boxplot(data, showfliers=False)
        plt.xticks(list(range(1, len(data) + 1)), names)
        Memory.store_visual(plt, name + "_box", "runtime")

    @staticmethod
    def performance(data):
        """
        Create and save bar charts. Visualizes the performances of different feature selection approaches.
        """
        for frame, name in data:
            width = 0.1
            _, ax = plt.subplots()

            anova = frame.loc["ANOVA"].values
            mi = frame.loc["MI"].values
            bagging = frame.loc["Bagging"].values
            pimp = frame.loc["PIMP"].values
            baseline = frame.loc["Baseline"].values
            meta = frame.loc["MetaLFI"].values

            x = np.arange(len(anova))

            ax.bar(x - 5 * width / 2, anova, width, label="ANOVA")
            ax.bar(x - 3 * width / 2, mi, width, label="MI")
            ax.bar(x - width / 2, bagging, width, label="Bagging")
            ax.bar(x + width / 2, pimp, width, label="PIMP")
            ax.bar(x + 3 * width / 2, baseline, width, label="Baseline")
            ax.bar(x + 5 * width / 2, meta, width, label="MetaLFI")

            ax.set_ylabel("Acc. Scores")
            ax.set_yticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.775, 0.8, 0.825, 0.85])
            ax.set_xticks(x)
            ax.set_xticklabels(list(frame.columns))
            ax.legend()
            plt.ylim([0.5, 0.85])

            Memory.store_visual(plt, name, "selection")

    @staticmethod
    def meta_feature_importance():
        """
        Create and save bar charts. Visualizes the importance of meta-features for different meta-targets.
        """
        directory = "output/importance"
        file_names = Memory.get_contents(directory)
        data = [(Memory.load(name, directory), name) for name in file_names if ".csv" in name]
        frame = data[0][0].sort_values(by="PIMP")
        for base_model in set(frame["base_model"]):
            for imp in set(frame["importance_measure"]):
                new_frame = frame[frame["base_model"] == base_model]
                new_frame = new_frame[new_frame["importance_measure"] == imp]
                plt.barh(list(new_frame["meta-features"])[-15:], list(new_frame["PIMP"])[-15:])
                plt.yticks(list(new_frame["meta-features"])[-15:])
                Memory.store_visual(plt, base_model + "x" + imp, "importance")

    @staticmethod
    def compare_means(data: List[Tuple[pd.DataFrame, str]], folder: str):
        """
        Determine, whether the differences in meta-model performance are significant:
        Group data across cross validation splits and employ the non-parametric Friedman test.
        If the differences between the cross validation splits are significant,
        employ the Nemenyi post-hoc test and visualize its results using a Critical Differences (CD) diagram.

        Parameters
        ----------
            data :
                List of cross validation results as a tuple. Each tuple contains meta-model performance estimates
                on a given cross validation split and the name of said split.
            folder :
                Name of the subdirectory of metalfi/data, in which the diagrams are supposed to be saved.
        """
        for data_frame, metric in data:
            d = list()
            names = list()
            ranks = [0] * len(data_frame.columns)

            for i in range(len(data_frame.index)):
                copy = data_frame.iloc[i].values
                values = np.array(copy) if ("RMSE" in metric) else np.array(list(map(lambda x: -x, copy)))
                temp = values.argsort()
                current_ranks = np.array([0] * len(values))
                current_ranks[temp] = np.arange(len(values))
                current_ranks = list(map(lambda x: x + 1, current_ranks))
                ranks = list(map(np.add, ranks, current_ranks))

            ranks = list(map(lambda x: x / len(data_frame.index), ranks))

            for column in data_frame.columns:
                names.append(column)
                d.append(data_frame[column].values)

            if 24 >= len(names) >= 3:
                _, p_value = ss.friedmanchisquare(*d)
                if p_value < 0.05:
                    Visualization.__create_cd_diagram(names, ranks, metric, d, folder)

    @staticmethod
    def __create_cd_diagram(names: List[str], ranks: List[float], metric: str, data: List[np.array], folder: str):
        cd = Orange.evaluation.compute_CD(ranks, 28) if len(ranks) < 21 else 3.616
        Orange.evaluation.graph_ranks(ranks, names, cd=cd)
        Memory.store_visual(plt, metric + "_cd", folder)
        plt.close()

        _, ax = plt.subplots()
        ax.boxplot(data, notch=True, showfliers=False)
        ax.set_xticks(list(range(1, len(data) + 1)))
        ax.set_xticklabels(names)
        Memory.store_visual(plt, metric + "_means", folder)
        plt.close()

    @staticmethod
    def correlate_metrics():
        """
        Compute pairwise Spearman correlation coefficients between performance metrics.
        """
        new = {metric: list() for metric in Parameters.metrics}
        directory = "output/predictions"
        file_names = Memory.get_contents(directory)
        data = [(Memory.load(name, directory), name) for name in file_names if "x" not in name]

        columns = data[0][0].columns

        for d, n in data:
            for column in columns[1:]:
                new[n[:-4]] += list(d[column].values)

        frame = pd.DataFrame.from_dict(new)
        corr = frame.corr("spearman")
        Memory.store_data_frame(corr, "metrics_corr", "", True)

    @staticmethod
    def correlate_targets():
        """
        Compute pairwise Spearman correlation coefficients between meta-features and meta-targets,
        grouped by feature importance measure.

        Returns
        -------
            Mean and maximum values over all correlation coefficients for each group.

        """
        directory = "input"
        file_names = Memory.get_contents(directory)
        sc = StandardScaler()
        data = list()

        for name in file_names:
            d = Memory.load(name, directory)
            df = pd.DataFrame(data=sc.fit_transform(d), columns=d.columns)
            data.append(df)

        frame = pd.concat(data)

        lofo = [x for x in frame.columns if "LOFO" in x]
        shap = [x for x in frame.columns if "SHAP" in x]
        pimp = [x for x in frame.columns if "PIMP" in x]
        lime = [x for x in frame.columns if "LIME" in x]
        lm = [x for x in frame.columns if not x.startswith("target_")]

        matrix = frame.corr("spearman")
        matrix = matrix.drop(lofo + shap + pimp + lime + lm, axis=0)
        matrix = matrix.drop([x for x in list(frame.columns) if x not in lofo + shap + pimp + lime], axis=1)

        def __f(targets):
            return np.round(np.mean([np.mean(list([val for val in list(map(abs, matrix[x].values)) if val < 1]))
                                     for x in targets]), 2)

        def __f_2(targets):
            return np.round(np.max([np.mean(list([val for val in list(map(abs, matrix[x].values)) if val < 1]))
                                    for x in targets]), 2)

        d = {'lofo': [__f(lofo), __f_2(lofo)], 'shap': [__f(shap), __f_2(shap)],
             'lime': [__f(lime), __f_2(lime)], 'pimp': [__f(pimp), __f_2(pimp)]}
        data_frame = pd.DataFrame(data=d, index=["mean", "max"], columns=["lofo", "shap", "lime", "pimp"])
        Memory.store_data_frame(data_frame, "target_corr", "tables", True)

    @staticmethod
    def create_histograms():
        """
        Group meta-targets by feature importance measure and visualize their distributions as histograms.
        """
        directory = "input"
        file_names = Memory.get_contents(directory)
        data = list()

        for name in file_names:
            d = Memory.load(name, directory)
            df = pd.DataFrame(data=d, columns=d.columns)
            data.append(df)

        frame = pd.concat(data)

        meta_targets = [([x for x in frame.columns if "LOFO" in x], "LOFO", 0, 0),
                        ([x for x in frame.columns if "SHAP" in x], "SHAP", 0, 1),
                        ([x for x in frame.columns if "LIME" in x], "LIME", 1, 0),
                        ([x for x in frame.columns if "PIMP" in x], "PIMP", 1, 1)]

        _, axs = plt.subplots(2, 2)
        for target, name, x, y in meta_targets:
            values = list()
            for value in [list(frame[column].values) for column in target]:
                values += value

            axs[x, y].hist(x=values, rwidth=1, bins=len(values))
            axs[x, y].set_title(name)
            axs[x, y].set_xlim(np.quantile(values, 0.05), np.quantile(values, 0.75))

        Memory.store_visual(plt, "Histograms", "tables")
