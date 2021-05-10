import os
import re
import Orange

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scikit_posthocs as sp
from pandas import DataFrame
from decimal import Decimal
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler

from metalfi.src.memory import Memory
from metalfi.src.parameters import Parameters


class Visualization:

    @staticmethod
    def fetch_runtime_data(substring, threshold=1000000):
        directory = "output/runtime"

        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if (name.endswith('.csv') and substring in name)]

        data = list()
        for name in file_names:
            file = Memory.load(name, directory)
            data.append((file, name))

        summary = {}
        columns = list(data[0][0].columns)
        columns.pop(0)
        for column in columns:
            summary[column] = DataFrame(columns=["size", column])

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
        target_data = Visualization.fetch_runtime_data("XtargetX")
        meta_data = Visualization.fetch_runtime_data("XmetaX")
        for x in target_data:
            if x == "LOFO" or x == "SHAP" or x == "LIME" or x == "total":
                continue
            target_data[x][x] /= 5
            plt.plot(target_data[x].columns[0], x, data=target_data[x], linewidth=2)

        for x in meta_data:
            if x == "total" or x == "multivariate":
                continue
            plt.plot(meta_data[x].columns[0], x, data=meta_data[x], linewidth=2)

        plt.legend()
        Memory.storeVisual(plt, name, "runtime")

    @staticmethod
    def runtime_boxplot(threshold, targets, meta, name):
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

        fig, ax = plt.subplots()
        ax.boxplot(data, showfliers=False)
        plt.xticks(list(range(1, len(data) + 1)), names)
        Memory.storeVisual(plt, name + "_box", "runtime")

    @staticmethod
    def fetch_predictions():
        directory = "output/predictions"

        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]

        for name in file_names:
            frame = Memory.load(name, directory).set_index("Unnamed: 0")
            for column in frame.columns:
                frame = frame.round({column: 3})

            path = Memory.getPath() / (directory + "/" + name)
            frame.to_csv(path, header=True)

        data = [(Memory.load(name, directory).set_index("Unnamed: 0"), name) for name in file_names]

        return data

    @staticmethod
    def performance():
        directory = "output/selection"
        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]

        data = [(Memory.load(name, directory).set_index("Unnamed: 0"), name) for name in file_names]
        for frame, name in data:
            width = 0.2
            fig, ax = plt.subplots()

            anova = frame.loc["ANOVA"].values
            mi = frame.loc["MI"].values
            fi = frame.loc["FI"].values
            meta = frame.loc["MetaLFI"].values

            x = np.arange(len(anova))

            pos_anova = ax.bar(x - 1.5 * width, anova, width, label="ANOVA")
            pos_mi = ax.bar(x - width / 2, mi, width, label="MI")
            pos_fi = ax.bar(x + width / 2, fi, width, label="FI")
            pos_meta = ax.bar(x + 1.5 * width, meta, width, label="MetaLFI")

            """plt.bar(pos_anova, anova, label="ANOVA")
            plt.bar(pos_mi, mi, label="MI")"""
            """plt.bar(pos_fi, fi, label="FI")
            plt.bar(pos_meta, meta, label="MetaLFI")"""

            ax.set_ylabel("Acc. Scores")
            ax.set_yticks([0.775, 0.8, 0.825, 0.85])
            ax.set_xticks(x)
            ax.set_xticklabels(list(frame.columns))
            ax.legend()
            plt.ylim([0.75, 0.85])

            Memory.storeVisual(plt, name[:-4], "selection")

    @staticmethod
    def metaFeatureImportance():
        directory = "output/importance"
        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]
        data = [(Memory.load(name, directory), name) for name in file_names if ".csv" in name]

        for frame, name in data:
            frame = frame.sort_values(by="mean absolute SHAP")
            plt.barh(list(frame["meta-features"])[-15:], list(frame["mean absolute SHAP"])[-15:])
            plt.yticks(list(frame["meta-features"])[-15:])
            Memory.storeVisual(plt, name[:-4], "importance")

    @staticmethod
    def compareMeans(data, folder):
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

            if len(names) <= 24:
                val, p_value = ss.friedmanchisquare(*d)
                if p_value < 0.05:
                    #sp.sign_array(sp.posthoc_nemenyi_friedman(np.array(d).T))
                    Visualization.createTimeline(names, ranks, metric, d, folder)

    @staticmethod
    def createTimeline(names, ranks, metric, data, folder):
        cd = Orange.evaluation.compute_CD(ranks, 28) if len(ranks) < 21 else 3.616
        Orange.evaluation.graph_ranks(ranks, names, cd=cd)
        Memory.storeVisual(plt, metric + "_cd", folder)
        plt.close()

        fig, ax = plt.subplots()
        ax.boxplot(data, notch=True, showfliers=False)
        ax.set_xticks(list(range(1, len(data) + 1)))
        ax.set_xticklabels(names)
        Memory.storeVisual(plt, metric + "_means", folder)
        plt.close()

    @staticmethod
    def correlateMetrics():
        new = {metric: list() for metric in Parameters.metrics.values()}
        directory = "output/predictions"
        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]
        data = [(Memory.load(name, directory), name) for name in file_names if "x" not in name]

        columns = data[0][0].columns

        for d, n in data:
            for column in columns[1:]:
                new[n[:-4]] += list(d[column].values)

        frame = DataFrame.from_dict(new)
        corr = frame.corr("spearman")
        Memory.storeDataFrame(corr, "metrics_corr", "", True)

        return data

    @staticmethod
    def correlateTargets():
        directory = "input"
        path = (Memory.getPath() / directory)
        sc = StandardScaler()
        data = list()
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]

        for name in file_names:
            d = Memory.load(name, directory)
            df = DataFrame(data=sc.fit_transform(d), columns=d.columns)
            data.append(df)

        frame = pd.concat(data)

        lofo = [x for x in frame.columns if "LOFO" in x]
        shap = [x for x in frame.columns if "SHAP" in x]
        pimp = [x for x in frame.columns if "PIMP" in x]
        lime = [x for x in frame.columns if "LIME" in x]
        lm = [x for x in frame.columns if not x.startswith("target_")]

        matrix = frame.corr("spearman")
        matrix = matrix.drop([x for x in lofo + shap + pimp + lime + lm], axis=0)
        matrix = matrix.drop([x for x in list(frame.columns) if x not in lofo + shap + pimp + lime], axis=1)

        def f(targets): return np.round(np.mean([np.mean(list([val for val in list(map(abs, matrix[x].values)) if val < 1])) for x in targets]), 2)

        def f_2(targets): return np.round(np.max([np.mean(list([val for val in list(map(abs, matrix[x].values)) if val < 1])) for x in targets]), 2)

        d = {'lofo': [f(lofo), f_2(lofo)], 'shap': [f(shap), f_2(shap)], 'lime': [f(lime), f_2(lime)], 'pimp': [f(pimp), f_2(pimp)]}

        data_frame = pd.DataFrame(data=d, index=["mean", "max"], columns=["lofo", "shap", "lime", "pimp"])

        Memory.storeDataFrame(data_frame, "target_corr", "", True)

    @staticmethod
    def createHistograms():
        directory = "input"
        path = (Memory.getPath() / directory)
        data = list()
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]

        for name in file_names:
            d = Memory.load(name, directory)
            df = DataFrame(data=d, columns=d.columns)
            data.append(df)

        frame = pd.concat(data)

        meta_targets = [([x for x in frame.columns if "LOFO" in x], "LOFO", 0, 0),
                        ([x for x in frame.columns if "SHAP" in x], "SHAP", 0, 1),
                        ([x for x in frame.columns if "LIME" in x], "LIME", 1, 0),
                        ([x for x in frame.columns if "PIMP" in x], "PIMP", 1, 1)]

        fig, axs = plt.subplots(2, 2)
        for target, name, x, y in meta_targets:
            values = list()
            for value in [list(frame[column].values) for column in target]:
                values += value

            n, _, _ = axs[x, y].hist(x=values, rwidth=1, bins=len(values))
            axs[x, y].set_title(name)
            axs[x, y].set_xlim(np.quantile(values, 0.05), np.quantile(values, 0.75))

        Memory.storeVisual(plt, "Histograms", "")

    @staticmethod
    def cleanUp():
        directory = "output/predictions"
        path = (Memory.getPath() / directory)

        file_names = [name for name in os.listdir(path) if name.endswith('.csv') and "x" in name]

        data = list()
        for name in file_names:
            file = Memory.load(name, directory)
            data.append((file, name))

        for data_frame, name in data:
            for i in range(len(data_frame.index)):
                for j in range(1, len(data_frame.columns)):
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

            data_frame = data_frame.set_index("Unnamed: 0")
            Memory.storeDataFrame(data_frame, name[:-4], "predictions", True)
