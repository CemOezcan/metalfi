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
        Memory.storeVisual(plt, name)

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
        Memory.storeVisual(plt, name)

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

            Memory.storeVisual(plt, name[:-4])

    @staticmethod
    def metaFeatureImportance():
        directory = "output/importance"
        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]
        data = [(Memory.load(name, directory), name) for name in file_names]

        for frame, name in data:
            frame = frame.sort_values(by="mean absolute SHAP")
            plt.barh(list(frame["meta-features"])[-15:], list(frame["mean absolute SHAP"])[-15:])
            plt.yticks(list(frame["meta-features"])[-15:])
            Memory.storeVisual(plt, name[:-4])

    @staticmethod
    def compareMeans(folder):
        directory = "output/questions/" + folder
        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]
        data = [(Memory.load(name, directory).set_index("Unnamed: 0"), name) for name in file_names]

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

            if len(names) < 10:
                val, p_value = ss.friedmanchisquare(*d)
                if p_value < 0.05:
                    #Visualization.createCriticalDifferencesPlot(names, ranks)
                    Visualization.createTimeline(names, ranks, metric,
                                                 sp.sign_array(sp.posthoc_nemenyi_friedman(np.array(d).T)), d)

    @staticmethod
    def createTimeline(names, ranks, metric, sign_matrix, data):
        """fig, ax = plt.subplots(2)

        levels = np.tile([-6, 6, -4, 4, -2, 2], len(ranks))[:len(ranks)]
        marker, _, _ = ax[0].stem(ranks, levels, linefmt="C3--", basefmt="k-", use_line_collection=True)
        marker.set_ydata(np.zeros(len(ranks)))

        plt.setp(marker, mec="k", mfc="k")
        vert = np.array(list(map(lambda x: "top" if x > 0 else "bottom", levels)))

        for i in range(len(ranks)):
            ax[0].annotate(names[i], (ranks[i], levels[i]), va=vert[i], xytext=(3, 3), textcoords="offset points")

        ax[0].get_yaxis().set_visible(False)
        ax[0].spines["left"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)

        d = {name: [] for name in names}
        remove = list()
        for i in range(len(sign_matrix) - 1):
            for j in range(i + 1, len(sign_matrix[0])):
                if (sign_matrix[i][j] == 0) and (j not in remove):
                    d[names[i]].append(j)
                    remove.append(j)

        colors = ["forestgreen", "royalblue", "gold"]
        c = 0
        for i in range(len(d.keys())):
            indices = d[names[i]]
            indices.append(i)

            if len(indices) > 1:
                values = [ranks[index] for index in indices]
                ax[0].axvspan(max(values), min(values), facecolor=colors[c % len(colors)], alpha=0.2)
                c += 1

        ax[1].boxplot(data, notch=True, showfliers=False)
        ax[1].set_xticks(list(range(1, len(data) + 1)))
        ax[1].set_xticklabels(names)"""

        cd = Orange.evaluation.compute_CD(ranks, 28)
        Orange.evaluation.graph_ranks(ranks, names, cd=cd)
        Memory.storeVisual(plt, metric[:-4] + "_cd")
        plt.close()

        fig, ax = plt.subplots()
        ax.boxplot(data, notch=True, showfliers=False)
        ax.set_xticks(list(range(1, len(data) + 1)))
        ax.set_xticklabels(names)
        Memory.storeVisual(plt, metric[:-4] + "_means")
        plt.close()

    @staticmethod
    def correlateMetrics():
        new = {"r2": list(), "r": list(), "rmse": list()}
        directory = "output/predictions"
        path = (Memory.getPath() / directory)
        file_names = [name for name in os.listdir(path) if not name.endswith(".gitignore")]
        data = [(Memory.load(name, directory), name) for name in file_names]

        columns = data[0][0].columns

        for d, n in data:
            for column in columns[1:]:
                new[n[:-9]] += list(d[column].values)

        frame = DataFrame.from_dict(new)
        corr = frame.corr("spearman")
        path = Memory.getPath() / ("visual/metrics_corr.csv")
        corr.to_csv(path, header=True)

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

        path = Memory.getPath() / ("visual/target_corr.csv")
        data_frame.to_csv(path, header=True)

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
            axs[x, y].set_xlim(np.quantile(values, 0.10), np.quantile(values, 0.75))

            if name == "LIME":
                axs[x, y].set_ylim(0, 30)

        Memory.storeVisual(plt, "Histograms")

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
