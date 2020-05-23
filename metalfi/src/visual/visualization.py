import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scikit_posthocs as sp
from pandas import DataFrame
import scipy.stats as ss
from pandas.plotting import table
from metalfi.src.data.memory import Memory


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
    def runtime_graph():
        target_data = Visualization.fetch_runtime_data("XtargetX", 20000)
        meta_data = Visualization.fetch_runtime_data("XmetaX", 20000)
        for x in target_data:
            if x == "total":
                continue
            target_data[x][x] /= 5
            plt.plot(target_data[x].columns[0], x, data=target_data[x], linewidth=2)

        for x in meta_data:
            if x == "univariate" or x == "landmarking" or x == "multivariate" or x == "data":
                continue
            plt.plot(meta_data[x].columns[0], x, data=meta_data[x], linewidth=2)

        plt.legend()
        plt.show()

    @staticmethod
    def runtime_boxplot():
        target_data = Visualization.fetch_runtime_data("XtargetX")
        meta_data = Visualization.fetch_runtime_data("XmetaX")

        data = list()
        names = list()
        for x in target_data:
            if x == "total" or x == "SHAP" or x == "LIME":
                continue
            names.append(x)
            target_data[x][x] /= 5
            data.append(target_data[x][x].values)

        for x in meta_data:
            if x == "total" or x == "multivariate":
                continue
            names.append(x)
            data.append(meta_data[x][x].values)

        fig, ax = plt.subplots()
        ax.boxplot(data)
        plt.xticks(list(range(1, len(data) + 1)), names)

        plt.show()

    @staticmethod
    def fetch_predictions():
        directory = "output/predictions"

        path = (Memory.getPath() / directory)

        for name in os.listdir(path):
            frame = Memory.load(name, directory).set_index("Unnamed: 0")
            for column in frame.columns:
                frame = frame.round({column: 3})

            path = Memory.getPath() / (directory + "/" + name)
            frame.to_csv(path, header=True)

        data = [(Memory.load(name, directory).set_index("Unnamed: 0"), name) for name in os.listdir(path)]

        return data

    @staticmethod
    def performance():
        directory = "output/selection"
        path = (Memory.getPath() / directory)

        data = [(Memory.load(name, directory).set_index("Unnamed: 0"), name) for name in os.listdir(path)]
        for frame, name in data:
            width = 0.2
            fig, ax = plt.subplots()

            anova = frame.loc["ANOVA"].values
            mi = frame.loc["MI"].values
            fi = frame.loc["FI"].values
            meta = frame.loc["MetaLFI"].values

            x = np.arange(len(anova))

            pos_anova = ax.bar(x - 1.5*width, anova, width, label="ANOVA")
            pos_mi = ax.bar(x - width/2, mi, width, label="MI")
            pos_fi = ax.bar(x + width/2, fi, width, label="FI")
            pos_meta = ax.bar(x + 1.5*width, meta, width, label="MetaLFI")

            """plt.bar(pos_anova, anova, label="ANOVA")
            plt.bar(pos_mi, mi, label="MI")"""
            """plt.bar(pos_fi, fi, label="FI")
            plt.bar(pos_meta, meta, label="MetaLFI")"""

            ax.set_ylabel("Acc. Scores")
            ax.set_yticks([0.775, 0.8, 0.825, 0.85])
            ax.set_title(name[:-4])
            ax.set_xticks(x)
            ax.set_xticklabels(list(frame.columns))
            ax.legend()
            plt.ylim([0.75, 0.85])
            plt.show()

    @staticmethod
    def metaFeatureImportance():
        directory = "output/importance"
        path = (Memory.getPath() / directory)
        data = [(Memory.load(name, directory).set_index("Unnamed: 0"), name) for name in os.listdir(path)]

        for frame, name in data:
            frame = frame.sort_values(by="mean absolute SHAP")
            plt.barh(list(frame["meta-features"])[:15], list(frame["mean absolute SHAP"])[:15])
            plt.yticks(list(frame["meta-features"])[:15], list(frame["meta-features"])[:15])
            plt.title(name[:-4])
            plt.show()

    @staticmethod
    def compareMeans(folder):
        directory = "output/questions/" + folder
        path = (Memory.getPath() / directory)

        data = [(Memory.load(name, directory).set_index("Unnamed: 0"), name) for name in os.listdir(path)]

        for data_frame, metric in data:
            d = list()
            names = list()
            ranks = [0] * len(data_frame.index)

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
                print("mean " + column + ": ", np.mean(data_frame[column].values))
                d.append(data_frame[column].values)

            val, p_value = ss.friedmanchisquare(*d)
            print(p_value)
            if p_value < 0.05:
                print(sp.sign_array(sp.posthoc_nemenyi_friedman(np.array(d).T)))

