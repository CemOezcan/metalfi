import os
from typing import Dict, List, Tuple

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
    def performance(data: List[Tuple[pd.DataFrame, str]]) -> None:
        """
        Create and save bar charts. Visualizes the performances of different feature selection approaches.
        """
        for frame, file_name in data:
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

            plt.savefig(Parameters.output_dir + "feature_selection_performance/" + file_name + ".pdf")
            plt.close()

    @staticmethod
    def meta_feature_importance() -> None:
        """
        Create and save bar charts. Visualizes the importance of meta-features for different meta-targets.
        """
        directory = Parameters.output_dir + "meta_feature_importance/"
        data = [(pd.read_csv(directory + file_name), file_name)
                for file_name in os.listdir(directory) if ".csv" in file_name]
        frame = data[0][0].sort_values(by="PIMP")
        for base_model in set(frame["base_model"]):
            for imp in set(frame["importance_measure"]):
                if "SHAP" in imp:
                    new_frame = frame[frame["base_model"] == base_model]
                    new_frame = new_frame[new_frame["importance_measure"] == imp]
                    plt.barh(list(new_frame["meta-features"])[-15:], list(new_frame["PIMP"])[-15:])
                    plt.yticks(list(new_frame["meta-features"])[-15:])
                    plt.savefig(Parameters.output_dir + f"meta_feature_importance/metaFeatureImp x {base_model}x{imp}.pdf")
                    plt.close()

    @staticmethod
    def compare_means(data: List[Tuple[pd.DataFrame, str]], folder: str) -> None:
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
            d = []
            names = []
            ranks = np.zeros(len(data_frame.columns))

            for i in range(len(data_frame.index)):
                values = np.array(data_frame.iloc[i].values)
                if "RMSE" not in metric:
                    values = -values
                temp = values.argsort()
                current_ranks = np.zeros(len(values))
                current_ranks[temp] = np.arange(len(values))
                current_ranks += 1
                ranks += current_ranks

            ranks /= len(data_frame.index)

            for column in data_frame.columns:
                names.append(column)
                d.append(data_frame[column].values)

            if 24 >= len(names) >= 3:
                _, p_value = ss.friedmanchisquare(*d)
                if p_value < 0.05:
                    Visualization.__create_cd_diagram(names, ranks.tolist(), metric, d, folder)

    @staticmethod
    def __create_cd_diagram(names: List[str], ranks: List[float], file_name: str,
                            data: List[np.array], folder: str) -> None:
        cd = Orange.evaluation.compute_CD(ranks, 28) if len(ranks) < 21 else 3.616
        Orange.evaluation.graph_ranks(ranks, names, cd=cd)
        plt.savefig(Parameters.output_dir + folder + file_name + "_cd.pdf")
        plt.close()

        _, ax = plt.subplots()
        ax.boxplot(data, notch=True, showfliers=False)
        ax.set_xticks(list(range(1, len(data) + 1)))
        ax.set_xticklabels(names)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.savefig(Parameters.output_dir + folder + file_name + "_means.pdf")
        plt.close()

    @staticmethod
    def correlate_targets() -> None:
        """
        Compute pairwise Spearman correlation coefficients between meta-features and meta-targets,
        grouped by feature importance measure.

        Returns
        -------
            Mean and maximum values over all correlation coefficients for each group.

        """
        sc = StandardScaler()
        data = []

        for file_name in [x for x in os.listdir(Parameters.meta_dataset_dir) if x.endswith('.csv')]:
            d = pd.read_csv(Parameters.meta_dataset_dir + file_name)
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
            return np.round(np.mean([np.mean([abs(x) for x in matrix[target].values if abs(x) < 1])
                                     for target in targets]), 2)

        def __f_2(targets):
            return np.round(np.max([np.mean([abs(x) for x in matrix[target].values if abs(x) < 1])
                                    for target in targets]), 2)

        d = {'lofo': [__f(lofo), __f_2(lofo)], 'shap': [__f(shap), __f_2(shap)],
             'lime': [__f(lime), __f_2(lime)], 'pimp': [__f(pimp), __f_2(pimp)]}
        data_frame = pd.DataFrame(data=d, index=["mean", "max"], columns=["lofo", "shap", "lime", "pimp"])
        Memory.store_data_frame(data_frame, "target_corr", "meta_prediction_performance", True)

    @staticmethod
    def create_histograms() -> None:
        """
        Group meta-targets by feature importance measure and visualize their distributions as histograms.
        """
        data = []

        for file_name in [x for x in os.listdir(Parameters.meta_dataset_dir) if x.endswith('.csv')]:
            d = pd.read_csv(Parameters.meta_dataset_dir + file_name)
            df = pd.DataFrame(data=d, columns=d.columns)
            data.append(df)

        frame = pd.concat(data)

        meta_targets = [([x for x in frame.columns if "LOFO" in x], "LOFO", 0, 0),
                        ([x for x in frame.columns if "SHAP" in x], "SHAP", 0, 1),
                        ([x for x in frame.columns if "LIME" in x], "LIME", 1, 0),
                        ([x for x in frame.columns if "PIMP" in x], "PIMP", 1, 1)]

        _, axs = plt.subplots(2, 2)
        for target, name, x, y in meta_targets:
            values = []
            for value in [list(frame[column].values) for column in target]:
                values += value

            axs[x, y].hist(x=values, rwidth=1, bins=len(values))
            axs[x, y].set_title(name)
            axs[x, y].set_xlim(np.quantile(values, 0.05), np.quantile(values, 0.75))

        plt.savefig(Parameters.output_dir + "meta_prediction_performance/Histograms.pdf")
        plt.close()
