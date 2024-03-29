import time
from typing import Dict, List, Tuple

import pandas as pd

from metalfi.src.metadata.metafeatures import MetaFeatures


class Dataset:

    def __init__(self, data_frame: pd.DataFrame, target: str):
        self.__data_frame = data_frame
        self.__target = target

    def get_data_frame(self) -> pd.DataFrame:
        return self.__data_frame

    def get_target(self) -> str:
        return self.__target

    def train_meta_data(self) -> Tuple[pd.DataFrame, List[str], Tuple[Dict[str, float], Dict[str, float]], int, int]:
        mf = MetaFeatures(self)

        start_d_total = time.time()
        d_time, u_time, mf_time, mt_time, l_time = mf.calculate_meta_features()
        end_d_total = time.time()
        d_total = end_d_total - start_d_total

        start_t_total = time.time()
        targets, d, p, l, s = mf.create_target()
        end_t_total = time.time()
        t_total = end_t_total - start_t_total

        data = mf.get_meta_data()

        data_time = {"data": d_time, "univariate": u_time, "multivariate_ff": mf_time,
                     "multivariate_ft": mt_time, "landmarking": l_time, "total": d_total}
        target_time = {"LOFO": d, "PIMP": p, "LIME": l, "SHAP": s, "total": t_total}

        return (data, targets, (data_time, target_time), len(self.__data_frame.columns) - 1,
                len(self.__data_frame.index))

    def test_meta_data(self) -> pd.DataFrame:
        mf = MetaFeatures(self)
        mf.calculate_meta_features()

        return mf.get_meta_data()
