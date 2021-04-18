import time

from metalfi.src.metadata.metafeatures import MetaFeatures


class Dataset:

    def __init__(self, data_frame, target):
        self.__data_frame = data_frame
        self.__target = target

    def getDataFrame(self):
        return self.__data_frame

    def getTarget(self):
        return self.__target

    def trainMetaData(self):
        mf = MetaFeatures(self)

        start_d_total = time.time()
        d_time, u_time, m_time, l_time = mf.calculateMetaFeatures()
        end_d_total = time.time()
        d_total = end_d_total - start_d_total

        start_t_total = time.time()
        targets, d, p, l, s = mf.createTarget()
        end_t_total = time.time()
        t_total = end_t_total - start_t_total

        data = mf.getMetaData()

        data_time = {"metadata": d_time, "univariate": u_time, "multivariate": m_time, "landmarking": l_time,
                     "total": d_total}
        target_time = {"LOFO": d, "PIMP": p, "LIME": l, "SHAP": s, "total": t_total}

        return data, targets, (data_time, target_time), len(self.__data_frame.columns) - 1, len(self.__data_frame.index)

    def testMetaData(self):
        mf = MetaFeatures(self)
        mf.calculateMetaFeatures()

        return mf.getMetaData()
