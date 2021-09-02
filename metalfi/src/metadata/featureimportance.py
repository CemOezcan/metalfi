import re
import sys
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from io import StringIO

from metalfi.src.parameters import Parameters


class FeatureImportance(ABC):
    """
    Superclass for computing different feature importance estimates.

    Attributes
    ----------
        _data_frame : (DataFrame)
            The underlying base-data set, whose feature importance values are supposed to be estimated.
        _target : (str)
            Name of the target variable of the base-data set.
        _models : (Dict[str, Dict[str, sklearn estimator]])
            Maps meta-model types and names to meta-models.
        _all_models : (List[sklearn estimator])
            Meta-models.
        _feature_importances : (List[DataFrame])
            Contains feature importances.
        _name : (str)
            Name of the feature importance measure.
        __model_names : (List[str])
            Names of all meta-models.
    """

    def __init__(self, dataset: 'Dataset'):
        if dataset is None:
            return

        self._data_frame = dataset.get_data_frame()
        self._target = dataset.get_target()
        self._models = dict()

        for model, model_name, model_type in Parameters.base_models:
            try:
                self._models[model_type][model_name] = model
            except KeyError:
                self._models[model_type] = dict()
                self._models[model_type][model_name] = model

        self.__model_names = sum([list(d.keys()) for d in self._models.values()], [])
        self._all_models = sum([list(d.values()) for d in self._models.values()], [])

        self._feature_importances = []
        self._name = ""

    def get_model_names(self):
        return self.__model_names

    def get_feature_importances(self):
        return self._feature_importances

    def get_name(self):
        return self._name

    @contextmanager
    def ignore_np_deprecation_warning(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="tostring.*")
        try:
            yield
        finally:
            warnings.filterwarnings("default")

    @contextmanager
    def ignore_progress_bars(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=".*(is a deprecated alias for the builtin|tostring).*")
        temp = StringIO()
        sys.stderr = temp
        try:
            yield
        finally:
            warnings.filterwarnings("default")
            sys.stderr = sys.__stderr__
            pattern = re.compile(r".*\d+%\|.*\|\s\d+/\d+\s\[.*<.*,.*(it/s|s/it)].*")
            for warning in {x for x in temp.getvalue().splitlines() if not re.match(pattern, x) and x != ""}:
                warnings.warn(warning)

    @abstractmethod
    def calculate_scores(self) -> None:
        """
        Compute feature importance estimates and store them, so they can be retrieved later.
        """
        raise NotImplementedError('Abstract method.')
