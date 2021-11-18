import abc
import argparse
import copy
import json
import logging
import pathlib

import math
import os.path
import pprint
import re
import requests
import sys
import traceback
import typing
from urllib import parse as url_parse

from . import base
from d3m import deprecate, exceptions, utils

__all__ = ('TaskKeyword', 'PerformanceMetric', 'Problem')

logger = logging.getLogger(__name__)

# Comma because we unpack the list of validators returned from "load_schema_validators".
PROBLEM_SCHEMA_VALIDATOR, = utils.load_schema_validators(base.SCHEMAS, ('problem.json',))

PROBLEM_SCHEMA_VERSION = 'https://metadata.datadrivendiscovery.org/schemas/v0/problem.json'


def sigmoid(x: float) -> float:
    """
    Numerically stable scaled logistic function.

    Maps all values ``x`` to [0, 1]. Values between -1000 and 1000 are
    mapped reasonably far from 0 and 1, after which the function
    converges to bounds.

    Parameters
    ----------
    x:
        Input.

    Returns
    -------
    Output.
    """

    scale = 1 / 1000

    if x >= 0:
        ex = math.exp(scale * -x)
        return 1 / (1 + ex)
    else:
        ex = math.exp(scale * x)
        return ex / (1 + ex)


class TaskKeywordBase:
    _d3m_map: typing.Dict[str, 'TaskKeywordBase'] = {}

    @classmethod
    def get_map(cls) -> dict:
        """
        Returns the map between D3M problem description JSON string and enum values.

        Returns
        -------
        The map.
        """

        return cls._d3m_map

    @classmethod
    def parse(cls, name: str) -> 'TaskKeywordBase':
        """
        Converts D3M problem description JSON string into enum value.

        Parameters
        ----------
        name:
            D3M problem description JSON string.

        Returns
        -------
        Enum value.
        """

        return cls.get_map()[name]

    def unparse(self) -> str:
        """
        Converts enum value to D3M problem description JSON string.

        Returns
        -------
        D3M problem description JSON string.
        """

        for key, value in self.get_map().items():
            if self == value:
                return key

        raise exceptions.InvalidStateError("Cannot convert {self}.".format(self=self))

    def __ge__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) >= list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented

    def __gt__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) > list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented

    def __le__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) <= list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented

    def __lt__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) < list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented


TaskKeyword = utils.create_enum_from_json_schema_enum(
    'TaskKeyword', base.DEFINITIONS_JSON,
    'definitions.problem.properties.task_keywords.items.oneOf[*].enum[*]',
    module=__name__, base_class=TaskKeywordBase,
)

TaskKeyword._d3m_map.update({
    'classification': TaskKeyword.CLASSIFICATION,
    'regression': TaskKeyword.REGRESSION,
    'clustering': TaskKeyword.CLUSTERING,
    'linkPrediction': TaskKeyword.LINK_PREDICTION,
    'vertexNomination': TaskKeyword.VERTEX_NOMINATION,
    'vertexClassification': TaskKeyword.VERTEX_CLASSIFICATION,
    'communityDetection': TaskKeyword.COMMUNITY_DETECTION,
    'graphMatching': TaskKeyword.GRAPH_MATCHING,
    'forecasting': TaskKeyword.FORECASTING,
    'collaborativeFiltering': TaskKeyword.COLLABORATIVE_FILTERING,
    'objectDetection': TaskKeyword.OBJECT_DETECTION,
    'semiSupervised': TaskKeyword.SEMISUPERVISED,
    'binary': TaskKeyword.BINARY,
    'multiClass': TaskKeyword.MULTICLASS,
    'multiLabel': TaskKeyword.MULTILABEL,
    'univariate': TaskKeyword.UNIVARIATE,
    'multivariate': TaskKeyword.MULTIVARIATE,
    'overlapping': TaskKeyword.OVERLAPPING,
    'nonOverlapping': TaskKeyword.NONOVERLAPPING,
    'tabular': TaskKeyword.TABULAR,
    'relational': TaskKeyword.RELATIONAL,
    'nested': TaskKeyword.NESTED,
    'image': TaskKeyword.IMAGE,
    'audio': TaskKeyword.AUDIO,
    'video': TaskKeyword.VIDEO,
    'speech': TaskKeyword.SPEECH,
    'text': TaskKeyword.TEXT,
    'graph': TaskKeyword.GRAPH,
    'multiGraph': TaskKeyword.MULTIGRAPH,
    'timeSeries': TaskKeyword.TIME_SERIES,
    'grouped': TaskKeyword.GROUPED,
    'geospatial': TaskKeyword.GEOSPATIAL,
    'remoteSensing': TaskKeyword.REMOTE_SENSING,
    'lupi': TaskKeyword.LUPI,
    'missingMetadata': TaskKeyword.MISSING_METADATA,
    'multipleInstanceLearning': TaskKeyword.MULTIPLE_INSTANCE_LEARNING,
})


class PerformanceMetricBase:
    _d3m_map: typing.ClassVar[typing.Dict[str, 'PerformanceMetricBase']] = {}
    _requires_score_set: typing.ClassVar[typing.Set['PerformanceMetricBase']] = set()
    _requires_rank_set: typing.ClassVar[typing.Set['PerformanceMetricBase']] = set()
    _best_value_map: typing.ClassVar[typing.Dict['PerformanceMetricBase', float]] = {}
    _worst_value_map: typing.ClassVar[typing.Dict['PerformanceMetricBase', float]] = {}
    _additional_score_class_map: typing.ClassVar[typing.Dict['PerformanceMetricBase', type]] = {}

    @classmethod
    def get_map(cls) -> dict:
        """
        Returns the map between D3M problem description JSON string and enum values.

        Returns
        -------
        The map.
        """

        return cls._d3m_map

    @classmethod
    def parse(cls, name: str) -> 'PerformanceMetricBase':
        """
        Converts D3M problem description JSON string into enum value.

        Parameters
        ----------
        name:
            D3M problem description JSON string.

        Returns
        -------
        Enum value.
        """

        return cls.get_map()[name]

    def unparse(self) -> str:
        """
        Converts enum value to D3M problem description JSON string.

        Returns
        -------
        D3M problem description JSON string.
        """

        for key, value in self.get_map().items():
            if self == value:
                return key

        raise exceptions.InvalidStateError("Cannot convert {self}.".format(self=self))

    def requires_score(self) -> bool:
        """
        Returns ``True`` if this metric requires score column.

        Returns
        -------
        ``True`` if this metric requires score column.
        """

        return self in self._requires_score_set

    def requires_rank(self) -> bool:
        """
        Returns ``True`` if this metric requires rank column.

        Returns
        -------
        ``True`` if this metric requires rank column.
        """

        return self in self._requires_rank_set

    def best_value(self) -> float:
        """
        The best possible value of the metric.

        Returns
        -------
        The best possible value of the metric.
        """

        return self._best_value_map[self]

    def worst_value(self) -> float:
        """
        The worst possible value of the metric.

        Returns
        -------
        The worst possible value of the metric.
        """

        return self._worst_value_map[self]

    def normalize(self, value: float) -> float:
        """
        Normalize the ``value`` for this metric so that it is between 0 and 1,
        inclusive, where 1 is the best score and 0 is the worst.

        Parameters
        ----------
        value:
            Value of this metric to normalize.

        Returns
        -------
        A normalized metric.
        """

        worst_value = self.worst_value()
        best_value = self.best_value()

        return self._normalize(worst_value, best_value, value)

    @classmethod
    def _normalize(cls, worst_value: float, best_value: float, value: float) -> float:
        assert worst_value <= value <= best_value or worst_value >= value >= best_value, (worst_value, value, best_value)

        if math.isinf(best_value) and math.isinf(worst_value):
            value = sigmoid(value)
            if best_value > worst_value:  # "best_value" == inf, "worst_value" == -inf
                best_value = 1.0
                worst_value = 0.0
            else:  # "best_value" == -inf, "worst_value" == inf
                best_value = 0.0
                worst_value = 1.0
        elif math.isinf(best_value):
            value = sigmoid(value - worst_value)
            if best_value > worst_value:  # "best_value" == inf
                best_value = 1.0
                worst_value = 0.5
            else:  # "best_value" == -inf
                best_value = 0.0
                worst_value = 0.5
        elif math.isinf(worst_value):
            value = sigmoid(best_value - value)
            if best_value > worst_value:  # "worst_value" == -inf
                best_value = 0.5
                worst_value = 1.0
            else:  # "worst_value" == inf
                best_value = 0.5
                worst_value = 0.0

        return (value - worst_value) / (best_value - worst_value)

    def get_class(self) -> typing.Any:
        """
        Returns a class suitable for computing this metric.
        """

        # Importing here to prevent import cycle.
        from d3m import metrics

        if self in metrics.class_map:
            return metrics.class_map[self]

        if self in self._additional_score_class_map:
            return self._additional_score_class_map[self]

        raise exceptions.NotSupportedError("Computing metric {metric} is not supported.".format(metric=self))

    @classmethod
    def register_metric(cls, name: str, *, best_value: float, worst_value: float, score_class: type, requires_score: bool = False, requires_rank: bool = False) -> None:
        cls.register_value(name, name)  # type: ignore
        cls._best_value_map[cls[name]] = best_value  # type: ignore
        cls._worst_value_map[cls[name]] = worst_value  # type: ignore
        cls._additional_score_class_map[cls[name]] = score_class  # type: ignore

        if requires_score:
            PerformanceMetric._requires_score_set.add(cls[name])  # type: ignore

        if requires_rank:
            PerformanceMetric._requires_rank_set.add(cls[name])  # type: ignore


PerformanceMetric = utils.create_enum_from_json_schema_enum(
    'PerformanceMetric', base.DEFINITIONS_JSON,
    'definitions.performance_metric.oneOf[*].properties.metric.enum[*]',
    module=__name__, base_class=PerformanceMetricBase,
)

PerformanceMetric._d3m_map.update({
    'accuracy': PerformanceMetric.ACCURACY,
    'precision': PerformanceMetric.PRECISION,
    'recall': PerformanceMetric.RECALL,
    'f1': PerformanceMetric.F1,
    'f1Micro': PerformanceMetric.F1_MICRO,
    'f1Macro': PerformanceMetric.F1_MACRO,
    'rocAuc': PerformanceMetric.ROC_AUC,
    'rocAucMicro': PerformanceMetric.ROC_AUC_MICRO,
    'rocAucMacro': PerformanceMetric.ROC_AUC_MACRO,
    'meanSquaredError': PerformanceMetric.MEAN_SQUARED_ERROR,
    'rootMeanSquaredError': PerformanceMetric.ROOT_MEAN_SQUARED_ERROR,
    'meanAbsoluteError': PerformanceMetric.MEAN_ABSOLUTE_ERROR,
    'rSquared': PerformanceMetric.R_SQUARED,
    'normalizedMutualInformation': PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION,
    'jaccardSimilarityScore': PerformanceMetric.JACCARD_SIMILARITY_SCORE,
    'precisionAtTopK': PerformanceMetric.PRECISION_AT_TOP_K,
    'objectDetectionAP': PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION,
    'hammingLoss': PerformanceMetric.HAMMING_LOSS,
    'meanReciprocalRank': PerformanceMetric.MEAN_RECIPROCAL_RANK,
    'hitsAtK': PerformanceMetric.HITS_AT_K,
})
PerformanceMetric._requires_score_set.update({
    PerformanceMetric.ROC_AUC,
    PerformanceMetric.ROC_AUC_MICRO,
    PerformanceMetric.ROC_AUC_MACRO,
    PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION,
})
PerformanceMetric._requires_rank_set.update({
    PerformanceMetric.MEAN_RECIPROCAL_RANK,
    PerformanceMetric.HITS_AT_K,
})
PerformanceMetric._best_value_map.update({
    PerformanceMetric.ACCURACY: 1.0,
    PerformanceMetric.PRECISION: 1.0,
    PerformanceMetric.RECALL: 1.0,
    PerformanceMetric.F1: 1.0,
    PerformanceMetric.F1_MICRO: 1.0,
    PerformanceMetric.F1_MACRO: 1.0,
    PerformanceMetric.ROC_AUC: 1.0,
    PerformanceMetric.ROC_AUC_MICRO: 1.0,
    PerformanceMetric.ROC_AUC_MACRO: 1.0,
    PerformanceMetric.MEAN_SQUARED_ERROR: 0.0,
    PerformanceMetric.ROOT_MEAN_SQUARED_ERROR: 0.0,
    PerformanceMetric.MEAN_ABSOLUTE_ERROR: 0.0,
    PerformanceMetric.R_SQUARED: 1.0,
    PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION: 1.0,
    PerformanceMetric.JACCARD_SIMILARITY_SCORE: 1.0,
    PerformanceMetric.PRECISION_AT_TOP_K: 1.0,
    PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION: 1.0,
    PerformanceMetric.HAMMING_LOSS: 0.0,
    PerformanceMetric.MEAN_RECIPROCAL_RANK: 1.0,
    PerformanceMetric.HITS_AT_K: 1.0,
})
PerformanceMetric._worst_value_map.update({
    PerformanceMetric.ACCURACY: 0.0,
    PerformanceMetric.PRECISION: 0.0,
    PerformanceMetric.RECALL: 0.0,
    PerformanceMetric.F1: 0.0,
    PerformanceMetric.F1_MICRO: 0.0,
    PerformanceMetric.F1_MACRO: 0.0,
    PerformanceMetric.ROC_AUC: 0.0,
    PerformanceMetric.ROC_AUC_MICRO: 0.0,
    PerformanceMetric.ROC_AUC_MACRO: 0.0,
    PerformanceMetric.MEAN_SQUARED_ERROR: float('inf'),
    PerformanceMetric.ROOT_MEAN_SQUARED_ERROR: float('inf'),
    PerformanceMetric.MEAN_ABSOLUTE_ERROR: float('inf'),
    PerformanceMetric.R_SQUARED: float('-inf'),
    PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION: 0.0,
    PerformanceMetric.JACCARD_SIMILARITY_SCORE: 0.0,
    PerformanceMetric.PRECISION_AT_TOP_K: 0.0,
    PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION: 0.0,
    PerformanceMetric.HAMMING_LOSS: 1.0,
    PerformanceMetric.MEAN_RECIPROCAL_RANK: 0.0,
    PerformanceMetric.HITS_AT_K: 0.0,
})

# Here are all legacy (before v4.0.0) task types and task subtypes mapped to task keywords.
TASK_TYPE_TO_KEYWORDS_MAP: typing.Dict[typing.Optional[str], typing.List] = {
    None: [],
    'classification': [TaskKeyword.CLASSIFICATION],
    'regression': [TaskKeyword.REGRESSION],
    'clustering': [TaskKeyword.CLUSTERING],
    'linkPrediction': [TaskKeyword.LINK_PREDICTION],
    'vertexClassification': [TaskKeyword.VERTEX_CLASSIFICATION],
    'vertexNomination': [TaskKeyword.VERTEX_NOMINATION],
    'communityDetection': [TaskKeyword.COMMUNITY_DETECTION],
    'graphMatching': [TaskKeyword.GRAPH_MATCHING],
    'timeSeriesForecasting': [TaskKeyword.TIME_SERIES, TaskKeyword.FORECASTING],
    'collaborativeFiltering': [TaskKeyword.COLLABORATIVE_FILTERING],
    'objectDetection': [TaskKeyword.OBJECT_DETECTION],
    'semiSupervisedClassification': [TaskKeyword.SEMISUPERVISED, TaskKeyword.CLASSIFICATION],
    'semiSupervisedRegression': [TaskKeyword.SEMISUPERVISED, TaskKeyword.REGRESSION],
    'binary': [TaskKeyword.BINARY],
    'multiClass': [TaskKeyword.MULTICLASS],
    'multiLabel': [TaskKeyword.MULTILABEL],
    'univariate': [TaskKeyword.UNIVARIATE],
    'multivariate': [TaskKeyword.MULTIVARIATE],
    'overlapping': [TaskKeyword.OVERLAPPING],
    'nonOverlapping': [TaskKeyword.NONOVERLAPPING],
}
JSON_TASK_TYPE_TO_KEYWORDS_MAP: typing.Dict[typing.Optional[str], typing.List] = {
    None: [],
    'CLASSIFICATION': [TaskKeyword.CLASSIFICATION],
    'REGRESSION': [TaskKeyword.REGRESSION],
    'CLUSTERING': [TaskKeyword.CLUSTERING],
    'LINK_PREDICTION': [TaskKeyword.LINK_PREDICTION],
    'VERTEX_CLASSIFICATION': [TaskKeyword.VERTEX_CLASSIFICATION],
    'VERTEX_NOMINATION': [TaskKeyword.VERTEX_NOMINATION],
    'COMMUNITY_DETECTION': [TaskKeyword.COMMUNITY_DETECTION],
    'GRAPH_MATCHING': [TaskKeyword.GRAPH_MATCHING],
    'TIME_SERIES_FORECASTING': [TaskKeyword.TIME_SERIES, TaskKeyword.FORECASTING],
    'COLLABORATIVE_FILTERING': [TaskKeyword.COLLABORATIVE_FILTERING],
    'OBJECT_DETECTION': [TaskKeyword.OBJECT_DETECTION],
    'SEMISUPERVISED_CLASSIFICATION': [TaskKeyword.SEMISUPERVISED, TaskKeyword.CLASSIFICATION],
    'SEMISUPERVISED_REGRESSION': [TaskKeyword.SEMISUPERVISED, TaskKeyword.REGRESSION],
    'BINARY': [TaskKeyword.BINARY],
    'MULTICLASS': [TaskKeyword.MULTICLASS],
    'MULTILABEL': [TaskKeyword.MULTILABEL],
    'UNIVARIATE': [TaskKeyword.UNIVARIATE],
    'MULTIVARIATE': [TaskKeyword.MULTIVARIATE],
    'OVERLAPPING': [TaskKeyword.OVERLAPPING],
    'NONOVERLAPPING': [TaskKeyword.NONOVERLAPPING],
}


class Loader(metaclass=utils.AbstractMetaclass):
    """
    A base class for problem loaders.
    """

    @abc.abstractmethod
    def can_load(self, problem_uri: str) -> bool:
        """
        Return ``True`` if this loader can load a problem from a given URI ``problem_uri``.

        Parameters
        ----------
        problem_uri:
            A URI to load a problem from.

        Returns
        -------
        ``True`` if this loader can load a problem from ``problem_uri``.
        """

    @abc.abstractmethod
    def load(self, problem_uri: str, *, problem_id: str = None, problem_version: str = None,
             problem_name: str = None, strict_digest: bool = False, handle_score_split: bool = True) -> 'Problem':
        """
        Loads the problem at ``problem_uri``.

        Parameters
        ----------
        problem_uri:
            A URI to load.
        problem_id:
            Override problem ID determined by the loader.
        problem_version:
            Override problem version determined by the loader.
        problem_name:
            Override problem name determined by the loader.
        strict_digest:
            If computed digest does not match the one provided in metadata, raise an exception?
        handle_score_split:
            Rename a scoring problem to not have the same name as testing problem
            and update dataset references.

        Returns
        -------
        A loaded problem.
        """

    @classmethod
    def get_problem_class(cls) -> 'typing.Type[Problem]':
        return Problem


class Saver(metaclass=utils.AbstractMetaclass):
    """
    A base class for problem savers.
    """

    @abc.abstractmethod
    def can_save(self, problem_uri: str) -> bool:
        """
        Return ``True`` if this saver can save a problem to a given URI ``problem_uri``.

        Parameters
        ----------
        problem_uri:
            A URI to save a problem to.

        Returns
        -------
        ``True`` if this saver can save a problem to ``problem_uri``.
        """

    @abc.abstractmethod
    def save(self, problem: 'Problem', problem_uri: str) -> None:
        """
        Saves the dataset ``dataset`` to ``dataset_uri``.

        Parameters
        ----------
        problem:
            A problem to save.
        problem_uri:
            A URI to save to.
        """


class D3MProblemLoader(Loader):
    """
    A class for loading of D3M problems.

    Loader support only loading from a local file system.
    URI should point to the ``problemDoc.json`` file in the D3M problem directory.
    """

    SUPPORTED_VERSIONS = {'3.0', '3.1', '3.1.1', '3.1.2', '3.2.0', '3.2.1', '3.3.0', '3.3.1', '4.0.0', '4.1.0', '4.1.1'}

    def can_load(self, problem_uri: str) -> bool:
        try:
            path = utils.uri_to_path(problem_uri)
        except exceptions.InvalidArgumentValueError:
            return False

        if path.name != 'problemDoc.json':
            return False

        return True

    # "strict_digest" is not used because there is no digest in D3M problem descriptions.
    def load(self, problem_uri: str, *, problem_id: str = None, problem_version: str = None,
             problem_name: str = None, strict_digest: bool = False, handle_score_split: bool = True) -> 'Problem':
        assert self.can_load(problem_uri)

        problem_doc_path = pathlib.Path(utils.uri_to_path(problem_uri))

        try:
            with open(problem_doc_path, 'r', encoding='utf8') as problem_doc_file:
                problem_doc = json.load(problem_doc_file)
        except FileNotFoundError as error:
            raise exceptions.ProblemNotFoundError(
                "D3M problem '{problem_uri}' cannot be found.".format(problem_uri=problem_uri),
            ) from error

        problem_schema_version = problem_doc.get('about', {}).get('problemSchemaVersion', '3.3.0')
        if problem_schema_version not in self.SUPPORTED_VERSIONS:
            logger.warning("Loading a problem with unsupported schema version '%(version)s'. Supported versions: %(supported_versions)s", {
                'version': problem_schema_version,
                'supported_versions': self.SUPPORTED_VERSIONS,
            })

        # To be compatible with problem descriptions which do not adhere to the schema and have only one entry for data.
        if not isinstance(problem_doc['inputs']['data'], list):
            problem_doc['inputs']['data'] = [problem_doc['inputs']['data']]

        performance_metrics = []
        for performance_metric in problem_doc['inputs']['performanceMetrics']:
            params = {}

            if 'posLabel' in performance_metric:
                params['pos_label'] = performance_metric['posLabel']

            if 'K' in performance_metric:
                params['k'] = performance_metric['K']

            performance_metrics.append({
                'metric': PerformanceMetric.parse(performance_metric['metric']),
            })

            if params:
                performance_metrics[-1]['params'] = params

        inputs = []
        for data in problem_doc['inputs']['data']:
            targets = []
            for target in data['targets']:
                targets.append({
                    'target_index': target['targetIndex'],
                    'resource_id': target['resID'],
                    'column_index': target['colIndex'],
                    'column_name': target['colName'],
                })

                if 'numClusters' in target:
                    targets[-1]['clusters_number'] = target['numClusters']

            privileged_data_columns = []
            for privileged_data in data.get('privilegedData', []):
                privileged_data_columns.append({
                    'privileged_data_index': privileged_data['privilegedDataIndex'],
                    'resource_id': privileged_data['resID'],
                    'column_index': privileged_data['colIndex'],
                    'column_name': privileged_data['colName'],
                })

            problem_input = {
                'dataset_id': data['datasetID'],
            }

            if targets:
                problem_input['targets'] = targets

            if privileged_data_columns:
                problem_input['privileged_data'] = privileged_data_columns

            if data.get('forecastingHorizon', {}).get('horizonValue', None):
                problem_input['forecasting_horizon'] = {
                    'resource_id': data['forecastingHorizon']['resID'],
                    'column_index': data['forecastingHorizon']['colIndex'],
                    'column_name': data['forecastingHorizon']['colName'],
                    'horizon_value': data['forecastingHorizon']['horizonValue'],
                }

            inputs.append(problem_input)

        document_problem_id = problem_doc['about']['problemID']
        # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
        # They are the same as TEST dataset splits, but we present them differently, so that
        # SCORE dataset splits have targets as part of data. Because of this we also update
        # corresponding problem ID.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
        if handle_score_split and (problem_doc_path.parent / '..' / 'targets.csv').exists() and document_problem_id.endswith('_TEST'):
            document_problem_id = document_problem_id[:-5] + '_SCORE'

            # Also update dataset references.
            for data in problem_doc.get('inputs', {}).get('data', []):
                if data['datasetID'].endswith('_TEST'):
                    data['datasetID'] = data['datasetID'][:-5] + '_SCORE'

        # "dataSplits" is not exposed as a problem description. One should provide splitting
        # configuration to a splitting pipeline instead. Similarly, "outputs" are not exposed either.
        description: typing.Dict[str, typing.Any] = {
            'schema': PROBLEM_SCHEMA_VERSION,
            'id': problem_id or document_problem_id,
            'version': problem_version or problem_doc['about'].get('problemVersion', '1.0'),
            'name': problem_name or problem_doc['about']['problemName'],
            'location_uris': [
                # We reconstruct the URI to normalize it.
                utils.path_to_uri(problem_doc_path),
            ],
            'problem': {},
        }

        task_keywords: typing.List = []

        # Legacy (before v4.0.0).
        task_keywords += TASK_TYPE_TO_KEYWORDS_MAP[problem_doc['about'].get('taskType', None)]
        task_keywords += TASK_TYPE_TO_KEYWORDS_MAP[problem_doc['about'].get('taskSubType', None)]

        if problem_doc['about'].get('taskKeywords', []):
            for task_keyword in problem_doc['about']['taskKeywords']:
                task_keywords.append(TaskKeyword.parse(task_keyword))

        if task_keywords:
            description['problem']['task_keywords'] = sorted(set(task_keywords))

        if performance_metrics:
            description['problem']['performance_metrics'] = performance_metrics

        if problem_doc['about'].get('problemDescription', None):
            description['description'] = problem_doc['about']['problemDescription']

        if problem_doc['about'].get('problemURI', None):
            typing.cast(typing.List[str], description['location_uris']).append(problem_doc['about']['problemURI'])

        if problem_doc['about'].get('sourceURI', None):
            description['source'] = {
                'uris': [problem_doc['about']['sourceURI']],
            }

        if inputs:
            description['inputs'] = inputs

        if 'dataAugmentation' in problem_doc:
            description['data_augmentation'] = problem_doc['dataAugmentation']

        # We do not want empty objects.
        if not description['problem']:
            del description['problem']

        problem_class = self.get_problem_class()

        return problem_class(description)


class D3MProblemSaver(Saver):
    """
    A class for saving of D3M problems.

    This saver supports only saving to local file system.
    URI should point to the ``problemDoc.json`` file in the D3M dataset directory.
    """

    VERSION = '4.1.1'

    def can_save(self, problem_uri: str) -> bool:
        if not self._is_problem(problem_uri):
            return False

        if not self._is_local_file(problem_uri):
            return False

        return True

    def _is_problem(self, uri: str) -> bool:
        try:
            parsed_uri = url_parse.urlparse(uri, allow_fragments=False)
        except Exception:
            return False

        parsed_uri_path = pathlib.PurePosixPath(url_parse.unquote(parsed_uri.path))

        if parsed_uri_path.name != 'problemDoc.json':
            return False

        return True

    def _is_local_file(self, uri: str) -> bool:
        try:
            utils.uri_to_path(uri)
        except exceptions.InvalidArgumentValueError:
            return False

        return True

    def save(self, problem: 'Problem', problem_uri: str) -> None:
        # Sanity check
        assert self.can_save(problem_uri)

        problem_path = utils.uri_to_path(problem_uri).parent
        os.makedirs(problem_path, 0o755, exist_ok=False)

        required_fields = ['id', 'name', 'version']
        for field in required_fields:
            if not problem.get(field, None):
                exceptions.InvalidProblemError(f"Problem field '{field}' is required when saving.")

        about = {
            'problemID': problem['id'],
            'problemName': problem['name'],
            'problemVersion': problem['version'],
            'problemSchemaVersion': self.VERSION,
        }

        if 'description' in problem:
            about['problemDescription'] = problem['description']

        remote_location_uris = [location_uri for location_uri in problem.get('location_uris', []) if not self._is_local_file(location_uri)]
        if remote_location_uris:
            # We are only using the first URI due to design of D3M problem format.
            about['problemURI'] = remote_location_uris[0]

        remote_source_uris = [source_uri for source_uri in problem.get('source', {}).get('uris', []) if not self._is_local_file(source_uri)]
        if remote_source_uris:
            # We are only using the first URI due to design of D3M problem format.
            about['sourceURI'] = remote_source_uris[0]

        if 'data_augmentation' in problem:
            about['dataAugmentation'] = problem['data_augmentation']

        performance_metrics = []
        if 'problem' not in problem:
            exceptions.InvalidProblemError(f"Problem field 'problem.task_keywords' is required when saving.")

        if 'task_keywords' not in problem['problem']:
            exceptions.InvalidProblemError(f"Problem field 'problem.task_keywords' is required when saving.")

        about['taskKeywords'] = []
        for task_keyword in problem['problem']['task_keywords']:
            try:
                about['taskKeywords'].append(task_keyword.unparse())
            except exceptions.InvalidStateError:
                logger.warning(
                    "D3M problem format does not support keyword '%(task_keyword)', skipping it.",
                    {
                        'task_keyword': task_keyword,
                    },
                )

        for performance_metric in problem['problem'].get('performance_metrics', []):
            params = {}

            if 'pos_label' in performance_metric:
                params['posLabel'] = performance_metric['pos_label']

            if 'K' in performance_metric:
                params['k'] = performance_metric['K']

            try:
                params['metric'] = performance_metric['metric'].unparse()
                performance_metrics.append(params)
            except exceptions.InvalidStateError:
                logger.warning(
                    "D3M problem format does not support metric '%(metric)', skipping it.",
                    {
                        'metric': performance_metric['metric'],
                    },
                )

        data = []
        for problem_input in problem['inputs']:
            _input = {
                'datasetID': problem_input['dataset_id'],
            }

            targets = []
            for target in problem_input['targets']:
                _target = {
                    'targetIndex': target['target_index'],
                    'resID': target['resource_id'],
                    'colIndex': target['column_index'],
                    'colName': target['column_name'],
                }

                if 'clusters_number' in target:
                    _target['numClusters'] = target['clusters_number']

                targets.append(_target)

            privileged_data_columns = []
            for privileged_data in problem_input.get('privileged_data', []):
                privileged_data_columns.append({
                    'privilegedDataIndex': privileged_data['privileged_data_index'],
                    'resID': privileged_data['resource_id'],
                    'colIndex': privileged_data['column_index'],
                    'colName': privileged_data['column_name'],
                })

            if problem_input.get('forecasting_horizon', {}).get('horizon_value', None):
                _input['forecastingHorizon'] = {
                    'resID': problem_input['forecasting_horizon']['resource_id'],
                    'colIndex': problem_input['forecasting_horizon']['column_index'],
                    'colName': problem_input['forecasting_horizon']['column_name'],
                    'horizonValue': problem_input['forecasting_horizon']['horizon_value'],
                }

            if targets:
                _input['targets'] = targets

            if privileged_data_columns:
                _input['privilegedData'] = privileged_data_columns

            data.append(_input)

        inputs = {
            'performanceMetrics': performance_metrics,
            'data': data
        }

        problem_doc = {
            'about': about,
            'inputs': inputs,
        }

        problem_description_path = problem_path / 'problemDoc.json'

        with open(problem_description_path, 'x', encoding='utf8') as f:
            json.dump(problem_doc, f, indent=2, allow_nan=False)


# Expects uri: https://www.openml.org/t/{problem_id}
OPENML_ID_REGEX = re.compile(r'^/t/(\d+)$')

# TODO: Include mapping from OpenML
# Tasks Types: https://www.openml.org/search?type=task_type
# Metrics: https://www.openml.org/search?q=%2520measure_type%3Aevaluation_measure&type=measure


class OpenMLTaskType(utils.Enum):
    SUPERVISED_CLASSIFICATION = 1
    SUPERVISED_REGRESSION = 2


class OpenMLProblemLoader(Loader):
    """
    A class for loading OpenML problems.
    """

    def can_load(self, problem_uri: str) -> bool:
        try:
            parsed_uri = url_parse.urlparse(problem_uri, allow_fragments=False)
        except Exception:
            return False

        if parsed_uri.scheme != 'https':
            return False

        if 'www.openml.org' != parsed_uri.netloc:
            return False

        parsed_uri_path = url_parse.unquote(parsed_uri.path)

        if OPENML_ID_REGEX.search(parsed_uri_path) is None:
            return False

        return True

    # "strict_digest" are ignored because there is no digest in OpenML problem descriptions.
    # "handle_score_split" is ignored.
    def load(self, problem_uri: str, *, problem_id: str = None, problem_version: str = None,
             problem_name: str = None, strict_digest: bool = False, handle_score_split: bool = True) -> 'Problem':
        assert self.can_load(problem_uri)

        parsed_uri = url_parse.urlparse(problem_uri, allow_fragments=False)
        parsed_uri_path = url_parse.unquote(parsed_uri.path)
        task_id = int(OPENML_ID_REGEX.search(parsed_uri_path)[1])  # type: ignore

        try:
            response = requests.get(
                "https://www.openml.org/api/v1/json/task/{task_id}".format(task_id=task_id)
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                raise exceptions.ProblemNotFoundError(
                    "OpenML problem '{problem_uri}' cannot be found.".format(problem_uri=problem_uri),
                ) from error
            else:
                raise error

        openml_task = response.json()['task']

        for field in openml_task['input']:
            if field['name'] == 'source_data':
                source_data = field['data_set']
                break
        else:
            raise exceptions.NotFoundError("OpenML dataset reference could not be found.")

        try:
            response = requests.get(
                'https://www.openml.org/api/v1/json/data/features/{dataset_id}'.format(
                    dataset_id=source_data['data_set_id']
                )
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                raise exceptions.DatasetNotFoundError(
                    "OpenML dataset \"{dataset_id}\" cannot be found.".format(dataset_id=source_data['data_set_id']),
                ) from error
            else:
                raise error

        data_features = response.json()['data_features']['feature']

        try:
            response = requests.get(
                'https://www.openml.org/d/{dataset_id}/json'.format(
                    dataset_id=source_data['data_set_id']
                )
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                raise exceptions.DatasetNotFoundError(
                    "OpenML dataset \"{dataset_id}\" cannot be found.".format(dataset_id=source_data['data_set_id']),
                ) from error
            else:
                raise error

        # We use this as a workaround for additional metadata about a target column.
        # See: https://github.com/openml/OpenML/issues/1085
        data_features2 = response.json()['features']

        target = None
        has_index_column = False
        for feature in data_features:
            if feature['name'] == source_data['target_feature']:
                target = feature

            if 'is_row_identifier' in feature and feature['is_row_identifier'] == 'true':
                has_index_column = True

        if target is None:
            raise exceptions.InvalidProblemError(
                "Target column \"{target_column}\" not found in OpenML dataset \"{dataset_id}\".".format(
                    target_column=source_data['target_feature'], dataset_id=source_data['data_set_id']
                )
            )

        # We use this as a workaround for additional metadata about a target column.
        # See: https://github.com/openml/OpenML/issues/1085
        for feature in data_features2:
            if feature['name'] == source_data['target_feature']:
                target2 = feature
                break
        else:
            raise exceptions.InvalidProblemError(
                "Target column \"{target_column}\" not found in OpenML dataset \"{dataset_id}\".".format(
                    target_column=source_data['target_feature'], dataset_id=source_data['data_set_id']
                )
            )

        # Case when there is no index column on the dataset, we will be adding d3mIndex later.
        target_column_index = int(target['index'])
        if not has_index_column:
            target_column_index += 1

        inputs = [{
            'dataset_id': 'openml_dataset_{dataset_id}'.format(dataset_id=source_data['data_set_id']),
            'targets': [{
                'target_index': 0,
                'resource_id': 'learningData',
                'column_index': target_column_index,
                'column_name': source_data['target_feature'],
            }]
        }]

        task_keywords = [TaskKeyword.TABULAR]
        performance_metrics = []

        if openml_task['task_type_id'] == str(OpenMLTaskType.SUPERVISED_CLASSIFICATION.value):
            task_keywords.append(TaskKeyword.CLASSIFICATION)
            performance_metrics.append({'metric': PerformanceMetric.ACCURACY})

            # Sanity check
            assert int(target2['distinct']) > 1

            # We need to know if the task is binary or multiclass.
            if int(target2['distinct']) > 2:
                task_keywords.append(TaskKeyword.MULTICLASS)
            else:
                task_keywords.append(TaskKeyword.BINARY)

        elif openml_task['task_type_id'] == str(OpenMLTaskType.SUPERVISED_REGRESSION.value):
            performance_metrics.append({'metric': PerformanceMetric.MEAN_ABSOLUTE_ERROR})
            task_keywords.append(TaskKeyword.REGRESSION)

        else:
            raise exceptions.NotSupportedError('Task {task_type} not supported for OpenMLProblemLoader'.format(
                task_type=openml_task['task_type'])
            )

        problem = {
            'performance_metrics': performance_metrics,
            'task_keywords': sorted(set(task_keywords)),
        }

        description = {
            'schema': PROBLEM_SCHEMA_VERSION,
            'id': problem_id or 'openml_problem_{task_id}'.format(task_id=openml_task['task_id']),
            'version': problem_version or '1.0',
            'name': problem_name or openml_task['task_name'],
            'location_uris': [problem_uri],
            'source': {
                'uris': [f"https://www.openml.org/t/{openml_task['task_id']}"],
            },
            'inputs': inputs,
            'problem': problem,
        }

        if openml_task.get('tag', []):
            if utils.is_sequence(openml_task['tag']):
                description['keywords'] = openml_task['tag']
            else:
                description['keywords'] = [openml_task['tag']]

        problem_class = self.get_problem_class()
        return problem_class(description)


P = typing.TypeVar('P', bound='Problem')


# TODO: It should be probably immutable.
class Problem(dict):
    """
    A class representing a problem.
    """

    def __init__(self, problem_description: typing.Dict = None, *, strict_digest: bool = False) -> None:
        if problem_description is not None:
            super().__init__(problem_description)
        else:
            super().__init__()

        PROBLEM_SCHEMA_VALIDATOR.validate(self)

        if 'digest' in self:
            digest = self.get_digest()

            if digest != self['digest']:
                if strict_digest:
                    raise exceptions.DigestMismatchError(
                        "Digest for problem description '{problem_id}' does not match a computed one. Provided digest: {problem_digest}. Computed digest: {new_problem_digest}.".format(
                            problem_id=self['id'],
                            problem_digest=self['digest'],
                            new_problem_digest=digest,
                        )
                    )
                else:
                    logger.warning(
                        "Digest for problem description '%(problem_id)s' does not match a computed one. Provided digest: %(problem_digest)s. Computed digest: %(new_problem_digest)s.",
                        {
                            'problem_id': self['id'],
                            'problem_digest': self['digest'],
                            'new_problem_digest': digest,
                        },
                    )

            # We do not want it to be stored in the object because it can become
            # obsolete. Use "get_digest" to get the current digest.
            del self['digest']

    loaders: typing.List[Loader] = [
        D3MProblemLoader(),
        OpenMLProblemLoader(),
    ]
    savers: typing.List[Saver] = [
        D3MProblemSaver(),
    ]

    @classmethod
    def load(cls, problem_uri: str, *, problem_id: str = None, problem_version: str = None,
             problem_name: str = None, strict_digest: bool = False, handle_score_split: bool = True) -> 'Problem':
        """
        Tries to load problem from ``problem_uri`` using all registered problem loaders.

        Parameters
        ----------
        problem_uri:
            A URI to load.
        problem_id:
            Override problem ID determined by the loader.
        problem_version:
            Override problem version determined by the loader.
        problem_name:
            Override problem name determined by the loader.
        strict_digest:
            If computed digest does not match the one provided in metadata, raise an exception?
        handle_score_split:
            Rename a scoring problem to not have the same name as testing problem
            and update dataset references.

        Returns
        -------
        A loaded problem.
        """

        for loader in cls.loaders:
            if loader.can_load(problem_uri):
                return loader.load(
                    problem_uri, problem_id=problem_id, problem_version=problem_version,
                    problem_name=problem_name, strict_digest=strict_digest,
                    handle_score_split=handle_score_split,
                )

        raise exceptions.ProblemUriNotSupportedError(
            "No known loader could load problem from '{problem_uri}'.".format(problem_uri=problem_uri)
        )

    def save(self, problem_uri: str) -> None:
        """
        Tries to save dataset to ``problem_uri`` using all registered problem savers.

        Parameters
        ----------
        problem_uri:
            A URI to save to.
        """

        for saver in self.savers:
            if saver.can_save(problem_uri):
                saver.save(self, problem_uri)
                return

        raise exceptions.ProblemUriNotSupportedError("No known saver could save problem to '{problem_uri}'.".format(problem_uri=problem_uri))

    # TODO: Allow one to specify priority which would then insert loader at a different place and not at the end?
    @classmethod
    def register_loader(cls, loader: Loader) -> None:
        """
        Registers a new problem loader.

        Parameters
        ----------
        loader:
            An instance of the loader class implementing a new loader.
        """

        cls.loaders.append(loader)

    # TODO: Allow one to specify priority which would then insert saver at a different place and not at the end?
    @classmethod
    def register_saver(cls, saver: Saver) -> None:
        """
        Registers a new dataset saver.

        Parameters
        ----------
        saver:
            An instance of the saver class implementing a new saver.
        """

        cls.savers.append(saver)

    def __repr__(self) -> str:
        return self.__str__()

    def _get_description_keys(self) -> typing.Sequence[str]:
        return 'id', 'name', 'location_uris'

    def __str__(self) -> str:
        return '{class_name}({description})'.format(
            class_name=type(self).__name__,
            description=', '.join('{key}=\'{value}\''.format(key=key, value=self[key]) for key in self._get_description_keys() if key in self),
        )

    def copy(self: P) -> P:
        return copy.deepcopy(self)

    @classmethod
    def _canonical_problem_description(cls: typing.Type[P], problem_description: typing.Dict) -> P:
        """
        Before we compute digest of the problem description, we have to convert it to a
        canonical structure.

        Currently, this is just removing any local URIs the description might have.
        """

        # Making a copy.
        problem_description = dict(problem_description)

        utils.filter_local_location_uris(problem_description)

        if 'digest' in problem_description:
            del problem_description['digest']

        return cls(problem_description)

    def get_digest(self) -> str:
        # We use "to_json_structure" here and not "to_reversible_json_structure"
        # because pickled values might not be deterministic.
        return utils.compute_digest(utils.to_json_structure(self._to_simple_structure(canonical=True)))

    def _to_simple_structure(self, *, canonical: bool = False) -> typing.Dict:
        problem_description = self

        if canonical:
            problem_description = self._canonical_problem_description(self)

        return dict(problem_description)

    def to_simple_structure(self, *, canonical: bool = False) -> typing.Dict:
        problem_description = self._to_simple_structure(canonical=canonical)

        problem_description['digest'] = self.get_digest()

        return problem_description

    @classmethod
    def from_simple_structure(cls: typing.Type[P], structure: typing.Dict, *, strict_digest: bool = False) -> P:
        return cls(structure, strict_digest=strict_digest)

    def to_json_structure(self, *, canonical: bool = False) -> typing.Dict:
        """
        For standard enumerations we map them to strings. Non-standard problem
        description fields we convert in a reversible manner.
        """

        PROBLEM_SCHEMA_VALIDATOR.validate(self)

        simple_structure = copy.deepcopy(self.to_simple_structure(canonical=canonical))

        if simple_structure.get('problem', {}).get('task_keywords', []):
            simple_structure['problem']['task_keywords'] = [task_keyword.name for task_keyword in simple_structure['problem']['task_keywords']]
        if simple_structure.get('problem', {}).get('performance_metrics', []):
            for metric in simple_structure['problem']['performance_metrics']:
                metric['metric'] = metric['metric'].name

        return utils.to_reversible_json_structure(simple_structure)

    @classmethod
    def from_json_structure(cls: typing.Type[P], structure: typing.Dict, *, strict_digest: bool = False) -> P:
        """
        For standard enumerations we map them from strings. For non-standard problem
        description fields we used a reversible conversion.
        """

        simple_structure = utils.from_reversible_json_structure(structure)

        # Legacy (before v4.0.0).
        legacy_task_keywords: typing.List[TaskKeyword] = []  # type: ignore
        legacy_task_keywords += JSON_TASK_TYPE_TO_KEYWORDS_MAP[simple_structure.get('problem', {}).get('task_type', None)]
        legacy_task_keywords += JSON_TASK_TYPE_TO_KEYWORDS_MAP[simple_structure.get('problem', {}).get('task_subtype', None)]

        if legacy_task_keywords:
            # We know "problem" field exists.
            simple_structure['problem']['task_keywords'] = simple_structure['problem'].get('task_keywords', []) + legacy_task_keywords

        if simple_structure.get('problem', {}).get('task_keywords', []):
            mapped_task_keywords = []
            for task_keyword in simple_structure['problem']['task_keywords']:
                if isinstance(task_keyword, str):
                    mapped_task_keywords.append(TaskKeyword[task_keyword])
                else:
                    mapped_task_keywords.append(task_keyword)
            simple_structure['problem']['task_keywords'] = mapped_task_keywords
        if simple_structure.get('problem', {}).get('performance_metrics', []):
            for metric in simple_structure['problem']['performance_metrics']:
                if isinstance(metric['metric'], str):
                    metric['metric'] = PerformanceMetric[metric['metric']]

        return cls.from_simple_structure(simple_structure, strict_digest=strict_digest)


@deprecate.function(message="use Problem.load class method instead")
def parse_problem_description(problem_doc_path: str) -> Problem:
    """
    Parses problem description according to ``problem.json`` metadata schema.

    It converts constants to enumerations when suitable.

    Parameters
    ----------
    problem_doc_path:
        File path to the problem description (``problemDoc.json``).

    Returns
    -------
    A parsed problem.
    """

    return Problem.load(problem_uri=utils.path_to_uri(problem_doc_path))


def problem_serializer(obj: Problem) -> dict:
    """
    Serializer to be used with PyArrow.
    """

    data: typing.Dict = {
        'problem': dict(obj),
    }

    if type(obj) is not Problem:
        data['type'] = type(obj)

    return data


def problem_deserializer(data: dict) -> Problem:
    """
    Deserializer to be used with PyArrow.
    """

    problem = data.get('type', Problem)(data['problem'])
    return problem


def get_problem(problem_uri: str, *, strict_digest: bool = False, datasets_dir: str = None, handle_score_split: bool = True) -> Problem:
    if datasets_dir is not None:
        datasets, problem_descriptions = utils.get_datasets_and_problems(datasets_dir, handle_score_split)

        if problem_uri in problem_descriptions:
            problem_uri = problem_descriptions[problem_uri]

    problem_uri = utils.path_to_uri(problem_uri)

    return Problem.load(problem_uri, strict_digest=strict_digest)


def describe_handler(
    arguments: argparse.Namespace, *, problem_resolver: typing.Callable = None,
) -> None:
    if problem_resolver is None:
        problem_resolver = get_problem

    output_stream = getattr(arguments, 'output', sys.stdout)

    has_errored = False

    for problem_path in arguments.problems:
        if getattr(arguments, 'list', False):
            print(problem_path, file=output_stream)

        try:
            problem = problem_resolver(problem_path, strict_digest=getattr(arguments, 'strict_digest', False))
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error parsing problem: {problem_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error parsing problem: {problem_path}") from error

        try:
            problem_description = problem.to_json_structure(canonical=True)

            if getattr(arguments, 'print', False):
                pprint.pprint(problem_description, stream=output_stream)
            elif not getattr(arguments, 'no_print', False):
                json.dump(
                    problem_description,
                    output_stream,
                    indent=(getattr(arguments, 'indent', 2) or None),
                    sort_keys=getattr(arguments, 'sort_keys', False),
                    allow_nan=False,
                )
                output_stream.write('\n')
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error describing problem: {problem_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error describing problem: {problem_path}") from error

    if has_errored:
        sys.exit(1)


def convert_handler(arguments: argparse.Namespace, *, problem_resolver: typing.Callable = None) -> None:
    if problem_resolver is None:
        problem_resolver = get_problem

    try:
        problem = problem_resolver(arguments.input_uri, strict_digest=getattr(arguments, 'strict_digest', False))
    except Exception as error:
        raise Exception(f"Error loading problem '{arguments.input_uri}'.") from error

    output_uri = utils.path_to_uri(arguments.output_uri)

    try:
        problem.save(output_uri)
    except Exception as error:
        raise Exception(f"Error saving problem '{arguments.input_uri}' to '{output_uri}'.") from error


def main(argv: typing.Sequence) -> None:
    raise exceptions.NotSupportedError("This CLI has been removed. Use \"python3 -m d3m problem describe\" instead.")


if __name__ == '__main__':
    main(sys.argv)
