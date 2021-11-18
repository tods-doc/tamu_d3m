import contextlib
import json
import gzip
import io
import logging
import os.path
import pickle
import random
import shutil
import sys
import tempfile
import traceback
import typing
import unittest
import uuid

import pandas

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common-primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')
sys.path.insert(0, TEST_PRIMITIVES_DIR)

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.fixed_split import FixedSplitDatasetSplitPrimitive
from common_primitives.kfold_split import KFoldDatasetSplitPrimitive
from common_primitives.no_split import NoSplitDatasetSplitPrimitive
from common_primitives.random_forest import RandomForestClassifierPrimitive
from common_primitives.redact_columns import RedactColumnsPrimitive
from common_primitives.train_score_split import TrainScoreDatasetSplitPrimitive

from test_primitives.random_classifier import RandomClassifierPrimitive
from test_primitives.fake_score import FakeScorePrimitive

from d3m import cli, exceptions, index, runtime, utils
from d3m.container import dataset as dataset_module
from d3m.contrib import pipelines as contrib_pipelines
from d3m.contrib.primitives.compute_scores import ComputeScoresPrimitive
from d3m.metadata import base as metadata_base, pipeline as pipeline_module, pipeline_run as pipeline_run_module, problem as problem_module

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PROBLEM_DIR = os.path.join(TEST_DATA_DIR, 'problems')
DATASET_DIR = os.path.join(TEST_DATA_DIR, 'datasets')
PIPELINE_DIR = os.path.join(TEST_DATA_DIR, 'pipelines')


class TestCLIRuntime(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @classmethod
    def setUpClass(cls):
        to_register = [
            ColumnParserPrimitive,
            # We do not have to load the scoring primitive, but loading it here prevents the package from loading all primitives.
            ComputeScoresPrimitive,
            ConstructPredictionsPrimitive,
            DatasetToDataFramePrimitive,
            FakeScorePrimitive,
            FixedSplitDatasetSplitPrimitive,
            KFoldDatasetSplitPrimitive,
            NoSplitDatasetSplitPrimitive,
            RandomClassifierPrimitive,
            RandomForestClassifierPrimitive,
            RedactColumnsPrimitive,
            TrainScoreDatasetSplitPrimitive,
        ]

        # To hide any logging or stdout output.
        with utils.silence():
            for primitive in to_register:
                index.register_primitive(primitive.metadata.query()['python_path'], primitive)

    def _call_cli_runtime(self, arg) -> typing.Sequence[logging.LogRecord]:
        logger = logging.getLogger('d3m.runtime')
        with utils.silence():
            with self.assertLogs(logger=logger) as cm:
                # So that at least one message is logged.
                logger.warning("Debugging.")
                cli.main(arg)
        # We skip our "debugging" message.
        return cm.records[1:]

    def _call_cli_runtime_without_fail(self, arg):
        try:
            return self._call_cli_runtime(arg)
        except Exception as e:
            self.fail(traceback.format_exc())

    def _assert_valid_saved_pipeline_runs(self, pipeline_run_path):
        with open(pipeline_run_path, 'r') as f:
            for pipeline_run_dict in list(utils.yaml_load_all(f)):
                try:
                    pipeline_run_module.validate_pipeline_run(pipeline_run_dict)
                except Exception as e:
                    self.fail(traceback.format_exc())

    def _validate_previous_pipeline_run_ids(self, pipeline_run_path):
        ids = set()
        prev_ids = set()
        with open(pipeline_run_path, 'r') as f:
            for pipeline_run_dict in list(utils.yaml_load_all(f)):
                ids.add(pipeline_run_dict['id'])
                if 'previous_pipeline_run' in pipeline_run_dict:
                    prev_ids.add(pipeline_run_dict['previous_pipeline_run']['id'])
        self.assertTrue(
            prev_ids.issubset(ids),
            'Some previous pipeline run ids {} are not in the set of pipeline run ids {}'.format(prev_ids, ids)
        )

    def test_fit_multi_input(self):
        self._generate_pipeline_run('fit', use_multi_input=True)
        self._assert_valid_saved_pipeline_runs(self._get_pipeline_run_path())
        self._assert_standard_output_metadata()

    def _assert_list_files(self, dir, expected):
        self.assertEqual([f.as_posix() for f in utils.list_files(dir)], expected)

    def test_fit_multi_input_without_problem(self):
        self._generate_pipeline_run('fit', use_multi_input=True, use_problem=False)

        self._assert_list_files(self.test_dir, [
            'fitted-pipeline.pickle',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])

        self._assert_valid_saved_pipeline_runs(self._get_pipeline_run_path())

        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=11225, outputs_path=self._get_exposed_outputs_path(outputs_name='outputs.0'))
        self._assert_prediction_sum(prediction_sum=11225, outputs_path=self._get_output_path())

    # TODO: test_<fit, fit_produce, produce>_without_problem (single input)

    def test_produce_multi_input_without_problem(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_multi_input=True, use_problem=False)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('produce', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path, use_multi_input=True)

        self._assert_list_files(produce_test_dir, [
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])

        self._assert_valid_saved_pipeline_runs(self._get_pipeline_run_path(produce_test_dir))

        self._assert_standard_output_metadata(produce_test_dir)
        self._assert_prediction_sum(prediction_sum=11008, outputs_path=self._get_exposed_outputs_path(produce_test_dir, 'outputs.0'))
        self._assert_prediction_sum(prediction_sum=11008, outputs_path=self._get_output_path(produce_test_dir))

    def test_fit_produce_multi_input_without_problem(self):
        self._generate_pipeline_run('fit-produce', use_multi_input=True, use_problem=False)

        self._assert_list_files(self.test_dir, [
            'fitted-pipeline.pickle',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])

        pipeline_run_path = self._get_pipeline_run_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=11008, outputs_path=self._get_exposed_outputs_path(outputs_name='outputs.0'))
        self._assert_prediction_sum(prediction_sum=11008, outputs_path=self._get_output_path())

    def test_nonstandard_fit_without_problem(self):
        self._generate_pipeline_run('fit', use_standard_pipeline=False, use_problem=False)

        self._assert_list_files(self.test_dir, [
            'fitted-pipeline.pickle',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'outputs.1/data.csv',
            'outputs.1/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
        ])

        self._assert_valid_saved_pipeline_runs(self._get_pipeline_run_path())

        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=10710, outputs_path=self._get_exposed_outputs_path(outputs_name='outputs.0'))
        self._assert_nonstandard_output(outputs_name='outputs.1')

    def test_nonstandard_produce_without_problem(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_standard_pipeline=False, use_problem=False)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run(
            'produce', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path, use_standard_pipeline=False, use_problem=False
        )

        self._assert_list_files(produce_test_dir, [
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'outputs.1/data.csv',
            'outputs.1/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json'
        ])

        pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        self._assert_standard_output_metadata(produce_test_dir)
        self._assert_prediction_sum(prediction_sum=12106, outputs_path=self._get_exposed_outputs_path(produce_test_dir, 'outputs.0'))
        self._assert_nonstandard_output(test_dir=produce_test_dir, outputs_name='outputs.1')

    def test_nonstandard_fit_produce_without_problem(self):
        self._generate_pipeline_run('fit-produce', use_standard_pipeline=False, use_problem=False)

        self._assert_list_files(self.test_dir, [
            'fitted-pipeline.pickle',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'outputs.1/data.csv',
            'outputs.1/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
        ])

        pipeline_run_path = self._get_pipeline_run_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=12106, outputs_path=self._get_exposed_outputs_path(outputs_name='outputs.0'))
        self._assert_nonstandard_output(outputs_name='outputs.1')

    def test_fit_produce_multi_input(self):
        self._generate_pipeline_run('fit-produce', use_multi_input=True)

        self._assert_list_files(self.test_dir, [
            'fitted-pipeline.pickle',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json',
        ])

        pipeline_run_path = self._get_pipeline_run_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=11008, outputs_path=self._get_exposed_outputs_path(outputs_name='outputs.0'))

    def test_fit_score(self):
        self._generate_pipeline_run('fit-score')
        pipeline_run_path = self._get_pipeline_run_path()

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        scores_path = self._get_scores_path()
        scores_df = pandas.read_csv(scores_path)
        self.assertEqual(list(scores_df.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(scores_df.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0], ['ACCURACY', 1.0, 1.0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_fit_score_without_problem(self):
        logging_records = self._generate_pipeline_run('fit-score', use_problem=False)
        pipeline_run_path = self._get_pipeline_run_path()
        scores_path = self._get_scores_path()

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    @staticmethod
    def _get_iris_dataset_path():
        return os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json')

    @staticmethod
    def _get_iris_problem_path():
        return os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json')

    @staticmethod
    def _get_iris_split_path():
        return os.path.join(PROBLEM_DIR, 'iris_problem_1/dataSplits.csv')

    @classmethod
    def _get_iris_split_test_indices(cls):
        split_df = pandas.read_csv(
            cls._get_iris_split_path(),
            dtype=str,
            header=0,
            na_filter=False,
            encoding='utf8',
            low_memory=False,
            memory_map=True,
        )
        return split_df.loc[split_df['type'] == 'TEST']['d3mIndex'].tolist()

    @staticmethod
    def _get_sklearn_iris_dataset_path():
        return 'sklearn://iris'

    @staticmethod
    def _get_random_forest_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'random-forest-classifier.yml')

    @staticmethod
    def _get_no_split_data_pipeline_path():
        return contrib_pipelines.NO_SPLIT_TABULAR_SPLIT_PIPELINE_PATH

    @staticmethod
    def _get_fixed_split_data_pipeline_path():
        return contrib_pipelines.FIXED_SPLIT_TABULAR_SPLIT_PIPELINE_PATH

    @staticmethod
    def _get_train_test_split_data_pipeline_path():
        return contrib_pipelines.TRAIN_TEST_TABULAR_SPLIT_PIPELINE_PATH

    def _get_unique_test_dir(self):
        unique_test_dir = os.path.join(self.test_dir, str(uuid.uuid4()))
        os.mkdir(unique_test_dir)
        return unique_test_dir

    def _get_pipeline_run_path(self, test_dir=None):
        if test_dir is None:
            test_dir = self.test_dir
        return os.path.join(test_dir, 'pipeline_run.yml')

    def _get_scores_path(self, test_dir=None):
        if test_dir is None:
            test_dir = self.test_dir
        return os.path.join(test_dir, 'scores.csv')

    def _generate_pipeline_run(
        self, command: str, test_dir: str = None, *, dataset: str = 'iris', use_problem: bool = True,
        use_standard_pipeline: bool = True, use_multi_input: bool = False, use_metrics: bool = True,
        fitted_pipeline_path: str = None, cache_pipelines: bool = False, split_method: str = None,
        train_score_ratio: float = None, number_of_folds: int = None, shuffle: bool = False,
        save_scores: bool = True, random_seed: int = None
    ) -> typing.Sequence[logging.LogRecord]:
        """
        A utility method for generating various cases of pipeline runs for testing.
        Uses the Iris dataset.
        Some parameters are ignored, depending on the values of other parameters, instead of raising errors, to make this method more usable and less brittle.

        Parameters
        ----------
        command
            Runtime command from ['fit', 'produce', 'score', 'fit-produce', 'fit-score', 'evaluate'].
        test_dir
            Base directory to write all output files, default of None resolves to self.test_dir.
        dataset
            One of ['iris', 'sklearn-iris'].
        use_problem
            Whether to pass a problem document for the Iris dataset to the runtime.
            Ignored when command not in ['fit', 'fit-produce', 'score', fit-score', 'evaluate'].
        use_standard_pipeline
            Whether to use a standard (True) or non-standard (False) pipeline.
        use_multi_input
            Whether to use a pipeline that takes multiple datasets as input.
            When True, the Iris dataset is input twice.
            Ignored when use_standard_pipeline is False.
            TODO: make a test non-standard pipeline that use multiple inputs
        use_metrics
            Whether to specify to the runtime to use the '--metric' flag.
            When True, F1_MACRO and ACCURACY are both used.
            This has no effect when use_problem is False because a fake scoring pipeline is used.
            Ignored when command not in ['score', 'fit-score', 'evaluate'].
        fitted_pipeline_path
            When command is one of ['produce', 'score'], a path to a fitted pipeline must be provided.
        cache_pipelines
            Whether to copy the pipeline(s) to test_dir.
        split_method
            The type of data splitting pipeline to use, one of ['train-test', 'cross-validation', 'no-split', 'fixed-split'].
            Default of None indicates no data preparation pipeline.
        train_score_ratio
            When split_method is 'train-test', this determines the size of the train split.
        number_of_folds
            When split_method is 'cross-validation', this determines the number of folds.
        shuffle
            Whether the data splitting pipeline should shuffle the data.
        save_scores
            When the `command` is in ['score', 'fit-score', 'evaluate'], this determines whether to set the '--scores' flag.

        Return
        ------
        logging_records
            Logging records generated by _call_cli_runtime_without_fail.
        """

        if command not in ['fit', 'produce', 'score', 'fit-produce', 'fit-score', 'evaluate']:
            raise exceptions.InvalidArgumentValueError("command {command} not in ['fit', 'produce', 'score', 'fit-produce', 'fit-score', 'evaluate']".format(command=command))

        if command in ['score', 'fit-score', 'evaluate'] and not use_standard_pipeline:
            raise exceptions.InvalidArgumentValueError('cannot score a non-standard pipeline')

        if command in ['produce', 'score'] and fitted_pipeline_path is None:
            raise exceptions.InvalidArgumentValueError('{command} requires a fitted_pipeline_path'.format(command=command))

        if test_dir is None:
            test_dir = self.test_dir

        dataset_path = None
        problem_path = None
        if dataset == 'iris':
            dataset_path = self._get_iris_dataset_path()
            if use_problem:
                problem_path = self._get_iris_problem_path()
        elif dataset == 'sklearn-iris':
            dataset_path = self._get_sklearn_iris_dataset_path()
            if use_problem:
                problem_path = self._get_sklearn_iris_problem_path()
        else:
            raise exceptions.InvalidArgumentValueError(
                "dataset {dataset} not in ['iris', 'sklearn-iris']".format(dataset=dataset)
            )

        pipeline_path = self._get_test_pipeline_path(use_problem=use_problem, use_standard_pipeline=use_standard_pipeline, use_multi_input=use_multi_input)

        args = ['d3m', 'runtime']

        # --random-seed
        if random_seed is not None:
            args += ['--random-seed', str(random_seed)]

        args += [
            command,
            '--output-run', self._get_pipeline_run_path(test_dir)
        ]

        # TODO: --strict-digest?

        # redundant if blocks help separate logic for separate runtime flags

        # --pipeline
        if command in ['fit', 'fit-produce', 'fit-score', 'evaluate']:
            args += ['--pipeline', pipeline_path]
            if cache_pipelines:
                self._cache_pipeline_for_rerun(pipeline_path, test_dir)

        # --not-standard-pipeline
            if not use_standard_pipeline:
                args += ['--not-standard-pipeline']

        # --fitted-pipeline
        if command in ['produce', 'score']:
            args += ['--fitted-pipeline', fitted_pipeline_path]

        if split_method is not None:

        # --data-pipeline
            if split_method == 'train-test':
                data_pipeline_path = self._get_train_test_split_data_pipeline_path()

        # --data-param train_score_ratio
                if train_score_ratio is not None:
                    args += ['--data-param', 'train_score_ratio', str(train_score_ratio)]

        # --data-param shuffle
                if shuffle:
                    args += ['--data-param', 'shuffle', 'true']

            elif split_method == 'cross-validation':
                data_pipeline_path = self._get_cross_validation_data_pipeline_path()

        # --data-param number_of_folds
                if number_of_folds is not None:
                    args += ['--data-param', 'number_of_folds', str(number_of_folds)]

        # --data-param shuffle
                if shuffle:
                    args += ['--data-param', 'shuffle', 'true']

            elif split_method == 'no-split':
                data_pipeline_path = self._get_no_split_data_pipeline_path()

            elif split_method == 'fixed-split':
                data_pipeline_path = self._get_fixed_split_data_pipeline_path()
                args += ['--data-param', 'primary_index_values', json.dumps(self._get_iris_split_test_indices())]

            else:
                raise exceptions.InvalidArgumentValueError(
                    "split_method '{split_method}' is invalid, must be one of ['train-test', 'cross-validation']".format(split_method=split_method)
                )

            args += ['--data-pipeline', data_pipeline_path]
            if cache_pipelines:
                self._cache_pipeline_for_rerun(data_pipeline_path, test_dir)

        # --scoring-pipeline
        if command in ['score', 'fit-score', 'evaluate'] and not use_problem:
            # standard default scoring pipeline needs the true target column which is set by the problem doc
            scoring_pipeline_path = self._get_fake_scoring_pipeline_path()
            args += ['--scoring-pipeline', scoring_pipeline_path]

            if cache_pipelines:
                self._cache_pipeline_for_rerun(scoring_pipeline_path, test_dir)

        # --metric
        if use_metrics and command in ['score', 'fit-score', 'evaluate']:
            # fake scoring pipeline does not use these
            args += ['--metric', 'F1_MACRO', '--metric', 'ACCURACY']

        # --problem
        if use_problem and command in ['fit', 'fit-produce', 'fit-score', 'evaluate']:
            args += ['--problem', problem_path]

        # --input
        if command in ['fit', 'evaluate'] or (command in ['fit-produce', 'fit-score'] and split_method is None):
            args += ['--input', dataset_path]

            if use_multi_input and use_standard_pipeline:
                # TODO: there is not currently a non standard pipeline that takes multi inputs
                args += ['--input', dataset_path]

        # --test-input
        if command in ['produce'] or (command in ['score', 'fit-produce', 'fit-score'] and split_method is None):
            args += ['--test-input', dataset_path]

            if use_multi_input and use_standard_pipeline:
                # TODO: there is not currently a non standard pipeline that takes multi inputs
                args += ['--test-input', dataset_path]

        # --score-input
        if command in ['score', 'fit-score'] and split_method is None:
            args += ['--score-input', dataset_path]

        # --full-input
        if command in ['score', 'fit-produce', 'fit-score'] and split_method is not None:
            args += ['--full-input', dataset_path]

        # --output
        if command in ['fit', 'produce', 'score', 'fit-produce', 'fit-score'] and use_standard_pipeline:  # cannot save predictions for non standard pipeline
            args += ['--output', self._get_output_path(test_dir)]

        # --scores
        if command in ['score', 'fit-score', 'evaluate'] and save_scores:
            args += ['--scores', self._get_scores_path(test_dir)]

        # --save
        if command in ['fit', 'fit-produce', 'fit-score']:
            args += ['--save', self._get_fitted_pipeline_path(test_dir)]

        # --expose-produced-outputs
        if command in ['fit', 'produce', 'score', 'fit-produce', 'fit-score']:
            args += ['--expose-produced-outputs', test_dir]

        # TODO: include?
        # '--input-run'
        # '--scoring-param'
        # '--scoring-random-seed'
        # '--data-param'
        # '--data-random-seed'

        # TODO: make failed pipeline run option?
        return self._call_cli_runtime_without_fail(args)

    def test_fit(self):
        self._generate_pipeline_run('fit')
        self._assert_valid_saved_pipeline_runs(self._get_pipeline_run_path())
        self.assertTrue(os.path.isfile(self._get_fitted_pipeline_path()))

    def test_evaluate_no_split(self):
        self._generate_pipeline_run('evaluate', split_method='no-split')
        pipeline_run_path = self._get_pipeline_run_path()
        scores_path = self._get_scores_path()

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0, 0], ['ACCURACY', 1.0, 1.0, 0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_evaluate_fixed_split(self):
        self._generate_pipeline_run('evaluate', split_method='fixed-split')
        pipeline_run_path = self._get_pipeline_run_path()
        scores_path = self._get_scores_path()

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_evaluate_fixed_split_sklearn_dataset(self):
        self._generate_pipeline_run('evaluate', dataset='sklearn-iris', split_method='fixed-split')
        pipeline_run_path = self._get_pipeline_run_path()
        scores_path = self._get_scores_path()

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_evaluate_without_scores(self):
        self._generate_pipeline_run('evaluate', split_method='no-split', save_scores=False)
        pipeline_run_path = self._get_pipeline_run_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        self.assertFalse(os.path.exists(self._get_scores_path()))

    def test_evaluate_without_problem(self):
        logging_records = self._generate_pipeline_run('evaluate', use_problem=False, split_method='train-test')
        pipeline_run_path = self._get_pipeline_run_path()
        scores_path = self._get_scores_path()

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_score(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        score_unique_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('score', score_unique_test_dir, fitted_pipeline_path=fitted_pipeline_path, use_metrics=True)
        score_pipeline_run_path = self._get_pipeline_run_path(score_unique_test_dir)
        self._assert_valid_saved_pipeline_runs(score_pipeline_run_path)

        scores_path = self._get_scores_path(score_unique_test_dir)
        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0], ['ACCURACY', 1.0, 1.0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, score_pipeline_run_path)

    def test_score_without_problem_without_metric(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_problem=False, use_metrics=False)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        score_unique_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run(
            'score', score_unique_test_dir, use_problem=False, use_metrics=False, fitted_pipeline_path=fitted_pipeline_path
        )

        scores_path = self._get_scores_path(score_unique_test_dir)
        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        pipeline_run_path = self._get_pipeline_run_path(score_unique_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_score_without_problem(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_problem=False)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        score_unique_test_dir = self._get_unique_test_dir()
        logging_records = self._generate_pipeline_run('score', score_unique_test_dir, use_problem=False, fitted_pipeline_path=fitted_pipeline_path)
        pipeline_run_path = self._get_pipeline_run_path(score_unique_test_dir)
        scores_path = self._get_scores_path(score_unique_test_dir)

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_produce(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir)

        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('produce', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path)
        pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

    def test_score_predictions(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir)
        predictions_path = self._get_output_path(fit_test_dir)
        self.assertTrue(os.path.isfile(predictions_path))

        score_unique_test_dir = self._get_unique_test_dir()
        scores_path = self._get_scores_path(score_unique_test_dir)
        arg = [
            '',
            'runtime',
            'score-predictions',
            '--score-input',
            self._get_iris_dataset_path(),
            '--problem',
            self._get_iris_problem_path(),
            '--predictions',
            predictions_path,
            '--metric',
            'ACCURACY',
            '--metric',
            'F1_MACRO',
            '--scores',
            scores_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0], ['F1_MACRO', 1.0, 1.0]])

    def test_sklearn_dataset_fit_produce(self):
        self._generate_pipeline_run('fit-produce', dataset='sklearn-iris', use_multi_input=True)

        pipeline_run_path = self._get_pipeline_run_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        self._assert_list_files(self.test_dir, [
            'fitted-pipeline.pickle',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'problemDoc.json',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])
        self._assert_standard_output_metadata(prediction_type='numpy.int64')
        self._assert_prediction_sum(prediction_sum=10648, outputs_path=self._get_exposed_outputs_path(outputs_name='outputs.0'))

    def test_sklearn_dataset_fit_produce_without_problem(self):
        self._generate_pipeline_run('fit-produce', dataset='sklearn-iris', use_problem=False)
        output_path = self._get_output_path()
        pipeline_run_path = self._get_pipeline_run_path()

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        self._assert_list_files(self.test_dir, [
            'fitted-pipeline.pickle',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json',
        ])
        self._assert_standard_output_metadata(prediction_type='numpy.int64')
        self._assert_prediction_sum(prediction_sum=10648, outputs_path=self._get_exposed_outputs_path(outputs_name='outputs.0'))
        self._assert_prediction_sum(prediction_sum=10648, outputs_path=output_path)

    def _get_sklearn_iris_problem_path(self, test_dir=None):
        if test_dir is None:
            test_dir = self.test_dir

        sklearn_iris_problem_path = os.path.join(test_dir, 'problemDoc.json')

        if not os.path.exists(sklearn_iris_problem_path):
            with open(self._get_iris_problem_path(), 'r', encoding='utf8') as problem_doc_file:
                problem_doc = json.load(problem_doc_file)

            problem_doc['inputs']['data'][0]['datasetID'] = 'sklearn://iris'

            with open(sklearn_iris_problem_path, 'x', encoding='utf8') as problem_doc_file:
                json.dump(problem_doc, problem_doc_file)

        return sklearn_iris_problem_path

    def test_sklearn_dataset_evaluate(self):
        self._generate_pipeline_run('evaluate', dataset='sklearn-iris', split_method='no-split')
        pipeline_run_path = self._get_pipeline_run_path()
        scores_path = self._get_scores_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0, 0], ['ACCURACY', 1.0, 1.0, 0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_sklearn_dataset_evaluate_without_problem(self):
        logging_records = self._generate_pipeline_run(
            'evaluate', dataset='sklearn-iris', use_problem=False, split_method='no-split'
        )
        pipeline_run_path = self._get_pipeline_run_path()
        scores_path = self._get_scores_path()

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def _assert_prediction_sum(self, prediction_sum, outputs_path):
        if prediction_sum is not None:
            with open(outputs_path, 'r') as csv_file:
                self.assertEqual(sum([int(v) for v in list(csv_file)[1:]]), prediction_sum)

    def _assert_standard_output_metadata(self, test_dir=None, outputs_name='outputs.0', prediction_type='str'):
        outputs_metadata_path = self._get_outputs_metadata_path(test_dir, outputs_name)
        with open(outputs_metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)

        self.assertEqual(
            metadata,
            [
                {
                    "selector": [],
                    "metadata": {
                        "dimension": {
                            "length": 150,
                            "name": "rows",
                            "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularRow"],
                        },
                        "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json",
                        "semantic_types": ["https://metadata.datadrivendiscovery.org/types/Table"],
                        "structural_type": "d3m.container.pandas.DataFrame",
                    },
                },
                {
                    "selector": ["__ALL_ELEMENTS__"],
                    "metadata": {
                        "dimension": {
                            "length": 1,
                            "name": "columns",
                            "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularColumn"],
                        }
                    },
                },
                {"selector": ["__ALL_ELEMENTS__", 0],
                 "metadata": {"name": "predictions", "structural_type": prediction_type}},
            ],
        )

    def _assert_nonstandard_output(self, test_dir=None, outputs_name='outputs.1'):
        exposed_outputs_path = self._get_exposed_outputs_path(test_dir, outputs_name)
        with open(exposed_outputs_path, 'r') as csv_file:
            output_dataframe = pandas.read_csv(csv_file, index_col=False)
            learning_dataframe = pandas.read_csv(
                os.path.join(DATASET_DIR, 'iris_dataset_1/tables/learningData.csv'), index_col=False)
            self.assertTrue(learning_dataframe.equals(output_dataframe))

        outputs_metadata_path = self._get_outputs_metadata_path(test_dir, outputs_name)
        with open(outputs_metadata_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)

        self.assertEqual(
            metadata,
            [
                {
                    "metadata": {
                        "dimension": {
                            "length": 150,
                            "name": "rows",
                            "semantic_types": [
                                "https://metadata.datadrivendiscovery.org/types/TabularRow"
                            ]
                        },
                        "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json",
                        "semantic_types": [
                            "https://metadata.datadrivendiscovery.org/types/Table"
                        ],
                        "structural_type": "d3m.container.pandas.DataFrame"
                    },
                    "selector": []
                },
                {
                    "metadata": {
                        "dimension": {
                            "length": 6,
                            "name": "columns",
                            "semantic_types": [
                                "https://metadata.datadrivendiscovery.org/types/TabularColumn"
                            ]
                        }
                    },
                    "selector": [
                        "__ALL_ELEMENTS__"
                    ]
                },
                {
                    "metadata": {
                        "name": "d3mIndex",
                        "semantic_types": [
                            "http://schema.org/Integer",
                            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        0
                    ]
                },
                {
                    "metadata": {
                        "name": "sepalLength",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        1
                    ]
                },
                {
                    "metadata": {
                        "name": "sepalWidth",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        2
                    ]
                },
                {
                    "metadata": {
                        "name": "petalLength",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        3
                    ]
                },
                {
                    "metadata": {
                        "name": "petalWidth",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        4
                    ]
                },
                {
                    "metadata": {
                        "name": "species",
                        "semantic_types": [
                            "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                            "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        5
                    ]
                }
            ]
        )

    def _assert_pipeline_runs_equal(self, pipeline_run_path_1, pipeline_run_path_2):
        with open(pipeline_run_path_1, 'r') as f:
            pipeline_runs_1 = list(utils.yaml_load_all(f))

        with open(pipeline_run_path_2, 'r') as f:
            pipeline_runs_2 = list(utils.yaml_load_all(f))

        self.assertEqual(len(pipeline_runs_1), len(pipeline_runs_2))

        for pipeline_run_1, pipeline_run_2 in zip(pipeline_runs_1, pipeline_runs_2):
            self.assertTrue(pipeline_run_module.PipelineRun.json_structure_equals(pipeline_run_1, pipeline_run_2))

    def test_pipeline_run_json_structure_equals(self):
        test_dir_1 = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', test_dir_1)
        pipeline_run_path_1 = self._get_pipeline_run_path(test_dir_1)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path_1)

        test_dir_2 = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', test_dir_2)
        pipeline_run_path_2 = self._get_pipeline_run_path(test_dir_2)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path_2)

        self._assert_pipeline_runs_equal(pipeline_run_path_1, pipeline_run_path_2)

    def _cache_pipeline_for_rerun(self, pipeline_path, test_dir=None):
        """
        Makes pipeline searchable by id in test_dir.
        """
        with open(pipeline_path, 'r') as f:
            pipeline = utils.yaml_load(f)
        if test_dir is None:
            test_dir = self.test_dir
        temp_pipeline_path = os.path.join(test_dir, pipeline['id'] + '.yml')
        with open(temp_pipeline_path, 'w') as f:
            utils.yaml_dump(pipeline, f)

    @staticmethod
    def _generate_seed():
        return random.randint(2**31, 2**32-1)

    def test_fit_rerun_with_hyperparams(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()

        fit_test_dir = self._get_unique_test_dir()
        pipeline_run_path = self._get_pipeline_run_path(fit_test_dir)

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, hyperparams=hyperparams,
                random_seed=random_seed, context=metadata_base.Context.TESTING,
            )

        with open(pipeline_run_path, 'w') as f:
            fit_result.pipeline_run.to_yaml(f)

        self._cache_pipeline_for_rerun(pipeline_path, fit_test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('fit', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=fit_test_dir)
        pipeline_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, pipeline_rerun_path)

    def test_fit_rerun(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, cache_pipelines=True)
        pipeline_run_path = self._get_pipeline_run_path(fit_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('fit', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=fit_test_dir)
        fit_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(fit_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, fit_rerun_path)

    def test_fit_rerun_multi_input(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_multi_input=True, cache_pipelines=True)
        pipeline_run_path = self._get_pipeline_run_path(fit_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('fit', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=fit_test_dir)
        fit_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(fit_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, fit_rerun_path)

    def test_fit_rerun_without_problem(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_problem=False, cache_pipelines=True)
        pipeline_run_path = self._get_pipeline_run_path(fit_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('fit', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=fit_test_dir)
        fit_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(fit_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, fit_rerun_path)

    def test_produce_rerun(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('produce', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path)
        produce_pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_pipeline_run_path)

        produce_rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'produce', produce_rerun_test_dir, input_run_path=produce_pipeline_run_path, pipelines_path=fit_test_dir, fitted_pipeline_path=fitted_pipeline_path
        )
        produce_rerun_path = self._get_pipeline_run_path(produce_rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_rerun_path)

        self._assert_pipeline_runs_equal(produce_pipeline_run_path, produce_rerun_path)

    def test_produce_rerun_multi_input(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_multi_input=True, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('produce', produce_test_dir, use_multi_input=True, fitted_pipeline_path=fitted_pipeline_path)
        produce_pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_pipeline_run_path)

        produce_rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'produce', produce_rerun_test_dir, input_run_path=produce_pipeline_run_path, pipelines_path=fit_test_dir, fitted_pipeline_path=fitted_pipeline_path
        )
        produce_rerun_path = self._get_pipeline_run_path(produce_rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_rerun_path)

        self._assert_pipeline_runs_equal(produce_pipeline_run_path, produce_rerun_path)

    def test_produce_rerun_without_problem(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_problem=False, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('produce', produce_test_dir, use_problem=False, fitted_pipeline_path=fitted_pipeline_path)
        produce_pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_pipeline_run_path)

        produce_rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'produce', produce_rerun_test_dir, input_run_path=produce_pipeline_run_path, pipelines_path=fit_test_dir, fitted_pipeline_path=fitted_pipeline_path
        )
        produce_rerun_path = self._get_pipeline_run_path(produce_rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_rerun_path)

        self._assert_pipeline_runs_equal(produce_pipeline_run_path, produce_rerun_path)

    def _assert_scores_equal(self, scores_path, rescores_path):
        scores = pandas.read_csv(scores_path)
        rescores = pandas.read_csv(rescores_path)
        self.assertTrue(scores.equals(rescores), '\n{}\n\n{}'.format(scores, rescores))

    def _assert_scores_equal_pipeline_run(self, scores_path, pipeline_run_path):
        csv_scores_df = pandas.read_csv(scores_path)

        with open(pipeline_run_path) as f:
            pipeline_runs = list(utils.yaml_load_all(f))

        all_pipeline_run_scores_list = []
        for pipeline_run in pipeline_runs:
            if pipeline_run['run']['phase'] == metadata_base.PipelineRunPhase.PRODUCE.name:
                pipeline_run_scores_df = pandas.DataFrame(pipeline_run['run']['results']['scores'])
                pipeline_run_scores_df['metric'] = pipeline_run_scores_df['metric'].map(lambda cell: cell['metric'])
                pipeline_run_scores_df['randomSeed'] = [pipeline_run['random_seed']] * pipeline_run_scores_df.shape[0]
                if 'fold_group' in pipeline_run['run']:
                    pipeline_run_scores_df['fold'] = [pipeline_run['run']['fold_group']['fold']] * pipeline_run_scores_df.shape[0]
                all_pipeline_run_scores_list.append(pipeline_run_scores_df)
        all_pipeline_run_scores_df = pandas.concat(all_pipeline_run_scores_list, axis=0, ignore_index=True)

        if (
            'fold' in all_pipeline_run_scores_df.columns and
            'fold' not in csv_scores_df.columns and
            csv_scores_df.shape[0] == all_pipeline_run_scores_df.shape[0]
        ):
            all_pipeline_run_scores_df.drop('fold', axis=1, inplace=True)

        self.assertEqual(sorted(csv_scores_df.columns), sorted(all_pipeline_run_scores_df.columns))
        all_pipeline_run_scores_df = all_pipeline_run_scores_df[csv_scores_df.columns.tolist()]

        # metric and/or fold order in pipeline runs may not match order in scores csv
        sort_columns = []
        if 'fold' in csv_scores_df.columns:
            sort_columns.append('fold')
        sort_columns.append('metric')
        csv_scores_df.sort_values(sort_columns, inplace=True)
        # ignore_index=True was not supported before pandas==1.0.0, so reset_index is used for backwards compatibility
        csv_scores_df.reset_index(inplace=True)
        all_pipeline_run_scores_df.sort_values(sort_columns, inplace=True)
        all_pipeline_run_scores_df.reset_index(inplace=True)

        pandas.testing.assert_frame_equal(csv_scores_df, all_pipeline_run_scores_df)

    def test_score_rerun(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()

        fit_test_dir = self._get_unique_test_dir()
        pipeline_run_path = self._get_pipeline_run_path(fit_test_dir)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        scores_path = self._get_scores_path(fit_test_dir)

        random_seed = self._generate_seed()
        metrics = runtime.get_metrics_from_list(['ACCURACY', 'F1_MACRO'])
        scoring_params = {'add_normalized_scores': 'false'}
        scoring_random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, random_seed=random_seed,
                context=metadata_base.Context.TESTING,
            )
            fit_result.check_success()
            with open(fitted_pipeline_path, 'wb') as f:
                pickle.dump(fitted_pipeline, f)

            predictions, produce_result = runtime.produce(fitted_pipeline, inputs)

            scores, score_result = runtime.score(
                predictions, inputs, scoring_pipeline=scoring_pipeline,
                problem_description=problem, metrics=metrics, predictions_random_seed=random_seed,
                context=metadata_base.Context.TESTING, scoring_params=scoring_params,
                random_seed=scoring_random_seed
            )

            self.assertFalse(score_result.has_error(), score_result.error)

            scores.to_csv(scores_path)

            runtime.combine_pipeline_runs(
                produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=inputs,
                metrics=metrics, scores=scores
            )
            with open(pipeline_run_path, 'w') as f:
                produce_result.pipeline_run.to_yaml(f)

        self.assertTrue(os.path.isfile(fitted_pipeline_path))
        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, random_seed], ['F1_MACRO', 1.0, random_seed]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

        self._cache_pipeline_for_rerun(pipeline_path, fit_test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'score', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=fit_test_dir, fitted_pipeline_path=fitted_pipeline_path
        )

        pipeline_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        self.assertTrue(os.path.isfile(pipeline_rerun_path))
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_path)
        rescores_path = self._get_scores_path(rerun_test_dir)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, pipeline_rerun_path)

    def test_score_rerun_without_problem(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, use_problem=False, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        score_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('score', score_test_dir, use_problem=False, fitted_pipeline_path=fitted_pipeline_path, cache_pipelines=True)
        produce_pipeline_run_path = self._get_pipeline_run_path(score_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_pipeline_run_path)

        score_rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'score', score_rerun_test_dir, input_run_path=produce_pipeline_run_path, pipelines_path=[fit_test_dir, score_test_dir],
            fitted_pipeline_path=fitted_pipeline_path
        )
        produce_rerun_path = self._get_pipeline_run_path(score_rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_rerun_path)

        self._assert_pipeline_runs_equal(produce_pipeline_run_path, produce_rerun_path)

    def test_fit_produce_rerun(self):
        test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit-produce', test_dir, cache_pipelines=True)
        fit_produce_pipeline_run_path = self._get_pipeline_run_path(test_dir)
        self._assert_valid_saved_pipeline_runs(fit_produce_pipeline_run_path)
        fitted_pipeline_path = self._get_fitted_pipeline_path(test_dir)
        self.assertTrue(os.stat(fitted_pipeline_path).st_size > 0)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'fit-produce', rerun_test_dir, input_run_path=fit_produce_pipeline_run_path,
            pipelines_path=test_dir, fitted_pipeline_path=fitted_pipeline_path
        )
        rerun_fit_produce_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(rerun_fit_produce_pipeline_run_path)

        self._assert_pipeline_runs_equal(fit_produce_pipeline_run_path, rerun_fit_produce_pipeline_run_path)

    def test_fit_produce_rerun_with_hyperparams(self):
        test_dir = self._get_unique_test_dir()
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_path = self._get_pipeline_run_path(test_dir)

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, hyperparams=hyperparams,
                random_seed=random_seed, context=metadata_base.Context.TESTING,
            )
            predictions, produce_result = runtime.produce(fitted_pipeline, inputs)

        with open(pipeline_run_path, 'w') as f:
            fit_result.pipeline_run.to_yaml(f)
            produce_result.pipeline_run.to_yaml(f, appending=True)

        self._cache_pipeline_for_rerun(pipeline_path, test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'fit-produce', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=test_dir,
        )
        rerun_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(rerun_pipeline_run_path)

        self._assert_pipeline_runs_equal(pipeline_run_path, rerun_pipeline_run_path)

    def test_fit_produce_rerun_multi_input(self):
        test_dir = self._get_unique_test_dir()
        random_seed = self._generate_seed()
        self._generate_pipeline_run(
            'fit-produce', test_dir, use_multi_input=True, cache_pipelines=True,
            random_seed=random_seed,
        )
        fit_produce_pipeline_run_path = self._get_pipeline_run_path(test_dir)
        self._assert_valid_saved_pipeline_runs(fit_produce_pipeline_run_path)
        fitted_pipeline_path = self._get_fitted_pipeline_path(test_dir)
        self.assertTrue(os.stat(fitted_pipeline_path).st_size > 0)

        rerun_test_dir = self._get_unique_test_dir()
        # TODO: add strict digest again, it was removed in refactoring
        self._generate_pipeline_rerun(
            'fit-produce', rerun_test_dir, input_run_path=fit_produce_pipeline_run_path,
            pipelines_path=test_dir
        )
        rerun_fit_produce_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(rerun_fit_produce_pipeline_run_path)

        self._assert_pipeline_runs_equal(fit_produce_pipeline_run_path, rerun_fit_produce_pipeline_run_path)

    def test_fit_produce_rerun_without_problem(self):
        test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run(
            'fit-produce', test_dir, use_problem=False, cache_pipelines=True
        )
        fit_produce_pipeline_run_path = self._get_pipeline_run_path(test_dir)
        self._assert_valid_saved_pipeline_runs(fit_produce_pipeline_run_path)
        fitted_pipeline_path = self._get_fitted_pipeline_path(test_dir)
        self.assertTrue(os.stat(fitted_pipeline_path).st_size > 0)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'fit-produce', rerun_test_dir, input_run_path=fit_produce_pipeline_run_path,
            pipelines_path=test_dir
        )
        rerun_fit_produce_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(rerun_fit_produce_pipeline_run_path)

        self._assert_pipeline_runs_equal(fit_produce_pipeline_run_path, rerun_fit_produce_pipeline_run_path)

    def test_fit_score_rerun_with_hyperparams(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()

        fit_score_test_dir = self._get_unique_test_dir()
        pipeline_run_path = self._get_pipeline_run_path(fit_score_test_dir)
        scores_path = self._get_scores_path(fit_score_test_dir)

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()
        metrics = runtime.get_metrics_from_list(['ACCURACY', 'F1_MACRO'])
        scoring_params = {'add_normalized_scores': 'false'}
        scoring_random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, hyperparams=hyperparams,
                random_seed=random_seed, context=metadata_base.Context.TESTING,
            )
            self.assertFalse(fit_result.has_error(), fit_result.error)

            predictions, produce_result = runtime.produce(fitted_pipeline, inputs)
            self.assertFalse(produce_result.has_error(), produce_result.error)

            scores, score_result = runtime.score(
                predictions, inputs, scoring_pipeline=scoring_pipeline,
                problem_description=problem, metrics=metrics,
                predictions_random_seed=fitted_pipeline.random_seed,
                context=metadata_base.Context.TESTING, scoring_params=scoring_params, random_seed=scoring_random_seed
            )

            self.assertFalse(score_result.has_error(), score_result.error)
            scores.to_csv(scores_path)

            runtime.combine_pipeline_runs(
                produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=inputs,
                metrics=metrics, scores=scores
            )

        with open(pipeline_run_path, 'w') as f:
            fit_result.pipeline_run.to_yaml(f)
            produce_result.pipeline_run.to_yaml(f, appending=True)

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

        self._cache_pipeline_for_rerun(pipeline_path, fit_score_test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('fit-score', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=fit_score_test_dir)

        pipeline_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_path)
        rescores_path = self._get_scores_path(rerun_test_dir)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, pipeline_rerun_path)

    def test_fit_score_rerun_without_problem(self):
        test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run(
            'fit-score', test_dir, use_problem=False, cache_pipelines=True
        )
        fit_produce_pipeline_run_path = self._get_pipeline_run_path(test_dir)
        self._assert_valid_saved_pipeline_runs(fit_produce_pipeline_run_path)
        fitted_pipeline_path = self._get_fitted_pipeline_path(test_dir)
        self.assertTrue(os.stat(fitted_pipeline_path).st_size > 0)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'fit-score', rerun_test_dir, input_run_path=fit_produce_pipeline_run_path,
            pipelines_path=test_dir
        )
        rerun_fit_produce_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(rerun_fit_produce_pipeline_run_path)

        self._assert_pipeline_runs_equal(fit_produce_pipeline_run_path, rerun_fit_produce_pipeline_run_path)

    def test_evaluate_rerun_with_hyperparams(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        data_pipeline_path = self._get_train_test_split_data_pipeline_path()

        evaluate_test_dir = self._get_unique_test_dir()
        pipeline_run_path = self._get_pipeline_run_path(evaluate_test_dir)
        scores_path = self._get_scores_path(evaluate_test_dir)

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()
        metrics = runtime.get_metrics_from_list(['ACCURACY', 'F1_MACRO'])
        scoring_params = {'add_normalized_scores': 'false'}
        scoring_random_seed = self._generate_seed()
        data_params = {'shuffle': 'true', 'stratified': 'true', 'train_score_ratio': '0.59'}
        data_random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(data_pipeline_path) as f:
            data_pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            dummy_runtime_environment = pipeline_run_module.RuntimeEnvironment(worker_id='dummy worker id')

            all_scores, all_results = runtime.evaluate(
                pipeline, inputs, data_pipeline=data_pipeline, scoring_pipeline=scoring_pipeline,
                problem_description=problem, data_params=data_params, metrics=metrics,
                context=metadata_base.Context.TESTING, scoring_params=scoring_params,
                hyperparams=hyperparams, random_seed=random_seed,
                data_random_seed=data_random_seed, scoring_random_seed=scoring_random_seed,
                runtime_environment=dummy_runtime_environment,
            )

            self.assertEqual(len(all_scores), 1)
            scores = runtime.combine_folds(all_scores)
            scores.to_csv(scores_path)

            if any(result.has_error() for result in all_results):
                self.fail([result.error for result in all_results if result.has_error()][0])

        with open(pipeline_run_path, 'w') as f:
            for i, pipeline_run in enumerate(all_results.pipeline_runs):
                pipeline_run.to_yaml(f, appending=i>0)

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

        self._cache_pipeline_for_rerun(pipeline_path, evaluate_test_dir)
        self._cache_pipeline_for_rerun(data_pipeline_path, evaluate_test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('evaluate', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=evaluate_test_dir)

        pipeline_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_path)
        rescores_path = self._get_scores_path(rerun_test_dir)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, pipeline_rerun_path)

    def test_evaluate_rerun_without_problem(self):
        test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run(
            'evaluate', test_dir, use_problem=False, split_method='no-split', cache_pipelines=True
        )
        pipeline_run_path = self._get_pipeline_run_path(test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun(
            'evaluate', rerun_test_dir, input_run_path=pipeline_run_path,
            pipelines_path=test_dir
        )
        rerun_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        self._assert_valid_saved_pipeline_runs(rerun_pipeline_run_path)

        self._assert_pipeline_runs_equal(pipeline_run_path, rerun_pipeline_run_path)

    # See: https://gitlab.com/datadrivendiscovery/d3m/issues/406
    # TODO: Test rerun validation code (that we throw exceptions on invalid pipeline runs).

    def test_rerun_fit_validation_zero_pipeline_runs(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_path = self._get_pipeline_run_path()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            _, _, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem,
                context=metadata_base.Context.TESTING,
            )

        # Create pipeline run file and insert zero pipline run documents
        open(pipeline_run_path, 'w').close()

        self._cache_pipeline_for_rerun(pipeline_path)

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'fit',
            '--input-run',
            pipeline_run_path,
        ]

        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Pipeline run file must contain at least one pipeline run document.')

    def test_rerun_fit_validation_invalid_pipeline_run_document(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_path = self._get_pipeline_run_path()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            _, _, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem,
                context=metadata_base.Context.TESTING,
            )

        # Create invalid pipeline run file
        with open(pipeline_run_path, 'w') as f:
            f.write('Invalid')

        self._cache_pipeline_for_rerun(pipeline_path)

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'fit',
            '--input-run',
            pipeline_run_path,
        ]

        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Provided pipeline run document is not valid.')

    def test_rerun_fit_validation_greater_than_one_pipeline_runs(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_path = self._get_pipeline_run_path()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            _, _, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem,
                context=metadata_base.Context.TESTING,
            )

        # Create pipeline run file with more than one pipeline run document
        with open(pipeline_run_path, 'w') as f:
            fit_result.pipeline_run.to_yaml(f)
            fit_result.pipeline_run.to_yaml(f, appending=True)

        self._cache_pipeline_for_rerun(pipeline_path)

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'fit',
            '--input-run',
            pipeline_run_path,
        ]

        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Fit requires exactly one pipeline run. 2 provided.')

    def test_rerun_fit_validation_invalid_phase(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_path = self._get_pipeline_run_path()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            fitted_pipeline, _, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem,
                context=metadata_base.Context.TESTING,
            )

            _, produce_result = runtime.produce(fitted_pipeline, inputs)

        # Create pipeline run file with incorrect PHASE
        with open(pipeline_run_path, 'w') as f:
            produce_result.pipeline_run.to_yaml(f)

        self._cache_pipeline_for_rerun(pipeline_path)

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'fit',
            '--input-run',
            pipeline_run_path,
        ]

        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Fit requires a FIT phase pipeline run. PRODUCE phase provided.')

    def test_produce_rerun_validation_empty_pipeline_runs(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('produce', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path)
        produce_pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_pipeline_run_path)

        open(produce_pipeline_run_path, 'w').close()

        rerun_arg = [
            'd3m',
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'produce',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--input-run',
            produce_pipeline_run_path,
        ]

        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Pipeline run file must contain at least one pipeline run document.')

    def test_produce_rerun_validation_invalid_pipeline_run(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('produce', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path)
        produce_pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(produce_pipeline_run_path)

        with open(produce_pipeline_run_path, 'w') as f:
            f.write('Invalid')

        rerun_arg = [
            'd3m',
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'produce',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--input-run',
            produce_pipeline_run_path,
        ]

        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Provided pipeline run document is not valid.')

    def test_rerun_score_validation_empty_pipeline_run(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        score_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('score', score_test_dir, fitted_pipeline_path=fitted_pipeline_path)
        score_pipeline_run_path = self._get_pipeline_run_path(score_test_dir)
        self._assert_valid_saved_pipeline_runs(score_pipeline_run_path)

        open(score_pipeline_run_path, 'w').close()

        rerun_test_dir = self._get_unique_test_dir()
        rerun_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        rerun_arg = [
            'd3m',
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'score',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--input-run',
            score_pipeline_run_path,
            '--output-run',
            rerun_pipeline_run_path,
        ]
        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Pipeline run file must contain at least one pipeline run document.')

    def test_score_rerun_validation_invalid_pipeline_run(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, cache_pipelines=True)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        score_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('score', score_test_dir, fitted_pipeline_path=fitted_pipeline_path)
        score_pipeline_run_path = self._get_pipeline_run_path(score_test_dir)
        self._assert_valid_saved_pipeline_runs(score_pipeline_run_path)

        with open(score_pipeline_run_path, 'w') as f:
            f.write('Invalid')

        rerun_test_dir = self._get_unique_test_dir()
        rerun_pipeline_run_path = self._get_pipeline_run_path(rerun_test_dir)
        rerun_arg = [
            'd3m',
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'score',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--input-run',
            score_pipeline_run_path,
            '--output-run',
            rerun_pipeline_run_path,
        ]
        with self.assertRaises(exceptions.InvalidArgumentValueError) as ctx:
            self._call_cli_runtime(rerun_arg)
        self.assertEqual(str(ctx.exception), 'Provided pipeline run document is not valid.')

    # TODO: Test rerun with multiple inputs (non-standard pipeline).
    # TODO: Test evaluate rerun with data split file.

    def test_fit_with_data_preparation_pipeline(self):
        self._generate_pipeline_run('fit', split_method='train-test')
        output_path = self._get_output_path()
        self.assertTrue(os.path.isfile(output_path))

        # Count the number of rows generated
        with open(output_path, 'r') as f:
            self.assertEqual(len(list(f)), 113)

    def test_produce_with_data_preparation_pipeline(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir, split_method='train-test')
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run(
            'produce', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path, split_method='train-test'
        )
        output_path = self._get_output_path(produce_test_dir)

        with open(output_path, 'r') as f:
            self.assertEqual(len(list(f)), 39)

        # TODO: add more checks

    def test_score_with_data_preparation_pipeline(self):
        fit_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('fit', fit_test_dir)
        fitted_pipeline_path = self._get_fitted_pipeline_path(fit_test_dir)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        produce_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run(
            'score', produce_test_dir, fitted_pipeline_path=fitted_pipeline_path, split_method='train-test'
        )

        pipeline_run_path = self._get_pipeline_run_path(produce_test_dir)
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)

        scores_path = self._get_scores_path(produce_test_dir)
        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0], ['ACCURACY', 1.0, 1.0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_fit_produce_with_data_preparation_pipeline(self):
        self._generate_pipeline_run('fit-produce', split_method='train-test')
        output_path = self._get_output_path()
        self.assertTrue(os.path.isfile(output_path))

        # Count the number of rows generated
        with open(output_path, 'r') as f:
            self.assertEqual(len(list(f)), 39)

        # TODO: add more checks

    def test_fit_score_with_data_preparation_pipeline(self):
        self._generate_pipeline_run('fit-score', split_method='train-test')
        output_path = self._get_output_path()
        scores_path = self._get_scores_path()
        pipeline_run_path = self._get_pipeline_run_path()

        self.assertTrue(os.path.isfile(output_path))
        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertNotEqual(dataframe.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0], ['ACCURACY', 1.0, 1.0, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

        # Count the number of rows generated
        with open(output_path, 'r') as f:
            self.assertEqual(len(list(f)), 39)

    def test_score_predictions_with_data_preparation_pipeline(self):
        self._generate_pipeline_run('fit-produce', split_method='no-split')
        output_path = self._get_output_path()
        self.assertTrue(os.path.isfile(output_path))

        scores_path = self._get_scores_path()
        arg = [
            '',
            'runtime',
            'score-predictions',
            '--problem',
            self._get_iris_problem_path(),
            '--predictions',
            output_path,
            '--data-pipeline',
            self._get_no_split_data_pipeline_path() ,
            '--score-input',
            self._get_iris_dataset_path(),
            '--scores',
            scores_path,
            '--metric',
            'F1_MACRO',
            '--metric',
            'ACCURACY',
            '--predictions-random-seed',
            '0',
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0], ['ACCURACY', 1.0, 1.0, 0]])

    def _get_gzipped_pipeline_run_path(self, pipeline_run_path=None):
        if pipeline_run_path is None:
            pipeline_run_path = self._get_pipeline_run_path()
        return '{pipeline_run_path}.gz'.format(pipeline_run_path=pipeline_run_path)

    def test_validate_gzipped_pipeline_run(self):
        # First, generate the pipeline run file
        self._generate_pipeline_run('fit')
        pipeline_run_path = self._get_pipeline_run_path()
        gzip_pipeline_run_path = self._get_gzipped_pipeline_run_path(pipeline_run_path)

        # Second, gzip the pipeline run file
        with open(pipeline_run_path, 'rb') as file_in:
            with gzip.open(gzip_pipeline_run_path, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
        os.remove(pipeline_run_path)

        # Third, ensure that calling 'pipeline-run validate' on the gzipped pipeline run file is successful
        arg = [
            '',
            'pipeline-run',
            'validate',
            gzip_pipeline_run_path,
        ]
        self._call_cli_runtime_without_fail(arg)

    def test_help_message(self):
        arg = [
            '',
            'runtime',
            'fit',
            '--version',
        ]

        with io.StringIO() as buffer:
            with contextlib.redirect_stderr(buffer):
                with self.assertRaises(SystemExit):
                    cli.main(arg)

            help = buffer.getvalue()
            self.assertTrue('usage: d3m runtime fit' in help, help)

    @staticmethod
    def _get_cross_validation_data_pipeline_path():
        return contrib_pipelines.K_FOLD_TABULAR_SPLIT_PIPELINE_PATH

    @staticmethod
    def _get_random_classifier_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'random-classifier.yml')

    @staticmethod
    def _get_fake_scoring_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'fake_compute_score.yml')

    def test_evaluate_with_cross_validation(self):
        self._generate_pipeline_run('evaluate', split_method='cross-validation', number_of_folds=3, shuffle=True)
        pipeline_run_path = self._get_pipeline_run_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        scores_path = self._get_scores_path()
        scores_df = pandas.read_csv(scores_path)
        self.assertEqual(scores_df['fold'].tolist(), [0,0,1,1,2,2])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_evaluate_with_cross_validation_without_problem(self):
        self._generate_pipeline_run('evaluate', split_method='cross-validation', use_problem=False, number_of_folds=3, shuffle=True)
        pipeline_run_path = self._get_pipeline_run_path()
        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)
        scores_path = self._get_scores_path()
        scores_df = pandas.read_csv(scores_path)
        self.assertEqual(scores_df['fold'].tolist(), [0,1,2])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

    def test_evaluate_rerun_with_cross_validation(self):
        test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('evaluate', test_dir, split_method='cross-validation', number_of_folds=3, shuffle=True, cache_pipelines=True)
        pipeline_run_path = self._get_pipeline_run_path(test_dir)
        scores_path = self._get_scores_path(test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('evaluate', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=test_dir)
        pipeline_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        rescores_path = self._get_scores_path(rerun_test_dir)

        self._assert_valid_saved_pipeline_runs(pipeline_rerun_path)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(rescores_path, pipeline_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, pipeline_rerun_path)

    def test_evaluate_rerun_with_cross_validation_without_problem(self):
        test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('evaluate', test_dir, use_problem=False, split_method='cross-validation', number_of_folds=3, shuffle=True, cache_pipelines=True)
        pipeline_run_path = self._get_pipeline_run_path(test_dir)
        scores_path = self._get_scores_path(test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('evaluate', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=test_dir)
        pipeline_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        rescores_path = self._get_scores_path(rerun_test_dir)

        self._assert_valid_saved_pipeline_runs(pipeline_rerun_path)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(rescores_path, pipeline_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, pipeline_rerun_path)

    def test_evaluate_rerun_with_fixed_split(self):
        test_dir = self._get_unique_test_dir()
        self._generate_pipeline_run('evaluate', test_dir, split_method='fixed-split', cache_pipelines=True)
        pipeline_run_path = self._get_pipeline_run_path(test_dir)
        scores_path = self._get_scores_path(test_dir)

        rerun_test_dir = self._get_unique_test_dir()
        self._generate_pipeline_rerun('evaluate', rerun_test_dir, input_run_path=pipeline_run_path, pipelines_path=test_dir)
        pipeline_rerun_path = self._get_pipeline_run_path(rerun_test_dir)
        rescores_path = self._get_scores_path(rerun_test_dir)

        self._assert_valid_saved_pipeline_runs(pipeline_rerun_path)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(rescores_path, pipeline_rerun_path)
        self._assert_pipeline_runs_equal(pipeline_run_path, pipeline_rerun_path)

    @staticmethod
    def _get_multi_input_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'multi-input-test.json')

    @staticmethod
    def _get_non_standard_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'semi-standard-pipeline.json')

    def _get_fitted_pipeline_path(self, test_dir=None):
        if test_dir is None:
            test_dir = self.test_dir
        return os.path.join(test_dir, 'fitted-pipeline.pickle')

    def _get_test_pipeline_path(self, *, use_problem, use_standard_pipeline, use_multi_input) -> str:
        if use_standard_pipeline:
            if use_multi_input:
                return self._get_multi_input_pipeline_path()
            else:
                if use_problem:
                    # random forest pipeline needs to know the true target column set by problem doc
                    return self._get_random_forest_pipeline_path()
                else:
                    return self._get_random_classifier_pipeline_path()
        else:
            # TODO: create multi input non standard pipeline and add if/else multi_input
            return self._get_non_standard_pipeline_path()

    def _get_output_path(self, test_dir=None):
        if test_dir is None:
            test_dir = self.test_dir
        return os.path.join(test_dir, 'output.csv')

    def _get_exposed_outputs_path(self, test_dir=None, outputs_name='outputs.0'):
        if test_dir is None:
            test_dir = self.test_dir
        return os.path.join(test_dir, outputs_name, 'data.csv')

    def _get_outputs_metadata_path(self, test_dir=None, outputs_name='outputs.0'):
        if test_dir is None:
            test_dir = self.test_dir
        return os.path.join(test_dir, outputs_name, 'metadata.json')

    def _generate_pipeline_rerun(
        self, command: str, test_dir: str = None, *, input_run_path: str,
        pipelines_path: typing.Union[str, typing.List], fitted_pipeline_path: str = None
    ) -> typing.Sequence[logging.LogRecord]:
        """
        A utility method for rerunning pipeline runs.
        Uses the Iris dataset.

        Parameters
        ----------
        command
            Runtime command from ['fit', 'produce', 'score', 'fit-produce', 'fit-score', 'evaluate'].
        test_dir
            Base directory to write all output files, default of None resolves to self.test_dir.
        input_run_path
            The pipeline run to re-run.
        pipelines_path
            The directory where the pipeline to re-run resides.
        fitted_pipeline_path
            When command is one of ['produce', 'score'], a path to a fitted pipeline must be provided.

        Return
        ------
        logging_records
            Logging records generated by _call_cli_runtime_without_fail.
        """

        if test_dir is None:
            test_dir = self.test_dir

        if isinstance(pipelines_path, str):
            pipelines_path = [pipelines_path]

        args = ['d3m']

        for path in pipelines_path:
            args += ['--pipelines-path', path]

        args += [
            'runtime',
            '--datasets', TEST_DATA_DIR,
            command,
            '--input-run', input_run_path,
            '--output-run', self._get_pipeline_run_path(test_dir),
        ]

        # --fitted-pipeline
        if command in ['produce', 'score']:
            if fitted_pipeline_path is None:
                raise exceptions.InvalidArgumentValueError('{command} requires a fitted_pipeline_path'.format(command=command))
            args += ['--fitted-pipeline', fitted_pipeline_path]

        # --scores
        if command in ['score', 'fit-score', 'evaluate']:
            args += ['--scores', self._get_scores_path(test_dir)]

        # TODO: other output files

        return self._call_cli_runtime_without_fail(args)

    def test_prepare_data(self):
        data_preparation_pipeline_path = os.path.join(PIPELINE_DIR, 'data-preparation-train-test-split.yml')
        save_path = os.path.join(self.test_dir, 'prepared_data')

        arg = [
            '',
            'runtime',
            'prepare-data',
            '--problem',
            self._get_iris_problem_path(),
            '--input',
            self._get_iris_dataset_path(),
            '--save',
            save_path,
            '--data-pipeline',
            data_preparation_pipeline_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertTrue(os.path.isdir(save_path))

        dataset_doc = 'datasetDoc.json'
        table = os.path.join('tables', 'learningData.csv')
        dataset_dir = lambda directory: os.path.join(save_path, directory, 'dataset_{}'.format(directory))

        problem_dir = lambda directory: os.path.join(
            save_path, directory, 'problem_{}'.format(directory), 'problemDoc.json'
        )

        for split in ['TRAIN', 'TEST', 'SCORE']:
            # Check if learningdata is created
            self.assertEqual(os.path.isfile(os.path.join(dataset_dir(split), table)), True)

            # Check if datasetDoc is created
            self.assertEqual(os.path.isfile(os.path.join(dataset_dir(split), dataset_doc)), True)

            # Check for problemDoc is created
            self.assertEqual(os.path.isfile(problem_dir(split)), True)

    def test_prepare_data_fit_score(self):
        data_preparation_pipeline_path = os.path.join(PIPELINE_DIR, 'data-preparation-train-test-split.yml')
        save_path = os.path.join(self.test_dir, 'prepared_data')

        arg = [
            '',
            'runtime',
            'prepare-data',
            '--problem',
            self._get_iris_problem_path(),
            '--input',
            self._get_iris_dataset_path(),
            '--save',
            save_path,
            '--data-pipeline',
            data_preparation_pipeline_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertTrue(os.path.isdir(save_path))

        dataset_dir = lambda directory: os.path.join(
            save_path, directory, 'dataset_{}'.format(directory), 'datasetDoc.json'
        )

        train = dataset_dir('TRAIN')
        test = dataset_dir('TEST')
        score = dataset_dir('SCORE')
        pipeline_run_path = self._get_pipeline_run_path()

        scores_path = self._get_scores_path()
        arg = [
            '',
            'runtime',
            'fit-score',
            '--input',
            train,
            '--problem',
            os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json'),
            '--test-input',
            test,
            '--score-input',
            score,
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-forest-classifier.yml'),
            '--scores',
            scores_path,
            '--data-pipeline-run',
            os.path.join(save_path, 'data_preparation_pipeline_run.pkl'),
            '-O',
            pipeline_run_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 0.7631578947368421, 0.7631578947368421, 0]])
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_run_path)

        # Check that we have data_preparation field on the pipeline_run.
        have_data_preparation_field = False
        with open(pipeline_run_path, 'r') as f:
            for pipeline_run_dict in list(utils.yaml_load_all(f)):
                if 'data_preparation' in pipeline_run_dict['run']:
                    have_data_preparation_field = True
                    break

        self.assertEqual(have_data_preparation_field, True)


if __name__ == '__main__':
    unittest.main()
