import json
import os
import pickle
import shutil
import sys
import tempfile
import typing
import unittest

import jsonschema
import pandas
import numpy

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common_primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.redact_columns import RedactColumnsPrimitive
from common_primitives.train_score_split import TrainScoreDatasetSplitPrimitive
from common_primitives.random_forest import RandomForestClassifierPrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.no_split import NoSplitDatasetSplitPrimitive
from common_primitives.remove_columns import RemoveColumnsPrimitive
from common_primitives.simple_profiler import SimpleProfilerPrimitive
from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive
from common_primitives.kfold_split import KFoldDatasetSplitPrimitive

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')
sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.monomial import MonomialPrimitive
from test_primitives.random import RandomPrimitive
from test_primitives.sum import SumPrimitive
from test_primitives.increment import IncrementPrimitive, Hyperparams as IncrementHyperparams
from test_primitives.primitive_sum import PrimitiveSumPrimitive
from test_primitives.null import NullUnsupervisedLearnerPrimitive
from test_primitives.null import NullTransformerPrimitive
from test_primitives.random_classifier import RandomClassifierPrimitive
from test_primitives.fail import FailPrimitive
from test_primitives.data_hyperparam import DataHyperparamPrimitive
from test_primitives.abs_sum import AbsSumPrimitive
from test_primitives.container_hyperparam import ContainerHyperparamPrimitive
from test_primitives.multi_data_hyperparam import MultiDataHyperparamPrimitive
from test_primitives.primitive_hyperparam import PrimitiveHyperparamPrimitive

from d3m import container, exceptions, index, runtime, utils
from d3m.metadata import base as metadata_base, hyperparams, params, pipeline as pipeline_module, pipeline_run as pipeline_run_module, problem
from d3m.metadata.pipeline_run import PIPELINE_RUN_SCHEMA_VALIDATOR, PipelineRun, RuntimeEnvironment, _validate_pipeline_run_status_consistency, _validate_pipeline_run_random_seeds, _validate_pipeline_run_timestamps
from d3m.primitive_interfaces import base, transformer


TEST_PIPELINE_1 = """
{
    "created": "2018-11-05T04:14:02.720699Z",
    "id": "3ffcc6a0-313e-44ae-b551-2ade1386c11e",
    "inputs": [
        {
            "name": "inputs1"
        },
        {
            "name": "inputs2"
        },
        {
            "name": "inputs3"
        }
    ],
    "outputs": [
        {
            "data": "steps.1.produce",
            "name": "Metafeatures"
        }
    ],
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
    "steps": [
        {
            "arguments": {
                "inputs": {
                    "data": [
                        "inputs.0",
                        "inputs.1",
                        "inputs.2"
                    ],
                    "type": "CONTAINER"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ],
            "primitive": {
                "id": "8a8a8c15-bb69-488e-834c-f129de2dd2f6",
                "name": "Vertical Concatenate Primitive",
                "python_path": "d3m.primitives.data_transformation.vertical_concatenate.Test",
                "version": "0.1.0"
            },
            "type": "PRIMITIVE"
        },
        {
            "arguments": {
                "inputs": {
                    "data": "steps.0.produce",
                    "type": "CONTAINER"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ],
            "primitive": {
                "id": "aea7fc39-f40b-43ce-b926-89758e560e50",
                "name": "Voting Primitive",
                "python_path": "d3m.primitives.classification.voting.Test",
                "version": "0.1.0"
            },
            "type": "PRIMITIVE"
        }
    ]
}
"""


class Resolver(pipeline_module.Resolver):
    def _get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        # To hide any logging or stdout output.
        with utils.silence():
            return super()._get_primitive(primitive_description)


class Hyperparams(hyperparams.Hyperparams):
    pass


DataFramesInputs = container.List
DataFrameOutputs = container.DataFrame


class VerticalConcatenatePrimitive(transformer.TransformerPrimitiveBase[DataFramesInputs, DataFrameOutputs, Hyperparams]):
    """Description."""

    metadata = metadata_base.PrimitiveMetadata({
        'id': '8a8a8c15-bb69-488e-834c-f129de2dd2f6',
        'version': '0.1.0',
        'name': "Vertical Concatenate Primitive",
        'python_path': 'd3m.primitives.data_transformation.vertical_concatenate.Test',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.ARRAY_CONCATENATION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION
    })

    def produce(self, *, inputs: DataFramesInputs, timeout: float = None, iterations: int = None) -> base.CallResult[DataFrameOutputs]:
        for i in range(len(inputs)):
            if not inputs.metadata.has_semantic_type((i, metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'):
                raise Exception("Required metadata missing.")

        outputs = pandas.concat(inputs, ignore_index=True)
        outputs.metadata = outputs.metadata.generate(outputs)
        return base.CallResult(outputs)


VotingInputs = container.DataFrame
VotingOutputs = container.DataFrame


class VotingPrimitive(transformer.TransformerPrimitiveBase[VotingInputs, VotingOutputs, Hyperparams]):
    """Description."""

    metadata = metadata_base.PrimitiveMetadata({
        'id': 'aea7fc39-f40b-43ce-b926-89758e560e50',
        'version': '0.1.0',
        'name': "Voting Primitive",
        'python_path': 'd3m.primitives.classification.voting.Test',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.AGGREGATE_FUNCTION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION
    })

    def produce(self, *, inputs: VotingInputs, timeout: float = None, iterations: int = None) -> base.CallResult[VotingOutputs]:
        result = inputs.groupby('d3mIndex').apply(lambda x: x['class'].mode())
        result.columns = ['class']
        result = result.reset_index()
        return base.CallResult(container.DataFrame(result, generate_metadata=True))


def set_additionProperties_False(schema_json):
    if isinstance(schema_json, typing.Dict):
        if 'additionalProperties' in schema_json:
            schema_json['additionalProperties'] = False
        for key, value in schema_json.items():
            set_additionProperties_False(value)
    elif isinstance(schema_json, typing.List):
        for item in schema_json:
            set_additionProperties_False(item)


ExtraArgumentInputs = container.Dataset
ExtraArgumentOutputs = container.Dataset


class ExtraArgumentHyperparams(hyperparams.Hyperparams):
    pass


class ExtraArgumentParams(params.Params):
    pass


class TestRuntime(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @classmethod
    def setUpClass(cls):
        to_register = {
            'd3m.primitives.regression.monomial.Test': MonomialPrimitive,
            'd3m.primitives.data_generation.random.Test': RandomPrimitive,
            'd3m.primitives.operator.sum.Test': SumPrimitive,
            'd3m.primitives.operator.increment.Test': IncrementPrimitive,
            'd3m.primitives.operator.primitive_sum.Test': PrimitiveSumPrimitive,
            'd3m.primitives.classification.voting.Test': VotingPrimitive,
            'd3m.primitives.data_transformation.vertical_concatenate.Test': VerticalConcatenatePrimitive,
            'd3m.primitives.operator.null.FailTest': FailPrimitive,
            'd3m.primitives.operator.sum.ContainerHyperparamTest': ContainerHyperparamPrimitive,
            'd3m.primitives.operator.sum.DataHyperparamTest': DataHyperparamPrimitive,
            'd3m.primitives.operator.sum.MultiDataHyperparamTest': MultiDataHyperparamPrimitive,
            'd3m.primitives.operator.sum.PrimitiveHyperparamTest': PrimitiveHyperparamPrimitive,
            'd3m.primitives.operator.sum.AbsTest': AbsSumPrimitive,
            'd3m.primitives.operator.null.UnsupervisedLearnerTest': NullUnsupervisedLearnerPrimitive,
            'd3m.primitives.operator.null.TransformerTest': NullTransformerPrimitive,
            'd3m.primitives.data_transformation.dataset_to_dataframe.Common': DatasetToDataFramePrimitive,
            'd3m.primitives.classification.random_classifier.Test': RandomClassifierPrimitive,
            'd3m.primitives.evaluation.redact_columns.Common': RedactColumnsPrimitive,
            'd3m.primitives.evaluation.train_score_dataset_split.Common': TrainScoreDatasetSplitPrimitive,
            'd3m.primitives.classification.random_forest.Common': RandomForestClassifierPrimitive,
            'd3m.primitives.data_transformation.column_parser.Common': ColumnParserPrimitive,
            'd3m.primitives.data_transformation.construct_predictions.Common': ConstructPredictionsPrimitive,
            'd3m.primitives.evaluation.no_split_dataset_split.Common': NoSplitDatasetSplitPrimitive,
            'd3m.primitives.data_transformation.remove_columns.Common': RemoveColumnsPrimitive,
            'd3m.primitives.schema_discovery.profiler.Common': SimpleProfilerPrimitive,
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common': ExtractColumnsBySemanticTypesPrimitive,
            'd3m.primitives.evaluation.kfold_dataset_split.Common': KFoldDatasetSplitPrimitive,
        }

        # To hide any logging or stdout output.
        with utils.silence():
            for python_path, primitive in to_register.items():
                index.register_primitive(python_path, primitive)

            from common_primitives.dataset_map import DataFrameDatasetMapPrimitive

            # We have to do it here because it depends on other primitives being first registered.
            index.register_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon', DataFrameDatasetMapPrimitive)

            class ExtraArgumentPrimitive(transformer.TransformerPrimitiveBase[ExtraArgumentInputs, ExtraArgumentOutputs, ExtraArgumentHyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': '13d2174a-5502-4eeb-84b2-a12f0e7e8b6e',
                    'version': '0.1.0',
                    'name': "Extra Argument Primitive",
                    'python_path': 'd3m.primitives.test.ExtraArgumentPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: ExtraArgumentInputs, timeout: float = None, iterations: int = None) -> base.CallResult[ExtraArgumentOutputs]:
                    return base.CallResult(inputs)

                def produce_another(self, *, inputs: ExtraArgumentInputs, extra_data: ExtraArgumentInputs, timeout: float = None, iterations: int = None) -> base.CallResult[ExtraArgumentOutputs]:
                    return base.CallResult(inputs)

                def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: ExtraArgumentInputs, extra_data: ExtraArgumentInputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                    return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, extra_data=extra_data)

                def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: ExtraArgumentInputs, extra_data: ExtraArgumentInputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                    return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, extra_data=extra_data)

            index.register_primitive('d3m.primitives.test.ExtraArgumentPrimitive', ExtraArgumentPrimitive)

        # We create runtime environment ourselves so that it is done only once.
        with utils.silence():
            cls.runtime_enviroment = RuntimeEnvironment(
                worker_id='test',
                base_docker_image={
                    'image_name': 'test',
                    'image_digest': 'sha256:' + ('0' * 64),
                },
                docker_image={
                    'image_name': 'test',
                    'image_digest': 'sha256:' + ('0' * 64),
                },
            )

    def test_produce_memory_leak(self):
        # Produce method had a resource leak when certain errors occurred, e.g.,
        # temporary files were left around. This test is to prevent a regression.

        tmp_before = os.listdir("/tmp/")
        self.test_pipeline_run_failure()
        tmp_after = os.listdir("/tmp/")
        self.assertEqual(tmp_after, tmp_before)

    def test_basic(self):
        with open(os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'random-sample.yml'), 'r') as pipeline_file:
            p = pipeline_module.Pipeline.from_yaml(pipeline_file, resolver=Resolver())

        r = runtime.Runtime(p, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)

        inputs = [container.List([0, 1, 42], generate_metadata=True)]

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertTrue(result.pipeline_run)

        self.assertEqual(len(result.values), 1)

        dataframe = result.values['outputs.0']

        self.assertEqual(dataframe.values.tolist(), [
            [1.764052345967664 + 1],
            [0.4001572083672233 + 1],
            [-1.7062701906250126 + 1],
        ])

        result = r.produce(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)
        self.assertTrue(result.pipeline_run)

        dataframe = result.values['outputs.0']

        self.assertEqual(dataframe.values.tolist(), [
            [1.764052345967664 + 1],
            [0.4001572083672233 + 1],
            [-1.7062701906250126 + 1],
        ])

        pickled = pickle.dumps(r)
        restored = pickle.loads(pickled)

        result = restored.produce(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)
        self.assertTrue(result.pipeline_run)

        dataframe = result.values['outputs.0']

        self.assertEqual(dataframe.values.tolist(), [
            [1.764052345967664 + 1],
            [0.4001572083672233 + 1],
            [-1.7062701906250126 + 1],
        ])

        pickle.dumps(r)

        r = runtime.Runtime(p, random_seed=42, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)

        inputs = [container.List([0, 1, 42], generate_metadata=True)]

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)
        self.assertTrue(result.pipeline_run)

        dataframe = result.values['outputs.0']

        self.assertEqual(dataframe.values.tolist(), [
            [0.4967141530112327 + 1],
            [-0.13826430117118466 + 1],
            [-0.11564828238824053 + 1],
        ])

        r = runtime.Runtime(p, [{}, {'amount': 10}], random_seed=42, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)

        pickle.dumps(r)

        inputs = [container.List([0, 1, 42], generate_metadata=True)]

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)
        self.assertTrue(result.pipeline_run)

        dataframe = result.values['outputs.0']

        self.assertEqual(dataframe.values.tolist(), [
            [0.4967141530112327 + 10],
            [-0.13826430117118466 + 10],
            [-0.11564828238824053 + 10],
        ])

        pickle.dumps(r)

    def test_argument_list(self):
        p = pipeline_module.Pipeline.from_json(TEST_PIPELINE_1, resolver=Resolver())

        r = runtime.Runtime(p, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)

        inputs = [
            container.DataFrame({'d3mIndex': [1, 2, 3], 'class': [0, 0, 0]}, generate_metadata=True),
            container.DataFrame({'d3mIndex': [1, 2, 3], 'class': [0, 0, 1]}, generate_metadata=True),
            container.DataFrame({'d3mIndex': [1, 2, 3], 'class': [0, 1, 1]}, generate_metadata=True),
        ]

        for df in inputs:
            df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()
        dataframe = result.values['outputs.0']

        self.assertEqual(dataframe.values.tolist(), [[1, 0], [2, 0], [3, 1]])

        pickle.dumps(r)

    def test_pipeline_with_primitives_as_hyperparams_from_pipeline(self):
        # We create the pipeline.
        pipeline_description = pipeline_module.Pipeline()
        pipeline_description.add_input(name='input_0')
        pipeline_description.add_input(name='input_1')

        step_0_primitive = index.get_primitive('d3m.primitives.regression.monomial.Test')
        step_0_primitive_metadata = step_0_primitive.metadata.query()
        step_0_primitive_description = {
            'id': step_0_primitive_metadata['id'],
            'version': step_0_primitive_metadata['version'],
            'python_path': step_0_primitive_metadata['python_path'],
            'name': step_0_primitive_metadata['name'],
            'digest': step_0_primitive_metadata['digest'],
        }

        step_0 = pipeline_module.PrimitiveStep(primitive_description=step_0_primitive_description)
        step_0.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='inputs.0')
        step_0.add_argument(name='outputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='inputs.1')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        step_1_primitive = index.get_primitive('d3m.primitives.operator.primitive_sum.Test')
        step_1_primitive_metadata = step_1_primitive.metadata.query()
        step_1_primitive_description = {
            'id': step_1_primitive_metadata['id'],
            'version': step_1_primitive_metadata['version'],
            'python_path': step_1_primitive_metadata['python_path'],
            'name': step_1_primitive_metadata['name'],
            'digest': step_1_primitive_metadata['digest'],
        }

        step_1 = pipeline_module.PrimitiveStep(primitive_description=step_1_primitive_description)
        step_1.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='inputs.0')
        step_1.add_hyperparameter(name='primitive_1', argument_type=metadata_base.ArgumentType.PRIMITIVE, data=0)
        step_1.add_hyperparameter(name='primitive_2', argument_type=metadata_base.ArgumentType.PRIMITIVE, data=0)
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

        pipeline_description.add_output(name='output', data_reference='steps.1.produce')

        r = runtime.Runtime(pipeline_description, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)

        inputs = [container.List([1, 2, 3, 4, 5], generate_metadata=True), container.List([2, 4, 6, 8, 100], generate_metadata=True)]

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)

        results = result.values['outputs.0']

        self.assertEqual(results, [
            11.2,
            22.4,
            33.599999999999994,
            44.8,
            56.0,
        ])

        # Random seed should be different from 0 for hyper-parameter primitive instance.
        self.assertEqual(result.pipeline_run.steps[1].hyperparams['primitive_1'].random_seed, 1)
        # Primitive should not be the same instance.
        self.assertIsNot(result.pipeline_run.steps[1].hyperparams['primitive_1'], result.pipeline_run.steps[1].hyperparams['primitive_2'])

        result = r.produce(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)

        results = result.values['outputs.0']

        self.assertEqual(results, [
            11.2,
            22.4,
            33.599999999999994,
            44.8,
            56.0,
        ])

        pickle._dumps(r)

    def test_pipeline_with_primitives_as_hyperparams_as_class_value(self):
        # We create the pipeline.
        pipeline_description = pipeline_module.Pipeline()
        pipeline_description.add_input(name='input_0')

        null_primitive = index.get_primitive('d3m.primitives.operator.null.TransformerTest')

        step_0_primitive = index.get_primitive('d3m.primitives.operator.primitive_sum.Test')
        step_0_primitive_metadata = step_0_primitive.metadata.query()
        step_0_primitive_description = {
            'id': step_0_primitive_metadata['id'],
            'version': step_0_primitive_metadata['version'],
            'python_path': step_0_primitive_metadata['python_path'],
            'name': step_0_primitive_metadata['name'],
            'digest': step_0_primitive_metadata['digest'],
        }

        step_0 = pipeline_module.PrimitiveStep(primitive_description=step_0_primitive_description)
        step_0.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='inputs.0')
        step_0.add_hyperparameter(name='primitive_1', argument_type=metadata_base.ArgumentType.VALUE, data=null_primitive)
        step_0.add_hyperparameter(name='primitive_2', argument_type=metadata_base.ArgumentType.VALUE, data=null_primitive)
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        pipeline_description.add_output(name='output', data_reference='steps.0.produce')

        r = runtime.Runtime(pipeline_description, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)

        inputs = [container.List([1, 2, 3, 4, 5], generate_metadata=True)]

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)

        results = result.values['outputs.0']

        self.assertEqual(results, [
            2, 4, 6, 8, 10,
        ])

        # Primitive should not be the same instance.
        self.assertIsNot(result.pipeline_run.steps[0].hyperparams['primitive_1'], result.pipeline_run.steps[0].hyperparams['primitive_2'])

        result = r.produce(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)

        results = result.values['outputs.0']

        self.assertEqual(results, [
            2, 4, 6, 8, 10,
        ])

        pickle.dumps(r)

    def test_pipeline_with_primitives_as_hyperparams_as_instance_value(self):
        # We create the pipeline.
        pipeline_description = pipeline_module.Pipeline()
        pipeline_description.add_input(name='input_0')

        null_primitive = index.get_primitive('d3m.primitives.operator.null.TransformerTest')

        hyperparams_class = null_primitive.metadata.get_hyperparams()

        primitive = null_primitive(hyperparams=hyperparams_class.defaults())

        step_0_primitive = index.get_primitive('d3m.primitives.operator.primitive_sum.Test')
        step_0_primitive_metadata = step_0_primitive.metadata.query()
        step_0_primitive_description = {
            'id': step_0_primitive_metadata['id'],
            'version': step_0_primitive_metadata['version'],
            'python_path': step_0_primitive_metadata['python_path'],
            'name': step_0_primitive_metadata['name'],
            'digest': step_0_primitive_metadata['digest'],
        }

        step_0 = pipeline_module.PrimitiveStep(primitive_description=step_0_primitive_description)
        step_0.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='inputs.0')
        step_0.add_hyperparameter(name='primitive_1', argument_type=metadata_base.ArgumentType.VALUE, data=primitive)
        step_0.add_hyperparameter(name='primitive_2', argument_type=metadata_base.ArgumentType.VALUE, data=primitive)
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        pipeline_description.add_output(name='output', data_reference='steps.0.produce')

        r = runtime.Runtime(pipeline_description, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)

        inputs = [container.List([1, 2, 3, 4, 5], generate_metadata=True)]

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)

        results = result.values['outputs.0']

        self.assertEqual(results, [
            2, 4, 6, 8, 10,
        ])

        # Primitive should not be the same instance.
        self.assertIsNot(null_primitive, result.pipeline_run.steps[0].hyperparams['primitive_1'])
        self.assertIsNot(result.pipeline_run.steps[0].hyperparams['primitive_1'], result.pipeline_run.steps[0].hyperparams['primitive_2'])

        result = r.produce(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)

        results = result.values['outputs.0']

        self.assertEqual(results, [
            2, 4, 6, 8, 10,
        ])

        pickle.dumps(r)

    def _fake_inputs(self, runtime, pipeline_run, inputs):
        # We fake that inputs were added even if this is not a standard pipeline.
        # TODO: Make tests not require this.
        for input_dataset in inputs:
            pipeline_run.add_input_dataset(input_dataset)
        if runtime is not None:
            runtime._previous_pipeline_run_id = pipeline_run.get_id()

    def _build_pipeline(self, pipeline_id: str, sequence=None):
        if sequence is None:
            sequence = [{'primitive_class': RandomPrimitive}, {'primitive_class': IncrementPrimitive}]

        pipeline_description = {
            'source': {
                'name': 'Test team'
            },
            'name': 'Test pipeline',
            'description': 'Pipeline created to test pipeline-run'
        }

        pipe = pipeline_module.Pipeline(
            pipeline_id,
            source=pipeline_module.Pipeline._get_source(pipeline_description),
            name=pipeline_description['name'],
            description=pipeline_description['description'],
        )

        pipe.add_input('input_data')

        for index, element in enumerate(sequence):
            # default input, argument name is 'inputs', value specified below
            if index == 0:
                inputs = 'inputs.0'
            else:
                inputs = 'steps.{}.produce'.format(index - 1)

            if isinstance(element, pipeline_module.Pipeline):
                step = pipeline_module.SubpipelineStep(element.to_json_structure(nest_subpipelines=True))
                step.add_input(inputs)
            elif isinstance(element, dict):
                primitive_description = element['primitive_class'].metadata.query()
                step = pipeline_module.PrimitiveStep(primitive_description)
                if 'INPUTS' in element:
                    for arg_name, value in element['INPUTS']:
                        value_str = 'steps.{}.produce'.format(value)
                        step.add_argument(arg_name, metadata_base.ArgumentType.CONTAINER, value_str)
                else:
                    # if not specified, use default
                    step.add_argument('inputs', metadata_base.ArgumentType.CONTAINER, inputs)
                if 'HYPERPARAMS' in element:
                    for hyperparam_name in element['HYPERPARAMS']:
                        hyperparam = element['HYPERPARAMS'][hyperparam_name]
                        step.add_hyperparameter(hyperparam_name, hyperparam['TYPE'], hyperparam['DATA'])
            else:
                raise exceptions.InvalidArgumentTypeError(
                    'Unknown type {} in parameter \'sequence\''.format(type(element)))
            step.add_output('produce')
            pipe.add_step(step)

        pipe.add_output('steps.{}.produce'.format(len(sequence) - 1))

        return pipe

    def _get_inputs(self):
        # TODO: Make tests use a real Dataset instead of a list. Pipeline runs are defined on standard pipelines.
        input_data = container.List([1, 3, 4, 2, 5, 3], generate_metadata=True)
        # First have to add dummy metadata to the list, which otherwise exist in the dataset.
        input_data.metadata = input_data.metadata.update((), {
            'id': '0000000000000000000000000000000000000000000000000000000000000000',
            'digest': '0000000000000000000000000000000000000000000000000000000000000000'
        })
        inputs = [input_data]
        return inputs

    def _fit_pipeline(
        self, pipeline, inputs, problem_description=None, context=metadata_base.Context.TESTING, outputs_to_expose=None
    ):
        r = runtime.Runtime(
            pipeline, problem_description=problem_description, context=context,
            environment=self.runtime_enviroment,
        )
        fit_result = r.fit(inputs, outputs_to_expose=outputs_to_expose)
        self.assertTrue(fit_result.pipeline_run)
        # We fake that inputs were added even if this is not a standard pipeline.
        # TODO: Make tests not require this.
        for input_dataset in inputs:
            fit_result.pipeline_run.add_input_dataset(input_dataset)
        return fit_result.pipeline_run

    def _fit_and_produce_pipeline(
        self, pipeline, inputs, problem_description = None, context = metadata_base.Context.TESTING
    ):
        r = runtime.Runtime(
            pipeline, problem_description=problem_description, context=context,
            environment=self.runtime_enviroment,
        )
        fit_result = r.fit(inputs)
        self.assertTrue(fit_result.pipeline_run)
        self._fake_inputs(r, fit_result.pipeline_run, inputs)
        self._check_pipelines_valid_and_succeeded([fit_result.pipeline_run])

        produce_result = r.produce(inputs)
        self.assertTrue(produce_result.pipeline_run)
        self._fake_inputs(r, produce_result.pipeline_run, inputs)
        self._check_pipelines_valid_and_succeeded([produce_result.pipeline_run])

        return (fit_result.pipeline_run, produce_result.pipeline_run)

    def _is_pipeline_run_successful(self, pipeline_run_json):
        if pipeline_run_json['status']['state'] == metadata_base.PipelineRunStatusState.SUCCESS:
            return True
        elif pipeline_run_json['status']['state'] == metadata_base.PipelineRunStatusState.FAILURE:
            return False
        else:
            self.fail('Pipeline-run document status state set to invalid value')

    def _validate_pipeline_run_structure(self, json_structure):
        try:
            PIPELINE_RUN_SCHEMA_VALIDATOR.validate(json_structure)
            _validate_pipeline_run_status_consistency(json_structure)
            _validate_pipeline_run_timestamps(json_structure)
            _validate_pipeline_run_random_seeds(json_structure)
        except jsonschema.exceptions.ValidationError as error:
            print('\n', error, '\n')
            print("##### PRINTING RECURSIVE SUBERRORS #####\n")
            self.print_recursive_suberrors(error, indent='\n')
            self.fail("Pipeline_run document failed to validate against the schema")

    def _invalidate_pipeline_run_structure(self, json_structure):
        is_valid = False
        try:
            PIPELINE_RUN_SCHEMA_VALIDATOR.validate(json_structure)
            is_valid = True
        except jsonschema.exceptions.ValidationError as error:
            pass
        if is_valid:
            self.fail("Pipeline_run document should not have validated against the schema")

    def _check_pipelines_valid_and_succeeded(self, pipeline_runs):
        for pipeline_run in pipeline_runs:
            pipeline_run_json = pipeline_run.to_json_structure()
            self._validate_pipeline_run_structure(pipeline_run_json)
            self.assertTrue(self._is_pipeline_run_successful(pipeline_run_json), json.dumps(pipeline_run_json, indent=4))

    def _check_pipelines_valid_and_failed(self, pipeline_runs):
        for pipeline_run in pipeline_runs:
            pipeline_run_json = pipeline_run.to_json_structure()
            self._validate_pipeline_run_structure(pipeline_run_json)
            self.assertFalse(self._is_pipeline_run_successful(pipeline_run_json))

    def _check_pipelines_invalid(self, pipeline_runs):
        for pipeline_run in pipeline_runs:
            pipeline_run_json = pipeline_run.to_json_structure()
            self._invalidate_pipeline_run_structure(pipeline_run_json)

    def test_basic_pipeline_run(self):
        inputs = self._get_inputs()
        pipe = self._build_pipeline('1490432b-b48a-4a62-8977-5a56e52a3e85')
        pipeline_runs = self._fit_and_produce_pipeline(pipe, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

    def test_pipeline_fit_with_outputs_to_expose(self):
        inputs = self._get_inputs()
        pipe = self._build_pipeline('cf2e4f93-4b9a-4a49-9ab5-92927b3125df')
        pipeline_runs = self._fit_pipeline(pipe, inputs, outputs_to_expose=['steps.0.produce'])
        self._check_pipelines_valid_and_succeeded([pipeline_runs])

    def test_pipeline_run_failure(self):
        inputs = self._get_inputs()
        for hyperparam in ('__init__', 'set_training_data', 'fit', 'produce'):
            failure_pipeline = self._build_pipeline('18e96ab3-e3c5-4b29-a446-3e81982eba9c', sequence=[{'primitive_class': RandomPrimitive},
                                                                                                      {'primitive_class': FailPrimitive, 'HYPERPARAMS': {'method_to_fail': {'TYPE': metadata_base.ArgumentType.VALUE, 'DATA': hyperparam}}}])
            fit_pipeline_run = self._fit_pipeline(failure_pipeline, inputs)
            self._check_pipelines_valid_and_failed([fit_pipeline_run])

    def test_pipeline_run_failure_return_error(self):
        inputs = self._get_inputs()
        pipeline = self._build_pipeline('80dee50d-9ca4-4ad5-9a52-7ea30f3eb3e5', sequence=[{'primitive_class': RandomPrimitive},
                                                                                                  {'primitive_class': FailPrimitive, 'HYPERPARAMS': {'method_to_fail': {'TYPE': metadata_base.ArgumentType.VALUE, 'DATA': 'fit'}}}])
        r = runtime.Runtime(
            pipeline, context=metadata_base.Context.TESTING,
            environment=self.runtime_enviroment,
        )
        fit_result = r.fit(inputs)

        self.assertTrue(fit_result.error)
        self.assertEqual(str(fit_result.error), 'Step 1 for pipeline 80dee50d-9ca4-4ad5-9a52-7ea30f3eb3e5 failed.')
        self.assertIsInstance(fit_result.error, exceptions.StepFailedError)

        with self.assertRaises(exceptions.StepFailedError) as cm:
            fit_result.check_success()

        self.assertEqual(str(cm.exception), 'Step 1 for pipeline 80dee50d-9ca4-4ad5-9a52-7ea30f3eb3e5 failed.')

    def test_pipeline_run_failure_with_subpipeline(self):
        inputs = self._get_inputs()
        for hyperparam in ('__init__', 'set_training_data', 'fit', 'produce'):
            failure_subpipeline = self._build_pipeline('bcd96144-34ae-4a67-a1b5-b911a07d03ed', sequence=[{'primitive_class': FailPrimitive, 'HYPERPARAMS': {'method_to_fail': {'TYPE': metadata_base.ArgumentType.VALUE, 'DATA': hyperparam}}}])
            failure_pipeline = self._build_pipeline('cbec1cb2-64df-4d4a-81ea-a829eeac0612', sequence=[{'primitive_class': RandomPrimitive}, failure_subpipeline, {'primitive_class': IncrementPrimitive}])
            fit_pipeline_run = self._fit_pipeline(failure_pipeline, inputs)
            self._check_pipelines_valid_and_failed([fit_pipeline_run])

    # tests previous_pipeline_run when it should be None, and when it should be full
    def test_all_previous_pipeline_run_types(self):
        inputs = self._get_inputs()
        pipe = self._build_pipeline('2617ca0c-552a-4014-a999-2904184ed648')
        fit_pipeline_run, produce_pipeline_run = self._fit_and_produce_pipeline(pipe, inputs)
        self._check_pipelines_valid_and_succeeded([fit_pipeline_run, produce_pipeline_run])
        fit_pipeline_run_json = fit_pipeline_run.to_json_structure()
        self.assertTrue(
            'previous_pipeline_run' not in fit_pipeline_run_json,
            'pipeline_run should not contain previous_pipeline_run'
        )
        produce_pipeline_run_json = produce_pipeline_run.to_json_structure()
        self.assertNotEqual(produce_pipeline_run_json['previous_pipeline_run'], None)
        self.assertEqual(fit_pipeline_run_json['id'], produce_pipeline_run_json['previous_pipeline_run']['id'])

    # tests pipeline_run given each type of context
    def test_all_pipeline_run_context_types(self):
        inputs = self._get_inputs()
        pipe = self._build_pipeline('4fb64b4b-baa6-404a-afe3-1ad68a1993c1')

        for context in metadata_base.Context:
            pipeline_runs = self._fit_and_produce_pipeline(
                pipe, inputs, context=context
            )
            self._check_pipelines_valid_and_succeeded(pipeline_runs)

        class InvalidContext:
            def __init__(self, name):
                self.name = name

        invalid_context = InvalidContext('INVALID_CONTEXT')
        pipe = self._build_pipeline('1c05ae77-1f74-48bd-9341-c31338a9c9f0')
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            pipeline_runs = self._fit_and_produce_pipeline(pipe, inputs, context=invalid_context)

    # tests pipeline_run given primitive steps and given subpipeline steps
    def test_all_pipeline_run_step_types(self):
        inputs = self._get_inputs()

        pipeline_without_subpipeline = self._build_pipeline('dca8efbe-4daa-47a6-a811-9ca633ffc90b', [{'primitive_class': RandomPrimitive}, {'primitive_class': IncrementPrimitive}, {'primitive_class': IncrementPrimitive}, {'primitive_class': IncrementPrimitive}])
        pipeline_runs = self._fit_and_produce_pipeline(pipeline_without_subpipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

        subpipeline = self._build_pipeline('06dfb07a-f151-467c-9f1c-51a6bf6378a3', [{'primitive_class': IncrementPrimitive}, {'primitive_class': IncrementPrimitive}])
        pipeline_with_subpipeline = self._build_pipeline('293c1883-f81a-459d-a1a8-ba19467d5ad6', [{'primitive_class': RandomPrimitive}, subpipeline, {'primitive_class': IncrementPrimitive}])
        pipeline_runs = self._fit_and_produce_pipeline(pipeline_with_subpipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

    # tests when there is a subpipeline within a subpipeline
    def test_recursive_subpipeline(self):
        inputs = self._get_inputs()
        subpipeline = self._build_pipeline('1eba8278-45da-448e-92a8-a6daf780563f', [{'primitive_class': IncrementPrimitive}, {'primitive_class': IncrementPrimitive}])
        subpipeline = self._build_pipeline('b350beb3-4421-4627-906c-92cbbe900834', [{'primitive_class': IncrementPrimitive}, subpipeline, {'primitive_class': IncrementPrimitive}])
        pipeline_with_recursive_subpipeline = self._build_pipeline('17e3ae59-e132-4c56-8573-20be6f84ea05', [{'primitive_class': RandomPrimitive}, subpipeline, {'primitive_class': IncrementPrimitive}])
        pipeline_runs = self._fit_and_produce_pipeline(pipeline_with_recursive_subpipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

    def test_all_pipeline_run_hyperparam_types(self):
        inputs = self._get_inputs()

        # test value_argument hyperparams (runtime sets defaults)
        pipeline = self._build_pipeline('301702a9-cf1e-4332-9116-696c9908586a')
        pipeline_runs = self._fit_and_produce_pipeline(pipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

        # test container_argument
        pipeline = self._build_pipeline('8390ab6f-d619-4cc5-b343-22b91f81eecd', sequence=[{'primitive_class': RandomPrimitive},
                                                                                          {'primitive_class': ContainerHyperparamPrimitive, 'HYPERPARAMS': {'dataframe': {'TYPE': metadata_base.ArgumentType.CONTAINER, 'DATA': 'steps.0.produce'}}}])
        pipeline_runs = self._fit_and_produce_pipeline(pipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

        # test data_argument
        pipeline = self._build_pipeline('f0e0e370-97db-4e67-9eff-5e9b79f253e6', sequence=[{'primitive_class': RandomPrimitive}, {'primitive_class': AbsSumPrimitive},
                                                                                          {'primitive_class': DataHyperparamPrimitive, 'INPUTS': [('inputs', 0)], 'HYPERPARAMS': {'value': {'TYPE': metadata_base.ArgumentType.DATA, 'DATA': 'steps.1.produce'}}}])
        pipeline_runs = self._fit_and_produce_pipeline(pipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

        # test data_arguments
        pipeline = self._build_pipeline('ab71ff74-5cd1-4e36-8c63-c2cd79085173', sequence=[{'primitive_class': RandomPrimitive}, {'primitive_class': AbsSumPrimitive}, {'primitive_class': AbsSumPrimitive, 'INPUTS': [('inputs', 0)]},
                                                                                          {'primitive_class': MultiDataHyperparamPrimitive, 'INPUTS': [('inputs', 0)], 'HYPERPARAMS': {'values': {'TYPE': metadata_base.ArgumentType.DATA, 'DATA': ['steps.1.produce', 'steps.2.produce']}}}])
        pipeline_runs = self._fit_and_produce_pipeline(pipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

        # test primitive argument
        pipeline = self._build_pipeline('c8b291f1-ff67-49e0-b8a3-a0e6a2d6f013', sequence=[{'primitive_class': RandomPrimitive}, {'primitive_class': AbsSumPrimitive},
                                                                                          {'primitive_class': PrimitiveHyperparamPrimitive, 'INPUTS': [('inputs', 0)], 'HYPERPARAMS': {'primitive': {'TYPE': metadata_base.ArgumentType.PRIMITIVE, 'DATA': 1}}}])
        pipeline_runs = self._fit_and_produce_pipeline(pipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

    def test_all_pipeline_run_method_call_base_metadata_types(self):
        pipeline = pipeline_module.Pipeline.from_json(TEST_PIPELINE_1, resolver=Resolver())
        pipeline_run = PipelineRun(
            pipeline, phase=metadata_base.PipelineRunPhase.FIT, context=metadata_base.Context.TESTING,
            environment=self.runtime_enviroment, random_seed=0
        )
        inputs = self._get_inputs()[0]
        pipeline_run.add_input_dataset(inputs)
        pipeline_run.run_started()
        pipeline_run.step_started(0)
        primitive_step_id = pipeline_run.add_primitive_step(pipeline.steps[0])
        method_call_id = pipeline_run.add_method_call_to_primitive_step(primitive_step_id, 'fit')
        pipeline_run.method_call_started(method_call_id)
        result = base.CallResult(inputs)
        pipeline_run.method_call_successful(method_call_id)
        pipeline_run.set_method_call_result_metadata(method_call_id, result)
        pipeline_run.step_successful(primitive_step_id)
        pipeline_run.run_successful()
        self._validate_pipeline_run_structure(pipeline_run.to_json_structure())

    # test that the phase is set correctly for fit and produce
    def test_all_pipeline_run_phase_types(self):
        inputs = self._get_inputs()
        pipeline = self._build_pipeline('d95a9816-8ede-4fe2-89c5-f5c9d9f1d9fd')
        pipeline_runs = self._fit_and_produce_pipeline(pipeline, inputs)
        self._check_pipelines_valid_and_succeeded(pipeline_runs)

        fit_pipeline_run = pipeline_runs[0]
        fit_pipeline_run_json = fit_pipeline_run.to_json_structure()
        self.assertEqual(fit_pipeline_run_json['run']['phase'], 'FIT')

        produce_pipeline_run = pipeline_runs[1]
        produce_pipeline_run_json = produce_pipeline_run.to_json_structure()
        self.assertEqual(produce_pipeline_run_json['run']['phase'], 'PRODUCE')

    # tests that the first method_call of each step is __init__()
    def test_pipeline_run_init_method_calls(self):
        inputs = self._get_inputs()
        pipeline = self._build_pipeline('5a9321df-7e40-443b-9e12-f1d840a677cd')
        pipeline_runs = self._fit_and_produce_pipeline(pipeline, inputs)
        for pipeline_run in pipeline_runs:
            pipeline_run_json = pipeline_run.to_json_structure()
            if pipeline_run_json['run']['phase'] == 'FIT':
                for step in pipeline_run_json['steps']:
                    first_method_call = step['method_calls'][0]
                    self.assertEqual(first_method_call['name'], '__init__')

    def print_recursive_suberrors(self, error, indent):
        for suberror in sorted(error.context, key=lambda e: e.schema_path):
            print(f'{indent}', list(suberror.schema_path), ", ", suberror.message)
            self.print_recursive_suberrors(suberror, indent + '\t')

    def get_data(self, dataset_name='iris_dataset_1', problem_name='iris_problem_1'):
        if problem_name:
            problem_doc_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 'data', 'problems', problem_name, 'problemDoc.json'
            ))
            problem_description = problem.Problem.load(utils.path_to_uri(problem_doc_path))
        else:
            problem_description = None

        datasetDoc_path = utils.path_to_uri(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', dataset_name, 'datasetDoc.json')))
        iris_dataset = container.Dataset.load(datasetDoc_path)
        return problem_description, iris_dataset

    def test_recording_hyperparams(self):
        pipeline = self._build_pipeline(
            '84d5dbb8-6e82-4187-801e-83a46069608f',
            sequence=[
                {
                    'primitive_class': IncrementPrimitive
                },
                {
                    'primitive_class': IncrementPrimitive,
                    'HYPERPARAMS': {
                        'amount': {
                            'TYPE': metadata_base.ArgumentType.VALUE,
                            'DATA': 3.14
                        }
                    }
                },
                {
                    'primitive_class': IncrementPrimitive
                }
            ],
        )
        runtime_hyperparams = [{}, {}, {'amount': 2.72}]
        inputs = [container.DataFrame({'a': [1,2,3], 'b': [3,5,8]}, generate_metadata=True)]
        # TODO: Make tests use a real Dataset instead of a dataframe. Pipeline runs are defined on standard pipelines.
        # First have to add dummy metadata to the dataframe, which otherwise exist in the dataset.
        inputs[0].metadata = inputs[0].metadata.update((), {
            'id': '0000000000000000000000000000000000000000000000000000000000000000',
            'digest': '0000000000000000000000000000000000000000000000000000000000000000'
        })
        r = runtime.Runtime(pipeline, runtime_hyperparams, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)
        fit_result = r.fit(inputs=inputs)
        self._fake_inputs(r, fit_result.pipeline_run, inputs)
        fit_pipeline_run_json = fit_result.pipeline_run.to_json_structure()

        # test default hyperparams recorded in pipeline_run
        self.assertTrue(
            'amount' in fit_pipeline_run_json['steps'][0]['hyperparams'],
            'default hyperparams not recorded in pipeline_run'
        )
        self.assertEqual(
            IncrementHyperparams.defaults().values_to_json_structure()['amount'],
            fit_pipeline_run_json['steps'][0]['hyperparams']['amount']['data'],
            'defualt hyperparams incorrectly recorded in pipeline_run'
        )

        # test hyperparams specified in pipeline not recored in pipeline_run
        self.assertFalse(
            'hyperparams' in fit_pipeline_run_json['steps'][1],
            'hyperparams specified in the pipeline should not be recorded in the pipeline_run'
        )

        # test hyperparams set at runtime recored in pipeline_run
        self.assertTrue(
            'amount' in fit_pipeline_run_json['steps'][2]['hyperparams'],
            'runtime hyperparams not recorded in pipeline_run'
        )
        self.assertEqual(
            runtime_hyperparams[2]['amount'],
            fit_pipeline_run_json['steps'][2]['hyperparams']['amount']['data'],
            'defualt hyperparams incorrectly recorded in pipeline_run'
        )

        produce_result = r.produce(inputs=inputs)
        self._fake_inputs(r, produce_result.pipeline_run, inputs)
        for step in produce_result.pipeline_run.to_json_structure()['steps']:
            self.assertFalse(
                'hyperparams' in step,
                'hyperparams should not be set in produce pipeline_runs'
            )

    def test_recording_arguments(self):
        pipeline = self._build_pipeline('46bb32a5-f9a0-4c33-97c8-f426ed147e0a')
        inputs = self._get_inputs()
        r = runtime.Runtime(pipeline, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)
        fit_result = r.fit(inputs=inputs)
        self._fake_inputs(r, fit_result.pipeline_run, inputs)
        fit_pipeline_run_json = fit_result.pipeline_run.to_json_structure()

        pipeline_json_structure = pipeline.to_json_structure()
        for pipeline_step, pipeline_run_step in zip(pipeline_json_structure['steps'], fit_pipeline_run_json['steps']):
            if 'arguments' in pipeline_run_step:
                for argument_name in pipeline_step['arguments']:
                    self.assertFalse(
                        argument_name in pipeline_run_step['arguments'],
                        'pipeline step arguments should not be recorded in pipeline_run method_call arguments'
                    )

        produce_result = r.produce(inputs=inputs)
        self._fake_inputs(r, produce_result.pipeline_run, inputs)
        produce_pipeline_run_json = produce_result.pipeline_run.to_json_structure()

        for pipeline_step, pipeline_run_step in zip(pipeline_json_structure['steps'], produce_pipeline_run_json['steps']):
            if 'arguments' in pipeline_run_step:
                for argument_name in pipeline_step['arguments']:
                    self.assertFalse(
                        argument_name in pipeline_run_step['arguments'],
                        'pipeline step arguments should not be recorded in pipeline_run method_call arguments'
                    )

    def test_saving_to_file(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        inputs = self._get_inputs()
        pipeline = self._build_pipeline('4327ce61-0580-48b3-9aeb-d3e35c09376d')

        r = runtime.Runtime(pipeline, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment)
        fit_result = r.fit(inputs=inputs)
        self._fake_inputs(r, fit_result.pipeline_run, inputs)
        fit_pipeline_run = fit_result.pipeline_run
        fit_pipeline_run_json = fit_pipeline_run.to_json_structure()
        fit_file_name = '{}.json'.format(fit_pipeline_run_json['id'])
        fit_file_path = os.path.join(self.test_dir, fit_file_name)
        with open(fit_file_path, 'w') as fit_file:
            fit_pipeline_run.to_yaml(fit_file)
        self.assertTrue(os.path.exists(fit_file_path), 'The fit pipeline_run object should have been saved to {}'.format(fit_file_path))
        with open(fit_file_path, 'r') as fit_file:
            fit_json = utils.yaml_load(fit_file)
        self._validate_pipeline_run_structure(fit_json)
        self.assertEqual(fit_json['id'], fit_pipeline_run_json['id'])
        self.assertEqual(len(fit_json['steps']), len(fit_pipeline_run.steps))
        self.assertEqual(fit_json['status'], fit_pipeline_run.status)

        produce_result = r.produce(inputs=inputs)
        self._fake_inputs(r, produce_result.pipeline_run, inputs)
        produce_pipeline_run = produce_result.pipeline_run
        produce_pipeline_run_json = produce_pipeline_run.to_json_structure()
        fit_produce_file_name = 'produce_pipeline.json'
        fit_produce_file_path = os.path.join(self.test_dir, fit_produce_file_name)
        with open(fit_produce_file_path, 'w') as fit_produce_file:
            fit_pipeline_run.to_yaml(fit_produce_file)
            produce_pipeline_run.to_yaml(fit_produce_file, appending=True)
        self.assertTrue(os.path.exists(fit_produce_file_path), 'The fit and produce pipeline_run objects should have been saved to {}'.format(fit_produce_file_path))
        with open(fit_produce_file_path, 'r') as fit_produce_file:
            fit_produce_jsons = list(utils.yaml_load_all(fit_produce_file))
        self.assertIsInstance(fit_produce_jsons, typing.Sequence, 'The fit_produce_file should contain a sequence of pipeline_run objects')
        self.assertEqual(len(fit_produce_jsons), 2, 'The fit_produce_file should contain 2 pipeline_run objects')
        fit_json = fit_produce_jsons[0]
        self._validate_pipeline_run_structure(fit_json)
        self.assertEqual(fit_json['id'], fit_pipeline_run_json['id'])
        self.assertEqual(len(fit_json['steps']), len(fit_pipeline_run.steps))
        self.assertEqual(fit_json['status'], fit_pipeline_run.status)
        produce_json = fit_produce_jsons[1]
        self._validate_pipeline_run_structure(produce_json)
        self.assertEqual(produce_json['id'], produce_pipeline_run_json['id'])
        self.assertEqual(len(produce_json['steps']), len(produce_pipeline_run.steps))
        self.assertEqual(produce_json['status'], produce_pipeline_run.status)

    def test_fit(self):
        pipeline = self._build_pipeline(
            '6e79c2cc-e36d-4f22-9016-8184d3385714',
            sequence=[
                {
                    'primitive_class': DatasetToDataFramePrimitive,
                },
                {
                    'primitive_class': RandomClassifierPrimitive,
                    'INPUTS': [('inputs', 0), ('outputs', 0)],
                },
            ],
        )
        iris_problem, iris_dataset = self.get_data()
        inputs = [iris_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment,
        )
        self._validate_pipeline_run_structure(fit_result.pipeline_run.to_json_structure())

    def test_prepare_data(self):
        with open(
            os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'data-preparation-no-split.yml'),
            'r',
        ) as data_pipeline_file:
            data_pipeline = pipeline_module.Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())

        with open(
            os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'random-forest-classifier.yml'),
            'r',
        ) as data_pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())

        iris_problem, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='iris_problem_1')
        inputs = [iris_dataset]
        outputs, data_result = runtime.prepare_data(
            data_pipeline=data_pipeline, problem_description=iris_problem, inputs=inputs,
            data_params={}, context=metadata_base.Context.TESTING, runtime_environment=self.runtime_enviroment)

        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment,
        )
        self.assertFalse(fit_result.has_error(), fit_result.error)
        self.assertFalse(data_result.has_error(), data_result.error)

        self._validate_pipeline_run_structure(data_result.pipeline_run.to_json_structure())

        runtime.combine_pipeline_runs(
            fit_result.pipeline_run, data_pipeline_run=data_result.pipeline_run,
        )
        self.assertFalse(fit_result.has_error(), fit_result.error)
        self.assertEqual(len(outputs), 3)
        self._validate_pipeline_run_structure(fit_result.pipeline_run.to_json_structure())

    def test_save_prepare_data(self):
        with open(
            os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'data-preparation-no-split.yml'),
            'r',
        ) as data_pipeline_file:
            data_pipeline = pipeline_module.Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())

        iris_problem, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='iris_problem_1')
        inputs = [iris_dataset]

        save_path = os.path.join(self.test_dir, 'iris_dataset_1')
        runtime.prepare_data_and_save(
            save_dir=save_path, data_pipeline=data_pipeline, problem_description=iris_problem, inputs=inputs, data_params={},
            context=metadata_base.Context.TESTING, runtime_environment=self.runtime_enviroment,
        )

        pipeline_run_path = os.path.join(save_path, 'data_preparation_pipeline_run.pkl')

        # Check if the pipeline run exists.
        self.assertEqual(os.path.isfile(pipeline_run_path), True)

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
            self.assertEqual(os.path.isfile(os.path.join(problem_dir(split))), True)

    def test_evaluate_with_prepared_data(self):
        with open(
            os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'data-preparation-tabular-cross-validation.yml'),
            'r',
        ) as data_pipeline_file:
            data_pipeline = pipeline_module.Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())

        with open(
            os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'random-forest-classifier.yml'),
            'r',
        ) as data_pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())

        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        iris_problem, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='iris_problem_1')
        inputs = [iris_dataset]
        data_params = {
            'number_of_folds': '2',
            'stratified': 'false',
            'shuffle': 'true',
        }

        save_path = os.path.join(self.test_dir, 'iris_dataset_1')
        runtime.prepare_data_and_save(
            save_dir=save_path, data_pipeline=data_pipeline, problem_description=iris_problem, inputs=inputs,
            data_params=data_params, context=metadata_base.Context.TESTING, runtime_environment=self.runtime_enviroment,
        )

        all_scores, all_results = runtime.evaluate_with_prepared_data(
            pipeline=pipeline, inputs_dir=save_path, scoring_pipeline=scoring_pipeline, problem_description=iris_problem,
            metrics=[{'metric': problem.PerformanceMetric.ACCURACY}, {'metric': problem.PerformanceMetric.F1_MACRO}],
            context=metadata_base.Context.TESTING, runtime_environment=self.runtime_enviroment,
        )

        all_results.check_success()
        self.assertEqual(len(all_scores), 2)
        self.assertEqual(list(all_scores[0].columns), ['metric', 'value', 'normalized', 'randomSeed'])

        self._check_pipelines_valid_and_succeeded(all_results.pipeline_runs)

    def test_compare_evaluate_with_prepared_data_with_direct_execution(self):
        with open(
            os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'data-preparation-train-test-split.yml'),
            'r',
        ) as data_pipeline_file:
            data_pipeline = pipeline_module.Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())

        with open(
            os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'random-forest-classifier.yml'),
            'r',
        ) as data_pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())

        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        iris_problem, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='iris_problem_1')
        inputs = [iris_dataset]
        data_params = {
            'shuffle': 'true',
        }

        evaluate_scores, all_results = runtime.evaluate(
            pipeline=pipeline, inputs=inputs, data_pipeline=data_pipeline, scoring_pipeline=scoring_pipeline, problem_description=iris_problem,
            data_params=data_params,
            metrics=[{'metric': problem.PerformanceMetric.ACCURACY}, {'metric': problem.PerformanceMetric.F1_MACRO}],
            context=metadata_base.Context.TESTING, runtime_environment=self.runtime_enviroment,
        )
        all_results.check_success()
        self.assertEqual(len(evaluate_scores), 1)

        evaluate_pipeline_runs = all_results.pipeline_runs
        self._check_pipelines_valid_and_succeeded(evaluate_pipeline_runs)
        self.assertEqual(len(evaluate_pipeline_runs), 2)
        self.assertTrue(evaluate_pipeline_runs[0].run.get('data_preparation', None))
        self.assertTrue(evaluate_pipeline_runs[1].run.get('data_preparation', None))

        save_path = os.path.join(self.test_dir, 'iris_dataset_1_evaluate')
        runtime.prepare_data_and_save(
            save_dir=save_path, data_pipeline=data_pipeline, problem_description=iris_problem, inputs=inputs,
            data_params=data_params, context=metadata_base.Context.TESTING, runtime_environment=self.runtime_enviroment,
        )

        evaluate_with_prepared_data_scores, all_results = runtime.evaluate_with_prepared_data(
            pipeline=pipeline, inputs_dir=save_path, scoring_pipeline=scoring_pipeline, problem_description=iris_problem,
            metrics=[{'metric': problem.PerformanceMetric.ACCURACY}, {'metric': problem.PerformanceMetric.F1_MACRO}],
            context=metadata_base.Context.TESTING, runtime_environment=self.runtime_enviroment,
        )
        all_results.check_success()
        self.assertEqual(len(evaluate_with_prepared_data_scores), 1)

        evaluate_with_prepared_data_pipeline_runs = all_results.pipeline_runs
        self._check_pipelines_valid_and_succeeded(evaluate_with_prepared_data_pipeline_runs)
        self.assertEqual(len(evaluate_with_prepared_data_pipeline_runs), 2)

        self.assertEqual(evaluate_scores[0].values.tolist(), evaluate_with_prepared_data_scores[0].values.tolist())
        self.assertTrue(evaluate_with_prepared_data_pipeline_runs[0].run.get('data_preparation', None))
        self.assertTrue(evaluate_with_prepared_data_pipeline_runs[1].run.get('data_preparation', None))

        self.assertEqual(evaluate_with_prepared_data_pipeline_runs[0].datasets, evaluate_pipeline_runs[0].datasets)
        self.assertEqual(evaluate_with_prepared_data_pipeline_runs[1].datasets, evaluate_pipeline_runs[1].datasets)

        for pipeline_run1, pipeline_run2 in zip(evaluate_with_prepared_data_pipeline_runs, evaluate_pipeline_runs):
            self.assertTrue(pipeline_run_module.PipelineRun.json_structure_equals(pipeline_run1.to_json_structure(), pipeline_run2.to_json_structure()))

    def test_multi_input_fit(self):
        with open(
                os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'multi-input-test.json'), 'r'
        ) as pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_json(pipeline_file, resolver=Resolver())

        iris_problem, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='multi_dataset_problem')
        _, boston_dataset = self.get_data(dataset_name='boston_dataset_1', problem_name='')
        inputs = [iris_dataset, boston_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment,
        )
        self._validate_pipeline_run_structure(fit_result.pipeline_run.to_json_structure())

    def test_multi_input_fit_without_problem(self):
        with open(
                os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'multi-input-test.json'), 'r'
        ) as pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_json(pipeline_file, resolver=Resolver())

        _, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='')
        _, boston_dataset = self.get_data(dataset_name='boston_dataset_1', problem_name='')
        inputs = [iris_dataset, boston_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        r = runtime.Runtime(pipeline, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment,
                            hyperparams=hyperparams, random_seed=random_seed, volumes_dir=volumes_dir)
        r.fit(inputs=inputs).check_success()

    def test_multi_input_fit_with_one_dataset_associated(self):
        with open(
                os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'multi-input-test.json'), 'r'
        ) as pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_json(pipeline_file, resolver=Resolver())
        _, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='')
        boston_problem, boston_dataset = self.get_data(dataset_name='boston_dataset_1', problem_name='boston_problem_1')
        inputs = [iris_dataset, boston_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        r = runtime.Runtime(pipeline, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment,
                            hyperparams=hyperparams, random_seed=random_seed, volumes_dir=volumes_dir,
                            problem_description=boston_problem)
        r.fit(inputs=inputs).check_success()

    def test_produce(self):
        pipeline = self._build_pipeline(
            'c99ae185-2a74-4919-88b1-66d02e2e21b2',
            sequence=[
                {
                    'primitive_class': DatasetToDataFramePrimitive
                },
                {
                    'primitive_class': RandomClassifierPrimitive,
                    'INPUTS': [('inputs', 0), ('outputs', 0)],
                },
            ],
        )
        iris_problem, iris_dataset = self.get_data()
        inputs = [iris_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment,
        )
        predictions, produce_result = runtime.produce(fitted_pipeline, inputs)
        self._validate_pipeline_run_structure(produce_result.pipeline_run.to_json_structure())

    def test_expose_outputs_fit(self):
        pipeline = self._build_pipeline(
            '5ce0a679-e058-41a0-a8c0-8999b073466e',
            sequence=[
                {
                    'primitive_class': DatasetToDataFramePrimitive
                },
                {
                    'primitive_class': ColumnParserPrimitive
                },
                {
                    'primitive_class': ExtractColumnsBySemanticTypesPrimitive,
                    'INPUTS': [('inputs', 1)],
                    'HYPERPARAMS': {
                        'semantic_types': {
                            'TYPE': metadata_base.ArgumentType.VALUE,
                            'DATA': ['https://metadata.datadrivendiscovery.org/types/Attribute']
                        },
                        'use_columns': {
                            'TYPE': metadata_base.ArgumentType.VALUE,
                            'DATA': [1, 2, 3, 4]
                        }
                    }
                },
                {
                    'primitive_class': ExtractColumnsBySemanticTypesPrimitive,
                    'INPUTS': [('inputs', 1)],
                    'HYPERPARAMS': {'semantic_types': {
                        'TYPE': metadata_base.ArgumentType.VALUE,
                        'DATA': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']
                    }}
                },
                {
                    'primitive_class': RandomForestClassifierPrimitive,
                    'INPUTS': [('inputs', 2), ('outputs', 3)],
                },
            ],
        )

        iris_problem, iris_dataset = self.get_data()
        inputs = [iris_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None

        # Case boolean True
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, expose_produced_outputs=True
        )

        self.assertEqual(
            list(fit_result.values.keys()),
            ['outputs.0', 'steps.0.produce', 'steps.1.produce', 'steps.2.produce', 'steps.3.produce', 'steps.4.produce']
        )

        # Case boolean False
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, expose_produced_outputs=False
        )

        self.assertEqual(list(fit_result.values.keys()), ['outputs.0'])

        # Check exposing values from invalid produce methods.
        expose_produced_outputs = ['steps.4.produce', 'steps.4.produce_feature_importances', 'steps.1.produce']
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, outputs_to_expose=expose_produced_outputs,
        )

        for expose_output in expose_produced_outputs:
            self.assertIn(expose_output, fit_result.values)

        # Check for different cases such as index value out of range, random method call and that other methods that are not produced.
        cases_expose_produced_outputs = [
            ['steps.5.produce'], ['steps.1.random_method'], ['steps.4.log_likelihoods'], ['test'], ['test.test'], ['test.test.test']]
        for expose_produced_outputs in cases_expose_produced_outputs:
            with self.assertRaises(ValueError) as cm:
                fitted_pipeline, predictions, fit_result = runtime.fit(
                    pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams,
                    random_seed=random_seed, volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
                    runtime_environment=self.runtime_enviroment, outputs_to_expose=expose_produced_outputs,
                )

        # Same cases but having a subpipeline as step
        pipeline_with_subpipeline = self._build_pipeline('3d7be7d2-458f-4d57-94ce-5f1e1619b477', [pipeline])

        # Case boolean True
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline_with_subpipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, expose_produced_outputs=True
        )

        self.assertEqual(
            list(fit_result.values.keys()),
            ['outputs.0', 'steps.0.outputs.0', 'steps.0.produce', 'steps.0.steps.0.produce', 'steps.0.steps.1.produce',
             'steps.0.steps.2.produce', 'steps.0.steps.3.produce', 'steps.0.steps.4.produce']
        )

        # Case boolean False
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline_with_subpipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, expose_produced_outputs=False
        )

        self.assertEqual(list(fit_result.values.keys()), ['outputs.0'])

        # Check exposing values from invalid produce methods.
        expose_produced_outputs = ['steps.0.steps.4.produce', 'steps.0.steps.4.produce_feature_importances',
                                   'steps.0.steps.1.produce', 'steps.0.produce']
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline_with_subpipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, outputs_to_expose=expose_produced_outputs,
        )

        for expose_output in expose_produced_outputs:
            self.assertIn(expose_output, fit_result.values)

        # Check for different cases such as index value out of range, random method call and that other methods that are not produced.
        cases_expose_produced_outputs = [
            ['steps.5.produce'], ['steps.1.random_method'], ['steps.4.log_likelihoods'], ['test'], ['test.test'],
            ['test.test.test'], ['steps.0.'], ['steps.0.steps.4.log_likelihoods']]
        for expose_produced_outputs in cases_expose_produced_outputs:
            with self.assertRaises(ValueError) as cm:
                pipeline_with_subpipeline, predictions, fit_result = runtime.fit(
                    pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams,
                    random_seed=random_seed, volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
                    runtime_environment=self.runtime_enviroment, outputs_to_expose=expose_produced_outputs,
                )

        # Case for pipeline with no outputs and expossing all produce methods
        pipeline.outputs = []
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, expose_produced_outputs=True, is_standard_pipeline=False
        )

        self.assertEqual(
            list(fit_result.values.keys()),
            ['steps.0.produce', 'steps.1.produce', 'steps.2.produce', 'steps.3.produce', 'steps.4.produce']
        )

        # Case for a non-standard pipeline with multiple outputs.
        outputs_to_check = ['steps.2.produce', 'steps.3.produce', 'steps.4.produce']
        for output in outputs_to_check:
            pipeline.add_output(output)

        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, expose_produced_outputs=True, is_standard_pipeline=False
        )

        self.assertEqual(
            list(fit_result.values.keys()),
            ['outputs.0', 'outputs.1', 'outputs.2', 'steps.0.produce', 'steps.1.produce', 'steps.2.produce',
             'steps.3.produce', 'steps.4.produce']
        )

    def test_expose_outputs_produce(self):
        pipeline = self._build_pipeline(
            'c8e08a03-68b4-4ce8-8828-1a0fb422418d',
            sequence=[
                {
                    'primitive_class': DatasetToDataFramePrimitive
                },
                {
                    'primitive_class': ColumnParserPrimitive
                },
                {
                    'primitive_class': ExtractColumnsBySemanticTypesPrimitive,
                    'INPUTS': [('inputs', 1)],
                    'HYPERPARAMS': {
                        'semantic_types': {
                            'TYPE': metadata_base.ArgumentType.VALUE,
                            'DATA': ['https://metadata.datadrivendiscovery.org/types/Attribute']
                        },
                        'use_columns': {
                            'TYPE': metadata_base.ArgumentType.VALUE,
                            'DATA': [1, 2, 3, 4]
                        }
                    }
                },
                {
                    'primitive_class': ExtractColumnsBySemanticTypesPrimitive,
                    'INPUTS': [('inputs', 1)],
                    'HYPERPARAMS': {'semantic_types': {
                        'TYPE': metadata_base.ArgumentType.VALUE,
                        'DATA': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']
                    }}
                },
                {
                    'primitive_class': RandomForestClassifierPrimitive,
                    'INPUTS': [('inputs', 2), ('outputs', 3)],
                },
            ],
        )
        iris_problem, iris_dataset = self.get_data()
        inputs = [iris_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment
        )

        # Check exposing values from produce methods that are not in the pipeline.
        expose_produced_outputs = ['steps.4.produce', 'steps.4.produce_feature_importances', 'steps.1.produce']
        predictions, produce_result = runtime.produce(fitted_pipeline, inputs, outputs_to_expose=expose_produced_outputs)

        for expose_output in expose_produced_outputs:
            self.assertIn(expose_output, produce_result.values)

        self._validate_pipeline_run_structure(produce_result.pipeline_run.to_json_structure())

        # Check for different cases such as index value out of range, random method call and that other methods that are not produced.
        cases_expose_produced_outputs = [
            ['steps.5.produce'], ['steps.1.random_method'], ['steps.4.log_likelihoods'],
            ['test'], ['test.test'], ['test.test.test']
        ]
        for expose_produced_outputs in cases_expose_produced_outputs:
            with self.assertRaises(ValueError) as cm:
                predictions, produce_result = runtime.produce(
                    fitted_pipeline, inputs, outputs_to_expose=expose_produced_outputs)

        # Same cases but having a subpipeline as step
        pipeline_with_subpipeline = self._build_pipeline('3d7be7d2-458f-4d57-94ce-5f1e1619b477', [pipeline])

        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline_with_subpipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams,
            random_seed=random_seed, volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment
        )

        # Check exposing values from produce methods that are not in the pipeline.
        expose_produced_outputs = ['steps.0.steps.4.produce', 'steps.0.steps.4.produce_feature_importances',
                                   'steps.0.steps.1.produce', 'steps.0.produce']
        predictions, produce_result = runtime.produce(
            fitted_pipeline, inputs,
            outputs_to_expose=expose_produced_outputs,
        )

        for expose_output in expose_produced_outputs:
            self.assertIn(expose_output, produce_result.values)

        self._validate_pipeline_run_structure(produce_result.pipeline_run.to_json_structure())

        # Check for different cases such as index value out of range, random method call and that other methods that are not produced.
        cases_expose_produced_outputs = [
            ['steps.5.produce'], ['steps.1.random_method'], ['steps.4.log_likelihoods'], ['test'], ['test.test'],
            ['test.test.test'], ['steps.0.'], ['steps.0.steps.4.log_likelihoods']]

        for expose_produced_outputs in cases_expose_produced_outputs:
            with self.assertRaises(ValueError) as cm:
                predictions, produce_result = runtime.produce(
                    fitted_pipeline, inputs, outputs_to_expose=expose_produced_outputs)

    def test_multi_input_produce(self):
        with open(
                os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'multi-input-test.json'), 'r'
        ) as pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_json(pipeline_file, resolver=Resolver())
        iris_problem, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='multi_dataset_problem')
        _, iris_dataset_2 = self.get_data(dataset_name='boston_dataset_1', problem_name='')
        inputs = [iris_dataset, iris_dataset_2]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        r = runtime.Runtime(pipeline, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment,
                            hyperparams=hyperparams, random_seed=random_seed, volumes_dir=volumes_dir,
                            problem_description=iris_problem)
        r.fit(inputs=inputs).check_success()
        r.produce(inputs=inputs).check_success()

    def test_multi_input_produce_without_problem(self):
        with open(
                os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'multi-input-test.json'), 'r'
        ) as pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_json(pipeline_file, resolver=Resolver())
        _, iris_dataset = self.get_data(dataset_name='iris_dataset_1', problem_name='')
        _, boston_dataset = self.get_data(dataset_name='boston_dataset_1', problem_name='')
        inputs = [iris_dataset, boston_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        r = runtime.Runtime(pipeline, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment,
                            hyperparams=hyperparams, random_seed=random_seed, volumes_dir=volumes_dir)
        r.fit(inputs=inputs).check_success()
        r.produce(inputs=inputs).check_success()

    def test_multi_input_produce_with_one_dataset_associated(self):
        with open(
                os.path.join(os.path.dirname(__file__), 'data', 'pipelines', 'multi-input-test.json'), 'r'
        ) as pipeline_file:
            with utils.silence():
                pipeline = pipeline_module.Pipeline.from_json(pipeline_file, resolver=Resolver())
        _, iris_dataset_1 = self.get_data(dataset_name='iris_dataset_1', problem_name='')
        boston_problem, iris_dataset_2 = self.get_data(dataset_name='boston_dataset_1', problem_name='boston_problem_1')
        inputs = [iris_dataset_1, iris_dataset_2]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None
        r = runtime.Runtime(pipeline, context=metadata_base.Context.TESTING, environment=self.runtime_enviroment,
                            hyperparams=hyperparams, random_seed=random_seed, volumes_dir=volumes_dir,
                            problem_description=boston_problem)
        r.fit(inputs=inputs).check_success()
        r.produce(inputs=inputs).check_success()

    @staticmethod
    def _build_fail_runtime(method_name, message):

        class FailRuntime(runtime.Runtime):
            pass

        def fail_method(*args, **kwargs):
            raise Exception(message)

        setattr(FailRuntime, method_name, fail_method)

        return FailRuntime

    def test_error_propgation(self):
        for method_name in [
            '_call_primitive_method', '_create_pipeline_primitive',
            '_run_primitive', '_run_subpipeline', '_run_step', '_do_run_step', '_do_run',
        ]:
            error_message = 'runtime failed in method "{}"'.format(method_name)

            inputs = self._get_inputs()
            subpipeline = self._build_pipeline('06dfb07a-f151-467c-9f1c-51a6bf6378a3', [{'primitive_class': IncrementPrimitive}, {'primitive_class': IncrementPrimitive}])
            pipeline_with_subpipeline = self._build_pipeline('293c1883-f81a-459d-a1a8-ba19467d5ad6', [{'primitive_class': RandomPrimitive}, subpipeline, {'primitive_class': IncrementPrimitive}])
            fail_runtime_class = self._build_fail_runtime(method_name, error_message)

            r = fail_runtime_class(
                pipeline_with_subpipeline, context=metadata_base.Context.TESTING,
                environment=self.runtime_enviroment,
            )

            fit_result = r.fit(inputs)
            self.assertTrue(fit_result.pipeline_run)
            self._fake_inputs(r, fit_result.pipeline_run, inputs)
            self._check_pipelines_valid_and_failed([fit_result.pipeline_run])
            self.assertTrue(
                str(fit_result.error) in [
                    error_message,
                    'Step 0 for pipeline 293c1883-f81a-459d-a1a8-ba19467d5ad6 failed.',
                    'Step 1 for pipeline 293c1883-f81a-459d-a1a8-ba19467d5ad6 failed.',
                ],
                'Unexpected error message: {}'.format(fit_result.error)
            )

    def test_get_singleton_value(self):
        l = container.List([1], generate_metadata=True)
        l.metadata = l.metadata.update((0,), {'custom': 'metadata'})

        s = runtime.get_singleton_value(l)

        self.assertEqual(s, 1)

        l = container.List([container.List([1], generate_metadata=True)], generate_metadata=True)
        l.metadata = l.metadata.update((0,), {'custom': 'metadata1'})
        l.metadata = l.metadata.update((0, 0), {'custom': 'metadata2'})

        s = runtime.get_singleton_value(l)

        self.assertEqual(s, [1])
        self.assertEqual(utils.to_json_structure(s.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'custom': 'metadata1',
               'dimension': {'length': 1},
               'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
               'structural_type': 'd3m.container.list.List'
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'structural_type': 'int'},
        }, {
            'selector': [0],
            'metadata': {'custom': 'metadata2'},
        }])

        d = container.DataFrame({'a': [1], 'b': ['one']}, generate_metadata=True)

        s = runtime.get_singleton_value(d)

        self.assertEqual(s, [1, 'one'])
        self.assertEqual(utils.to_json_structure(s.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'dimension': {
                    'length': 2,
                    # TODO: "name" and "semantic_types" here should be removed.
                    #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/336
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'structural_type': 'd3m.container.list.List',
            },
        }, {
            'selector': [0],
            'metadata': {'name': 'a', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [1],
            'metadata': {'name': 'b', 'structural_type': 'str'},
        }])

    def test_unfitted_primitive(self):
        pipeline = pipeline_module.Pipeline()
        pipeline.add_input()

        step = pipeline_module.PrimitiveStep(
            {
                'id': '3b09ba74-cc90-4f22-9e0a-0cf4f29a7e28',
                'version': '0.1.0',
                'name': "Removes columns",
                'python_path': 'd3m.primitives.data_transformation.remove_columns.Common',
            },
            resolver=pipeline_module.Resolver(),
        )
        step.add_hyperparameter('columns', metadata_base.ArgumentType.VALUE, [3])

        pipeline.add_step(step)

        step = pipeline_module.PrimitiveStep(
            {
                'id': '5bef5738-1638-48d6-9935-72445f0eecdc',
                'version': '0.1.0',
                'name': "Map DataFrame resources to new resources using provided primitive",
                'python_path': 'd3m.primitives.operator.dataset_map.DataFrameCommon',
            },
            resolver=pipeline_module.Resolver(),
        )
        step.add_argument('inputs', metadata_base.ArgumentType.CONTAINER, 'inputs.0')
        step.add_output('produce')
        step.add_hyperparameter('primitive', metadata_base.ArgumentType.PRIMITIVE, 0)

        pipeline.add_step(step)

        pipeline.add_output('steps.1.produce')

        pipeline.check(allow_placeholders=False, standard_pipeline=False, input_types={'inputs.0': container.Dataset})

        _, dataset = self.get_data()

        self.assertEqual(dataset['learningData'].shape, (150, 6))

        hyperparams = dict(pipeline.steps[1].primitive.metadata.get_hyperparams().defaults())
        del hyperparams['primitive']

        r = runtime.Runtime(
            pipeline,
            # We explicitly pass default hyper-parameters to test this code path as well.
            [
                {},
                hyperparams,
            ],
            context=metadata_base.Context.TESTING,
            is_standard_pipeline=False,
            environment=self.runtime_enviroment,
        )

        inputs = [dataset]

        result = r.fit(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertTrue(result.pipeline_run)

        self.assertEqual(len(result.values), 1)

        output_dataset = result.values['outputs.0']

        self.assertEqual(output_dataset['learningData'].shape, (150, 5))

        result = r.produce(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)
        self.assertTrue(result.pipeline_run)

        output_dataset = result.values['outputs.0']

        self.assertEqual(output_dataset['learningData'].shape, (150, 5))

        pickled = pickle.dumps(r)
        restored = pickle.loads(pickled)

        result = restored.produce(inputs, outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertEqual(len(result.values), 1)
        self.assertTrue(result.pipeline_run)

        output_dataset = result.values['outputs.0']

        self.assertEqual(output_dataset['learningData'].shape, (150, 5))

        pickle.dumps(r)

    def test_pipeline_openml(self):
        # Creating pipeline
        pipeline_description = pipeline_module.Pipeline()
        pipeline_description.add_input(name='inputs')

        # Step 0: dataset_to_dataframe
        step_0 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'),
        )
        step_0.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='inputs.0')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        # Step 1: profiler
        step_1 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.schema_discovery.profiler.Common'),
        )
        step_1.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.0.produce')
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

        # Step 2: column_parser
        step_2 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'),
        )
        step_2.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.1.produce')
        step_2.add_output('produce')
        pipeline_description.add_step(step_2)

        # Step 4: random_forest
        step_3 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.classification.random_forest.Common'),
        )
        step_3.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.2.produce')
        step_3.add_argument(name='outputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.2.produce')
        step_3.add_hyperparameter(name='return_result', argument_type=metadata_base.ArgumentType.VALUE, data='replace')
        step_3.add_output('produce')
        pipeline_description.add_step(step_3)

        # Step 5: construct predictions
        step_4 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'),
        )
        step_4.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.3.produce')
        step_4.add_argument(name='reference', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.2.produce')
        step_4.add_output('produce')
        pipeline_description.add_step(step_4)

        # Final Output
        pipeline_description.add_output(name='output predictions', data_reference='steps.4.produce')

        # Load OpenML Dataset
        dataset_id = 61
        dataset_name = 'iris'
        openml_dataset_uri = 'https://www.openml.org/d/{dataset_id}'.format(dataset_id=dataset_id)
        ds = container.Dataset.load(openml_dataset_uri, dataset_id=str(dataset_id), dataset_name=dataset_name)

        with utils.silence():
            r = runtime.Runtime(pipeline=pipeline_description, context=metadata_base.Context.TESTING)
            r.fit(inputs=[ds]).check_success()
            result = r.produce(inputs=[ds])

        result.check_success()
        predictions = result.values['outputs.0']

        self.assertEqual(predictions.shape, (150, 2))
        self.assertTrue(predictions.metadata.has_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            'https://metadata.datadrivendiscovery.org/types/PredictedTarget'),
        )
        self.assertFalse(predictions.metadata.has_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            'https://metadata.datadrivendiscovery.org/types/TrueTarget'),
        )


    def test_problem_pipeline_openml(self):
        # Creating pipeline
        pipeline_description = pipeline_module.Pipeline()
        pipeline_description.add_input(name='inputs')

        # Step 0: dataset_to_dataframe
        step_0 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'),
        )
        step_0.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='inputs.0')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        # Step 1: profiler
        step_1 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.schema_discovery.profiler.Common'),
        )
        step_1.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.0.produce')
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

        # Step 2: column_parser
        step_2 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'),
        )
        step_2.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.1.produce')
        step_2.add_output('produce')
        pipeline_description.add_step(step_2)

        # Step 4: random_forest
        step_3 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.classification.random_forest.Common'),
        )
        step_3.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.2.produce')
        step_3.add_argument(name='outputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.2.produce')
        step_3.add_hyperparameter(name='return_result', argument_type=metadata_base.ArgumentType.VALUE, data='replace')
        step_3.add_output('produce')
        pipeline_description.add_step(step_3)

        # Step 5: construct predictions
        step_4 = pipeline_module.PrimitiveStep(
            primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'),
        )
        step_4.add_argument(name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.3.produce')
        step_4.add_argument(name='reference', argument_type=metadata_base.ArgumentType.CONTAINER, data='steps.2.produce')
        step_4.add_output('produce')
        pipeline_description.add_step(step_4)

        # Final Output
        pipeline_description.add_output(name='output predictions', data_reference='steps.4.produce')

        # Load problem and dataset
        task_id = 59
        openml_problem_uri = 'https://www.openml.org/t/{task_id}'.format(task_id=task_id)
        problem_description = problem.Problem.load(openml_problem_uri)

        # We get the dataset id from the problem
        dataset_id = int(problem_description['inputs'][0]['dataset_id'].split('_')[-1])
        openml_dataset_uri = 'https://www.openml.org/d/{dataset_id}'.format(dataset_id=dataset_id)
        ds = container.Dataset.load(openml_dataset_uri)

        with utils.silence():
            r = runtime.Runtime(pipeline=pipeline_description, problem_description=problem_description,
                                context=metadata_base.Context.TESTING)
            r.fit(inputs=[ds]).check_success()
            result = r.produce(inputs=[ds])

        result.check_success()
        predictions = result.values['outputs.0']

        self.assertEqual(predictions.shape, (150, 2))
        self.assertTrue(predictions.metadata.has_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            'https://metadata.datadrivendiscovery.org/types/PredictedTarget'),
        )
        self.assertFalse(predictions.metadata.has_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            'https://metadata.datadrivendiscovery.org/types/TrueTarget'),
        )

    def test_sort_pipeline_runs(self):
        self.assertEqual(runtime._sort_pipeline_runs([
            {
                'id': 'foo1',
            },
            {
                'id': 'foo2',
                'previous_pipeline_run': {
                    'id': 'foo1',
                },
            },
            {
                'id': 'foo3'
            }
        ]), [
            [{'id': 'foo1'}, {'id': 'foo2', 'previous_pipeline_run': {'id': 'foo1'}}],
            [{'id': 'foo3'}],
        ])

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'No first pipeline run'):
            runtime._sort_pipeline_runs([
                {
                    'id': 'foo2',
                    'previous_pipeline_run': {
                        'id': 'foo1',
                    },
                },
            ])

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, "Pipeline runs with previous pipeline runs which were not provided: \['foo2'\]"):
            runtime._sort_pipeline_runs([
                {
                    'id': 'foo1',
                },
                {
                    'id': 'foo2',
                    'previous_pipeline_run': {
                        'id': 'foo4',
                    },
                },
                {
                    'id': 'foo3'
                }
            ])

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, "Pipeline runs with previous pipeline runs which were not provided: \['foo1', 'foo2', 'foo3'\]"):
            runtime._sort_pipeline_runs([
                {
                    'id': 'foo0',
                },
                {
                    'id': 'foo1',
                    'previous_pipeline_run': {
                        'id': 'foo2',
                    },
                },
                {
                    'id': 'foo2',
                    'previous_pipeline_run': {
                        'id': 'foo3',
                    },
                },
                {
                    'id': 'foo3',
                    'previous_pipeline_run': {
                        'id': 'foo1',
                    },
                }
            ])

    def test_extra_argument_in_extra_produce(self):
        pipeline = pipeline_module.Pipeline.from_json("""
            {
              "id": "90e99f5b-fdec-4840-b5c3-dcdbd22eaa5e",
              "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
              "created": "2020-02-28T09:42:27.443844Z",
              "name": "Test pipeline",
              "description": "Just a test pipeline",
              "inputs": [
                {}
              ],
              "outputs": [
                {
                  "data": "steps.0.produce"
                }
              ],
              "steps": [
                {
                  "type": "PRIMITIVE",
                  "primitive": {
                    "id": "13d2174a-5502-4eeb-84b2-a12f0e7e8b6e",
                    "version": "0.1.0",
                    "python_path": "d3m.primitives.test.ExtraArgumentPrimitive",
                    "name": "Extra Argument Primitive"
                  },
                  "arguments": {
                    "inputs": {
                      "type": "CONTAINER",
                      "data": "inputs.0"
                    }
                  },
                  "outputs": [
                    {
                      "id": "produce"
                    }
                  ]
                }
              ]
            }
        """, resolver=Resolver())

        iris_problem, iris_dataset = self.get_data()
        inputs = [iris_dataset]
        hyperparams = None
        random_seed = 0
        volumes_dir: str = None

        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, is_standard_pipeline=False,
        )
        fit_result.check_success()

        with self.assertLogs(logger=runtime.logger) as cm:
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
                volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
                runtime_environment=self.runtime_enviroment, is_standard_pipeline=False, outputs_to_expose=['steps.0.produce_another'],
            )
            fit_result.check_success()

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, "Additional output to expose 'produce_another' does not have all necessary arguments available. Skipping exposing. Missing arguments: ['extra_data']")

        pipeline = pipeline_module.Pipeline.from_json("""
            {
              "id": "90e99f5b-fdec-4840-b5c3-dcdbd22eaa5e",
              "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
              "created": "2020-02-28T09:42:27.443844Z",
              "name": "Test pipeline",
              "description": "Just a test pipeline",
              "inputs": [
                {}
              ],
              "outputs": [
                {
                  "data": "steps.0.produce"
                }
              ],
              "steps": [
                {
                  "type": "PRIMITIVE",
                  "primitive": {
                    "id": "13d2174a-5502-4eeb-84b2-a12f0e7e8b6e",
                    "version": "0.1.0",
                    "python_path": "d3m.primitives.test.ExtraArgumentPrimitive",
                    "name": "Extra Argument Primitive"
                  },
                  "arguments": {
                    "inputs": {
                      "type": "CONTAINER",
                      "data": "inputs.0"
                    },
                    "extra_data": {
                      "type": "CONTAINER",
                      "data": "inputs.0"
                    }
                  },
                  "outputs": [
                    {
                      "id": "produce"
                    }
                  ]
                }
              ]
            }
        """, resolver=Resolver())

        fitted_pipeline, predictions, fit_result = runtime.fit(
            pipeline, inputs, problem_description=iris_problem, hyperparams=hyperparams, random_seed=random_seed,
            volumes_dir=volumes_dir, context=metadata_base.Context.TESTING,
            runtime_environment=self.runtime_enviroment, is_standard_pipeline=False, outputs_to_expose=['steps.0.produce_another'],
        )
        fit_result.check_success()

    def _run_pipeline_with_static_primitive_input(self, pipeline):
        pipeline.check(allow_placeholders=False, standard_pipeline=False, input_types={})

        r = runtime.Runtime(
            pipeline,
            context=metadata_base.Context.TESTING,
            is_standard_pipeline=False,
            environment=self.runtime_enviroment,
        )

        result = r.fit([], outputs_to_expose=['outputs.0'])
        result.check_success()

        self.assertTrue(result.pipeline_run)

        self.assertEqual(len(result.values), 1)

        outputs = result.values['outputs.0']

        self.assertTrue(numpy.allclose(outputs.values, container.ndarray([1.76405235, 0.40015721, 0.97873798, 2.2408932]).reshape((4, 1))))

    def test_static_primitive_input(self):
        pipeline = pipeline_module.Pipeline()

        input_numbers = list(range(4))
        inputs = container.List(input_numbers, generate_metadata=True)

        step = pipeline_module.PrimitiveStep(
            {
                'id': 'df3153a1-4411-47e2-bbc0-9d5e9925ad79',
                'version': '0.1.0',
                'name': "Random Samples",
                'python_path': 'd3m.primitives.data_generation.random.Test',
            },
            resolver=pipeline_module.Resolver(),
        )
        step.add_argument('inputs', metadata_base.ArgumentType.VALUE, inputs)
        step.add_output('produce')

        pipeline.add_step(step)

        pipeline.add_output('steps.0.produce')

        self._run_pipeline_with_static_primitive_input(pipeline)

        # This should pickle the list because it is container.List.
        pipeline_json = pipeline.to_json()

        # This should unpickle the list.
        pipeline = pipeline_module.Pipeline.from_json(pipeline_json)

        self._run_pipeline_with_static_primitive_input(pipeline)

        # Let's make a simpler pipeline without container.List. This is supported to make it easier to write those by hand.
        pipeline_structure = json.loads(pipeline_json)
        pipeline_structure['steps'][0]['arguments']['inputs']['data'] = input_numbers
        del pipeline_structure['digest']
        pipeline_json = json.dumps(pipeline_structure)

        pipeline = pipeline_module.Pipeline.from_json(pipeline_json)

        self._run_pipeline_with_static_primitive_input(pipeline)


if __name__ == '__main__':
    unittest.main()
