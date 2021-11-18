import argparse
import inspect
import itertools
import json
import logging
import os
import os.path
import pickle
import re
import sys
import tempfile
import traceback
import typing
import uuid

import jsonschema
import frozendict
import pandas

from d3m import container, deprecate, exceptions, types, utils
from d3m.container import dataset as dataset_module
from d3m.container import utils as container_utils
from d3m.contrib import pipelines as contrib_pipelines
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, pipeline_run as pipeline_run_module, problem
from d3m.primitive_interfaces import base

logger = logging.getLogger(__name__)

DEFAULT_SCORING_PIPELINE_ID = contrib_pipelines.SCORING_PIPELINE_ID
DEFAULT_SCORING_PIPELINE_PATH = contrib_pipelines.SCORING_PIPELINE_PATH

DATASET_ID_REGEX = re.compile(r'(_FOLD_\d+)?(_TRAIN|_TEST|_SCORE)$')

DATA_PIPELINE_RUN_FILENAME = 'data_preparation_pipeline_run.pkl'


class Result:
    """
    Results from running a pipeline.

    Parameters
    ----------
    pipeline_run:
        A pipeline run description.
    values:
        A map between data references and their values computed during pipeline run.
    error:
        If during a run an exception occurred, then it is available here.
    """

    def __init__(self, pipeline_run: pipeline_run_module.PipelineRun, values: typing.Dict[str, typing.Any], error: Exception = None) -> None:
        self.pipeline_run = pipeline_run
        self.values = values
        self.error = error

    def has_error(self) -> bool:
        """
        Returns ``True`` if pipeline has not successfully finished.
        """

        return self.error is not None

    def check_success(self) -> None:
        """
        Throws an exception if pipeline has not successfully finished.
        """

        if self.has_error():
            raise self.error  # type: ignore

    def get_standard_pipeline_output(self) -> typing.Optional[container.DataFrame]:
        """
        Returns the output value if exists and its from a standard pipeline.
        """

        if self.has_error() or not self.pipeline_run.is_standard_pipeline:
            output = None
        else:
            output = self.values['outputs.0']

            if not isinstance(output, container.DataFrame):
                raise TypeError(
                    "A standard pipeline's output should be of a container DataFrame type, not {output_type}.".format(
                        output_type=type(output),
                    )
                )

        return output


class MultiResult(typing.List[Result]):
    """
    Results of running a pipeline multiple times.
    """

    @property
    def pipeline_runs(self) -> typing.Sequence[pipeline_run_module.PipelineRun]:
        return [result.pipeline_run for result in self]

    def has_error(self) -> bool:
        """
        Returns ``True`` if any of pipelines has not successfully finished.
        """

        return any(result.has_error() for result in self)

    def check_success(self) -> None:
        """
        Throws an exception if pipeline has not successfully finished in any of the runs.
        """

        for result in self:
            result.check_success()


def get_singleton_value(value: typing.Any) -> typing.Any:
    """
    A helper to extract a value from a singleton value (extracting a sole element of a
    container of length 1).
    """

    if isinstance(value, pandas.DataFrame):
        # Fetch the row as a list. This assures different columns can be of a different type.
        singleton_value = container.List([value.iloc[0, k] for k in range(len(value.columns))])
    else:
        singleton_value = value[0]

    if isinstance(singleton_value, types.Container):
        singleton_value.metadata = metadata_base.DataMetadata()
        singleton_value.metadata = value.metadata.copy_to(
            singleton_value.metadata,
            (0,),
        )
        # TODO: We should also remove table metadata which might not hold true anymore.
        #       If original value was tabular, we now copied also metadata about tabular column dimension,
        #       but that is not true anymore for this singleton value, it is not tabular anymore.
        #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/336
        singleton_value.metadata = singleton_value.metadata.generate(singleton_value)

    return singleton_value


# TODO: Add debug logging to the runtime.
class Runtime:
    """
    Reference runtime to fit and produce a pipeline.

    Parameters
    ----------
    pipeline:
        A pipeline to run.
    hyperparams:
        Values for free hyper-parameters of the pipeline. It should be a list, where each element corresponds
        to free hyper-parameters of the corresponding pipeline step. Not all free hyper-parameters have to be
        specified. Default values are used for those which are not. Optional.
    problem_description:
        A parsed problem description in standard problem description schema.
    context:
        In which context to run pipelines.
    random_seed:
        A random seed to use for every run. This control all randomness during the run.
    volumes_dir:
        Path to a directory with static files required by primitives.
        In the standard directory structure (as obtained running ``python3 -m d3m primitive download``).
    scratch_dir:
        Path to a directory to store any temporary files needed during execution.
    is_standard_pipeline:
        Is the pipeline a standard pipeline?
    environment:
        A description of the runtime environment, including engine versions,
        Docker images, compute resources, and benchmarks. If not provided,
        an attempt is made to determine it automatically.
    users:
        Users associated with running the pipeline.
    """

    #: A pipeline to run.
    pipeline: pipeline_module.Pipeline
    #: Values for free hyper-parameters of the pipeline. It should be a list, where each element corresponds
    #: to free hyper-parameters of the corresponding pipeline step. Not all free hyper-parameters have to be
    #: specified. Default values are used for those which are not. Optional.
    hyperparams: typing.Optional[typing.Sequence]
    #: A parsed problem description in standard problem description schema.
    problem_description: typing.Optional[problem.Problem]
    #: In which context to run pipelines.
    context: metadata_base.Context
    #: A random seed to use for every run. This control all randomness during the run.
    random_seed: int
    #: Path to a directory with static files required by primitives.
    #: In the standard directory structure (as obtained running ``python3 -m d3m primitive download``).
    volumes_dir: typing.Optional[str]
    #: Path to a directory to store any temporary files needed during execution.
    scratch_dir: typing.Optional[str]
    #: Is the pipeline a standard pipeline?
    is_standard_pipeline: bool
    #: A description of the runtime environment, including engine versions,
    #: Docker images, compute resources, and benchmarks. If not provided,
    #: an attempt is made to determine it automatically.
    environment: pipeline_run_module.RuntimeEnvironment
    #: Users associated with running the pipeline.
    users: typing.Optional[typing.Sequence[pipeline_run_module.User]]
    #: Which step is currently being ran.
    current_step: int
    #: Which phase are we currently running.
    phase: metadata_base.PipelineRunPhase
    #: A current instance of pipeline run.
    pipeline_run: typing.Optional[pipeline_run_module.PipelineRun]
    #: Which step outputs should the runtime keep during a pipeline run, even after they are necessary.
    #: Outputs which would otherwise not be produced are allowed and that forces those outputs to be produced.
    outputs_to_expose: typing.Iterable[str]
    #: Map between available data references and their values during the run.
    data_values: typing.Dict[str, typing.Any]
    #: Fitted state for each step of the pipeline.
    steps_state: typing.List[typing.Union[typing.Any, typing.List]]

    _runtime_environment: typing.ClassVar[typing.Optional[pipeline_run_module.RuntimeEnvironment]] = None

    def __init__(
        self, pipeline: pipeline_module.Pipeline, hyperparams: typing.Sequence = None, *,
        problem_description: problem.Problem = None, context: metadata_base.Context,
        random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None,
        is_standard_pipeline: bool = False, environment: pipeline_run_module.RuntimeEnvironment = None,
        users: typing.Sequence[pipeline_run_module.User] = None,
    ) -> None:
        self.pipeline = pipeline
        self.hyperparams = hyperparams
        self.problem_description = problem_description
        self.context = context
        self.random_seed = random_seed
        self.volumes_dir = volumes_dir
        self.scratch_dir = scratch_dir
        self.is_standard_pipeline = is_standard_pipeline
        self.users = users

        if environment is None:
            if type(self)._runtime_environment is None:
                type(self)._runtime_environment = pipeline_run_module.RuntimeEnvironment()
            self.environment = type(self)._runtime_environment  # type: ignore
        else:
            self.environment = environment

        # Preliminary check.
        self.pipeline.check(allow_placeholders=False, standard_pipeline=self.is_standard_pipeline)

        if self.hyperparams is not None:
            self._check_hyperparams(self.pipeline, self.hyperparams)

        self.steps_state: typing.List[typing.Union[typing.Any, typing.List, None]] = [None for step in self.pipeline.steps]

        self._previous_pipeline_run: typing.Optional[pipeline_run_module.PipelineRun] = None

        self._initialize_run_state([], None, [])

    def _initialize_data_values(self, inputs: typing.Sequence[typing.Any]) -> None:
        # TODO: Remove values from the "data_values" once they are not needed anymore to optimize memory use.
        self.data_values: typing.Dict[str, typing.Any] = {}

        if self.phase is None:
            return

        marked_problem_inputs: typing.Set[int] = set()
        if self.problem_description is None:
            problem_inputs: typing.List[typing.Dict] = []
        else:
            problem_inputs = self.problem_description.get('inputs', [])

        for i, input_value in enumerate(inputs):
            if isinstance(input_value, container.Dataset):
                if problem_inputs:
                    input_value, marked_problem_indices = self._mark_columns(problem_inputs, input_value)
                    marked_problem_inputs.update(marked_problem_indices)
            else:
                # All standard pipeline inputs should be Datasets.
                assert not self.is_standard_pipeline

            self.data_values['inputs.{i}'.format(i=i)] = input_value

        if len(marked_problem_inputs) != len(problem_inputs):
            unmarked_problem_inputs = sorted(set(range(len(problem_inputs))) - marked_problem_inputs)

            raise exceptions.InvalidProblemError(
                "Not all problem description inputs could be applied to input datasets: {inputs}".format(
                    inputs=', '.join(str(problem_inputs[unmarked_problem_input]) for unmarked_problem_input in unmarked_problem_inputs),
                )
            )

    def _clear_data_values(self) -> None:
        self.data_values = {}

    def _initialize_run_state(
        self, inputs: typing.Sequence[typing.Any],
        phase: typing.Optional[metadata_base.PipelineRunPhase],
        outputs_to_expose: typing.Iterable[str],
    ) -> None:
        self.current_step = 0
        self.phase = phase
        self.outputs_to_expose = outputs_to_expose

        self._initialize_data_values(inputs)

        self._initialize_base_temporary_directory()

        self._initialize_pipeline_run()

    def _get_all_outputs(self) -> typing.Sequence[str]:
        return ['outputs.{i}'.format(i=i) for i, output_description in enumerate(self.pipeline.outputs)]

    def _clear_run_state(self) -> None:
        """
        After a pipeline run, we clear state which was necessary while pipeline was running, but it is not needed anymore.
        """

        # We keep "steps_state" so that we can produce.

        self.current_step = 0
        self.phase = None
        self.outputs_to_expose = []

        self._clear_data_values()
        self._clear_base_temporary_directory()
        self._clear_pipeline_run()

    def _check_hyperparams(self, pipeline: pipeline_module.Pipeline, hyperparams: typing.Sequence) -> None:
        """
        Check provided values for free hyper-parameters.
        """

        if not utils.is_sequence(hyperparams):
            raise exceptions.InvalidArgumentTypeError("Hyper-parameter values for the pipeline '{pipeline_id}' is not a sequence.".format(
                pipeline_id=pipeline.id,
            ))

        if len(hyperparams) != len(pipeline.steps):
            raise exceptions.InvalidArgumentValueError(
                "Hyper-parameter values for the pipeline '{pipeline_id}' do not match the number of steps in the pipeline: {hyperparams_steps} vs. {pipeline_steps}".format(
                    pipeline_id=pipeline.id,
                    hyperparams_steps=len(hyperparams),
                    pipeline_steps=len(pipeline.steps),
                ),
            )

        for step_index, (hyperparams_for_step, step) in enumerate(zip(hyperparams, pipeline.steps)):
            # Placeholder step is not really allowed, but we have it here for completeness.
            # Its "get_free_hyperparams" returns an empty list.
            if isinstance(step, pipeline_module.PlaceholderStep):
                if not utils.is_sequence(hyperparams_for_step):
                    raise exceptions.InvalidArgumentTypeError("Hyper-parameter values for placeholder step {step_index} of pipeline '{pipeline_id}' is not a sequence.".format(
                        step_index=step_index,
                        pipeline_id=pipeline.id,
                    ))

            elif isinstance(step, pipeline_module.SubpipelineStep):
                if step.pipeline is None:
                    raise exceptions.InvalidStateError("Pipeline has not been resolved.")

                self._check_hyperparams(step.pipeline, hyperparams_for_step)

            elif isinstance(step, pipeline_module.PrimitiveStep):
                if not isinstance(hyperparams_for_step, (dict, frozendict.frozendict)):
                    raise exceptions.InvalidArgumentTypeError("Hyper-parameter values for primitive step {step_index} of pipeline '{pipeline_id}' is not a dict.".format(
                        step_index=step_index,
                        pipeline_id=pipeline.id,
                    ))

                hyperparams_for_step_keys = set(hyperparams_for_step.keys())
                free_hyperparams_keys = set(step.get_free_hyperparams().keys())
                all_hyperparams_keys = set(step.get_all_hyperparams().keys())

                if hyperparams_for_step_keys - all_hyperparams_keys:
                    raise exceptions.InvalidArgumentValueError(
                        "Hyper-parameter values for primitive step {step_index} of pipeline '{pipeline_id}' contain values for non-existent hyper-parameters: {hyperparams}".format(
                            step_index=step_index,
                            pipeline_id=pipeline.id,
                            hyperparams=sorted(hyperparams_for_step_keys - all_hyperparams_keys),
                        ),
                    )
                elif hyperparams_for_step_keys - free_hyperparams_keys:
                    raise exceptions.InvalidArgumentValueError(
                        "Hyper-parameter values for primitive step {step_index} of pipeline '{pipeline_id}' are overriding hyper-parameters fixed in the pipeline: {hyperparams}".format(
                            step_index=step_index,
                            pipeline_id=pipeline.id,
                            hyperparams=sorted(hyperparams_for_step_keys - free_hyperparams_keys),
                        ),
                    )

    def _get_pipeline_run_class(self) -> typing.Type[pipeline_run_module.PipelineRun]:
        return pipeline_run_module.PipelineRun

    def _initialize_pipeline_run(self) -> None:
        if self.phase is None:
            self.pipeline_run = None
            return

        self.pipeline_run = self._get_pipeline_run_class()(
            pipeline=self.pipeline,
            problem_description=self.problem_description,
            phase=self.phase,
            context=self.context,
            previous_pipeline_run=self._previous_pipeline_run,
            environment=self.environment,
            random_seed=self.random_seed,
            is_standard_pipeline=self.is_standard_pipeline,
            users=self.users
        )

        # We make sure we always set this ID as soon as possible, so even if the current phase run fails
        # even with an internal error which never produces a pipeline run, it can at least be visible
        # that some pipeline run is missing in the sequence of phase runs.
        self._previous_pipeline_run = self.pipeline_run

        input_values = []
        for i, input_value in sorted((int(data_reference.split('.')[1]), input_value) for data_reference, input_value in self.data_values.items() if data_reference.startswith('inputs.')):
            input_values.append(input_value)

        all_input_values_datasets = all(isinstance(input_value, container.Dataset) for input_value in input_values)
        assert all_input_values_datasets or not self.is_standard_pipeline

        # Even if the pipeline is not a standard pipeline, we still record Dataset inputs (if all are Dataset inputs)
        # into pipeline run to allow generation of pipeline runs for a subset of non-standard pipelines, especially
        # those computing metafeatures. Because having inputs recorded is required for a pipeline run, any other
        # (for other types of inputs) pipeline run is not a valid stand-alone pipeline run and you get an error if
        # you want to serialize it to JSON. This is on purpose. (We could have a better error message though.)
        # You can still build a pipeline run object for non-standard pipelines. This is being used for data
        # preparation or scoring pipelines.
        # See: https://gitlab.com/datadrivendiscovery/metalearning/issues/64
        if all_input_values_datasets:
            for input_value in input_values:
                self.pipeline_run.add_input_dataset(input_value)

    def _clear_pipeline_run(self) -> None:
        self.pipeline_run = None

    def _initialize_base_temporary_directory(self) -> None:
        if self.phase is None:
            self._base_temporary_directory = None
            self._base_temporary_directory_path = None
            return

        self._base_temporary_directory = tempfile.TemporaryDirectory(dir=self.scratch_dir)
        self._base_temporary_directory_path = os.path.abspath(self._base_temporary_directory.name)

    def _clear_base_temporary_directory(self) -> None:
        if self._base_temporary_directory is not None:
            self._base_temporary_directory.cleanup()
            self._base_temporary_directory = None
            self._base_temporary_directory_path = None

    def _check_pipeline(self, inputs: typing.Sequence[typing.Any], outputs_to_expose: typing.Iterable[str]) -> typing.Iterable[str]:
        """
        Check with known inputs and outputs to expose.
        """

        input_types = {}
        for i, input_value in enumerate(inputs):
            input_types['inputs.{i}'.format(i=i)] = type(input_value)

        self.pipeline.check(allow_placeholders=False, standard_pipeline=self.is_standard_pipeline, input_types=input_types)

        exposable_outputs = self.pipeline.get_exposable_outputs()
        outputs_to_expose_set = set(outputs_to_expose)

        not_exposable_outputs = outputs_to_expose_set - exposable_outputs
        if not_exposable_outputs:
            raise exceptions.InvalidArgumentValueError('{not_exposable_outputs} are not exposable outputs.'.format(
                not_exposable_outputs=sorted(not_exposable_outputs),
            ))

        for i, step in enumerate(self.pipeline.steps):
            if not isinstance(step, pipeline_module.PrimitiveStep):
                continue

            if step.primitive is None:
                raise exceptions.InvalidStateError("Primitive has not been resolved.")

            arguments_set = set(step.arguments.keys())
            primitive_arguments_without_defaults = step._get_primitive_arguments_without_defaults()
            instance_methods = step.primitive.metadata.query()['primitive_code'].get('instance_methods', {})
            step_reference_prefix = 'steps.{i}.'.format(i=i)
            # We iterate over "outputs_to_expose" but we modify "outputs_to_expose_set".
            for output_to_expose in outputs_to_expose:
                if output_to_expose.startswith(step_reference_prefix):
                    produce_method = output_to_expose[len(step_reference_prefix):]
                    # Produce method should not contain a dot.
                    assert '.' not in produce_method, produce_method
                    produce_methods = step.outputs
                    if produce_method not in produce_methods:
                        produce_method_arguments = set(instance_methods.get(produce_method, {}).get('arguments', [])) & primitive_arguments_without_defaults
                        missing_arguments = produce_method_arguments - arguments_set
                        if missing_arguments:
                            logger.warning(
                                "Additional output to expose '%(produce_method)s' does not have all necessary arguments available. Skipping exposing. Missing arguments: %(missing_arguments)s",
                                {
                                    'produce_method': produce_method,
                                    'missing_arguments': sorted(missing_arguments),
                                },
                            )
                            outputs_to_expose_set.remove(output_to_expose)

        # We sort to have deterministic order.
        return sorted(outputs_to_expose_set)

    def _run_placeholder(self, step: pipeline_module.PlaceholderStep) -> None:
        raise exceptions.InvalidPipelineError("Step {step_index} of pipeline '{pipeline_id}' is a placeholder but there should be no placeholders.".format(
            step_index=self.current_step,
            pipeline_id=self.pipeline.id,
        ))

    # TODO: Make return type be equal to the current's class type, so that it adapts if this class is subclassed.
    def _create_subpipeline(self, pipeline: pipeline_module.Pipeline, hyperparams: typing.Optional[typing.Sequence]) -> 'Runtime':
        """
        Creates an instance of the subpipeline's runtime.
        """

        # We change the random seed in a deterministic way so that it does not matter in which order we run steps.
        # Subpipelines are generally not a standard pipeline.
        return type(self)(
            pipeline,
            hyperparams,
            # TODO: Should we pass "problem_description" as well, but make it so that it does not try to mark columns again?
            problem_description=None,
            context=self.context,
            random_seed=self.random_seed + self.current_step,
            volumes_dir=self.volumes_dir,
            scratch_dir=self.scratch_dir,
            is_standard_pipeline=False,
            environment=self.environment,
            users=self.users,
        )

    def _run_subpipeline(self, step: pipeline_module.SubpipelineStep) -> None:
        assert self.pipeline_run is not None

        if step.pipeline is None:
            raise exceptions.InvalidPipelineError("Pipeline has not been resolved.")

        subpipeline_inputs: typing.List[typing.Any] = []
        for i, data_reference in enumerate(step.inputs):
            subpipeline_inputs.append(self.data_values[data_reference])

        if self.hyperparams is not None:
            hyperparams = self.hyperparams[self.current_step]

            # We checked this already in "_check_hyperparams".
            assert utils.is_sequence(hyperparams), hyperparams
        else:
            hyperparams = None

        subpipeline = self._create_subpipeline(step.pipeline, hyperparams)

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            assert self.steps_state[self.current_step] is None
        else:
            subpipeline.set_params(typing.cast(typing.List, self.steps_state[self.current_step]))

        outputs_to_expose_map = {}
        outputs_to_expose = set()
        for i, output_id in enumerate(step.outputs):
            # "output_id" can be "None" if this output is not used and should be skipped.
            if output_id is not None:
                data_reference = 'outputs.{i}'.format(i=i)
                outputs_to_expose.add(data_reference)
                outputs_to_expose_map['steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)] = data_reference

        step_reference_prefix = 'steps.{i}.'.format(i=step.index)
        for output_to_expose in self.outputs_to_expose:
            # We process recursive data references for this subpipeline.
            # We check that "output_to_expose" is not in "outputs_to_expose_map" because data
            # references of the format "steps.{i}.{output_id}" have "step_reference_prefix"
            # as a prefix but are not really a recursive data reference.
            # But all references of that format are already in "outputs_to_expose_map".
            if output_to_expose.startswith(step_reference_prefix) and output_to_expose not in outputs_to_expose_map:
                data_reference = output_to_expose[len(step_reference_prefix):]
                # Data reference at this point should contain at least one dot, because all with the prefix
                # which do not contain a dot we filtered out by checking them against "outputs_to_expose_map".
                assert '.' in data_reference, data_reference
                outputs_to_expose.add(data_reference)
                outputs_to_expose_map[output_to_expose] = data_reference

        # We sort "outputs_to_expose" to have deterministic order.
        result = subpipeline._run(subpipeline_inputs, self.phase, outputs_to_expose=sorted(outputs_to_expose))
        self.pipeline_run.add_subpipeline_step(result.pipeline_run)
        result.check_success()

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            assert self.steps_state[self.current_step] is None
            self.steps_state[self.current_step] = subpipeline.get_params()

        for step_data_reference, subpipeline_data_reference in outputs_to_expose_map.items():
            self.data_values[step_data_reference] = result.values[subpipeline_data_reference]

    def _get_singleton_value(self, value: typing.Any, is_argument: bool, name: str) -> typing.Any:
        """
        A helper to extract a value from a singleton value (extracting a sole element of a
        container of length 1).
        """

        if len(value) != 1:
            if is_argument:
                raise exceptions.InvalidPipelineError(
                    "Argument '{argument_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, but available data is not.".format(
                        argument_name=name,
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                    ),
                )
            else:
                raise exceptions.InvalidPipelineError(
                    "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, but available data is not.".format(
                        hyperparameter_name=name,
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                    ),
                )

        return get_singleton_value(value)

    def _prepare_primitive_arguments(self, step: pipeline_module.PrimitiveStep) -> typing.Dict[str, typing.Any]:
        arguments = {}
        for argument_name, argument_description in step.arguments.items():

            if argument_description['type'] == metadata_base.ArgumentType.DATA:
                argument_value = self.data_values[argument_description['data']]
                # We have to extract a singleton value out.
                argument_value = self._get_singleton_value(argument_value, True, argument_name)

            elif argument_description['type'] == metadata_base.ArgumentType.CONTAINER:
                if utils.is_sequence(argument_description['data']):
                    values = [self.data_values[data_reference] for data_reference in argument_description['data']]
                    # We have to create a container List.
                    argument_value = self._get_list_value(values)
                else:
                    argument_value = self.data_values[argument_description['data']]

            elif argument_description['type'] == metadata_base.ArgumentType.VALUE:
                argument_value = argument_description['data']

            else:
                raise exceptions.UnexpectedValueError("Unknown argument type: {argument_type}".format(argument_type=argument_description['type']))

            arguments[argument_name] = argument_value

        return arguments

    def _get_list_value(self, values: typing.Sequence) -> container.List:
        """
        Creates a container List from ``values``. It reuses existing metadata in ``values``
        to create metadata of the container List.
        """

        container_list = container.List(values, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List,
            'dimension': {
                'length': len(values),
            },
        })

        for value_index, value in enumerate(values):
            container_list.metadata = value.metadata.copy_to(container_list.metadata, (), (value_index,))

        return container_list

    def _get_default_hyperparams(self, step_index: int, step: pipeline_module.PrimitiveStep) -> hyperparams_module.Hyperparams:
        return step.get_primitive_hyperparams().defaults()

    def _get_runtime_hyperparams(self, step_index: int, step: pipeline_module.PrimitiveStep) -> typing.Dict:
        if self.hyperparams is not None:
            runtime_hyperparams = self.hyperparams[step_index]

            # We checked this already in "_check_hyperparams".
            assert isinstance(runtime_hyperparams, (dict, frozendict.frozendict)), runtime_hyperparams
        else:
            runtime_hyperparams = {}

        return runtime_hyperparams

    def _get_pipeline_hyperparams(self, step_index: int, step: pipeline_module.PrimitiveStep) -> typing.Dict:
        pipeline_hyperparams = {}
        for hyperparameter_name, hyperparameter_description in step.hyperparams.items():
            if hyperparameter_description['type'] == metadata_base.ArgumentType.DATA:
                if utils.is_sequence(hyperparameter_description['data']):
                    pipeline_hyperparams[hyperparameter_name] = [
                        self._get_singleton_value(self.data_values[data_reference], False, hyperparameter_name)
                        for data_reference in hyperparameter_description['data']
                    ]
                else:
                    pipeline_hyperparams[hyperparameter_name] = self._get_singleton_value(self.data_values[hyperparameter_description['data']], False, hyperparameter_name)

            elif hyperparameter_description['type'] == metadata_base.ArgumentType.PRIMITIVE:
                if utils.is_sequence(hyperparameter_description['data']):
                    primitive_references = hyperparameter_description['data']
                else:
                    primitive_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])

                primitives = []
                for primitive_reference in primitive_references:
                    # We make an instance of a primitive which is almost the same as the pipeline primitive
                    # (see "_create_pipeline_primitive"), but with a different random seed because of a different
                    # "current_step". Then we clone it (using "_clone_primitive") in "_handle_primitive_hyperparams"
                    # which uses the final random seed. This way we are handling all primitives in hyper-parameters
                    # the same no matter the source (it could be somebody somehow passes a primitive instance through
                    # produce method's output or something).
                    # TODO: See if an optimization (no additional clone) here is needed and how hard is to implement it.
                    # TODO: Try to re-use existing primitive instances.
                    #       We currently do not store primitive instances of prior steps, but we could those we know we
                    #       will need in later steps and then just use them here, instead of creating them from scratch.
                    primitive = self._create_primitive_reference_primitive(primitive_reference, hyperparameter_name)
                    primitives.append(primitive)

                if utils.is_sequence(hyperparameter_description['data']):
                    pipeline_hyperparams[hyperparameter_name] = primitives
                else:
                    assert len(primitives) == 1

                    pipeline_hyperparams[hyperparameter_name] = primitives[0]  # type: ignore

            elif hyperparameter_description['type'] == metadata_base.ArgumentType.CONTAINER:
                pipeline_hyperparams[hyperparameter_name] = self.data_values[hyperparameter_description['data']]

            elif hyperparameter_description['type'] == metadata_base.ArgumentType.VALUE:
                pipeline_hyperparams[hyperparameter_name] = hyperparameter_description['data']

            else:
                raise exceptions.UnexpectedValueError("Unknown hyper-parameter type: {hyperparameter_type}".format(hyperparameter_type=hyperparameter_description['type']))

        return pipeline_hyperparams

    def _prepare_primitive_hyperparams(self, step_index: int, step: pipeline_module.PrimitiveStep) -> typing.Tuple[hyperparams_module.Hyperparams, typing.Dict]:
        default_hyperparams = self._get_default_hyperparams(step_index, step)
        pipeline_hyperparams = self._get_pipeline_hyperparams(step_index, step)
        runtime_hyperparams = self._get_runtime_hyperparams(step_index, step)

        # Pipeline hyper-parameters should be disjoint with runtime hyper-parameters.
        # We check this in "_check_hyperparams" call from the constructor.
        assert set(pipeline_hyperparams.keys()).isdisjoint(set(runtime_hyperparams.keys())), (pipeline_hyperparams, runtime_hyperparams)

        hyperparams = default_hyperparams.replace(pipeline_hyperparams).replace(runtime_hyperparams)

        # We have to handle all primitive values present in hyper-parameters.
        return self._handle_primitive_hyperparams(hyperparams, 0), pipeline_hyperparams

    def _filter_arguments(self, primitive_class: typing.Type[base.PrimitiveBase], method_name: str, arguments: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """
        Primitive as a whole gets arguments for all its methods, so here we then filter out
        only those arguments expected by a given method.
        """

        method_arguments = primitive_class.metadata.query()['primitive_code'].get('instance_methods', {}).get(method_name, {}).get('arguments', [])

        filtered_arguments = {}
        for argument_name in method_arguments:
            if argument_name in arguments:
                filtered_arguments[argument_name] = arguments[argument_name]

        return filtered_arguments

    def _get_primitive_volumes(self, primitive_class: typing.Type[base.PrimitiveBase]) -> typing.Dict:
        volumes = {}
        for entry in primitive_class.metadata.get_volumes():
            if self.volumes_dir is None:
                raise exceptions.InvalidArgumentValueError(
                    "Primitive '{primitive_id}' of step {step_index} of pipeline '{pipeline_id}' requires static files (volumes) but volumes are not available.".format(
                        primitive_id=primitive_class.metadata.query()['id'],
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                    ),
                )

            volume_path = os.path.join(self.volumes_dir, entry['file_digest'])
            if not os.path.exists(volume_path):
                raise exceptions.InvalidArgumentValueError(
                    "Primitive '{primitive_id}' of step {step_index} of pipeline '{pipeline_id}' requires static files (volume) but volume for key '{key}' is not available.".format(
                        primitive_id=primitive_class.metadata.query()['id'],
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                        key=entry['key'],
                    ),
                )

            volumes[entry['key']] = volume_path

        return volumes

    def _get_primitive_temporary_directory(self, primitive_class: typing.Type[base.PrimitiveBase]) -> str:
        return tempfile.mkdtemp(dir=self._base_temporary_directory_path)

    def _create_primitive_arguments(self, primitive_class: typing.Type[base.PrimitiveBase], hyperparams: hyperparams_module.Hyperparams, random_seed_offset: int) -> typing.Dict:
        constructor_arguments = {
            'hyperparams': hyperparams,
            # We change the random seed in a deterministic way so that it does not matter in which order we run steps.
            'random_seed': self.random_seed + self.current_step + random_seed_offset,
            'volumes': self._get_primitive_volumes(primitive_class),
            'temporary_directory': self._get_primitive_temporary_directory(primitive_class),
        }

        filtered_arguments = self._filter_arguments(primitive_class, '__init__', constructor_arguments)

        return filtered_arguments

    def _create_primitive(self, primitive_class: typing.Type[base.PrimitiveBase], hyperparams: hyperparams_module.Hyperparams, random_seed_offset: int) -> base.PrimitiveBase:
        """
        Creates an instance of a non-pipeline primitive.

        Constructor call is not recorded in pipeline run.
        """

        arguments = self._create_primitive_arguments(primitive_class, hyperparams, random_seed_offset)

        return primitive_class(**arguments)

    def _clone_primitive(self, primitive: base.PrimitiveBase, random_seed_offset: int) -> base.PrimitiveBase:
        """
        Clone a primitive. It reuses hyper-parameters and params, but provides a
        potentially different random seed and other constructor arguments.

        We are creating a new instance and not a deep copy because primitive instance might have
        been created outside of the runtime and might not have valid constructor argument values.
        """

        # We have to handle all primitive values present in hyper-parameters.
        # They are all already an instance, but we have to make their copies.
        hyperparams = self._handle_primitive_hyperparams(primitive.hyperparams, random_seed_offset + 1)

        primitive_clone = self._create_primitive(type(primitive), hyperparams, random_seed_offset)

        primitive_clone.set_params(params=primitive.get_params())

        return primitive_clone

    def _create_pipeline_primitive(self, primitive_class: typing.Type[base.PrimitiveBase], hyperparams: hyperparams_module.Hyperparams) -> base.PrimitiveBase:
        """
        Creates an instance of a pipeline primitive.

        Constructor call is recorded in pipeline run.
        """

        assert self.pipeline_run is not None

        arguments = self._create_primitive_arguments(primitive_class, hyperparams, 0)

        if 'random_seed' in arguments:
            self.pipeline_run.set_primitive_step_random_seed(self.current_step, arguments['random_seed'])

        return self._call_primitive_method(primitive_class, arguments)

    def _create_hyperparameter_primitive(self, primitive_class: typing.Type[base.PrimitiveBase], random_seed_offset: int) -> base.PrimitiveBase:
        """
        Creates an instance of the non-pipeline primitive with default hyper-parameters.
        """

        hyperparams_class = primitive_class.metadata.get_hyperparams()

        return self._create_primitive(primitive_class, hyperparams_class.defaults(), random_seed_offset)

    def _create_primitive_reference_primitive(self, primitive_reference: int, hyperparameter_name: str) -> base.PrimitiveBase:
        """
        Creates an instance of a primitive based on its primitive reference (step index), meaning the instance
        of a primitive is almost the same as the pipeline primitive (see "_create_pipeline_primitive") at that
        step index, but with a different random seed because of a probably different "current_step".

        Constructor call is not recorded in pipeline run.
        """

        # It could point to a sub-pipeline and not primitive.
        if not isinstance(self.pipeline.steps[primitive_reference], pipeline_module.PrimitiveStep):
            raise exceptions.InvalidPipelineError(
                "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' does not point to a primitive step (step {primitive_reference}).".format(  # noqa
                    hyperparameter_name=hyperparameter_name,
                    step_index=self.current_step,
                    pipeline_id=self.pipeline.id,
                    primitive_reference=primitive_reference,
                ),
            )

        step = typing.cast(pipeline_module.PrimitiveStep, self.pipeline.steps[primitive_reference])
        if step.primitive is None:
            raise exceptions.InvalidStateError("Primitive has not been resolved.")
        hyperparams, pipeline_hyperparams = self._prepare_primitive_hyperparams(primitive_reference, step)
        # We use 0 for "random_seed_offset" because we are creating a primitive instance
        # which should be the same as the pipeline primitive (see "_create_pipeline_primitive").
        primitive = self._create_primitive(step.primitive, hyperparams, 0)
        primitive.set_params(params=self.steps_state[primitive_reference])
        return primitive

    def _transform_primitive_hyperparameter(self, hyperparameter: hyperparams_module.Hyperparameter, value: typing.Any, index: int) -> typing.Any:
        value_is_type = utils.is_type(value)
        if value_is_type and issubclass(value, base.PrimitiveBase):
            return self._create_hyperparameter_primitive(value, index)
        elif not value_is_type and isinstance(value, base.PrimitiveBase):
            return self._clone_primitive(value, index)
        else:
            # Not a primitive instance or a primitive class, do not do anything.
            return value

    def _handle_primitive_hyperparams(self, hyperparams: base.Hyperparams, random_seed_offset: int) -> base.Hyperparams:
        """
        Handles a special case when the value is a primitive instance or a primitive class.
        In this case we have to make sure we create a new instance reusing its hyper-parameters,
        or create an instance from the class using default hyper-parameters.
        """

        return hyperparams.transform_value(hyperparams, self._transform_primitive_hyperparameter, random_seed_offset)

    def _run_primitive(self, step: pipeline_module.PrimitiveStep) -> None:
        assert self.pipeline_run is not None

        if step.primitive is None:
            raise exceptions.InvalidPipelineError("Primitive has not been resolved.")

        self.pipeline_run.add_primitive_step(step)
        arguments = self._prepare_primitive_arguments(step)

        hyperparams, pipeline_hyperparams = self._prepare_primitive_hyperparams(self.current_step, step)

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            self.pipeline_run.set_primitive_step_hyperparams(self.current_step, hyperparams, pipeline_hyperparams)

        # We create a primitive just before it is being run. This assures that any primitives it depends on through its
        # hyper-parameters have already been run (because they are in prior steps). Similarly, any pipeline-based value
        # being passed to a hyper-parameter has already been computed.
        primitive = self._create_pipeline_primitive(step.primitive, hyperparams)

        # If primitive step has no arguments we do not fit or produce it. It is meant to be used as
        # unfitted primitive for another primitive's hyper-parameter.
        if not arguments:
            return

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            assert self.steps_state[self.current_step] is None
        else:
            primitive.set_params(params=self.steps_state[self.current_step])

        arguments_set = set(arguments.keys())
        # Required arguments are all arguments required by produce methods used in step outputs and "set_training_data".
        required_arguments = step._get_required_arguments()
        instance_methods = step.primitive.metadata.query()['primitive_code'].get('instance_methods', {})

        # This should already be checked by "PrimitiveStep.check_add", but we check it here as well,
        # to provide a more user friendly error if somebody is subclassing the runtime breaking this.
        missing_arguments = required_arguments - arguments_set
        if missing_arguments:
            raise exceptions.InvalidArgumentValueError(
                "Not all required arguments are provided for the primitive: {missing_arguments}".format(
                    missing_arguments=missing_arguments,
                )
            )

        # "multi_produce" and "fit_multi_produce" accept all possible arguments to the primitive.
        # But if not all produce methods are being called, some of those arguments are not
        # necessary and we pass "None" for them. We know that at this point "arguments" include
        # arguments for all produce methods used in step outputs and "set_training_data".
        # So we can iterate over all produce methods (which might include those not among step outputs)
        # and set value for any missing arguments to "None".
        for method_name, method_metadata in instance_methods.items():
            if method_metadata['kind'] == metadata_base.PrimitiveMethodKind.PRODUCE:
                for argument in method_metadata['arguments']:
                    if argument not in arguments:
                        arguments[argument] = None

        # Get the produce methods from the union of the step outputs and outputs to expose.
        produce_methods = list(step.outputs)
        step_reference_prefix = 'steps.{i}.'.format(i=step.index)
        for output_to_expose in self.outputs_to_expose:
            if output_to_expose.startswith(step_reference_prefix):
                produce_method = output_to_expose[len(step_reference_prefix):]
                # Produce method should not contain a dot.
                assert '.' not in produce_method, produce_method
                if produce_method not in produce_methods:
                    produce_methods.append(produce_method)

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            fit_multi_produce_arguments = self._filter_arguments(step.primitive, 'fit_multi_produce', dict(arguments, produce_methods=produce_methods))

            # We fit and produce once, without any limits on iterations/time.
            multi_call_result = self._call_primitive_method(primitive.fit_multi_produce, fit_multi_produce_arguments)
            if not multi_call_result.has_finished:
                # Because we have not set any limits on iterations/time, the primitive should finish and not stop early.
                # One should be able to control through a hyper-parameter or hyper-parameters stopping criteria for the primitive.
                raise exceptions.InvalidReturnValueError(
                    "\"fit_multi_produce\" call result should have \"has_finished\" set to true because iterations/time limits were set and the primitive should finish and not stop early.",
                )
            outputs = multi_call_result.values

        elif self.phase == metadata_base.PipelineRunPhase.PRODUCE:
            multi_produce_arguments = self._filter_arguments(step.primitive, 'multi_produce', dict(arguments, produce_methods=produce_methods))

            # We produce once, without any limits on iterations/time.
            multi_call_result = self._call_primitive_method(primitive.multi_produce, multi_produce_arguments)
            if not multi_call_result.has_finished:
                # Because we have not set any limits on iterations/time, the primitive should finish and not stop early.
                # One should be able to control through a hyper-parameter or hyper-parameters stopping criteria for the primitive.
                raise exceptions.InvalidReturnValueError(
                    "\"multi_produce\" call result should have \"has_finished\" set to true because iterations/time limits were set and the primitive should finish and not stop early.",
                )
            outputs = multi_call_result.values

        else:
            # TODO: Allow dispatch to a general method so that subclasses of this class can handle them if necessary.
            raise exceptions.UnexpectedValueError("Unknown phase: {phase}".format(phase=self.phase))

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            assert self.steps_state[self.current_step] is None
            self.steps_state[self.current_step] = primitive.get_params()

        for output_id in produce_methods:
            output_data_reference = 'steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)

            if output_id in outputs:
                self.data_values[output_data_reference] = outputs[output_id]
            else:
                raise exceptions.InvalidReturnValueError("Missing declared output '{output_id}' in computed primitive's outputs.".format(output_id=output_id))

    def _call_primitive_method(self, method: typing.Callable, arguments: typing.Dict) -> typing.Any:
        """
        Calls a primitive method (or constructor). Records relevant information in pipeline run.

        Parameters
        ----------
        method:
            Primitive's method or constructor to call.
        arguments:
            Arguments to pass to the method.

        Returns
        -------
        The result of calling the method. It method is a constructor,
        returns an instance.
        """

        assert self.pipeline_run is not None

        # A special case for the constructor.
        if inspect.isclass(method):
            method_name = '__init__'
        else:
            method_name = method.__name__

        pipeline_run_method_call_id = self.pipeline_run.add_method_call_to_primitive_step(self.current_step, method_name)

        callback = self.pipeline_run.get_method_call_logging_callback(pipeline_run_method_call_id)
        logging_handler = utils.CallbackHandler(callback)

        root = logging.getLogger()
        redirect_logger = logging.getLogger('redirect')

        old_level = root.level
        old_handler_levels = [handler.level for handler in root.handlers]
        old_propagate = redirect_logger.propagate
        try:
            # We are just about to modify the root logger level, so we change levels
            # of all existing handlers to retain same configuration.
            for handler in root.handlers:
                # If existing handler has level already set to something more restrictive than what the
                # root logger has, we do not change that. Otherwise, we set it to the root logger's level.
                if handler.level < old_level:
                    handler.setLevel(old_level)
            # Record all logging which happens during the call.
            root.setLevel(logging.DEBUG)
            root.addHandler(logging_handler)
            # We do not want to print logging from "redirect_logger" because pass-through is enabled, so we
            # disable propagation from it to the root logger (by default there is a stream handler on the root
            # logger which prints all logging) and install our handler directly on the redirect logger.
            redirect_logger.propagate = False
            redirect_logger.addHandler(logging_handler)

            # TODO: All this redirection works in a single thread, what about multi-threaded or async?
            #       Reference engine is single threaded, but maybe a subclass would not be?
            # We redirect all stdout/stderr to logging, but pass it through to stdout/stderr as well.
            with utils.redirect_to_logging(logger=redirect_logger, pass_through=True):
                with utils.global_randomness_warning():
                    self.pipeline_run.method_call_started(pipeline_run_method_call_id)

                    try:
                        result = method(**arguments)
                    except Exception as error:
                        self.pipeline_run.method_call_failed(pipeline_run_method_call_id, traceback.format_exc())

                        raise error

                    self.pipeline_run.method_call_successful(pipeline_run_method_call_id)

        finally:
            # Restore original logging configuration.
            root.removeHandler(logging_handler)
            root.setLevel(old_level)
            for i, level in enumerate(old_handler_levels):
                root.handlers[i].setLevel(level)
            # Just to be consistent, if somebody is doing something with the same logger.
            redirect_logger.propagate = old_propagate
            redirect_logger.removeHandler(logging_handler)

        self.pipeline_run.set_method_call_result_metadata(pipeline_run_method_call_id, result)

        return result

    def _run_step(self, step: pipeline_module.StepBase) -> None:
        if isinstance(step, pipeline_module.PlaceholderStep):
            self._run_placeholder(step)
        elif isinstance(step, pipeline_module.SubpipelineStep):
            self._run_subpipeline(step)
        elif isinstance(step, pipeline_module.PrimitiveStep):
            self._run_primitive(step)
        else:
            # TODO: Allow dispatch to a general method so that subclasses of this class can handle them if necessary.
            raise exceptions.UnexpectedValueError("Unknown step type: {step_type}".format(step_type=type(step)))

    def _do_run_step(self, step: pipeline_module.StepBase) -> None:
        assert self.pipeline_run is not None

        self.pipeline_run.step_started(self.current_step)

        try:
            self._before_step_run()
            self._run_step(step)
            self._after_step_run()
        except Exception as error:
            self.pipeline_run.step_failed(self.current_step, traceback.format_exc())

            raise exceptions.StepFailedError(
                "Step {step_index} for pipeline {pipeline_id} failed.".format(
                    step_index=self.current_step, pipeline_id=self.pipeline.id,
                ),
            ) from error

        self.pipeline_run.step_successful(self.current_step)

    def _do_run(self) -> None:
        for step_index, step in enumerate(self.pipeline.steps):
            self.current_step = step_index

            self._do_run_step(step)

    def _run(
        self, inputs: typing.Sequence[typing.Any], phase: metadata_base.PipelineRunPhase,
        outputs_to_expose: typing.Optional[typing.Iterable[str]]
    ) -> Result:
        if outputs_to_expose is None:
            outputs_to_expose = self._get_all_outputs()
        else:
            # We sort to have deterministic order.
            outputs_to_expose = sorted(set(outputs_to_expose))

        outputs_to_expose = self._check_pipeline(inputs, outputs_to_expose)

        self._initialize_run_state(inputs, phase, outputs_to_expose)

        assert self.pipeline_run is not None
        error: typing.Optional[Exception] = None
        values: typing.Dict = {}

        try:
            self.pipeline_run.run_started()

            try:
                self._do_run()
            except Exception as run_error:
                self.pipeline_run.run_failed(traceback.format_exc())

                error = run_error

            if error is None:
                self.pipeline_run.run_successful()

                self._populate_output_values()

                if self.is_standard_pipeline:
                    self.pipeline_run.set_predictions(self.data_values['outputs.0'])

            values = self._get_exposed_outputs(error)

        finally:
            pipeline_run = self.pipeline_run

            self._clear_run_state()

            return Result(pipeline_run, values, error)

    def _get_exposed_outputs(self, error: typing.Optional[Exception]) -> typing.Dict:
        outputs = {}
        for name in self.outputs_to_expose:
            try:
                outputs[name] = self.data_values[name]
            except KeyError as value_error:
                # We try to return whichever outputs we can, even in the case of an error.
                if error is None:
                    raise value_error

        return outputs

    def _before_step_run(self) -> None:
        pass

    def _after_step_run(self) -> None:
        self._delete_unnecessary_outputs()

    def _delete_unnecessary_outputs(self) -> None:
        outputs_needed = set()

        # Which outputs are explicitly required to be kept until the end?
        for output in self.outputs_to_expose:
            outputs_needed.add(output)

        # Pipeline outputs need step outputs.
        for i, output_description in enumerate(self.pipeline.outputs):
            if 'outputs.{i}'.format(i=i) in self.outputs_to_expose:
                outputs_needed.add(output_description['data'])

        # Future steps also need outputs.
        for step in self.pipeline.steps[self.current_step + 1:]:
            outputs_needed.update(step.get_input_data_references())

        # Pipeline run for a standard pipeline needs predictions.
        if self.is_standard_pipeline:
            outputs_needed.add(self.pipeline.outputs[0]['data'])

        # Delete any output which is not needed anymore.
        # We iterate over a list so that we can change dict while iterating.
        for data_reference in list(self.data_values.keys()):
            if data_reference not in outputs_needed:
                del self.data_values[data_reference]

    @deprecate.arguments('return_values', message="use outputs_to_expose instead")
    def fit(
        self, inputs: typing.Sequence[typing.Any], *, outputs_to_expose: typing.Iterable[str] = None,
        return_values: typing.Iterable[str] = None,
    ) -> Result:
        """
        Does a "fit" phase of the pipeline.

        Parameters
        ----------
        inputs:
            A list of inputs to the pipeline.
        outputs_to_expose:
            Data references of all outputs of all steps to return.
            Requesting a data reference of an output which would otherwise not be produced
            is allowed and it forces that output to be produced, but all inputs necessary
            have to be provided to the primitive, otherwise an error is logged and output
            is skipped. If ``None``, the outputs of the whole pipeline are returned.
        return_values:
            DEPRECATED: use ``outputs_to_expose`` instead.

        Returns
        -------
        A result object with kept values, pipeline run description, and any exception.
        """

        return self._run(inputs, metadata_base.PipelineRunPhase.FIT, outputs_to_expose or return_values)

    @deprecate.arguments('return_values', message="use outputs_to_expose instead")
    def produce(
        self, inputs: typing.Sequence[typing.Any], *, outputs_to_expose: typing.Iterable[str] = None,
        return_values: typing.Iterable[str] = None,
    ) -> Result:
        """
        Does a "produce" phase of the pipeline and returns outputs.

        Parameters
        ----------
        inputs:
            A list of inputs to the pipeline.
        outputs_to_expose:
            Data references of all outputs of all steps to return.
            Requesting a data reference of an output which would otherwise not be produced
            is allowed and it forces that output to be produced, but all inputs necessary
            have to be provided to the primitive, otherwise an error is logged and output
            is skipped. If ``None``, the outputs of the whole pipeline are returned.
        return_values:
            DEPRECATED: use ``outputs_to_expose`` instead.

        Returns
        -------
        A result object with kept values, pipeline run description, and any exception.
        """

        return self._run(inputs, metadata_base.PipelineRunPhase.PRODUCE, outputs_to_expose or return_values)

    def get_params(self) -> typing.List[typing.Union[typing.Any, typing.List]]:
        return self.steps_state

    def set_params(self, params: typing.List[typing.Union[typing.Any, typing.List]]) -> None:
        if not isinstance(params, typing.List):
            raise exceptions.InvalidArgumentValueError("Parameters not a list.")

        self._clear_run_state()
        self.steps_state = params

    def _populate_output_values(self) -> None:
        for i, output_description in enumerate(self.pipeline.outputs):
            # Outputs might not be available because they were not requested to be returned from the run.
            if output_description['data'] in self.data_values:
                self.data_values['outputs.{i}'.format(i=i)] = self.data_values[output_description['data']]

    @classmethod
    def _normalize_dataset_id(cls, dataset_id: str) -> str:
        return DATASET_ID_REGEX.sub('', dataset_id)

    @classmethod
    def _dataset_ids_match(cls, first_dataset_id: str, second_dataset_id: str) -> bool:
        if first_dataset_id == second_dataset_id:
            return True

        if cls._normalize_dataset_id(first_dataset_id) == cls._normalize_dataset_id(second_dataset_id):
            return True

        return False

    @classmethod
    def _mark_columns(cls, problem_inputs: typing.Sequence[typing.Dict], dataset: container.Dataset) -> typing.Tuple[container.Dataset, typing.Sequence[int]]:
        dataset = dataset.copy()
        dataset_id = dataset.metadata.query(())['id']

        marked_problem_indices = []
        for problem_index, problem_input in enumerate(problem_inputs):
            if not cls._dataset_ids_match(problem_input['dataset_id'], dataset_id):
                continue

            marked_problem_indices.append(problem_index)

            for target in problem_input.get('targets', []):
                if target['resource_id'] not in dataset:
                    raise exceptions.NotFoundError(
                        "Error marking target column: dataset does not contain resource with resource ID '{resource_id}'.".format(
                            resource_id=target['resource_id'],
                        ),
                    )
                if not isinstance(dataset[target['resource_id']], container.DataFrame):
                    raise TypeError(
                        "Error marking target column: resource '{resource_id}' is not a DataFrame.".format(
                            resource_id=target['resource_id'],
                        ),
                    )
                if not 0 <= target['column_index'] < dataset[target['resource_id']].shape[1]:
                    raise ValueError(
                        "Error marking target column: resource '{resource_id}' does not have a column with index '{column_index}'.".format(
                            resource_id=target['resource_id'],
                            column_index=target['column_index'],
                        ),
                    )

                dataset.metadata = dataset.metadata.add_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/Target',
                )
                dataset.metadata = dataset.metadata.add_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                )
                # If column is marked as a target, it cannot be attribute as well.
                # This allows one to define in problem description otherwise attribute columns as targets.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/265
                dataset.metadata = dataset.metadata.remove_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                )

            # TODO: Warn if privileged data columns are not set on attributes.
            for privileged_data in problem_input.get('privileged_data', []):
                if privileged_data['resource_id'] not in dataset:
                    raise exceptions.NotFoundError(
                        "Error marking privileged data column: dataset does not contain resource with resource ID '{resource_id}'.".format(
                            resource_id=privileged_data['resource_id'],
                        ),
                    )
                if not isinstance(dataset[privileged_data['resource_id']], container.DataFrame):
                    raise TypeError(
                        "Error marking privileged data column: resource '{resource_id}' is not a DataFrame.".format(
                            resource_id=privileged_data['resource_id'],
                        ),
                    )
                if not 0 <= privileged_data['column_index'] < dataset[privileged_data['resource_id']].shape[1]:
                    raise ValueError(
                        "Error marking privileged data column: resource '{resource_id}' does not have a column with index '{column_index}'.".format(
                            resource_id=privileged_data['resource_id'],
                            column_index=privileged_data['column_index'],
                        ),
                    )

                dataset.metadata = dataset.metadata.add_semantic_type(
                    (privileged_data['resource_id'], metadata_base.ALL_ELEMENTS, privileged_data['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/PrivilegedData',
                )

        return dataset, marked_problem_indices


def _prepare_data_and_scoring_hyperparams(free_hyperparams: typing.Sequence, hyperparameter_values: typing.Dict) -> typing.Tuple[typing.Sequence, typing.Set[str]]:
    """
    Values in ``hyperparameter_values`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    hyperparams: typing.List[typing.Union[typing.Dict, typing.Sequence]] = []

    hyperparameter_values_used = set()

    for free_hyperparams_for_step in free_hyperparams:
        if isinstance(free_hyperparams_for_step, (dict, frozendict.frozendict)):
            values = {}
            for name, hyperparameter in free_hyperparams_for_step.items():
                if name in hyperparameter_values:
                    values[name] = hyperparameter.value_from_json_structure(json.loads(hyperparameter_values[name]))
                    hyperparameter_values_used.add(name)
            hyperparams.append(values)
        elif utils.is_sequence(free_hyperparams_for_step):
            step_hyperparams, step_hyperparameter_values_used = _prepare_data_and_scoring_hyperparams(free_hyperparams_for_step, hyperparameter_values)
            hyperparams.append(step_hyperparams)
            hyperparameter_values_used.update(step_hyperparameter_values_used)
        else:
            raise exceptions.UnexpectedValueError("Unknown hyper-parameters type: {hyperparams_type}".format(hyperparams_type=type(free_hyperparams_for_step)))

    return hyperparams, hyperparameter_values_used


def _get_outputs_to_expose(
    pipeline: pipeline_module.Pipeline, is_standard_pipeline: bool, expose_produced_outputs: bool, outputs_to_expose: typing.Optional[typing.Iterable[str]],
) -> typing.Sequence[str]:
    """
    Determines which exposable outputs should be exposed from the runtime.
    This is based on whether the pipeline is standard and based on those requested.
    """

    data_references: typing.Set[str] = set()

    if outputs_to_expose is not None:
        data_references.update(outputs_to_expose)

    if expose_produced_outputs:
        data_references.update(pipeline.get_producing_outputs())

    if is_standard_pipeline:
        data_references.add('outputs.0')

    # We sort to have deterministic order.
    return sorted(data_references)


# TODO: Add debug logging.
def fit(
    pipeline: pipeline_module.Pipeline, inputs: typing.Sequence[container.Dataset], *,
    problem_description: typing.Optional[problem.Problem], context: metadata_base.Context,
    hyperparams: typing.Sequence = None, random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None,
    runtime_environment: pipeline_run_module.RuntimeEnvironment = None, is_standard_pipeline: bool = True,
    expose_produced_outputs: bool = False, outputs_to_expose: typing.Iterable[str] = None, data_pipeline: pipeline_module.Pipeline = None,
    data_params: typing.Dict[str, str] = None, data_random_seed: int = 0,
    data_pipeline_run: pipeline_run_module.PipelineRun = None,
    fold_group_uuid: uuid.UUID = None, fold_index: int = 0,
) -> typing.Tuple[typing.Optional[Runtime], typing.Optional[container.DataFrame], Result]:
    if data_pipeline is not None and data_pipeline_run is not None:
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    for input in inputs:
        if not isinstance(input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(input),
            ))

    if is_standard_pipeline and len(pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(pipeline.outputs),
        ))

    data_result: typing.Optional[Result]
    if data_pipeline is not None:
        outputs, data_result = prepare_data(
            inputs,
            data_pipeline=data_pipeline,
            problem_description=problem_description,
            data_params=data_params,
            context=context,
            random_seed=data_random_seed,
            volumes_dir=volumes_dir,
            scratch_dir=scratch_dir,
            runtime_environment=runtime_environment,
        )
        if data_result.has_error():
            return None, None, data_result
        if len(outputs[0]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of train data but {folds} folds.".format(
                folds=len(outputs[0]),
            ))
        inputs = [outputs[0][0]]
    else:
        data_result = None

    runtime = Runtime(
        pipeline, hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
        is_standard_pipeline=is_standard_pipeline, environment=runtime_environment,
    )

    outputs_to_expose = _get_outputs_to_expose(pipeline, is_standard_pipeline, expose_produced_outputs, outputs_to_expose)

    result = runtime.fit(inputs, outputs_to_expose=outputs_to_expose)

    if fold_group_uuid is None:
        fold_group_uuid = uuid.uuid4()
    if data_result is not None:
        assert data_pipeline_run is None
        data_pipeline_run = data_result.pipeline_run
    if data_pipeline_run is not None:
        # Modifies "result.pipeline_run" in-place.
        combine_pipeline_runs(
            result.pipeline_run, data_pipeline_run=data_pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index,
        )

    if result.has_error():
        runtime = None  # type: ignore

    return runtime, result.get_standard_pipeline_output(), result


# TODO: Add debug logging.
def produce(
    fitted_pipeline: Runtime, test_inputs: typing.Sequence[container.Dataset], *,
    expose_produced_outputs: bool = False, outputs_to_expose: typing.Iterable[str] = None, data_pipeline: pipeline_module.Pipeline = None,
    data_params: typing.Dict[str, str] = None, data_random_seed: int = 0,
    data_pipeline_run: pipeline_run_module.PipelineRun = None,
    fold_group_uuid: uuid.UUID = None, fold_index: int = 0,
) -> typing.Tuple[typing.Optional[container.DataFrame], Result]:
    if data_pipeline is not None and data_pipeline_run is not None:
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    for test_input in test_inputs:
        if not isinstance(test_input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(test_input),
            ))

    data_result: typing.Optional[Result]
    if data_pipeline is not None:
        outputs, data_result = prepare_data(
            test_inputs,
            data_pipeline=data_pipeline,
            problem_description=fitted_pipeline.problem_description,
            data_params=data_params,
            context=fitted_pipeline.context,
            random_seed=data_random_seed,
            volumes_dir=fitted_pipeline.volumes_dir,
            scratch_dir=fitted_pipeline.scratch_dir,
            runtime_environment=fitted_pipeline.environment,
        )
        if data_result.has_error():
            return None, data_result
        if len(outputs[1]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of test data but {folds} folds.".format(
                folds=len(outputs[1]),
            ))
        test_inputs = [outputs[1][0]]
    else:
        data_result = None

    # This is checked in "fit" already, but maybe somebody fitter a pipeline not through "fit".
    if fitted_pipeline.is_standard_pipeline and len(fitted_pipeline.pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(fitted_pipeline.pipeline.outputs),
        ))

    outputs_to_expose = _get_outputs_to_expose(fitted_pipeline.pipeline, fitted_pipeline.is_standard_pipeline, expose_produced_outputs, outputs_to_expose)

    result = fitted_pipeline.produce(test_inputs, outputs_to_expose=outputs_to_expose)

    if fold_group_uuid is None:
        fold_group_uuid = uuid.uuid4()
    if data_result is not None:
        assert data_pipeline_run is None
        data_pipeline_run = data_result.pipeline_run
    if data_pipeline_run is not None:
        # Modifies "result.pipeline_run" in-place.
        combine_pipeline_runs(
            result.pipeline_run, data_pipeline_run=data_pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index,
        )

    return result.get_standard_pipeline_output(), result


# TODO: Add debug logging.
def score(
    predictions: container.DataFrame, score_inputs: typing.Sequence[container.Dataset], *, scoring_pipeline: pipeline_module.Pipeline,
    problem_description: typing.Optional[problem.Problem], metrics: typing.Sequence[typing.Dict], predictions_random_seed: int = None,
    context: metadata_base.Context, scoring_params: typing.Dict[str, str] = None, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
    data_pipeline: pipeline_module.Pipeline = None, data_params: typing.Dict[str, str] = None, data_random_seed: int = 0,
    data_pipeline_run: pipeline_run_module.PipelineRun = None,
    fold_group_uuid: uuid.UUID = None, fold_index: int = 0,
) -> typing.Tuple[typing.Optional[container.DataFrame], Result]:
    if data_pipeline is not None and data_pipeline_run is not None:
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    for score_input in score_inputs:
        if not isinstance(score_input, container.Dataset):
            raise TypeError("A scoring pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(score_input),
            ))

    if len(scoring_pipeline.outputs) != 1:
        raise ValueError("A scoring pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(scoring_pipeline.outputs),
        ))

    data_result: typing.Optional[Result]
    if data_pipeline is not None:
        outputs, data_result = prepare_data(
            score_inputs,
            data_pipeline=data_pipeline,
            problem_description=problem_description,
            data_params=data_params,
            context=context,
            random_seed=data_random_seed,
            volumes_dir=volumes_dir,
            scratch_dir=scratch_dir,
            runtime_environment=runtime_environment,
        )
        if data_result.has_error():
            return None, data_result
        if len(outputs[2]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of score data but {folds} folds.".format(
                folds=len(outputs[2]),
            ))
        score_inputs = [outputs[2][0]]
    else:
        data_result = None

    metrics_hyperparameter = []
    for metric in metrics:
        # Structure should match what "value_from_json_structure" would
        # return for "ComputeScoresPrimitive" hyper-parameter.
        # TODO: Once "ComputeScoresPrimitive" is moved to core package, use its default hyper-parameters here.
        metric_hyperparameter = {'metric': metric['metric'].name, 'k': None, 'pos_label': None}
        metric_hyperparameter.update(metric.get('params', {}))
        metrics_hyperparameter.append(metric_hyperparameter)

    if scoring_params is None:
        scoring_params = {}

    if metrics_hyperparameter:
        # We have to JSON-serialize it because "_prepare_data_and_scoring_hyperparams"
        # expects all values to be JSON-serialized.
        scoring_params['metrics'] = json.dumps(metrics_hyperparameter)

    scoring_hyperparams, scoring_params_used = _prepare_data_and_scoring_hyperparams(scoring_pipeline.get_free_hyperparams(), scoring_params)

    scoring_params_keys_set = set(scoring_params.keys())
    if scoring_params_keys_set - scoring_params_used:
        logger.warning("Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s", {
            'pipeline_id': scoring_pipeline.id,
            'unused_params': ', '.join(sorted(scoring_params_keys_set - scoring_params_used)),
        })

    runtime = Runtime(
        scoring_pipeline, scoring_hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
        environment=runtime_environment,
    )

    inputs = [predictions] + list(score_inputs)  # type: ignore

    # Fit + produce on same data.
    result = runtime.fit(inputs, outputs_to_expose=['outputs.0'])

    if fold_group_uuid is None:
        fold_group_uuid = uuid.uuid4()
    if data_result is not None:
        assert data_pipeline_run is None
        data_pipeline_run = data_result.pipeline_run
    if data_pipeline_run is not None:
        # Modifies "result.pipeline_run" in-place.
        combine_pipeline_runs(
            result.pipeline_run, data_pipeline_run=data_pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index,
        )

    if result.has_error():
        return None, result

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A scoring pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    if predictions_random_seed is not None:
        output = combine_random_seed(output, predictions_random_seed)

    return output, result


def _get_number_of_folds(data_params: typing.Optional[typing.Dict[str, str]]) -> int:
    if data_params is None:
        return 1
    elif 'number_of_folds' in data_params:
        number_of_folds = int(data_params['number_of_folds'])
        if number_of_folds < 1:
            raise exceptions.InvalidArgumentValueError("Number of folds cannot be less than 1, but it is {number_of_folds}.".format(number_of_folds=number_of_folds))
        return number_of_folds
    else:
        # For now we assume other data preparation pipelines do only one fold. We should standardize
        # more hyper-parameters to gather how many folds have to be made (and not really folds, but
        # more how many input indices have to be passed to the pipeline).
        return 1


# TODO: Add debug logging.
def prepare_data(
    inputs: typing.Sequence[container.Dataset], *, data_pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem],
    data_params: typing.Dict[str, str] = None, context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.List, Result]:
    """
    This function calls a data preparation pipeline. That pipeline can take as input one or more datasets but must always return only
    one dataset split into training, testing, and scoring splits (e.g., the pipeline combines multiple input datasets).
    Each split can be across multiple folds. So the data preparation pipeline must have three pipeline outputs, each
    returning a list of datasets, where every list item corresponds to a fold index.

    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    if data_params is None:
        data_params = {}

    for input in inputs:
        if not isinstance(input, container.Dataset):
            raise TypeError("A data preparation pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(input),
            ))

    if len(data_pipeline.outputs) != 3:
        raise ValueError("A data preparation pipeline should have exactly three outputs, not {outputs}.".format(
            outputs=len(data_pipeline.outputs),
        ))

    number_of_folds = _get_number_of_folds(data_params)
    assert number_of_folds != 0

    data_hyperparams, data_params_used = _prepare_data_and_scoring_hyperparams(data_pipeline.get_free_hyperparams(), data_params)

    data_params_keys_set = set(data_params.keys())
    if data_params_keys_set - data_params_used:
        logger.warning("Not all provided hyper-parameters for the data preparation pipeline {pipeline_id} were used: {unused_params}".format(
            pipeline_id=data_pipeline.id,
            unused_params=sorted(data_params_keys_set - data_params_used),
        ))

    runtime = Runtime(
        data_pipeline, data_hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, environment=runtime_environment,
    )

    # Fit + produce on same data. The inputs are the list of indices of folds
    # to generate and a dataset to split.
    result = runtime.fit([container.List(range(number_of_folds))] + list(inputs), outputs_to_expose=['outputs.0', 'outputs.1', 'outputs.2'])
    # This is not set by the runtime because not all inputs are datasets. So we set input datasets here.
    for input_value in inputs:
        result.pipeline_run.add_input_dataset(input_value)
    if result.has_error():
        return [], result

    outputs = [result.values['outputs.0'], result.values['outputs.1'], result.values['outputs.2']]

    for output in outputs:
        if not isinstance(output, container.List):
            raise TypeError("A data preparation pipeline's output should be of a container List type, not {input_type}.".format(
                input_type=type(output),
            ))
        if len(output) != number_of_folds:
            raise ValueError("A data preparation pipeline's output should contain {number_of_folds} datasets, not {length}.".format(
                number_of_folds=number_of_folds,
                length=len(output),
            ))
        # This holds because of number_of_folds != 0 assert.
        assert len(output)
        dataset_id = output[0].metadata.query_field((), 'id')
        for dataset in output[1:]:
            second_dataset_id = dataset.metadata.query_field((), 'id')
            if dataset_id != second_dataset_id:
                raise ValueError(
                    f"All datasets (for all folds) for the same split type returned from the data preparation pipeline should have the same dataset ID, "
                    f"but one has \"{dataset_id}\" and another \"{second_dataset_id}\".",
                )

    return outputs, result


# TODO: Add debug logging.
def evaluate(
    pipeline: pipeline_module.Pipeline, inputs: typing.Sequence[container.Dataset], *, data_pipeline: pipeline_module.Pipeline,
    scoring_pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem],
    data_params: typing.Dict[str, str] = None, metrics: typing.Sequence[typing.Dict], context: metadata_base.Context,
    scoring_params: typing.Dict[str, str] = None, hyperparams: typing.Sequence = None, random_seed: int = 0,
    data_random_seed: int = 0, scoring_random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.List[container.DataFrame], MultiResult]:
    """
    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    outputs, data_result = prepare_data(
        inputs, data_pipeline=data_pipeline, problem_description=problem_description, data_params=data_params,
        context=context, random_seed=data_random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, runtime_environment=runtime_environment,
    )
    if data_result.has_error():
        return [], MultiResult([data_result])

    fold_group_uuid = uuid.uuid4()

    all_scores: typing.List[container.DataFrame] = []
    all_results = MultiResult()
    for fold_index, (train_inputs, test_inputs, score_inputs) in enumerate(zip(*outputs)):
        evaluate_fold(
            pipeline, [train_inputs], [test_inputs], [score_inputs], all_scores, all_results,
            fold_index=fold_index, fold_group_uuid=fold_group_uuid, data_pipeline_run=data_result.pipeline_run,
            scoring_pipeline=scoring_pipeline, problem_description=problem_description,
            metrics=metrics, context=context,
            scoring_params=scoring_params, hyperparams=hyperparams, random_seed=random_seed,
            scoring_random_seed=scoring_random_seed, volumes_dir=volumes_dir,
            scratch_dir=scratch_dir, runtime_environment=runtime_environment,
        )

        if all_results.has_error():
            break

    return all_scores, all_results


def evaluate_fold(
    pipeline: pipeline_module.Pipeline, train_inputs: typing.Sequence[container.Dataset],
    test_inputs: typing.Sequence[container.Dataset], score_inputs: typing.Sequence[container.Dataset],
    all_scores: typing.List[container.DataFrame], all_results: MultiResult, *,
    data_pipeline_run: pipeline_run_module.PipelineRun, fold_group_uuid: uuid.UUID, fold_index: int,
    scoring_pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem],
    metrics: typing.Sequence[typing.Dict], context: metadata_base.Context,
    scoring_params: typing.Dict[str, str] = None, hyperparams: typing.Sequence = None, random_seed: int = 0,
    scoring_random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> None:
    fitted_pipeline, predictions, fit_result = fit(
        pipeline, train_inputs, problem_description=problem_description, context=context, hyperparams=hyperparams,
        random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
        runtime_environment=runtime_environment, data_pipeline_run=data_pipeline_run,
        fold_group_uuid=fold_group_uuid, fold_index=fold_index,
    )

    all_results.append(fit_result)
    if fit_result.has_error():
        assert all_results.has_error()
        return

    assert fitted_pipeline is not None

    predictions, produce_result = produce(
        fitted_pipeline, test_inputs,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=fold_group_uuid, fold_index=fold_index,
    )

    all_results.append(produce_result)
    if produce_result.has_error():
        assert all_results.has_error()
        return

    assert predictions is not None

    scores, score_result = score(
        predictions, score_inputs, scoring_pipeline=scoring_pipeline, problem_description=problem_description, metrics=metrics,
        predictions_random_seed=random_seed, scoring_params=scoring_params, context=context, random_seed=scoring_random_seed,
        volumes_dir=volumes_dir, scratch_dir=scratch_dir, runtime_environment=runtime_environment,
    )

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run,
    )
    # Sets the error, if there are any.
    produce_result.error = score_result.error

    # We modified "produce_result.pipeline_run" in-place and "produce_result"
    # is already among "all_results", so we do not add it again.
    if score_result.has_error():
        assert all_results.has_error()
        return

    assert scores is not None

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, metrics=metrics, scores=scores,
    )

    all_scores.append(scores)


is_uri = deprecate.function(message="use d3m.utils.is_uri instead")(utils.is_uri)

get_dataset = deprecate.function(message="use d3m.container.dataset.get_dataset instead")(dataset_module.get_dataset)
get_problem = deprecate.function(message="use d3m.metadata.problem.get_problem instead")(problem.get_problem)
get_pipeline = deprecate.function(message="use d3m.metadata.pipeline.get_pipeline instead")(pipeline_module.get_pipeline)


@deprecate.function(message="use d3m.utils.get_datasets_and_problems instead")
def _get_datasets_and_problems(
    datasets_dir: str, handle_score_split: bool = True,
) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, str]]:
    return utils.get_datasets_and_problems(datasets_dir, handle_score_split)


def _resolve_pipeline_run_datasets(
    pipeline_run_datasets: typing.Sequence[typing.Dict[str, str]], *,
    dataset_resolver: typing.Callable, compute_digest: dataset_module.ComputeDigest, strict_digest: bool,
    strict_resolving: bool, datasets_dir: typing.Optional[str], handle_score_split: bool,
) -> typing.Sequence[container.Dataset]:
    resolved_datasets = []

    for dataset_reference in pipeline_run_datasets:
        resolved_dataset = dataset_resolver(
            dataset_reference['id'], compute_digest=compute_digest, strict_digest=strict_digest,
            datasets_dir=datasets_dir, handle_score_split=handle_score_split,
        )

        resolved_dataset_digest = resolved_dataset.metadata.query(()).get('digest', None)

        if resolved_dataset_digest != dataset_reference['digest']:
            if strict_resolving:
                raise exceptions.DigestMismatchError(
                    "Digest for dataset '{dataset_id}' does not match the one specified in the dataset reference. "
                    "Dataset reference digest: {dataset_digest}. Resolved dataset digest: {resolved_dataset_digest}.".format(
                        dataset_id=dataset_reference['id'],
                        dataset_digest=dataset_reference['digest'],
                        resolved_dataset_digest=resolved_dataset_digest,
                    )
                )
            else:
                logger.warning(
                    "Digest for dataset '%(dataset_id)s' does not match the one specified in the dataset reference. "
                    "Dataset reference digest: %(dataset_digest)s. Resolved dataset digest: %(resolved_dataset_digest)s.",
                    {
                        'dataset_id': dataset_reference['id'],
                        'dataset_digest': dataset_reference['digest'],
                        'resolved_dataset_digest': resolved_dataset_digest,
                    },
                )

        resolved_datasets.append(resolved_dataset)

    return resolved_datasets


def parse_pipeline_run(
    pipeline_run_file: typing.IO[typing.Any], pipeline_search_paths: typing.Sequence[str], datasets_dir: typing.Optional[str], *,
    pipeline_resolver: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None, strict_resolving: bool = False,
    compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False, handle_score_split: bool = True,
) -> typing.Sequence[typing.Dict[str, typing.Any]]:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    pipeline_runs = list(utils.yaml_load_all(pipeline_run_file))

    if not pipeline_runs:
        raise exceptions.InvalidArgumentValueError("Pipeline run file must contain at least one pipeline run document.")

    for pipeline_run in pipeline_runs:
        try:
            pipeline_run_module.validate_pipeline_run(pipeline_run)
        except jsonschema.exceptions.ValidationError as error:
            raise exceptions.InvalidArgumentValueError("Provided pipeline run document is not valid.") from error

        pipeline_run['datasets'] = _resolve_pipeline_run_datasets(
            pipeline_run['datasets'], dataset_resolver=dataset_resolver,
            compute_digest=compute_digest, strict_digest=strict_digest,
            strict_resolving=strict_resolving, datasets_dir=datasets_dir,
            handle_score_split=handle_score_split,
        )

        if 'problem' in pipeline_run:
            pipeline_run['problem'] = problem_resolver(
                pipeline_run['problem']['id'],
                strict_digest=strict_digest,
                datasets_dir=datasets_dir,
                handle_score_split=handle_score_split,
            )

        pipeline_run['pipeline'] = pipeline_resolver(
            pipeline_run['pipeline']['id'],
            strict_resolving=strict_resolving,
            strict_digest=strict_digest,
            pipeline_search_paths=pipeline_search_paths,
        )

        if 'data_preparation' in pipeline_run['run']:
            pipeline_run['run']['data_preparation']['pipeline'] = pipeline_resolver(
                pipeline_run['run']['data_preparation']['pipeline']['id'],
                strict_resolving=strict_resolving,
                strict_digest=strict_digest,
                pipeline_search_paths=pipeline_search_paths,
            )

        if 'scoring' in pipeline_run['run']:
            if 'datasets' in pipeline_run['run']['scoring']:
                assert 'data_preparation' not in pipeline_run['run']
                pipeline_run['run']['scoring']['datasets'] = _resolve_pipeline_run_datasets(
                    pipeline_run['run']['scoring']['datasets'], dataset_resolver=dataset_resolver,
                    compute_digest=compute_digest, strict_digest=strict_digest, strict_resolving=strict_resolving,
                    datasets_dir=datasets_dir, handle_score_split=handle_score_split,
                )

            if pipeline_run['run']['scoring']['pipeline']['id'] == DEFAULT_SCORING_PIPELINE_ID:
                pipeline_run['run']['scoring']['pipeline'] = pipeline_resolver(
                    DEFAULT_SCORING_PIPELINE_PATH,
                    strict_resolving=strict_resolving,
                    strict_digest=strict_digest,
                    pipeline_search_paths=pipeline_search_paths,
                )
            else:
                pipeline_run['run']['scoring']['pipeline'] = pipeline_resolver(
                    pipeline_run['run']['scoring']['pipeline']['id'],
                    strict_resolving=strict_resolving,
                    strict_digest=strict_digest,
                    pipeline_search_paths=pipeline_search_paths,
                )

    return pipeline_runs


def _get_runtime_hyperparams_from_pipeline_run(pipeline: pipeline_module.Pipeline, pipeline_run_steps: typing.Sequence[typing.Dict]) -> typing.Sequence[typing.Union[typing.Dict, typing.Sequence]]:
    free_hyperparams = pipeline.get_free_hyperparams()

    # We want to allow missing steps for failed pipeline runs.
    if len(free_hyperparams) >= len(pipeline_run_steps):
        pipeline_run_steps = list(pipeline_run_steps)
        for i in range(len(pipeline_run_steps), len(free_hyperparams)):
            pipeline_run_steps.append({})
    else:
        raise exceptions.InvalidPipelineRunError("Number of steps in the pipeline run does not match the number of steps of the pipeline.")

    hyperparams: typing.List[typing.Union[typing.Dict, typing.Sequence]] = []

    for free_hyperparams_for_step, pipeline_run_step in zip(free_hyperparams, pipeline_run_steps):
        if isinstance(free_hyperparams_for_step, (dict, frozendict.frozendict)):
            values = {}
            hyperparams_from_step = pipeline_run_step.get('hyperparams', {})
            for name, hyperparameter in free_hyperparams_for_step.items():
                if name in hyperparams_from_step:
                    if hyperparams_from_step[name]['type'] == metadata_base.ArgumentType.VALUE.name:
                        values[name] = hyperparameter.value_from_json_structure(hyperparams_from_step[name]['data'])
                    else:
                        raise exceptions.UnexpectedValueError("Hyper-parameter '{name}' of type '{type}' cannot be set at runtime.".format(name=name, type=hyperparams_from_step[name]['type']))
            hyperparams.append(values)

            extra_hyperparams_set = set(hyperparams_from_step.keys()) - set(free_hyperparams_for_step.keys())
            if extra_hyperparams_set:
                logger.warning("Pipeline run contains values for additional hyper-parameters: %(extra_hyperparams)s", {
                    'extra_hyperparams': sorted(extra_hyperparams_set),
                })

        elif utils.is_sequence(free_hyperparams_for_step):
            step_hyperparams = _get_runtime_hyperparams_from_pipeline_run(free_hyperparams_for_step, pipeline_run_step.get('steps', []))
            hyperparams.append(step_hyperparams)
        else:
            raise exceptions.UnexpectedValueError("Unknown hyper-parameters type: {hyperparams_type}".format(hyperparams_type=type(free_hyperparams_for_step)))

    return hyperparams


def _get_data_and_scoring_params_from_pipeline_run(pipeline_run_steps: typing.Sequence[typing.Dict]) -> typing.Dict:
    params: typing.Dict[str, typing.Any] = {}

    for pipeline_run_step in pipeline_run_steps:
        if pipeline_run_step['type'] == metadata_base.PipelineStepType.PRIMITIVE.name:
            new_params = {}

            for hyperparameter_name, hyperparameter in pipeline_run_step.get('hyperparams', {}).items():
                if hyperparameter['type'] == metadata_base.ArgumentType.VALUE.name:
                    # We are comparing JSON serializations, so we need it to be deterministic, so we sort keys.
                    new_params[hyperparameter_name] = json.dumps(hyperparameter['data'], sort_keys=True)
                else:
                    raise exceptions.UnexpectedValueError("Hyper-parameter '{name}' of type '{type}' cannot be set at runtime.".format(name=hyperparameter_name, type=hyperparameter['type']))

        elif pipeline_run_step['type'] == metadata_base.PipelineStepType.SUBPIPELINE.name:
            new_params = _get_data_and_scoring_params_from_pipeline_run(pipeline_run_step.get('steps', []))

        else:
            raise exceptions.UnexpectedValueError("Unknown step type: {step_type}".format(step_type=pipeline_run_step['type']))

        for name, value in new_params.items():
            if name in params:
                if params[name] != value:
                    raise exceptions.UnexpectedValueError(
                        "Hyper-parameter '{name}' does not have the same value across the whole pipeline: {value1} vs {value2}.".format(
                            name=name, value1=params[name], value2=value,
                        ),
                    )
            else:
                params[name] = value

    return params


def combine_random_seed(scores: container.DataFrame, random_seed: int) -> container.DataFrame:
    random_seed_column = container.DataFrame({'randomSeed': [random_seed] * scores.shape[0]})
    # We add the new column at the end so that we do not have to do complicated changes to the metadata.
    output_scores = pandas.concat([scores, random_seed_column], axis=1)
    # There is one more column now, so we update metadata for it.
    output_scores.metadata = scores.metadata.update((metadata_base.ALL_ELEMENTS,), {
        'dimension': {
            'length': output_scores.shape[1],
        },
    })
    output_scores.metadata = output_scores.metadata.update_column(output_scores.shape[1] - 1, {
        'name': 'randomSeed',
        'structural_type': int,
    })

    return output_scores


def combine_folds(scores_list: typing.List[container.DataFrame]) -> container.DataFrame:
    # We combine multiple scores tables into one output table by adding a "fold" column.
    for fold, scores in enumerate(scores_list):
        fold_column = container.DataFrame({'fold': [fold] * scores.shape[0]})
        # We add the new column at the end so that we do not have to do complicated
        # changes to the metadata.
        scores_list[fold] = pandas.concat([scores, fold_column], axis=1)
        # There is one more column now, so we update metadata for it.
        scores_list[fold].metadata = scores.metadata.update((metadata_base.ALL_ELEMENTS,), {
            'dimension': {
                'length': scores_list[fold].shape[1],
            },
        })
        scores_list[fold].metadata = scores_list[fold].metadata.update_column(scores_list[fold].shape[1] - 1, {
            'name': 'fold',
            'structural_type': int,
        })

    scores = pandas.concat(scores_list, axis=0).reset_index(drop=True)
    # We reuse metadata from the first fold and update the number of rows which is now
    # combined across all folds.
    scores.metadata = scores_list[0].metadata.update((), {
        'dimension': {
            'length': scores.shape[0],
        },
    })

    return scores


def combine_pipeline_runs(
    standard_pipeline_run: pipeline_run_module.PipelineRun, *,
    data_pipeline_run: pipeline_run_module.PipelineRun = None, scoring_pipeline_run: pipeline_run_module.PipelineRun = None,
    score_inputs: typing.Sequence[typing.Any] = None, metrics: typing.Sequence[typing.Dict] = None, scores: container.DataFrame = None,
    fold_group_uuid: uuid.UUID = None, fold_index: int = None,
) -> None:
    fold_args_provided = (item is None for item in (fold_group_uuid, fold_index))
    if any(fold_args_provided) and not all(fold_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'fold_group_uuid' and 'fold_index' are provided, they must all be provided.")

    scores_args_provided = (item is None for item in (scores, metrics))
    if any(scores_args_provided) and not all(scores_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'scores' or 'metrics' is provided, they must both be provided.")

    if data_pipeline_run is not None:
        standard_pipeline_run.set_data_preparation_pipeline_run(data_pipeline_run)

    if fold_group_uuid is not None:
        assert fold_index is not None
        standard_pipeline_run.set_fold_group(fold_group_uuid, fold_index)

    if scoring_pipeline_run is not None:
        standard_pipeline_run.set_scoring_pipeline_run(scoring_pipeline_run, score_inputs)

    if scores is not None:
        assert metrics is not None
        standard_pipeline_run.set_scores(scores, metrics)


@deprecate.function(message="use extended DataFrame.to_csv method instead")
def export_dataframe(dataframe: container.DataFrame, output_file: typing.IO[typing.Any] = None) -> typing.Optional[str]:
    return dataframe.to_csv(output_file)


def _check_duplicate_metrics(metrics: typing.Sequence[typing.Dict]) -> None:
    """
    In results from scoring we identify each score by its metric name. So to map those rows in scoring
    output back to requested metrics, names must be unique. Otherwise we would not know to which
    metric configuration the score belongs to.
    """

    only_metrics = [metric['metric'] for metric in metrics]

    if utils.has_duplicates(only_metrics):
        raise exceptions.InvalidArgumentValueError("Same metric listed multiple times.")


def get_metrics_from_list(metrics: typing.Sequence[str]) -> typing.Sequence[typing.Dict]:
    metric_descriptions = [{'metric': problem.PerformanceMetric[metric]} for metric in metrics]

    _check_duplicate_metrics(metric_descriptions)

    return metric_descriptions


def get_metrics_from_problem_description(problem_description: typing.Optional[problem.Problem]) -> typing.Sequence[typing.Dict]:
    if problem_description is None:
        return []

    metric_descriptions = problem_description['problem'].get('performance_metrics', [])

    _check_duplicate_metrics(metric_descriptions)

    return metric_descriptions


def _output_pipeline_runs(arguments: argparse.Namespace, pipeline_runs: typing.Sequence[pipeline_run_module.PipelineRun]) -> None:
    if not getattr(arguments, 'output_run', None):
        return

    first = True
    for pipeline_run in pipeline_runs:
        pipeline_run.to_yaml(arguments.output_run, appending=not first)
        first = False

    # Make sure the handle is flushed so that no data is lost. CLI file handles are generally
    # used outside of a context manager which would otherwise handle that.
    # See: https://gitlab.com/datadrivendiscovery/d3m/issues/436
    arguments.output_run.flush()


class InputsConfig(typing.NamedTuple):
    # This can be or train data or full data (if data pipeline is used as well).
    inputs: typing.Sequence[container.Dataset]
    test_inputs: typing.Sequence[container.Dataset]
    score_inputs: typing.Sequence[container.Dataset]
    data_pipeline: typing.Optional[pipeline_module.Pipeline]
    data_params: typing.Optional[typing.Dict[str, str]]
    data_random_seed: int


def _get_inputs_config_from_arguments(
    *, arguments: argparse.Namespace, pipeline_resolver: typing.Callable, dataset_resolver: typing.Callable,
    fit_pipeline_run: typing.Dict[str, typing.Any] = None, produce_pipeline_run: typing.Dict[str, typing.Any] = None,
) -> InputsConfig:
    """
    A helper to retrieve the inputs from CLI arguments. In addition to inputs also a preparation pipeline and
    its parameters can be returned if configured through CLI arugments.
    """

    inputs_config: typing.Dict[str, typing.Any] = {
        'inputs': None,
        'test_inputs': None,
        'score_inputs': None,
        'data_pipeline': None,
        'data_params': {},
        'data_random_seed': 0,
    }

    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]

    if fit_pipeline_run is not None:
        inputs_config['inputs'] = fit_pipeline_run['datasets']
    else:
        if getattr(arguments, 'inputs', []):
            inputs_config['inputs'] = [
                dataset_resolver(
                    input_uri, compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
                )
                for input_uri in arguments.inputs
            ]
        elif getattr(arguments, 'full_inputs', []):
            inputs_config['inputs'] = [
                dataset_resolver(
                    input_uri, compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
                )
                for input_uri in arguments.full_inputs
            ]

    if produce_pipeline_run is not None:
        inputs_config['test_inputs'] = produce_pipeline_run['datasets']
        if 'scoring' in produce_pipeline_run['run'] and 'datasets' in produce_pipeline_run['run']['scoring']:
            inputs_config['score_inputs'] = produce_pipeline_run['run']['scoring']['datasets']
    else:
        if getattr(arguments, 'test_inputs', []):
            inputs_config['test_inputs'] = [
                dataset_resolver(
                    input_uri, compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
                )
                for input_uri in arguments.test_inputs
            ]
        elif getattr(arguments, 'full_inputs', []):
            inputs_config['test_inputs'] = [
                dataset_resolver(
                    input_uri, compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
                )
                for input_uri in arguments.full_inputs
            ]
        if getattr(arguments, 'score_inputs', []):
            inputs_config['score_inputs'] = [
                dataset_resolver(
                    input_uri, compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
                )
                for input_uri in arguments.score_inputs
            ]
        elif getattr(arguments, 'full_inputs', []):
            inputs_config['score_inputs'] = [
                dataset_resolver(
                    input_uri, compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
                )
                for input_uri in arguments.full_inputs
            ]

    # TODO: Check that data preparation pipelines match between both pipeline runs.
    #       Try to reuse code from _process_pipeline_runs_for_evaluate_handler.
    pipeline_run = fit_pipeline_run or produce_pipeline_run
    if pipeline_run is not None:
        if 'data_preparation' in pipeline_run['run']:
            inputs_config['data_pipeline'] = pipeline_run['run']['data_preparation']['pipeline']
            # Currently, "random_seed" is not yet required.
            inputs_config['data_random_seed'] = pipeline_run['run']['data_preparation'].get('random_seed', 0)
            inputs_config['data_params'] = _get_data_and_scoring_params_from_pipeline_run(pipeline_run['run']['data_preparation'].get('steps', []))
    else:
        if getattr(arguments, 'data_pipeline', None):
            inputs_config['data_pipeline'] = pipeline_resolver(
                arguments.data_pipeline,
                strict_resolving=getattr(arguments, 'strict_resolving', False),
                strict_digest=getattr(arguments, 'strict_digest', False),
                pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
            )

        inputs_config['data_random_seed'] = getattr(arguments, 'data_random_seed', 0)

        if getattr(arguments, 'data_params', None) is not None:
            inputs_config['data_params'] = {name: value for name, value in arguments.data_params}

        if getattr(arguments, 'data_split_file', None) is not None:
            if arguments.data_split_file == '-':
                data_split_file = sys.stdin.buffer
            else:
                data_split_file = arguments.data_split_file
            split_file = pandas.read_csv(
                data_split_file,
                # We do not want to do any conversion of values at this point.
                # This should be done by primitives later on.
                dtype=str,
                # We always expect one row header.
                header=0,
                # We want empty strings and not NaNs.
                na_filter=False,
                encoding='utf8',
                low_memory=False,
                memory_map=True,
            )

            # We use just the "d3mIndex" column and ignore multi-key indices.
            # This works for now because it seems that every current multi-key
            # dataset in fact has an unique value in "d3mIndex" alone.
            # See: https://gitlab.com/datadrivendiscovery/data-supply/issues/117
            # Hyper-parameter value has to be JSON-serialized.
            inputs_config['data_params']['primary_index_values'] = json.dumps(list(split_file.loc[split_file['type'] == 'TEST']['d3mIndex']))

    return InputsConfig(**inputs_config)


# TODO: Replace "dataset_view_maps" with an instance of "ScoringConfiguration".
#       See: https://gitlab.com/datadrivendiscovery/d3m/-/issues/515
def _save_problem_description(problem_description: problem.Problem, problem_path: str, *, dataset_view_maps: typing.Mapping[str, typing.Sequence[typing.Mapping[str, str]]] = None) -> None:
    """
    Saves the problem description, but also adds a dataset view map to saved problem description, which is otherwise not part of problem description,
    but it is part of D3M format problem description.
    """

    problem_uri = utils.path_to_uri(problem_path)
    problem_description.save(problem_uri)

    if dataset_view_maps:
        with open(problem_path, 'r', encoding='utf8') as file:
            problem_description_json = json.load(file)
        problem_description_json['inputs']['dataSplits'] = {
            'datasetViewMaps': dataset_view_maps,
        }
        with open(problem_path, 'w', encoding='utf8') as file:
            json.dump(problem_description_json, file, indent=2, allow_nan=False)


# TODO: Replace "dataset_view_maps" with an instance of "ScoringConfiguration".
#       See: https://gitlab.com/datadrivendiscovery/d3m/-/issues/515
def _get_dataset_id_from_view_maps(dataset_view_maps: typing.Mapping[str, typing.Sequence[typing.Mapping[str, str]]], split_type: str, dataset_id: str) -> str:
    for view_map_entry in dataset_view_maps[split_type.lower()]:
        if view_map_entry['from'] == dataset_id:
            return view_map_entry['to']

    # This should not happen if view maps were generated using "_generate_dataset_view_maps".
    raise KeyError(f"Could not find a view map entry for \"{dataset_id}\".")


# TODO: Replace "dataset_view_maps" with an instance of "ScoringConfiguration".
#       See: https://gitlab.com/datadrivendiscovery/d3m/-/issues/515
def _save_dataset_fold(
    path: str, split_type: str, inputs: typing.Sequence[container.Dataset], split_dataset: container.Dataset, problem_description: typing.Optional[problem.Problem],
    dataset_view_maps: typing.Mapping[str, typing.Sequence[typing.Mapping[str, str]]], fold_index: int = None,
    force_split_dataset_id: bool = False,
) -> None:
    if fold_index is not None:
        save_path = os.path.abspath(os.path.join(path, 'folds', str(fold_index), split_type))
    else:
        save_path = os.path.abspath(os.path.join(path, split_type))
    os.makedirs(save_path, 0o755, exist_ok=False)

    split_dataset_id = _get_split_dataset_id(inputs, split_dataset, split_type, fold_index)
    # If split dataset does not already have the mapped dataset ID, we update it now.
    if force_split_dataset_id or split_dataset.metadata.query_field((), 'id') != split_dataset_id:
        # We make a copy for the case that no-split is use.
        split_dataset = split_dataset.copy()
        split_dataset.metadata = split_dataset.metadata.update((), {'id': split_dataset_id})

    dataset_path = os.path.join(save_path, 'dataset_{split_type}'.format(split_type=split_type), 'datasetDoc.json')
    dataset_uri = utils.path_to_uri(dataset_path)
    # We od not save metadata because data preparation pipelines might not have updated all metadata (like metafeatures).
    split_dataset.save(dataset_uri, preserve_metadata=False)

    if problem_description is not None:
        problem_path = os.path.join(save_path, 'problem_{split_type}'.format(split_type=split_type), 'problemDoc.json')
        _save_problem_description(problem_description, problem_path, dataset_view_maps=dataset_view_maps)


def _get_split_dataset_id(inputs: typing.Sequence[container.Dataset], split_dataset: container.Dataset, split_type: str, fold_index: int = None) -> str:
    split_dataset_id = split_dataset.metadata.query_field((), 'id')
    input_ids = {input_dataset.metadata.query_field((), 'id') for input_dataset in inputs}
    # If data preparation pipeline has not updated the output dataset ID, we map it here.
    if split_dataset_id in input_ids:
        if fold_index is not None:
            split_dataset_id = f'{split_dataset_id}_FOLD_{fold_index}_{split_type.upper()}'
        else:
            split_dataset_id = f'{split_dataset_id}_{split_type.upper()}'
        # Sanity check, the new dataset ID should not be one used on the input.
        assert split_dataset_id not in input_ids, (split_dataset_id, input_ids)
    return split_dataset_id


# TODO: Replace "dataset_view_maps" with an instance of "ScoringConfiguration".
#       See: https://gitlab.com/datadrivendiscovery/d3m/-/issues/515
def _generate_dataset_view_maps(
    inputs: typing.Sequence[container.Dataset], train_dataset: dataset_module.Dataset, test_dataset: dataset_module.Dataset,
    score_dataset: dataset_module.Dataset, fold_index: int = None,
) -> typing.Mapping[str, typing.Sequence[typing.Mapping[str, str]]]:
    """
    Dataset view map map dataset ID from full dataset ID to a corresponding split dataset ID. This is used because
    same problem description (which references inputs using full dataset ID) is always used, but split datasets have a different
    dataset ID (because they are different datasets).

    This function generates a dataset view map for data preparation pipelines as supported by
    "prepare_data" function, namely data preparation pipelines which always return only one dataset,
    even if they have multiple input datasets (e.g., they combine multiple input datasets).
    """

    view_maps: typing.Dict[str, typing.List[typing.Dict[str, str]]] = {}
    for split_type, split_dataset in zip(['train', 'test', 'score'], [train_dataset, test_dataset, score_dataset]):
        view_maps[split_type] = []
        for input_dataset in inputs:
            view_maps[split_type].append({
                'from': input_dataset.metadata.query_field((), 'id'),
                'to': _get_split_dataset_id(inputs, split_dataset, split_type, fold_index),
            })

    return view_maps


# TODO: Replace "dataset_view_maps" with an instance of "ScoringConfiguration".
#       See: https://gitlab.com/datadrivendiscovery/d3m/-/issues/515
def prepare_data_and_save(
    save_dir: str, inputs: typing.Sequence[container.Dataset], *, data_pipeline: pipeline_module.Pipeline,
    problem_description: typing.Optional[problem.Problem], data_params: typing.Dict[str, str] = None,
    context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
    dataset_view_maps: typing.Sequence[typing.Mapping[str, typing.Sequence[typing.Mapping[str, str]]]] = None,
) -> None:
    os.makedirs(save_dir, 0o755, exist_ok=False)

    outputs, data_result = prepare_data(
        inputs, data_pipeline=data_pipeline, problem_description=problem_description, data_params=data_params,
        context=context, random_seed=random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, runtime_environment=runtime_environment,
    )
    data_result.check_success()

    # Store pickled prepared data pipeline run.
    pipeline_run_path = os.path.join(save_dir, DATA_PIPELINE_RUN_FILENAME)
    with open(pipeline_run_path, 'wb') as file:
        pickle.dump(data_result.pipeline_run, file)

    # Checked already by "prepare_data".
    assert len(outputs[0]) > 0

    if dataset_view_maps is not None and len(dataset_view_maps) != len(outputs[0]):
        raise exceptions.InvalidArgumentValueError(
            f"\"dataset_view_maps\" parameter ({len(dataset_view_maps)}) does not match the number of folds made by the data preparation pipeline ({len(outputs[0])}).",
        )

    # Store folds.
    if len(outputs[0]) == 1:
        train_inputs, test_inputs, score_inputs = list(zip(*outputs))[0]
        if dataset_view_maps is None:
            view_maps = _generate_dataset_view_maps(inputs, train_inputs, test_inputs, score_inputs)
        else:
            view_maps = dataset_view_maps[0]
        _save_dataset_fold(
            path=save_dir, split_type='TRAIN', inputs=inputs, split_dataset=train_inputs, problem_description=problem_description,
            dataset_view_maps=view_maps, force_split_dataset_id=dataset_view_maps is not None,
        )
        _save_dataset_fold(
            path=save_dir, split_type='TEST', inputs=inputs, split_dataset=test_inputs, problem_description=problem_description,
            dataset_view_maps=view_maps, force_split_dataset_id=dataset_view_maps is not None,
        )
        _save_dataset_fold(
            path=save_dir, split_type='SCORE', inputs=inputs, split_dataset=score_inputs, problem_description=problem_description,
            dataset_view_maps=view_maps, force_split_dataset_id=dataset_view_maps is not None,
        )
    else:
        for fold_index, (train_inputs, test_inputs, score_inputs) in enumerate(zip(*outputs)):
            if dataset_view_maps is None:
                view_maps = _generate_dataset_view_maps(inputs, train_inputs, test_inputs, score_inputs, fold_index)
            else:
                view_maps = dataset_view_maps[fold_index]
            _save_dataset_fold(
                path=save_dir, split_type='TRAIN', inputs=inputs, split_dataset=train_inputs, problem_description=problem_description, fold_index=fold_index,
                dataset_view_maps=view_maps, force_split_dataset_id=dataset_view_maps is not None,
            )
            _save_dataset_fold(
                path=save_dir, split_type='TEST', inputs=inputs, split_dataset=test_inputs, problem_description=problem_description, fold_index=fold_index,
                dataset_view_maps=view_maps, force_split_dataset_id=dataset_view_maps is not None,
            )
            _save_dataset_fold(
                path=save_dir, split_type='SCORE', inputs=inputs, split_dataset=score_inputs, problem_description=problem_description, fold_index=fold_index,
                dataset_view_maps=view_maps, force_split_dataset_id=dataset_view_maps is not None,
            )


def _load_dataset_fold(
    path: str, split_type: str, *,
    dataset_resolver: typing.Callable = None, compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False,
) -> container.Dataset:
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset

    dataset_uri = utils.path_to_uri(os.path.join(path, split_type, 'dataset_{split_type}'.format(split_type=split_type), 'datasetDoc.json'))
    return dataset_resolver(
        dataset_uri, compute_digest=compute_digest, strict_digest=strict_digest,
    )


def evaluate_with_prepared_data(
    pipeline: pipeline_module.Pipeline, inputs_dir: str, *,
    scoring_pipeline: pipeline_module.Pipeline, problem_description: typing.Optional[problem.Problem],
    metrics: typing.Sequence[typing.Dict], context: metadata_base.Context,
    scoring_params: typing.Dict[str, str] = None, hyperparams: typing.Sequence = None, random_seed: int = 0,
    scoring_random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
    dataset_resolver: typing.Callable = None, compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False,
) -> typing.Tuple[typing.List[container.DataFrame], MultiResult]:
    with open(os.path.join(inputs_dir, DATA_PIPELINE_RUN_FILENAME), 'rb') as file:
        data_pipeline_run = pickle.load(file)

    folds_dir = os.path.join(inputs_dir, 'folds')
    if os.path.exists(folds_dir):
        fold_dirs = []
        for file_name in os.listdir(folds_dir):
            fold_dir = os.path.join(folds_dir, file_name)
            if not os.path.isdir(fold_dir):
                continue
            try:
                fold_index = int(file_name)
            except ValueError:
                raise ValueError("\"{file_name}\" is not a fold index in \"{folds_dir}\".".format(file_name=file_name, folds_dir=folds_dir))
            fold_dirs.append((fold_index, fold_dir))

    else:
        fold_dirs = [(0, inputs_dir)]

    fold_group_uuid = uuid.uuid4()

    all_scores: typing.List[container.DataFrame] = []
    all_results = MultiResult()

    for fold_index, fold_dir in fold_dirs:
        try:
            train_inputs = _load_dataset_fold(fold_dir, 'TRAIN', dataset_resolver=dataset_resolver, compute_digest=compute_digest, strict_digest=strict_digest)
            test_inputs = _load_dataset_fold(fold_dir, 'TEST', dataset_resolver=dataset_resolver, compute_digest=compute_digest, strict_digest=strict_digest)
            score_inputs = _load_dataset_fold(fold_dir, 'SCORE', dataset_resolver=dataset_resolver, compute_digest=compute_digest, strict_digest=strict_digest)
        except Exception as error:
            raise exceptions.InvalidDatasetError("Cannot load dataset splits from \"{fold_dir}\".".format(fold_dir=fold_dir)) from error

        evaluate_fold(
            pipeline, [train_inputs], [test_inputs], [score_inputs], all_scores, all_results,
            fold_index=fold_index, fold_group_uuid=fold_group_uuid, data_pipeline_run=data_pipeline_run,
            scoring_pipeline=scoring_pipeline, problem_description=problem_description,
            metrics=metrics, context=context,
            scoring_params=scoring_params, hyperparams=hyperparams, random_seed=random_seed,
            scoring_random_seed=scoring_random_seed, volumes_dir=volumes_dir,
            scratch_dir=scratch_dir, runtime_environment=runtime_environment,
        )

        if all_results.has_error():
            break

    return all_scores, all_results


def prepare_data_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, pipeline_run_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]
    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )
    if getattr(arguments, 'problem', None) is not None:
        problem_description = problem_resolver(arguments.problem, strict_digest=getattr(arguments, 'strict_digest', False))
    else:
        problem_description = None

    assert getattr(arguments, 'input_run', None) is None

    inputs_config = _get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
    )

    if inputs_config.data_pipeline is None:
        raise exceptions.InvalidArgumentValueError("Data preparation pipeline is missing in arguments.")
    if inputs_config.inputs is None:
        raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")

    inputs = inputs_config.inputs
    data_pipeline = inputs_config.data_pipeline
    data_params = inputs_config.data_params
    data_random_seed = inputs_config.data_random_seed

    prepare_data_and_save(
        save_dir=arguments.save_dir,
        inputs=inputs,
        data_pipeline=data_pipeline,
        problem_description=problem_description,
        data_params=data_params,
        context=context,
        random_seed=data_random_seed,
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment
    )


def fit_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    pipeline_run_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if pipeline_run_parser is None:
        pipeline_run_parser = parse_pipeline_run
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    if getattr(arguments, 'data_pipeline', None) and getattr(arguments, 'data_pipeline_run', None):
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    context = metadata_base.Context[arguments.context]
    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]
    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    if getattr(arguments, 'input_run', None) is not None:
        parsed_pipeline_runs = pipeline_run_parser(
            arguments.input_run, getattr(arguments, 'pipeline_search_paths', []), getattr(arguments, 'datasets_dir', None),
            pipeline_resolver=pipeline_resolver, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
        )

        if len(parsed_pipeline_runs) != 1:
            raise exceptions.InvalidArgumentValueError(
                "Fit requires exactly one pipeline run. {pipeline_runs} provided.".format(pipeline_runs=len(parsed_pipeline_runs))
            )
        if parsed_pipeline_runs[0]['run']['phase'] != metadata_base.PipelineRunPhase.FIT.name:
            raise exceptions.InvalidArgumentValueError(
                "Fit requires a FIT phase pipeline run. {phase} phase provided.".format(phase=parsed_pipeline_runs[0]['run']['phase'])
            )
        fit_pipeline_run = parsed_pipeline_runs[0]

        pipeline = fit_pipeline_run['pipeline']
        problem_description = fit_pipeline_run.get('problem', None)
        # Currently, "random_seed" is not yet required.
        random_seed = fit_pipeline_run.get('random_seed', 0)
        hyperparams: typing.Optional[typing.Sequence[typing.Union[typing.Dict, typing.Sequence]]] = \
            _get_runtime_hyperparams_from_pipeline_run(fit_pipeline_run['pipeline'], fit_pipeline_run.get('steps', []))
        # Currently, "is_standard_pipeline" is not yet required.
        is_standard_pipeline = fit_pipeline_run['run'].get('is_standard_pipeline', True)

    else:
        fit_pipeline_run = None

        pipeline = pipeline_resolver(
            arguments.pipeline,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        )

        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem, strict_digest=getattr(arguments, 'strict_digest', False))
        else:
            problem_description = None

        random_seed = getattr(arguments, 'random_seed', 0)
        # We use default hyper-parameter values for now.
        hyperparams = None
        is_standard_pipeline = getattr(arguments, 'standard_pipeline', True)

    inputs_config = _get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
        fit_pipeline_run=fit_pipeline_run,
    )

    if inputs_config.inputs is None:
        if inputs_config.data_pipeline is not None:
            if fit_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input full data is missing in the pipeline run.")
        else:
            if fit_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input train data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input train data is missing in the pipeline run.")

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    data_pipeline_run = None
    if inputs_config.data_pipeline is None and getattr(arguments, 'data_pipeline_run', None) is not None:
        with open(arguments.data_pipeline_run, 'rb') as file:
            data_pipeline_run = pickle.load(file)

        if 'data_preparation' not in data_pipeline_run.run:
            raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not a data preparation pipeline run.")
        if data_pipeline_run.run['data_preparation']['status']['state'] != metadata_base.PipelineRunStatusState.SUCCESS.name:
            raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not successful.")

    fitted_pipeline, predictions, result = fit(
        pipeline, inputs_config.inputs,
        problem_description=problem_description,
        context=context,
        hyperparams=hyperparams,
        random_seed=random_seed,
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
        is_standard_pipeline=is_standard_pipeline,
        expose_produced_outputs=expose_produced_outputs,
        data_pipeline=inputs_config.data_pipeline,
        data_params=inputs_config.data_params,
        data_random_seed=inputs_config.data_random_seed,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=getattr(arguments, 'fold_group_uuid', None),
        fold_index=getattr(arguments, 'fold_index', 0),
    )

    if expose_produced_outputs:
        save_exposed_outputs(result, arguments.expose_produced_outputs_dir)

    _output_pipeline_runs(arguments, [result.pipeline_run])

    result.check_success()

    assert fitted_pipeline is not None

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)
        # Make sure the handle is flushed so that no data is lost. CLI file handles are generally
        # used outside of a context manager which would otherwise handle that.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/436
        arguments.save.flush()

    if getattr(arguments, 'output', None) is not None:
        assert is_standard_pipeline
        assert predictions is not None
        predictions.to_csv(arguments.output)


# We have "problem_resolver" as argument (even if we are not using it in this function)
# so that the signature is the same for all handlers.
def produce_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    pipeline_run_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if pipeline_run_parser is None:
        pipeline_run_parser = parse_pipeline_run
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset

    if getattr(arguments, 'data_pipeline', None) and getattr(arguments, 'data_pipeline_run', None):
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]
    fitted_pipeline = pickle.load(arguments.fitted_pipeline)

    if not fitted_pipeline.is_standard_pipeline and getattr(arguments, 'output', None) is not None:
        raise exceptions.InvalidArgumentValueError("You cannot save predictions for a non-standard pipeline.")

    if getattr(arguments, 'input_run', None) is not None:
        parsed_pipeline_runs = pipeline_run_parser(
            arguments.input_run, getattr(arguments, 'pipeline_search_paths', []), getattr(arguments, 'datasets_dir', None),
            pipeline_resolver=pipeline_resolver, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
        )

        if len(parsed_pipeline_runs) != 1:
            raise exceptions.InvalidArgumentValueError(
                "Produce requires exactly one pipeline run. {pipeline_runs} provided.".format(pipeline_runs=len(parsed_pipeline_runs))
            )
        if parsed_pipeline_runs[0]['run']['phase'] != metadata_base.PipelineRunPhase.PRODUCE.name:
            raise exceptions.InvalidArgumentValueError(
                "Produce requires a PRODUCE phase pipeline run. {phase} phase provided.".format(phase=parsed_pipeline_runs[0]['run']['phase'])
            )
        produce_pipeline_run = parsed_pipeline_runs[0]

        # TODO: Check that pipeline (and hyperparams, is_standard_pipeline flag) and problem match those in the fitted_pipeline.

    else:
        produce_pipeline_run = None

    inputs_config = _get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
        produce_pipeline_run=produce_pipeline_run,
    )

    if inputs_config.test_inputs is None:
        if inputs_config.data_pipeline is not None:
            if produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input full data is missing in the pipeline run.")
        else:
            if produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input test data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input test data is missing in the pipeline run.")

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    data_pipeline_run = None
    if inputs_config.data_pipeline is None and getattr(arguments, 'data_pipeline_run', None) is not None:
        with open(arguments.data_pipeline_run, 'rb') as file:
            data_pipeline_run = pickle.load(file)

        if 'data_preparation' not in data_pipeline_run.run:
            raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not a data preparation pipeline run.")
        if data_pipeline_run.run['data_preparation']['status']['state'] != metadata_base.PipelineRunStatusState.SUCCESS.name:
            raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not successful.")

    predictions, result = produce(
        fitted_pipeline, inputs_config.test_inputs,
        expose_produced_outputs=expose_produced_outputs,
        data_pipeline=inputs_config.data_pipeline,
        data_params=inputs_config.data_params,
        data_random_seed=inputs_config.data_random_seed,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=getattr(arguments, 'fold_group_uuid', None),
        fold_index=getattr(arguments, 'fold_index', 0),
    )

    if expose_produced_outputs:
        save_exposed_outputs(result, arguments.expose_produced_outputs_dir)

    _output_pipeline_runs(arguments, [result.pipeline_run])

    result.check_success()

    if getattr(arguments, 'output', None) is not None:
        assert fitted_pipeline.is_standard_pipeline
        assert predictions is not None
        predictions.to_csv(arguments.output)


# We have "problem_resolver" as argument (even if we are not using it in this function)
# so that the signature is the same for all handlers.
def score_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    pipeline_run_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if pipeline_run_parser is None:
        pipeline_run_parser = parse_pipeline_run
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset

    if getattr(arguments, 'data_pipeline', None) and getattr(arguments, 'data_pipeline_run', None):
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]
    fitted_pipeline = pickle.load(arguments.fitted_pipeline)

    if not fitted_pipeline.is_standard_pipeline:
        raise exceptions.InvalidArgumentValueError("You cannot score a non-standard pipeline.")

    if getattr(arguments, 'input_run', None) is not None:
        parsed_pipeline_runs = pipeline_run_parser(
            arguments.input_run, getattr(arguments, 'pipeline_search_paths', []), getattr(arguments, 'datasets_dir', None),
            pipeline_resolver=pipeline_resolver, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
        )

        if len(parsed_pipeline_runs) != 1:
            raise exceptions.InvalidArgumentValueError(
                "Score requires exactly one pipeline run. {pipeline_runs} provided.".format(pipeline_runs=len(parsed_pipeline_runs))
            )
        if parsed_pipeline_runs[0]['run']['phase'] != metadata_base.PipelineRunPhase.PRODUCE.name:
            raise exceptions.InvalidArgumentValueError(
                "Score requires a PRODUCE phase pipeline run. {phase} phase provided.".format(phase=parsed_pipeline_runs[0]['run']['phase'])
            )
        produce_pipeline_run = parsed_pipeline_runs[0]

        if 'scoring' not in produce_pipeline_run['run']:
            raise exceptions.InvalidArgumentValueError("Score requires a pipeline run with scoring.")

        # TODO: Check that pipeline (and hyperparams, is_standard_pipeline flag) and problem match those in the fitted_pipeline.

        scoring_pipeline = produce_pipeline_run['run']['scoring']['pipeline']
        # Currently, "random_seed" is not yet required.
        random_seed = produce_pipeline_run['run']['scoring'].get('random_seed', 0)
        # We do not have to set metrics, because they should already be included in hyper-paramters.
        metrics: typing.Sequence[typing.Dict] = []
        scoring_params = _get_data_and_scoring_params_from_pipeline_run(produce_pipeline_run['run']['scoring'].get('steps', []))

    else:
        produce_pipeline_run = None

        scoring_pipeline = pipeline_resolver(
            arguments.scoring_pipeline,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        )

        random_seed = getattr(arguments, 'random_seed', 0)

        if getattr(arguments, 'metrics', None) is not None:
            metrics = get_metrics_from_list(arguments.metrics)
        else:
            metrics = get_metrics_from_problem_description(fitted_pipeline.problem_description)

        if getattr(arguments, 'scoring_params', None) is not None:
            scoring_params = {name: value for name, value in arguments.scoring_params}
        else:
            scoring_params = {}

    inputs_config = _get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
        produce_pipeline_run=produce_pipeline_run,
    )

    data_pipeline_run = None
    data_result: typing.Optional[Result]
    test_inputs: typing.Sequence[container.Dataset]
    score_inputs: typing.Sequence[container.Dataset]
    # We run a data preparation pipeline here instead of passing it into
    # "produce" and "score" functions so that we run it only once.
    if inputs_config.data_pipeline is not None:
        if inputs_config.test_inputs is None:
            if produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input full data is missing in the pipeline run.")

        outputs, data_result = prepare_data(
            inputs_config.test_inputs,
            data_pipeline=inputs_config.data_pipeline,
            problem_description=fitted_pipeline.problem_description,
            data_params=inputs_config.data_params,
            context=fitted_pipeline.context,
            random_seed=inputs_config.data_random_seed,
            volumes_dir=fitted_pipeline.volumes_dir,
            scratch_dir=fitted_pipeline.scratch_dir,
            runtime_environment=fitted_pipeline.environment,
        )
        data_result.check_success()
        if len(outputs[1]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of test data but {folds} folds.".format(
                folds=len(outputs[1]),
            ))
        if len(outputs[2]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of score data but {folds} folds.".format(
                folds=len(outputs[2]),
            ))
        test_inputs = [outputs[1][0]]
        score_inputs = [outputs[2][0]]
    else:
        if inputs_config.test_inputs is None:
            if produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input test data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input test data is missing in the pipeline run.")
        if inputs_config.score_inputs is None:
            if produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input score data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input score data is missing in the pipeline run.")

        if getattr(arguments, 'data_pipeline_run', None) is not None:
            with open(arguments.data_pipeline_run, 'rb') as file:
                data_pipeline_run = pickle.load(file)

            if 'data_preparation' not in data_pipeline_run.run:
                raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not a data preparation pipeline run.")
            if data_pipeline_run.run['data_preparation']['status']['state'] != metadata_base.PipelineRunStatusState.SUCCESS.name:
                raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not successful.")

        data_result = None
        test_inputs = inputs_config.test_inputs
        score_inputs = inputs_config.score_inputs

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    fold_group_uuid = getattr(arguments, 'fold_group_uuid', None)
    if fold_group_uuid is None:
        fold_group_uuid = uuid.uuid4()
    if data_result is not None:
        assert data_pipeline_run is None
        data_pipeline_run = data_result.pipeline_run

    predictions, produce_result = produce(
        fitted_pipeline, test_inputs,
        expose_produced_outputs=expose_produced_outputs,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=fold_group_uuid, fold_index=getattr(arguments, 'fold_index', 0),
    )

    if expose_produced_outputs:
        save_exposed_outputs(produce_result, arguments.expose_produced_outputs_dir)

    if produce_result.has_error():
        _output_pipeline_runs(arguments, [produce_result.pipeline_run])

        produce_result.check_success()

        assert False

    assert predictions is not None

    if getattr(arguments, 'output', None) is not None:
        predictions.to_csv(arguments.output)

    scores, score_result = score(
        predictions,
        score_inputs,
        scoring_pipeline=scoring_pipeline,
        problem_description=fitted_pipeline.problem_description,
        metrics=metrics,
        predictions_random_seed=fitted_pipeline.random_seed,
        scoring_params=scoring_params,
        context=fitted_pipeline.context,
        random_seed=random_seed,
        volumes_dir=fitted_pipeline.volumes_dir,
        scratch_dir=fitted_pipeline.scratch_dir,
        runtime_environment=fitted_pipeline.environment,
    )

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=score_inputs if data_result is None and data_pipeline_run is None else None,
    )

    if score_result.has_error():
        _output_pipeline_runs(arguments, [produce_result.pipeline_run])

        score_result.check_success()

        assert False

    # Modifies "produce_pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, metrics=metrics, scores=scores,
    )

    _output_pipeline_runs(arguments, [produce_result.pipeline_run])

    assert scores is not None

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


def fit_produce_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    pipeline_run_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if pipeline_run_parser is None:
        pipeline_run_parser = parse_pipeline_run
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    if getattr(arguments, 'data_pipeline', None) and getattr(arguments, 'data_pipeline_run', None):
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    context = metadata_base.Context[arguments.context]
    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]
    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    if getattr(arguments, 'input_run', None) is not None:
        parsed_pipeline_runs = pipeline_run_parser(
            arguments.input_run, getattr(arguments, 'pipeline_search_paths', []), getattr(arguments, 'datasets_dir', None),
            pipeline_resolver=pipeline_resolver, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
        )

        if len(parsed_pipeline_runs) != 2:
            raise exceptions.InvalidArgumentValueError(
                "Fit-produce requires exactly two pipeline runs. {pipeline_runs} provided.".format(pipeline_runs=len(parsed_pipeline_runs))
            )
        # TODO: We might not want to require that the order in the file is strict.
        #       We could just require that pipeline runs belong together (using previous_pipeline_run)
        #       and are of FIT and PRODUCE phase and then run them in the correct order.
        pipeline_run_0_phase = parsed_pipeline_runs[0]['run']['phase']
        if pipeline_run_0_phase != metadata_base.PipelineRunPhase.FIT.name:
            raise exceptions.InvalidArgumentValueError(
                "Fit-produce requires the first pipeline run to be a FIT phase. {phase} phase provided.".format(phase=pipeline_run_0_phase)
            )
        pipeline_run_1_phase = parsed_pipeline_runs[1]['run']['phase']
        if pipeline_run_1_phase != metadata_base.PipelineRunPhase.PRODUCE.name:
            raise exceptions.InvalidArgumentValueError(
                "Fit-produce requires the second pipeline run to be a PRODUCE phase. {phase} phase provided.".format(phase=pipeline_run_1_phase)
            )
        fit_pipeline_run = parsed_pipeline_runs[0]
        produce_pipeline_run = parsed_pipeline_runs[1]

        if produce_pipeline_run['previous_pipeline_run']['id'] != fit_pipeline_run['id']:
            raise exceptions.InvalidArgumentValueError("Fit-produce requires that the PRODUCE phase pipeline run must reference FIT phase pipeline run in \"previous_pipeline_run\".")
        if fit_pipeline_run['pipeline'].id != produce_pipeline_run['pipeline'].id or fit_pipeline_run['pipeline'].get_digest() != produce_pipeline_run['pipeline'].get_digest():
            raise exceptions.InvalidArgumentValueError("Fit-produce requires that both the FIT phase and PRODUCE phase pipeline runs reference the same pipeline.")
        if ('problem' in fit_pipeline_run) != ('problem' in produce_pipeline_run):
            raise exceptions.InvalidArgumentValueError('fit-produce requires that both FIT phase PRODUCE phase pipeline runs reference the same problem, if any')
        if 'problem' in fit_pipeline_run and fit_pipeline_run['problem']['id'] != produce_pipeline_run['problem']['id']:
            raise exceptions.InvalidArgumentValueError('FIT phase and PRODUCE phase pipeline runs reference different problem ids')
        if 'problem' in fit_pipeline_run and fit_pipeline_run['problem'].get_digest() != produce_pipeline_run['problem'].get_digest():
            raise exceptions.InvalidArgumentValueError('FIT phase and PRODUCE phase pipeline runs reference different problem digests')

        # TODO: Check that hyperparams match between both pipeline runs (but allow failed runs).
        # TODO: Check that inputs match between both pipeline runs.

        pipeline = fit_pipeline_run['pipeline']
        problem_description = fit_pipeline_run.get('problem', None)
        # Currently, "random_seed" is not yet required.
        random_seed = fit_pipeline_run.get('random_seed', 0)
        hyperparams: typing.Optional[typing.Sequence[typing.Union[typing.Dict, typing.Sequence]]] = \
            _get_runtime_hyperparams_from_pipeline_run(fit_pipeline_run['pipeline'], fit_pipeline_run.get('steps', []))
        # Currently, "is_standard_pipeline" is not yet required.
        is_standard_pipeline = fit_pipeline_run['run'].get('is_standard_pipeline', True)

    else:
        fit_pipeline_run = None
        produce_pipeline_run = None

        pipeline = pipeline_resolver(
            arguments.pipeline,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        )

        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem, strict_digest=getattr(arguments, 'strict_digest', False))
        else:
            problem_description = None

        random_seed = getattr(arguments, 'random_seed', 0)
        # We use default hyper-parameter values for now.
        hyperparams = None
        is_standard_pipeline = getattr(arguments, 'standard_pipeline', True)

    inputs_config = _get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
        fit_pipeline_run=fit_pipeline_run,
        produce_pipeline_run=produce_pipeline_run,
    )

    data_pipeline_run = None
    data_result: typing.Optional[Result]
    inputs: typing.Sequence[container.Dataset]
    test_inputs: typing.Sequence[container.Dataset]
    # We run a data preparation pipeline here instead of passing it into
    # "fit" and "produce" functions so that we run it only once.
    if inputs_config.data_pipeline is not None:
        if inputs_config.inputs is None:
            if fit_pipeline_run is None and produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input full data is missing in the pipeline run.")

        outputs, data_result = prepare_data(
            inputs_config.inputs,
            data_pipeline=inputs_config.data_pipeline,
            problem_description=problem_description,
            data_params=inputs_config.data_params,
            context=context,
            random_seed=inputs_config.data_random_seed,
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            scratch_dir=getattr(arguments, 'scratch_dir', None),
            runtime_environment=runtime_environment,
        )
        data_result.check_success()
        if len(outputs[0]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of train data but {folds} folds.".format(
                folds=len(outputs[0]),
            ))
        if len(outputs[1]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of test data but {folds} folds.".format(
                folds=len(outputs[1]),
            ))
        inputs = [outputs[0][0]]
        test_inputs = [outputs[1][0]]
    else:
        if inputs_config.inputs is None:
            if fit_pipeline_run is None and produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input train data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input train data is missing in the pipeline run.")
        if inputs_config.test_inputs is None:
            if fit_pipeline_run is None and produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input test data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input test data is missing in the pipeline run.")

        if getattr(arguments, 'data_pipeline_run', None) is not None:
            with open(arguments.data_pipeline_run, 'rb') as file:
                data_pipeline_run = pickle.load(file)

            if 'data_preparation' not in data_pipeline_run.run:
                raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not a data preparation pipeline run.")
            if data_pipeline_run.run['data_preparation']['status']['state'] != metadata_base.PipelineRunStatusState.SUCCESS.name:
                raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run is not successful.")

        data_result = None
        inputs = inputs_config.inputs
        test_inputs = inputs_config.test_inputs

    fold_group_uuid = getattr(arguments, 'fold_group_uuid', None)
    if fold_group_uuid is None:
        fold_group_uuid = uuid.uuid4()
    if data_result is not None:
        assert data_pipeline_run is None
        data_pipeline_run = data_result.pipeline_run

    fitted_pipeline, predictions, fit_result = fit(
        pipeline, inputs,
        problem_description=problem_description,
        context=context,
        hyperparams=hyperparams,
        random_seed=random_seed,
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
        is_standard_pipeline=is_standard_pipeline,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=fold_group_uuid, fold_index=getattr(arguments, 'fold_index', 0),
    )

    if fit_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run])

        fit_result.check_success()

        assert False

    assert fitted_pipeline is not None

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)
        # Make sure the handle is flushed so that no data is lost. CLI file handles are generally
        # used outside of a context manager which would otherwise handle that.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/436
        arguments.save.flush()

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    predictions, produce_result = produce(
        fitted_pipeline, test_inputs,
        expose_produced_outputs=expose_produced_outputs,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=fold_group_uuid, fold_index=getattr(arguments, 'fold_index', 0),
    )

    if expose_produced_outputs:
        save_exposed_outputs(produce_result, arguments.expose_produced_outputs_dir)

    _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

    produce_result.check_success()

    if getattr(arguments, 'output', None) is not None:
        assert is_standard_pipeline
        assert predictions is not None
        predictions.to_csv(arguments.output)


def fit_score_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    pipeline_run_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if pipeline_run_parser is None:
        pipeline_run_parser = parse_pipeline_run
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    if getattr(arguments, 'data_pipeline', None) and getattr(arguments, 'data_pipeline_run', None):
        raise exceptions.InvalidArgumentValueError("\"data_pipeline\" and \"data_pipeline_run\" cannot be both provided.")

    context = metadata_base.Context[arguments.context]
    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]
    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    if getattr(arguments, 'input_run', None) is not None:
        parsed_pipeline_runs = pipeline_run_parser(
            arguments.input_run, getattr(arguments, 'pipeline_search_paths', []), getattr(arguments, 'datasets_dir', None),
            pipeline_resolver=pipeline_resolver, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
        )

        if len(parsed_pipeline_runs) != 2:
            raise exceptions.InvalidArgumentValueError(
                "Fit-score requires exactly two pipeline runs. {pipeline_runs} provided.".format(pipeline_runs=len(parsed_pipeline_runs))
            )
        # TODO: We might not want to require that the order in the file is strict.
        #       We could just require that pipeline runs belong together (using previous_pipeline_run)
        #       and are of FIT and PRODUCE phase and then run them in the correct order.
        pipeline_run_0_phase = parsed_pipeline_runs[0]['run']['phase']
        if pipeline_run_0_phase != metadata_base.PipelineRunPhase.FIT.name:
            raise exceptions.InvalidArgumentValueError(
                "Fit-score requires the first pipeline run to be a FIT phase. {phase} phase provided.".format(phase=pipeline_run_0_phase)
            )
        pipeline_run_1_phase = parsed_pipeline_runs[1]['run']['phase']
        if pipeline_run_1_phase != metadata_base.PipelineRunPhase.PRODUCE.name:
            raise exceptions.InvalidArgumentValueError(
                "Fit-score requires the second pipeline run to be a PRODUCE phase. {phase} phase provided.".format(phase=pipeline_run_1_phase)
            )
        fit_pipeline_run = parsed_pipeline_runs[0]
        produce_pipeline_run = parsed_pipeline_runs[1]

        if produce_pipeline_run['previous_pipeline_run']['id'] != fit_pipeline_run['id']:
            raise exceptions.InvalidArgumentValueError("Fit-score requires that the PRODUCE phase pipeline run must reference FIT phase pipeline run in \"previous_pipeline_run\".")
        if fit_pipeline_run['pipeline'].id != produce_pipeline_run['pipeline'].id or fit_pipeline_run['pipeline'].get_digest() != produce_pipeline_run['pipeline'].get_digest():
            raise exceptions.InvalidArgumentValueError("Fit-score requires that both the FIT phase and PRODUCE phase pipeline runs reference the same pipeline.")
        if ('problem' in fit_pipeline_run) != ('problem' in produce_pipeline_run):
            raise exceptions.InvalidArgumentValueError('fit-score requires that both FIT phase PRODUCE phase pipeline runs reference the same problem, if any')
        if 'problem' in fit_pipeline_run and fit_pipeline_run['problem']['id'] != produce_pipeline_run['problem']['id']:
            raise exceptions.InvalidArgumentValueError('FIT phase and PRODUCE phase pipeline runs reference different problem ids')
        if 'problem' in fit_pipeline_run and fit_pipeline_run['problem'].get_digest() != produce_pipeline_run['problem'].get_digest():
            raise exceptions.InvalidArgumentValueError('FIT phase and PRODUCE phase pipeline runs reference different problem digests')
        if 'scoring' not in produce_pipeline_run['run']:
            raise exceptions.InvalidArgumentValueError("Fit-score requires the PRODUCE phase pipeline run to be a pipeline run with scoring.")

        # TODO: Check that hyperparams match between both pipeline runs (but allow failed runs).
        # TODO: Check that inputs match between both pipeline runs.
        # TODO: Check that scoring pipelines match between both pipeline runs.

        pipeline = fit_pipeline_run['pipeline']
        scoring_pipeline = produce_pipeline_run['run']['scoring']['pipeline']
        problem_description = fit_pipeline_run.get('problem', None)
        # Currently, "random_seed" is not yet required.
        random_seed = fit_pipeline_run.get('random_seed', 0)
        hyperparams: typing.Optional[typing.Sequence[typing.Union[typing.Dict, typing.Sequence]]] = \
            _get_runtime_hyperparams_from_pipeline_run(fit_pipeline_run['pipeline'], fit_pipeline_run.get('steps', []))
        # Currently, "random_seed" is not yet required.
        scoring_random_seed = produce_pipeline_run['run']['scoring'].get('random_seed', 0)
        # We do not have to set metrics, because they should already be included in hyper-paramters.
        metrics: typing.Sequence[typing.Dict] = []
        scoring_params = _get_data_and_scoring_params_from_pipeline_run(produce_pipeline_run['run']['scoring'].get('steps', []))

    else:
        fit_pipeline_run = None
        produce_pipeline_run = None

        pipeline = pipeline_resolver(
            arguments.pipeline,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        )
        scoring_pipeline = pipeline_resolver(
            arguments.scoring_pipeline,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        )

        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem, strict_digest=getattr(arguments, 'strict_digest', False))
        else:
            problem_description = None

        random_seed = getattr(arguments, 'random_seed', 0)
        hyperparams = None
        scoring_random_seed = getattr(arguments, 'scoring_random_seed', 0)

        if getattr(arguments, 'metrics', None) is not None:
            metrics = get_metrics_from_list(arguments.metrics)
        else:
            metrics = get_metrics_from_problem_description(problem_description)

        if getattr(arguments, 'scoring_params', None) is not None:
            scoring_params = {name: value for name, value in arguments.scoring_params}
        else:
            scoring_params = {}

    inputs_config = _get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
        fit_pipeline_run=fit_pipeline_run,
        produce_pipeline_run=produce_pipeline_run,
    )

    data_pipeline_run = None
    data_result: typing.Optional[Result]
    inputs: typing.Sequence[container.Dataset]
    test_inputs: typing.Sequence[container.Dataset]
    score_inputs: typing.Sequence[container.Dataset]
    # We run a data preparation pipeline here instead of passing it into
    # "fit", "produce", and "score" functions so that we run it only once.
    if inputs_config.data_pipeline is not None:
        if inputs_config.inputs is None:
            if fit_pipeline_run is None and produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input full data is missing in the pipeline run.")

        outputs, data_result = prepare_data(
            inputs_config.inputs,
            data_pipeline=inputs_config.data_pipeline,
            problem_description=problem_description,
            data_params=inputs_config.data_params,
            context=context,
            random_seed=inputs_config.data_random_seed,
            volumes_dir=getattr(arguments, 'volumes_dir', None),
            scratch_dir=getattr(arguments, 'scratch_dir', None),
            runtime_environment=runtime_environment,
        )
        data_result.check_success()
        if len(outputs[0]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of train data but {folds} folds.".format(
                folds=len(outputs[0]),
            ))
        if len(outputs[1]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of test data but {folds} folds.".format(
                folds=len(outputs[1]),
            ))
        if len(outputs[2]) != 1:
            raise ValueError("Data preparation pipeline has not returned 1 fold of score data but {folds} folds.".format(
                folds=len(outputs[2]),
            ))
        inputs = [outputs[0][0]]
        test_inputs = [outputs[1][0]]
        score_inputs = [outputs[2][0]]
    else:
        if inputs_config.inputs is None:
            if fit_pipeline_run is None and produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input train data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input train data is missing in the pipeline run.")
        if inputs_config.test_inputs is None:
            if fit_pipeline_run is None and produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input test data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input test data is missing in the pipeline run.")
        if inputs_config.score_inputs is None:
            if fit_pipeline_run is None and produce_pipeline_run is None:
                raise exceptions.InvalidArgumentValueError("Input score data is missing in arguments.")
            else:
                raise exceptions.InvalidPipelineRunError("Input score data is missing in the pipeline run.")

        if getattr(arguments, 'data_pipeline_run', None) is not None:
            with open(arguments.data_pipeline_run, 'rb') as file:
                data_pipeline_run = pickle.load(file)

            if data_pipeline_run.is_failed():
                raise exceptions.InvalidPipelineRunError("Provided data preparation pipeline run has failed.")

        data_result = None
        inputs = inputs_config.inputs
        test_inputs = inputs_config.test_inputs
        score_inputs = inputs_config.score_inputs

    fold_group_uuid = getattr(arguments, 'fold_group_uuid', None)
    if fold_group_uuid is None:
        fold_group_uuid = uuid.uuid4()
    if data_result is not None:
        assert data_pipeline_run is None
        data_pipeline_run = data_result.pipeline_run

    fitted_pipeline, predictions, fit_result = fit(
        pipeline, inputs,
        problem_description=problem_description,
        context=context,
        hyperparams=hyperparams,
        random_seed=random_seed,
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=fold_group_uuid, fold_index=getattr(arguments, 'fold_index', 0),
    )

    if fit_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run])

        fit_result.check_success()

        assert False

    assert fitted_pipeline is not None

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)
        # Make sure the handle is flushed so that no data is lost. CLI file handles are generally
        # used outside of a context manager which would otherwise handle that.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/436
        arguments.save.flush()

    expose_produced_outputs = getattr(arguments, 'expose_produced_outputs_dir', None) is not None

    predictions, produce_result = produce(
        fitted_pipeline, test_inputs,
        expose_produced_outputs=expose_produced_outputs,
        data_pipeline_run=data_pipeline_run,
        fold_group_uuid=fold_group_uuid, fold_index=getattr(arguments, 'fold_index', 0),
    )

    if expose_produced_outputs:
        save_exposed_outputs(produce_result, arguments.expose_produced_outputs_dir)

    if produce_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

        produce_result.check_success()

        assert False

    assert predictions is not None

    if getattr(arguments, 'output', None) is not None:
        predictions.to_csv(arguments.output)

    scores, score_result = score(
        predictions, score_inputs,
        scoring_pipeline=scoring_pipeline,
        problem_description=problem_description,
        metrics=metrics,
        predictions_random_seed=fitted_pipeline.random_seed,
        scoring_params=scoring_params, context=context,
        random_seed=scoring_random_seed,
        volumes_dir=fitted_pipeline.volumes_dir,
        scratch_dir=fitted_pipeline.scratch_dir,
        runtime_environment=runtime_environment,
    )

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=score_inputs if data_result is None and data_pipeline_run is None else None,
    )

    if score_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

        score_result.check_success()

        assert False

    assert scores is not None

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, metrics=metrics, scores=scores,
    )

    _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


# We have "pipeline_run_parser" as an arguments (even if we are not
# using it in this function) so that the signature is the same for all handlers.
def score_predictions_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    pipeline_run_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]
    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )
    if scoring_pipeline is None:
        raise exceptions.InvalidStateError("Pipeline has not been resolved.")

    if getattr(arguments, 'problem', None) is not None:
        problem_description = problem_resolver(arguments.problem, strict_digest=getattr(arguments, 'strict_digest', False))
    else:
        problem_description = None

    if arguments.data_split_file == '-':
        predictions_file = sys.stdin.buffer
    else:
        predictions_file = arguments.predictions
    predictions_dataframe = pandas.read_csv(
        predictions_file,
        # We do not want to do any conversion of values at this point.
        # This should be done by primitives later on.
        dtype=str,
        # We always expect one row header.
        header=0,
        # We want empty strings and not NaNs.
        na_filter=False,
        encoding='utf8',
        low_memory=False,
        memory_map=True,
    )
    predictions_random_seed = getattr(arguments, 'predictions_random_seed', None)
    scoring_random_seed = getattr(arguments, 'scoring_random_seed', 0)

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(problem_description)

    if getattr(arguments, 'scoring_params', None) is not None:
        scoring_params = {name: value for name, value in arguments.scoring_params}
    else:
        scoring_params = {}

    # Convert pandas DataFrame to container DataFrame.
    predictions = container.DataFrame(predictions_dataframe, generate_metadata=True)

    inputs_config = _get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
    )

    if inputs_config.score_inputs is None:
        if inputs_config.data_pipeline is not None:
            raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")
        else:
            raise exceptions.InvalidArgumentValueError("Input score data is missing in arguments.")

    scores, score_result = score(
        predictions, inputs_config.score_inputs,
        scoring_pipeline=scoring_pipeline,
        problem_description=problem_description,
        metrics=metrics,
        predictions_random_seed=predictions_random_seed,
        scoring_params=scoring_params,
        context=context,
        random_seed=scoring_random_seed,
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
        data_pipeline=inputs_config.data_pipeline,
        data_params=inputs_config.data_params,
        data_random_seed=inputs_config.data_random_seed,
    )

    score_result.check_success()

    assert scores is not None

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


def _group_pipeline_runs_by_fold_group(
    pipeline_runs: typing.Sequence[typing.Dict[str, typing.Any]], default_group: str = '',
) -> typing.Dict[str, typing.Sequence[typing.Dict[str, typing.Any]]]:
    """
    Groups pipeline runs by pipeline_run.run.fold_group.id.

    Params
    ------
    pipeline_runs:
        The pipeline runs to group.
    default_group:
        Groups all pipeline runs that do not have a fold group under this key.

    Return
    ------
    grouped_pipeline_runs:
        The pipeline runs grouped by fold group id.
    """

    grouped_pipeline_runs: typing.Dict[str, typing.Sequence[typing.Dict[str, typing.Any]]] = {
        key: list(value) for key, value in itertools.groupby(
            pipeline_runs,
            lambda pipeline_run: pipeline_run['run'].get('fold_group', {}).get('id', default_group)
        )
    }

    assert sum(len(group) for group in grouped_pipeline_runs.values()) == len(pipeline_runs)

    return grouped_pipeline_runs


def _sort_pipeline_runs(
    pipeline_runs: typing.Sequence[typing.Dict[str, typing.Any]]
) -> typing.Sequence[typing.Sequence[typing.Dict[str, typing.Any]]]:
    """
    Performs a topological sort of pipeline runs by pipeline_run.previous_pipeline_run.id. This separates disconnected trees into separate sequences.

    Params
    ------
    pipeline_runs:
        The pipeline runs to sort.

    Return
    ------
    sorted_pipeline_runs:
        The pipeline runs sorted by previous pipeline run id.
    """

    pipeline_run_id_map: typing.Dict[str, typing.Dict[str, typing.Any]] = {}  # id: pipeline_run
    next_id_map: typing.Dict[str, typing.List[str]] = {}  # id: list of ids that point to key id
    tree_roots: typing.List[str] = []  # ids that don't have a previous pipeline run
    for pipeline_run in pipeline_runs:
        id_ = pipeline_run['id']
        pipeline_run_id_map[id_] = pipeline_run
        if 'previous_pipeline_run' in pipeline_run:
            prev_id = pipeline_run['previous_pipeline_run']['id']
            if prev_id not in next_id_map:
                next_id_map[prev_id] = []
            next_id_map[prev_id].append(id_)
        else:
            tree_roots.append(id_)

    if not tree_roots:
        raise exceptions.InvalidArgumentValueError("No first pipeline run.")

    sorted_pipeline_runs = []
    for root_id in tree_roots:
        sorted_tree = [pipeline_run_id_map[root_id]]
        if root_id in next_id_map:
            stack = list(next_id_map[root_id])
            while stack:
                id_ = stack.pop()
                sorted_tree.append(pipeline_run_id_map[id_])
                if id_ in next_id_map:
                    stack += next_id_map[id_]
        sorted_pipeline_runs.append(sorted_tree)

    all_pipeline_run_ids = set(pipeline_run_id_map.keys())
    sorted_pipeline_run_ids = {pipeline_run['id'] for tree in sorted_pipeline_runs for pipeline_run in tree}
    extra_pipeline_run_ids = all_pipeline_run_ids - sorted_pipeline_run_ids
    if extra_pipeline_run_ids:
        raise exceptions.InvalidArgumentValueError("Pipeline runs with previous pipeline runs which were not provided: {extra_pipeline_run_ids}".format(
            extra_pipeline_run_ids=sorted(extra_pipeline_run_ids),
        ))

    return sorted_pipeline_runs


def _group_sort_pipeline_runs(
    pipeline_runs: typing.Sequence[typing.Dict[str, typing.Any]], default_group: str = ''
) -> typing.Dict[str, typing.Sequence[typing.Sequence[typing.Dict[str, typing.Any]]]]:
    """
    Groups pipeline runs by pipeline_run.run.fold_group.id.
    Then performs a topological sort of each group of pipeline runs by pipeline_run.previous_pipeline_run.id.

    Params
    ------
    pipeline_runs:
        The pipeline runs to group.
    default_group:
        Groups all pipeline runs that do not have a fold group under this key.

    Return
    ------
    grouped_pipeline_runs:
        The pipeline runs grouped by fold group id.
    """

    grouped_pipeline_runs = {}
    for group_id, group in _group_pipeline_runs_by_fold_group(pipeline_runs, default_group).items():
        try:
            grouped_pipeline_runs[group_id] = _sort_pipeline_runs(group)
        except Exception as error:
            raise exceptions.InvalidArgumentValueError("Failed sorting pipeline runs in fold group \"{group_id}\".".format(group_id=group_id)) from error
    return grouped_pipeline_runs


def _validate_fold_of_pipeline_runs_for_evaluate_handler(pipeline_runs: typing.Sequence[typing.Dict[str, typing.Any]]) -> None:
    if len(pipeline_runs) != 2:
        raise exceptions.InvalidArgumentValueError("Each fold should only have 2 pipeline runs.")
    if pipeline_runs[0]['run']['phase'] != metadata_base.PipelineRunPhase.FIT:
        raise exceptions.InvalidArgumentValueError("The first pipeline run in fold must be phase FIT, not {phase}.".format(phase=pipeline_runs[0]['run']['phase']))
    if pipeline_runs[1]['run']['phase'] != metadata_base.PipelineRunPhase.PRODUCE:
        raise exceptions.InvalidArgumentValueError("The second pipeline run in fold must be phase PRODUCE, not {phase}.".format(phase=pipeline_runs[1]['run']['phase']))


def _process_pipeline_runs_for_evaluate_handler(
    pipeline_runs: typing.Sequence[typing.Dict[str, typing.Any]],
) -> typing.Tuple[
    pipeline_module.Pipeline, typing.Sequence[container.Dataset],
    pipeline_module.Pipeline, pipeline_module.Pipeline, typing.Optional[problem.Problem],
    typing.Optional[typing.Sequence], typing.Optional[typing.Dict[str, str]],
    typing.Optional[typing.Dict[str, str]], int, int, int,
]:
    if not pipeline_runs:
        raise exceptions.InvalidArgumentValueError("No pipeline runs.")

    grouped_sorted_pipeline_runs = _group_sort_pipeline_runs(pipeline_runs)

    if len(grouped_sorted_pipeline_runs) > 1:
        raise exceptions.InvalidArgumentValueError("All pipeline runs must belong to the same fold group, but got {fold_groups}.".format(
            fold_groups=sorted(grouped_sorted_pipeline_runs.keys()),
        ))

    sorted_pipeline_runs = grouped_sorted_pipeline_runs.popitem()[1]

    # We know there should be at least one pipeline run.
    assert len(sorted_pipeline_runs) >= 1

    first_pipeline_runs = sorted_pipeline_runs[0]

    _validate_fold_of_pipeline_runs_for_evaluate_handler(first_pipeline_runs)

    fit_pipeline_run = first_pipeline_runs[0]
    produce_pipeline_run = first_pipeline_runs[1]

    if 'scoring' not in produce_pipeline_run['run']:
        raise exceptions.InvalidArgumentValueError("Evaluate requires the PRODUCE phase pipeline run to be a pipeline run with scoring.")
    if 'data_preparation' not in fit_pipeline_run['run']:
        raise exceptions.InvalidArgumentValueError("Evaluate requires the FIT phase pipeline run to be a pipeline run with data preparation.")

    pipeline = fit_pipeline_run['pipeline']
    inputs = fit_pipeline_run['datasets']
    data_pipeline = fit_pipeline_run['run']['data_preparation']['pipeline']
    scoring_pipeline = produce_pipeline_run['run']['scoring']['pipeline']
    problem_description = fit_pipeline_run.get('problem', None)
    hyperparams = _get_runtime_hyperparams_from_pipeline_run(fit_pipeline_run['pipeline'], fit_pipeline_run.get('steps', []))
    # Currently, "random_seed" is not yet required.
    random_seed = fit_pipeline_run.get('random_seed', 0)
    # Currently, "random_seed" is not yet required.
    data_random_seed = fit_pipeline_run['run']['data_preparation'].get('random_seed', 0)
    # Currently, "random_seed" is not yet required.
    scoring_random_seed = produce_pipeline_run['run']['scoring'].get('random_seed', 0)
    data_params = _get_data_and_scoring_params_from_pipeline_run(fit_pipeline_run['run']['data_preparation'].get('steps', []))
    scoring_params = _get_data_and_scoring_params_from_pipeline_run(produce_pipeline_run['run']['scoring'].get('steps', []))
    number_of_folds = _get_number_of_folds(data_params)

    if len(sorted_pipeline_runs) != number_of_folds:
        raise exceptions.InvalidArgumentValueError(
            "The number of folds configured for the data preparation pipeline ({folds1}) does not match the number of folds of provided pipeline runs ({folds2}).".format(
                folds1=number_of_folds,
                folds2=len(sorted_pipeline_runs),
            ),
        )

    fold_indices = set()
    # We redo validation of the first pipeline runs partially to fully validate them.
    for fold_pipeline_runs in sorted_pipeline_runs:
        # Validation inside a single fold.
        _validate_fold_of_pipeline_runs_for_evaluate_handler(fold_pipeline_runs)

        # Fold index validation.
        if fold_pipeline_runs[0]['run']['fold_group']['fold'] != fold_pipeline_runs[1]['run']['fold_group']['fold']:
            raise exceptions.InvalidArgumentValueError("Pipeline run fold indices do not match: {index1} vs. {index2}".format(
                index1=fold_pipeline_runs[0]['run']['fold_group']['fold'],
                index2=fold_pipeline_runs[1]['run']['fold_group']['fold'],
            ))
        fold_index = fold_pipeline_runs[0]['run']['fold_group']['fold']
        if fold_index >= number_of_folds:
            raise exceptions.InvalidArgumentValueError(
                "The fold index for a pipeline run ({fold_index}) is out of bounds for the number of folds ({number_of_folds}) in the data preparation pipeline.".format(
                    fold_index=fold_index,
                    number_of_folds=number_of_folds,
                ),
            )
        if fold_index in fold_indices:
            raise exceptions.InvalidArgumentValueError("Pipeline runs have duplicate fold index {fold_index}.".format(
                fold_index=fold_index,
            ))
        fold_indices.add(fold_index)

        # Consistency across all folds validation.
        for pipeline_run in fold_pipeline_runs:
            # Validate same pipeline.
            if pipeline_run['pipeline'].id != pipeline.id or pipeline_run['pipeline'].get_digest() != pipeline.get_digest():
                raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same pipeline.")

            # Validate same data preparation pipeline.
            # TODO: Validate that the phase of this pipeline run is FIT.
            if (
                pipeline_run['run']['data_preparation']['pipeline'].id != data_pipeline.id or
                pipeline_run['run']['data_preparation']['pipeline'].get_digest() != data_pipeline.get_digest()
            ):
                raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same data preparation pipeline.")

            # Validate same scoring pipeline.
            # TODO: Validate that the phase of this pipeline run is FIT.
            if pipeline_run['run']['phase'] == metadata_base.PipelineRunPhase.PRODUCE.name:
                if (
                    pipeline_run['run']['scoring']['pipeline'].id != scoring_pipeline.id or
                    pipeline_run['run']['scoring']['pipeline'].get_digest() != scoring_pipeline.get_digest()
                ):
                    raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same scoring pipeline.")

            # Validate same problem.
            if (
                problem_description is None and pipeline_run.get('problem', None) is not None or
                problem_description != pipeline_run.get('problem', None)
            ):
                raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same problem description.")

            # Validate same inputs.
            if (
                set(dataset.metadata.query(())['id'] for dataset in pipeline_run['datasets']) != set(dataset.metadata.query(())['id'] for dataset in inputs) or
                set(dataset.metadata.query(())['digest'] for dataset in pipeline_run['datasets']) != set(dataset.metadata.query(())['digest'] for dataset in inputs)
            ):
                raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same input datasets.")

            # Validate same random seed.
            if pipeline_run.get('random_seed', 0) != random_seed:
                raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same random seed.")

            # Validate that hyperparams match between both pipeline runs (but allow failed runs).
            if pipeline_run['run']['phase'] == metadata_base.PipelineRunPhase.FIT.name:
                if _get_runtime_hyperparams_from_pipeline_run(pipeline_run['pipeline'], pipeline_run.get('steps', [])) != hyperparams:
                    raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same runtime hyper-parameters.")

            # Validate same data prep random seed.
            if pipeline_run['run']['data_preparation'].get('random_seed', 0) != data_random_seed:
                raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same data preparation pipeline random seed.")

            # Validate same scoring random seed.
            if pipeline_run['run']['phase'] == metadata_base.PipelineRunPhase.PRODUCE.name:
                if pipeline_run['run']['scoring'].get('random_seed', 0) != scoring_random_seed:
                    raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same scoring pipeline random seed.")

            # Validate same data preparation params.
            if _get_data_and_scoring_params_from_pipeline_run(pipeline_run['run']['data_preparation'].get('steps', [])) != data_params:
                raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same data preparation pipeline params.")

            # Validate same scoring params.
            if pipeline_run['run']['phase'] == metadata_base.PipelineRunPhase.PRODUCE.name:
                if _get_data_and_scoring_params_from_pipeline_run(pipeline_run['run']['scoring'].get('steps', [])) != scoring_params:
                    raise exceptions.InvalidArgumentValueError("Pipeline runs do not all use the same scoring params.")

    expected_fold_indices = set(range(number_of_folds))
    if fold_indices != expected_fold_indices:
        raise exceptions.InvalidArgumentValueError("Not expected fold indices: {fold_indices}".format(
          fold_indices=sorted(fold_indices),
        ))

    return pipeline, inputs, data_pipeline, scoring_pipeline, problem_description, \
        hyperparams, data_params, scoring_params, random_seed, data_random_seed, \
        scoring_random_seed


def evaluate_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, pipeline_run_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if pipeline_run_parser is None:
        pipeline_run_parser = parse_pipeline_run
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem.get_problem

    context = metadata_base.Context[arguments.context]
    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]
    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    if getattr(arguments, 'input_run', None) is not None:
        parsed_pipeline_runs = pipeline_run_parser(
            arguments.input_run, getattr(arguments, 'pipeline_search_paths', []), getattr(arguments, 'datasets_dir', None),
            pipeline_resolver=pipeline_resolver, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            compute_digest=compute_digest, strict_digest=getattr(arguments, 'strict_digest', False),
        )

        # We require that the input_run was generated using this or equivalent evaluate method.
        pipeline, inputs, data_pipeline, scoring_pipeline, problem_description, \
            hyperparams, data_params, scoring_params, random_seed, data_random_seed, \
            scoring_random_seed = _process_pipeline_runs_for_evaluate_handler(parsed_pipeline_runs)

        # We do not have to populate metrics, because they should already be included in hyper-paramters.
        metrics: typing.Sequence[typing.Dict] = []

    else:
        pipeline = typing.cast(pipeline_module.Pipeline, pipeline_resolver(
            arguments.pipeline,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        ))
        scoring_pipeline = typing.cast(pipeline_module.Pipeline, pipeline_resolver(
            arguments.scoring_pipeline,
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        ))

        if getattr(arguments, 'problem', None) is not None:
            problem_description = problem_resolver(arguments.problem, strict_digest=getattr(arguments, 'strict_digest', False))
        else:
            problem_description = None

        random_seed = getattr(arguments, 'random_seed', 0)
        hyperparams = None
        scoring_random_seed = getattr(arguments, 'scoring_random_seed', 0)

        if getattr(arguments, 'metrics', None) is not None:
            metrics = get_metrics_from_list(arguments.metrics)
        else:
            metrics = get_metrics_from_problem_description(problem_description)

        if getattr(arguments, 'scoring_params', None) is not None:
            scoring_params = {name: value for name, value in arguments.scoring_params}
        else:
            scoring_params = {}

        inputs_config = _get_inputs_config_from_arguments(
            arguments=arguments,
            pipeline_resolver=pipeline_resolver,
            dataset_resolver=dataset_resolver,
        )

        if inputs_config.data_pipeline is None:
            raise exceptions.InvalidArgumentValueError("Data preparation pipeline is missing in arguments.")
        if inputs_config.inputs is None:
            raise exceptions.InvalidArgumentValueError("Input full data is missing in arguments.")

        inputs = inputs_config.inputs
        data_pipeline = inputs_config.data_pipeline
        data_params = inputs_config.data_params
        data_random_seed = inputs_config.data_random_seed

    scores_list, results_list = evaluate(
        pipeline,
        inputs,
        data_pipeline=data_pipeline,
        scoring_pipeline=scoring_pipeline,
        problem_description=problem_description,
        data_params=data_params,
        metrics=metrics,
        scoring_params=scoring_params,
        context=context,
        hyperparams=hyperparams,
        random_seed=random_seed,
        data_random_seed=data_random_seed,
        scoring_random_seed=scoring_random_seed,
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    _output_pipeline_runs(arguments, results_list.pipeline_runs)

    results_list.check_success()

    scores = combine_folds(scores_list)

    if getattr(arguments, 'scores', None) is not None:
        scores.to_csv(arguments.scores)


@deprecate.function(message="use save_exposed_outputs instead")
def save_steps_outputs(results: typing.Union[Result, MultiResult], output_dir: str) -> None:
    save_exposed_outputs(results, output_dir)


def save_exposed_outputs(results: typing.Union[Result, MultiResult], output_dir: str) -> None:
    if isinstance(results, Result):
        for key, step_output in results.values.items():
            container_utils.save_container(step_output, os.path.join(output_dir, key))
    elif isinstance(results, MultiResult):
        for i, result in enumerate(results):
            for key, step_output in result.values.items():
                container_utils.save_container(step_output, os.path.join(output_dir, str(i), key))
    else:
        raise exceptions.UnexpectedTypeError("Type: {results_type}".format(results_type=type(results)))


def main(argv: typing.Sequence) -> None:
    # We have to disable importing while type checking because it makes
    # an import cycle in mypy which makes many typing errors.
    if not typing.TYPE_CHECKING:
        # Importing here to prevent import cycle.
        from d3m import cli

        logging.basicConfig()

        logger.warning("This CLI is deprecated. Use \"python3 -m d3m runtime\" instead.")

        parser = argparse.ArgumentParser(description="Run D3M pipelines.")
        cli.runtime_configure_parser(parser)

        arguments = parser.parse_args(argv[1:])

        cli.runtime_handler(arguments, parser)


if __name__ == '__main__':
    main(sys.argv)
