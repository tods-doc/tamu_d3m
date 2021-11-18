import argparse
import shutil

import logging
import os
import requests
import sys
import typing

from d3m import exceptions, runtime, utils
from d3m.metadata import base as metadata_base
from d3m.metadata import pipeline as pipeline_module
from d3m.container import dataset as dataset_module
from d3m.metadata import pipeline_run as pipeline_run_module, problem as problem_module

logger = logging.getLogger(__name__)


def crawl_openml_task(
    datasets: typing.Dict[str, str], task_id: int, save_dir: str, *, data_pipeline: pipeline_module.Pipeline,
    data_params: typing.Dict[str, str] = None, context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
    dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
    compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False,
) -> None:
    """
    A function that crawls an OpenML task and corresponding dataset, do the split using a data
    preparation pipeline, and stores the splits as D3M dataset and problem description.

    Parameters
    ----------
    datasets:
        A mapping between known dataset IDs and their paths. Is updated in-place.
    task_id:
        An integer representing and OpenML task id to crawl and convert.
    save_dir:
        A directory where to save datasets and problems.
    data_pipeline:
        A data preparation pipeline used for splitting.
    data_params:
        A dictionary that contains the hyper-parameters for the data prepration pipeline.
    context:
        In which context to run pipelines.
    random_seed:
        A random seed to use for every run. This control all randomness during the run.
    volumes_dir:
        Path to a directory with static files required by primitives.
    scratch_dir:
        Path to a directory to store any temporary files needed during execution.
    runtime_environment:
        A description of the runtime environment.
    dataset_resolver:
        A dataset resolver to use.
    problem_resolver:
        A problem description resolver to use.
    compute_digest:
        Compute a digest over the data?
    strict_digest:
        If computed digest does not match the one provided in metadata, raise an exception?
    """

    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem_module.get_problem

    number_of_folds = runtime._get_number_of_folds(data_params)
    assert number_of_folds != 0

    problem_uri = f'https://www.openml.org/t/{task_id}'
    problem_description = problem_resolver(problem_uri, strict_digest=strict_digest)

    if len(problem_description['inputs']) != 1:
        raise exceptions.NotSupportedError("OpenML problem descriptions with multiple inputs are not supported.")

    problem_description_input = problem_description['inputs'][0]
    input_dataset_id = problem_description_input['dataset_id']

    known_datasets_set = set(datasets.keys())
    needed_splits_set = set()
    # We make sure when splitting that the output dataset has the same ID as the input dataset
    # with additional suffix for split type, and we are taking the advantage of this here.
    # The naming scheme matches "runtime._get_split_dataset_id".
    if number_of_folds == 1:
        needed_splits_set.add(f'{input_dataset_id}_TRAIN')
        needed_splits_set.add(f'{input_dataset_id}_TEST')
        needed_splits_set.add(f'{input_dataset_id}_SCORE')
        dataset_view_maps = [{
            'train': [
                {
                    'from': input_dataset_id,
                    'to': f'{input_dataset_id}_TRAIN',
                },
            ],
            'test': [
                {
                    'from': input_dataset_id,
                    'to': f'{input_dataset_id}_TEST',
                },
            ],
            'score': [
                {
                    'from': input_dataset_id,
                    'to': f'{input_dataset_id}_SCORE',
                },
            ],
        }]
    else:
        dataset_view_maps = []
        for fold_index in range(number_of_folds):
            needed_splits_set.add(f'{input_dataset_id}_FOLD_{fold_index}_TRAIN')
            needed_splits_set.add(f'{input_dataset_id}_FOLD_{fold_index}_TEST')
            needed_splits_set.add(f'{input_dataset_id}_FOLD_{fold_index}_SCORE')
            dataset_view_maps.append({
                'train': [
                    {
                        'from': input_dataset_id,
                        'to': f'{input_dataset_id}_FOLD_{fold_index}_TRAIN',
                    },
                ],
                'test': [
                    {
                        'from': input_dataset_id,
                        'to': f'{input_dataset_id}_FOLD_{fold_index}_TEST',
                    },
                ],
                'score': [
                    {
                        'from': input_dataset_id,
                        'to': f'{input_dataset_id}_FOLD_{fold_index}_SCORE',
                    },
                ],
            })

    # We already have this split, we can just reuse it.
    if problem_description_input['dataset_id'] in known_datasets_set and needed_splits_set <= known_datasets_set:
        logger.debug("Copying existing splits.")

        # Copy splits.
        if number_of_folds == 1:
            view_maps = dataset_view_maps[0]
            for split_type in ['train', 'test', 'score']:
                shutil.copytree(
                    os.path.dirname(datasets[runtime._get_dataset_id_from_view_maps(view_maps, split_type, input_dataset_id)]),
                    os.path.join(save_dir, split_type.upper(), f'dataset_{split_type.upper()}'),
                )

                # Save problem description for the split. We do not copy because we copy only datasets.
                problem_path = os.path.abspath(os.path.join(save_dir, split_type.upper(), f'problem_{split_type.upper()}', 'problemDoc.json'))
                runtime._save_problem_description(problem_description, problem_path, dataset_view_maps=view_maps)
        else:
            for fold_index, view_maps in enumerate(dataset_view_maps):
                for split_type in ['train', 'test', 'score']:
                    shutil.copytree(
                        os.path.dirname(datasets[runtime._get_dataset_id_from_view_maps(view_maps, split_type, input_dataset_id)]),
                        os.path.join(save_dir, 'folds', str(fold_index), split_type.upper(), f'dataset_{split_type.upper()}'),
                    )

                    # Save problem description for the split. We do not copy because we copy only datasets.
                    problem_path = os.path.abspath(os.path.join(save_dir, 'folds', str(fold_index), split_type.upper(), f'problem_{split_type.upper()}', 'problemDoc.json'))
                    runtime._save_problem_description(problem_description, problem_path, dataset_view_maps=view_maps)

        # Copy data preparation pipeline run pickle.
        shutil.copy2(
            os.path.join(os.path.dirname(datasets[input_dataset_id]), '..', runtime.DATA_PIPELINE_RUN_FILENAME),
            os.path.join(save_dir, runtime.DATA_PIPELINE_RUN_FILENAME),
        )

        # Copy full dataset.
        shutil.copytree(
            os.path.dirname(datasets[input_dataset_id]),
            os.path.join(save_dir, input_dataset_id),
        )

    else:
        logger.debug("Running a data preparation pipeline.")

        openml_dataset_id = int(input_dataset_id.split('_')[-1])
        dataset_uri = f'https://www.openml.org/d/{openml_dataset_id}'
        dataset = dataset_resolver(
            dataset_uri, compute_digest=compute_digest, strict_digest=strict_digest,
        )
        dataset_id = dataset.metadata.query_field((), 'id')

        if input_dataset_id != dataset_id:
            raise exceptions.InvalidDatasetError(f"Loaded dataset (\"{dataset_id}\") does not have the expected dataset ID (\"{input_dataset_id}\").")

        # Make splits and save them. This saves the pipeline run made by the data preparation pipeline, too.
        runtime.prepare_data_and_save(
            save_dir=save_dir, inputs=[dataset], data_pipeline=data_pipeline, problem_description=problem_description,
            data_params=data_params, context=context,
            random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
            runtime_environment=runtime_environment,
            # We provide "dataset_view_maps" to force split dataset IDs.
            dataset_view_maps=dataset_view_maps,
        )

        # Save full dataset.
        dataset_path = os.path.abspath(os.path.join(save_dir, dataset_id, 'datasetDoc.json'))
        dataset_uri = utils.path_to_uri(dataset_path)
        dataset.save(dataset_uri)

        # Updating known datasets.
        datasets[dataset_id] = dataset_path
        # We make sure when splitting that the output dataset has the same ID as the input dataset
        # with additional suffix for split type, and we are taking the advantage of this here.
        # The naming scheme matches "runtime._get_split_dataset_id".
        if number_of_folds == 1:
            for split_type in ['TRAIN', 'TEST', 'SCORE']:
                datasets[f'{dataset_id}_{split_type}'] = os.path.join(save_dir, split_type, f'dataset_{split_type}', 'datasetDoc.json')
        else:
            for fold_index in range(number_of_folds):
                for split_type in ['TRAIN', 'TEST', 'SCORE']:
                    datasets[f'{dataset_id}_FOLD_{fold_index}_{split_type}'] = os.path.join(save_dir, 'folds', str(fold_index), split_type, f'dataset_{split_type}', 'datasetDoc.json')

    # Save problem description. For splits, problem description is saved by "runtime.prepare_data_and_save".
    problem_path = os.path.abspath(os.path.join(save_dir, problem_description['id'], 'problemDoc.json'))
    # We do not save "dataset_view_maps" for this problem description.
    runtime._save_problem_description(problem_description, problem_path)


def crawl_openml(
    save_dir: str, task_types: typing.Sequence[problem_module.OpenMLTaskType], *,
    data_pipeline: pipeline_module.Pipeline,
    data_params: typing.Dict[str, str] = None,
    context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
    max_tasks: typing.Optional[int] = None,
    ignore_tasks: typing.Sequence[int] = [],
    ignore_datasets: typing.Sequence[int] = [],
    dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
    compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False,
) -> bool:
    """
    A function that crawls OpenML tasks and corresponding datasets
    and converts them to D3M datasets and problems.

    Parameters
    ----------
    save_dir:
        A directory where to save datasets and problems.
    task_types:
        Task types to crawl.
    data_pipeline:
        A data preparation pipeline used for splitting.
    data_params:
        A dictionary that contains the hyper-parameters for the data prepration pipeline.
    context:
        In which context to run pipelines.
    random_seed:
        A random seed to use for every run. This control all randomness during the run.
    volumes_dir:
        Path to a directory with static files required by primitives.
    scratch_dir:
        Path to a directory to store any temporary files needed during execution.
    runtime_environment:
        A description of the runtime environment.
    max_tasks:
        Maximum number of tasks to crawl, no limit if ``None`` or 0.
    dataset_resolver:
        A dataset resolver to use.
    problem_resolver:
        A problem description resolver to use.
    compute_digest:
        Compute a digest over the data?
    strict_digest:
        If computed digest does not match the one provided in metadata, raise an exception?

    Returns
    -------
    A boolean set to true if there was an error during the call.
    """

    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem_module.get_problem

    # We load already saved datasets.
    datasets = utils.get_datasets_and_problems(save_dir, ignore_duplicate=True)[0]

    has_errored = False
    current_tasks = 0

    for task_type in task_types:
        try:
            response = requests.get(f'https://www.openml.org/api/v1/json/task/list/type/{task_type.value}')
            response.raise_for_status()
            tasks = response.json()['tasks']['task']
        except requests.HTTPError:
            has_errored = True
            logger.exception("Tasks for task type '%(task_type)s' failed to load, skipping them.", {'task_type': task_type})
            continue

        logger.info("Loaded %(tasks)s tasks for task type '%(task_type)s'.", {'tasks': len(tasks), 'task_type': task_type})

        for task in tasks:
            if max_tasks is not None and max_tasks > 0 and max_tasks == current_tasks:
                break

            # Check if the task is active.
            if task['status'] != 'active':
                logger.info("Task '%(task_id)s' not active. Skipping.", {'task_id': task['task_id']})
                continue

            task_path = os.path.join(save_dir, f"openml_task_{task['task_id']}")
            if os.path.exists(task_path):
                logger.info("Output directory '%(task_path)s' for task '%(task_id)s' already exists. Skipping.", {'task_path': task_path, 'task_id': task['task_id']})
                continue

            if ignore_tasks or ignore_datasets:
                openml_problem_uri = 'https://www.openml.org/t/{task_id}'.format(task_id=task['task_id'])
                try:
                    problem_description = problem_module.Problem.load(openml_problem_uri)

                    # We get the dataset id from the problem
                    dataset_id = int(problem_description['inputs'][0]['dataset_id'].split('_')[-1])

                    if task['task_id'] in ignore_tasks or dataset_id in ignore_datasets:
                        logger.info("Ignore task '%(task_id)s' and dataset '%(input_dataset_id)s'.",
                                    {'task_id': task['task_id'], 'input_dataset_id': dataset_id})
                        continue
                except Exception:
                    has_errored = True
                    logger.exception("Error crawling task '%(task_id)s'.", {'task_id': task['task_id']})

                    # Cleanup.
                    if os.path.isdir(task_path):
                        shutil.rmtree(task_path)
                    continue

            try:
                # "crawl_openml_task" modifies datasets in-place.
                crawl_openml_task(
                    datasets, task_id=task['task_id'], save_dir=task_path, data_pipeline=data_pipeline, data_params=data_params,
                    context=context, random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
                    runtime_environment=runtime_environment, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver,
                    compute_digest=compute_digest, strict_digest=strict_digest,
                )
                current_tasks += 1
                logger.info("Crawled task '%(task_id)s' into '%(task_path)s'.", {'task_path': task_path, 'task_id': task['task_id']})
            except Exception:
                has_errored = True
                # Cleanup.
                if os.path.isdir(task_path):
                    shutil.rmtree(task_path)
                logger.exception("Error crawling task '%(task_id)s'.", {'task_id': task['task_id']})

    return has_errored


def crawl_openml_handler(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = pipeline_module.get_pipeline
    if dataset_resolver is None:
        dataset_resolver = dataset_module.get_dataset
    if problem_resolver is None:
        problem_resolver = problem_module.get_problem

    context = metadata_base.Context[arguments.context]
    compute_digest = dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)]
    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    task_types = [problem_module.OpenMLTaskType[task_type] for task_type in arguments.task_types]
    if utils.has_duplicates(task_types):
        raise exceptions.InvalidArgumentValueError("Same task type listed multiple times.")

    assert task_types

    inputs_config = runtime._get_inputs_config_from_arguments(
        arguments=arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
    )

    assert inputs_config.data_pipeline

    has_errored = crawl_openml(
        save_dir=arguments.save_dir, task_types=task_types,
        data_pipeline=inputs_config.data_pipeline,
        data_params=inputs_config.data_params,
        context=context,
        random_seed=inputs_config.data_random_seed,
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
        max_tasks=arguments.max_tasks,
        ignore_tasks=arguments.ignore_tasks or [],
        ignore_datasets=arguments.ignore_datasets or [],
        dataset_resolver=dataset_resolver,
        problem_resolver=problem_resolver,
        compute_digest=compute_digest,
        strict_digest=getattr(arguments, 'strict_digest', False),
    )

    if has_errored:
        sys.exit(1)
