import argparse
import importlib
import uuid

import faulthandler
import logging
import sys
import typing

from d3m import exceptions, index, runtime, utils, __version__
from d3m.container import dataset as dataset_module
from d3m.contrib.openml import crawler as openml_crawler
from d3m.metadata import base as metadata_base, pipeline as pipeline_module, pipeline_run, problem as problem_module

logger = logging.getLogger(__name__)


def pipeline_run_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser,
) -> None:
    # Call a handler for the command.
    arguments.pipeline_run_handler(
        arguments,
    )


def pipeline_run_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    subparsers = parser.add_subparsers(dest='pipeline_run_command', title='commands')
    subparsers.required = True

    validate_parser = subparsers.add_parser(
        'validate', help="validate pipeline runs",
        description="Validate pipeline runs for use in metalearning database.",
    )

    if 'list' not in skip_arguments:
        validate_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path of pipeline run being validated",
        )
    if 'continue' not in skip_arguments:
        validate_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after pipeline run validation error",
        )
    if 'pipeline_runs' not in skip_arguments:
        validate_parser.add_argument(
            'pipeline_runs', metavar='PIPELINE_RUN', nargs='+',
            help="path to a pipeline run",
        )
    validate_parser.set_defaults(pipeline_run_handler=pipeline_run.pipeline_run_handler)


def dataset_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    dataset_resolver: typing.Callable = None,
) -> None:
    # Call a handler for the command.
    arguments.dataset_handler(
        arguments,
        dataset_resolver=dataset_resolver,
    )


def dataset_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    subparsers = parser.add_subparsers(dest='dataset_command', title='commands')
    subparsers.required = True

    describe_parser = subparsers.add_parser(
        'describe', help="generate JSON description of datasets",
        description="Generates JSON descriptions of datasets.",
    )
    convert_parser = subparsers.add_parser(
        'convert', help="convert datasets",
        description="Converts one dataset to another.",
    )
    validate_parser = subparsers.add_parser(
        'validate', help="validate datasets",
        description="Validate dataset descriptions for use in metalearning database.",
    )

    if 'list' not in skip_arguments:
        describe_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path or URI of dataset being described",
        )
    if 'indent' not in skip_arguments:
        describe_parser.add_argument(
            '-i', '--indent', type=int, default=2, action='store',
            help="indent JSON by this much, 0 disables indentation, default 2",
        )
    if 'sort_keys' not in skip_arguments:
        describe_parser.add_argument(
            '-s', '--sort-keys', default=False, action='store_true',
            help="sort keys in JSON"
        )
    if 'print' not in skip_arguments:
        describe_parser.add_argument(
            '-p', '--print', default=False, action='store_true',
            help="pretty print dataset contents instead of printing JSON description",
        )
    if 'metadata' not in skip_arguments:
        describe_parser.add_argument(
            '-m', '--metadata', default=False, action='store_true',
            help="pretty print dataset metadata instead of printing JSON description",
        )
    if 'lazy' not in skip_arguments:
        describe_parser.add_argument(
            '-L', '--lazy', default=False, action='store_true',
            help="load dataset lazily",
        )
    if 'time' not in skip_arguments:
        describe_parser.add_argument(
            '-t', '--time', default=False, action='store_true',
            help="time dataset loading instead of printing JSON description",
        )
    if 'continue' not in skip_arguments:
        describe_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after dataset loading error",
        )
    if 'output' not in skip_arguments:
        describe_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save output to a file, default stdout",
        )
    if 'datasets' not in skip_arguments:
        describe_parser.add_argument(
            'datasets', metavar='DATASET', nargs='*',
            help="path or URI of a dataset",
        )
    describe_parser.set_defaults(dataset_handler=dataset_module.describe_handler)

    if 'input_uri' not in skip_arguments:
        convert_parser.add_argument(
            '-i', '--input', dest='input_uri',
            help="input path or URI of a dataset",
        )
    if 'output_uri' not in skip_arguments:
        convert_parser.add_argument(
            '-o', '--output', dest='output_uri',
            help="output path or URI of a dataset",
        )
    if 'preserve_metadata' not in skip_arguments:
        convert_parser.add_argument(
            '--no-metadata', default=True, action='store_false', dest='preserve_metadata',
            help="do not preserve metadata",
        )
    convert_parser.set_defaults(dataset_handler=dataset_module.convert_handler)

    if 'list' not in skip_arguments:
        validate_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path or URI of dataset being validated",
        )
    if 'continue' not in skip_arguments:
        validate_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after dataset validation error",
        )
    if 'datasets' not in skip_arguments:
        validate_parser.add_argument(
            'datasets', metavar='DATASET', nargs='+',
            help="path to a dataset description",
        )
    validate_parser.set_defaults(dataset_handler=pipeline_run.dataset_handler)


def problem_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    problem_resolver: typing.Callable = None,
) -> None:
    # Call a handler for the command.
    arguments.problem_handler(
        arguments,
        problem_resolver=problem_resolver,
    )


def problem_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    subparsers = parser.add_subparsers(dest='problem_command', title='commands')
    subparsers.required = True

    describe_parser = subparsers.add_parser(
        'describe', help="generate JSON description of problems",
        description="Generates JSON descriptions of problems.",
    )
    convert_parser = subparsers.add_parser(
        'convert', help="convert problems",
        description="Converts one problem to another.",
    )
    validate_parser = subparsers.add_parser(
        'validate', help="validate problems",
        description="Validate problem descriptions for use in metalearning database.",
    )

    if 'list' not in skip_arguments:
        describe_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path or URI of problem being described",
        )
    if 'indent' not in skip_arguments:
        describe_parser.add_argument(
            '-i', '--indent', type=int, default=2, action='store',
            help="indent JSON by this much, 0 disables indentation, default 2",
        )
    if 'sort_keys' not in skip_arguments:
        describe_parser.add_argument(
            '-s', '--sort-keys', default=False, action='store_true',
            help="sort keys in JSON"
        )
    if 'print' not in skip_arguments:
        describe_parser.add_argument(
            '-p', '--print', default=False, action='store_true',
            help="pretty print problem description instead of printing JSON",
        )
    if 'continue' not in skip_arguments:
        describe_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after problem parsing error",
        )
    if 'output' not in skip_arguments:
        describe_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save output to a file, default stdout",
        )
    if 'no_print' not in skip_arguments:
        describe_parser.add_argument(
            '--no-print', default=False, action='store_true',
            help="do not print JSON",
        )
    if 'problems' not in skip_arguments:
        describe_parser.add_argument(
            'problems', metavar='PROBLEM', nargs='+',
            help="path or URI to a problem description",
        )
    describe_parser.set_defaults(problem_handler=problem_module.describe_handler)

    if 'input_uri' not in skip_arguments:
        convert_parser.add_argument(
            '-i', '--input', dest='input_uri',
            help="input path or URI of a problem",
        )
    if 'output_uri' not in skip_arguments:
        convert_parser.add_argument(
            '-o', '--output', dest='output_uri',
            help="output path or URI of a problem",
        )
    convert_parser.set_defaults(problem_handler=problem_module.convert_handler)

    if 'list' not in skip_arguments:
        validate_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path or URI of problem being validated",
        )
    if 'continue' not in skip_arguments:
        validate_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after problem validation error",
        )
    if 'problems' not in skip_arguments:
        validate_parser.add_argument(
            'problems', metavar='PROBLEM', nargs='+',
            help="path to a problem description",
        )
    validate_parser.set_defaults(problem_handler=pipeline_run.problem_handler)


def primitive_handler(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Call a handler for the command.
    arguments.primitive_handler(arguments)


def primitive_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    subparsers = parser.add_subparsers(dest='primitive_command', title='commands')
    subparsers.required = True

    search_parser = subparsers.add_parser(
        'search', help="search locally available primitives",
        description="Searches locally available primitives. Lists registered Python paths for primitives installed on the system.",
    )
    describe_parser = subparsers.add_parser(
        'describe', help="generate JSON description of primitives",
        description="Generates JSON descriptions of primitives.",
    )
    download_parser = subparsers.add_parser(
        'download', help="download files for primitives' volumes",
        description="Downloads static files needed by primitives.",
    )
    validate_parser = subparsers.add_parser(
        'validate', help="validate primitive descriptions",
        description="Validate primitive descriptions for use in metalearning database.",
    )

    if 'prefix' not in skip_arguments:
        search_parser.add_argument(
            '-p', '--prefix', action='store',
            help="primitive path prefix to limit search results to",
        )
    search_parser.set_defaults(primitive_handler=index.search_handler)

    if 'list' not in skip_arguments:
        describe_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path or ID of primitive being described",
        )
    if 'indent' not in skip_arguments:
        describe_parser.add_argument(
            '-i', '--indent', type=int, default=2, action='store',
            help="indent JSON by this much, 0 disables indentation, default 2",
        )
    if 'sort_keys' not in skip_arguments:
        describe_parser.add_argument(
            '-s', '--sort-keys', default=False, action='store_true',
            help="sort keys in JSON"
        )
    if 'print' not in skip_arguments:
        describe_parser.add_argument(
            '-p', '--print', default=False, action='store_true',
            help="pretty print primitive description instead of printing JSON",
        )
    if 'continue' not in skip_arguments:
        describe_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after primitive loading error",
        )
    if 'output' not in skip_arguments:
        describe_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save output to a file, default stdout",
        )
    if 'primitives' not in skip_arguments:
        describe_parser.add_argument(
            'primitives', metavar='PRIMITIVE', nargs='+',
            help="primitive path od primitive ID",
        )
    describe_parser.set_defaults(primitive_handler=index.describe_handler)

    if 'output' not in skip_arguments:
        download_parser.add_argument(
            '-o', '--output', default=index.DEFAULT_OUTPUT, action='store',
            help="path of a directory to download to, default current directory",
        )
    if 'redownload' not in skip_arguments:
        download_parser.add_argument(
            '-r', '--redownload', default=False, action='store_true',
            help="redownload files again, even if they already exist",
        )
    if 'prefix' not in skip_arguments:
        download_parser.add_argument(
            '-p', '--prefix', action='store',
            help="primitive path prefix to limit download to",
        )
    download_parser.set_defaults(primitive_handler=index.download_handler)

    if 'list' not in skip_arguments:
        validate_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path of primitive description being validated",
        )
    if 'continue' not in skip_arguments:
        validate_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after primitive description validation error",
        )
    if 'primitives' not in skip_arguments:
        validate_parser.add_argument(
            'primitives', metavar='PRIMITIVE', nargs='+',
            help="path to a primitive description",
        )
    validate_parser.set_defaults(primitive_handler=pipeline_run.primitive_handler)


def pipeline_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    resolver_class: typing.Type[pipeline_module.Resolver] = None,
    no_resolver_class: typing.Type[pipeline_module.Resolver] = None,
    pipeline_class: typing.Type[pipeline_module.Pipeline] = None,
) -> None:
    # Call a handler for the command.
    arguments.pipeline_handler(
        arguments,
        resolver_class=resolver_class,
        no_resolver_class=no_resolver_class,
        pipeline_class=pipeline_class,
    )


def pipeline_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    subparsers = parser.add_subparsers(dest='pipeline_command', title='commands')
    subparsers.required = True

    describe_parser = subparsers.add_parser(
        'describe', help="generate JSON description of pipelines",
        description="Generates JSON descriptions of pipelines.",
    )
    validate_parser = subparsers.add_parser(
        'validate', help="validate pipelines",
        description="Validate pipeline descriptions for use in metalearning database.",
    )

    if 'no_resolving' not in skip_arguments:
        describe_parser.add_argument(
            '-n', '--no-resolving', default=False, action='store_true',
            help="do not resolve primitives and pipelines, this prevents checking to be fully done though",
        )
    if 'check' not in skip_arguments:
        describe_parser.add_argument(
            '-C', '--no-check', default=True, action='store_false', dest='check',
            help="do not check a pipeline, just parse it",
        )
    if 'allow_placeholders' not in skip_arguments:
        describe_parser.add_argument(
            '-a', '--allow-placeholders', default=False, action='store_true',
            help="allow placeholders in a pipeline",
        )
    if 'standard_pipeline' not in skip_arguments:
        describe_parser.add_argument(
            '-t', '--not-standard-pipeline', default=True, action='store_false', dest='standard_pipeline',
            help="allow a pipeline to not have standard inputs and outputs",
        )
    if 'list' not in skip_arguments:
        describe_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path of pipeline being described",
        )
    if 'indent' not in skip_arguments:
        describe_parser.add_argument(
            '-i', '--indent', type=int, default=2, action='store',
            help="indent JSON by this much, 0 disables indentation, default 2",
        )
    if 'sort_keys' not in skip_arguments:
        describe_parser.add_argument(
            '-s', '--sort-keys', default=False, action='store_true',
            help="sort keys in JSON"
        )
    if 'print' not in skip_arguments:
        describe_parser.add_argument(
            '-p', '--print', default=False, action='store_true',
            help="pretty print pipeline description instead of printing JSON",
        )
    if 'continue' not in skip_arguments:
        describe_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after pipeline parsing error",
        )
    if 'set_source_name' not in skip_arguments:
        describe_parser.add_argument(
            '--set-source-name', action='store',
            help="set pipeline's source name",
        )
    if 'output' not in skip_arguments:
        describe_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save output to a file, default stdout",
        )
    if 'pipelines' not in skip_arguments:
        describe_parser.add_argument(
            'pipelines', metavar='PIPELINE', nargs='+',
            help="path to a pipeline (.json, .yml, or .yaml)",
        )
    describe_parser.set_defaults(pipeline_handler=pipeline_module.describe_handler)

    if 'list' not in skip_arguments:
        validate_parser.add_argument(
            '-l', '--list', default=False, action='store_true',
            help="print path of pipeline being validated",
        )
    if 'continue' not in skip_arguments:
        validate_parser.add_argument(
            '-c', '--continue', default=False, action='store_true',
            help="continue after pipeline validation error",
        )
    if 'pipelines' not in skip_arguments:
        validate_parser.add_argument(
            'pipelines', metavar='PIPELINE', nargs='*',
            help="path to a pipeline (.json, .yml, or .yaml)",
        )
    validate_parser.set_defaults(pipeline_handler=pipeline_run.pipeline_handler)


def openml_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    pipeline_resolver: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    # Call a handler for the command.
    arguments.openml_handler(
        arguments,
        pipeline_resolver=pipeline_resolver,
        dataset_resolver=dataset_resolver,
        problem_resolver=problem_resolver,
    )


def openml_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    subparsers = parser.add_subparsers(dest='openml_command', title='commands')
    subparsers.required = True

    crawl_parser = subparsers.add_parser(
        'crawl', help="crawl OpenML tasks",
        description="Crawl OpenML tasks and corresponding datasets and convert them to D3M datasets and problems.",
    )

    if 'save_dir' not in skip_arguments:
        crawl_parser.add_argument(
            '-s', '--save', action='store', dest='save_dir', required=True,
            help="directory where to save datasets and problems",
        )
    if 'task_types' not in skip_arguments:
        crawl_parser.add_argument(
            '-t', '--task-type', choices=utils.EnumArgProxy(problem_module.OpenMLTaskType),
            action='append', dest='task_types', required=True,
            help="task type to crawl, can be specified multiple times",
        )
    if 'max_tasks' not in skip_arguments:
        crawl_parser.add_argument(
            '--max-tasks', type=int, action='store', default=5,
            help="maximum number of tasks to crawl, no limit if 0, default 5",
        )
    if 'ignore_tasks' not in skip_arguments:
        crawl_parser.add_argument(
            '--ignore_tasks', type=int, action='append', dest='ignore_tasks',
            help="tasks to ignore/skip",
        )
    if 'ignore_datasets' not in skip_arguments:
        crawl_parser.add_argument(
            '--ignore_datasets', type=int, action='append', dest='ignore_datasets',
            help="datasets to ignore/skip",
        )
    _common_runtime_configure_parser(crawl_parser, skip_arguments=skip_arguments)
    _data_preprocessing_configure_parser(crawl_parser, skip_arguments=skip_arguments)
    crawl_parser.set_defaults(openml_handler=openml_crawler.crawl_openml_handler)


def runtime_handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    pipeline_resolver: typing.Callable = None, pipeline_run_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
) -> None:
    # Dynamically fetch which subparser was used.
    subparser = parser._subparsers._group_actions[0].choices[arguments.runtime_command]  # type: ignore

    # TODO: These arguments are required, but this is not visible from the usage line. These arguments are marked as optional there.
    if getattr(arguments, 'input_run', None) is None:
        required_without_pipeline_run = {
            'fit': [
                ('-i/--input', 'inputs'), ('-p/--pipeline', 'pipeline'),
            ],
            'produce': [
                ('-t/--test-input', 'test_inputs'),
            ],
            'fit-produce': [
                ('-p/--pipeline', 'pipeline'),
            ],
            'fit-score': [
                ('-p/--pipeline', 'pipeline'),
            ],
            'evaluate': [
                ('-i/--input', 'inputs'), ('-p/--pipeline', 'pipeline'), ('-d/--data-pipeline', 'data_pipeline'),
            ],
            'prepare-data': [
                ('-i/--input', 'inputs'), ('-d/--data-pipeline', 'data_pipeline'), ('-s/--save-path', 'save_dir')
            ],
        }.get(arguments.runtime_command, [])
        if any(getattr(arguments, dest, None) is None for (name, dest) in required_without_pipeline_run):
            subparser.error(
                '{command} requires the following arguments if run without -u/--input-run: {required_arguments}'.format(
                    command=arguments.runtime_command,
                    required_arguments=', '.join(
                        name for (name, dest) in required_without_pipeline_run if getattr(arguments, dest, None) is None
                    ),
                )
            )

        allowed_only_with_data_pipeline = {command: [
            ('-y/--data-param', 'data_params', None), ('--data-split-file', 'data_split_file', None), ('-I/--full-input', 'full_inputs', None), ('--data-random-seed', 'data_random_seed', 0)
        ] for command in ['fit', 'produce', 'score', 'fit-produce', 'fit-score', 'evaluate', 'prepare-data']}.get(arguments.runtime_command, [])
        if getattr(arguments, 'data_pipeline', None) is None and any(getattr(arguments, dest, None) not in [default, None] for (name, dest, default) in allowed_only_with_data_pipeline):
            subparser.error(
                'The following arguments can be used only with -d/--data-pipeline: {arguments}'.format(
                    arguments=', '.join(
                        name for (name, dest, default) in allowed_only_with_data_pipeline if getattr(arguments, dest, None) not in [default, None]
                    ),
                )
            )

        if arguments.runtime_command != 'evaluate':
            required_for_data_pipeline = {
                'score': [
                    ('-I/--full-input', 'full_inputs', None),
                ],
                'fit-produce': [
                    ('-I/--full-input', 'full_inputs', None),
                ],
                'fit-score': [
                    ('-I/--full-input', 'full_inputs', None),
                ],
            }.get(arguments.runtime_command, [])
            mutually_exclusive_with_data_pipeline = {
                'score': [
                    ('-t/--test-input', 'test_inputs', None), ('-a/--score-input', 'score_inputs', None),
                ],
                'fit-produce': [
                    ('-i/--input', 'inputs', None), ('-t/--test-input', 'test_inputs', None),
                ],
                'fit-score': [
                    ('-i/--input', 'inputs', None), ('-t/--test-input', 'test_inputs', None), ('-a/--score-input', 'score_inputs', None),
                ],
            }.get(arguments.runtime_command, [])

            if getattr(arguments, 'data_pipeline', None) is not None:
                if any(getattr(arguments, dest, None) not in [default, None] for (name, dest, default) in mutually_exclusive_with_data_pipeline):
                    subparser.error(
                        '{command} with -d/--data-pipeline cannot have any of the following arguments: {arguments}'.format(
                            command=arguments.runtime_command,
                            arguments=', '.join(
                                name for (name, dest, default) in mutually_exclusive_with_data_pipeline if getattr(arguments, dest, None) not in [default, None]
                            ),
                        )
                    )
                if any(getattr(arguments, dest, None) in [default, None] for (name, dest, default) in required_for_data_pipeline):
                    subparser.error(
                        '{command} with -d/--data-pipeline must have the following arguments: {arguments}'.format(
                            command=arguments.runtime_command,
                            arguments=', '.join(
                                name for (name, dest, default) in required_for_data_pipeline if getattr(arguments, dest, None) in [default, None]
                            ),
                        )
                    )
            else:
                if any(getattr(arguments, dest, None) in [default, None] for (name, dest, default) in mutually_exclusive_with_data_pipeline):
                    subparser.error(
                        '{command} requires either -d/--data-pipeline or the following arguments: {arguments}'.format(
                            command=arguments.runtime_command,
                            arguments=', '.join(
                                name for (name, dest, default) in mutually_exclusive_with_data_pipeline if getattr(arguments, dest, None) in [default, None]
                            ),
                        )
                    )

    else:
        mutually_exclusive_with_pipeline_run = [
            ('-i/--input', 'inputs', None), ('-t/--test-input', 'test_inputs', None), ('-a/--score-input', 'score_inputs', None), ('-I/--full-input', 'full_inputs', None),
            ('-r/--problem', 'problem', None), ('-p/--pipeline', 'pipeline', None), ('-d/--data-pipeline', 'data_pipeline', None),
            ('-n/--random-seed', 'random_seed', 0), ('-e/--metric', 'metrics', None), ('-Y/--scoring-param', 'scoring_params', None),
            ('--scoring-random-seed', 'scoring_random_seed', 0), ('-n/--scoring-pipeline', 'scoring_pipeline', runtime.DEFAULT_SCORING_PIPELINE_PATH),
            ('-y/--data-param', 'data_params', None), ('--data-split-file', 'data_split_file', None), ('--data-random-seed', 'data_random_seed', 0),
            ('--not-standard-pipeline', 'standard_pipeline', True),
        ]
        if any(getattr(arguments, dest, None) not in [default, None] for (name, dest, default) in mutually_exclusive_with_pipeline_run):
            subparser.error(
                '-u/--input-run cannot be used with the following arguments: {mutually_exclusive}'.format(
                    mutually_exclusive=', '.join(
                        name for (name, dest, default) in mutually_exclusive_with_pipeline_run if getattr(arguments, dest, None) not in [default, None]
                    ),
                )
            )

    if not getattr(arguments, 'standard_pipeline', True) and getattr(arguments, 'output', None) is not None:
        subparser.error("you cannot save predictions for a non-standard pipeline")

    if getattr(arguments, 'data_pipeline', None) and getattr(arguments, 'data_pipeline_run', None):
        subparser.error("-d/--data-pipeline and --data-pipeline-run cannot be used at the same time")

    # Call a handler for the command.
    arguments.runtime_handler(
        arguments,
        pipeline_resolver=pipeline_resolver,
        pipeline_run_parser=pipeline_run_parser,
        dataset_resolver=dataset_resolver,
        problem_resolver=problem_resolver,
    )


def _data_preprocessing_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    if 'data_pipeline' not in skip_arguments:
        parser.add_argument(
            '-d', '--data-pipeline', action='store',
            help="path to a data preparation pipeline file (.json, .yml, or .yaml) or pipeline ID",
        )
    if 'data_params' not in skip_arguments:
        parser.add_argument(
            '-y', '--data-param', nargs=2, action='append', metavar=('NAME', 'VALUE'), dest='data_params',
            help="hyper-parameter name and its value for data preparation pipeline, can be specified multiple times, value should be JSON-serialized",
        )
    if 'data_split_file' not in skip_arguments:
        parser.add_argument(
            '--data-split-file', action='store',
            help="reads the split file and populates \"primary_index_values\" hyper-parameter for data preparation pipeline with "
                 "values from the \"d3mIndex\" column corresponding to the test data, use \"-\" for stdin",
        )
    if 'data_random_seed' not in skip_arguments:
        parser.add_argument(
            '--data-random-seed', type=int, action='store', default=0,
            help="random seed to use for data preparation",
        )


def _common_runtime_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    if 'context' not in skip_arguments:
        parser.add_argument(
            '-x', '--context', choices=utils.EnumArgProxy(metadata_base.Context), default=metadata_base.Context.TESTING.name, action='store',
            help="in which context to run pipelines, default is TESTING",
        )
    if 'volumes_dir' not in skip_arguments:
        parser.add_argument(
            '-v', '--volumes', action='store', dest='volumes_dir',
            help="path to a directory with static files required by primitives, in the standard directory structure (as obtained running \"python3 -m d3m primitive download\")",
        )
    if 'datasets_dir' not in skip_arguments:
        parser.add_argument(
            '-D', '--datasets', action='store', dest='datasets_dir',
            help="path to a directory with datasets (and problem descriptions) to resolve IDs in pipeline run files",
        )
    if 'scratch_dir' not in skip_arguments:
        parser.add_argument(
            '--scratch', action='store', dest='scratch_dir',
            help="path to a directory to store any temporary files needed during execution",
        )
    if 'worker_id' not in skip_arguments:
        parser.add_argument(
            '--worker-id', action='store',
            help="globally unique identifier for the machine on which the runtime is running, has priority over D3M_WORKER_ID environment variable",
        )


def runtime_configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    if 'random_seed' not in skip_arguments:
        parser.add_argument(
            '-n', '--random-seed', type=int, default=0, action='store', metavar='SEED',
            help="random seed to use",
        )
    _common_runtime_configure_parser(parser, skip_arguments=skip_arguments)

    subparsers = parser.add_subparsers(dest='runtime_command', title='commands')
    subparsers.required = True

    fit_parser = subparsers.add_parser(
        'fit', help="fit a pipeline",
        description="Fits a pipeline on train data, resulting in a fitted pipeline. Outputs also produced predictions during fitting on train data.",
    )
    produce_parser = subparsers.add_parser(
        'produce', help="produce using a fitted pipeline",
        description="Produce predictions on test data given a fitted pipeline.",
    )
    score_parser = subparsers.add_parser(
        'score', help="produce using a fitted pipeline and score results",
        description="Produce predictions on test data given a fitted pipeline and compute scores.",
    )
    fit_produce_parser = subparsers.add_parser(
        'fit-produce', help="fit a pipeline and then produce using it",
        description="Fit a pipeline on train data and produce predictions on test data.",
    )
    fit_score_parser = subparsers.add_parser(
        'fit-score', help="fit a pipeline, produce using it and score results",
        description="Fit a pipeline on train data, then produce predictions on test data and compute scores.",
    )
    score_predictions_parser = subparsers.add_parser(
        'score-predictions', help="score a predictions file",
        description="Compute scores given a file with predictions.",
    )
    evaluate_parser = subparsers.add_parser(
        'evaluate', help="evaluate a pipeline",
        description="Run pipeline multiple times using an evaluation approach and compute scores for each run.",
    )
    prepare_data_parser = subparsers.add_parser(
        'prepare-data', help="prepare data and save the results",
        description="Apply a data preparation pipeline to the input data and save the results.",
    )

    if 'pipeline' not in skip_arguments:
        fit_parser.add_argument(
            '-p', '--pipeline', action='store',
            help="path to a pipeline file (.json, .yml, or .yaml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        fit_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train data, or an input full data when used in combination with the data preparation pipeline",
        )
    if 'input_run' not in skip_arguments:
        fit_parser.add_argument(
            '-u', '--input-run', type=utils.FileType('r', encoding='utf8'), action='store',
            help="path to a pipeline run file with configuration, use \"-\" for stdin",
        )
    if 'data_pipeline_run' not in skip_arguments:
        fit_parser.add_argument(
            '--data-pipeline-run', action='store',
            help="path to a pickled data preparation pipeline run",
        )
    if 'fold_group_uuid' not in skip_arguments:
        fit_parser.add_argument(
            '--fold-group-uuid', action='store', type=uuid.UUID,
            help="fold group UUID to be set in the pipeline run",
        )
    if 'fold_index' not in skip_arguments:
        fit_parser.add_argument(
            '--fold-index', action='store', type=int, default=0,
            help="fold index to be set in the pipeline run",
        )
    if 'save' not in skip_arguments:
        fit_parser.add_argument(
            '-s', '--save', type=utils.FileType('wb'), action='store',
            help="save fitted pipeline to a file, use \"-\" for stdout",
        )
    if 'output' not in skip_arguments:
        fit_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions during fitting to a file, use \"-\" for stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_parser.add_argument(
            '-O', '--output-run', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file, use \"-\" for stdout",
        )
    if 'standard_pipeline' not in skip_arguments:
        fit_parser.add_argument(
            '--not-standard-pipeline', default=True, action='store_false', dest='standard_pipeline',
            help="allow a pipeline to not have standard inputs and outputs",
        )
    if 'expose_produced_outputs_dir' not in skip_arguments:
        fit_parser.add_argument(
            '-E', '--expose-produced-outputs', action='store', dest='expose_produced_outputs_dir',
            help="save to a directory produced outputs of all primitives from pipeline's fit run",
        )
    _data_preprocessing_configure_parser(fit_parser, skip_arguments=skip_arguments)
    fit_parser.set_defaults(runtime_handler=runtime.fit_handler)

    if 'fitted_pipeline' not in skip_arguments:
        produce_parser.add_argument(
            '-f', '--fitted-pipeline', type=utils.FileType('rb'), action='store', required=True,
            help="path to a saved fitted pipeline, use \"-\" for stdin",
        )
    if 'test_inputs' not in skip_arguments:
        produce_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test data, or an input full data when used in combination with the data preparation pipeline",
        )
    if 'input_run' not in skip_arguments:
        produce_parser.add_argument(
            '-u', '--input-run', type=utils.FileType('r', encoding='utf8'), action='store',
            help="path to a pipeline run file with configuration, use \"-\" for stdin",
        )
    if 'data_pipeline_run' not in skip_arguments:
        produce_parser.add_argument(
            '--data-pipeline-run', action='store',
            help="path to a pickled data preparation pipeline run",
        )
    if 'fold_group_uuid' not in skip_arguments:
        produce_parser.add_argument(
            '--fold-group-uuid', action='store', type=uuid.UUID,
            help="fold group UUID to be set in the pipeline run",
        )
    if 'fold_index' not in skip_arguments:
        produce_parser.add_argument(
            '--fold-index', action='store', type=int, default=0,
            help="fold index to be set in the pipeline run",
        )
    if 'output' not in skip_arguments:
        produce_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file, use \"-\" for stdout",
        )
    if 'output_run' not in skip_arguments:
        produce_parser.add_argument(
            '-O', '--output-run', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file, use \"-\" for stdout",
        )
    if 'expose_produced_outputs_dir' not in skip_arguments:
        produce_parser.add_argument(
            '-E', '--expose-produced-outputs', action='store', dest='expose_produced_outputs_dir',
            help="save to a directory produced outputs of all primitives from pipeline's produce run",
        )
    _data_preprocessing_configure_parser(produce_parser, skip_arguments=skip_arguments)
    produce_parser.set_defaults(runtime_handler=runtime.produce_handler)

    if 'fitted_pipeline' not in skip_arguments:
        score_parser.add_argument(
            '-f', '--fitted-pipeline', type=utils.FileType('rb'), action='store', required=True,
            help="path to a saved fitted pipeline, use \"-\" for stdin",
        )
    if 'scoring_pipeline' not in skip_arguments:
        score_parser.add_argument(
            '-n', '--scoring-pipeline', default=runtime.DEFAULT_SCORING_PIPELINE_PATH, action='store',
            help="path to a scoring pipeline file (.json, .yml, or .yaml) or pipeline ID, default is standard scoring pipeline",
        )
    if 'test_inputs' not in skip_arguments:
        score_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test data",
        )
    if 'score_inputs' not in skip_arguments:
        score_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs',
            help="path or URI of an input score data",
        )
    if 'input_run' not in skip_arguments:
        score_parser.add_argument(
            '-u', '--input-run', type=utils.FileType('r', encoding='utf8'), action='store',
            help="path to a pipeline run file with configuration, use \"-\" for stdin",
        )
    if 'data_pipeline_run' not in skip_arguments:
        score_parser.add_argument(
            '--data-pipeline-run', action='store',
            help="path to a pickled data preparation pipeline run",
        )
    if 'fold_group_uuid' not in skip_arguments:
        score_parser.add_argument(
            '--fold-group-uuid', action='store', type=uuid.UUID,
            help="fold group UUID to be set in the pipeline run",
        )
    if 'fold_index' not in skip_arguments:
        score_parser.add_argument(
            '--fold-index', action='store', type=int, default=0,
            help="fold index to be set in the pipeline run",
        )
    if 'metrics' not in skip_arguments:
        score_parser.add_argument(
            '-e', '--metric', choices=utils.EnumArgProxy(problem_module.PerformanceMetric),
            action='append', metavar='METRIC', dest='metrics',
            help="metric to use, can be specified multiple times, default from problem description",
        )
    if 'scoring_params' not in skip_arguments:
        score_parser.add_argument(
            '-Y', '--scoring-param', nargs=2, action='append', metavar=('NAME', 'VALUE'), dest='scoring_params',
            help="hyper-parameter name and its value for scoring pipeline, can be specified multiple times, value should be JSON-serialized",
        )
    if 'output' not in skip_arguments:
        score_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file, use \"-\" for stdout",
        )
    if 'scores' not in skip_arguments:
        score_parser.add_argument(
            '-c', '--scores', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        score_parser.add_argument(
            '-O', '--output-run', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file, use \"-\" for stdout",
        )
    if 'expose_produced_outputs_dir' not in skip_arguments:
        score_parser.add_argument(
            '-E', '--expose-produced-outputs', action='store', dest='expose_produced_outputs_dir',
            help="save to a directory produced outputs of all primitives from pipeline's produce run",
        )
    if 'full_inputs' not in skip_arguments:
        score_parser.add_argument(
            '-I', '--full-input', action='append', metavar='INPUT', dest='full_inputs',
            help="path or URI of an input full data, used in combination with the data preparation pipeline",
        )
    _data_preprocessing_configure_parser(score_parser, skip_arguments=skip_arguments)
    score_parser.set_defaults(runtime_handler=runtime.score_handler)

    if 'pipeline' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-p', '--pipeline', action='store',
            help="path to a pipeline file (.json, .yml, or .yaml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train data",
        )
    if 'test_inputs' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test data",
        )
    if 'input_run' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-u', '--input-run', type=utils.FileType('r', encoding='utf8'), action='store',
            help="path to a pipeline run file with configuration, use \"-\" for stdin",
        )
    if 'data_pipeline_run' not in skip_arguments:
        fit_produce_parser.add_argument(
            '--data-pipeline-run', action='store',
            help="path to a pickled data preparation pipeline run",
        )
    if 'fold_group_uuid' not in skip_arguments:
        fit_produce_parser.add_argument(
            '--fold-group-uuid', action='store', type=uuid.UUID,
            help="fold group UUID to be set in the pipeline run",
        )
    if 'fold_index' not in skip_arguments:
        fit_produce_parser.add_argument(
            '--fold-index', action='store', type=int, default=0,
            help="fold index to be set in the pipeline run",
        )
    if 'save' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-s', '--save', type=utils.FileType('wb'), action='store',
            help="save fitted pipeline to a file, use \"-\" for stdout",
        )
    if 'output' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file, use \"-\" for stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-O', '--output-run', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file, use \"-\" for stdout",
        )
    if 'standard_pipeline' not in skip_arguments:
        fit_produce_parser.add_argument(
            '--not-standard-pipeline', default=True, action='store_false', dest='standard_pipeline',
            help="allow a pipeline to not have standard inputs and outputs",
        )
    if 'expose_produced_outputs_dir' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-E', '--expose-produced-outputs', action='store', dest='expose_produced_outputs_dir',
            help="save to a directory produced outputs of all primitives from pipeline's produce run",
        )
    if 'full_inputs' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-I', '--full-input', action='append', metavar='INPUT', dest='full_inputs',
            help="path or URI of an input full data, used in combination with the data preparation pipeline",
        )
    _data_preprocessing_configure_parser(fit_produce_parser, skip_arguments=skip_arguments)
    fit_produce_parser.set_defaults(runtime_handler=runtime.fit_produce_handler)

    if 'pipeline' not in skip_arguments:
        fit_score_parser.add_argument(
            '-p', '--pipeline', action='store',
            help="path to a pipeline file (.json, .yml, or .yaml) or pipeline ID",
        )
    if 'scoring_pipeline' not in skip_arguments:
        fit_score_parser.add_argument(
            '-n', '--scoring-pipeline', default=runtime.DEFAULT_SCORING_PIPELINE_PATH, action='store',
            help="path to a scoring pipeline file (.json, .yml, or .yaml) or pipeline ID, default is standard scoring pipeline",
        )
    if 'problem' not in skip_arguments:
        fit_score_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train data",
        )
    if 'test_inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test data",
        )
    if 'score_inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs',
            help="path or URI of an input score data",
        )
    if 'input_run' not in skip_arguments:
        fit_score_parser.add_argument(
            '-u', '--input-run', type=utils.FileType('r', encoding='utf8'), action='store',
            help="path to a pipeline run file with configuration, use \"-\" for stdin",
        )
    if 'data_pipeline_run' not in skip_arguments:
        fit_score_parser.add_argument(
            '--data-pipeline-run', action='store',
            help="path to a pickled data preparation pipeline run",
        )
    if 'fold_group_uuid' not in skip_arguments:
        fit_score_parser.add_argument(
            '--fold-group-uuid', action='store', type=uuid.UUID,
            help="fold group UUID to be set in the pipeline run",
        )
    if 'fold_index' not in skip_arguments:
        fit_score_parser.add_argument(
            '--fold-index', action='store', type=int, default=0,
            help="fold index to be set in the pipeline run",
        )
    if 'metrics' not in skip_arguments:
        fit_score_parser.add_argument(
            '-e', '--metric', choices=utils.EnumArgProxy(problem_module.PerformanceMetric),
            action='append', metavar='METRIC', dest='metrics',
            help="metric to use, can be specified multiple times, default from problem description",
        )
    if 'scoring_params' not in skip_arguments:
        fit_score_parser.add_argument(
            '-Y', '--scoring-param', nargs=2, action='append', metavar=('NAME', 'VALUE'), dest='scoring_params',
            help="hyper-parameter name and its value for scoring pipeline, can be specified multiple times, value should be JSON-serialized",
        )
    if 'save' not in skip_arguments:
        fit_score_parser.add_argument(
            '-s', '--save', type=utils.FileType('wb'), action='store',
            help="save fitted pipeline to a file, use \"-\" for stdout",
        )
    if 'output' not in skip_arguments:
        fit_score_parser.add_argument(
            '-o', '--output', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file, use \"-\" for stdout",
        )
    if 'scores' not in skip_arguments:
        fit_score_parser.add_argument(
            '-c', '--scores', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_score_parser.add_argument(
            '-O', '--output-run', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file, use \"-\" for stdout",
        )
    if 'scoring_random_seed' not in skip_arguments:
        fit_score_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    if 'expose_produced_outputs_dir' not in skip_arguments:
        fit_score_parser.add_argument(
            '-E', '--expose-produced-outputs', action='store', dest='expose_produced_outputs_dir',
            help="save to a directory produced outputs of all primitives from pipeline's produce run",
        )
    if 'full_inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-I', '--full-input', action='append', metavar='INPUT', dest='full_inputs',
            help="path or URI of an input full data, used in combination with the data preparation pipeline",
        )
    _data_preprocessing_configure_parser(fit_score_parser, skip_arguments=skip_arguments)
    fit_score_parser.set_defaults(runtime_handler=runtime.fit_score_handler)

    if 'scoring_pipeline' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-n', '--scoring-pipeline', default=runtime.DEFAULT_SCORING_PIPELINE_PATH, action='store',
            help="path to a scoring pipeline file (.json, .yml, or .yaml) or pipeline ID, default is standard scoring pipeline",
        )
    if 'problem' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'predictions' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-p', '--predictions', action='store', required=True,
            help="path to a predictions file, use \"-\" for stdin",
        )
    if 'score_inputs' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs', required=True,
            help="path or URI of an input score data, or an input full data when used in combination with the data preparation pipeline",
        )
    if 'metrics' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-e', '--metric', choices=utils.EnumArgProxy(problem_module.PerformanceMetric),
            action='append', metavar='METRIC', dest='metrics',
            help="metric to use, can be specified multiple times, default from problem description",
        )
    if 'scoring_params' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-Y', '--scoring-param', nargs=2, action='append', metavar=('NAME', 'VALUE'), dest='scoring_params',
            help="hyper-parameter name and its value for scoring pipeline, can be specified multiple times, value should be JSON-serialized",
        )
    if 'scores' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-c', '--scores', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'scoring_random_seed' not in skip_arguments:
        score_predictions_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    if 'predictions_random_seed' not in skip_arguments:
        score_predictions_parser.add_argument(
            '--predictions-random-seed', type=int, action='store', default=None,
            help="random seed used for predictions",
        )
    _data_preprocessing_configure_parser(score_predictions_parser, skip_arguments=skip_arguments)
    score_predictions_parser.set_defaults(runtime_handler=runtime.score_predictions_handler)

    if 'pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-p', '--pipeline', action='store',
            help="path to a pipeline file (.json, .yml, or .yaml) or pipeline ID"
        )
    if 'scoring_pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-n', '--scoring-pipeline', default=runtime.DEFAULT_SCORING_PIPELINE_PATH, action='store',
            help="path to a scoring pipeline file (.json, .yml, or .yaml) or pipeline ID, default is standard scoring pipeline",
        )
    if 'problem' not in skip_arguments:
        evaluate_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        evaluate_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input full data",
        )
    if 'input_run' not in skip_arguments:
        evaluate_parser.add_argument(
            '-u', '--input-run', type=utils.FileType('r', encoding='utf8'), action='store',
            help="path to a pipeline run file with configuration, use \"-\" for stdin",
        )
    if 'metrics' not in skip_arguments:
        evaluate_parser.add_argument(
            '-e', '--metric', choices=utils.EnumArgProxy(problem_module.PerformanceMetric), action='append', metavar='METRIC', dest='metrics',
            help="metric to use, can be specified multiple times, default from problem description",
        )
    if 'scoring_params' not in skip_arguments:
        evaluate_parser.add_argument(
            '-Y', '--scoring-param', nargs=2, action='append', metavar=('NAME', 'VALUE'), dest='scoring_params',
            help="hyper-parameter name and its value for scoring pipeline, can be specified multiple times, value should be JSON-serialized",
        )
    if 'scores' not in skip_arguments:
        evaluate_parser.add_argument(
            '-c', '--scores', type=utils.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        evaluate_parser.add_argument(
            '-O', '--output-run', type=utils.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file, use \"-\" for stdin",
        )
    if 'scoring_random_seed' not in skip_arguments:
        evaluate_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    _data_preprocessing_configure_parser(evaluate_parser, skip_arguments=skip_arguments)
    evaluate_parser.set_defaults(runtime_handler=runtime.evaluate_handler)

    if 'problem' not in skip_arguments:
        prepare_data_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        prepare_data_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input full data",
        )
    if 'save_dir' not in skip_arguments:
        prepare_data_parser.add_argument(
            '-s', '--save', action='store', dest='save_dir', required=True,
            help="directory where to save the data preparation results",
        )
    _data_preprocessing_configure_parser(prepare_data_parser, skip_arguments=skip_arguments)
    prepare_data_parser.set_defaults(runtime_handler=runtime.prepare_data_handler)


def handler(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
    pipeline_resolver: typing.Callable = None, pipeline_run_parser: typing.Callable = None,
    dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None,
    resolver_class: typing.Type[pipeline_module.Resolver] = None,
    no_resolver_class: typing.Type[pipeline_module.Resolver] = None,
    pipeline_class: typing.Type[pipeline_module.Pipeline] = None,
) -> None:
    # Dynamically fetch which subparser was used.
    subparser = parser._subparsers._group_actions[0].choices[arguments.d3m_command]  # type: ignore

    if arguments.d3m_command == 'primitive':
        primitive_handler(
            arguments,
            subparser,
        )

    elif arguments.d3m_command == 'index':
        logger.warning("\"index\" CLI command is deprecated. Use \"primitive\" CLI command instead.")

        primitive_handler(
            arguments,
            subparser,
        )

    elif arguments.d3m_command == 'pipeline':
        pipeline_handler(
            arguments,
            subparser,
            resolver_class=resolver_class,
            no_resolver_class=no_resolver_class,
            pipeline_class=pipeline_class,
        )

    elif arguments.d3m_command == 'problem':
        problem_handler(
            arguments,
            subparser,
            problem_resolver=problem_resolver,
        )

    elif arguments.d3m_command == 'dataset':
        dataset_handler(
            arguments,
            subparser,
            dataset_resolver=dataset_resolver,
        )

    elif arguments.d3m_command == 'pipeline-run':
        pipeline_run_handler(
            arguments,
            subparser,
        )

    elif arguments.d3m_command == 'runtime':
        runtime_handler(
            arguments,
            subparser,
            pipeline_resolver=pipeline_resolver,
            pipeline_run_parser=pipeline_run_parser,
            dataset_resolver=dataset_resolver,
            problem_resolver=problem_resolver,
        )

    elif arguments.d3m_command == 'openml':
        openml_handler(
            arguments,
            subparser,
            pipeline_resolver=pipeline_resolver,
            dataset_resolver=dataset_resolver,
            problem_resolver=problem_resolver,
        )

    else:
        raise exceptions.InvalidStateError("Cannot find a suitable command handler.")


# A fixed parser which correctly shows the error message for unknown arguments to the sub-command.
# See: https://gitlab.com/datadrivendiscovery/d3m/-/issues/409
class _ArgumentParser(argparse.ArgumentParser):
    # "parse_known_args" is made to behave exactly like "parse_args".
    def parse_known_args(self, args: typing.Sequence[str] = None, namespace: argparse.Namespace = None) -> typing.Tuple[argparse.Namespace, typing.List[str]]:
        namespace, argv = super().parse_known_args(args, namespace)
        if argv:
            msg = argparse._('unrecognized arguments: %s')  # type: ignore
            self.error(msg % ' '.join(argv))
        return namespace, argv


def configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    if 'pipeline_search_paths' not in skip_arguments:
        parser.add_argument(
            '-p', '--pipelines-path', action='append', metavar='PATH', dest='pipeline_search_paths',
            help="path to a directory with pipelines to resolve from (<pipeline id>.json, <pipeline id>.yml, or <pipeline id>.yaml), "
                 "can be specified multiple times, has priority over PIPELINES_PATH environment variable",
        )
    if 'logging_level' not in skip_arguments:
        parser.add_argument(
            '-l', '--logging-level', default='info', action='store',
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            help="logging level to use for the console",
        )
    if 'compute_digest' not in skip_arguments:
        parser.add_argument(
            '--compute-digest', choices=utils.EnumArgProxy(dataset_module.ComputeDigest),
            default=dataset_module.ComputeDigest.ONLY_IF_MISSING.name, action='store',
            help="when loading datasets, when to compute their digests, default is ONLY_IF_MISSING",
        )
    if 'strict_resolving' not in skip_arguments:
        parser.add_argument(
            '--strict-resolving', default=False, action='store_true',
            help="fail resolving if a resolved pipeline, primitive, or dataset, does not fully match specified reference",
        )
    if 'strict_digest' not in skip_arguments:
        parser.add_argument(
            '--strict-digest', default=False, action='store_true',
            help="when loading datasets, pipelines, primitives, or problem descriptions, if computed digest does not match the one provided in metadata, raise an exception?"
        )
    if 'modules' not in skip_arguments:
        parser.add_argument(
            '-M', '--module', action='append', metavar='PATH', dest='modules',
            help="a Python module name to import at startup, can be specified multiple times",
        )
    if 'version' not in skip_arguments:
        parser.add_argument(
            '-V', '--version', action='version', version=str(__version__),
            help="print d3m package version and exit",
        )

    subparsers = parser.add_subparsers(dest='d3m_command', title='commands', parser_class=_ArgumentParser)
    subparsers.required = True

    primitive_parser = subparsers.add_parser(
        'primitive', help="describe, validate, explore, and manage primitives",
        description="Describe, explore, and manage primitives.",
    )
    # Legacy command name. Deprecated. We do not use "aliases" argument to "add_parser"
    # because we want this command to be hidden.
    subparsers._name_parser_map['index'] = primitive_parser

    primitive_configure_parser(primitive_parser, skip_arguments=skip_arguments)

    pipeline_parser = subparsers.add_parser(
        'pipeline', help="describe and validate pipelines",
        description="Describe and validate pipelines.",
    )

    pipeline_configure_parser(pipeline_parser, skip_arguments=skip_arguments)

    problem_parser = subparsers.add_parser(
        'problem', help="describe and validate problems",
        description="Describe and validate problems.",
    )

    problem_configure_parser(problem_parser, skip_arguments=skip_arguments)

    dataset_parser = subparsers.add_parser(
        'dataset', help="describe and validate datasets",
        description="Describe and validate datasets.",
    )

    dataset_configure_parser(dataset_parser, skip_arguments=skip_arguments)

    pipeline_run_parser = subparsers.add_parser(
        'pipeline-run', help="validate pipeline runs",
        description="Validate pipeline runs.",
    )

    pipeline_run_configure_parser(pipeline_run_parser, skip_arguments=skip_arguments)

    runtime_parser = subparsers.add_parser(
        'runtime', help="run D3M pipelines",
        description="Run D3M pipelines.",
    )

    runtime_configure_parser(runtime_parser, skip_arguments=skip_arguments)

    openml_parser = subparsers.add_parser(
        'openml', help="use OpenML",
        description="Use OpenML.",
    )

    openml_configure_parser(openml_parser, skip_arguments=skip_arguments)

    # We set metavar at the end, when we know all subparsers. We want
    # "index" command to be hidden because it is deprecated.
    subparsers.metavar = '{' + ','.join(name for name in subparsers._name_parser_map.keys() if name != 'index') + '}'


def parse_modules_arguments(argv: typing.Sequence) -> typing.Sequence:
    parser = argparse.ArgumentParser(prog='d3m', description="Run a D3M core package command.")
    parser.add_argument(
        '-M', '--module', action='append', metavar='PATH', dest='modules',
        help="a Python module name to import at startup, can be specified multiple times",
    )

    # We care only about known arguments here.
    arguments, _ = parser.parse_known_args(argv[1:])

    return getattr(arguments, 'modules', []) or []


def main(argv: typing.Sequence) -> None:
    # We do this, before any real parsing of arguments. This allows imported modules to modify how parsing
    # of arguments is done (e.g., register additional metrics).
    for module in parse_modules_arguments(argv):
        try:
            importlib.import_module(module)
        except Exception as error:
            raise ImportError("Error importing \"{module}\".".format(module=module)) from error

    # Now we parse arguments once more, for real.
    parser = argparse.ArgumentParser(prog='d3m', description="Run a D3M core package command.")
    configure_parser(parser)

    arguments = parser.parse_args(argv[1:])

    try:
        logging.basicConfig(level=arguments.logging_level.upper())

        if not faulthandler.is_enabled():
            faulthandler.enable(file=sys.__stderr__)

        handler(arguments, parser)

    finally:
        # We close any file handlers.
        for value in arguments.__dict__.values():
            if hasattr(value, 'close'):
                value.close()
