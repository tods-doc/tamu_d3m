import os
import pathlib
import shutil
import sys
import tempfile
import unittest

from d3m import index, utils
from d3m.metadata import base as metadata_base, problem as problem_module
from d3m.metadata.pipeline import Resolver, Pipeline
from d3m.contrib.openml import crawler

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common_primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

from common_primitives.train_score_split import TrainScoreDatasetSplitPrimitive

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PIPELINE_DIR = os.path.join(TEST_DATA_DIR, 'pipelines')


class TestContribOpenML(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @classmethod
    def setUpClass(cls):
        to_register = {
            'd3m.primitives.evaluation.train_score_dataset_split.Common': TrainScoreDatasetSplitPrimitive,
        }

        # To hide any logging or stdout output.
        with utils.silence():
            for python_path, primitive in to_register.items():
                index.register_primitive(python_path, primitive)

    def _get_dir_structure(self, directory):
        structure = []
        for dirpath, dirnames, filenames in os.walk(directory):
            # Make sure we traverse in deterministic order.
            dirnames[:] = sorted(dirnames)
            for dirname in dirnames:
                structure.append(os.path.join(dirpath, dirname))
            for filename in sorted(filenames):
                structure.append(os.path.join(dirpath, filename))
        return structure

    def _assert_dir_structure(self, save_dir, files):
        save_dir_path = pathlib.PurePath(save_dir)
        self.assertEqual(self._get_dir_structure(save_dir), [str(save_dir_path / pathlib.PurePosixPath(f)) for f in files])

    def test_convert_openml_task(self):
        self.maxDiff = None

        with open(os.path.join(os.path.join(PIPELINE_DIR, 'data-preparation-train-test-split.yml')), 'r') as data_pipeline_file:
            data_pipeline = Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())
        data_params = {
            'train_score_ratio': '0.8',
            'shuffle': 'true',
            'stratified': 'true',
        }
        task_id = 8
        save_dir = os.path.join(self.test_dir, 'single_dataset')
        save_dir_path = pathlib.PurePath(save_dir)

        datasets = {}
        crawler.crawl_openml_task(
            datasets=datasets, task_id=task_id, save_dir=save_dir,
            data_pipeline=data_pipeline, data_params=data_params,
            context=metadata_base.Context.TESTING,
        )
        self.assertEqual(datasets, {
            'openml_dataset_8': str(save_dir_path / pathlib.PurePosixPath('openml_dataset_8/datasetDoc.json')),
            'openml_dataset_8_TRAIN': str(save_dir_path / pathlib.PurePosixPath('TRAIN/dataset_TRAIN/datasetDoc.json')),
            'openml_dataset_8_TEST': str(save_dir_path / pathlib.PurePosixPath('TEST/dataset_TEST/datasetDoc.json')),
            'openml_dataset_8_SCORE': str(save_dir_path / pathlib.PurePosixPath('SCORE/dataset_SCORE/datasetDoc.json')),
        })

        self._assert_dir_structure(save_dir, [
            'SCORE',
            'TEST',
            'TRAIN',
            'openml_dataset_8',
            'openml_problem_8',
            'data_preparation_pipeline_run.pkl',
            'SCORE/dataset_SCORE',
            'SCORE/problem_SCORE',
            'SCORE/dataset_SCORE/tables',
            'SCORE/dataset_SCORE/datasetDoc.json',
            'SCORE/dataset_SCORE/tables/learningData.csv',
            'SCORE/problem_SCORE/problemDoc.json',
            'TEST/dataset_TEST',
            'TEST/problem_TEST',
            'TEST/dataset_TEST/tables',
            'TEST/dataset_TEST/datasetDoc.json',
            'TEST/dataset_TEST/tables/learningData.csv',
            'TEST/problem_TEST/problemDoc.json',
            'TRAIN/dataset_TRAIN',
            'TRAIN/problem_TRAIN',
            'TRAIN/dataset_TRAIN/tables',
            'TRAIN/dataset_TRAIN/datasetDoc.json',
            'TRAIN/dataset_TRAIN/tables/learningData.csv',
            'TRAIN/problem_TRAIN/problemDoc.json',
            'openml_dataset_8/tables',
            'openml_dataset_8/datasetDoc.json',
            'openml_dataset_8/tables/learningData.csv',
            'openml_problem_8/problemDoc.json',
        ])

    def test_crawl_openml_task(self):
        self.maxDiff = None

        with open(os.path.join(os.path.join(PIPELINE_DIR, 'data-preparation-train-test-split.yml')), 'r') as data_pipeline_file:
            data_pipeline = Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())
        data_params = {
            'train_score_ratio': '0.8',
            'shuffle': 'true',
            'stratified': 'true',
        }
        save_dir = os.path.join(self.test_dir, 'multi_dataset')
        max_tasks = 3
        has_errored = crawler.crawl_openml(
            save_dir=save_dir, task_types=(problem_module.OpenMLTaskType.SUPERVISED_CLASSIFICATION,),
            data_pipeline=data_pipeline, data_params=data_params,
            context=metadata_base.Context.TESTING,
            max_tasks=max_tasks,
        )
        self.assertFalse(has_errored)

        self._assert_dir_structure(save_dir, [
            'openml_task_2',
            'openml_task_3',
            'openml_task_4',
            'openml_task_2/SCORE',
            'openml_task_2/TEST',
            'openml_task_2/TRAIN',
            'openml_task_2/openml_dataset_2',
            'openml_task_2/openml_problem_2',
            'openml_task_2/data_preparation_pipeline_run.pkl',
            'openml_task_2/SCORE/dataset_SCORE',
            'openml_task_2/SCORE/problem_SCORE',
            'openml_task_2/SCORE/dataset_SCORE/tables',
            'openml_task_2/SCORE/dataset_SCORE/datasetDoc.json',
            'openml_task_2/SCORE/dataset_SCORE/tables/learningData.csv',
            'openml_task_2/SCORE/problem_SCORE/problemDoc.json',
            'openml_task_2/TEST/dataset_TEST',
            'openml_task_2/TEST/problem_TEST',
            'openml_task_2/TEST/dataset_TEST/tables',
            'openml_task_2/TEST/dataset_TEST/datasetDoc.json',
            'openml_task_2/TEST/dataset_TEST/tables/learningData.csv',
            'openml_task_2/TEST/problem_TEST/problemDoc.json',
            'openml_task_2/TRAIN/dataset_TRAIN',
            'openml_task_2/TRAIN/problem_TRAIN',
            'openml_task_2/TRAIN/dataset_TRAIN/tables',
            'openml_task_2/TRAIN/dataset_TRAIN/datasetDoc.json',
            'openml_task_2/TRAIN/dataset_TRAIN/tables/learningData.csv',
            'openml_task_2/TRAIN/problem_TRAIN/problemDoc.json',
            'openml_task_2/openml_dataset_2/tables',
            'openml_task_2/openml_dataset_2/datasetDoc.json',
            'openml_task_2/openml_dataset_2/tables/learningData.csv',
            'openml_task_2/openml_problem_2/problemDoc.json',
            'openml_task_3/SCORE',
            'openml_task_3/TEST',
            'openml_task_3/TRAIN',
            'openml_task_3/openml_dataset_3',
            'openml_task_3/openml_problem_3',
            'openml_task_3/data_preparation_pipeline_run.pkl',
            'openml_task_3/SCORE/dataset_SCORE',
            'openml_task_3/SCORE/problem_SCORE',
            'openml_task_3/SCORE/dataset_SCORE/tables',
            'openml_task_3/SCORE/dataset_SCORE/datasetDoc.json',
            'openml_task_3/SCORE/dataset_SCORE/tables/learningData.csv',
            'openml_task_3/SCORE/problem_SCORE/problemDoc.json',
            'openml_task_3/TEST/dataset_TEST',
            'openml_task_3/TEST/problem_TEST',
            'openml_task_3/TEST/dataset_TEST/tables',
            'openml_task_3/TEST/dataset_TEST/datasetDoc.json',
            'openml_task_3/TEST/dataset_TEST/tables/learningData.csv',
            'openml_task_3/TEST/problem_TEST/problemDoc.json',
            'openml_task_3/TRAIN/dataset_TRAIN',
            'openml_task_3/TRAIN/problem_TRAIN',
            'openml_task_3/TRAIN/dataset_TRAIN/tables',
            'openml_task_3/TRAIN/dataset_TRAIN/datasetDoc.json',
            'openml_task_3/TRAIN/dataset_TRAIN/tables/learningData.csv',
            'openml_task_3/TRAIN/problem_TRAIN/problemDoc.json',
            'openml_task_3/openml_dataset_3/tables',
            'openml_task_3/openml_dataset_3/datasetDoc.json',
            'openml_task_3/openml_dataset_3/tables/learningData.csv',
            'openml_task_3/openml_problem_3/problemDoc.json',
            'openml_task_4/SCORE',
            'openml_task_4/TEST',
            'openml_task_4/TRAIN',
            'openml_task_4/openml_dataset_4',
            'openml_task_4/openml_problem_4',
            'openml_task_4/data_preparation_pipeline_run.pkl',
            'openml_task_4/SCORE/dataset_SCORE',
            'openml_task_4/SCORE/problem_SCORE',
            'openml_task_4/SCORE/dataset_SCORE/tables',
            'openml_task_4/SCORE/dataset_SCORE/datasetDoc.json',
            'openml_task_4/SCORE/dataset_SCORE/tables/learningData.csv',
            'openml_task_4/SCORE/problem_SCORE/problemDoc.json',
            'openml_task_4/TEST/dataset_TEST',
            'openml_task_4/TEST/problem_TEST',
            'openml_task_4/TEST/dataset_TEST/tables',
            'openml_task_4/TEST/dataset_TEST/datasetDoc.json',
            'openml_task_4/TEST/dataset_TEST/tables/learningData.csv',
            'openml_task_4/TEST/problem_TEST/problemDoc.json',
            'openml_task_4/TRAIN/dataset_TRAIN',
            'openml_task_4/TRAIN/problem_TRAIN',
            'openml_task_4/TRAIN/dataset_TRAIN/tables',
            'openml_task_4/TRAIN/dataset_TRAIN/datasetDoc.json',
            'openml_task_4/TRAIN/dataset_TRAIN/tables/learningData.csv',
            'openml_task_4/TRAIN/problem_TRAIN/problemDoc.json',
            'openml_task_4/openml_dataset_4/tables',
            'openml_task_4/openml_dataset_4/datasetDoc.json',
            'openml_task_4/openml_dataset_4/tables/learningData.csv',
            'openml_task_4/openml_problem_4/problemDoc.json',
        ])

    def test_ignore_openml_task(self):
        self.maxDiff = None

        with open(os.path.join(os.path.join(PIPELINE_DIR, 'data-preparation-train-test-split.yml')), 'r') as data_pipeline_file:
            data_pipeline = Pipeline.from_yaml(data_pipeline_file, resolver=Resolver())
        data_params = {
            'train_score_ratio': '0.8',
            'shuffle': 'true',
            'stratified': 'true',
        }
        save_dir = os.path.join(self.test_dir, 'ignore_dataset')
        max_tasks = 1
        has_errored = crawler.crawl_openml(
            save_dir=save_dir, task_types=(problem_module.OpenMLTaskType.SUPERVISED_CLASSIFICATION,),
            data_pipeline=data_pipeline, data_params=data_params,
            context=metadata_base.Context.TESTING,
            max_tasks=max_tasks,
            ignore_tasks=[3],
            ignore_datasets=[2],
        )
        self.assertFalse(has_errored)

        self._assert_dir_structure(save_dir, [
            'openml_task_4',
            'openml_task_4/SCORE',
            'openml_task_4/TEST',
            'openml_task_4/TRAIN',
            'openml_task_4/openml_dataset_4',
            'openml_task_4/openml_problem_4',
            'openml_task_4/data_preparation_pipeline_run.pkl',
            'openml_task_4/SCORE/dataset_SCORE',
            'openml_task_4/SCORE/problem_SCORE',
            'openml_task_4/SCORE/dataset_SCORE/tables',
            'openml_task_4/SCORE/dataset_SCORE/datasetDoc.json',
            'openml_task_4/SCORE/dataset_SCORE/tables/learningData.csv',
            'openml_task_4/SCORE/problem_SCORE/problemDoc.json',
            'openml_task_4/TEST/dataset_TEST',
            'openml_task_4/TEST/problem_TEST',
            'openml_task_4/TEST/dataset_TEST/tables',
            'openml_task_4/TEST/dataset_TEST/datasetDoc.json',
            'openml_task_4/TEST/dataset_TEST/tables/learningData.csv',
            'openml_task_4/TEST/problem_TEST/problemDoc.json',
            'openml_task_4/TRAIN/dataset_TRAIN',
            'openml_task_4/TRAIN/problem_TRAIN',
            'openml_task_4/TRAIN/dataset_TRAIN/tables',
            'openml_task_4/TRAIN/dataset_TRAIN/datasetDoc.json',
            'openml_task_4/TRAIN/dataset_TRAIN/tables/learningData.csv',
            'openml_task_4/TRAIN/problem_TRAIN/problemDoc.json',
            'openml_task_4/openml_dataset_4/tables',
            'openml_task_4/openml_dataset_4/datasetDoc.json',
            'openml_task_4/openml_dataset_4/tables/learningData.csv',
            'openml_task_4/openml_problem_4/problemDoc.json',
        ])

if __name__ == '__main__':
    unittest.main()
