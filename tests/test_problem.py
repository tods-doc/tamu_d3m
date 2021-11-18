import os.path
import pickle
import shutil
import unittest
import tempfile

from d3m import utils
from d3m.metadata import problem, pipeline_run


class TestProblem(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_basic(self):
        self.maxDiff = None

        problem_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'problems', 'iris_problem_1', 'problemDoc.json'))

        problem_uri = utils.path_to_uri(problem_doc_path)

        problem_description = problem.Problem.load(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), {
            'id': 'iris_problem_1',
            'digest': '1a12135422967aa0de0c4629f4f58d08d39e97f9133f7b50da71420781aa18a5',
            'version': '4.0.0',
            'location_uris': [
                problem_uri,
            ],
            'name': 'Distinguish Iris flowers',
            'description': 'Distinguish Iris flowers of three related species.',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'problem': {
                'task_keywords': [problem.TaskKeyword.CLASSIFICATION, problem.TaskKeyword.MULTICLASS],
                'performance_metrics': [
                    {
                        'metric': problem.PerformanceMetric.ACCURACY,
                    }
                ]
            },
            'inputs': [
                {
                    'dataset_id': 'iris_dataset_1',
                    'targets': [
                        {
                            'target_index': 0,
                            'resource_id': 'learningData',
                            'column_index': 5,
                            'column_name': 'species',
                        }
                    ]
                }
            ],
        })

        self.assertEqual(problem_description.to_json_structure(), {
            'id': 'iris_problem_1',
            'digest': '1a12135422967aa0de0c4629f4f58d08d39e97f9133f7b50da71420781aa18a5',
            'version': '4.0.0',
            'location_uris': [
                problem_uri,
            ],
            'name': 'Distinguish Iris flowers',
            'description': 'Distinguish Iris flowers of three related species.',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'problem': {
                'task_keywords': [problem.TaskKeyword.CLASSIFICATION, problem.TaskKeyword.MULTICLASS],
                'performance_metrics': [
                    {
                        'metric': problem.PerformanceMetric.ACCURACY,
                    }
                ]
            },
            'inputs': [
                {
                    'dataset_id': 'iris_dataset_1',
                    'targets': [
                        {
                            'target_index': 0,
                            'resource_id': 'learningData',
                            'column_index': 5,
                            'column_name': 'species',
                        }
                    ]
                }
            ],
        })

        self.assertEqual(problem_description.to_json_structure(), {
            'id': 'iris_problem_1',
            'digest': '1a12135422967aa0de0c4629f4f58d08d39e97f9133f7b50da71420781aa18a5',
            'version': '4.0.0',
            'location_uris': [
                problem_uri,
            ],
            'name': 'Distinguish Iris flowers',
            'description': 'Distinguish Iris flowers of three related species.',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'problem': {
                'task_keywords': ['CLASSIFICATION', 'MULTICLASS'],
                'performance_metrics': [
                    {
                        'metric': 'ACCURACY',
                    }
                ]
            },
            'inputs': [
                {
                    'dataset_id': 'iris_dataset_1',
                    'targets': [
                        {
                            'target_index': 0,
                            'resource_id': 'learningData',
                            'column_index': 5,
                            'column_name': 'species',
                        }
                    ]
                }
            ],
        })

        pipeline_run.validate_problem(problem_description.to_json_structure(canonical=True))
        problem.PROBLEM_SCHEMA_VALIDATOR.validate(problem_description.to_json_structure(canonical=True))

    def test_conversion(self):
        problem_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'problems', 'iris_problem_1', 'problemDoc.json'))

        problem_uri = utils.path_to_uri(problem_doc_path)

        problem_description = problem.Problem.load(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), problem.Problem.from_json_structure(problem_description.to_json_structure(), strict_digest=True).to_simple_structure())

        # Legacy.
        self.assertEqual(utils.to_json_structure(problem_description.to_simple_structure()), problem.Problem.from_json_structure(utils.to_json_structure(problem_description.to_simple_structure()), strict_digest=True).to_simple_structure())

        self.assertIs(problem.Problem.from_json_structure(problem_description.to_json_structure(), strict_digest=True)['problem']['task_keywords'][0], problem.TaskKeyword.CLASSIFICATION)

    def test_unparse(self):
        self.assertEqual(problem.TaskKeyword.CLASSIFICATION.unparse(), 'classification')
        self.assertEqual(problem.TaskKeyword.MULTICLASS.unparse(), 'multiClass')
        self.assertEqual(problem.PerformanceMetric.ACCURACY.unparse(), 'accuracy')

    def test_normalize(self):
        self.assertEqual(problem.PerformanceMetric._normalize(0, 1, 0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(0, 2, 0.5), 0.25)
        self.assertEqual(problem.PerformanceMetric._normalize(1, 2, 1.5), 0.5)

        self.assertEqual(problem.PerformanceMetric._normalize(-1, 0, -0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(-2, 0, -1.5), 0.25)
        self.assertEqual(problem.PerformanceMetric._normalize(-2, -1, -1.5), 0.5)

        self.assertEqual(problem.PerformanceMetric._normalize(1, 0, 0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(2, 0, 0.5), 0.75)
        self.assertEqual(problem.PerformanceMetric._normalize(2, 1, 1.5), 0.5)

        self.assertEqual(problem.PerformanceMetric._normalize(0, -1, -0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(0, -2, -1.5), 0.75)
        self.assertEqual(problem.PerformanceMetric._normalize(-1, -2, -1.5), 0.5)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 0.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 1000.0), 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 5000.0), 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 1.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 1000.0), 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 5000.0), 0.013399004523107192)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, -1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, -0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, 1000.0), 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, 5000.0), 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, 0.0), 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, 0.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, -0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, -1000.0), 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, -5000.0), 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, 1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, 0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, -1000.0), 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, -5000.0), 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, 0.0), 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -1.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -1000.0), 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -5000.0), 0.013399004523107192)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 0.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 1000.0), 1 - 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 5000.0), 1 - 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 1.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 1000.0), 1 - 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 5000.0), 1 - 0.013399004523107192)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), -1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), -0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), 1000.0), 1 - 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), 5000.0), 1 - 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), 0.0), 1 - 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), 0.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), -0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), -1000.0), 1 - 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), -5000.0), 1 - 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), 1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), 0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), -1000.0), 1 - 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), -5000.0), 1 - 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), 0.0), 1 - 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -1.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -1000.0), 1 - 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -5000.0), 1 - 0.013399004523107192)

    def test_pickle(self):
        value = problem.PerformanceMetric.ACCURACY

        pickled = pickle.dumps(value)
        unpickled = pickle.loads(pickled)

        self.assertEqual(value, unpickled)
        self.assertIs(value.get_class(), unpickled.get_class())

    def test_save_d3m_problem(self):
        self.maxDiff = None

        problem_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'problems', 'iris_problem_1', 'problemDoc.json'))

        problem_uri = utils.path_to_uri(problem_doc_path)
        problem_description = problem.Problem.load(problem_uri)

        problem_path = os.path.join(os.path.abspath(self.test_dir), 'problem', 'problemDoc.json')
        saved_problem_uri = utils.path_to_uri(problem_path)
        problem_description.save(saved_problem_uri)
        saved_problem_description = problem.Problem.load(saved_problem_uri)

        original = problem_description.to_simple_structure()
        saved = saved_problem_description.to_simple_structure()
        del original['location_uris']
        del saved['location_uris']

        self.assertEqual(original, saved)

    def test_openml_binary_classification(self):
        problem_uri = 'https://www.openml.org/t/{problem_id}'.format(problem_id=8)
        problem_description = problem.get_problem(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), {
            'id': 'openml_problem_8',
            'digest': '1a0194c4c439312cee69fbeb2b02744a640b1cd51cfb8d8f0fae533afec926a2',
            'version': '1.0',
            'location_uris': ['https://www.openml.org/t/8'],
            'name': 'Task 8: liver-disorders (Supervised Classification)',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'problem': {
                'task_keywords': [
                    problem.TaskKeyword.CLASSIFICATION,
                    problem.TaskKeyword.BINARY,
                    problem.TaskKeyword.TABULAR,
                ],
                'performance_metrics': [{'metric': problem.PerformanceMetric.ACCURACY}],
            },
            'source': {'uris': ['https://www.openml.org/t/8']},
            'inputs': [{
                'dataset_id': 'openml_dataset_8',
                'targets': [{
                    'target_index': 0,
                    'resource_id': 'learningData',
                    'column_index': 7,
                    'column_name': 'selector',
                }]
            }],
            'keywords': [
                'basic',
                'study_107',
                'study_50',
                'study_73',
                'under100k',
                'under1m',
            ],
        })

    def test_openml_multiclass_classification(self):
        problem_uri = 'https://www.openml.org/t/{problem_id}'.format(problem_id=1786)
        problem_description = problem.get_problem(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), {
            'id': 'openml_problem_1786',
            'digest': '70848cb6a19d62ede0aff14f795e81e46359883f6b6e36f16117cb1a62a94457',
            'version': '1.0',
            'keywords': ['under100k', 'under1m'],
            'location_uris': ['https://www.openml.org/t/1786'],
            'name': 'Task 1786: mfeat-zernike (Supervised Classification)',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'source': {'uris': ['https://www.openml.org/t/1786']},
            'problem': {
                'task_keywords': [
                    problem.TaskKeyword.CLASSIFICATION,
                    problem.TaskKeyword.MULTICLASS,
                    problem.TaskKeyword.TABULAR,
                ],
                'performance_metrics': [{'metric': problem.PerformanceMetric.ACCURACY}],
            },
            'inputs': [{
                'dataset_id': 'openml_dataset_22',
                'targets': [{
                    'target_index': 0,
                    'resource_id': 'learningData',
                    'column_index': 48,
                    'column_name': 'class',
                }]
            }]
        })

    def test_openml_regression(self):
        problem_uri = 'https://www.openml.org/t/{problem_id}'.format(problem_id=2295)
        problem_description = problem.get_problem(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), {
            'id': 'openml_problem_2295',
            'digest': '4565a39b5ebd774e900980d2969a4a6641237785982429c8763dbbd569180f1e',
            'version': '1.0',
            'location_uris': ['https://www.openml.org/t/2295'],
            'keywords': ['under1m'],
            'name': 'Task 2295: cholesterol (Supervised Regression)',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'source': {'uris': ['https://www.openml.org/t/2295']},
            'problem': {
                'task_keywords': [
                    problem.TaskKeyword.REGRESSION,
                    problem.TaskKeyword.TABULAR,
                ],
                'performance_metrics': [{'metric': problem.PerformanceMetric.MEAN_ABSOLUTE_ERROR}]
            },
            'inputs': [{
                'dataset_id': 'openml_dataset_204',
                'targets': [{
                    'target_index': 0,
                    'resource_id': 'learningData',
                    'column_index': 14,
                    'column_name': 'chol'
                }]
            }],
        })

    def test_openml_classification_with_index(self):
        problem_uri = 'https://www.openml.org/t/{problem_id}'.format(problem_id=168861)
        problem_description = problem.get_problem(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), {
            'id': 'openml_problem_168861',
            'digest': '3269c8a61d39ccc59ed10400e16d9921dc1d178e2c6017377f08924768a13de6',
            'version': '1.0',
            'location_uris': ['https://www.openml.org/t/168861'],
            'name': 'Task 168861: DRSongsLyrics (Supervised Classification)',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'source': {'uris': ['https://www.openml.org/t/168861']},
            'problem': {
                'task_keywords': [
                    problem.TaskKeyword.CLASSIFICATION,
                    problem.TaskKeyword.BINARY,
                    problem.TaskKeyword.TABULAR,
                ],
                'performance_metrics': [{'metric': problem.PerformanceMetric.ACCURACY}],
            },
            'inputs': [{
                'dataset_id': 'openml_dataset_41496',
                'targets': [{
                    'target_index': 0,
                    'resource_id': 'learningData',
                    'column_index': 2,
                    'column_name': 'class'
                }]
            }],
        })


if __name__ == '__main__':
    unittest.main()
