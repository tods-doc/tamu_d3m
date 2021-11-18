import pandas as pd
import numpy.random

import d3m.metrics
from d3m import container
from d3m.metadata import base as metadata_base
from d3m.contrib.primitives import compute_scores


class BenchMetrics:
    params = [[100, 1000, 10000, 50000]]
    param_names = ['rows']

    def time_vectorize_columns(self, rows):
        df = pd.DataFrame({
            'd3mIndex': range(rows),
            'col0': (1,) * rows
        })
        vdf = d3m.metrics.Metric.vectorize_columns(df)

    def time_classification_scores(self, rows):
        # This has been cut-and-paste from test_compute_scores.py
        truth = container.DataFrame({
            'd3mIndex': range(rows),
            'col0': (1,) * rows
        })

        truth_dataset = container.Dataset({'learningData': truth}, generate_metadata=True)
        truth_dataset.metadata = truth_dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        truth_dataset.metadata = truth_dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Target')
        truth_dataset.metadata = truth_dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

        # predictions are identical to truth, should have no impact on performance.
        predictions = truth

        # configure primitive
        hyperparams_class = compute_scores.ComputeScoresPrimitive.metadata.get_hyperparams()
        metrics_class = hyperparams_class.configuration['metrics'].elements
        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'ACCURACY',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'F1_MICRO',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'F1_MACRO',
                'pos_label': None,
                'k': None,
            })],
        }))

        # run scoring.
        scores = primitive.produce(inputs=predictions, score_dataset=truth_dataset).value
