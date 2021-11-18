Register a Custom Metric
========================

New problem metrics can be added to D3M core package by using the register_metric() call.
See :py:func:`d3m.metadata.problem.PerformanceMetricBase.register_metric`
An example of adding a new custom metric by the name CUSTOM_METRIC is shown below-

Example
~~~~~~~

.. code:: python

    from d3m.metadata import problem
    from d3m import metrics as d3m_metrics
    from sklearn import metrics

    # Class representing the new metric to be added
    class CustomMetric(d3m_metrics.Metric):
        # Score function computing the actual score using true labels and predictions
        # Input arguments "truth" and "predictions" are data frames with "d3mIndex" column aligning the rows in both frames.
        def score(self, truth: d3m_metrics.Truth, predictions: d3m_metrics.Predictions) -> float:
            predictions = self.align(truth, predictions)
            truth_targets = self.get_target_columns(truth)
            predictions_targets = self.get_target_columns(predictions)
            # Customized scoring to be done here!
            return metrics.accuracy_score(truth_targets, predictions_targets)

    # Registering "CustomMetric" with the d3m core package
    problem.PerformanceMetric.register_metric('CUSTOM_METRIC', best_value=1.0, worst_value=0.0, score_class=CustomMetric)

    # Score predictions-vs-true labels using CustomMetric
    new_metric = CustomMetric()
    result = new_metric.score(y_true, y_pred)

The score() function expects inputs "truth" and "predictions" to be data frames of equal sizes with "d3mIndex" column aligning the rows in both frames.
Target columns from each set (truth, predictions) are evaluated after removing the columns with the names "d3mIndex", "rank" and "confidence". 
Predictions for several D3M metrics are structured as listed `here <https://gitlab.com/datadrivendiscovery/data-supply/-/blob/shared/documentation/problemSchema.md#predictions-file>`__.

Sample unit test for scoring custom metric is available at test_custom_scoring_metric() at tests/test_metrics.py.
See :py:func:`TestMetrics.test_custom_scoring_metric`

This completes addition of a new scoring metric.


Register a Custom Problem Type
==============================

New problem types can be added to D3M core package by using the register_value() call.
See :py:func:`d3m.utils.Enum.register_value`
An example of adding a new custom task by the name CUSTOM_TASK is shown below-

Example
~~~~~~~

.. code:: python

    from d3m.metadata import problem

    # Registering "CUSTOM_TASK" as a new TaskKeyword
    problem.TaskKeyword.register_value('CUSTOM_TASK', 'CUSTOM_TASK')

Sample unit test for adding new custom task is available at test_extendable_enum() at tests/test_utils.py.
See :py:func:`TestUtils.test_extendable_enum`

This completes addition of a new problem task.

