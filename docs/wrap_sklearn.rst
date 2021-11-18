Wrap a Sklearn-compatible Primitive
===================================

This tutorial will walk-through and explain select parts of the `SKRandomForestClassifier Primitive code <https://gitlab.com/datadrivendiscovery/sklearn-wrap/-/blob/dev-dist/sklearn_wrap/SKRandomForestClassifier.py>`__
For more information on DO's and DONT's please visit :ref:`Write a Good Primitive <write_primitive>`

SKRandomForestClassifier Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Custom import commands if any
    from sklearn.ensemble.forest import RandomForestClassifier


    from d3m.container.numpy import ndarray as d3m_ndarray
    from d3m.container import DataFrame as d3m_dataframe
    from d3m.metadata import hyperparams, params, base as metadata_base
    from d3m import utils
    from d3m.base import utils as base_utils
    from d3m.exceptions import PrimitiveNotFittedError
    from d3m.primitive_interfaces.base import CallResult, DockerContainer

    from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
    from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
    from d3m import exceptions


These are the necessary imports from ``d3m`` core which are utilized in every primitive as well as our custom import of
:class:`~sklearn.ensemble.forest.RandomForestClassifier` from ``sklearn``. Since we are wrapping a supervised classification primitive we must also
import the :class:`~d3m.primitive_interfaces.supervised_learning.SupervisedLearnerPrimitiveBase` base class from ``d3m``.

.. code:: python

    class Params(params.Params):
        estimators_: Optional[List[sklearn.tree.DecisionTreeClassifier]]
        classes_: Optional[Union[ndarray, List[ndarray]]]
        n_classes_: Optional[Union[int, List[int]]]
        n_features_: Optional[int]
        n_outputs_: Optional[int]
        oob_score_: Optional[float]
        oob_decision_function_: Optional[ndarray]
        base_estimator_: Optional[object]
        estimator_params: Optional[tuple]
        base_estimator: Optional[object]
        input_column_names: Optional[pandas.core.indexes.base.Index]
        target_names_: Optional[Sequence[Any]]
        training_indices_: Optional[Sequence[int]]
        target_column_indices_: Optional[Sequence[int]]
        target_columns_metadata_: Optional[List[OrderedDict]]

Next we defined our primitive's parameters as class attributes with type annotations. The majority of this list is comprised
of the attributes from the `sklearn source code <https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/ensemble/_forest.py#L1072-L1115>`__.
In addition, we have added a few more parameters which contains information we would like to store during fitting such as
``target_names_``, ``training_indices_`` etc. More information can be found in :ref:`Parameters <parameters>`

.. code:: python

    class Hyperparams(hyperparams.Hyperparams):
        n_estimators = hyperparams.Bounded[int](
            default=10,
            lower=1,
            upper=None,
            description='The number of trees in the forest.',
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )
        criterion = hyperparams.Enumeration[str](
            values=['gini', 'entropy'],
            default='gini',
            description='The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. Note: this parameter is tree-specific.',
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )
        .
        .
        .
        use_inputs_columns = hyperparams.Set(
            elements=hyperparams.Hyperparameter[int](-1),
            default=(),
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="A set of column indices to force primitive to use as training input. If any specified column cannot be parsed, it is skipped.",
        )
        use_outputs_columns = hyperparams.Set(
            elements=hyperparams.Hyperparameter[int](-1),
            default=(),
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="A set of column indices to force primitive to use as training target. If any specified column cannot be parsed, it is skipped.",
        )
        exclude_inputs_columns = hyperparams.Set(
            elements=hyperparams.Hyperparameter[int](-1),
            default=(),
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="A set of column indices to not use as training inputs. Applicable only if \"use_columns\" is not provided.",
        )
        exclude_outputs_columns = hyperparams.Set(
            elements=hyperparams.Hyperparameter[int](-1),
            default=(),
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="A set of column indices to not use as training target. Applicable only if \"use_columns\" is not provided.",
        )
        return_result = hyperparams.Enumeration(
            values=['append', 'replace', 'new'],
            default='new',
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
        )
        use_semantic_types = hyperparams.UniformBool(
            default=False,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
        )
        add_index_columns = hyperparams.UniformBool(
            default=False,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
        )
        error_on_no_input = hyperparams.UniformBool(
            default=True,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.",
        )
        return_semantic_type = hyperparams.Enumeration[str](
            values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
            default='https://metadata.datadrivendiscovery.org/types/PredictedTarget',
            description='Decides what semantic type to attach to generated output',
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
        )

Above we added the Hyper-parameters from the `sklearn source code <https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/ensemble/_forest.py#L899-L1070>`__
(this has been abridged in the example block above). Following the Hyper-parameters from the original source code we then
add the standard ``d3m`` Hyper-parameters to the :class:`~d3m.metadata.hyperparams.Hyperparams` class.
These Hyper-parameters include ``return_result`` which indicate whether the output
should append to the original dataframe (``append``), replace the altered columns on the original dataframe (``replace``),
or simply return the output as is (``new``). More information can be found in :ref:`Hyper-parameters <hyperparameters>`.

.. code:: python

    class SKRandomForestClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                              ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams]):

        metadata = metadata_base.PrimitiveMetadata({
             "algorithm_types": [metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST, ],
             "name": "sklearn.ensemble.forest.RandomForestClassifier",
             "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
             "python_path": "d3m.primitives.classification.random_forest.SKlearn",
             "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html']},
             "version": "2020.12.1",
             "id": "1dd82833-5692-39cb-84fb-2455683075f3",
             "hyperparams_to_tune": ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
             'installation': [
                            {'type': metadata_base.PrimitiveInstallationType.PIP,
                               'package_uri': 'git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@{git_commit}#egg=sklearn_wrap'.format(
                                   git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                                ),
                               }]
        })

We then add the Primitive metadata which describes the algorithm type and family, name, id etc. The ``id`` should be unique
for every primitive and we recommend using :meth:`~uuid.uuid4()` to generate. We also list our recommendations for which Hyper-Parameters
to tune in ``hyperparams_to_tune``. More information can be found in :ref:`Primitive metadata <metadata>` and :ref:`Primitive family <primitive_family>`.

.. code:: python

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = RandomForestClassifier(
              n_estimators=self.hyperparams['n_estimators'],
              criterion=self.hyperparams['criterion'],
                .
                .
                .
              random_state=self.random_seed,
              verbose=_verbose
        )

        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._fitted = False
        self._new_training_data = False

In our ``init`` we initialize all of our parameters as well as the :class:`~sklearn.ensemble.forest.RandomForestClassifier`.

Note: you should use ``self.random_seed`` for ``random_state`` instead of adding it as a Hyper-parameter.

.. code:: python

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._fitted = False
        self._new_training_data = True

Next we add our ``set_training_data`` method which will be used by TA2 systems to set the inputs and output. Any
pre-processing or data selection should be done in the ``fit`` method instead of ``set_training_data``. More information
can be found in :ref:`Input/Output types <input_output_types>`

.. code:: python

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._training_outputs, self._target_names, self._target_column_indices = self._get_targets(self._outputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns.astype(str)

        if len(self._training_indices) > 0 and len(self._target_column_indices) > 0:
            self._target_columns_metadata = self._get_target_columns_metadata(self._training_outputs.metadata, self.hyperparams)
            sk_training_output = self._training_outputs.values

            shape = sk_training_output.shape
            if len(shape) == 2 and shape[1] == 1:
                sk_training_output = numpy.ravel(sk_training_output)

            self._clf.fit(self._training_inputs, sk_training_output)
            self._fitted = True

        return CallResult(None)

In the ``fit`` method we select the training input and outputs using ``self._get_columns_to_fit`` and ``self._get_targets``.
TA2 systems can choose to use semantic types for filtering columns in input dataframe, this is set in the
Hyper-parameter ``use_semantic_types``. ``self._get_columns_to_fit`` and ``self._get_targets`` should check ``use_semantic_types``
and support both methods of selecting columns. Then we ``fit`` the primitive using the selected training inputs and outputs.

.. code:: python

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        sk_inputs, columns_to_use = self._get_columns_to_fit(inputs, self.hyperparams)
        output = []
        if len(sk_inputs.columns):

            sk_output = self._clf.predict(sk_inputs)

            if not self._fitted:
                raise PrimitiveNotFittedError("Primitive not fitted.")

            if sparse.issparse(sk_output):
                sk_output = pandas.DataFrame.sparse.from_spmatrix(sk_output)

            output = self._wrap_predictions(inputs, sk_output)
            output.columns = self._target_names
            output = [output]

        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                               add_index_columns=self.hyperparams['add_index_columns'],
                                               inputs=inputs, column_indices=self._target_column_indices,
                                               columns_list=output)

        return CallResult(outputs)

In the ``produce`` method we use our fitted model to predict the outputs. We then use ``self._wrap_predictions`` to add
metadata to the predicted output and add the target column names. Finally ``combine_columns`` will return the appropriate
``return_result`` and add the ``d3mIndex`` column. Produce methods and some other methods return results wrapped in CallResult.

.. code:: python

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                estimators_=None,
                classes_=None,
                .
                .
                .
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            estimators_=getattr(self._clf, 'estimators_', None),
            classes_=getattr(self._clf, 'classes_', None),
            .
            .
            .
            target_columns_metadata_=self._target_columns_metadata
        )

.. code:: python

    def set_params(self, *, params: Params) -> None:
        self._clf.estimators_ = params['estimators_']
        self._clf.classes_ = params['classes_']
        .
        .
        .
        self._target_columns_metadata = params['target_columns_metadata_']

        if params['estimators_'] is not None:
            self._fitted = True
        if params['classes_'] is not None:
            self._fitted = True
        .
        .
        .

An instance of the :mod:`d3m.metadata.params` subclass should be returned from primitive’s :meth:`~d3m.metadata.params.Params.get_params`
method, and accepted in :meth:`~d3m.metadata.params.Params.set_params`. All model attributes and custom parameters found
in :mod:`d3m.metadata.params` should also be in :meth:`~d3m.metadata.params.Params.get_params` and
:meth:`~d3m.metadata.params.Params.set_params`