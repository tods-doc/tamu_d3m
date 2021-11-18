Advanced Primitives
===================

TODO: Describe how to define a new hyper-parameters class.

Primitive Interfaces
--------------------

A collection of standard Python interfaces for TA1 primitives. All
primitives should extend one of the base classes available and
optionally implement available mixins.

There are a variety of :mod:`primitive interfaces/classes <d3m.primitive_interfaces>` available. As an example,
a primitive doing just attribute extraction without requiring any fitting, a :class:`~d3m.primitive_interfaces.transformer.TransformerPrimitiveBase`
from :mod:`~d3m.primitive_interfaces.transformer` module can be used.

Each primitives can have it's own :mod:`hyper-parameters <d3m.metadata.hyperparams>`. Some example hyper-parameter types one can use to describe
primitive's hyper-parameters are: :class:`~d3m.metadata.hyperparams.Constant`, :class:`~d3m.metadata.hyperparams.UniformBool`,
:class:`~d3m.metadata.hyperparams.UniformInt`, :class:`~d3m.metadata.hyperparams.Choice`, :class:`~d3m.metadata.hyperparams.List`.

Also, each hyper-parameter should be defined as one or more of the four :ref:`hyper-parameter semantic types <hyperparameters>`:

* `https://metadata.datadrivendiscovery.org/types/TuningParameter <https://metadata.datadrivendiscovery.org/types/TuningParameter>`__
* `https://metadata.datadrivendiscovery.org/types/ControlParameter <https://metadata.datadrivendiscovery.org/types/ControlParameter>`__
* `https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter <https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter>`__
* `https://metadata.datadrivendiscovery.org/types/MetafeatureParameter <https://metadata.datadrivendiscovery.org/types/MetafeatureParameter>`__

Example
~~~~~~~

.. code:: python

    from d3m.primitive_interfaces import base, transformer
    from d3m.metadata import base as metadata_base, hyperparams

    __all__ = ('ExampleTransformPrimitive',)


    class Hyperparams(hyperparams.Hyperparams):
        learning_rate = hyperparams.Uniform(lower=0.0, upper=1.0, default=0.001, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ])
        clusters = hyperparams.UniformInt(lower=1, upper=100, default=10, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ])


    class ExampleTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        """
        The docstring is very important and must to be included. It should contain
        relevant information about the hyper-parameters, primitive functionality, etc.
        """

        def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
            pass

Design Principles
~~~~~~~~~~~~~~~~~

Standard TA1 primitive interfaces have been designed to be possible for
TA2 systems to call primitives automatically and combine them into
pipelines.

Some design principles applied:

-  Use of a de facto standard language for "glue" between different
   components and libraries, Python.
-  Use of keyword-only arguments for all methods so that caller does not
   have to worry about the order of arguments.
-  Every primitive should implement only one functionality, more or less
   a function, with clear inputs and outputs. All parameters of the
   function do not have to be known in advance and function can be
   "fitted" as part of the training step of the pipeline.
-  Use of Python 3 typing extensions to annotate methods and classes
   with typing information to make it easier for TA2 systems to prune
   incompatible combinations of inputs and outputs and to reuse existing
   Python type-checking tooling.
-  Typing information can serve both detecting issues and
   incompatibilities in primitive implementations and help with pipeline
   construction.
-  All values being passed through a primitive have metadata associated
   with them.
-  Primitives can operate only at a metadata level to help guide the
   pipeline construction process without having to operate on data
   itself.
-  Primitive metadata is close to the source, primitive code, and not in
   separate files to minimize chances that it is goes out of sync.
   Metadata which can be automatically determined from the code should
   be automatically determined from the code. Similarly for data
   metadata.
-  All randomness of primitives is captured by a random seed argument to
   assure reproducibility.
-  Operations can work in iterations, under time budgets, and caller
   might not always want to compute values fully.
-  Through use of mixins primitives can signal which capabilities they
   support.
-  Primitives are to be composed and executed in a data-flow manner.

Main Concepts
~~~~~~~~~~~~~

Interface classes, mixins, and methods are documented in detail through
use of docstrings and typing annotations. Here we note some higher-level
concept which can help understand basic ideas behind interfaces and what
they are trying to achieve, the big picture. This section is not
normative.

A primitive should extend one of the base classes available and
optionally mixins as well. Not all mixins apply to all primitives. That
being said, you probably do not want to subclass ``PrimitiveBase``
directly, but instead one of other base classes to signal to a caller
more about what your primitive is doing. If your primitive belong to a
larger set of primitives no exiting non-\ ``PrimitiveBase`` base class
suits well, consider suggesting that a new base class is created by
opening an issue or making a merge request.

Base class and mixins have generally four type arguments you have to
provide: ``Inputs``, ``Outpus``, ``Params``, and ``Hyperparams``. One
can see a primitive as parameterized by those four type arguments. You
can access them at runtime through metadata:

.. code:: python

    FooBarPrimitive.metadata.query()['class_type_arguments']

``Inputs`` should be set to a primary input type of a primitive.
Primary, because you can define additional inputs your primitive might
need, but we will go into these details later. Similarly for
``Outputs``. ``produce`` method then produces outputs from inputs. Other
primitive methods help the primitive (and its ``produce`` method)
achieve that, or help the runtime execute the primitive as a whole, or
optimize its behavior.

Both ``Inputs`` and ``Outputs`` should be of a
:ref:`container_types`. We allow a limited set of value types being
passed between primitives so that both TA2 and TA3 systems can
implement introspection for those values if needed, or user interface
for them, etc. Moreover this allows us also to assure that they can be
efficiently used with Arrow/Plasma store.

Container values can then in turn contain values of an :ref:`extended but
still limited set of data types <data_types>`.

Those values being passed between primitives also hold metadata.
Metadata is available on their ``metadata`` attribute. Metadata on
values is stored in an instance of
:class:`~d3m.metadata.base.DataMetadata` class. This is a
reason why we have :ref:`our own versions of some standard container
types <container_types>`: to have the ``metadata`` attribute.

All metadata is immutable and updating a metadata object returns a new,
updated, copy. Metadata internally remembers the history of changes, but
there is no API yet to access that. But the idea is that you will be
able to follow the whole history of change to data in a pipeline through
metadata. See :ref:`metadata API <metadata_api>` for more information
how to manipulate metadata.

Primitives have a similar class ``PrimitiveMetadata``, which when
created automatically analyses its primitive and populates parts of
metadata based on that. In this way author does not have to have
information in two places (metadata and code) but just in code and
metadata is extracted from it. When possible. Some metadata author of
the primitive stil has to provide directly.

Currently most standard interface base classes have only one ``produce``
method, but design allows for multiple: their name has to be prefixed
with ``produce_``, have similar arguments and same semantics as all
produce methods. The main motivation for this is that some primitives
might be able to expose same results in different ways. Having multiple
produce methods allow the caller to pick which type of the result they
want.

To keep primitive from outside simple and allow easier compositionality
in pipelines, primitives have arguments defined per primitive and not
per their method. The idea here is that once a caller satisfies
(computes a value to be passed to) an argument, any method which
requires that argument can be called on a primitive.

There are three types of arguments:

-  pipeline – arguments which are provided by the pipeline, they are
   required (otherwise caller would be able to trivially satisfy them by
   always passing ``None`` or another default value)
-  runtime – arguments which caller provides during pipeline execution
   and they control various aspects of the execution
-  hyper-parameter – a method can declare that primitive's
   hyper-parameter can be overridden for the call of the method, they
   have to match hyper-parameter definition

Methods can accept additional pipeline and hyper-parameter arguments and
not just those from the standard interfaces.

Produce methods and some other methods return results wrapped in
``CallResult``. In this way primitives can expose information about
internal iterative or optimization process and allow caller to decide
how long to run.

When calling a primitive, to access ``Hyperparams`` class you can do:

.. code:: python

    hyperparams_class = FooBarPrimitive.metadata.query()['class_type_arguments']['Hyperparams']

You can now create an instance of the class by directly providing values
for hyper-parameters, use available simple sampling, or just use default
values:

.. code:: python

    hp1 = hyperparams_class({'threshold': 0.01})
    hp2 = hyperparams_class.sample(random_state=42)
    hp3 = hyperparams_class.defaults

You can then pass those instances as the ``hyperparams`` argument to
primitive's constructor.

Author of a primitive has to define what internal parameters does the
primitive have, if any, by extending the ``Params`` class. It is just a
fancy dict, so you can both create an instance of it in the same way,
and access its values:

.. code:: python

    class Params(params.Params):
        coefficients: numpy.ndarray

    ps = Params({'coefficients': numpy.array[1, 2, 3]})
    ps['coefficients']

``Hyperparams`` class and ``Params`` class have to be pickable and
copyable so that instances of primitives can be serialized and restored
as needed.

Primitives (and some other values) are uniquely identified by their ID
and version. ID does not change through versions.

Primitives should not modify in-place any input argument but always
first make a copy before any modification.

Checklist for Creating a New Primitive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Implement as many interfaces as are applicable to your
   primitive. An up-to-date list of mixins you can implement can be
   found at
   <https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/primitive_interfaces/base.py>

2. Create unit tests to test all methods you implement

3. Include all relevant hyperparameters and use appropriate
   ``Hyperparameter`` subclass for specifying the range of values a
   hyperparameter can take. Try to provide good default values where
   possible. Also include all relevant ``semantic_types``
   <https://metadata.datadrivendiscovery.org/types/>

4. Include ``metadata`` and ``__author__`` fields in your class
   definition. The ``__author__`` field should include a name or team
   as well as email. The ``metadata`` object has many fields which should
   be filled in:

   * id, this is a uuid unique to this primitive. It can be generated with :code:`import uuid; uuid.uuid4()`
   * version
   * python_path, the name you want to be import this primitive through
   * keywords, keywords you want your primitive to be discovered by
   * installation, how to install the package which has this primitive. This is easiest if this is just a python package on PyPI
   * algorithm_types, specify which PrimitiveAlgorithmType the algorithm is, a complete list can be found in TODO
   * primitive_family, specify the broad family a primitive falls under, a complete list can be found in TODO
   * hyperparameters_to_tune, specify which hyperparameters you would prefer a TA2 system tune

5. Make sure primitive uses the correct container type

6. If container type is a dataframe, specify which column is the
   target value, which columns are the input values, and which columns
   are the output values.

7. Create an example pipeline which includes this primitive and uses one of the seed datasets as input.

Examples
~~~~~~~~

Examples of simple primitives using these interfaces can be found `in
this
repository <https://gitlab.com/datadrivendiscovery/tests-data/tree/master/primitives>`__:

-  `MonomialPrimitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py>`__
   is a simple regressor which shows how to use ``container.List``,
   define and use ``Params`` and ``Hyperparams``, and implement multiple
   methods needed by a supervised learner primitive
-  `IncrementPrimitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/increment.py>`__
   is a transformer and shows how to have ``container.ndarray`` as
   inputs and outputs, and how to set metadata for outputs
-  `SumPrimitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py>`__
   is a transformer as well, but it is just a wrapper around a Docker
   image, it shows how to define Docker image in metadata and how to
   connect to a running Docker container, moreover, it also shows how
   inputs can be a union type of multiple other types
-  `RandomPrimitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/random.py>`__
   is a generator which shows how to use ``random_seed``, too.

High-level Primitives Base Classes
----------------------------------

High-level primitives base classes provides tools to the developers
to easily create new primitives by abstracting some unnecessary and
repetitive work.

``FileReaderPrimitiveBase``:  A primitive base class for reading files referenced in columns.

``DatasetSplitPrimitiveBase``: A base class for primitives which fit on a
``Dataset`` object to produce splits of that ``Dataset`` when producing.

``TabularSplitPrimitiveBase``: A primitive base class for splitting tabular datasets.

Examples
~~~~~~~~

Examples of primitives using these base classes can be found `in
this
repository <https://gitlab.com/datadrivendiscovery/common-primitives/-/tree/master/common_primitives>`__:

-  `DataFrameImageReaderPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/dataframe_image_reader.py>`__
    A primitive which reads columns referencing image files.
-  `FixedSplitDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/fixed_split.py>`__
   A primitive which splits a tabular Dataset in a way that uses for the test
   (score) split a fixed list of primary index values or row indices of the main
   resource to be used. All other rows are added used for the train split.
-  `KFoldDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/kfold_split.py>`__
   A primitive which splits a tabular Dataset for k-fold cross-validation.
-  `KFoldTimeSeriesSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/kfold_split_timeseries.py>`__
   A primitive which splits a tabular time-series Dataset for k-fold cross-validation.
-  `NoSplitDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/no_split.py>`__
   A primitive which splits a tabular Dataset in a way that for all splits it
   produces the same (full) Dataset.
-  `TrainScoreDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/train_score_split.py>`__
   A primitive which splits a tabular Dataset into random train and score subsets.

.. _parameters:

Parameters
----------

A base class to be subclassed and used as a type for :class:`~d3m.metadata.params.Params` type
argument in primitive interfaces can be found in the
:mod:`d3m.metadata.params` module. An
instance of this subclass should be returned from primitive's
:meth:`~d3m.metadata.params.Params.get_params` method, and accepted in :meth:`~d3m.metadata.params.Params.set_params`.

To define parameters a primitive has you should subclass this base class
and define parameters as class attributes with type annotations.
Example:

.. code:: python

    import numpy
    from d3m.metadata import params

    class Params(params.Params):
        weights: numpy.ndarray
        bias: float

:class:`~d3m.metadata.params.Params` class is just a fancy Python dict which checks types of
parameters and requires all of them to be set. You can create it like:

.. code:: python

    ps = Params({'weights': weights, 'bias': 0.1})
    ps['bias']

::

    0.01

``weights`` and ``bias`` do not exist as an attributes on the class or
instance. In the class definition, they are just type annotations to
configure which parameters are there.

    **Note:** :class:`~d3m.metadata.params.Params` class uses ``parameter_name: type`` syntax
    while :class:`~d3m.metadata.hyperparams.Hyperparams` class uses
    ``hyperparameter_name = Descriptor(...)`` syntax. Do not confuse
    them.

.. _primitive_metadata:

Primitive Metadata
------------------

It is very crucial to define :ref:`primitive metadata <primitive_metadata>` for the primitive properly.
Primitive metadata can be used by TA2 systems to metalearn about primitives and in general decide which primitive to use when.

Example
~~~~~~~

.. code:: python

    from d3m.primitive_interfaces import base, transformer
    from d3m.metadata import base as metadata_base, hyperparams

    __all__ = ('ExampleTransformPrimitive',)

    class ExampleTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        """
        Docstring.
        """

        metadata = metadata_base.PrimitiveMetadata({
            'id': <Unique-ID, generated using UUID>,
            'version': <Primitive-development-version>,
            'name': <Primitive-Name>,
            'python_path': 'd3m.primitives.<>.<>.<>' # Must match path in setup.py,
            'source': {
                'name': <Project-maintainer-name>,
                'uris': [<GitHub-link-to-project>],
                'contact': 'mailto:<Author E-Mail>'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+<git-link-to-project>@{git_commit}#egg=<Package_name>'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                # Check https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.algorithm_types for all available algorithm types.
                # If algorithm type s not available a Merge Request should be made to add it to core package.
                metadata_base.PrimitiveAlgorithmType.<Choose-the-algorithm-type-that-best-describes-the-primitive>,
            ],
            # Check https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.primitive_family for all available primitive family types.
            # If primitive family is not available a Merge Request should be made to add it to core package.
            'primitive_family': metadata_base.PrimitiveFamily.<Choose-the-primitive-family-that-closely-associates-to-the-primitive>
        })

        ...

Metadata
~~~~~~~~

Part of primitive metadata can be automatically obtained from
primitive's code, some can be computed through evaluation of primitives,
but some has to be provided by primitive's author. Details of which
metadata is currently standardized and what values are possible can be
found in primitive's JSON schema. This section describes author's
metadata into more detail. Example of primitive's metadata provided by
an author from `Monomial test
primitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py#L32>`__,
slightly modified:

.. code:: python

    metadata = metadata_module.PrimitiveMetadata({
        'id': '4a0336ae-63b9-4a42-860e-86c5b64afbdd',
        'version': '0.1.0',
        'name': "Monomial Regressor",
        'keywords': ['test primitive'],
        'source': {
            'name': 'Test team',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.test.MonomialPrimitive',
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.LINEAR_REGRESSION,
        ],
        'primitive_family': metadata_module.PrimitiveFamily.REGRESSION,
    })

-  Primitive's metadata provided by an author is defined as a class
   attribute and instance of :class:`~d3m.metadata.base.PrimitiveMetadata`.
-  When class is defined, class is automatically analyzed and metadata
   is extended with automatically obtained values from class code.
-  ``id`` can be simply generated using :func:`uuid.uuid4` in Python and
   should never change. **Do not reuse IDs and do not use the ID from
   this example.**
-  When primitive's code changes you should update the version, a `PEP
   440 <https://www.python.org/dev/peps/pep-0440/>`__ compatible one.
   Consider updating a version every time you change code, potentially
   using `semantic versioning <https://semver.org/>`__, but nothing of
   this is enforced.
-  ``name`` is a human-friendly name of the primitive.
-  ``keywords`` can be anything you want to convey to users of the
   primitive and which could help with primitive's discovery.
-  ``source`` describes where the primitive is coming from. The required
   value is ``name`` to tell information about the author, but you might
   be interested also in ``contact`` where you can put an e-mail like
   ``mailto:author@example.com`` as a way to contact the author.
   ``uris`` can be anything. In above, one points to the code in GitLab,
   and another to the repo. If there is a website for the primitive, you
   might want to add it here as well. These URIs are not really meant
   for automatic consumption but are more as a reference. See
   ``location_uris`` for URIs to the code.
-  ``installation`` is important because it describes how can your
   primitive be automatically installed. Entries are installed in order
   and currently the following types of entries are supported:
-  A ``PIP`` package available on PyPI or some other package registry:

   ::

       ```
       {
         'type': metadata_module.PrimitiveInstallationType.PIP,
         'package': 'my-primitive-package',
         'version': '0.1.0',
       }
       ```

-  A ``PIP`` package available at some URI. If this is a git repository,
   then an exact git hash and ``egg`` name should be provided. ``egg``
   name should match the package name installed. Because here we have a
   chicken and an egg problem: how can one commit a hash of code version
   if this changes the hash, you can use a helper utility function to
   provide you with a hash automatically at runtime. ``subdirectory``
   part of the URI suffix is not necessary and is here just because this
   particular primitive happens to reside in a subdirectory of the
   repository.
-  A ``DOCKER`` image which should run while the primitive is operating.
   Starting and stopping of a Docker container is managed by a caller,
   which passes information about running container through primitive's
   ``docker_containers`` ``__init__`` argument. The argument is a
   mapping between the ``key`` value and address and ports at which the
   running container is available. See `Sum test
   primitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py#L66>`__
   for an example:

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.DOCKER,
           'key': 'summing',
           'image_name': 'registry.gitlab.com/datadrivendiscovery/tests-data/summing',
           'image_digest': 'sha256:07db5fef262c1172de5c1db5334944b2f58a679e4bb9ea6232234d71239deb64',
       }
       ```

-  A ``UBUNTU`` entry can be used to describe a system library or
   package required for installation or operation of your primitive. If
   your other dependencies require a system library to be installed
   before they can be installed, list this entry before them in
   ``installation`` list.

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.UBUNTU,
           'package': 'ffmpeg',
           'version': '7:3.3.4-2',
       }
       ```

-  A ``FILE`` entry allows a primitive to specify a static file
   dependency which should be provided by a caller to a primitive.
   Caller passes information about the file path of downloaded file
   through primitive's ``volumes`` ``__init__`` argument. The argument
   is a mapping between the ``key`` value and file path. The filename
   portion of the provided path does not necessary match the filename
   portion of the file's URI.

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.FILE,
           'key': 'model',
           'file_uri': 'http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/googlenet_finetune_web_car_iter_10000.caffemodel',
           'file_digest': '6bdf72f703a504cd02d7c3efc6c67cbbaf506e1cbd9530937db6a698b330242e',
       }
       ```

-  A ``TGZ`` entry allows a primitive to specify a static directory
   dependency which should be provided by a caller to a primitive.
   Caller passes information about the directory path of downloaded and
   extracted file through primitive's ``volumes`` ``__init__`` argument.
   The argument is a mapping between the ``key`` value and directory
   path.

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.TGZ,
           'key': 'mails',
           'file_uri': 'https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz',
           'file_digest': 'b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7',
       }
       ```

-  If you can provide, ``location_uris`` points to an exact code used by
   the primitive. This can be obtained through installing a primitive,
   but it can be helpful to have an online resource as well.
-  ``python_path`` is a path under which the primitive will get mapped
   through ``setup.py`` entry points. This is very important to keep in
   sync.
-  ``algorithm_types`` and ``primitive_family`` help with discovery of a
   primitive. They are required and if suitable values are not available
   for you, make a merge request and propose new values. As you see in
   the code here and in ``installation`` entries, you can use directly
   Python enumerations to populate these values.

Some other metadata you might be interested to provide to help callers
use your primitive better are ``preconditions`` (what preconditions
should exist on data for primitive to operate well), ``effects`` (what
changes does a primitive do to data), and a ``hyperparams_to_tune`` hint
to help callers know which hyper-parameters are most important to focus
on.

Primitive metadata also includes descriptions of a primitive and its
methods. These descriptions are automatically obtained from primitive's
docstrings. Docstrings should be made according to :ref:`numpy docstring
format <numpydoc:format>`
(`examples <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__).

.. _primitive_family:

Primitive Family
~~~~~~~~~~~~~~~~

As mentioned above, ``primitive_family`` is a required value which helps with
the discovery of a primitive. Hence, it is important to select the correct
``primitive_family`` when describing your primitive.

-  List of all supported ``primitive_family`` values
   `https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.primitive_family <https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.primitive_family>`__

Often there is confusion between ``DATA_CLEANING, ``DATA_PREPROCESSING``, and ``DATA_TRANSFORMATION``.
Here is a quick cheat-sheat for these primitive families:

-  ``DATA_TRANSFORMATION``: Primitives which affect type casting,
   dimension/structure changes (i.e., changing columns), semantic type changes, data encoders, or
   file readers. In short, if it changes type of data or structure
   of data, it is a transformation.

-  ``DATA_CLEANING``: Primitives which impute, normalize, filter rows, or remove outliers.
   In short, if it improves on existing data values, but not structure, it
   is data cleaning.

-  ``FEATURE_EXTRACTION``: Primitives which takes initial data and builds a set
   of derived values/features, these include component analysis and vectorizers.

If there is a primitive family not in the list you are welcome to suggest adding it.

.. _input_output_types:

Input/Output types
------------------

The acceptable inputs/outputs of a primitive must be pre-defined. D3M supports a variety of
standard input/output :ref:`container types <container_types>` such as:

- ``pandas.DataFrame`` (as :class:`d3m.container.pandas.DataFrame`)

- ``numpy.ndarray`` (as :class:`d3m.container.numpy.ndarray`)

- ``list`` (as :class:`d3m.container.list.List`)

.. note::
    Even thought D3M container types behave mostly as standard types, the D3M container types must be used for inputs/outputs, because D3M container types support D3M metadata.

Example
~~~~~~~

.. code:: python

    from d3m import container

    Inputs  = container.DataFrame
    Outputs = container.DataFrame


    class ExampleTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        ...

.. note::
    When returning the output DataFrame, its metadata should be updated with the correct semantic and structural types.

Example
~~~~~~~

.. code:: python

    # Update metadata for each DataFrame column.
    for column_index in range(outputs.shape[1]):
        column_metadata = {}
        column_metadata['structural_type'] = type(1.0)
        column_metadata['name'] = "column {i}".format(i=column_index)
        column_metadata["semantic_types"] = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute",)
        outputs.metadata = outputs.metadata.update((metadata_base.ALL_ELEMENTS, column_index), column_metadata)
