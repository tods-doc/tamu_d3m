.. _hyperparameters:

Hyper-parameters
================

A base class for hyper-parameters description for primitives can be
found in the
:mod:`d3m.metadata.hyperparams` module.

To define a hyper-parameters space you should subclass this base class
and define hyper-parameters as class attributes. Example:

.. code:: python

    from d3m.metadata import hyperparams

    class Hyperparams(hyperparams.Hyperparams):
        learning_rate = hyperparams.Uniform(lower=0.0, upper=1.0, default=0.001, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ])
        clusters = hyperparams.UniformInt(lower=1, upper=100, default=10, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ])

To access hyper-parameters space configuration, you can now call:

.. code:: python

    Hyperparams.configuration

::

    OrderedDict([('learning_rate', Uniform(lower=0.0, upper=1.0, q=None, default=0.001)), ('clusters', UniformInt(lower=1, upper=100, default=10))])

To get a random sample of all hyper-parameters, call:

.. code:: python

    hp1 = Hyperparams.sample(random_state=42)

::

    Hyperparams({'learning_rate': 0.3745401188473625, 'clusters': 93})

To get an instance with all default values:

.. code:: python

    hp2 = Hyperparams.defaults()

::

    Hyperparams({'learning_rate': 0.001, 'clusters': 10})

:class:`~d3m.metadata.hyperparams.Hyperparams` class is just a fancy read-only Python dict. You can
also manually create its instance:

.. code:: python

    hp3 = Hyperparams({'learning_rate': 0.01, 'clusters': 20})
    hp3['learning_rate']

::

    0.01

If you want to use most of default values, but set some, you can thus
use this dict-construction approach:

.. code:: python

    hp4 = Hyperparams(Hyperparams.defaults(), clusters=30)

::

    Hyperparams({'learning_rate': 0.001, 'clusters': 30})

There is no class- or instance-level attribute ``learning_rate`` or
``clusters``. In the class definition, they were used only for defining
the hyper-parameters space, but those attributes were extracted out and
put into ``configuration`` attribute.

There are four types of hyper-parameters:

* tuning parameters which
  should be tuned during hyper-parameter optimization phase
* control
  parameters which should be determined during pipeline construction phase
  and are part of the logic of the pipeline
* parameters which control the use of resources by the primitive
* parameters which control which meta-features are computed by the primitive

You can use hyper-parameter's semantic type to differentiate between
those types of hyper-parameters using the following URIs:

* https://metadata.datadrivendiscovery.org/types/TuningParameter
* https://metadata.datadrivendiscovery.org/types/ControlParameter
* https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter
* https://metadata.datadrivendiscovery.org/types/MetafeatureParameter

Once you define a :class:`~d3m.metadata.hyperparams.Hyperparams` class for your primitive you can pass
it as a class type argument in your primitive's class definition:

.. code:: python

    class MyPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
        ...

Those class type arguments are then automatically extracted from the
class definition and made part of primitive's metadata. This allows the
caller to access the :class:`~d3m.metadata.hyperparams.Hyperparams` class to crete an instance to pass
to primitive's constructor:

.. code:: python

    hyperparams_class = MyPrimitive.metadata.get_hyperparams()
    primitive = MyPrimitive(hyperparams=hyperparams_class.defaults())

.. note::

    :class:`~d3m.metadata.hyperparams.Hyperparams` class uses
    ``hyperparameter_name = Descriptor(...)`` syntax while :class:`~d3m.metadata.params.Params`
    class uses ``parameter_name: type`` syntax. Do not confuse them.
