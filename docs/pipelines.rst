.. _advanced_pipelines:

Advanced Pipelines
==================

TODO: Document that custom/additional fields are allowed (which are part of digest). Document _prefix fields (which are not part of digest).

TODO: Document sub-pipeline step. Document how data references for sub-pipelines are done.

TODO: Document placeholder step.

TODO: Document resolving of pipelines (by filename based on ID in the pipeline search path).

.. _interaction_with_problem:

Interaction with Problem Description
------------------------------------

TODO: Passing true targets and LUPI through semantic types from the problem description.

.. _container_types:

Container types
---------------

All input and output (container) values passed between primitives should
expose a ``Sequence``
`protocol <https://www.python.org/dev/peps/pep-0544/>`__ (sequence in
samples) and provide ``metadata`` attribute with metadata.

``d3m.container`` module exposes such standard types:

-  ``Dataset`` – a class representing datasets, including D3M datasets,
   implemented in
   :mod:`d3m.container.dataset` module
-  ``DataFrame`` –
   :class:`pandas.DataFrame`
   with support for ``metadata`` attribute, implemented in
   :mod:`d3m.container.pandas` module
-  ``ndarray`` –
   :class:`numpy.ndarray`
   with support for ``metadata`` attribute, implemented in
   :mod:`d3m.container.numpy` module
-  ``List`` – a standard :class:`list` with support for ``metadata``
   attribute, implemented in
   :mod:`d3m.container.list` module

``List`` can be used to create a simple list container.

It is strongly encouraged to use the :class:`~d3m.container.pandas.DataFrame` container type for
primitives which do not have strong reasons to use something else
(:class:`~d3m.container.dataset.Dataset`\ s to operate on initial pipeline input, or optimized
high-dimensional packed data in :class:`~numpy.ndarray`\ s, or :class:`list`\ s to pass as
values to hyper-parameters). This makes it easier to operate just on
columns without type casting while the data is being transformed to make
it useful for models.

When deciding which container type to use for inputs and outputs of a
primitive, consider as well where an expected place for your primitive
is in the pipeline. Generally, pipelines tend to have primitives
operating on :class:`~d3m.container.dataset.Dataset` at the beginning, then use :class:`~d3m.container.pandas.DataFrame` and
then convert to :class:`~numpy.ndarray`.

.. _data_types:

Data types
----------

Container types can contain values of the following types:

* container types themselves
* Python builtin primitive types:

  * ``str``
  * ``bytes``
  * ``bool``
  * ``float``
  * ``int``
  * ``dict`` (consider using :class:`typing.Dict`, :class:`typing.NamedTuple`, or :ref:`TypedDict <mypy:typeddict>`)
  * ``NoneType``

Placeholders
------------

Placeholders can be used to define pipeline templates to be used outside
of the metalearning context. A placeholder is replaced with a pipeline
step to form a pipeline. Restrictions of placeholders may apply on the
number of them, their position, allowed inputs and outputs, etc.

