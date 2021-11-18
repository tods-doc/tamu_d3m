.. _metadata:

Metadata
========

All values being passed between primitives have additional metadata associated with them to help primitives
make better sense of data. Metadata also serves as a way to pass additional information to other primitives.
Primitives themselves can be described with metadata as well. And pipelines, problem descriptions, and records of pipeline
runs are also seen as metadata.

Metadata is a core component of any data-based system. This repository
is standardizing how we represent metadata in the D3M program and
focusing on three types of metadata: \* metadata associated with
primitives \* metadata associated with datasets \* metadata associated
with values passed inside pipelines

This repository is also standardizing types of values being passed
between primitives in pipelines. While theoretically any value could be
passed between primitives, limiting them to a known set of values can
make primitives more compatible, efficient, and values easier to
introspect by TA3 systems.

Metadata
--------

:mod:`d3m.metadata.base` module provides a
standard Python implementation for metadata object.

When thinking about metadata, it is useful to keep in mind that metadata
can apply to different contexts:

* primitives
* values being passed
  between primitives, which we call containers (and are container types)
* datasets are a special case of a container
* to parts of data
  contained inside a container
* for example, a cell in a table can have
  its own metadata

Containers and their data can be seen as multi-dimensional structures.
Dimensions can have numeric (arrays) or string indexes (string to value
maps, i.e., dicts). Moreover, even numeric indexes can still have names
associated with each index value, e.g., column names in a table.

If a container type has a concept of *shape*
(:attr:`DataFrame.shape <pandas.DataFrame.shape>`, :attr:`ndarray.shape <numpy.ndarray.shape>`),
dimensions go in that order. For tabular data and existing container
types this means that the first dimension of a container is always
traversing samples (e.g., rows in a table), and the second dimension
columns.

Values can have nested other values and metadata dimensions go over all
of them until scalar values. So if a Pandas DataFrame contains
3-dimensional ndarrays, the whole value has 5 dimensions: two for rows
and columns of the DataFrame (even if there is only one column), and 3
for the array.

To tell to which part of data contained inside a container metadata
applies, we use a *selector*. Selector is a tuple of strings, integers,
or special values. Selector corresponds to a series of ``[...]`` item
getter Python operations on most values, except for Pandas DataFrame
where it corresponds to
:attr:`iloc <pandas.DataFrame.iloc>`
position-based selection.

Special selector values:

-  ``ALL_ELEMENTS`` – makes metadata apply to all elements in a given
   dimension (a wildcard)

Metadata itself is represented as a (potentially nested) dict. If
multiple metadata dicts comes from different selectors for the same
resolved selector location, they are merged together in the order from
least specific to more specific, later overriding earlier. ``null``
metadata value clears the key specified from a less specific selector.

Example
~~~~~~~

To better understand how metadata is attached to various parts of the
value, A `simple tabular D3M
dataset <https://gitlab.com/datadrivendiscovery/tests-data/tree/master/datasets/iris_dataset_1>`__
could be represented as a multi-dimensional structure:

.. code:: yaml

    {
      "0": [
        [0, 5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
        [1, 4.9, 3, 1.4, 0.2, "Iris-setosa"],
        ...
      ]
    }

It contains one resource with ID ``"0"`` which is the first dimension
(using strings as index; it is a map not an array), then rows, which is
the second dimension, and then columns, which is the third dimension.
The last two dimensions are numeric.

In Python, accessing third column of a second row would be
``["0"][1][2]`` which would be value ``3``. This is also the selector if
we would want to attach metadata to that cell. If this metadata is
description for this cell, we can thus describe this datum metadata as a
pair of a selector and a metadata dict:

-  selector: ``["0"][1][2]``
-  metadata:
   ``{"description": "Measured personally by Ronald Fisher."}``

Dataset-level metadata have empty selector:

-  selector: ``[]``
-  metadata: ``{"id": "iris_dataset_1", "name": "Iris Dataset"}``

To describe first dimension itself, we set ``dimension`` metadata on the
dataset-level (container). ``dimension`` describes the next dimension at
that location in the data structure.

-  selector: ``[]``
-  metadata: ``{"dimension": {"name": "resources", "length": 1}}``

This means that the full dataset-level metadata is now:

.. code:: json

    {
      "id": "iris_dataset_1",
      "name": "Iris Dataset",
      "dimension": {
        "name": "resources",
        "length": 1
      }
    }

To attach metadata to the first (and only) resource, we can do:

-  selector: ``["0"]``
-  metadata:
   ``{"structural_type": "pandas.core.frame.DataFrame", "dimension": {"length": 150, "name": "rows"}``

``dimension`` describes rows.

Columns dimension:

-  selector: ``["0"][ALL_ELEMENTS]``
-  metadata: ``{"dimension": {"length": 6, "name": "columns"}}``

Observe that there is no requirement that dimensions are aligned from
the perspective of metadata. But in this case they are, so we can use
``ALL_ELEMENTS`` wildcard to describe columns for all rows.

Third column metadata:

-  selector: ``["0"][ALL_ELEMENTS][2]``
-  metadata:
   ``{"name": "sepalWidth", "structural_type": "builtins.str", "semantic_types": ["http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute"]}``

Column names belong to each particular column and not all columns. Using
``name`` can serve to assign a string name to otherwise numeric
dimension.

We attach names and types to datums themselves and not dimensions.
Because we use ``ALL_ELEMENTS`` selector, this is internally stored
efficiently. We see traditional approach of storing this information in
the header of a column as a special case of a ``ALL_ELEMENTS`` selector.

Note that the name of a column belongs to the metadata because it is
just an alternative way to reference values in an otherwise numeric
dimension. This is different from a case where a dimension has
string-based index (a map/dict) where names of values are part of the
data structure at that dimension. Which approach is used depends on the
structure of the container for which metadata is attached to.

Default D3M dataset loader found in this package parses all tabular
values as strings and add semantic types, if known, for what could those
strings be representing (a float) and its role (an attribute). This
allows primitives later in a pipeline to convert them to proper
structural types but also allows additional analysis on original values
before such conversion is done.

Fetching all metadata for ``["0"][1][2]`` now returns:

.. code:: json

    {
      "name": "sepalWidth",
      "structural_type": "builtins.str",
      "semantic_types": [
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/Attribute"
      ],
      "description": "Measured personally by Ronald Fisher."
    }

.. _metadata_api:

API
~~~

:mod:`d3m.metadata.base` module provides two
classes which serve for storing metadata on values: :class:`~d3m.metadata.base.DataMetadata` for
data values, and :class:`~d3m.metadata.base.PrimitiveMetadata` for primitives. It also exposes a
:const:`~d3m.metadata.base.ALL_ELEMENTS` constant to be used in selectors.

You can see public methods available on classes documented in their
code. Some main ones are:

-  ``__init__(metadata)`` – constructs a new instance of the metadata
   class and optionally initializes it with top-level metadata
-  ``update(selector, metadata)`` – updates metadata at a given location
   in data structure identified by a selector
-  ``query(selector)`` – retrieves metadata at a given location
-  ``query_with_exceptions(selector)`` – retrieves metadata at a given
   location, but also returns metadata for selectors which have metadata
   which differs from that of ``ALL_ELEMENTS``
-  ``remove(selector)`` – removes metadata at a given location
-  ``get_elements(selector)`` – lists element names which exists at a
   given location
-  ``to_json()`` – converts metadata to a JSON representation
-  ``pretty_print()`` – pretty-print all metadata

``PrimitiveMetadata`` differs from ``DataMetadata`` that it does not
accept selector in its methods because there is no structure in
primitives.

Standard metadata keys
~~~~~~~~~~~~~~~~~~~~~~

You can use custom keys for metadata, but the following keys are
standardized, so you should use those if you are trying to represent the
same metadata:
https://metadata.datadrivendiscovery.org/schemas/v0/definitions.json

The same key always have the same meaning and we reuse the same key in
different contexts when we need the same meaning. So instead of having
both ``primitive_name`` and ``dataset_name`` we have just ``name``.

Different keys are expected in different contexts:

-  ``primitive`` –
   https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json
-  ``container`` –
   https://metadata.datadrivendiscovery.org/schemas/v0/container.json
-  ``data`` –
   https://metadata.datadrivendiscovery.org/schemas/v0/data.json

A more user friendly visualization of schemas listed above is available
at https://metadata.datadrivendiscovery.org/.

Contribute: Standardizing metadata schemas are an ongoing process. Feel
free to contribute suggestions and merge requests with improvements.

Data metadata
~~~~~~~~~~~~~

Every value passed around a pipeline has metadata associated with it.
Defined container types have an attribute ``metadata`` to contain it.
API available to manipulate metadata is still evolving because many
operations one can do on data are reasonable also on metadata (e.g.,
slicing and combining data). Currently, every operation on data clears
and re-initializes associated metadata.

    **Note:** While part of primitive's metadata is obtained
    automatically nothing like that is currently done for data metadata.
    This means one has to manually populate with dimension and typing
    information. This will be improved in the future with automatic
    extraction of this metadata from data.

.. _semantic_type:

Semantic Types
--------------

All that is standardized through `JSON schemas <https://metadata.datadrivendiscovery.org>`__.
In addition, we use semantic types and maintain a `list of commonly
used semantic types <https://metadata.datadrivendiscovery.org/types/>`__ in the program.
