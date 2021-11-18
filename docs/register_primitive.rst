.. _register_primitive:

Register a Primitive
====================

New primitives can be added to D3M structure in two different ways: using the method
``d3m.index.register_primitive(primitive_path, primitive)`` or adding the primitive with a
d3m compatible entrypoint.

Registering a Primitive
-----------------------

For this example, the primitive
`RandomClassifierPrimitive <https://gitlab.com/datadrivendiscovery/tests-data/-/blob/master/primitives/test_primitives/random_classifier.py>`__ will be added to the index.

.. code:: python

    from d3m import index as d3m_index
    from test_primitives.random import RandomPrimitive

    # Register primitive in local d3m index.
    index.register_primitive('d3m.primitives.data_generation.random.Test', RandomPrimitive)

This method allows registering the primitive ``RandomPrimitive`` using the path locally
``d3m.primitives.data_generation.random.Test`` so it can be accessible from the d3m index.

Note: If the primitive is not installed, this code will need to be run at the beginning of
every execution so the primitive can be loaded in the index.

Installing a Primitive
----------------------

To install primitives, it is necessary to add entrypoints for every
primitive to be exposed in the ``setup.py``. An example is shown below-


.. code:: python

    entry_points = {
        'd3m.primitives': [
            'primitive_namespace.PrimitiveName = my_package.my_module:PrimitiveClassName',
        ],
    },

For a more elaborated example see `common primitives <https://gitlab.com/datadrivendiscovery/common-primitives>`__.

Primitives D3M Namespace
------------------------

The :mod:`d3m.primitives` module exposes all primitives under the same
``d3m.primitives`` namespace.

This is achieved using :ref:`Python entry points <setuptools:entry_points>`.
Python packages containing primitives should register them and expose
them under the common namespace by adding an entry like the following to
package's ``setup.py``:

.. code:: python

    entry_points = {
        'd3m.primitives': [
            'primitive_namespace.PrimitiveName = my_package.my_module:PrimitiveClassName',
        ],
    },

The example above would expose the
``my_package.my_module.PrimitiveClassName`` primitive under
``d3m.primitives.primitive_namespace.PrimitiveName``.

Configuring ``entry_points`` in your ``setup.py`` does not just put
primitives into a common namespace, but also helps with discovery of
your primitives on the system. Then your package with primitives just
have to be installed on the system and can be automatically discovered
and used by any other Python code.

    **Note:** Only primitive classes are available through the
    ``d3m.primitives`` namespace, no other symbols from a source
    module. In the example above, only ``PrimitiveClassName`` is
    available, not other symbols inside ``my_module`` (except if they
    are other classes also added to entry points).

    **Note:** Modules under ``d3m.primitives`` are created dynamically
    at run-time based on information from entry points. So some tools
    (IDEs, code inspectors, etc.) might not find them because there are
    no corresponding files and directories under ``d3m.primitives``
    module. You have to execute Python code for modules to be available.
    Static analysis cannot find them.

Primitives Discovery on PyPi
----------------------------

To facilitate automatic discovery of primitives on PyPi (or any other
compatible Python Package Index), publish a package with a keyword
``d3m_primitive`` in its ``setup.py`` configuration:

.. code:: python

    keywords='d3m_primitive'

    **Note:** Be careful when automatically discovering, installing, and
    using primitives from unknown sources. While primitives are designed
    to be bootstrapable and automatically installable without human
    involvement, there are no isolation mechanisms yet in place for
    running potentially malicious primitives. Currently recommended way
    is to use manually curated lists of known primitives.

See also the :mod:`d3m.index` module and its API.

Primitive Annotation
--------------------

Once primitive is constructed and unit testing is successful, the
final step in building a primitive is to generate the primitive annotation
which will be indexed and used by D3M.

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
    cd /mnt/d3m/example_primitive
    pip3 install -e .
    python3 -m d3m primitive describe -i 4 <primitive_name>

Alternatively, a `helper script <https://gitlab.com/datadrivendiscovery/docs-code/-/blob/master/quickstart_primitives/generate-primitive-json.py>`__
can be used to generate primitive annotations as well.
This can be more convenient when having to manage multiple primitives.
In this case, generating the primitive annotation is done as follows:

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
    cd /mnt/d3m/example_primitive
    pip3 install -e .
    python3 generate-primitive-json.py ...

Blocklist Primitives
====================
