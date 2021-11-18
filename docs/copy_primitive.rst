Create a Primitive Based on an Existing Primitive
=================================================

In this how-to series, we will examine:
    * Where to find a primitive of a specific type
    * How to make a copy of that primitive
    * What to change in the copy to make a new primitive
    * Adding the new primitive to the D3M library
    * Demonstrate copying and modifying Transformer and Estimator primitives


Introduction
------------
Rather than writing a primitive from scratch, it is generally easier to start from an existing primitive that performs
a similar function. This how-to explains where to find a suitable primitive to use as a starting point and what is
required to create a new primitive from it.


See Also
--------
This section includes links on related reading that should be reviewed to help understand this topic
  * Primitive `families`_: Schema for the types of primitive families currently supported in D3M
  * `Primitive repository`_: Repository containing the D3M primitives.
  * Primitives repository `readme`_: Documentation on how to write, test and submit a primitive to the primitives
    repository
  * `Marvin`_: The Marvin query tool
  * `Primitive Good Citizen Checklist`_: Explains the do's and dont's of writing a primitive
  * How to write primitive `unit tests`_


Where to find existing Primitives
---------------------------------

Before making a copy of a primitive we must first identify a suitable candidate. There are many types of primitives that
are organized into `families`_. Once a suitable primitive family has been identified, existing primitives that belong to
this family can be explored in the `primitive repository`_ and/or by using `marvin`_.

Primitives Repository
---------------------

* It is recommended that the `primitive repository`_ be cloned to make search and navigation of the primitives easier.
  Note that there are tags that corresponds to official releases of the primitives. It is recommended that the
  latest official release tag is chosen for review. For up to date information on how to work with the primitives
  repo, see the `readme`_ which has a lot of excellent information relevant to this topic. The primitives repo can be
  cloned as follows:

::

    git clone https://gitlab.com/<fork>/primitives
    cd primitives
    git checkout v2020.12.1
    git lfs fetch

* In the latest official release of the primitives library, the primitives are organized into a separate folder for
  each contributor. For example, the folder primitives/SRI contains the primitives written by SRI. As such, to find
  all examples of existing primitives from a particular primitive family, grep and ls can be used as follows:

::

    ls -al  primitives/*/* | grep classification

    This command yields the following:

::

    primitives/BBN/d3m.primitives.classification.mlp.BBNMLPClassifier:
    primitives/CMU/d3m.primitives.classification.cover_tree.Fastlvm:
    primitives/CMU/d3m.primitives.classification.search.Find_projections:
    primitives/CMU/d3m.primitives.classification.search_hybrid.Find_projections:
    primitives/CMU/d3m.primitives.semisupervised_classification.iterative_labeling.AutonBox:
    ...

* Within the primitive family, 'classification' in the above example, we can see there are various algorithm `types`_
  such as 'mlp', and 'cover_tree'. Review these to narrow the type of primitive to be used to create the new
  primitive.

Marvin
------

`Marvin`_ is a browser based system that provides access to important D3M resources. The primitive query service
provides comprehensive filters for narrowing the view of primitives.

Once a suitable primitive has been identified, clone the primitives repo (see previous section)

The primitives are located in the 'primitives` folder. It is worth reviewing the details of how the folder
`structure`_ is arranged for each primitive.


How to make a copy of that Primitive
------------------------------------

* There are two primary components to a primitive, the code and the wrapper. The code for a primitive lives in
  a code library that is referred to by the wrapper code. The wrapper code, cloned in the previous step, lives in
  the `primitive repository`_.
  When writing a primitive, one generally starts with the code. If the code is structured properly and the required
  metadata fields are populated, we can automatically generate the primitive wrapper from it. To find where the code
  for a particular primitive resides, use the Marvin `primitives`_ tab which has a `link` for each primitives source
  code. Alternatively, look at the primitive.json for the chosen primitive. For example:

::

    vi primitives/common-primitives/d3m.primitives.classification.random_forest.Common/0.4.0/primitive.json

    Look for the "source" section in the json file:

::

    "source": {
        "name": "common-primitives",
        "contact": "mailto:zinkov@robots.ox.ac.uk",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/random_forest.py",
            "https://gitlab.com/datadrivendiscovery/common-primitives.git"
        ]
    },

   In this example, we see that the source of the primitive can be found in git in the common-primitives repository.
   Copy this python module to the repository of the new primitive and change the name of the module as appropriate.


What to change in the copy to make a new primitive
--------------------------------------------------

* The primitive source code has important sections that must be populated properly, for complete details see the
  tutorial on `primitive source code`_. Pay particular attention to the following sections that will likely need to
  be changed for the new primitive:

* `Input and Outputs`_
* ``Params``: These are the parameters that will be provided to the new primitives algorithm at run time
* `Hyperparameters`_ are used to tune the algorithms learning process
* `Primitive Metadata`_ (see also `primitive source code`_ for helpful information). All the primitive metadata
  section should be reviewed, here are the most important fields:
  * ``id``: see `primitive source code`_ for details on how to generate a valid new id.
    **NOTE - failure to generate a new id is a very common source of problems for primitive authors - please make sure this is completed!**
  * ``author``: update to reflect the author of the new primitive. This allows issue or questions associated with
    the primitive to go to the correct person.
  * ``version``: update to reflect the version of the new primitive
  * ``python_path``: update to the module for the new primitive
  * `keywords`_: These terms describe what this primitive is or does
  * ``source``: points to the location of the code. This will be used to tell the primitive wrapper where to find t9
    the underlying algorithm.
  * ``installation``: describes how the primitive implementation can be installed in the D3M environment
  * ``algorithm_types``: describes the type of algorithm implemented in the primitive
  * The code in the primitive that does the work of the algorithm will also need to be replaced. The primary methods
    that should be implemented are fit, continue_fit, produce, produce_feature_importances, get_params and set_params

Here is an example from a primitives source code:

.. code:: python

    __author__ = "CHANGE HERE"
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'CHANGE HERE',
        'version': 'CHANGE HERE',
        'name': 'CHANGE HERE',
        'keywords': [CHANGE HERE],
        'source': {
            'name': 'CHANGE HERE',
            'contact': 'CHANGE HERE',
            'uris': [
                'CHANGE HERE',
                'CHANGE HERE',
            ],
        },
        'installation': [{
            'type': CHANGE HERE,
            'package_uri': CHANGE HERE
        }],
        'python_path': 'CHANGE HERE',
        'algorithm_types': [CHANGE HERE],
        'primitive_family': CHANGE HERE,
    })

* Once the primitive code has been updated to reflect the new primitive, write some `unit tests`_ to ensure the
  primitive behaves as expected.


Adding the new Primitive to the D3M Library
-------------------------------------------

Once the new primitive is complete, generate the `primitive wrapper`_. Before the primitive can be used by Auto ML
systems that rely on the D3M environment, it must be added to the D3M library. The first step is to fork the
primitives repo, make a new branch, and add the new primitive to this branch. See the guide in `primitive wrapper`_ for
details on how and why forks are used.


Example use cases
-----------------

In this section we will apply the guidance outlined above for two sample primitive type Transformer and Estimator

Transformer Primitive
This section needs to be written

Estimator Primitive
This section needs to be written


.. _families: https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.primitive_family
.. _primitive repository: https://gitlab.com/datadrivendiscovery/primitives
.. _marvin: https://marvin.datadrivendiscovery.org/primitives?InterfacesVersion=%5B%22devel%22%5D
.. _primitives: https://marvin.datadrivendiscovery.org/primitives
.. _readme: https://gitlab.com/datadrivendiscovery/primitives/-/blob/master/README.md
.. _types: https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.algorithm_types
.. _structure: https://gitlab.com/datadrivendiscovery/primitives#structure-of-repository
.. _primitive source code: https://docs.datadrivendiscovery.org/devel/quickstart.html#primitive-source-code
.. _input and outputs: https://docs.datadrivendiscovery.org/devel/tutorial.html#input-output-types
.. _hyperparameters: https://docs.datadrivendiscovery.org/devel/metadata.html#hyperparameters)
.. _primitive metadata: https://docs.datadrivendiscovery.org/devel/tutorial.html#primitive-metadata
.. _Keywords: https://metadata.datadrivendiscovery.org/devel/?primitive&expanded#Primitive_metadata.keywords
.. _Primitive Good Citizen Checklist: https://docs.datadrivendiscovery.org/devel/primitive-checklist.html#primitive-good-citizen
.. _unit tests: https://docs.datadrivendiscovery.org/devel/quickstart.html#primitive-unit-tests
.. _primitive wrapper: https://gitlab.com/datadrivendiscovery/primitives/-/blob/master/README.md#adding-a-primitive
