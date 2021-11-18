Write Tests for Primitives
==========================

Once the primitives are constructed, unit testing must be done to see if the
primitive works as intended.

**Sample Setup**

.. code:: python

    import os
    import unittest

    from d3m.container import dataset
    from d3m.metadata import base as metadata_base
    from common_primitives import dataset_to_dataframe

    from example_primitive import ExampleTransformPrimitive


    class ExampleTransformTest(unittest.TestCase):
        def test_happy_path():
            # Load a dataset.
            # Datasets can be obtained from: https://datasets.datadrivendiscovery.org/d3m/datasets
            base_path = '../datasets/training_datasets/seed_datasets_archive/'
            dataset_doc_path = os.path.join(base_path, '38_sick_dataset', 'datasetDoc.json')
            dataset = dataset.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

            dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
            dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
            dataframe = dataframe_primitive.produce(inputs=dataset).value

            # Call example transformer.
            hyperparams_class = SampleTransform.metadata.get_hyperparams()
            primitive  = SampleTransform(hyperparams=hyperparams_class.defaults())
            test_out   = primitive.produce(inputs=dataframe).value

            # Write assertions to make sure that the output (type, shape, metadata) is what is expected.
            self.assertEqual(...)

            ...


    if __name__ == '__main__':
        unittest.main()

It is recommended to do the testing inside the D3M Docker container:

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
    cd /mnt/d3m/example_primitive
    python3 primitive_name_test.py
