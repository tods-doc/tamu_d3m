.. _static_files:

Primitive with Static Files
===========================

When building primitives that uses external/static files i.e. pre-trained weights, the
metadata for the primitive must be properly define such dependency.
The static file can be hosted anywhere based on your preference, as long as the URL to the file is a direct download link. It must
be public so that users of your primitive can access the file. Be sure to keep the URL available, as
the older version of the primitive could potentially start failing if URL stops resolving.

.. note::

    Full code of this section can be found in the `docs code repository <https://gitlab.com/datadrivendiscovery/docs-code>`__.

Below is a description of primitive metadata definition required, named ``_weights_configs`` for
each static file.

.. code:: python

    _weights_configs = [{
        'type': 'FILE',
        'key': '<Weight File Name>',
        'file_uri': '<URL to directly Download the Weight File>',
        'file_digest':'sha256sum of the <Weight File>',
    }]


This ``_weights_configs`` should be directly added to the ``INSTALLATION`` field of the primitive metadata.

.. code:: python

    from d3m.primitive_interfaces import base, transformer
    from d3m.metadata import base as metadata_base, hyperparams

    __all__ = ('ExampleTransform',)

    class ExampleTransform(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        """
        Docstring.
        """

        _weights_configs = [{
            'type': 'FILE',
            'key': '<Weight File Name>',
            'file_uri': '<URL to directly Download the Weight File>',
            'file_digest':'sha256sum of the <Weight File>',
        }]

        metadata = ...
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+<git-link-to-project>@{git_commit}#egg=<Package_name>'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }] + _weights_configs,
            ...

        ...

After the primitive metadata definition, it is important to include code to return the path of files.
An example is given as follows:

.. code:: python

    def _find_weights_path(self, key_filename):
        if key_filename in self.volumes:
            weight_file_path = self.volumes[key_filename]
        else:
            weight_file_path = os.path.join('.', self._weights_configs['file_digest'], key_filename)

        if not os.path.isfile(weight_file_path):
            raise ValueError(
                "Can't get weights file from volumes by key '{key_filename}' and at path '{path}'.".format(
                    key_filename=key_filename,
                    path=weight_file_path,
                ),
            )

        return weight_file_path

In this example code,  ``_find_weights_path`` method will try to find the static files from volumes based on weight file key.
If it cannot be found (e.g., runtime was not provided with static files), then it looks into the current directory.
The latter fallback is useful during development.

To run a pipeline with such primitive, you have to download static files and provide them to the runtime:

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
      /bin/bash -c "cd /mnt/d3m; \
        pip3 install -e .; \
        cd pipelines; \
        mkdir /static
        python3 -m d3m primitive download -p d3m.primitives.path.of.Primitive -o /static; \
        python3 -m d3m runtime --volumes /static fit-produce \
                --pipeline feature_pipeline.json \
                --problem /datasets/seed_datasets_current/22_handgeometry/TRAIN/problem_TRAIN/problemDoc.json \
                --input /datasets/seed_datasets_current/22_handgeometry/TRAIN/dataset_TRAIN/datasetDoc.json \
                --test-input /datasets/seed_datasets_current/22_handgeometry/TEST/dataset_TEST/datasetDoc.json \
                --output 22_handgeometry_results.csv \
                --output-run feature_pipeline_run.yml; \
        exit"

The static files will be downloaded and stored locally based on ``file_digest`` of ``_weights_configs``.
In this way we don't duplicate same files used by multiple primitives:

.. code:: shell

    mkdir /static
    python3 -m d3m primitive download -p d3m.primitives.path.of.Primitive -o /static

``-p`` optional argument to download static files for a particular primitive, matching on its Python path.
``-o`` optional argument to download the static files into a common folder. If not provided, they are
downloaded into the current directory.

After the download, the file structure is given as follows::

    /static/
      <file_digest>/
        <file>
      <file_digest>/
        <file>
      ...
      ...
