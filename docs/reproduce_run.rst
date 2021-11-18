.. _reproduce_run:

Reproduce Pipeline Run
----------------------

TODO: Explain better how to do reproduction.

TODO: Describe how to reproduce a pipeline run from metalearning database.

TODO: Describe how to set Docker environment variables so that Docker image is recorded in pipeline run and how to then fetch exactly the same Docker image.

TODO: Document how to fetch a dataset from git repository based on dataset digest.

The reference runtime offers a way to pass an existing pipeline run file to a runtime command to allow it to be rerun.
Here is an example of this for the fit-produce call::

    $ python3 -m d3m runtime fit-produce -u pipeline_run.yml

Here is the guidance from the help menu::

      -u INPUT_RUN, --input-run INPUT_RUN
                        path to a pipeline run file with configuration, use
                        "-" for stdin
