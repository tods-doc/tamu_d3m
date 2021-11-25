import os
import os.path
import sys
from setuptools import setup, find_packages

PACKAGE_NAME = 'tamu_d3m'
MINIMUM_PYTHON_VERSION = 3, 7


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join('d3m', '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    raise KeyError("'{0}' not found in '{1}'".format(key, module_path))


def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()


def read_entry_points():
    with open('entry_points.ini') as entry_points:
        return entry_points.read()


check_python_version()
version = read_package_variable('__version__')
description = read_package_variable('__description__')
author = read_package_variable('__author__')

setup(
    name=PACKAGE_NAME,
    version=version,
    description=description,
    author=author,
    packages=find_packages(exclude=['contrib', 'docs', 'site', 'tests*']),
    package_data={PACKAGE_NAME: ['metadata/schemas/*/*.json', 'contrib/pipelines/*.yml', 'py.typed']},
    install_requires=[
        'scikit-learn>=0.21.3,<=0.24.2',
        'pytypes>=1.0b5',
        'frozendict==1.2',
        'numpy>=1.16.6,<=1.21.2',
        'jsonschema>=3.0.2,<=4.0.1',
        'requests>=2.19.1,<=2.26.0',
        'rfc3339-validator>=0.1,<0.2',
        'rfc3986-validator>=0.1,<0.2',
        'webcolors>=1.8.1,<=1.11.1',
        'dateparser>=0.7.0,<=1.1.0',
        'python-dateutil>=2.8.1,<=2.8.2',
        'pandas>=1.1.3,<=1.3.4',
        'typing-inspect==0.7.1',
        'GitPython>=3.1.0,<=3.1.24',
        'jsonpath-ng>=1.4.3,<=1.5.3',
        'custom-inherit>=2.2.0,<=2.3.2',
        'PyYAML>=5.1,<=5.4.1',
        'gputil>=1.3.0,<=1.4.0',
        'pyrsistent>=0.14.11,<=0.18.0',
        'scipy>=1.2.1,<=1.7.1',
        'openml==0.11.0',
    ],
    extras_require={
        'tests': [
            'asv==0.4.2',
            'docker[tls]==2.7',
            'pypiwin32==220 ; sys_platform=="win32"',
        ],
    },
    entry_points=read_entry_points(),
    url='https://github.com/tods-doc/tamu_d3m',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    classifiers=[
          'License :: OSI Approved :: Apache Software License',
    ],
    zip_safe=False,
)
