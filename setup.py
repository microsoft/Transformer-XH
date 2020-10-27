# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from distutils.core import setup
from setuptools import setup
from setuptools import find_packages

# put package dependencies here
# this will make it easy to use
# i.e. run: ``python setup.py develop'' will automatically install depedencies stated in this file and add the trasformer-xh folder into your python path
# it will be easy to make this a pip install-able package as well





setup(name='transformer-xh',
      version='0.1',
      description='learn representations of docs and users from the information point of view',
      author='Chen Zhao, Chenyan Xiong, and Corby Rosset',
      author_email='Chenyan.Xiong@microsoft.com',
      packages=['transformer-xh'],
      url="https://github.com/microsoft/Transformer-XH",
      install_requires=[
          'nltk',
          'numpy',
          'scipy',
          'sklearn',
          'torch',
          'torchsummary',
          'dgl',
          'tqdm',
          'pytorch_transformers',
          'rapidfuzz'
      ], 
      )