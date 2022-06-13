import sys

from setuptools import setup, find_packages

tf = 'tensorflow-macos' if sys.platform == 'darwin' else 'tensorflow'

setup(
    name='graph_ml',
    version='1.0',
    packages=find_packages(include=['graph_ml*']),
    install_requires=[f'{tf}>=2.0.0', 'numpy>=1.10.0'],
)
