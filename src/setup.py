import io
from setuptools import setup, find_packages

from bci import __version__

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('README.rst')
requirements = read('requirements.txt')


setup(
    # metadata
    name='bci',
    version=__version__,
    license='MIT',
    author='Intelligent Systems',
    author_email="mlalgorithms@gmail.com",
    description='lib for signal decoding, python package',
    long_description=readme,
    url='https://github.com/intsystems/bci',

    # options
    packages=find_packages(),
    install_requires=requirements,
)
