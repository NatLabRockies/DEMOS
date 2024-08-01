# Install setuptools if not installed.
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup

# read README as the long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='demos',
    version='0.1dev',
    description='Scripts to run demos',
    long_description=long_description,
    author='NREL',
    author_email='@nrel.gov',
    url='https://github.com/NREL/DEMOS_NREL',
    classifiers=['Programming Language :: Python :: 3.8'],
)
