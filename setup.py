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
    name='demos_urbansim',
    version='0.1dev',
    description='Scripts to run demos & urbansim',
    long_description=long_description,
    author='UrbanSim Inc.',
    author_email='info@urbansim.com',
    url='https://github.com/urbansim/DEMOS_URBANSIM',
    classifiers=['Programming Language :: Python :: 3.6'],
)
