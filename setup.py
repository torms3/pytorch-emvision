from setuptools import setup, find_packages


requirements = [
    'nose',
    'numpy',
    'torch',
]


setup(
    # Metadata
    name='emvision',
    version='0.0.1',
    author='Kisuk Lee',
    author_email='kisuklee@mit.edu',
    url='https://github.com/torms3/pytorch-emvision',
    description='3D EM models for torch deep learning',

    # Package info
    packages=find_packages(exclude=('test',)),

    # Test suite
    test_suite='nose.collector',

    zip_safe=True,
    install_requires=requirements,
)
