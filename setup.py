from setuptools import setup, Extension
import setuptools
import os
import sys


setup(
    name='EpiCommute',
    version="0.0.1",
    author="Frank Schlosser",
    author_email="frankfschlosser@gmail.com",
    url='https://github.com/franksh/EpiCommute',
    license="MIT",
    description="Simulate an epidemic on a metapopulation network with commuter-type mobility.",
    long_description='',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
                'networkx>=2.2',
                'numpy>=1.16',
                'scipy>=1.3.1',
    ],
    tests_require=[],
    setup_requires=[],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'
                 ],
    project_urls={
        'Source': 'https://github.com/franksh/EpiCommute',
        'Bug Reports': 'https://github.com/franksh/EpiCommute/issues',
    },
    include_package_data=True,
    zip_safe=False,
)
