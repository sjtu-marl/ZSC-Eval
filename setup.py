#!/usr/bin/env python

import os

import setuptools
from setuptools import setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("zsceval", "__init__.py")).read().split()
    return init[init.index("__version__") + 2][1:-1]


setup(
    name="zsceval",  # Replace with your own username
    version=get_version(),
    description="zero-shot coordination benchmark",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="multi-agent reinforcement learning platform pytorch",
    python_requires=">=3.6",
)
