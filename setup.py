# coding: utf-8
NAME = "fastpork"  # name of package directory

import sys
try:
    from setuptools import setup
except ImportError:
    print(f"Please install setuptools before installing {NAME}.", file=sys.stderr)
    exit(1)

if sys.version_info < (3,9):
    print(f"At least Python 3.9 is required for {NAME}.", file=sys.stderr)
    exit(1)


# load and set VERSION and DESCRIPTION
vcontent = open(f"{NAME}/_version.py").read()
exec(vcontent)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    zip_safe=False,
    license='MIT',
    url='None',
    packages=[NAME],
    entry_points={
        "console_scripts": [
            f"fastpork = {NAME}.main:main",
        ],
    },
    package_data={'': ['*.css', '*.sh', '*.html']},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ]
)
