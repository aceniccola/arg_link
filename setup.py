#!/usr/bin/env python3
"""
Setup script for Argument Link project.
This makes the src package importable from anywhere in the project.
"""

from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="argument-link",
    version="0.1.0",
    author="Andrew Ceniccola",
    description="AI-powered legal argument analysis and linking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aceniccola/arg_link",
    packages=find_packages(),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Legal",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "argument-link=src.main:main",
        ],
    },
)