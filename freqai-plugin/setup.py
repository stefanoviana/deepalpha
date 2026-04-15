"""
DeepAlpha FreqAI Plugin - Setup
================================

Package configuration for PyPI distribution.

Install for development:
    pip install -e .

Build for distribution:
    python setup.py sdist bdist_wheel

Upload to PyPI:
    twine upload dist/*
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepalpha-freqai",
    version="1.0.0",
    author="DeepAlpha Team",
    author_email="stefano@deepalpha.dev",
    description=(
        "A FreqAI-compatible model implementing Triple Barrier Labeling, "
        "SHAP feature selection, meta-labeling, and purged walk-forward CV."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanoviana/deepalpha",
    project_urls={
        "Bug Tracker": "https://github.com/stefanoviana/deepalpha/issues",
        "Documentation": "https://github.com/stefanoviana/deepalpha#readme",
        "Source Code": "https://github.com/stefanoviana/deepalpha",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    py_modules=["deepalpha_model"],
    python_requires=">=3.9",
    install_requires=[
        "lightgbm>=3.3.0",
        "shap>=0.41.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
        "freqtrade": [
            "freqtrade>=2024.1",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    keywords=[
        "freqtrade",
        "freqai",
        "trading",
        "machine-learning",
        "lightgbm",
        "shap",
        "triple-barrier",
        "meta-labeling",
        "quantitative-finance",
    ],
    license="MIT",
)
