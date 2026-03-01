"""
IPL Dataset Analysis - Source Package
=====================================

Reusable modules for IPL cricket data analysis covering 17 seasons
(2008-2024). Provides data loading, exploratory analysis helpers,
and machine-learning pipeline utilities.

Modules
-------
data_loader
    Functions for loading and cleaning IPL match and delivery datasets.
eda
    Helper functions for exploratory data analysis and visualization.
models
    ML pipeline builders and evaluation utilities.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ipl-dataset-analysis")
except PackageNotFoundError:
    __version__ = "0.1.0"
