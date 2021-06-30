# MergeTreeMetric

This is a Python implementation of computing pairwise distance matrices for time-varying data. These matrices are calculated using four metrics: bottleneck distance, Wasserstein distance, and labelled interleaving distance between two merge trees, and Euclidean distances between scalar fields.

The implementation is described in "Geometry Aware Merge Tree Comparisons for Time-Varying Data with Interleaving Distances" (under review).


- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Installation

Tested with Python 2.7&3.7, MacOS and Linux.

### Dependencies

- Python packages
This software requires [NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/), [NetworkX](https://networkx.github.io/), [Matplotlib](https://matplotlib.org/) and [VTK](https://vtk.org) to run.
