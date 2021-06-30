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

1. Python packages

- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [NetworkX](https://networkx.github.io/)
- [Matplotlib](https://matplotlib.org/) 
- [VTK](https://vtk.org) 

If you do not have these packages installed, please use the following command to intall them.

```bash
$ pip install numpy
$ pip install scipy
$ pip install networkx
$ pip install matplotlib
$ pip install vtk
```

2. TTK and Paraview

- We need TTK and Paraview to calculate the merge trees from scalar fields. Please following this [link](https://topology-tool-kit.github.io/installation.html) to install both TTK and Paraview.

- After installation of TTK and Paraview, adding environment variable for pvpython in whatever file you normally configure these (e.g. ~/.bash_profile):
```bash
export PV_PLUGIN_PATH="/PATH/TO/PVPYTHON"
```

3. Hera

- We use Hera to calculate bottleneck distance and Wasserstein distance between two persistence diagrams. Please following this [link](https://github.com/grey-narn/hera) to install Hera.

- Adding environment variables in whatever file you normally configure these (e.g. ~/.bash_profile):
```bash
export PATH="/PATH/TO/hera/wasserstein/build"
export PATH="/PATH/TO/hera/bottleneck/build"
```

### Installation

```bash
$ git clone https://github.com/tdavislab/MergeTreeMetric.git
$ cd MergeTreeMetric
$ python MergeTreeMetric.py [dir to files] [Name of scalar field] [Mapping Strategy: TD/ED/ET/MP]
 [Extending Strategy: dmyLeaf/dmyVert] [Tree Type: jt/st] [Glabal or Pairwise Mapping: GM/PM] 
 [Skip merge tree and morse smale calculation] [Output labelling result for global mapping] 
 [threshold for simplification (optional)]
```

## Features
```bash
$ python MergeTreeMetric.py [Path to files] [Name of scalar field] [Mapping Strategy: TD/ED/ET/MP]
 [Extending Strategy: dmyLeaf/dmyVert] [Tree Type: jt/st] [Glabal or Pairwise Mapping: GM/PM] 
 [Skip merge tree and morse smale calculation] [Output labelling result for global mapping] 
 [threshold for simplification (optional)]
```

#### Parameters

- **Path to files**
  - Path to the folder of time-varying dataset.
  - Each data file should has an index in filename to specify the time step.
  - Acceptable format of time-varying data: ".vtp" and ".vti"
 
- **Name of scalar field**
  - The name of attribute for computing merge tree. 

