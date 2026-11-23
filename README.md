# Moire-Excitons

Code accompanying the theoretical modeling in: 
Deng et al., "Correlation-Driven Ballistic Exciton Expansion in a Moiré Bose-Fermi Mixture", ?Journal? (2026). [?link?]

## Overview
The C++ code in `code/` generates the data in `data/`: it first solves the
self-consistent Hartree-Fock equation to obtain the electronic equilibrium
single-particle densities for a range of chemical potentials (fillings), and then performs
the Keldysh mean-field time evolution of the coupled electron-exciton system.
The Python script `python_utils/plots.py` reads the data and produces produces the figures of the paper (listed below).

## C++ dependencies
Eigen3 - https://libeigen.gitlab.io/
Boost (odeint, math) - https://www.boost.org/
OpenMP - https://www.openmp.org/
GSL - https://www.gnu.org/software/gsl/
FFTW3 - https://www.fftw.org/

## Usage

### Plotting (no simulation required)
The data used in the paper is already included in `data/`. To regenerate
the figures, run

    python python_utils/plots.py

The table below lists which function produces which figure.

### Simulation (optional)
To regenerate the data from scratch:

1. Build the C++ code:

       cmake -S code -B build -DCMAKE_BUILD_TYPE=Release
       cmake --build build

2. Run the executable (it takes no command-line arguments):

       ./build/kbe_timestepper

   `main()` performs the following steps:
   1. `mu_sweep()` solves for the electronic initial states for a range of chemical potentials.

   2. For each, `solve_bfKBE_markov()` performs the time evolution of the joint exciton-electron system.

   Output is written to `data/` and **overwrites the included paper data**.

All simulation parameters are hard-coded at the beginning of `mu_sweep()` and
`solve_bfKBE_markov()`. Edit them and rebuild to change the setup.

## Figures
| Figure            | Python function          |
|-------------------|--------------------------|
| Fig. 3d           | `plot_diam_heatmap()`    |
| Fig. 3e           | `plot_diam_trajectory()` |
| Ext. Data Fig. 9a | `phase_analysis()`       |
| Ext. Data Fig. 9b | `plot_occ_trajectory()`  |
