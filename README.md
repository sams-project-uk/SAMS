# SAMS - the Solar Atmospheric Modelling Suite

SAMS is a modular, performance portable framework for solar physics simulations. A simulation is built from packages, such as solvers, initial conditions and output control, which are selected at run time. The LARE3D Lagrangian remap MHD solver is included.

SAMS is at version 0.1.0 beta.

## Quick start

```bash
mkdir build
cd build
cmake ..
make -j
./sams {output_directory}
```

SAMS reads the input deck `input.deck` from the output directory and writes its output there.

## Documentation

* [Guide for SAMS Users](docs/Guide%20for%20SAMS%20Users.md) - Compiling and running SAMS, and writing input decks
* [SAMS For LARE Users](docs/SAMS%20For%20LARE%20Users.md) - Setting up problems in the style of the Fortran LARE3D code
* [Using SAMS Boundary Conditions](docs/Using%20SAMS%20Boundary%20Conditions.md) - Adding the built in boundary conditions to variables
* [Developing SAMS Packages](docs/Developing%20SAMS%20Packages.md) - How packages are written and the events that the runner calls
* [SAMS Harness Principles](docs/SAMS%20Harness%20Principles.md) - The axis and variable registries, MPI and the other parts of the harness
* [Example Alternate Solvers](docs/Example%20Alternate%20Solvers.md) - The HLLE3D and KT3D example solvers
