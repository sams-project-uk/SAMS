# SAMS User guide

There shouldn't be any SAMS users yet! We're still on version 0.1.0 beta. We will add user documentation here as we go

## Compiling SAMS

SAMS uses CMake as its build system, and it uses a conventional CMake out-of-source build. To compile SAMS, you first need to create a build directory, and then run CMake from that directory. For example:

```bash
mkdir build
cd build
cmake ..
make -j
```

There are various parameters that you can turn on using `-DPARAMETER=ON` when you run CMake. The most important ones are:

1) `MPI` - Turn on to enable MPI parallelism. This is required to run SAMS on more than one processor, but it also adds some overhead, so you may want to turn it off for small test problems.

2) `KOKKOS_OPENMP` - Use Kokkos OpenMP rather than native OpenMP. This is generally slower than native OpenMP support.

3) `KOKKOS_CUDA` - Use Kokkos CUDA. This allows SAMS to run on NVIDIA GPUs. This requires a CUDA capable GPU and the CUDA toolkit to be installed.

4) `KOKKOS_HIP` - Use Kokkos HIP. This allows SAMS to run on AMD GPUs. This requires a HIP capable GPU and the ROCm toolkit to be installed.

5) `KOKKOS_SYCL` - Use Kokkos SYCL. This allows SAMS to run on Intel GPUs. This requires a SYCL capable GPU and the oneAPI toolkit to be installed.

6) `TIMER_SUPPORT` - Turn on to enable timing support. This allows you to get detailed timing information about the various parts of the code, but it also adds some overhead, so you may want to turn it off for small test problems. Currently this is very primitive, but it does give you some information about how long the various parts of the code are taking.

7) `DEBUG_COMPILE` - Disable optimisations and compile with all warnings turned on. This is useful for debugging, but it will make the code run much slower, so you should only turn this on when you are trying to debug something. Note that all builds of SAMS include debug symbols, so you can use a debugger even with an optimized build.

8) `NATIVE_CUDA` - Compile all .cpp files with nvcc as CUDA.

9) `NATIVE_HIP` - Compile all .cpp files as HIP.

10) `HDF5` - Enable HDF5 output.

11) `ADIOS2` - Enable ADIOS2 output.

12) `PETSC` - Enable PETSc.

13) `DECK_PARSER` - Enable initial conditions read from the deck.

## Target platforms

SAMS is designed primarily to target Linux HPC systems. It has also been tested on macOS, but no testing has been performed on Windows. It should be possible to compile and run SAMS on Windows using WSL, but this has not been tested.

## Current state

Currently SAMS is in very early development. When you run SAMS you give it an output directory. SAMS reads the input deck `input.deck` from that directory and writes its output there:

```bash
./sams {output_directory}
```

You can instead specify the packages that you want to run as a comma separated list with the `--packages` option. There must then be no input deck in the output directory. For example, to run the Emery wind tunnel test problem, you would run:

```bash
./sams {output_directory} --packages=LARE3D,EmeryWindTunnel
```

If there is no `input.deck` in the output directory, SAMS reads `input.json`, a deck in JSON. The `--deckfile` option reads a deck with another name in the output directory instead. A deck whose name ends in `.json` is read as JSON.

### Restarting

A run can restart from a restart dump in the output directory with the `--restart` option:

```bash
./sams {output_directory} --restart=restart_final.h5
```

The run then uses the deck stored in the dump. If there is also a deck in the output directory, one of two more options must be given. `--allow-restart-override` makes the keys of that deck replace those of the deck in the dump, and `--restart-deck-only` ignores it. A deck can also give the dump to restart from with the `restart` key of the `SAMS` block. A dump is read as HDF5 if its name ends in `.h5`, as ADIOS2 if it ends in `.bp`, and in the simple format if it is a directory.

Restart dumps are written by output schedules that write the `restart` collection, and by two packages. `GenerateBookendRestart` is active by default, and writes `restart_initial` at the start of the run and `restart_final` at the end. A restarted run does not write `restart_initial`. `RollingRestart` is optional. It writes a restart dump every `dt_snapshots` of simulation time, set in the `RollingRestart` block, cycling through the number of dumps set by `files` in the same block, which defaults to 2. The dumps are named `restart_rolling_0`, `restart_rolling_1` and so on.

We will add more documentation here as we go.

The current built in test problems are

1) SodShockTube - A 1D shock tube problem with a discontinuity in density and pressure. This is a standard test problem for hydrodynamics codes.

2) BrioAndWu - A 1D shock tube problem with a discontinuity in density, pressure, and magnetic field. This is a standard test problem for magnetohydrodynamics codes.

3) MHDRotor - A 2D problem of a rotating magnetized disk. This is a standard test problem for magnetohydrodynamics codes.

4) OrszagTang - A 2D problem of a vortex in a magnetized fluid. This is a standard test problem for magnetohydrodynamics codes.

5) EmeryWindTunnel - A 2D problem of supersonic flow past a rigid rectangular obstacle. This is a standard test problem for hydrodynamics codes. Currently crashes - not sure why

6) KarmanVortex - A 2D problem of flow past a cylinder. Since we don't have actual viscosity this isn't a true Karman vortex street, but it shows the approach

7) OrszagTang3D - A three dimensional Orszag-Tang vortex, with a velocity perturbation that varies in z.

8) SoundWave - A sound wave pulse, the same problem as the SoundWave input deck.

## Creating simple input decks

An input deck is a text file made of blocks. Each block starts with `begin:name` and ends with `end:name`, and holds keys written as `key = value`. Blocks can contain other blocks. A `#` at the start of a line or after whitespace starts a comment that runs to the end of the line. A list is written in brackets, as `(A, B, C)`.

Numerical values can be written as expressions. They can use the constants defined in the `constants` block, and, with the deck initial conditions packages, the parameters `gas_gamma`, `mf` and `t_end`.

### Choosing packages

The `packages` key of the `SAMS` block lists the packages to run. An entry `~Name` deactivates a package that would otherwise be active, an entry `+Name` activates a package as a default, replacing the built in defaults, and an entry `+` on its own means no defaults. The `description` key of the `SAMS` block describes the problem and is printed when the run starts.

The `SAMS` block can also hold `progress_steps`, the number of steps between the progress messages, which defaults to 100. 0 turns the messages off. When SAMS is built with `TIMER_SUPPORT`, the time spent in each phase of each package is printed at the end of the run, and phases that take less than `timer_threshold` percent of the package's total, which defaults to 10, are collected into Other.

### Time and domain

The `time` block holds `t_end`, the end time of the simulation, and `nsteps`, the maximum number of steps. A negative `nsteps` means no limit.

A run must have an active package that can end it. `SimulationTimeTerminate` is active by default, and ends the run once the simulation time reaches `t_end` or the step count reaches `nsteps`. The optional `WalltimeTerminate` package ends the run once the wall time given by `walltime_end` in the `time` block, in HH:MM:SS, has passed.

The `domain` block holds the number of cells in each direction, `nx`, `ny` and `nz`, and the limits of the domain, `x_min`, `x_max`, `y_min`, `y_max`, `z_min` and `z_max`. With the deck initial conditions packages, `ny` and `nz` default to 2, and the limits in y and z default to starting at `x_min` with cells that are the same size as in x.

The `domain` block also holds the MPI decomposition. `MPI_decomposition` is a list of the number of ranks in each direction. A direction that is not given, or is given as 0, is chosen automatically, and if the key is absent the whole decomposition is automatic. `MPI_periodic` is a list of booleans saying whether each direction is periodic. Directions that are not given are not periodic.

### Physics parameters

Parameters of the LARE3D solver go in the `LARE` block, for example `units` and `gas_gamma`.

### Output

Output is controlled by `schedule` blocks in the `diagnostics` block. Each schedule writes output first at time 0 and then every `dt_snapshots` of simulation time. `collections` lists what it writes, which can be collections of outputs or single outputs given by name. `name` is the name of the output files, `format` is the output format and `ndigits` is the number of digits in the file numbers, which defaults to 5. Only one schedule can leave out `name`.

The outputs that a run can write are those of its active packages. LARE3D has `rho`, `energy_electron`, `energy_ion`, `vx`, `vy`, `vz`, `bx`, `by` and `bz`, and the `DerivedQuantities` package adds `pressure`, `temperature`, `magnetic_field_magnitude` and `column_density`. `DerivedQuantities` is not active by default, so it must be listed in `packages` in the `SAMS` block for these outputs to be available. It is mostly intended to demonstrate the output of derived quantities rather than to provide a full set of them.

There are two built in collections. `default` holds the outputs that are written by default, which include those of LARE3D but not those of `DerivedQuantities`, and `restart` holds the outputs that are needed to restart a run. More collections are defined with `collection` blocks in the `diagnostics` block. `name` is the name of the collection, `outputs` lists the outputs in it, and each `include` line adds the outputs of another collection. For example, to write the default outputs and the pressure and temperature every 0.1 of simulation time:

```
begin:diagnostics
  begin:collection
    name = withThermal
    outputs = (pressure, temperature)
    include : default
  end:collection

  begin:schedule
    name = diagnostics
    collections = (withThermal)
    dt_snapshots = 0.1
  end:schedule
end:diagnostics
```

The same output could be written with `collections = (default, pressure, temperature)` in the schedule, without the collection block.

The `format` key of the `diagnostics` block sets the output format of every schedule that does not set its own. It is `adios2`, `hdf5` or `simple`, and defaults to the first of these that SAMS was built with.

### Initial conditions

The `LAREDeckInitialConditions` package sets the variables of LARE3D from the `initial_conditions` block. Each key is the name of a variable, `rho`, `energy_electron`, `energy_ion`, `vx`, `vy`, `vz`, `bx`, `by` or `bz`, and its value is an expression in `x`, `y` and `z`, the position of each point of the variable. An expression can also use variables that have already been set. The keys are applied in the order of the deck.

The `conservedVariableDeckInitialConditions` package does the same for HLLE3D and KT3D, with the variables `rho`, `px`, `py`, `pz`, `energy`, `bx`, `by` and `bz`. Variables that the deck does not set are 0.

### Loading initial conditions from files

With either deck initial conditions package, a key in `initial_conditions` can load its variable from a file instead, with the value `file(uri)`. The scheme of the URI is the format of the file:

* `hdf5` - An HDF5 file, when SAMS is built with HDF5
* `adios2` - An ADIOS2 file, when SAMS is built with ADIOS2
* `binary` - A raw binary file
* `sams` - A data item in a SAMS output or restart dump. The dump is read as HDF5 if its name ends in `.h5`, as ADIOS2 if it ends in `.bp`, and in the simple format if it is a directory

A path that does not start with `/` is relative to the directory of the deck, and `hdf5://data/file.h5` is the same as `hdf5:data/file.h5`. The name of the array in the file, or of the data item in a SAMS dump, is given either as the fragment, after `#`, or with the `variable` key of the query, but not both. The other query keys are `step`, the step to read from an ADIOS2 file or dump, `byte_offset`, the position of the array in a binary file, and `order`, the order of the array in the file, which can only be `C`. A `#` in a URI does not start a comment, because it does not follow whitespace.

The array must have exactly the type of the variable and the size of its domain, without ghost cells. From a SAMS dump only the domain of the data item is read, so a dump that holds ghost cells can be used. For example:

```
begin:initial_conditions
  rho = file(hdf5:start.h5#density)
  energy_electron = file(sams:restart_final.h5?variable=energy_electron)
  energy_ion = energy_electron
end:initial_conditions
```

### Boundary conditions

Boundary conditions are set in the `boundaries` block. For each variable and each edge the condition is taken from the first of these that is present:

1) The `x_min` or `x_max` key of the block of that variable inside `boundaries`

2) The `x` key of the block of that variable

3) The `x_min` or `x_max` key of `boundaries`

4) The `x` key of `boundaries`

and in the same way for y and z. The conditions are

* `periodic` - Periodic with the opposite edge
* `symmetric` - The values in the domain are copied symmetrically across the edge
* `zero_gradient` - Zero gradient across the edge
* `clamp(value)` - The ghost cells are set to the value
* `mirror(value)` - The ghost cells are reflected about the value on the edge
* `mirror` - The ghost cells are reflected about the value on the edge found from the domain
* `expression(expression)` - The ghost cells are set to the expression, which can use `x`, `y`, `z` and `time`

With the deck initial conditions packages, `MPI_periodic` in the `domain` block defaults to periodic in each direction in which the boundary conditions of `rho` are periodic.

### Files written by the run

Before the first timestep, SAMS writes the deck that the run uses, with every value that the packages set or used, to `ran.deck` in the output directory, or `ran.json` if the input deck was JSON. It also reports any blocks or keys in the deck that were never read, which usually means that they are misspelt or not used by any active package.

### Example

This is the SoundWave input deck, which drives a sound wave pulse in from the lower x boundary.

```
begin:SAMS
  packages = (LARE3D, LAREDeckInitialConditions)
end:SAMS

begin:LARE
  units = normalised
  gas_gamma = 5 / 3
end:LARE

# Sound speed is sqrt(gamma * (gamma - 1) * e) with total specific energy e = 1
begin:constants
  c_s = sqrt(gas_gamma * (gas_gamma - 1))
  mach = 0.05
  t_peak = 0.1
  t_width = 0.03
end:constants

# The pulse peaks at time 0.1 and then crosses the domain
begin:time
  t_end = t_peak + 1 / c_s
end:time

begin:diagnostics
  begin:schedule
    name = default
    collections = (default)
    dt_snapshots = t_end / 20
    format = hdf5
  end:schedule
end:diagnostics

begin:domain
  nx = 256
  ny = 256
  x_min = 0
  x_max = 1
  y_min = 0
  y_max = 1
end:domain

begin:initial_conditions
  rho = 1
  energy_electron = 0.5
  energy_ion = 0.5
end:initial_conditions

begin:boundaries
  x = zero_gradient
  y = periodic
  z = periodic
  begin:vx
    # Mach 0.05 pulse, Gaussian in y about the middle of the face and in time about 0.1
    x_min = expression(mach * c_s * gauss(y, 0.5, 0.05) * gauss(time, t_peak, t_width))
  end:vx
end:boundaries
```
