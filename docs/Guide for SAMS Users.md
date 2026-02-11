# SAMS User guide

There shouldn't be any SAMS users yet! We're still on version 0.0.3. We will add user documentation here as we go

## Compiling SAMS

SAMS uses CMake as its build system, and it uses a conventional CMake out-of-source build. To compile SAMS, you first need to create a build directory, and then run CMake from that directory. For example:

```bash
mkdir build
cd build
cmake ..
make -j
```

There are various parameters that you can turn on using `-DPARAMETER=ON` when you run CMake. The most important ones are:

1) `USE_MPI` - Turn on to enable MPI parallelism. This is required to run SAMS on more than one processor, but it also adds some overhead, so you may want to turn it off for small test problems.

2) `KOKKOS_OPENMP` - Use Kokkos OpenMP rather than native OpenMP. This is generally slower than native OpenMP support.

3) `KOKKOS_CUDA` - Use Kokkos CUDA. This allows SAMS to run on NVIDIA GPUs. This requires a CUDA capable GPU and the CUDA toolkit to be installed.

4) `KOKKOS_HIP` - Use Kokkos HIP. This allows SAMS to run on AMD GPUs. This requires a HIP capable GPU and the ROCm toolkit to be installed.

5) `KOKKOS_SYCL` - Use Kokkos SYCL. This allows SAMS to run on Intel GPUs. This requires a SYCL capable GPU and the oneAPI toolkit to be installed.

6) `TIMER_SUPPORT` - Turn on to enable timing support. This allows you to get detailed timing information about the various parts of the code, but it also adds some overhead, so you may want to turn it off for small test problems. Currently this is very primitive, but it does give you some information about how long the various parts of the code are taking.

7) `DEBUG_COMPILE` - Disable optimisations and compile with all warnings turned on. This is useful for debugging, but it will make the code run much slower, so you should only turn this on when you are trying to debug something. Note that all builds of SAMS include debug symbols, so you can use a debugger even with an optimized build.

## Target platforms

SAMS is designed primarily to target Linux HPC systems. It has also been tested on macOS, but no testing has been performed on Windows. It should be possible to compile and run SAMS on Windows using WSL, but this has not been tested.

## Current state

Currently SAMS is in very early development, and there is no true input deck currently. When you run SAMS you simply specify the packages that you want to run as space separated arguments at the command line. For example, to run the Emery wind tunnel test problem, you would run:

```bash
./lare3d LARE3D EmeryWindTunnel
```

This approach is not likely to be the final one, but it is sufficient for now. We will add more documentation here as we go, and eventually we will have a proper input deck and user guide.

The current built in test problems are

1) SodShockTube - A 1D shock tube problem with a discontinuity in density and pressure. This is a standard test problem for hydrodynamics codes.

2) BrioAndWu - A 1D shock tube problem with a discontinuity in density, pressure, and magnetic field. This is a standard test problem for magnetohydrodynamics codes.

3) MHDRotor - A 2D problem of a rotating magnetized disk. This is a standard test problem for magnetohydrodynamics codes.

4) OrszagTangVortex - A 2D problem of a vortex in a magnetized fluid. This is a standard test problem for magnetohydrodynamics codes.

5) EmeryWindTunnel - A 2D problem of supersonic flow past a rigid rectangular obstacle. This is a standard test problem for hydrodynamics codes. Currently crashes - not sure why

6) KarmanVortex - A 2D problem of flow past a cylinder. Since we don't have actual viscosity this isn't a true Karman vortex street, but it shows the approach
