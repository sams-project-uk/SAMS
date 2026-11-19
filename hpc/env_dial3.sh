# Environment for building and running SAMS on DIaL3 (DiRAC, Leicester). Source it:
#     source hpc/env_dial3.sh
# Used by hpc/build_sams.sh and hpc/sams_job.slurm.
#
# The module names below are placeholders: check them with `module avail gcc openmpi hdf5 netcdf cmake`
# and edit this file once. Requirements:
#   - a C++20 compiler (GCC 12 or newer)
#   - MPI (the one PETSc was built with, if PETSc is used)
#   - HDF5 (serial is fine: every rank writes its own file) and NetCDF-C
#   - CMake 3.20 or newer, pkg-config
#   - PETSc 3.18 or newer: only for the implicit two-fluid solver; build with USE_PETSC=OFF otherwise

if command -v module > /dev/null 2>&1; then
    module purge
    module load gcc            # e.g. gcc/12.x or newer
    module load openmpi        # e.g. openmpi/4.1.x, built with the GCC above
    module load hdf5           # serial or parallel, built with the same compiler
    module load netcdf-c 2>/dev/null || module load netcdf
    module load cmake
else
    echo "env_dial3.sh: no 'module' command here; using the tools already on PATH"
fi

# PETSc (only if the implicit solver is wanted): point pkg-config at your PETSc install, e.g.
#     export PKG_CONFIG_PATH=$HOME/petsc/lib/pkgconfig:$PKG_CONFIG_PATH
#     export LD_LIBRARY_PATH=$HOME/petsc/lib:$LD_LIBRARY_PATH

# One MPI rank per core, no OpenMP threads (OpenMP gives no speed-up for SAMS; see the handoff)
export OMP_NUM_THREADS=1
# Split the domain in x only (the problem is 1D)
export SAMS_DECOMPOSITION=0,1,1

# Report anything missing
for tool in mpicxx cmake pkg-config nc-config; do
    command -v $tool > /dev/null || echo "env_dial3.sh: WARNING: '$tool' not found; check the module names in this file"
done
