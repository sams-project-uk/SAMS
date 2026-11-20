# Environment for building and running SAMS on DIaL3 (DiRAC, Leicester). Use with:
#     source hpc/env_dial3.sh
# Used by hpc/build_sams.sh and hpc/sams_job.slurm, so the compiler, MPI and HDF5 are always the same.
# Based on the environment that works for PIP on DIaL3 (Intel oneAPI + OpenMPI 4 + a parallel HDF5
# built in $HOME/sw), plus what SAMS needs on top: a C++20 compiler.
#
# SAMS needs: C++20 (concepts, <filesystem>), MPI, HDF5 (C API; serial or parallel), CMake >= 3.20,
# pkg-config, python3. NetCDF and PETSc are optional (build_sams.sh builds without them).

# Keep conda's compilers/MPI/HDF5 off the PATH
if type conda > /dev/null 2>&1; then
    conda deactivate 2> /dev/null || true
fi

if command -v module > /dev/null 2>&1; then
    module purge
    module load intel-oneapi/2022.1.2
    module load openmpi4/intel/4.0.5
    module load cmake 2> /dev/null || true
    # SAMS needs C++20, and the system GCC 8 is too old. Load a GCC module (newest 10-12 by default;
    # override with SAMS_GCC_MODULE=gcc/<version>). It is the compiler by default (SAMS_CXX=gnu), or
    # supplies the C++ standard library for icpx (SAMS_CXX=intel).
    if [ -z "${SAMS_GCC_MODULE:-}" ]; then
        SAMS_GCC_MODULE=$(module -t avail gcc 2>&1 | grep -E '^gcc[^/]*/1[0-2]([.][0-9]+)*$' | sort -V | tail -1)
    fi
    if [ -n "$SAMS_GCC_MODULE" ] && module load "$SAMS_GCC_MODULE" 2> /dev/null; then
        echo "env_dial3.sh: loaded $SAMS_GCC_MODULE"
    else
        echo "env_dial3.sh: WARNING: no GCC 10-12 module found (module -t avail gcc); C++20 will not compile." >&2
        echo "  Set SAMS_GCC_MODULE=<module> before sourcing this file." >&2
    fi
else
    echo "env_dial3.sh: no 'module' command here; using the tools already on PATH"
fi

# The compiler behind the OpenMPI wrappers (mpicxx/mpicc honour OMPI_CXX/OMPI_CC).
#   SAMS_CXX=gnu (default): GCC from the module above. SAMS uses only the C APIs of MPI and HDF5, so the
#     Intel-built OpenMPI and HDF5 link fine with it.
#   SAMS_CXX=intel: icpx/icx. icpx 2022 is clang 14, which lacks parts of C++20 that SAMS uses (e.g.
#     omitting 'typename' before dependent types, P0634), so it only works with a newer oneAPI module.
unset OMPI_CXXFLAGS SAMS_LINK_LIBS
GCC_LIBGCC=$(gcc -print-libgcc-file-name 2> /dev/null)
GCC_PREFIX=$(realpath -m "$(dirname "$GCC_LIBGCC")/../../../..")
if [ -n "$GCC_LIBGCC" ] && [ "$GCC_PREFIX" != /usr ]; then
    # The module's newer libstdc++ ahead of the system one at run time
    GCC_STDCXX_DIR=$(dirname "$(gcc -print-file-name=libstdc++.so)")
    [ -d "$GCC_STDCXX_DIR" ] && export LD_LIBRARY_PATH=$GCC_STDCXX_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
if [ "${SAMS_CXX:-gnu}" = intel ] && command -v icpx > /dev/null 2>&1; then
    export OMPI_CXX=icpx OMPI_CC=icx
    if [ -n "$GCC_LIBGCC" ] && [ "$GCC_PREFIX" != /usr ]; then
        # Point icpx at the module's GCC. The prefix comes from GCC itself (HPE's /opt/gcc/<v>/bin is a
        # front end; the installation is e.g. .../snos), and its C++ headers are where icpx does not
        # look (include/g++), so pass g++'s own C++ include directories too.
        GCC_CXX_INC=
        for d in $(echo | g++ -x c++ -E -v - 2>&1 | sed -n '/search starts here:/,/End of search list/p' | grep -E '^ /.*/(c|g)\+\+'); do
            GCC_CXX_INC="$GCC_CXX_INC -isystem $d"
        done
        export OMPI_CXXFLAGS="--gcc-toolchain=$GCC_PREFIX${GCC_CXX_INC:+ -nostdinc++$GCC_CXX_INC}"
    fi
else
    export OMPI_CXX=g++ OMPI_CC=gcc
    # The HDF5 (and OpenMPI) built with the Intel compilers call Intel runtime routines (libintlc etc.),
    # which icx links automatically but g++ does not. build_sams.sh adds these to the link line.
    SAMS_LINK_LIBS=
    for lib in libintlc.so.5 libimf.so libsvml.so libirng.so; do
        for d in ${LD_LIBRARY_PATH//:/ }; do
            if [ -e "$d/$lib" ]; then SAMS_LINK_LIBS="$SAMS_LINK_LIBS $d/$lib"; INTEL_RT_DIR=$d; break; fi
        done
    done
    [ -n "$SAMS_LINK_LIBS" ] && SAMS_LINK_LIBS="$SAMS_LINK_LIBS -Wl,-rpath,$INTEL_RT_DIR"
    export SAMS_LINK_LIBS
fi
echo "env_dial3.sh: mpicxx uses $(command -v $OMPI_CXX) ($($OMPI_CXX --version 2> /dev/null | head -1))"

# openmpi4/intel/4.0.5 was built against PMIx 3 (/cm/shared/apps/cm-pmix3/3.1.4), but that directory
# no longer holds libpmix.so.2, so the loader falls back to the system PMIx 2.2.5 in /lib64 (same
# soname) and mpirun aborts with "unexpected version of PMIx". Put the first PMIx 3 directory that
# really contains the library on LD_LIBRARY_PATH.
PMIX3_LIB=
for d in /cm/shared/apps/cm-pmix3/3.1.4/lib /cm/shared/apps/cm-pmix3/3.1.7/lib /cm/shared/apps/cm-pmix3/current/lib; do
    if [ -e "$d/libpmix.so.2" ]; then PMIX3_LIB=$d; break; fi
done
if [ -n "$PMIX3_LIB" ]; then
    export LD_LIBRARY_PATH=$PMIX3_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
elif [ -d /cm/shared/apps ]; then
    echo "env_dial3.sh: WARNING: no PMIx 3 libpmix.so.2 found under /cm/shared/apps/cm-pmix3" >&2
fi

# Parallel HDF5 built with this compiler and MPI (DIaL3 has no parallel HDF5 module). SAMS uses the
# C API only, so this HDF5 (which has no C++ library) is fine.
export HDF5_DIR=${HDF5_DIR:-$HOME/sw/hdf5-1.14.6-ompi4-intel}
export HDF5_ROOT=$HDF5_DIR
export PATH=$HDF5_DIR/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# PETSc (only for the implicit two-fluid solver, normally not built on DIaL3): point pkg-config at it
#     export PKG_CONFIG_PATH=$HOME/sw/petsc/lib/pkgconfig:$PKG_CONFIG_PATH
#     export LD_LIBRARY_PATH=$HOME/sw/petsc/lib:$LD_LIBRARY_PATH

# Stop glibc returning large freed arrays to the OS on every free (as for PIP). SAMS allocates and
# frees work arrays every remap sweep, so this may help it too; compare in the scaling test.
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=68719476736

# One MPI rank per core, no OpenMP threads; split the (1D) domain in x only
export OMP_NUM_THREADS=1
export SAMS_DECOMPOSITION=0,1,1

# Run directories live on scratch
export SAMS_SCRATCH=${SAMS_SCRATCH:-/scratch/dp446/$USER}

# Report anything missing
for tool in mpicxx mpicc cmake pkg-config python3 h5cc h5pcc; do
    command -v $tool > /dev/null || [ $tool = h5cc ] || [ $tool = h5pcc ] || \
        echo "env_dial3.sh: WARNING: '$tool' not found" >&2
done
command -v h5pcc > /dev/null || command -v h5cc > /dev/null || echo "env_dial3.sh: WARNING: no HDF5 (h5pcc/h5cc) on PATH; check HDF5_DIR" >&2
