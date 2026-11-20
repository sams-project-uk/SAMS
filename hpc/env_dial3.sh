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
    # The Intel compilers use the C++ standard library of a GCC installation; the system GCC is too old
    # for C++20 (<concepts>). Load a GCC module for its libstdc++: GCC 10-12 (icpx 2022 is clang 14,
    # which may not parse the GCC 13+ headers). Override with SAMS_GCC_MODULE=gcc/<version>.
    if [ -z "${SAMS_GCC_MODULE:-}" ]; then
        SAMS_GCC_MODULE=$(module -t avail gcc 2>&1 | grep -E '^gcc[^/]*/1[0-2]([.][0-9]+)*$' | sort -V | tail -1)
    fi
    if [ -n "$SAMS_GCC_MODULE" ] && module load "$SAMS_GCC_MODULE" 2> /dev/null; then
        echo "env_dial3.sh: using $SAMS_GCC_MODULE for the C++ standard library"
    else
        echo "env_dial3.sh: WARNING: no GCC 10-12 module found (module -t avail gcc); C++20 will not compile." >&2
        echo "  Set SAMS_GCC_MODULE=<module> before sourcing this file." >&2
    fi
else
    echo "env_dial3.sh: no 'module' command here; using the tools already on PATH"
fi

# Use the LLVM-based icpx/icx behind the OpenMPI wrappers: the classic icpc does not fully support
# C++20. (OpenMPI's mpicxx/mpicc honour OMPI_CXX/OMPI_CC.)
if command -v icpx > /dev/null 2>&1; then
    export OMPI_CXX=icpx OMPI_CC=icx
    # Point icpx explicitly at the GCC on PATH (the module's), not whatever it finds by default, and
    # put that GCC's newer libstdc++ ahead of the system one at run time.
    # The prefix is found from GCC itself: <prefix>/lib*/gcc/<triple>/<version>/libgcc.a. (HPE's
    # /opt/gcc/<v>/bin is only a front end; the installation icpx needs is elsewhere, e.g. .../snos.)
    GCC_LIBGCC=$(gcc -print-libgcc-file-name 2> /dev/null)
    GCC_PREFIX=$(realpath -m "$(dirname "$GCC_LIBGCC")/../../../..")
    if [ -n "$GCC_LIBGCC" ] && [ "$GCC_PREFIX" != /usr ]; then
        # HPE's GCC keeps its C++ headers in a non-standard place (e.g. include/g++) that icpx does not
        # look in, so give icpx g++'s own C++ include directories explicitly.
        GCC_CXX_INC=
        for d in $(echo | g++ -x c++ -E -v - 2>&1 | sed -n '/search starts here:/,/End of search list/p' | grep -E '^ /.*/(c|g)\+\+'); do
            GCC_CXX_INC="$GCC_CXX_INC -isystem $d"
        done
        export OMPI_CXXFLAGS="--gcc-toolchain=$GCC_PREFIX${GCC_CXX_INC:+ -nostdinc++$GCC_CXX_INC}${OMPI_CXXFLAGS:+ $OMPI_CXXFLAGS}"
        GCC_STDCXX_DIR=$(dirname "$(gcc -print-file-name=libstdc++.so)")
        [ -d "$GCC_STDCXX_DIR" ] && export LD_LIBRARY_PATH=$GCC_STDCXX_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        selected=$(echo | icpx --gcc-toolchain="$GCC_PREFIX" -x c++ -fsyntax-only -v - 2>&1 | grep 'Selected GCC installation')
        case "$selected" in
            *"$GCC_PREFIX"*) echo "env_dial3.sh: icpx uses GCC in $GCC_PREFIX" ;;
            *) echo "env_dial3.sh: WARNING: icpx did not accept $GCC_PREFIX (${selected:-no GCC selected}); libgcc is $GCC_LIBGCC" >&2 ;;
        esac
        if ! echo '#include <concepts>' | icpx -std=c++20 $OMPI_CXXFLAGS -x c++ -fsyntax-only - > /dev/null 2>&1; then
            echo "env_dial3.sh: WARNING: icpx still cannot find <concepts>. C++ includes from g++:${GCC_CXX_INC:- none}" >&2
        fi
    fi
fi

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
