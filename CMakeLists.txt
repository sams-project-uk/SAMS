cmake_minimum_required(VERSION 3.12)

#Check for flag support
enable_language(CXX)
include(CheckCXXCompilerFlag)

#Check for -march=native support
check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
#Check for OpenMP support
check_cxx_compiler_flag("-fopenmp" HAVE_OPENMP)

#Enable array bounds checking
option(ARRAY_BOUNDS_CHECKING "Enable array bounds checking" OFF)
#Reserved for future use: array set element checking
option(ARRAY_SET_CHECKER "Enable array set element checking" OFF)
#Enable MPI
option(MPI "Enable MPI" OFF)
#Enable Kokkos CUDA backend
option(KOKKOS_CUDA "Enable Kokkos with CUDA backend" OFF)
#Enable Kokkos HIP backend
option(KOKKOS_HIP "Enable Kokkos with HIP backend" OFF)
#Enable native HIP compilation
option(KOKKOS_SYCL "Enable Kokkos with SYCL backend" OFF)
#Enable Kokkos OpenMP backend
option(KOKKOS_OPENMP "Enable Kokkos with OpenMP backend" OFF)
#Enable native CUDA compilation
option(NATIVE_CUDA "Compile all .cpp files with nvcc as CUDA" OFF)
#Debug to stderr instead of log files
option(DEBUG_TO_STDERR "Send debug output to stderr instead of log files" OFF)
#Do not unroll loops in the native OpenMP implementation. Usually reduces performance.
option(NO_NATIVE_UNROLL "Do not unroll loops in the native OpenMP implementation" OFF)
#Native loop collapse level (0 = disabled). Usually collapsing loops reduces performance, but for very small domains it can help.
option(NATIVE_LOOP_COLLAPSE "Enable loop collapse in the native OpenMP implementation" 0)
#Enable timer support - prints times for each stage of each package at end of run
option(TIMER_SUPPORT "Enable timing of packages" OFF)
#Compile with -O0 (debug compile mode) also, -Wall -Wextra
option(DEBUG_COMPILE "Compile with -O0 (debug compile mode)" OFF)
#Allow rank inconsistency in queries for whole code events (terminate, output)
option(ALLOW_RANK_INCONSISTENCY "Allow rank inconsistency in queries for whole code events (terminate, output)" OFF)
#Enable PETSc
option(PETSC "Enable PETSc" OFF)
#Update the field of the parabolic solvers from the net flux of the solved field, so that problems in flux form conserve exactly
option(EXACT_FLUX_CONSERVATION "Exact flux conservation in the parabolic solvers" OFF)
#Relative tolerance of the parabolic solvers. If it is not set the default in the source code is used
set(SOLVER_TOLERANCE "" CACHE STRING "Relative tolerance of the parabolic solvers, for example 1e-10")
#Larsen's flux limiter in Braginskii thermal conduction. Turning it off defines NO_BRAGINSKII_LARSON
option(BRAGINSKII_LARSON "Larsen's flux limiter in Braginskii thermal conduction" ON)
#van Leer limiter of the cross conductances in Braginskii thermal conduction. Turning it off defines NO_BRAGINSKII_VAN_LEER
option(BRAGINSKII_VAN_LEER "van Leer limiter of the cross conductances in Braginskii thermal conduction" ON)

if (TIMER_SUPPORT)
  message(STATUS "Enabling timer support")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TIMER")
endif()

if(EXACT_FLUX_CONSERVATION)
  message(STATUS "Enabling exact flux conservation in the parabolic solvers")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXACT_FLUX_CONSERVATION")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DEXACT_FLUX_CONSERVATION")
endif()

if(NOT BRAGINSKII_LARSON)
  message(STATUS "Turning off Larsen's flux limiter in Braginskii thermal conduction")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_BRAGINSKII_LARSON")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNO_BRAGINSKII_LARSON")
endif()

if(NOT BRAGINSKII_VAN_LEER)
  message(STATUS "Turning off the van Leer limiter in Braginskii thermal conduction")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_BRAGINSKII_VAN_LEER")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNO_BRAGINSKII_VAN_LEER")
endif()

if(SOLVER_TOLERANCE)
  message(STATUS "Setting the tolerance of the parabolic solvers to ${SOLVER_TOLERANCE}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSOLVER_TOLERANCE=${SOLVER_TOLERANCE}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSOLVER_TOLERANCE=${SOLVER_TOLERANCE}")
endif()

# Debug level: numeric 0..3 (default 0)
set(DEBUG_LEVEL "0" CACHE STRING "Debug level: 0 (none), 1, 2, 3")
set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS 0 1 2 3)

#Debug level must be 0, 1, 2, or 3
if(NOT DEBUG_LEVEL MATCHES "^[0-3]$")
  message(WARNING "Invalid DEBUG_LEVEL='${DEBUG_LEVEL}'. Resetting to 0.")
  set(DEBUG_LEVEL "0" CACHE STRING "Debug level: 0 (none), 1, 2, 3" FORCE)
endif()

#Native loop collapse level (0 = disabled)
if(NATIVE_LOOP_COLLAPSE GREATER 0)
  message(STATUS "Enabling native loop collapse with level ${NATIVE_LOOP_COLLAPSE}")
  add_compile_definitions(NATIVE_LOOP_COLLAPSE=${NATIVE_LOOP_COLLAPSE})
endif()


# Make available to sources and flags
add_compile_definitions(DEBUG_LEVEL=${DEBUG_LEVEL})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSAMS_DEBUG=${DEBUG_LEVEL}")

if(ARRAY_BOUNDS_CHECKING)
  message(STATUS "Enabling array bounds checking")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARRAY_BOUNDS_CHECKING")
endif()

if(ARRAY_SET_CHECKER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSET_CHECKER")
endif()

if(MPI)
  find_program(MPI_CXX_COMPILER_PATH
    NAMES mpicxx mpic++ mpiCC
    PATHS ENV PATH
    NO_DEFAULT_PATH
  )
  if(NOT MPI_CXX_COMPILER_PATH)
    message(FATAL_ERROR "MPI=ON requested but mpicxx (or equivalent) was not found on PATH")
  endif()
  set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER_PATH}" CACHE FILEPATH "MPI C++ compiler" FORCE)
endif()

if(MPI)
  set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "MPI C++ compiler")

  message(STATUS "Enabling MPI support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")

  if(KOKKOS_CUDA OR NATIVE_CUDA)
    set(CMAKE_CUDA_HOST_COMPILER "mpicxx" CACHE FILEPATH "CUDA host C++ compiler")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_MPI")
  endif()
endif()

if(KOKKOS_CUDA)
  message(STATUS "Enabling Kokkos with CUDA backend")
  add_compile_definitions(KOKKOS_CUDA)
endif()

if(KOKKOS_HIP)
  message(STATUS "Enabling Kokkos with HIP backend")
  add_compile_definitions(KOKKOS_HIP)
endif()

if(KOKKOS_SYCL)
  message(STATUS "Enabling Kokkos with SYCL backend")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_SYCL")
endif()

if(KOKKOS_OPENMP)
  message(STATUS "Enabling Kokkos with OpenMP backend")
  add_compile_definitions(KOKKOS_OPENMP)
endif()

if(NATIVE_CUDA)
  message(STATUS "Enabling native CUDA compilation")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS ON)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NATIVE_HIP)
  message(STATUS "Enabling native HIP compilation")
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS ON)
  set(CMAKE_HIP_ARCHITECTURES native)
endif()

if(DEBUG_TO_STDERR)
  add_compile_definitions(SAMS_DEBUG_TO_STDERR)
endif()

if(NATIVE_CUDA)
  project(SAMS-PP LANGUAGES CXX CUDA)
else()
  project(SAMS-PP LANGUAGES CXX)
endif()

if(ALLOW_RANK_INCONSISTENCY)
  message(STATUS "Allowing rank inconsistency in queries for whole code events (terminate, output)")
  add_compile_definitions(ALLOW_RANK_INCONSISTENCY)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set optimization flags depending on DEBUG_COMPILE (separate from DEBUG_LEVEL)
if(DEBUG_COMPILE)
  message(STATUS "Debug compile mode enabled: using -O0 (no optimizations)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DUSE_HDF5 -Wall -Wextra")
  # Also set CUDA/HIP flags for debug compile mode if those languages are enabled
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O0 -g")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DUSE_HDF5 -DUNROLL_NATIVE_OPENMP_LOOPS")
  # Preserve existing CUDA/HIP default flags (they may be extended elsewhere)
endif()

if (HAVE_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
if (HAVE_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

find_package(HDF5 REQUIRED COMPONENTS CXX)

#Recursively gather all .cpp source files in src/
file(GLOB_RECURSE SAMS_SOURCES src/*.cpp)

if(NATIVE_CUDA)
  message(STATUS "Configuring for native CUDA compilation")
  foreach(src_file IN LISTS SAMS_SOURCES)
    set_source_files_properties(${src_file} PROPERTIES LANGUAGE CUDA)
  endforeach()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_CUDA  -DUSE_HDF5 --extended-lambda")
endif()

if(NATIVE_HIP)
  message(STATUS "Configuring for native HIP compilation")
  foreach(src_file IN LISTS SAMS_SOURCES)
    set_source_files_properties(${src_file} PROPERTIES LANGUAGE HIP)
  endforeach()
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -DUSE_HIP")
endif()

add_executable(sams ${SAMS_SOURCES})
message(STATUS "CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
  

target_include_directories(sams PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/LARE")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/LARE/include")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/Parabolic")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/ThermalConduction")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/LARE")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/SodShockTube")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/BrioAndWu")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/MHDRotor")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/OrszagTang")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/OrszagTang3D")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/EmeryWindTunnel")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/src/InitialConditions/KarmanVortex")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/include/harness")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/include/runner")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/include/boundaryConditions")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sams PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(sams PRIVATE stdc++)

if (KOKKOS_CUDA OR KOKKOS_HIP OR KOKKOS_SYCL OR KOKKOS_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_KOKKOS")
  # Allow Kokkos_DIR to come from env or be set on the cmake command line
  if(NOT Kokkos_DIR)
    set(Kokkos_DIR "$ENV{Kokkos_DIR}")
  endif()
  message(STATUS "Looking for Kokkos in Kokkos_DIR='${Kokkos_DIR}'")
  find_package(Kokkos REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${Kokkos_DIR}")
  message(STATUS "Found Kokkos ${Kokkos_VERSION} at ${Kokkos_DIR}")
  target_link_libraries(sams PRIVATE Kokkos::kokkos)
endif()

if (PETSC)
  message(STATUS "Enabling PETSc support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PETSC")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_PETSC")

  # Allow PETSC_DIR / PETSC_ARCH to come from env or be set on the cmake command line
  if(NOT PETSC_DIR)
    set(PETSC_DIR "$ENV{PETSC_DIR}")
  endif()
  if(NOT PETSC_ARCH)
    set(PETSC_ARCH "$ENV{PETSC_ARCH}")
  endif()

  set(PETSC_FOUND_MANUALLY FALSE)

  if(PETSC_DIR)
    message(STATUS "Using PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH}")
    set(ENV{PKG_CONFIG_PATH} "${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig:${PETSC_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  else()
    message(WARNING "PETSC=ON but PETSC_DIR is not set; relying on default PKG_CONFIG_PATH to find PETSc")
  endif()

  find_package(PkgConfig)
  if(PkgConfig_FOUND)
    unset(PETSC_FOUND CACHE)
    unset(PETSC_INCLUDE_DIRS CACHE)
    pkg_check_modules(PETSC QUIET PETSc)
  endif()

  if(NOT PETSC_FOUND)
    message(STATUS "pkg-config could not find PETSc.pc; falling back to manual detection via PETSC_DIR/PETSC_ARCH")
    if(NOT PETSC_DIR)
      message(FATAL_ERROR "PETSC=ON, pkg-config lookup failed, and PETSC_DIR is not set. Please set PETSC_DIR (and PETSC_ARCH if used).")
    endif()

    find_path(PETSC_INCLUDE_DIR petscksp.h
      HINTS "${PETSC_DIR}/${PETSC_ARCH}/include" "${PETSC_DIR}/include"
    )
    find_library(PETSC_LIBRARY
      NAMES petsc
      HINTS "${PETSC_DIR}/${PETSC_ARCH}/lib" "${PETSC_DIR}/lib"
    )

    if(NOT PETSC_INCLUDE_DIR OR NOT PETSC_LIBRARY)
      message(FATAL_ERROR "Could not locate PETSc headers/libraries under PETSC_DIR='${PETSC_DIR}' PETSC_ARCH='${PETSC_ARCH}'")
    endif()

    set(PETSC_INCLUDE_DIRS "${PETSC_INCLUDE_DIR}" "${PETSC_DIR}/include" "${PETSC_DIR}/${PETSC_ARCH}/include")
    set(PETSC_LDFLAGS "${PETSC_LIBRARY}")
    set(PETSC_FOUND_MANUALLY TRUE)
  endif()

  if(NOT EXISTS "${PETSC_INCLUDE_DIR}/petscksp.h")
    find_path(PETSC_INCLUDE_DIR petscksp.h
      HINTS ${PETSC_INCLUDE_DIRS} "${PETSC_DIR}/${PETSC_ARCH}/include" "${PETSC_DIR}/include"
    )
    if(NOT PETSC_INCLUDE_DIR)
      message(FATAL_ERROR "petscksp.h not found. PETSC_INCLUDE_DIRS='${PETSC_INCLUDE_DIRS}' did not contain it; check PETSC_DIR/PETSC_ARCH.")
    endif()
    list(APPEND PETSC_INCLUDE_DIRS "${PETSC_INCLUDE_DIR}")
  endif()

  message(STATUS "PETSC_INCLUDE_DIRS=${PETSC_INCLUDE_DIRS}")
  target_include_directories(sams PRIVATE ${PETSC_INCLUDE_DIRS})
  target_link_libraries(sams PRIVATE ${PETSC_LDFLAGS})
endif()
