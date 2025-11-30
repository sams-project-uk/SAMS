cmake_minimum_required(VERSION 3.10)

option(ARRAY_BOUNDS_CHECKING "Enable array bounds checking" OFF)
option(ARRAY_SET_CHECKER "Enable array set element checking" OFF)
option(MPI "Unable MPI" OFF)
option(KOKKOS_CUDA "Enable Kokkos with CUDA backend" OFF)
option(KOKKOS_HIP "Enable Kokkos with HIP backend" OFF)
option(KOKKOS_SYCL "Enable Kokkos with SYCL backend" OFF)
option(KOKKOS_OPENMP "Enable Kokkos with OpenMP backend" OFF)
option(NATIVE_CUDA "Compile all .cpp files with nvcc as CUDA" OFF)
option(DEBUG_TO_STDERR "Send debug output to stderr instead of log files" OFF)

# New: debug compile mode (separate from DEBUG_LEVEL)
option(DEBUG_COMPILE "Compile with -O0 (debug compile mode)" OFF)

# Debug level: numeric 0..3 (default 0)
set(DEBUG_LEVEL "0" CACHE STRING "Debug level: 0 (none), 1, 2, 3")
set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS 0 1 2 3)

if(NOT DEBUG_LEVEL MATCHES "^[0-3]$")
  message(WARNING "Invalid DEBUG_LEVEL='${DEBUG_LEVEL}'. Resetting to 0.")
  set(DEBUG_LEVEL "0" CACHE STRING "Debug level: 0 (none), 1, 2, 3" FORCE)
endif()

# Make available to sources and flags
add_compile_definitions(DEBUG_LEVEL=${DEBUG_LEVEL})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSAMS_DEBUG=${DEBUG_LEVEL}")

if(ARRAY_BOUNDS_CHECKING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARRAY_BOUNDS_CHECKING")
endif()

if(ARRAY_SET_CHECKER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSET_CHECKER")
endif()

if(MPI)
  set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "MPI C++ compiler" FORCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")

  # If also requesting CUDA/Kokkos CUDA, make mpicxx the CUDA host compiler so nvcc/nvcc_wrapper and Kokkos use it
  if(KOKKOS_CUDA OR NATIVE_CUDA)
    set(CMAKE_CUDA_HOST_COMPILER "mpicxx" CACHE FILEPATH "CUDA host C++ compiler" FORCE)
  endif()
endif()

if(KOKKOS_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_CUDA")
endif()

if(KOKKOS_HIP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_HIP")
endif()

if(KOKKOS_SYCL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_SYCL")
endif()

if(KOKKOS_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DKOKKOS_OPENMP")
endif()

if(NATIVE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS ON)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NATIVE_HIP)
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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set optimization flags depending on DEBUG_COMPILE (separate from DEBUG_LEVEL)
if(DEBUG_COMPILE)
  message(STATUS "Debug compile mode enabled: using -O0 (no optimizations)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fopenmp -DUSE_HDF5")
  # Also set CUDA/HIP flags for debug compile mode if those languages are enabled
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O0 -g")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -g -fopenmp -DUSE_HDF5")
  # Preserve existing CUDA/HIP default flags (they may be extended elsewhere)
endif()

find_package(HDF5 REQUIRED COMPONENTS CXX)

set(SAMS_SOURCES
  src/main.cpp
  src/PhysicsPackages/LARE/control.cpp
  src/PhysicsPackages/LARE/setup.cpp
  src/PhysicsPackages/LARE/boundary.cpp
  src/PhysicsPackages/LARE/lagran.cpp
  src/PhysicsPackages/LARE/diagnostics.cpp
  src/PhysicsPackages/LARE/remap.cpp
  src/PhysicsPackages/LARE/xremap.cpp
  src/PhysicsPackages/LARE/yremap.cpp
  src/PhysicsPackages/LARE/zremap.cpp
)

if(USE_NATIVE_CUDA)
  message(STATUS "Configuring for native CUDA compilation")
  foreach(src_file IN LISTS SAMS_SOURCES)
    set_source_files_properties(${src_file} PROPERTIES LANGUAGE CUDA)
  endforeach()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_CUDA --extended-lambda")
endif()

if(USE_NATIVE_HIP)
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
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/include/harness")
target_include_directories(sams PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sams PRIVATE ${HDF5_LIBRARIES})

if (KOKKOS_CUDA OR KOKKOS_HIP OR KOKKOS_SYCL OR KOKKOS_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_KOKKOS")
  find_package(Kokkos REQUIRED)
  target_link_libraries(sams PRIVATE Kokkos::kokkos)
endif()
