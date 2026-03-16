cmake_minimum_required(VERSION 3.12)

# Move Project declaration up so compilers are detected early
project(SAMS-PP LANGUAGES CXX)

option(ARRAY_BOUNDS_CHECKING "Enable array bounds checking" OFF)
option(ARRAY_SET_CHECKER "Enable array set element checking" OFF)
option(MPI "Enable MPI" OFF)
option(KOKKOS_CUDA "Enable Kokkos with CUDA backend" OFF)
option(KOKKOS_HIP "Enable Kokkos with HIP backend" OFF)
option(KOKKOS_SYCL "Enable Kokkos with SYCL backend" OFF)
option(KOKKOS_OPENMP "Enable Kokkos with OpenMP backend" OFF)
option(NATIVE_CUDA "Compile all .cpp files with nvcc as CUDA" OFF)
option(DEBUG_TO_STDERR "Send debug output to stderr instead of log files" OFF)
option(DEBUG_COMPILE "Compile with -O0 (debug compile mode)" OFF)

# --- THREADS & OPENMP SETUP (Crucial for macOS) ---
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Tell CMake where Homebrew's libomp is located
if(APPLE)
    # Automatically find the path where brew installed libomp
    execute_process(COMMAND brew --prefix libomp 
                    OUTPUT_VARIABLE LIBOMP_PREFIX 
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    
    # Help CMake find the package by setting the root hint
    set(OpenMP_ROOT "${LIBOMP_PREFIX}")
endif()
find_package(OpenMP REQUIRED)

# --- HDF5 SETUP ---
# Search for HDF5 AFTER Threads is found
find_package(HDF5 REQUIRED COMPONENTS CXX)

# --- NetCDF SETUP ---
# Pull in NetCDF C library for TWOFLUID utilities
find_package(NetCDF REQUIRED COMPONENTS C)

# Prefer imported NetCDF targets when available
if(TARGET NetCDF::NetCDF_C)
  set(NETCDF_LINK_TARGET NetCDF::NetCDF_C)
elseif(TARGET NetCDF::NetCDF)
  set(NETCDF_LINK_TARGET NetCDF::NetCDF)
endif()

# Fallback find for library/include if variables are empty
if(NOT NetCDF_INCLUDE_DIRS)
  find_path(NetCDF_INCLUDE_DIRS netcdf.h)
endif()
if(NOT NetCDF_LIBRARIES)
  find_library(NetCDF_LIBRARIES netcdf)
endif()


# Debug level setup
set(DEBUG_LEVEL "0" CACHE STRING "Debug level: 0 (none), 1, 2, 3")
set_property(CACHE DEBUG_LEVEL PROPERTY STRINGS 0 1 2 3)

if(NOT DEBUG_LEVEL MATCHES "^[0-3]$")
  message(WARNING "Invalid DEBUG_LEVEL='${DEBUG_LEVEL}'. Resetting to 0.")
  set(DEBUG_LEVEL "0" CACHE STRING "Debug level: 0 (none), 1, 2, 3" FORCE)
endif()

add_compile_definitions(DEBUG_LEVEL=${DEBUG_LEVEL})
add_compile_definitions(SAMS_DEBUG=${DEBUG_LEVEL})

if(ARRAY_BOUNDS_CHECKING)
  add_compile_definitions(ARRAY_BOUNDS_CHECKING)
endif()

if(ARRAY_SET_CHECKER)
  add_compile_definitions(SET_CHECKER)
endif()

if(MPI)
  find_package(MPI REQUIRED)
  message(STATUS "Enabling MPI support")
  ##set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "MPI C++ compiler")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")

  # If also requesting CUDA/Kokkos CUDA, make mpicxx the CUDA host compiler so nvcc/nvcc_wrapper and Kokkos use it
  if(KOKKOS_CUDA OR NATIVE_CUDA)
##    set(CMAKE_CUDA_HOST_COMPILER "mpicxx" CACHE FILEPATH "CUDA host C++ compiler")
  endif()
endif()

if(KOKKOS_CUDA)
  add_compile_definitions(KOKKOS_CUDA)
endif()

# Language specific logic
if(NATIVE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(DEBUG_TO_STDERR)
  add_compile_definitions(SAMS_DEBUG_TO_STDERR)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Logic
if(DEBUG_COMPILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
endif()

# Always define USE_HDF5
add_compile_definitions(USE_HDF5)

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

# Include Directories
target_include_directories(sams PRIVATE 
    ${HDF5_INCLUDE_DIRS}
    ${NetCDF_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/LARE"
    "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/LARE_neutral"
    "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/LARE/include"
    "${CMAKE_SOURCE_DIR}/src/PhysicsPackages/TWOFLUID"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/LARE"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/LARE_neutral"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/SodShockTube"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/BrioAndWu"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/MHDRotor"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/OrszagTang"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/OrszagTang3D"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/EmeryWindTunnel"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/KarmanVortex"
    "${CMAKE_SOURCE_DIR}/src/InitialConditions/TwoFluidTests"
    "${CMAKE_SOURCE_DIR}/include/harness"
    "${CMAKE_SOURCE_DIR}/include/runner"
    "${CMAKE_SOURCE_DIR}/include/boundaryConditions"
    "${CMAKE_SOURCE_DIR}/include"
)

# --- LINKING ---
# Link OpenMP::OpenMP_CXX instead of using -fopenmp flags
target_link_libraries(sams PRIVATE 
    ${HDF5_LIBRARIES} 
    Threads::Threads
    OpenMP::OpenMP_CXX
)

if(NETCDF_LINK_TARGET)
  target_link_libraries(sams PRIVATE ${NETCDF_LINK_TARGET})
else()
  target_link_libraries(sams PRIVATE ${NetCDF_LIBRARIES})
endif()

if(MPI)
  target_include_directories(sams PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_link_libraries(sams PRIVATE MPI::MPI_CXX)
endif()

if (KOKKOS_CUDA OR KOKKOS_HIP OR KOKKOS_SYCL OR KOKKOS_OPENMP)
  add_compile_definitions(USE_KOKKOS)
  find_package(Kokkos REQUIRED)
  target_link_libraries(sams PRIVATE Kokkos::kokkos)
endif()

