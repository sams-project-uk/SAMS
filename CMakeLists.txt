cmake_minimum_required(VERSION 3.12)

option(USE_KOKKOS_CUDA "Enable Kokkos with CUDA backend" OFF)
option(USE_KOKKOS_HIP "Enable Kokkos with HIP backend" OFF)
option(USE_KOKKOS_SYCL "Enable Kokkos with SYCL backend" OFF)
option(USE_KOKKOS_OPENMP "Enable Kokkos with OpenMP backend" OFF)
option(USE_NATIVE_CUDA "Compile all .cpp files with nvcc as CUDA" OFF)

if(USE_KOKKOS_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_CUDA")
endif()

if(USE_KOKKOS_HIP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_HIP")
endif()

if(USE_KOKKOS_SYCL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_SYCL")
endif()

if(USE_KOKKOS_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DKOKKOS_OPENMP")
endif()

if(USE_NATIVE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS on)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(USE_NATIVE_HIP)
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS on)
  set(CMAKE_HIP_ARCHITECTURES native)
endif()

if(USE_NATIVE_CUDA)
  project(SAMS-PP LANGUAGES CXX CUDA)
else()
  project(SAMS-PP LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS on)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -g -fopenmp -DUSE_HDF5")
find_package(HDF5 REQUIRED)

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
target_link_libraries(sams PRIVATE ${HDF5_LIBRARIES})

if (USE_KOKKOS_CUDA OR USE_KOKKOS_HIP OR USE_KOKKOS_SYCL OR USE_KOKKOS_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_KOKKOS")
  find_package(Kokkos REQUIRED)
  target_link_libraries(sams PRIVATE Kokkos::kokkos)
endif()
