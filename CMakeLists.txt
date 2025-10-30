cmake_minimum_required(VERSION 3.18)

option(USE_KOKKOS_CUDA "Enable Kokkos with CUDA backend" OFF)
option(USE_KOKKOS_HIP "Enable Kokkos with HIP backend" OFF)
option(USE_KOKKOS_SYCL "Enable Kokkos with SYCL backend" OFF)
option(USE_KOKKOS_OPENMP "Enable Kokkos with OpenMP backend" OFF)
option(USE_NATIVE_CUDA "Compile all .cpp files with nvcc as CUDA" OFF)

set(USE_KOKKOS False)
if(USE_KOKKOS_CUDA)
  set(USE_KOKKOS True)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_CUDA")
endif()

if(USE_KOKKOS_HIP)
  set(USE_KOKKOS True)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_HIP")
endif()

if(USE_KOKKOS_SYCL)
  set(USE_KOKKOS True)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_SYCL")
endif()

if(USE_KOKKOS_OPENMP)
  set(USE_KOKKOS True)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DKOKKOS_OPENMP")
endif()

if(USE_NATIVE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS ON)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(USE_NATIVE_HIP)
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS ON)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DUSE_HDF5")
find_package(HDF5 REQUIRED)

if (USE_KOKKOS)
  enable_language(CXX)
  find_package(Kokkos REQUIRED CONFIG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_KOKKOS")
  # NOTE(cmo): Kokkos compiling source as CUDA gives the best
  # kokkos/vscode/devcontainer intellisense experience.
  if (${Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE})
    enable_language(${Kokkos_COMPILE_LANGUAGE})
    # NOTE(cmo): Hack for older CMake versions
    if (NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
      set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
    endif()
    set(CMAKE_${Kokkos_COMPILE_LANGUAGE}_FLAGS "${CMAKE_${Kokkos_COMPILE_LANGUAGE}_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES})
    set(CMAKE_${Kokkos_COMPILE_LANGUAGE}_STANDARD ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_STANDARD})
  endif()
elseif(NOT (USE_NATIVE_CUDA OR USE_NATIVE_HIP))
  find_package(OpenMP REQUIRED)
endif()

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

if (NOT (USE_KOKKOS OR USE_NATIVE_CUDA OR USE_NATIVE_HIP))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
  target_link_libraries(sams PRIVATE OpenMP::OpenMP_CXX)
endif()

if (USE_KOKKOS)
  if (${Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE})
    foreach(src_file IN LISTS SAMS_SOURCES)
      set_source_files_properties(${src_file} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
    endforeach()
  endif()
  target_link_libraries(sams PRIVATE Kokkos::kokkos)
endif()
