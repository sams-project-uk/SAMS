#!/bin/bash
KOKKOS_VERSION="4.6.02"

rm -rf kokkos-${KOKKOS_VERSION}

# Call with $1 as a string of CMake flags e.g. -DKOKKOS_ENABLE_CUDA=ON
wget -O kokkos.tar.gz https://github.com/kokkos/kokkos/releases/download/${KOKKOS_VERSION}/kokkos-${KOKKOS_VERSION}.tar.gz
tar xvzf kokkos.tar.gz
cd kokkos-${KOKKOS_VERSION}
mkdir build
cd build
cmake $1 ..
make -j 10
make install