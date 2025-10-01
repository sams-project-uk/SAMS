/*
 *    Copyright 2025 SAMS Team
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef SAMS_HARNESSDEF_H
#define SAMS_HARNESSDEF_H

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#if defined(KOKKOS_CUDA)
#define KOKKOS_EXECUTION_SPACE Kokkos::Cuda
#elif defined(KOKKOS_HIP)
#define KOKKOS_EXECUTION_SPACE Kokkos::HIP
#elif defined(KOKKOS_SYCL)
#define KOKKOS_EXECUTION_SPACE Kokkos::SYCL
#elif defined(KOKKOS_OPENMP)
#define KOKKOS_EXECUTION_SPACE Kokkos::OpenMP
#else
#define KOKKOS_EXECUTION_SPACE Kokkos::DefaultExecutionSpace
#endif
#endif

namespace SAMS {

    /**
     * Maximum rank of a variable
     */
    constexpr int MAX_RANK = 7;

} //namespace SAMS

#endif