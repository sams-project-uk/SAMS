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
#ifndef DEFS_H
#define DEFS_H

#include "harnessDef.h"

#define SIGNED_INDEX_TYPE int64_t
#define UNSIGNED_INDEX_TYPE size_t
#define SIZE_TYPE size_t

#define COUNT_TYPE size_t

//We're using a custom tuple implementation
//That could easily be a bad idea, so this should make
//it easy to switch to std::tuple or thrust::tuple if needed
#define APPLY portableTuple::apply
#define MAKETUPLE portableTuple::make_tuple
#define GET portableTuple::get
#define TUPLE portableTuple::tuple

#ifdef USE_KOKKOS
#include <Kokkos_Core.hpp>
#define UNREPEATED inline
#define DEVICEPREFIX KOKKOS_FUNCTION
#define INLINE KOKKOS_FORCEINLINE_FUNCTION
#define LAMBDA KOKKOS_LAMBDA
#define CLASS_LAMBDA KOKKOS_CLASS_LAMBDA
#define NCLAMBDA(...) [__VA_ARGS__] DEVICEPREFIX
#define FUNCTORMETHODPREFIX KOKKOS_FUNCTION
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
//Kokkos has always used desul for atomic operations inside 
//atomic memory spaces, so we'll just use that here
//If not in C++20 then atomic is not available
#if __cplusplus >= 202002L
#define ATOMIC_REF_DEVICE(T) desul::AtomicRef<T, desul::MemoryOrderRelaxed, desul::MemoryScopeDevice>
#define ATOMIC_REF_HOST(T) std::atomic_ref<T>
#else
#define ATOMIC_REF_DEVICE(T) desul::AtomicRef<T, desul::MemoryOrderRelaxed, desul::MemoryScopeDevice>
#define ATOMIC_REF_HOST(T) void
#endif
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#include <cuda/atomic>
#define DEVICEPREFIX __device__ __host__
#define UNREPEATED inline
#define INLINE inline __attribute__((always_inline))
#define LAMBDA [=] __device__ __host__
#define CLASS_LAMBDA [=, *this] __device__ __host__
#define NCLAMBDA(...) [__VA_ARGS__] __device__ __host__
#define FUNCTORMETHODPREFIX __device__ __host__
#if __cplusplus >= 202002L
#define ATOMIC_REF_DEVICE(T) ::cuda::atomic_ref<T, ::cuda::thread_scope_device>
#define ATOMIC_REF_HOST(T) std::atomic_ref<T>
#else
#define ATOMIC_REF_DEVICE(T) ::cuda::atomic_ref<T, ::cuda::thread_scope_device>
#define ATOMIC_REF_HOST(T) void
#endif
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#define DEVICEPREFIX __device__ __host__
#define UNREPEATED inline
#define INLINE inline __attribute__((always_inline))
#define LAMBDA [=] __device__ __host__
#define CLASS_LAMBDA [=, *this] __device__ __host__
#define NCLAMBDA(...) [__VA_ARGS__] __device__ __host__
#define FUNCTORMETHODPREFIX __device__ __host__
#if __cplusplus >= 202002L
#define ATOMIC_REF_DEVICE(T) __hipc::atomic_ref<T, hip::thread_scope_device>
#define ATOMIC_REF_HOST(T) std::atomic_ref<T>
#else
#define ATOMIC_REF_DEVICE(T) __hipc::atomic_ref<T, hip::thread_scope_device>
#define ATOMIC_REF_HOST(T) void
#endif
#else
#define DEVICEPREFIX
#define UNREPEATED inline
#define INLINE inline __attribute__((always_inline))
#define LAMBDA [=]
#define CLASS_LAMBDA [=, *this]
#define NCLAMBDA(...) [__VA_ARGS__]
#define FUNCTORMETHODPREFIX
#if __cplusplus >= 202002L
#define ATOMIC_REF_DEVICE(T) std::atomic_ref<T>
#define ATOMIC_REF_HOST(T) std::atomic_ref<T>
#else
#define ATOMIC_REF_DEVICE(T) void
#define ATOMIC_REF_HOST(T) void
#endif
#endif

#if defined(USE_KOKKOS) + defined(USE_CUDA) + defined(USE_HIP) > 1
#error "Only one accelerated backend can be specified."
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace portableWrapper
{
    enum arrayTags
    {
        host = 1,
#if defined(USE_CUDA) || defined (USE_KOKKOS) || defined (USE_HIP)
        accelerated = 2,
#else
        accelerated = host, // If no acceleration is available, treat it as a host array
#endif
    };
    template <typename T, int i_rank, arrayTags tag = arrayTags::accelerated>
    class portableArray;
}

#endif
