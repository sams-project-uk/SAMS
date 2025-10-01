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
#ifndef PORTABLEWRAPPERFRONTEND_H
#define PORTABLEWRAPPERFRONTEND_H

namespace portableWrapper
{

    /**
     * Function to apply a kernel function in parallel over the specified ranges
     * Kernel function must capture all parameters that it needs BY VALUE and all
     * of those parameters must be trivially copyable.
     * Use the LAMBDA macro to create a lambda function that can be used as the kernel function.
     */
    template <typename T_func, typename... T_ranges>
    inline void applyKernel(T_func func, T_ranges... ranges)
    {
			static_assert(sizeof...(ranges)>0, "Must specify at least one range to apply a kernel over");
#ifdef USE_KOKKOS
        kokkos::applyKernel(std::forward<T_func>(func), ranges...);
#elif defined(USE_CUDA)
        cuda::applyKernel(std::forward<T_func>(func), ranges...);
#elif defined(USE_HIP)
        hip::applyKernel(std::forward<T_func>(func), ranges...);
#else
        openmp::applyKernel(std::forward<T_func>(func), ranges...);
#endif
    }

    /**
     * Version of applyKernel that only ever runs on the host.
     */
    template <typename T_func, typename... T_ranges>
    inline void applyKernelHost(T_func func, T_ranges... ranges){
        N_ary_tuple_type_t<SIGNED_INDEX_TYPE, sizeof...(ranges)> tuple;
        openmp::forEachCore<sizeof...(ranges)>(func, tuple, ranges...);
    }

    /**
     *  Function to apply a reduction operation in parallel over the specified ranges
     * @param mapper Function to map the input data to a value. Same signature as a normal kernel function, but should return the value to be reduced.
     * @param reducer Function to reduce the values. Should take two parameters of the same type as the return value of the mapper function.
     * @param initialValue Initial value for the reduction. This is used to initialize the reduction sites.
     * @param ranges Ranges over which to apply the reduction. The number of ranges must match the rank of the data.
     * */    
    template <typename T_ret, typename T_data_in, typename T_mapper, typename T_reducer, typename... T_ranges>
    inline auto applyReduction(T_mapper mapper, T_reducer reducer, T_data_in initialValue, T_ranges... ranges)
    {

        using T_data = std::conditional_t<std::is_void_v<T_ret>, typename far::callableTraits<T_mapper>::type, T_ret>;

#ifdef USE_KOKKOS
        return kokkos::reductionKokkosCore<T_data>(mapper, reducer, initialValue, ranges...);
#elif defined(USE_CUDA)
        return cuda::applyReduction<T_data>(std::forward<T_mapper>(mapper), std::forward<T_reducer>(reducer), initialValue, ranges...);
#elif defined(USE_HIP)
        return hip::applyReduction<T_data>(std::forward<T_mapper>(mapper), std::forward<T_reducer>(reducer), initialValue, ranges...);
#else
        std::vector<T_data> reductionSites(getOMPMaxThreads(), initialValue);
        N_ary_tuple_type_t<SIGNED_INDEX_TYPE, sizeof...(ranges)> tuple;
        openmp::reductionCore<sizeof...(ranges), T_data>(mapper, reducer, initialValue, reductionSites.data(), tuple, ranges...);
        return reductionSites[0];
#endif
    }

    /**
     *  Function to apply a reduction operation in parallel over the specified ranges
     * @param mapper Function to map the input data to a value. Same signature as a normal kernel function, but should return the value to be reduced.
     * @param reducer Function to reduce the values. Should take two parameters of the same type as the return value of the mapper function.
     * @param initialValue Initial value for the reduction. This is used to initialize the reduction sites.
     * @param ranges Ranges over which to apply the reduction. The number of ranges must match the rank of the data.
     * */    
    template <typename T_data_in, typename T_mapper, typename T_reducer, typename... T_ranges>
    inline typename far::callableTraits<T_mapper>::type applyReductionHost(T_mapper mapper, T_reducer reducer, T_data_in initialValue, T_ranges... ranges)
    {
        using T_data = typename far::callableTraits<T_mapper>::type;
        std::vector<T_data> reductionSites(getOMPMaxThreads(), initialValue);
        N_ary_tuple_type_t<SIGNED_INDEX_TYPE, sizeof...(ranges)> tuple;
        openmp::reductionCore<sizeof...(ranges), T_data>(mapper, reducer, initialValue, reductionSites.data(), tuple, ranges...);
        return reductionSites[0];
    }

    /**
     * Function to provide a fence point for the parallel operations.
     */
    inline void fence()
    {
#ifdef USE_KOKKOS
        kokkos::fence();
#elif defined(USE_CUDA)
        cuda::fence();
#elif defined(USE_HIP)
        hip::fence();
#else
        openmp::fence();
#endif
    }

    /**
     * Fence but only for host operations.
     */
    inline void fenceHost()
    {
        openmp::fence();
    }

    /**
     * Initialise the parallellization environment.
     * If the precompiler flag PRINT_PARALLELIZATION_INFO is set, it will print
     * information about the parallelization mode and the available resources.
     * This function should be called before any parallel operations are performed, but after MPI_Init if MPI is used.
     */

    inline void initialize(int &argc, char *argv[])
    {
#ifdef USE_KOKKOS
        kokkos::initialize(argc, argv);
#elif defined(USE_CUDA)
        cuda::initialize(argc, argv);
#elif defined(USE_HIP)
        hip::initialize(argc, argv);
#else
        openmp::initialize(argc, argv);
#endif

#ifdef USE_MPI
        mpi::initialize(argc, argv);
#endif

        {
#ifdef PRINT_PARALLELIZATION_INFO
        std::cout << "Parallelization mode: ";
#ifdef USE_KOKKOS
        kokkos::printInfo();
#elif defined(USE_CUDA)
        cuda::printInfo();
#elif defined(USE_HIP)
        hip::printInfo();
#else
#ifdef _OPENMP
        std::cout << "OpenMP" << std::endl;
        std::cout << "Number of threads: " << getOMPMaxThreads() << std::endl;
#else
        std::cout << "Serial" << std::endl;
#endif
#endif
#endif
        }
        }

inline void finalize()
{
    #ifdef USE_KOKKOS
    Kokkos::finalize();            
    #endif
}

} // namespace portableWrapper

#endif // PORTABLEWRAPPERFRONTEND_H