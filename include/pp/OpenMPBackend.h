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
#ifndef OPENMPBACKEND_H
#define OPENMPBACKEND_H

#include "defs.h"
#include "utils.h"
#include "range.h"
#include "array.h"
#include "OpenMPUnrolledBackend.h"
#include <cstring>
#include <vector>

namespace portableWrapper
{
    namespace openmp {

        //Constants for OpenMP version checking
        #define OMP1_0 199810
        #define OMP2_0 200203 //C++ OpenMP 2.0 Fortran = 200011
        #define OMP2_5 200505
        #define OMP3_0 200805
        #define OMP3_1 201107
        #define OMP4_0 201307
        #define OMP4_5 201511
        #define OMP5_0 201811
        #define OMP5_1 202011
        #define OMP5_2 202211
        #define OMP6_0 202311

        // Forward declaration for the core CPU forEach function
        template <int rank, int level = 0, bool serial=false, typename T_func, typename T_cRange, typename... T_oRanges>
        HOSTINLINE HOSTDEVICEPREFIX void forEachCore(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges);

        //And for the applyKernel function
        template <typename T_func, typename... T_ranges>
        HOSTUNREPEATED void applyKernel(T_func func, T_ranges... ranges);

        /**
         * CPU serial forEach function
         * Calls the core function for each element in the range.
         * This function is used when the level is not zero
         * i.e. OpenMP parallelism is only used on the outermost loop.
         */
        template <int rank, int level = 0, bool serial=false, typename T_func, typename T_cRange, typename... T_oRanges>
        HOSTINLINE HOSTDEVICEPREFIX void forEachSerial(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges)
        {
            auto [lower_bound, upper_bound] = getRange(cRange);
            if constexpr (level < rank)
            {
                for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                {
                    GET<level>(tuple) = i;
                    if constexpr (sizeof...(oRanges) > 0)
                    {
                        forEachCore<rank, level + 1, serial>(func, tuple, oRanges...);
                    }
                    else
                    {
                        applyToDataHost(func, tuple);
                    }
                }
            }
        }

        /**
         * CPU vectorized forEach function
         * Calls the core function for each element in the range.
         * This function is used when the level is maximal to allow vectorization
         */
        /*template <int rank, int level = 0, bool serial=false, typename T_func, typename T_cRange, typename... T_oRanges>
        HOSTDEVICEPREFIX HOSTINLINE void forEachVectorized(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges)
        {
            auto [lower_bound, upper_bound] = getRange(cRange);
            if constexpr (level < rank)
            {
                #pragma omp simd
                for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                {
                    GET<level>(tuple) = i;
                    if constexpr (sizeof...(oRanges) > 0)
                    {
                        forEachCore<rank, level + 1, serial>(func, tuple, oRanges...);
                    }
                    else
                    {
                        applyToDataHost(func, tuple);
                    }
                }
            }
        }*/

        /**
         * CPU parallel forEach function
         * Calls the core function for each element in the range.
         * This function is used when the level is zero to apply OpenMP parallelism
         * This function is identical to the serial version except that it uses OpenMP to parallelize the outermost loop.
         * I can't think of an even loosely elegant way to do this without duplicating the code.
         */
        template <int rank, int level = 0, bool serial = false, typename T_func, typename T_cRange, typename... T_oRanges>
        HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges)
        {
            auto lbub = getRange(cRange);
            SIGNED_INDEX_TYPE lower_bound = lbub.first;
            SIGNED_INDEX_TYPE upper_bound = lbub.second;
            if constexpr (level < rank)
            {
#pragma omp parallel 
                {
                    auto local_tuple = tuple;
                    #pragma omp for
                    for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                    {
                        GET<level>(tuple) = i;
                        if constexpr (sizeof...(oRanges) > 0)
                        {
                            forEachCore<rank, level + 1, serial>(func, tuple, oRanges...);
                        }
                        else
                        {
                            applyToDataHost(func, tuple);
                        }
                    }
                }
            }
        }

        /**
         * Selector function to choose between parallel and serial execution on CPU
         */
        template <int rank, int level, bool serial, typename T_func, typename T_cRange, typename... T_oRanges>
        HOSTDEVICEPREFIX HOSTINLINE void forEachCore(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges)
        {

            #ifndef NO_NATIVE_UNROLL
            //If unrolling is enabled, use the unrolled version (if possible)
            if constexpr(!serial && rank <= openmp::UNROLL_LIMIT) {
                forEachParallel(func, cRange, oRanges...);
            } else 
            #endif
            {
                if constexpr (level == 0 && !serial)
                {
                    forEachParallel<rank, level, serial>(func, tuple, cRange, oRanges...);
                }
                /*else if constexpr (level == rank - 1)
                {
                    forEachVectorized<rank, level, serial>(func, tuple, cRange, oRanges...);
                }*/
                else
                {
                    forEachSerial<rank, level, serial>(func, tuple, cRange, oRanges...);
                }
            }
        }

        /**
         * Forward declaration for the core CPU reduction function
         */
        template <int rank, typename T_data, int level = 0, typename T_mapper, typename T_reducer, typename... T_ranges>
        HOSTUNREPEATED void reductionCore(T_mapper mapper, T_reducer reducer, const T_data &initialValue, T_data *reductionSites, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple, T_ranges... ranges);

        /**
         * CPU serial reduction function
         * Calls the core function for each element in the range.
         * This function is used when the level is not zero
         * i.e. OpenMP parallelism is only used on the outermost loop.
         */
        template <int rank, typename T_data, int level = 0, typename T_reducer, typename T_mapper, typename T_cRange, typename... T_oRanges>
        HOSTINLINE void reductionSerial(
            T_mapper mapper,
            T_reducer reducer,
            const T_data & initialValue,
            T_data *reductionSites,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple, T_cRange cRange, T_oRanges... oRanges)
        {
            SIGNED_INDEX_TYPE lower_bound, upper_bound;
            if constexpr (std::is_same_v<T_cRange, Range>)
            {
                lower_bound = cRange.lower_bound;
                upper_bound = cRange.upper_bound;
            }
            else
            {
                lower_bound = 1;
                upper_bound = cRange;
            }
            if constexpr (level < rank)
            {
                for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                {
                    GET<level>(tuple) = i;
                    if constexpr (sizeof...(oRanges) > 0)
                    {
                        reductionCore<rank, T_data, level + 1>(mapper, reducer, initialValue, reductionSites, tuple, oRanges...);
                    }
                    else
                    {
                        auto result = applyToDataHost(mapper, tuple);
                        //Either this is running fully serial, in which case 0 is correct
                        //Or you are at a lower level of parallel reduction in which case you are using
                        //a thread local reduction variable - it will be put back into the true reduciton variable in reductionParallel
                        reducer(reductionSites[getOMPThreadID()], result);
                    }
                }
            }
        }

        /**
         * CPU parallel reduction function
         * Calls the core function for each element in the range.
         * This function is used when the level is zero to apply OpenMP parallelism
         * This function is identical to the serial version except that it uses OpenMP to parallelize the outermost loop.
         * I can't think of an even loosely elegant way to do this without duplicating the code.
         */
        template <int rank, typename T_data, int level = 0, typename T_reducer, typename T_mapper, typename T_cRange, typename... T_oRanges>
        HOSTINLINE void reductionParallel(
            T_mapper mapper,
            T_reducer reducer,
            const T_data & initialValue,
            T_data *reductionSites,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple, T_cRange cRange, T_oRanges... oRanges)
        {
            SIGNED_INDEX_TYPE lower_bound, upper_bound;
            if constexpr (std::is_same_v<T_cRange, Range>)
            {
                lower_bound = cRange.lower_bound;
                upper_bound = cRange.upper_bound;
            }
            else
            {
                lower_bound = 1;
                upper_bound = cRange;
            }
            if constexpr (level < rank)
            {
#pragma omp parallel firstprivate(tuple)
            {
                int tid = getOMPThreadID();
                int nthreads = static_cast<int>(getOMPMaxThreads());
                std::vector<T_data> iResult(nthreads, initialValue);
                #pragma omp for
                    for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                    {
                        GET<level>(tuple) = i;
                        if constexpr (sizeof...(oRanges) > 0)
                        {
                            reductionCore<rank, T_data, level + 1>(mapper, reducer, initialValue, iResult.data(), tuple, oRanges...);

                        }
                        else
                        {
                            auto result = applyToDataHost(mapper, tuple);
                            reducer(iResult, result);
                        }
                    }
                    #pragma omp barrier
                    for (int stride = 1; stride < nthreads; stride <<= 1) {
                        if ((tid % (stride << 1)) == 0) {
                            int other = tid + stride;
                            if (other < nthreads) {
                                reducer(iResult[tid], iResult[other]);
                            }
                        }
                        #pragma omp barrier
                    }

                    // write per-thread final slot out
                    reductionSites[tid] = iResult[tid];
                }
            }
        }

        /**
         * CPU reduction wrapper.
         */
        template <int rank, typename T_data, int level, typename T_mapper, typename T_reducer, typename... T_ranges>
        HOSTUNREPEATED void reductionCore(T_mapper mapper, T_reducer reducer, const T_data &initialValue, T_data *reductionSites, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple, T_ranges... ranges)
        {
            if constexpr (level==0)
            {
                reductionParallel<rank, T_data, level>(mapper, reducer, initialValue, reductionSites, tuple, ranges...);
            }
            else
            {
                reductionSerial<rank, T_data, level>(mapper, reducer, initialValue, reductionSites, tuple, ranges...);
            }
            if constexpr(level == 0) {
                size_t numThreads = getOMPMaxThreads();
                size_t offset = numThreads;
                while(true)  {
                    if (numThreads & 0x1 && numThreads > 1) {
                        reducer(reductionSites[0], reductionSites[numThreads-1]);
                    }
                    numThreads /= 2;
                    if (numThreads == 0) break;
                    offset/= 2;
                    #pragma omp parallel num_threads(numThreads)
                    {
                        int i = getOMPThreadID();
                        reducer(reductionSites[i], reductionSites[i+offset]);
                    }
                }
            }
        }

    /**
     * Functor to assign one portableArray to another.
     */
    template<typename T, int rank, arrayTags arrayTag>
      struct trivialAssignArray{
        using pa = portableArray<T, rank, arrayTag>;
        pa dest;
        pa src;

        INLINE trivialAssignArray(pa &dest, const pa &src)
          : dest(dest), src(src) {}

        template<typename... T_indices>
        INLINE void operator()(T_indices... indices) const
          {
            dest.getZB(indices...) = src.getZB(indices...); // Use the overloaded operator() to assign the value
          }
      };

        template<typename T=void, int rank=0, arrayTags tag=arrayTags::host>
            HOSTFLATTEN void trivialAssign(portableArray<T, rank, tag> dest, const portableArray<T, rank, tag> &src) {
            if ((&dest) != (&src))
            {
                if (dest.getElements() != src.getElements()) {
                throw std::runtime_error("Source and destination arrays must have the same number of elements.");
                }
                std_N_ary_tuple_type_t<Range,rank> ranges;
                portableWrapper::detail::arrayToRangesZB(src, ranges);
                auto tpl = std::tuple_cat(
                    std::make_tuple(trivialAssignArray(dest, src)),
                    ranges
                    );
                std::apply([](auto&&... args) {
                    ::portableWrapper::openmp::applyKernel(args...);
                }, tpl);
            }
            }

       template<typename T_data, int rankS, int rankD, arrayTags tagS, arrayTags tagD>
        HOSTUNREPEATED void copyData(portableArray<T_data, rankD, tagD> &destination, const portableArray<T_data, rankS, tagS> &source) {
            //Here in the OpenMP backend, given that we require that all types be trivially copyable,
            //we can just use memcpy to copy the data.
            if constexpr(std::is_trivially_copyable_v<T_data> && portableArray<T_data, rankD, tagD>::rowMajor() == portableArray<T_data, rankS, tagS>::rowMajor())
            {
                if (source.isContiguous() && destination.isContiguous()){
                    std::memcpy(destination.data(), source.data(), source.getElements() * sizeof(T_data));
                } else {
                    trivialAssign(destination, source);
                }
            } else {
                trivialAssign(destination, source);
            }
        }

        /**
         * Function to allocate device memory of a fixed number of elements
         * Memory allocated by this function can be unavailable on the host
         */
        template<typename T>
        HOSTUNREPEATED T* allocate(size_t elements) {
            //return static_cast<T*>(aligned_alloc(64, elements * sizeof(T)));
            return static_cast<T*>(std::malloc(elements * sizeof(T))); // Allocate memory using malloc
        }

        /**
         * Funtion to allocate memory of a fixed number of elements with a given alignment
         * Memory allocated by this function can be unavailable on the host
         */
        template<typename T>
        HOSTUNREPEATED T* allocateAligned(size_t elements, size_t alignment) {
            return static_cast<T*>(aligned_alloc(alignment, elements * sizeof(T))); // Allocate aligned memory
        }
            

        /**
         * Function to allocate shared memory of a fixed number of elements
         * Memory allocated by this function must be accessible from both host and device
         * or this function should return nullptr. If you cannot automatically
         * delete shared memory without KNOWING that it is shared memory, this function
         * should return nullptr.
         */
        template<typename T>
        HOSTUNREPEATED T* allocateShared(size_t elements) {
            //Shared data is the same as normal data in OpenMP
            return allocate<T>(elements); // Allocate memory using malloc
        }

        /**
         * Function to deallocate device memory
         */
        template<typename T>
        HOSTUNREPEATED void deallocate(T* data) {
            std::free(data);
        }

       HOSTUNREPEATED void fence() {
            // In OpenMP, we can use a barrier to synchronize threads
            //#pragma omp barrier
        }

        template <typename T_func, typename... T_ranges>
        HOSTUNREPEATED void applyKernel(T_func func, T_ranges... ranges)
        {
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, sizeof...(ranges)> tuple;
            openmp::forEachCore<sizeof...(ranges), 0, false>(func, tuple, ranges...);
        }

        template<typename T_func, typename... T_ranges>
        HOSTUNREPEATED void applyKernelSerial(T_func func, T_ranges... ranges)
        {
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, sizeof...(ranges)> tuple;
            openmp::forEachCore<sizeof...(ranges), 0, true>(func, tuple, ranges...);
        }

        HOSTUNREPEATED void initialize([[maybe_unused]] int& argc, [[maybe_unused]]char* argv[])
        {
            // OpenMP initialization can be done here if needed
            // For now, we assume OpenMP is already initialized by the compiler/runtime
        }

        HOSTUNREPEATED void printInfo()
        {
            SAMS::cout << "OpenMP" << std::endl;
            SAMS::cout << "OpenMP version: ";
#ifdef _OPENMP
            SAMS::cout << _OPENMP << std::endl;
#else
            SAMS::cout << "Not defined" << std::endl;
#endif
            SAMS::cout << "Number of threads: " << getOMPMaxThreads() << std::endl;
#ifndef NO_NATIVE_UNROLL
            SAMS::cout << "Loop unrolling enabled up to rank " << openmp::UNROLL_LIMIT << std::endl;
#ifdef NATIVE_LOOP_COLLAPSE
            SAMS::cout << "Loop collapse enabled beyond rank " << NATIVE_LOOP_COLLAPSE << std::endl;
#endif
#else
            SAMS::cout << "Native OpenMP loop unrolling disabled. WARNING This is likely to cause performance degradation." << std::endl;
#endif
        }

        namespace atomic {
            /**
             * Atomic addition operation
             * @param target The target variable to add to
             * @param value The value to add
             */
            template <typename T, typename T2>
            HOSTDEVICEPREFIX void Add(T& target, const T2 value)
            {
                #pragma omp atomic
                target += value;
            }

            /**
             * Atomic and operation
             * @param target The target variable to perform the operation on
             * @param value The value to and with
             */
            template <typename T, typename T2>            
            HOSTDEVICEPREFIX void And(T& target, const T2 value)
            {
                #pragma omp atomic
                target &= value;
            }

            /**
             * Atomic decrement operation
             * @param target The target variable to decrement
             */
            template <typename T>
            HOSTDEVICEPREFIX void Dec(T& target)
            {
                #pragma omp atomic
                --target;
            }

            /**
             * Atomic increment operation
             * @param target The target variable to increment
             */
            template <typename T>            
            HOSTDEVICEPREFIX void Inc(T& target)
            {
                #pragma omp atomic
                ++target;
            }

            /**
             * Atomic Max operation
             * @param target The target variable to perform the operation on
             * @param value The value to compare with
             */
            template< typename T, typename T2>            
            HOSTDEVICEPREFIX void Max(T& target, const T2 value)
            {
                //Check for OpenMP 5.1 or higher
                #if defined(_OPENMP) && (_OPENMP >= OMP5_1)
                #pragma omp atomic compare
                target = value>target ? value : target;
                #else
                #pragma omp critical
                {
                    target = value>target ? value : target;
                }
                #endif
            }

            /**
             * Atomic Min operation
             * @param target The target variable to perform the operation on
             * @param value The value to compare with
             */
            template<typename T, typename T2>            
            HOSTDEVICEPREFIX void Min(T& target, const T2 value)
            {
                //Check for OpenMP 5.1 or higher
                #if defined(_OPENMP) && (_OPENMP >= OMP5_1)
                #pragma omp atomic compare
                target = value<target ? value : target;
                #else
                #pragma omp critical
                {
                    target = value<target ? value : target;
                }
                #endif
            }

            /**
             * Atomic OR operation
             * @param target The target variable to perform the operation on
             * @param value The value to or with
             */
            template<typename T, typename T2>            
            HOSTDEVICEPREFIX void Or(T& target, const T2 value)
            {
                #pragma omp atomic
                target |= value;
            }

            /**
             * Atomic subtraction operation
             * @param target The target variable to subtract from
             * @param value The value to subtract
             */
            template<typename T, typename T2>            
            HOSTDEVICEPREFIX void Sub(T& target, const T2 value)
            {
                #pragma omp atomic
                target -= value;
            }

        } // namespace atomic

    } // namespace openmp
} // namespace portableWrapper

#endif
