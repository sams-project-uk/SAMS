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
#include <cstring>

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
        template <int rank, int level = 0, typename T_func, typename T_cRange, typename... T_oRanges>
        INLINE DEVICEPREFIX void forEachCore(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges);

        /**
         * CPU serial forEach function
         * Calls the core function for each element in the range.
         * This function is used when the level is not zero
         * i.e. OpenMP parallelism is only used on the outermost loop.
         */
        template <int rank, int level = 0, typename T_func, typename T_cRange, typename... T_oRanges>
        INLINE DEVICEPREFIX void forEachSerial(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges)
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
                        forEachCore<rank, level + 1>(func, tuple, oRanges...);
                    }
                    else
                    {
                        applyToData(func, tuple);
                    }
                }
            }
        }

        /**
         * CPU parallel forEach function
         * Calls the core function for each element in the range.
         * This function is used when the level is zero to apply OpenMP parallelism
         * This function is identical to the serial version except that it uses OpenMP to parallelize the outermost loop.
         * I can't think of an even loosely elegant way to do this without duplicating the code.
         */
        template <int rank, int level = 0, typename T_func, typename T_cRange, typename... T_oRanges>
        DEVICEPREFIX INLINE void forEachParallel(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges)
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
#pragma omp parallel for firstprivate(tuple)
                for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                {
                    GET<level>(tuple) = i;
                    if constexpr (sizeof...(oRanges) > 0)
                    {
                        forEachCore<rank, level + 1>(func, tuple, oRanges...);
                    }
                    else
                    {
                        applyToData(func, tuple);
                    }
                }
            }
        }

        /**
         * Selector function to choose between parallel and serial execution on CPU
         */
        template <int rank, int level, typename T_func, typename T_cRange, typename... T_oRanges>
        DEVICEPREFIX INLINE void forEachCore(
            T_func func,
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> &tuple, T_cRange cRange, T_oRanges... oRanges)
        {
            if constexpr (level == 0)
            {
                forEachParallel<rank, level>(func, tuple, cRange, oRanges...);
            }
            else
            {
                forEachSerial<rank, level>(func, tuple, cRange, oRanges...);
            }
        }

        /**
         * Forward declaration for the core CPU reduction function
         */
        template <int rank, typename T_data, int level = 0, typename T_mapper, typename T_reducer, typename... T_ranges>
        UNREPEATED void reductionCore(T_mapper mapper, T_reducer reducer, const T_data &initialValue, T_data *reductionSites, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple, T_ranges... ranges);

        /**
         * CPU serial reduction function
         * Calls the core function for each element in the range.
         * This function is used when the level is not zero
         * i.e. OpenMP parallelism is only used on the outermost loop.
         */
        template <int rank, typename T_data, int level = 0, typename T_reducer, typename T_mapper, typename T_cRange, typename... T_oRanges>
        INLINE void reductionSerial(
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
                        auto result = applyToData(mapper, tuple);
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
        INLINE void reductionParallel(
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
                T_data iResult = initialValue;
                #pragma omp for
                    for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                    {
                        GET<level>(tuple) = i;
                        if constexpr (sizeof...(oRanges) > 0)
                        {
                            reductionCore<rank, T_data, level + 1>(mapper, reducer, initialValue, &iResult, tuple, oRanges...);
                        }
                        else
                        {
                            auto result = applyToData(mapper, tuple);
                            reducer(iResult, result);
                        }
                    }
                    reducer(reductionSites[getOMPThreadID()], iResult);
                }
            }
        }

        /**
         * CPU reduction wrapper.
         */
        template <int rank, typename T_data, int level, typename T_mapper, typename T_reducer, typename... T_ranges>
        UNREPEATED void reductionCore(T_mapper mapper, T_reducer reducer, const T_data &initialValue, T_data *reductionSites, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple, T_ranges... ranges)
        {
            if constexpr (level==0 && false)
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

       template<typename T_data, int rankS, int rankD, arrayTags tagS, arrayTags tagD>
        UNREPEATED void copyData(portableArray<T_data, rankD, tagD> &destination, const portableArray<T_data, rankS, tagS> &source) {
            //Here in the OpenMP backend, given that we require that all types be trivially copyable,
            //we can just use memcpy to copy the data.
            std::memcpy(destination.data(), source.data(), source.getElements() * sizeof(T_data));
        }

        /**
         * Function to allocate device memory of a fixed number of elements
         * Memory allocated by this function can be unavailable on the host
         */
        template<typename T>
        UNREPEATED T* allocate(size_t elements) {
            return static_cast<T*>(std::malloc(elements * sizeof(T))); // Allocate memory using malloc
        }

        /**
         * Function to allocate shared memory of a fixed number of elements
         * Memory allocated by this function must be accessible from both host and device
         * or this function should return nullptr. If you cannot automatically
         * delete shared memory without KNOWING that it is shared memory, this function
         * should return nullptr.
         */
        template<typename T>
        UNREPEATED T* allocateShared(size_t elements) {
            //Shared data is the same as normal data in OpenMP
            return allocate<T>(elements); // Allocate memory using malloc
        }

        /**
         * Function to deallocate device memory
         */
        template<typename T>
        UNREPEATED void deallocate(T* data) {
            std::free(data);
        }

        UNREPEATED void fence() {
            // In OpenMP, we can use a barrier to synchronize threads
            //#pragma omp barrier
        }

        template <typename T_func, typename... T_ranges>
        UNREPEATED void applyKernel(T_func func, T_ranges... ranges)
        {
            N_ary_tuple_type_t<SIGNED_INDEX_TYPE, sizeof...(ranges)> tuple;
            openmp::forEachCore<sizeof...(ranges)>(func, tuple, ranges...);
        }

        UNREPEATED void initialize(int& argc, char* argv[])
        {
            // OpenMP initialization can be done here if needed
            // For now, we assume OpenMP is already initialized by the compiler/runtime
        }

        namespace atomic {
            /**
             * Atomic addition operation
             * @param target The target variable to add to
             * @param value The value to add
             */
            template <typename T, typename T2>
            DEVICEPREFIX void Add(T& target, const T2 value)
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
            DEVICEPREFIX void And(T& target, const T2 value)
            {
                #pragma omp atomic
                target &= value;
            }

            /**
             * Atomic decrement operation
             * @param target The target variable to decrement
             */
            template <typename T>
            DEVICEPREFIX void Dec(T& target)
            {
                #pragma omp atomic
                --target;
            }

            /**
             * Atomic increment operation
             * @param target The target variable to increment
             */
            template <typename T>            
            DEVICEPREFIX void Inc(T& target)
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
            DEVICEPREFIX void Max(T& target, const T2 value)
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
            DEVICEPREFIX void Min(T& target, const T2 value)
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
            DEVICEPREFIX void Or(T& target, const T2 value)
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
            DEVICEPREFIX void Sub(T& target, const T2 value)
            {
                #pragma omp atomic
                target -= value;
            }

        } // namespace atomic

    } // namespace openmp
} // namespace portableWrapper

#endif
