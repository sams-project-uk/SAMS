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
#ifndef KOKKOSBACKEND_H
#define KOKKOSBACKEND_H

#include "defs.h"
#include "utils.h"
#include "range.h"
#include "demangle.h"

#ifdef USE_KOKKOS

namespace portableWrapper{
    namespace kokkos {
        /**
         * Kokkos parallel forEach function
         * This function uses Kokkos to parallelize the execution of the provided function
         * over the specified ranges. It constructs a Kokkos RangePolicy or MDRangePolicy
         * based on the number of ranges provided and executes the function in parallel.
         */
        template <typename T_func, typename... T_ranges>
        UNREPEATED void forEachKokkosCore(const char *name, T_func func, T_ranges... ranges)
        {
            using iType = int64_t;
            Kokkos::Array<iType, sizeof...(ranges)> starts, ends;
            int i=0;
            ((starts[i] = ranges.lower_bound, ends[i] = ranges.upper_bound + 1, i++), ...);
            auto RangePolicy = [&starts, &ends]()
            {
                if constexpr (sizeof...(ranges) == 1)
                    return Kokkos::RangePolicy<KOKKOS_EXECUTION_SPACE,iType>(starts[0], ends[0]);
                else
                    //This avoids having to reverse the order of the ranges for Kokkos
                    return Kokkos::MDRangePolicy<Kokkos::Rank<sizeof...(T_ranges),Kokkos::Iterate::Right, Kokkos::Iterate::Right>,KOKKOS_EXECUTION_SPACE,iType>(starts, ends);
            }();
            Kokkos::parallel_for(
                name,
                RangePolicy,
                func
            );
        }

        /**
         * Kokkos reducer class for arbitrary types1
         * This wraps a reducer function that simply takes a source and a destination value
         * and combines them into the destination value.
         * This matches the Kokkos reducer concept
         */
        template <typename T, typename combinerFn>
        class kokkosArbitraryReducer
        {
            combinerFn combiner;

            T &value_;
            T initialValue_;

        public:
            typedef kokkosArbitraryReducer reducer;
            typedef T value_type;
            typedef Kokkos::View<value_type, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> result_view_type;

            KOKKOS_INLINE_FUNCTION
            void join(value_type &update, const value_type &input) const
            {
                combiner(update, input);
            }

            KOKKOS_INLINE_FUNCTION
            value_type &reference() const
            {
                return value_;
            }

            KOKKOS_INLINE_FUNCTION
            result_view_type view() const
            {
                return result_view_type(&value_, 1);
            }

            KOKKOS_INLINE_FUNCTION
            void init(value_type &update) const
            {
                update = initialValue_;
            }

            KOKKOS_INLINE_FUNCTION
            kokkosArbitraryReducer(combinerFn combiner, T initialValue, T &value) : combiner(combiner), value_(value), initialValue_(initialValue) {}
        };

        /**
         * Helper function to build a Kokkos reducer from a reducer function and the reduction variable
         */
        template <typename T_iv, typename T, typename T_reducer>
        UNREPEATED auto makeReducer(T_reducer reducer, T_iv initialValue, T &value)
        {
            return kokkosArbitraryReducer<T, T_reducer>(reducer, initialValue, value);
        }

        /**
         * Functor to map arbitrary items to a value and reduce it the first time
         * Has to be a functor because of CUDA's limitations with lambdas
         * @details
         * This functor wraps a mapper function. The mapper function looks like a normal kernel function,
         * but returns the value to be reduced. This is then reduced using the provided reducer function.
         * This is mapped to Kokkos "Index + Reduction" syntax
         */
        template<typename T_data, typename T_mapper, typename T_reducer>
        class kokkosArbitraryMapper{
            T_mapper mapper;
            T_reducer reducer;
            public:
            KOKKOS_INLINE_FUNCTION
            kokkosArbitraryMapper(T_mapper mapper, T_reducer reducer) : mapper(mapper), reducer(reducer) {}
            template <typename... T_items>
            KOKKOS_INLINE_FUNCTION 
            void operator()(T_items&&... items) const
            {
                portableTuple::tuple<T_items...> itemsTuple(items...);
                //Strip off the last item, which is the reduction variable
                auto &reductionValue = portableTuple::get<sizeof...(items) - 1>(itemsTuple);
                //Call the mapper with all but the last item and then use the reducer to combine the result with the reduction variable
                reducer(reductionValue, portableTuple::apply(mapper, portableTuple::tupleRemoveLast(itemsTuple)));
                
            }

        };

        /**
         * Helper function to use type deduction to create a Kokkos mapper
         */
        template <typename T_data, typename T_reducer, typename T_mapper>
        INLINE auto makeMapper(T_mapper mapper, T_reducer reducer)
        {
            return kokkosArbitraryMapper<T_data, T_mapper, T_reducer
            >(mapper, reducer);
        }

        template <typename T_data, typename T_mapper, typename T_reducer, typename... T_ranges>
        UNREPEATED auto reductionKokkosCore(
            T_mapper mapper, T_reducer reducer, T_data initialValue, T_ranges... ranges)
        {
            Kokkos::Array<int64_t, sizeof...(ranges)> starts, ends;
            int i = 0;
            ((starts[i] = ranges.lower_bound, ends[i] = ranges.upper_bound + 1, ++i), ...);
            auto RangePolicy = [&starts, &ends]()
            {
                if constexpr (sizeof...(ranges) == 1)
                    return Kokkos::RangePolicy<KOKKOS_EXECUTION_SPACE>(starts[0], ends[0]);
                else
                    return Kokkos::MDRangePolicy<Kokkos::Rank<sizeof...(T_ranges)>,KOKKOS_EXECUTION_SPACE>(starts, ends);
            }();
            T_data result = initialValue;
            Kokkos::parallel_reduce(
                "reduction",
                RangePolicy,
                makeMapper<T_data>(mapper, reducer),
                makeReducer<T_data>(reducer, initialValue, result)
                );
            return result;
        }

        /**
         * Function to apply a kernel in parallel using Kokkos
         */
        template <typename T_func, typename... T_ranges>
        UNREPEATED void applyKernel(T_func func, T_ranges... ranges)
        {
            kokkos::forEachKokkosCore("Parallel", func, ranges...);
        }

        /**
         * Initialise Kokkos
         */
        UNREPEATED void initialize(int &argc, char *argv[])
        {
            Kokkos::initialize(argc, argv);
        }

        /**
         * Finalize Kokkos
         */
        UNREPEATED void finalize()
        {
            Kokkos::finalize();
        }

        template<typename T>
        UNREPEATED T* allocate(size_t elements) {
            // Allocate memory using Kokkos
            // This can be used for both host and device memory depending on the execution space
                return static_cast<T*>(Kokkos::kokkos_malloc<KOKKOS_EXECUTION_SPACE>(elements * sizeof(T)));
        }

        template<typename T>
        UNREPEATED T* allocateShared(size_t elements) {
            #ifdef KOKKOS_ENABLE_CUDA
            if constexpr (std::is_same_v<KOKKOS_EXECUTION_SPACE, Kokkos::Cuda>)
            {
                return static_cast<T*>(Kokkos::kokkos_malloc<Kokkos::CudaUVMSpace>(elements * sizeof(T)));
            }
            #endif
            #ifdef KOKKOS_ENABLE_HIP
            if constexpr (std::is_same_v<KOKKOS_EXECUTION_SPACE, Kokkos::HIP>)
            {
                return static_cast<T*>(Kokkos::kokkos_malloc<Kokkos::HIPManagedSpace>(elements * sizeof(T)));
            }
            #endif
            #ifdef KOKKOS_ENABLE_SYCL 
            if constexpr (std::is_same_v<KOKKOS_EXECUTION_SPACE, Kokkos::SYCL>)
            {
                return static_cast<T*>(Kokkos::kokkos_malloc<Kokkos::SYCLManagedMemorySpace>(elements * sizeof(T)));
            }
            #endif
            return allocate<T>(elements); // Default case for other execution spaces
        }

        template<typename T>
        UNREPEATED void deallocate(T* data) {
            // Deallocate memory using Kokkos
            if (data != nullptr) {
                Kokkos::kokkos_free<KOKKOS_EXECUTION_SPACE>(data);
            }
        }

        UNREPEATED void fence() {
            // Synchronize the Kokkos execution space
            Kokkos::fence();
        }

        template<typename T_data, int rankS, int rankD, arrayTags tagS, arrayTags tagD>
        UNREPEATED void copyData(portableArray<T_data, rankD, tagD> &destination, const portableArray<T_data, rankS, tagS> &source) {

            using kokkosSource = std::conditional_t<tagS == arrayTags::host, Kokkos::HostSpace, KOKKOS_EXECUTION_SPACE::memory_space>;
            using kokkosDestination = std::conditional_t<tagD == arrayTags::host, Kokkos::HostSpace, KOKKOS_EXECUTION_SPACE::memory_space>;
            
            Kokkos::View<const T_data*, kokkosSource> sourceView(source.data(), source.getElements());
            Kokkos::View<T_data*, kokkosDestination> destinationView(destination.data(), destination.getElements());
            Kokkos::deep_copy(destinationView, sourceView);
            
        }

        /**
         * Function to print Kokkos information
         * This function prints the Kokkos execution space and concurrency information.
         */
        UNREPEATED void printInfo(){
            std::cout << "Kokkos" << std::endl;
            // Get the Kokkos execution space
            auto exec_space = KOKKOS_EXECUTION_SPACE();
            std::cout << "Kokkos execution space: " << exec_space.name() << std::endl;
            std::cout << "Kokkos concurrency: " << exec_space.concurrency() << std::endl;
        }

        template<typename T>
        UNREPEATED auto compare_and_swap(T *ptr, T expected, T desired)
        {
            // Use Kokkos atomic operations to compare and swap
            return Kokkos::atomic_compare_exchange(ptr, expected, desired);
        }
    }; // namespace kokkos
};// namespace portableWrapper

#endif //ifdef USE_KOKKOS
#endif //#ifdef KOKKOSBACKEND_H
