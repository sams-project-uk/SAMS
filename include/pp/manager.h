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
#ifndef MANAGER_H
#define MANAGER_H

#include <map>

#include "defs.h"
#include "range.h"
#include "utils.h"

//Agnostic tools are things like applyToData, which can be used
//to call a function/lambda/functor with a tuple of arguments.
//These tools are used by the backends to implement the forEach and reduction functions.
#include "agnosticTools.h"

//Now import the actual backends
#include "cudaBackend.h"
#include "hipBackend.h"
#include "KokkosBackend.h"
#include "OpenMPBackend.h"
#include "array.h"

//#include "pwmpi.h"

namespace portableWrapper {

    /**
     * Factory class to create and manage portableArray instances.
     * This is needed because of the requirement for trivial copyability
     * and hence trivial destructors for CUDA memory, so when the wrapper is captured
     * in a lambda, it can be copied without issues.
     */
    class portableArrayManager
    {
        
    public:
        /**
         * Functor class to initialize data using placement new.
         */
        template<typename T>
        struct placementNewFunctor{
            T *data;

            FUNCTORMETHODPREFIX placementNewFunctor(T *data)
                : data(data) {}

            FUNCTORMETHODPREFIX void operator()(SIGNED_INDEX_TYPE index) const
            {
                new (&data[index]) T();
            }
        };

        template<typename T>
        struct manualDestructorFunctor
        {
            T *data;

            FUNCTORMETHODPREFIX manualDestructorFunctor(T *data)
                : data(data) {}

            FUNCTORMETHODPREFIX void operator()(SIGNED_INDEX_TYPE index) const
            {
                data[index].~T(); // Call the destructor manually
            }
        };


        /**
         * Class representing arbitrary data and how to delete it when needed
         * Mucky type erasure ahead!
         */
        class destructingData
        {
            void *data;
            size_t elements;
            void (*deleter)(void *, size_t, arrayTags, bool) = nullptr;
            arrayTags tag;
            bool owned = true;
            
        public:
            template <typename T>
            destructingData(T *data, size_t elements, arrayTags tag, bool owned = true)
                : data(data), tag(tag), elements(elements), owned(owned)
            { 
                deleter = [](void *data, size_t elements, arrayTags tag, bool owned)
                {
                    if (data == nullptr) return; // Nothing to deallocate
                    if constexpr(!std::is_trivially_destructible_v<T>)
                    {
                        // If T is not trivially destructible, we need to call the destructor manually
                        applyKernel(manualDestructorFunctor<T>(static_cast<T *>(data)), Range(0, elements - 1));
                    }
                    if (!owned) return; // Do not deallocate if we do not own the data
                    if (tag == arrayTags::host)
                    {
                        openmp::deallocate(static_cast<T *>(data));
                    }
                    else 
                    {
                        #ifdef USE_KOKKOS
                        kokkos::deallocate(static_cast<T *>(data));
                        #elif defined(USE_CUDA)
                        cuda::deallocate(static_cast<T *>(data));
                        #elif defined(USE_HIP)
                        hip::deallocate(static_cast<T *>(data));
                        #else
                        openmp::deallocate(static_cast<T *>(data));
                        #endif
                    }
                };
            }
            destructingData(destructingData &&other) noexcept
                : data(other.data), deleter(other.deleter), tag(other.tag), elements(other.elements), owned(other.owned)
            {
                other.data = nullptr;
                other.deleter = nullptr;
            }
            destructingData(const destructingData &) = delete; // Disable copy constructor
            ~destructingData()
            {
                if (deleter)
                {
                    deleter(data, elements, tag, owned);
                }
            }
        }; // class destructingData

        /**
         * Map mapping the raw data pointer to the destructingData object.
         * This is used to keep track of the allocated memory and to allow
         * manual deallocation and reallocation of the data.
         */
        std::map<void *, destructingData> destructors;

        template<arrayTags tag, typename T>
        void storeData(T* data, size_t elements)
        {
            if (data != nullptr)
            {
                destructors.emplace(data, destructingData(data, elements, tag));
            }
            else
            {
                throw std::runtime_error("Failed to store data pointer in portableArrayManager.");
            }
        }

        template<arrayTags tag, typename T>
        void storeWrappedData(T* data, size_t elements)
        {
            if (data != nullptr)
            {
                destructors.emplace(data, destructingData(data, elements, tag, false));
            }
            else
            {
                throw std::runtime_error("Failed to store data pointer in portableArrayManager.");
            }
        }

        template <bool shared, typename T, arrayTags tag, typename... T_ranges>
        T* allocateCore(portableArray<T, sizeof...(T_ranges), tag> &wrapper, T_ranges... ranges)
        {
            wrapper.setSizes(ranges...);
            SIZE_TYPE elements = wrapper.getElements();
            wrapper.manage(this);
            T* data_;
            if constexpr(tag == arrayTags::host)
            {
                data_ = openmp::allocate<T>(elements);
            }
            else
            {
                if (shared)
                {
                    #ifdef USE_KOKKOS
                        data_ = kokkos::allocateShared<T>(elements);
                    #elif defined(USE_CUDA)
                        data_ = cuda::allocateShared<T>(elements);
                    #elif defined(USE_HIP)
                        data_ = hip::allocateShared<T>(elements);
                    #else
                        data_ = openmp::allocateShared<T>(elements);
                    #endif
                } else {
                    #ifdef USE_KOKKOS
                        data_ = kokkos::allocate<T>(elements);
                    #elif defined(USE_CUDA)
                        data_ = cuda::allocate<T>(elements);
                    #elif defined(USE_HIP)
                        data_ = hip::allocate<T>(elements);
                    #else
                        data_ = openmp::allocate<T>(elements);
                    #endif
                }
            }
            if constexpr (!std::is_trivially_default_constructible_v<T>)
            {
                // If T is not trivially default constructible, we need to initialize the data
                // using placement new.
                applyKernel(placementNewFunctor<T>(data_), Range(0, elements - 1));
            }
            wrapper.bind(data_);
            return data_;
        }

        template <bool shared, typename T, int rank, arrayTags tag, typename T_lower, typename T_upper>
        T* allocateCore(portableArray<T, rank, tag> &wrapper, const T_lower *lbounds, const T_upper *ubounds)
        {
            static_assert(std::is_default_constructible_v<T>, "Can only allocate arrays of types that have parameterless constructors.");
            wrapper.setSizes(lbounds, ubounds);
            SIZE_TYPE elements = wrapper.getElements();
            T* data_;
            if constexpr(tag == arrayTags::host)
            {
                data_ = openmp::allocate<T>(elements);
            }
            else
            {
                if constexpr(shared)
                {
                    #ifdef USE_KOKKOS
                        data_ = kokkos::allocateShared<T>(elements);
                    #elif defined(USE_CUDA)
                        data_ = cuda::allocateShared<T>(elements);
                    #elif defined(USE_HIP)
                        data_ = hip::allocateShared<T>(elements);
                    #else
                        data_ = openmp::allocateShared<T>(elements);
                    #endif
                } else {
                    #ifdef USE_KOKKOS
                        data_ = kokkos::allocate<T>(elements);
                    #elif defined(USE_CUDA)
                        data_ = cuda::allocate<T>(elements);
                    #elif defined(USE_HIP)
                        data_ = hip::allocate<T>(elements);
                    #else
                        data_ = openmp::allocate<T>(elements);
                    #endif
                }
            }
            wrapper.bind(data_);
            return data_;
        }


        template <bool shared, typename T, arrayTags tag, typename... T_ranges>
        T* wrapCore(portableArray<T, sizeof...(T_ranges), tag> &wrapper, T* data_, T_ranges... ranges)
        {
            wrapper.setSizes(ranges...);
            SIZE_TYPE elements = wrapper.getElements();
            wrapper.manage(this);
            if constexpr (!std::is_trivially_default_constructible_v<T>)
            {
                // If T is not trivially default constructible, we need to initialize the data
                // using placement new.
                applyKernel(placementNewFunctor<T>(data_), Range(0, elements - 1));
            }
            wrapper.bind(data_);
            return data_;
        }

        template <bool shared, typename T, int rank, arrayTags tag, typename T_lower, typename T_upper>
        T* wrapCore(portableArray<T, rank, tag> &wrapper, T* data_, const T_lower *lbounds, const T_upper *ubounds)
        {
            static_assert(std::is_default_constructible_v<T>, "Can only allocate arrays of types that have parameterless constructors.");
            wrapper.setSizes(lbounds, ubounds);
            SIZE_TYPE elements = wrapper.getElements();
            if constexpr (!std::is_trivially_default_constructible_v<T>)
            {
                // If T is not trivially default constructible, we need to initialize the data
                // using placement new.
                applyKernel(placementNewFunctor<T>(data_), Range(0, elements - 1));
            }
            wrapper.bind(data_);
            return data_;
        }

        public:

        //Allocate a portableArray with the specified ranges
        template <typename T, arrayTags tag, typename... T_ranges>
        void allocate(portableArray<T, sizeof...(T_ranges), tag> &wrapper, T_ranges... ranges)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            T* data_ = allocateCore<false>(wrapper, ranges...);
            storeData<tag>(data_, wrapper.getElements());
        }

        //Alocate returning an array
        template<typename T, arrayTags tag = arrayTags::accelerated, typename... T_ranges>
        auto allocate(T_ranges... ranges)
        {            
            portableArray<T, sizeof...(ranges), tag> wrapper;
            allocate(wrapper, ranges...);
            return wrapper;
        }


        //Wrap some existing memory in a portableArray with the specified ranges
        template <typename T, arrayTags tag, typename... T_ranges>
        void wrap(portableArray<T, sizeof...(T_ranges), tag> &wrapper, T* data_,T_ranges... ranges)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            wrapCore<false>(wrapper, data_, ranges...);
            storeWrappedData<tag>(data_, wrapper.getElements());
        }

        //Wrap returning an array
        template<typename T, arrayTags tag = arrayTags::accelerated, typename... T_ranges>
        auto wrap(T* data_, T_ranges... ranges)
        {   
            portableArray<T, sizeof...(ranges), tag> wrapper;
            wrap(wrapper, data_, ranges...);
            return wrapper;
        }

        template<typename T, arrayTags tag, typename... T_ranges>
        void allocateManaged(portableArray<T, sizeof...(T_ranges), tag> &wrapper, T_ranges... ranges)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            T* data_ = allocateCore<true>(wrapper, ranges...);
			wrapper.setManaged(true);
            storeData<tag>(data_, wrapper.getElements());
        }

        template<typename T, arrayTags tag = arrayTags::accelerated, typename... T_ranges>
        auto allocateManaged(T_ranges... ranges)
        {
            portableArray<T, sizeof...(ranges), tag> wrapper;
            allocateManaged(wrapper, ranges...);
            return wrapper;
        }

        template<typename T, arrayTags tag, typename... T_ranges>
        void wrapManaged(portableArray<T, sizeof...(T_ranges), tag> &wrapper, T* data_, T_ranges... ranges)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            wrapCore<true>(wrapper, data_, ranges...);
			wrapper.setManaged(true);
            storeWrappedData<tag>(data_, wrapper.getElements());
        }

        template<typename T, arrayTags tag = arrayTags::accelerated, typename... T_ranges>
        auto wrapManaged(T* data_, T_ranges... ranges)
        {

            portableArray<T, sizeof...(ranges), tag> wrapper;
            wrapManaged(wrapper, data_, ranges...);
            return wrapper;
        }

        //Allocate a portableArray with the specified lower and upper bounds
        template<typename T, int rank ,arrayTags tag, typename T_bounds>
        void allocate(portableArray<T, rank, tag> &wrapper, const T_bounds *lbounds, const T_bounds *ubounds)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            T* data_ = allocateCore<false>(wrapper, lbounds, ubounds);
            storeData<tag>(data_, wrapper.getElements());
        }

        template<typename T, int rank, arrayTags tag, typename T_bounds>
        void allocateManaged(portableArray<T, rank, tag> &wrapper, const T_bounds *lbounds, const T_bounds *ubounds)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            T* data_ = allocateCore<true>(wrapper, lbounds, ubounds);
			wrapper.setManaged(true);
            storeData<tag>(data_, wrapper.getElements());
        }

        //Wrap a portableArray with the specified lower and upper bounds
        template<typename T, int rank ,arrayTags tag, typename T_bounds>
        void wrap(portableArray<T, rank, tag> &wrapper, T* data_, const T_bounds *lbounds, const T_bounds *ubounds)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            wrapCore<false>(wrapper, data_, lbounds, ubounds);
            storeWrappedData<tag>(data_, wrapper.getElements());
        }

        template<typename T, int rank, arrayTags tag, typename T_bounds>
        void wrapManaged(portableArray<T, rank, tag> &wrapper, T* data_, const T_bounds *lbounds, const T_bounds *ubounds)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            wrapCore<true>(wrapper, lbounds, ubounds);
			wrapper.setManaged(true);
            storeWrappedData<tag>(data_, wrapper.getElements());
        }

        //Allocate an array matching the size of another array
        template <typename T, int rank, arrayTags tag, typename T2, arrayTags tag2>
        void mold(portableArray<T, rank, tag> &wrapper, const portableArray<T2, rank, tag2> &source)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            SIGNED_INDEX_TYPE lower_bound[rank];
            SIGNED_INDEX_TYPE upper_bound[rank];
            for (int i = 0; i < rank; ++i)
            {
                lower_bound[i] = source.getLowerBound(i);
                upper_bound[i] = source.getUpperBound(i);
            }
            this->allocate(wrapper, lower_bound, upper_bound);
        }

        //Allocate and return an array matching the size of another array
        template <typename T=void, arrayTags tag = arrayTags::accelerated, int rank, typename T2, arrayTags tag2>
        auto mold(const portableArray<T2, rank, tag2> &source)
        {
            using Tdest = std::conditional_t<std::is_void_v<T>, T2, T>;
            portableArray<Tdest, rank, tag> wrapper;
            mold(wrapper, source);
            return wrapper;
        }

        //Allocate an array matching the size of another array, returning a managed array
        template <typename T, int rank, arrayTags tag, typename T2, arrayTags tag2>
        void moldManaged(portableArray<T, rank, tag> &wrapper, const portableArray<T2, rank, tag2> &source)
        {
            deallocate(wrapper); // Ensure any previous allocation is cleaned up
            SIGNED_INDEX_TYPE lower_bound[rank];
            SIGNED_INDEX_TYPE upper_bound[rank];
            for (int i = 0; i < rank; ++i)
            {
                lower_bound[i] = source.getLowerBound(i);
                upper_bound[i] = source.getUpperBound(i);
            }
            this->allocateManaged(wrapper, lower_bound, upper_bound);
        }

        //Allocate and return an array matching the size of another array, returning a managed array
        template <typename T=void, arrayTags tag = arrayTags::accelerated, int rank, typename T2, arrayTags tag2>
        auto moldManaged(const portableArray<T2, rank, tag2> &source)
        {
            using Tdest = std::conditional_t<std::is_void_v<T>, T2, T>;
            portableArray<Tdest, rank, tag> wrapper;
            moldManaged(wrapper, source);
            return wrapper;
        }

        //Copy an array to a new array, allocating the new array
        template <typename T, int rank, arrayTags tag, typename T2, int rank2, arrayTags tag2>
        void copy(portableArray<T, rank, tag> &destination, const portableArray<T2, rank2, tag2> &source)
        {
            deallocate(destination); // Ensure any previous allocation is cleaned up
            this->mold(destination, source);
            assign(destination, source);
        }

        //Copy an array to a new array, allocating the new array, returning the new array
        template <typename T=void, arrayTags tag = arrayTags::accelerated, typename T2, int rank, arrayTags tag2>
        auto copy(const portableArray<T2, rank, tag2> &source)
        {
            using Tdest = std::conditional_t<std::is_void_v<T>, T2, T>;
            portableArray<Tdest, rank, tag> destination;
            this->copy(destination, source);
            return destination;
        }

        //Copy an array to a new managed array, allocating the new array in managed memory
        template <typename T, int rank, arrayTags tag, typename T2, int rank2, arrayTags tag2>
        void copyManaged(portableArray<T, rank, tag> &destination, const portableArray<T2, rank2, tag2> &source)
        {
            deallocate(destination); // Ensure any previous allocation is cleaned up
            this->moldManaged(destination, source);
            assign(destination, source);
            destination.setManaged(true);
        }

        //Copy an array to a new managed array, allocating the new array in managed memory, returning the new array
        template <typename T=void, arrayTags tag = arrayTags::accelerated, typename T2, int rank, arrayTags tag2>
        auto copyManaged(const portableArray<T2, rank, tag2> &source)
        {
            using Tdest = std::conditional_t<std::is_void_v<T>, T2, T>;
            portableArray<Tdest, rank, tag> destination;
            this->copyManaged(destination, source);
            destination.setManaged(true);
            return destination;
        }

        /**
         * Create a portableArray instance with the specified ranges.
         */
        template <typename T, typename... T_ranges>
        portableArray<T, sizeof...(T_ranges), arrayTags::accelerated> create(T_ranges... ranges)
        {            
            portableArray<T, sizeof...(T_ranges)> wrapper;
            allocate(wrapper, ranges...);
            return wrapper;
        }

        template<typename T, typename... T_ranges>
        portableArray<T, sizeof...(T_ranges), arrayTags::accelerated> createManaged(T_ranges... ranges)
        {
            portableArray<T, sizeof...(T_ranges)> wrapper;
            allocateManaged(wrapper, ranges...);
            return wrapper;
        }

        /**
         * Create a portableArray instance with the specified ranges.
         */
        template <typename T, typename... T_ranges>
        portableArray<T, sizeof...(T_ranges), arrayTags::host> createHost(T_ranges... ranges)
        {
            portableArray<T, sizeof...(T_ranges),arrayTags::host> wrapper;
            allocate(wrapper, ranges...);
            return wrapper;
        }

        template <typename T, int rank, arrayTags tags>
        void deallocate(portableArray<T, rank, tags> &wrapper)
        {
            if (wrapper.manager != this)
            {
                if (wrapper.manager) wrapper.manager->deallocate(wrapper);
                return;
            }
            if (wrapper.data() != nullptr && wrapper.ownsData)
            {
                auto it = destructors.find(wrapper.data());
                if (it != destructors.end())
                {
                    destructors.erase(it);
                }
            }
			wrapper.bind(nullptr);
        }

        void deallocate(void *data)
        {
            auto it = destructors.find(data);
            if (it != destructors.end())
            {
                destructors.erase(it);
            }
        }

        template <typename T, int rank, typename... T_ranges>
        void reallocate(portableArray<T, rank> &wrapper, T_ranges... ranges)
        {
            deallocate(wrapper);
            allocate(wrapper, ranges...);
        }

    /**
     * Function to copy data from one portableArray to another.
     * This function uses the appropriate backend to copy the data.
     * If the source and destination are the same, it does nothing.
     * Otherwise a copy will always be performed even if source and destination are on the same device.
     * The source and destination arrays must have the same type, and the same total number of elements,
     * but can have different ranks in which case the data is copied in me
     * @param destination The destination portableArray where the data will be copied to. Must be allocated to be large enough to hold the source data.
     * @param source The source portableArray from which the data will be copied.
     */
    template<typename T_data, int rankD, int rankS, arrayTags tagD, arrayTags tagS>
    void copyData(portableArray<T_data, rankD, tagD> &destination, const portableArray<T_data, rankS, tagS> &source)
    {
        if constexpr (tagS == tagD){
            //Check if the source and destination are the same, if so nothing to do
            if ((char*)source.data() == (char*)destination.data()) return;
        }
        
        #ifdef USE_KOKKOS
            kokkos::copyData(destination, source);
        #elif defined(USE_CUDA)
            cuda::copyData(destination, source);
        #elif defined(USE_HIP)
            hip::copyData(destination, source);
        #else
            openmp::copyData(destination, source);
        #endif
     
    }

    template<typename T_data, int rank, arrayTags tag>
    auto makeHostAvailable(const portableArray<T_data, rank, tag> &array)
    {
        portableArray<T_data, rank, arrayTags::host> hostArray;        
        if (array.isManaged() || tag == arrayTags::host)
        {
            //Array is already available on the host, just return it
            hostArray.pointTo(array);
            return hostArray;
        }
        else
        {
            this->allocate(hostArray, array.getLowerBounds(), array.getUpperBounds());
            this->copyData(hostArray, array);
            return hostArray;
        }
    }

   template<typename T_data, int rank, arrayTags tag>
    auto makeDeviceAvailable(const portableArray<T_data, rank, tag> &array)
    {
        portableArray<T_data, rank, arrayTags::accelerated> deviceArray;
        if (array.isManaged() || tag == arrayTags::accelerated)
        {
            //Array is already available on the device, just return it
            deviceArray.pointTo(array);
            return deviceArray;
        }
        else
        {
            this->allocate(deviceArray, array.getLowerBounds(), array.getUpperBounds());
            this->copyData(deviceArray, array);
            return deviceArray;
        }
    }

    void clear()
    {
        destructors.clear();
    }
    
    }; // class portableArrayManager

};

#endif // MANAGER_H
