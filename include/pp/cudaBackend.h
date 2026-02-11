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
#ifndef CUDABACKEND_H
#define CUDABACKEND_H

#include "defs.h"
#include "utils.h"
#include "range.h"
#include "demangle.h"
#include "callableTraits.h"
#include <iostream>
#include <stdio.h>
#include <array>

#define CUDA_ERROR_CHECK(call) \
    do { \
        cudaError_t ccerr = call; \
        if (ccerr != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(ccerr) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifdef USE_CUDA
namespace portableWrapper
{
    namespace cuda
    {

        namespace impl
        {

            // For CUDA <12.1 4096 bytes is the maximum size of a kernel argument
            // For CUDA >=12.1 this is increased to 32768 bytes
            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 1)
            constexpr size_t maxFormalArgSize = 32764/2; // Default value for CUDA devices with compute capability >= 12.1
            #elif defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 2)
            constexpr size_t maxFormalArgSize=4096; // Default value for CUDA devices with compute capability >= 2.0
            #else
            //This is ridiculously low, should we really support devices with compute capability < 2.0?
            constexpr size_t maxFormalArgSize=256; // Default value for CUDA devices with compute capability < 2.0
            #endif

            /**
             * Forward declaration of the buildDecompLogic function
             */
            template<bool sendAll, typename T_func, typename... T_ranges>
            void buildDecompLogic(dim3 &threads, dim3 &blocks, dim3& chunks, size_t shared_memory, const T_ranges... ranges);


            /**
             * Populate the right element of a dim3 structure with a range
             */
            template <int level, int rank, typename T_current>
            DEVICEPREFIX INLINE dim3 populateDim(const T_current currentRange, dim3 existing)
            {
                dim3 result = existing;
                if constexpr (level == rank - portableWrapper::max(rank,3)+2)
                {
                    result.x = currentRange.upper_bound - currentRange.lower_bound + 1;
                }
                else if constexpr (level == rank - portableWrapper::max(rank,3)+1)
                {
                    result.y = currentRange.upper_bound - currentRange.lower_bound + 1;
                }
                else if constexpr (level == rank - portableWrapper::max(rank,3))
                {
                    result.z = currentRange.upper_bound - currentRange.lower_bound + 1;
                }
                return result;
            }

            /**
             * Get the last 3 ranges from a list of ranges
             * This is used to create a dim3 structure for CUDA kernels
             * that can handle up to 3 dimensions.
             */
            template <int level = 0, int rank=0, typename T_current, typename... T_others>
            DEVICEPREFIX INLINE dim3 getLast3RangesCore(const T_current current, const T_others... others)
            {
                if constexpr (sizeof...(T_others) == 0)
                {
                    return populateDim<level,rank>(current, dim3(1, 1, 1));
                }
                else
                {
                    return populateDim<level,rank>(current, getLast3RangesCore<level + 1,rank>(others...));
                }
            }

            template<typename T_current, typename... T_others>
            DEVICEPREFIX INLINE dim3 getLast3Ranges(const T_current current, const T_others... others)
            {
                //If 3 or fewer ranges left, call the implementation with level 0
                if constexpr (sizeof...(T_others) < 3)
                {
                    return getLast3RangesCore<0,sizeof...(T_others)+1>(current, others...);
                }
                else
                {
                    //Otherwise call self to strip the first range
                    return getLast3Ranges(others...);
                }
            }

            /**
             * Class representing information about the current location
             * in a launched kernel
             */
            struct kernelInflightInfo{
                bool active; //Is a current location active for this thread
            };

            /*
            * CUDA reduction phase 1 functor
            * This functor performs the first phase of a reduction operation in CUDA. It maps input values using a mapper function,
            * performs a block-level reduction using a reducer function, and stores the intermediate results in global memory.
            */
            template<typename T_mapper, typename T_reducer, typename T_data, int rank>
            struct reductionPhase1{
                private:
                T_mapper mapper;
                T_reducer reducer;
                T_data *data;
                T_data initialValue;
                public:
                template<typename... T_indices>
                __device__ INLINE void operator()(kernelInflightInfo k, T_indices... ranges) const
                {
                    extern __shared__ T_data sharedReduction[];
                    UNSIGNED_INDEX_TYPE threadIndex = threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.y * blockDim.z;
                    if (k.active){
                        sharedReduction[threadIndex] = mapper(ranges...);
                    } else {
                        sharedReduction[threadIndex] = initialValue;
                    }
                    __syncthreads();
                    
                    size_t mthread= blockDim.x * blockDim.y * blockDim.z;
                    size_t delta = 1;
                    size_t mlocal = mthread;
                    UNSIGNED_INDEX_TYPE threadLocal = threadIndex;
                    while (mlocal>1)
                    {
                        if (mlocal &0x1){
                        //Combine 0 and last
                            if (threadIndex == 0)
                            {
                                reducer(sharedReduction[0], sharedReduction[(mlocal - 1)*delta]);
                                mlocal--;
                            }
                        }
                        if (!(threadLocal&0x1) && threadIndex + delta < mthread )
                        {
                            reducer(sharedReduction[threadIndex], sharedReduction[threadIndex + delta]);
                            threadLocal >>=1;
                        }
                        delta <<= 1; // Double the delta to reduce pairs of pairs
                        mlocal >>= 1; // Halve the local thread count
                        __syncthreads(); // Ensure all threads have completed the reduction before proceeding
                    }
                    
                    if (threadIndex == 0)
                    {
                        reducer(data[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x], sharedReduction[0]);
                    }
                    
                }
            
                reductionPhase1(T_mapper m, T_reducer r, T_data *d, T_data init=T_data())
                    : mapper(m), reducer(r), data(d), initialValue(init){}
            };

            /**
             * CUDA reduction phase 2 functor
             * This functor performs the second phase of a reduction operation in CUDA. It reduces the intermediate results
             * stored in global memory from the first phase and produces the final reduced value.
             */
            template<typename T_reducer, typename T_data>
            struct reductionPhase2{
                T_reducer reducer;
                T_data *data;
                size_t nBlocks;
                template<typename... T_ranges>
                __device__ INLINE void operator()(UNSIGNED_INDEX_TYPE i) const
                {
                    size_t threadID = threadIdx.x;
                    size_t blockID = blockIdx.x;
                    size_t globalID = blockID * blockDim.x + threadID;
                    size_t delta = 1;
                    //Can only reduce within a block, so we need to ensure we don't go out of bounds
                    size_t lBlocks = portableWrapper::min(nBlocks, blockDim.x);
                    //If there are an odd number of blocks, the last block will not have a pair to reduce with, so reduce it with 0

                    size_t rID = globalID;
                    while (lBlocks>1)
                    {
                        if ((lBlocks & 0x1) && globalID==0){
                            int i= data[0];
                            reducer(data[0], data[(lBlocks-1)*delta]);
                            lBlocks--;
                        }
                        if (!(rID & 0x1) && (globalID + delta < nBlocks)){
                            int i= data[globalID];
                            reducer(data[globalID], data[globalID + delta]);
                            rID >>= 1;
                        }
                        delta<<=1; // Double the delta to reduce pairs of pairs
                        lBlocks >>= 1; // Halve the local thread count
                        __syncthreads(); // Ensure all threads have completed the reduction before proceeding
                    }
                }
                reductionPhase2(T_reducer r, T_data *d, size_t n)
                    : reducer(r), data(d), nBlocks(n) {}
            };

            /**
             * CUDA parallel forEach device function
             * This function uses CUDA to parallelize the execution of the provided function
             * over the specified ranges. It constructs a dim3 structure based on the number of ranges
             * and launches a CUDA kernel to execute the function in parallel.
             */
            template <int level, bool sendAll, int rank, typename T_func, typename T_tuple, typename T_cRange, typename... T_others>
            DEVICEPREFIX INLINE void forEachCUDACore(const T_func &func, T_tuple tuple, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> chunkTuple, T_cRange cRange, T_others... others)
            {
                constexpr int rlevel = (rank-1) - level - portableWrapper::max(0, rank-3);
                constexpr UNSIGNED_INDEX_TYPE tupleOffset = sendAll ? 1 : 0;
                if constexpr(sendAll && level == 0) GET<0>(tuple).active =true;
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
                if constexpr (level < rank - 3)
                {
                    for (SIGNED_INDEX_TYPE i = lower_bound; i <= upper_bound; ++i)
                    {
                        GET<level+tupleOffset>(tuple) = i;
                        forEachCUDACore<level + 1, sendAll, rank>(func, tuple, chunkTuple, others...);
                    }
                }
                else
                {
                    UNSIGNED_INDEX_TYPE chunkOffset=0;
                    for (size_t ichunk = 0; ichunk < GET<rlevel>(chunkTuple); ++ichunk)
                    {
                        // Calculate the thread-local index based on the current chunk and level
                        SIGNED_INDEX_TYPE thread_local_index = [lower_bound, ichunk, upper_bound, &chunkOffset]()
                        {
                            SIGNED_INDEX_TYPE offset;
                            if constexpr (level == rank-portableWrapper::max(3,rank)+0)
                            {
                                offset = blockIdx.z * blockDim.z + threadIdx.z + lower_bound + chunkOffset;
                                chunkOffset+= blockDim.z * gridDim.z; // Increment chunkOffset for the next chunk
                            }
                            else if constexpr (level == rank-portableWrapper::max(3,rank)+1)
                            {
                                offset = blockIdx.y * blockDim.y + threadIdx.y + lower_bound + chunkOffset;
                                chunkOffset+= blockDim.y * gridDim.y; // Increment chunkOffset for the next chunk
                            }
                            else if constexpr (level == rank-portableWrapper::max(3,rank)+2)
                            {
                                offset = blockIdx.x * blockDim.x + threadIdx.x + lower_bound + chunkOffset;
                                chunkOffset+= blockDim.x * gridDim.x; // Increment chunkOffset for the next chunk
                            }
                            return offset;
                        }();
                        constexpr int loopedIndices = portableWrapper::max(0, rank-3);
                        GET<level+loopedIndices+tupleOffset>(tuple) = thread_local_index;
                        if (thread_local_index <= upper_bound || sendAll){
                            if constexpr(sendAll) GET<0>(tuple).active &= (thread_local_index <= upper_bound); 

                            if constexpr (sizeof...(T_others) > 0) {
                                forEachCUDACore<level + 1, sendAll, rank>(func, tuple, chunkTuple, others...);
                            } else {
                                applyToData(func, tuple);
                            }
                        } else {
                            break; // No more valid indices in this chunk
                        }
                    }
                }
            }

            /**
             * CUDA kernel function to launch the forEachCUDACore function
             * This function is the entry point for the CUDA kernel and initializes the tuple.
             */
            template <bool sendAll=false, typename T_func=void, typename... T_ranges>
            __global__ void forEachCUDA(T_func func, dim3 chunks, T_ranges... ranges)
            {
                static constexpr int rank = sizeof...(ranges);
                N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> chunkTuple;
                GET<0>(chunkTuple) = chunks.x;
                if constexpr (rank>1) GET<1>(chunkTuple) = chunks.y;
                if constexpr (rank>2) GET<2>(chunkTuple) = chunks.z;
                if constexpr (sendAll)
                {
                    using tupleType = portableTuple::tuple_cat_type_t<portableTuple::tuple<kernelInflightInfo>, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank>>;
                    tupleType tuple;
                    forEachCUDACore<0, sendAll, rank>(func, tuple, chunkTuple, ranges...);
                }
                else
                {
                    N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple;
                    forEachCUDACore<0, sendAll, rank>(func, tuple, chunkTuple, ranges...);
                }
            }

            template<bool sendAll=false, typename T_func=void, typename... T_ranges>
            __global__ void forEachCUDAPtr(T_func* func, dim3 chunks, T_ranges... ranges)
            {
                static constexpr int rank = sizeof...(ranges);
                N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> chunkTuple;
                GET<0>(chunkTuple) = chunks.x;
                if constexpr (rank>1) GET<1>(chunkTuple) = chunks.y;
                if constexpr (rank>2) GET<2>(chunkTuple) = chunks.z;
                if constexpr (sendAll)
                {
                    using tupleType = portableTuple::tuple_cat_type_t<portableTuple::tuple<kernelInflightInfo>, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank>>;
                    tupleType tuple;
                    forEachCUDACore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
                }
                else
                {
                    N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple;
                    forEachCUDACore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
                }
            }


            template<bool sendAll, typename T_func, typename... T_ranges>
            void buildDecompLogic(dim3 &threads, dim3 &blocks, dim3& chunks, size_t shared_memory, const T_ranges... ranges) {

            dim3 parallelRanges = impl::getLast3Ranges(ranges...);

            //Get max threads per block for the device
            int maxThreadsPerBlock = 0;
            CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));

            cudaFuncAttributes attr;
            if constexpr (sizeof(T_func) > maxFormalArgSize)
                CUDA_ERROR_CHECK(cudaFuncGetAttributes(&attr, forEachCUDAPtr<sendAll, T_func, T_ranges...>));
            else    
                CUDA_ERROR_CHECK(cudaFuncGetAttributes(&attr, forEachCUDA<sendAll, T_func, T_ranges...>));
            
            maxThreadsPerBlock = std::min(maxThreadsPerBlock, attr.maxThreadsPerBlock);

            //Get max threads in each dimension
            int maxThreadsPerBlockX = 0, maxThreadsPerBlockY = 0, maxThreadsPerBlockZ = 0;
            CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlockX, cudaDevAttrMaxBlockDimX, 0));
            CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlockY, cudaDevAttrMaxBlockDimY, 0));
            CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlockZ, cudaDevAttrMaxBlockDimZ, 0));

            int maxThreadsPerBlockPerDim = std::floor(std::pow(maxThreadsPerBlock, 1.0 / std::min(size_t(3), sizeof...(ranges))));

            //Set Max Threads per block to 2,4,8,16,32,64,128,256,512
            std::array<int, 9> maxThreadsPerBlockOptions = {2, 4, 8, 16, 32, 64, 128, 256, 512};
            int target;
            for (int option : maxThreadsPerBlockOptions)
            {
                if (option <= maxThreadsPerBlockPerDim)
                {
                    target = option;
                }
            }
            maxThreadsPerBlockPerDim = target;
            if constexpr (sizeof...(ranges) == 1)
            {
                int threadsPerBlockX = std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockX);
                // Rank 1, use 1D grid
                threads = dim3(threadsPerBlockX, 1, 1);
                blocks = dim3((parallelRanges.x + threadsPerBlockX - 1) / threadsPerBlockX, 1, 1);
            }
            else if constexpr (sizeof...(ranges) == 2)
            {
                int threadsPerBlockX = std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockX);
                int threadsPerBlockY = std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockY);
                
                // Rank 2, use 2D grid
                threads = dim3(threadsPerBlockX, threadsPerBlockY, 1);
                blocks = dim3((parallelRanges.x + threadsPerBlockX - 1) / threadsPerBlockX,
                            (parallelRanges.y + threadsPerBlockY - 1) / threadsPerBlockY, 1);
            }
            else if constexpr (sizeof...(ranges) >= 3)
            {
                int threadsPerBlockX = std::max(std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockX),1);
                int threadsPerBlockY = std::max(std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockY),1);
                int threadsPerBlockZ = std::max(std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockZ),1);

                // Rank 3 or more, use 3D grid
                threads = dim3(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
                blocks = dim3((parallelRanges.x + threadsPerBlockX - 1) / threadsPerBlockX,
                            (parallelRanges.y + threadsPerBlockY - 1) / threadsPerBlockY,
                            (parallelRanges.z + threadsPerBlockZ - 1) / threadsPerBlockZ);
            }
            // Required blocks per dim (ceil divisions)
            auto ceilDiv = [](unsigned long long a, unsigned long long b){ return (a + b - 1ULL)/b; };
            unsigned long long needX = ceilDiv(parallelRanges.x, threads.x ? threads.x : 1);
            unsigned long long needY = (sizeof...(ranges)>1)? ceilDiv(parallelRanges.y, threads.y ? threads.y : 1) : 1ULL;
            unsigned long long needZ = (sizeof...(ranges)>2)? ceilDiv(parallelRanges.z, threads.z ? threads.z : 1) : 1ULL;

            // Hardware grid limits
            int limX=1, limY=1, limZ=1;
            cudaDeviceGetAttribute(&limX, cudaDevAttrMaxGridDimX, 0);
            if constexpr (sizeof...(ranges)>1) cudaDeviceGetAttribute(&limY, cudaDevAttrMaxGridDimY, 0);
            if constexpr (sizeof...(ranges)>2) cudaDeviceGetAttribute(&limZ, cudaDevAttrMaxGridDimZ, 0);

            // Start with hardware-clamped blocks per dim
            unsigned long long useX = std::min<unsigned long long>(needX, limX);
            unsigned long long useY = std::min<unsigned long long>(needY, limY);
            unsigned long long useZ = std::min<unsigned long long>(needZ, limZ);

            // Occupancy: max active blocks per SM for this kernel & block config
            int smCount=0; cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

            // Pick correct kernel symbol (pointer vs by-value) for occupancy query
            void *kernelPtr;
            if constexpr (sizeof(T_func) > maxFormalArgSize)
                kernelPtr = (void*)forEachCUDAPtr<sendAll, T_func, T_ranges...>;
            else
                kernelPtr = (void*)forEachCUDA<sendAll, T_func, T_ranges...>;

            int activePerSM=0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &activePerSM,
                kernelPtr,
                threads.x * threads.y * threads.z,
                shared_memory*threads.x * threads.y * threads.z);
        

            unsigned long long inflightLimit = (unsigned long long)activePerSM * (unsigned long long)smCount;

            // If current (useX*useY*useZ) exceeds inflight resource limit, shrink dims
            auto trim = [&](unsigned long long &d, unsigned long long &needOther){
                while (useX * useY * useZ > inflightLimit && d > 1) {
                    d = (d + 1) / 2; // halve (ceil)
                }
            };
            trim(useX, needX); trim(useY, needY); trim(useZ, needZ);

            // Final chunk counts (additive progression in device code)
            unsigned long long chunksX = std::max(1ULL, ceilDiv(needX, useX));
            unsigned long long chunksY = std::max(1ULL, ceilDiv(needY, useY));
            unsigned long long chunksZ = std::max(1ULL, ceilDiv(needZ, useZ));

            blocks = dim3((unsigned)useX,
                        (sizeof...(ranges)>1)? (unsigned)useY : 1u,
                        (sizeof...(ranges)>2)? (unsigned)useZ : 1u);
            chunks = dim3((unsigned)chunksX,
                        (sizeof...(ranges)>1)? (unsigned)chunksY : 1u,
                        (sizeof...(ranges)>2)? (unsigned)chunksZ : 1u);
            }
        } // namespace impl

        /**
         * CUDA implementation of the applyKernel function
         */
        template <bool sendAll = false, typename T_func=void, typename... T_ranges>
        UNREPEATED void applyKernelCore(T_func func, dim3 threads, dim3 blocks, dim3 chunks, size_t shared_memory, T_ranges... ranges){

            if constexpr (sizeof(T_func) > impl::maxFormalArgSize)
            {
                // If the function is too large, we need to allocate it on the device
                T_func* data=nullptr;
                CUDA_ERROR_CHECK(cudaMalloc(&data, sizeof(T_func)));
                CUDA_ERROR_CHECK(cudaMemcpy(data, &func, sizeof(T_func), cudaMemcpyHostToDevice));
                impl::forEachCUDAPtr<sendAll,T_func><<<blocks, threads, shared_memory>>>(data, chunks, ranges...);
                CUDA_ERROR_CHECK(cudaGetLastError());
                CUDA_ERROR_CHECK(cudaFree(data)); //Free the data pointer after the kernel launch
            }
            else
            {
                impl::forEachCUDA<sendAll,T_func><<<blocks, threads, shared_memory>>>(func, chunks, ranges...);
                CUDA_ERROR_CHECK(cudaGetLastError());
            }
        }

        template <typename T_func=void, typename... T_ranges>
        UNREPEATED void applyKernel(T_func func, T_ranges... ranges){
            dim3 threads, blocks, chunks;
            impl::buildDecompLogic<false,T_func>(threads, blocks, chunks, 0, ranges...);
            applyKernelCore<false>(func, threads, blocks, chunks, 0, ranges...);
        }

 
        /**
         * CUDA implementation of the applyReduction function
         */
        template <typename T_ret=void, typename T_data_in=void, typename T_mapper=void, typename T_reducer=void, typename... T_ranges>
        UNREPEATED auto applyReduction(T_mapper mapper, T_reducer reducer, T_data_in initialValue, T_ranges... ranges)
        {
            using T_data = std::conditional_t<std::is_void_v<T_ret>,typename far::callableTraits<T_mapper>::type, T_ret>;

            dim3 threads, blocks, chunks;
            impl::buildDecompLogic<true,impl::reductionPhase1<T_mapper, T_reducer, T_data, sizeof...(T_ranges)>>(threads, blocks, chunks, sizeof(T_data), ranges...);

            size_t nBlocks = blocks.x * blocks.y * blocks.z;
            T_data *reductionSites;
            CUDA_ERROR_CHECK(cudaMallocManaged(&reductionSites, nBlocks * sizeof(T_data)));

            for (int i=0; i < nBlocks; ++i)
            {
                reductionSites[i] = initialValue; // Initialize all reduction sites to the initial value
            }

            impl::reductionPhase1<T_mapper, T_reducer, T_data, sizeof...(T_ranges)> rp1(mapper, reducer, reductionSites, initialValue);

            //Each thread will have its own shared memory for the reduction
            size_t shared_memory = threads.x * threads.y * threads.z * sizeof(T_data);

            applyKernelCore<true>(rp1, threads, blocks, chunks, shared_memory, ranges...);
            cudaDeviceSynchronize(); // Ensure all threads have completed before proceeding to phase 2

            //while (nBlocks > 1)
            {
                impl::reductionPhase2<T_reducer, T_data> rp2(reducer, reductionSites, nBlocks);
                impl::buildDecompLogic<false, decltype(rp2)>(threads, blocks, chunks, 0, Range(0,nBlocks-1));
                applyKernelCore(impl::reductionPhase2<T_reducer, T_data>(reducer, reductionSites, nBlocks), threads, blocks, chunks, 0 ,Range(0,nBlocks-1));
                cudaDeviceSynchronize(); // Ensure all threads have completed before next cycle
                nBlocks/=2;
            }

            T_data result;
            cudaMemcpy(&result, reductionSites, sizeof(T_data), cudaMemcpyDeviceToHost); // Copy the final result back to the host
            cudaFree(reductionSites); // Free the reduction sites after the reduction is done

            return result;
        }

        /**
         * Function to copy data from one portableArray to another
         */
       template<typename T_data, int rankS, int rankD, arrayTags tagS, arrayTags tagD>
        UNREPEATED void copyData(portableArray<T_data, rankD, tagD> &destination, const portableArray<T_data, rankS, tagS> &source) {
            //Use default memcpy to automatically choose the right memcpy direction
            //We do have the information from the tags, but no need to complicate things
            CUDA_ERROR_CHECK(cudaMemcpy(destination.data(), source.data(), source.getElements() * sizeof(T_data), cudaMemcpyDefault));
        }

        /**
         * Function to allocate device memory of a fixed number of elements
         * Memory allocated by this function can be unavailable on the host
         */
        template<typename T>
        UNREPEATED T* allocate(size_t elements) {
            T* data;
            CUDA_ERROR_CHECK(cudaMalloc(&data, elements * sizeof(T)));
            return data;
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
            // Allocate memory using Kokkos
            // This can be used for both host and device memory depending on the execution space
            T* data;
            CUDA_ERROR_CHECK(cudaMallocManaged(&data, elements * sizeof(T), cudaMemAttachGlobal));
            return data;
        }

        /**
         * Function to deallocate device memory
         */
        template<typename T>
        UNREPEATED void deallocate(T* data) {
            CUDA_ERROR_CHECK(cudaFree(data));
        }


        UNREPEATED void fence() {
            cudaDeviceSynchronize(); // Ensure all threads have completed
        }

        /**
         * Function to print CUDA device information
         */
        UNREPEATED void printInfo(){
            SAMS::cout << "CUDA" << std::endl;
            int device;
            CUDA_ERROR_CHECK(cudaGetDevice(&device));
            cudaDeviceProp prop;
            CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, device));
            SAMS::cout << "CUDA device: " << prop.name << std::endl;
            SAMS::cout << "CUDA device compute capability: " << prop.major << "." << prop.minor << std::endl;
            SAMS::cout << "CUDA device total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        }

        UNREPEATED void initialize(int &argc, char *argv[])
        {
            int deviceCount;
            CUDA_ERROR_CHECK(cudaGetDeviceCount(&deviceCount));
            if (deviceCount == 0)
            {
                std::cerr << "No CUDA devices found. Exiting." << std::endl;
                exit(EXIT_FAILURE);
            }
            //Read the PW_CUDA_DEVICE environment variable to set the device
            const char *cudaDeviceEnv = std::getenv("PW_CUDA_DEVICE");
            if (cudaDeviceEnv != nullptr)
            {
                int device = std::atoi(cudaDeviceEnv);
                if (device < 0 || device >= deviceCount)
                {
                    std::cerr << "Invalid CUDA device specified in PW_CUDA_DEVICE: " << device << "\n";
                    exit(EXIT_FAILURE);
                }
                CUDA_ERROR_CHECK(cudaSetDevice(device));
            } else {
                CUDA_ERROR_CHECK(cudaSetDevice(0)); // Set the first CUDA device as the default
            }
        }

        //CUDA atomic operations
        namespace atomic
        {
            /**
             * Atomic addition function for CUDA
             * @param data Data to perform the atomic addition on
             * @param val Value to add atomically
             */
            template<typename T1, typename T2>
            DEVICEPREFIX void Add(T1& data, T2 val) {
                ::atomicAdd(&data, val);
            }

            /**
             * Atomic AND function for CUDA
             * @param data Data to perform the atomic AND on
             * @param val Value to AND atomically
             */
            template<typename T>
            __device__ void And(T& data, T val) {
                if constexpr(std::is_integral_v<T>) {
                    ::atomicAnd(&data, val);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __float_as_int(__int_as_float(swapped) & val));
                    } while (swapped != orig);
                    data = __int_as_float(orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = int64_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __double_as_longlong(__longlong_as_double(swapped) & val));
                    } while (swapped != orig);
                    data = __longlong_as_double(orig);
                } else {
                    static_assert(alwaysFalse<T>::value, "Atomic And not supported for this type");
                }
            }

            /**
             * Atomic DECREMENT function for CUDA
             * @param data Data to perform the atomic DECREMENT on
             */
            template<typename T1>
            DEVICEPREFIX void Dec(T1& data) {
                ::atomicSub(&data, T1(1));
            }

            /**
             * Atomic INCREMENT function for CUDA
             * @param data Data to perform the atomic INCREMENT on
             */
            template<typename T1>
            DEVICEPREFIX void Inc(T1& data) {
                ::atomicAdd(&data, T1(1));
            }

            /**
             * Atomic MAX function for CUDA
             * @param data Data to perform the atomic MAX on
             * @param val Value to compare for MAX atomically
             */
            template<typename T>
            __device__ void Max(T& data, T val) {
                if constexpr(std::is_integral_v<T>) {
                    ::atomicMax(&data, val);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __float_as_int(fmaxf(val, __int_as_float(swapped))));
                    } while (swapped != orig);
                    data = __int_as_float(orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = int64_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __double_as_longlong(fmax(val, __longlong_as_double(swapped))));
                    } while (swapped != orig);
                    data = __longlong_as_double(orig);
                } else {
                    static_assert(alwaysFalse<T>::value, "Atomic Max not supported for this type");
                }
            }

            /**
             * Atomic MIN function for CUDA
             * @param data Data to perform the atomic MIN on
             * @param val Value to compare for MIN atomically
             */
            template<typename T>
            __device__ void Min(T& data, T val) {
                if constexpr(std::is_integral_v<T>) {
                    ::atomicMin(&data, val);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __float_as_int(fminf(val, __int_as_float(swapped))));
                    } while (swapped != orig);
                    data = __int_as_float(orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = int64_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __double_as_longlong(fmin(val, __longlong_as_double(swapped))));
                    } while (swapped != orig);
                    data = __longlong_as_double(orig);
                } else {
                    static_assert(alwaysFalse<T>::value, "Atomic Min not supported for this type");
                }
            }

            /**
             * Atomic OR function for CUDA
             * @param data Data to perform the atomic OR on
             * @param val Value to OR atomically
             */
            template<typename T>
            __device__ void Or(T& data, T val) {
                if constexpr(std::is_integral_v<T>) {
                    ::atomicOr(&data, val);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __float_as_int(__int_as_float(swapped) | val));
                    } while (swapped != orig);
                    data = __int_as_float(orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = int64_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __double_as_longlong(__longlong_as_double(swapped) | val));
                    } while (swapped != orig);
                    data = __longlong_as_double(orig);
                } else {
                    static_assert(alwaysFalse<T>::value, "Atomic Or not supported for this type");
                }
            }

            /**
             * Atomic Subtraction function for CUDA
             * @param data Data to perform the atomic Subtraction on
             * @param val Value to subtract atomically
             */
            template<typename T1, typename T2>
            DEVICEPREFIX void Sub(T1& data, T2 val) {
                ::atomicSub(&data, val);
            }

        } // namespace atomic

    } // namespace cuda
} // namespace portableWrapper
#endif // USE_CUDA

#endif
