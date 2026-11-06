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

            // For CUDA <12.1 4096 bytes is the maximum total size of a kernel's parameters
            // For CUDA >=12.1 this is increased to 32764 bytes
            #if defined(__CUDACC_VER_MAJOR__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 1)))
            constexpr size_t maxFormalArgSize = 32764; // Default value for CUDA >= 12.1
            #elif defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 2)
            constexpr size_t maxFormalArgSize=4096; // Default value for CUDA >= 2.0
            #else
            //This is ridiculously low, should we really support CUDA < 2.0?
            constexpr size_t maxFormalArgSize=256; // Default value for CUDA < 2.0
            #endif

            /**
             * Total size of a kernel's parameters, laid out with their natural alignment
             */
            template<typename... T_params>
            constexpr size_t getKernelArgSize()
            {
                size_t size = 0;
                ((size = (size + alignof(T_params) - 1) / alignof(T_params) * alignof(T_params) + sizeof(T_params)), ...);
                return size;
            }

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
            * Reduction phase 1. Each thread accumulates its values, then the block combines them. Also used for phase 2
            */
            template<typename T_mapper, typename T_reducer, typename T_data, int rank>
            struct reductionPhase1{
                private:
                T_mapper mapper;
                T_reducer reducer;
                T_data *data;
                T_data initialValue;

                __device__ INLINE UNSIGNED_INDEX_TYPE getThreadIndex() const
                {
                    return threadIdx.z + threadIdx.y * blockDim.z + threadIdx.x * blockDim.y * blockDim.z;
                }
                public:
                /**
                 * Called once per thread before any values
                 */
                __device__ INLINE void start() const
                {
                    extern __shared__ T_data sharedReduction[];
                    sharedReduction[getThreadIndex()] = initialValue;
                }

                /**
                 * Accumulate one value
                 */
                template<typename... T_indices>
                __device__ INLINE void operator()(kernelInflightInfo k, T_indices... ranges) const
                {
                    extern __shared__ T_data sharedReduction[];
                    if (k.active){
                        reducer(sharedReduction[getThreadIndex()], mapper(ranges...));
                    }
                }

                /**
                 * Called once per thread after all values
                 */
                __device__ INLINE void finish() const
                {
                    extern __shared__ T_data sharedReduction[];
                    UNSIGNED_INDEX_TYPE threadIndex = getThreadIndex();
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
                    
                    //Written once, so not initialised
                    if (threadIndex == 0)
                    {
                        data[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x] = sharedReduction[0];
                    }
                    
                }
            
                reductionPhase1(T_mapper m, T_reducer r, T_data *d, T_data init=T_data())
                    : mapper(m), reducer(r), data(d), initialValue(init){}
            };

            /**
             * Reduction phase 2 mapper, reading the phase 1 results
             */
            template<typename T_data>
            struct readPartial{
                const T_data *partials;
                __device__ INLINE T_data operator()(SIGNED_INDEX_TYPE i) const
                {
                    return partials[i];
                }
                readPartial(const T_data *p) : partials(p) {}
            };

            /**
             * Detect start and finish methods on a functor
             */
            template<typename T, typename = void>
            struct hasStartFinish : std::false_type {};
            template<typename T>
            struct hasStartFinish<T, std::void_t<decltype(std::declval<const T&>().start()), decltype(std::declval<const T&>().finish())>> : std::true_type {};

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
                    if constexpr (hasStartFinish<T_func>::value) func.start();
                    forEachCUDACore<0, sendAll, rank>(func, tuple, chunkTuple, ranges...);
                    if constexpr (hasStartFinish<T_func>::value) func.finish();
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
                    if constexpr (hasStartFinish<T_func>::value) func->start();
                    forEachCUDACore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
                    if constexpr (hasStartFinish<T_func>::value) func->finish();
                }
                else
                {
                    N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple;
                    forEachCUDACore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
                }
            }


            /**
             * Device properties, queried once
             */
            struct deviceLimits{
                int maxThreadsPerBlock = 0;
                int maxBlockDimX = 0, maxBlockDimY = 0, maxBlockDimZ = 0;
                int maxGridDimX = 0, maxGridDimY = 0, maxGridDimZ = 0;
                int smCount = 0;
            };

            /**
             * Get the device properties. Queried on the first call only
             */
            inline const deviceLimits& getDeviceLimits()
            {
                static const deviceLimits limits = [](){
                    deviceLimits queried;
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.maxBlockDimX, cudaDevAttrMaxBlockDimX, 0));
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.maxBlockDimY, cudaDevAttrMaxBlockDimY, 0));
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.maxBlockDimZ, cudaDevAttrMaxBlockDimZ, 0));
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.maxGridDimX, cudaDevAttrMaxGridDimX, 0));
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.maxGridDimY, cudaDevAttrMaxGridDimY, 0));
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.maxGridDimZ, cudaDevAttrMaxGridDimZ, 0));
                    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&queried.smCount, cudaDevAttrMultiProcessorCount, 0));
                    return queried;
                }();
                return limits;
            }

            /**
             * Queried once per kernel
             */
            template<bool sendAll, typename T_func, typename... T_ranges>
            int getKernelMaxThreadsPerBlock()
            {
                static const int maxThreads = [](){
                    cudaFuncAttributes attr;
                    if constexpr (impl::getKernelArgSize<T_func, dim3, T_ranges...>() > maxFormalArgSize)
                        CUDA_ERROR_CHECK(cudaFuncGetAttributes(&attr, forEachCUDAPtr<sendAll, T_func, T_ranges...>));
                    else
                        CUDA_ERROR_CHECK(cudaFuncGetAttributes(&attr, forEachCUDA<sendAll, T_func, T_ranges...>));
                    return attr.maxThreadsPerBlock;
                }();
                return maxThreads;
            }

            /**
             * Queried again only when the block size or shared memory changes
             */
            template<bool sendAll, typename T_func, typename... T_ranges>
            int getKernelActiveBlocksPerSM(int threadsPerBlock, size_t sharedMemory)
            {
                static int cachedThreadsPerBlock = -1;
                static size_t cachedSharedMemory = 0;
                static int activePerSM = 0;
                if (threadsPerBlock != cachedThreadsPerBlock || sharedMemory != cachedSharedMemory)
                {
                    // Pick correct kernel symbol (pointer vs by-value) for occupancy query
                    void *kernelPtr;
                    if constexpr (impl::getKernelArgSize<T_func, dim3, T_ranges...>() > maxFormalArgSize)
                        kernelPtr = (void*)forEachCUDAPtr<sendAll, T_func, T_ranges...>;
                    else
                        kernelPtr = (void*)forEachCUDA<sendAll, T_func, T_ranges...>;
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activePerSM, kernelPtr, threadsPerBlock, sharedMemory);
                    cachedThreadsPerBlock = threadsPerBlock;
                    cachedSharedMemory = sharedMemory;
                }
                return activePerSM;
            }

            template<bool sendAll, typename T_func, typename... T_ranges>
            void buildDecompLogic(dim3 &threads, dim3 &blocks, dim3& chunks, size_t shared_memory, const T_ranges... ranges) {

            dim3 parallelRanges = impl::getLast3Ranges(ranges...);

            //Get max threads per block for the device and for this kernel
            const deviceLimits &limits = getDeviceLimits();
            int maxThreadsPerBlock = std::min(limits.maxThreadsPerBlock, getKernelMaxThreadsPerBlock<sendAll, T_func, T_ranges...>());

            //Get max threads in each dimension
            int maxThreadsPerBlockX = limits.maxBlockDimX, maxThreadsPerBlockY = limits.maxBlockDimY, maxThreadsPerBlockZ = limits.maxBlockDimZ;

            //x is contiguous, so it gets up to a warp. No axis gets more threads than the power of 2 that covers it
            constexpr int targetThreadsPerBlock = 256;
            constexpr int warpWidth = 32;
            const int budget = std::min(targetThreadsPerBlock, maxThreadsPerBlock);
            auto coveringThreads = [](unsigned int extent, int limit){
                int count = 1;
                while (count < static_cast<int>(extent) && count * 2 <= limit) count *= 2;
                return count;
            };
            int threadsPerBlockX = coveringThreads(parallelRanges.x, std::min((sizeof...(ranges) == 1) ? budget : std::min(budget, warpWidth), maxThreadsPerBlockX));
            int threadsPerBlockY = (sizeof...(ranges) > 1) ? coveringThreads(parallelRanges.y, std::min(budget / threadsPerBlockX, maxThreadsPerBlockY)) : 1;
            int threadsPerBlockZ = (sizeof...(ranges) > 2) ? coveringThreads(parallelRanges.z, std::min(budget / (threadsPerBlockX * threadsPerBlockY), maxThreadsPerBlockZ)) : 1;
            //Give any spare threads to x
            while (threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ * 2 <= budget && threadsPerBlockX < static_cast<int>(parallelRanges.x) && threadsPerBlockX * 2 <= maxThreadsPerBlockX)
            {
                threadsPerBlockX *= 2;
            }
            threads = dim3(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
            // Required blocks per dim (ceil divisions)
            auto ceilDiv = [](unsigned long long a, unsigned long long b){ return (a + b - 1ULL)/b; };
            unsigned long long needX = ceilDiv(parallelRanges.x, threads.x ? threads.x : 1);
            unsigned long long needY = (sizeof...(ranges)>1)? ceilDiv(parallelRanges.y, threads.y ? threads.y : 1) : 1ULL;
            unsigned long long needZ = (sizeof...(ranges)>2)? ceilDiv(parallelRanges.z, threads.z ? threads.z : 1) : 1ULL;

            // Hardware grid limits
            int limX = limits.maxGridDimX;
            int limY = (sizeof...(ranges)>1) ? limits.maxGridDimY : 1;
            int limZ = (sizeof...(ranges)>2) ? limits.maxGridDimZ : 1;

            // Start with hardware-clamped blocks per dim
            unsigned long long useX = std::min<unsigned long long>(needX, limX);
            unsigned long long useY = std::min<unsigned long long>(needY, limY);
            unsigned long long useZ = std::min<unsigned long long>(needZ, limZ);

            // Occupancy: max active blocks per SM for this kernel & block config
            int smCount = limits.smCount;
            int activePerSM = getKernelActiveBlocksPerSM<sendAll, T_func, T_ranges...>(threads.x * threads.y * threads.z, shared_memory*threads.x * threads.y * threads.z);
        

            unsigned long long inflightLimit = (unsigned long long)activePerSM * (unsigned long long)smCount;

            // Only reductions trim the grid, because phase 2 is a single block
            // If current (useX*useY*useZ) exceeds inflight resource limit, shrink dims
            auto trim = [&](unsigned long long &d, [[maybe_unused]] unsigned long long &needOther){
                while (useX * useY * useZ > inflightLimit && d > 1) {
                    d = (d + 1) / 2; // halve (ceil)
                }
            };
            if constexpr (sendAll) {
                trim(useX, needX); trim(useY, needY); trim(useZ, needZ);
            }

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

            if constexpr (impl::getKernelArgSize<T_func, dim3, T_ranges...>() > impl::maxFormalArgSize)
            {
                //Large functors are copied to a device buffer, one per kernel, reused on the same stream
                static T_func* data=nullptr;
                if (!data) CUDA_ERROR_CHECK(cudaMalloc(&data, sizeof(T_func)));
                CUDA_ERROR_CHECK(cudaMemcpyAsync(data, &func, sizeof(T_func), cudaMemcpyHostToDevice));
                impl::forEachCUDAPtr<sendAll,T_func><<<blocks, threads, shared_memory>>>(data, chunks, ranges...);
                CUDA_ERROR_CHECK(cudaGetLastError());
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
            //One result per phase 1 block, plus the final result
            T_data *reductionSites;
            CUDA_ERROR_CHECK(cudaMalloc(&reductionSites, (nBlocks + 1) * sizeof(T_data)));

            impl::reductionPhase1<T_mapper, T_reducer, T_data, sizeof...(T_ranges)> rp1(mapper, reducer, reductionSites, initialValue);

            //Each thread will have its own shared memory for the reduction
            size_t shared_memory = threads.x * threads.y * threads.z * sizeof(T_data);

            applyKernelCore<true>(rp1, threads, blocks, chunks, shared_memory, ranges...);

            //Phase 2 is a single block on the same stream, so no synchronisation is needed
            impl::reductionPhase1<impl::readPartial<T_data>, T_reducer, T_data, 1> rp2(impl::readPartial<T_data>(reductionSites), reducer, reductionSites + nBlocks, initialValue);
            unsigned int phase2Threads = static_cast<unsigned int>(std::min<size_t>(nBlocks, 256));
            dim3 phase2Blocks(1, 1, 1);
            dim3 phase2ThreadBlock(phase2Threads, 1, 1);
            dim3 phase2Chunks(static_cast<unsigned int>((nBlocks + phase2Threads - 1) / phase2Threads), 1, 1);
            applyKernelCore<true>(rp2, phase2ThreadBlock, phase2Blocks, phase2Chunks, phase2Threads * sizeof(T_data), Range(0, nBlocks - 1));

            T_data result;
            //cudaMemcpy waits for both phases to finish
            CUDA_ERROR_CHECK(cudaMemcpy(&result, reductionSites + nBlocks, sizeof(T_data), cudaMemcpyDeviceToHost));
            CUDA_ERROR_CHECK(cudaFree(reductionSites));

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
            // Allocate managed memory, accessible from both host and device
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
            template<typename T, typename T2>
            __device__ void And(T& data, T2 val) {
                T value = static_cast<T>(val);
                if constexpr(std::is_integral_v<T>) {
                    ::atomicAnd(&data, value);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            swapped & __float_as_int(value));
                    } while (swapped != orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = unsigned long long int;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            swapped & __double_as_longlong(value));
                    } while (swapped != orig);
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
            template<typename T, typename T2>
            __device__ void Max(T& data, T2 val) {
                T value = static_cast<T>(val);
                if constexpr(std::is_integral_v<T>) {
                    ::atomicMax(&data, value);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __float_as_int(fmaxf(value, __int_as_float(swapped))));
                    } while (swapped != orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = unsigned long long int;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __double_as_longlong(fmax(value, __longlong_as_double(swapped))));
                    } while (swapped != orig);
                } else {
                    static_assert(alwaysFalse<T>::value, "Atomic Max not supported for this type");
                }
            }

            /**
             * Atomic MIN function for CUDA
             * @param data Data to perform the atomic MIN on
             * @param val Value to compare for MIN atomically
             */
            template<typename T, typename T2>
            __device__ void Min(T& data, T2 val) {
                T value = static_cast<T>(val);
                if constexpr(std::is_integral_v<T>) {
                    ::atomicMin(&data, value);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __float_as_int(fminf(value, __int_as_float(swapped))));
                    } while (swapped != orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = unsigned long long int;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            __double_as_longlong(fmin(value, __longlong_as_double(swapped))));
                    } while (swapped != orig);
                } else {
                    static_assert(alwaysFalse<T>::value, "Atomic Min not supported for this type");
                }
            }

            /**
             * Atomic OR function for CUDA
             * @param data Data to perform the atomic OR on
             * @param val Value to OR atomically
             */
            template<typename T, typename T2>
            __device__ void Or(T& data, T2 val) {
                T value = static_cast<T>(val);
                if constexpr(std::is_integral_v<T>) {
                    ::atomicOr(&data, value);
                } else if constexpr(std::is_same_v<T, float>) {
                    //Get an integer with the same bit length as the float
                    using intType = int32_t;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            swapped | __float_as_int(value));
                    } while (swapped != orig);
                } else if constexpr(std::is_same_v<T, double>) {
                    //Get an integer with the same bit length as the double
                    using intType = unsigned long long int;
                    intType* data_as_int = reinterpret_cast<intType*>(&data);
                    intType orig = *data_as_int, swapped;
                    do {
                        swapped = orig;
                        orig = ::atomicCAS(data_as_int, swapped,
                            swapped | __double_as_longlong(value));
                    } while (swapped != orig);
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
