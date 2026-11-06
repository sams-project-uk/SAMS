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
#ifndef HIPBACKEND_H
#define HIPBACKEND_H

#include "defs.h"
#include "utils.h"
#include "range.h"
#include "demangle.h"
#include "callableTraits.h"
#include <iostream>
#include <stdio.h>
#include <array>

#define HIP_ERROR_CHECK(call) \
    do { \
        hipError_t ccerr = call; \
        if (ccerr != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(ccerr) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifdef USE_HIP
namespace portableWrapper
{
    namespace hip
    {

        namespace impl
        {

            #if defined(__HIP_PLATFORM_NVIDIA__) && defined(__CUDACC_VER_MAJOR__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 1)))
            constexpr size_t maxFormalArgSize = 32764; // Default value for CUDA >= 12.1
            #else
            constexpr size_t maxFormalArgSize=4096;
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

            template<bool sendAll, typename T_func, typename... T_ranges>
            void buildDecompLogic(dim3 &threads, dim3 &blocks, dim3& chunks, size_t shared_memory, const T_ranges... ranges);

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
                if constexpr (sizeof...(T_others) < 3)
                {
                    return getLast3RangesCore<0,sizeof...(T_others)+1>(current, others...);
                }
                else
                {
                    return getLast3Ranges(others...);
                }
            }

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
                        delta <<= 1;
                        mlocal >>= 1;
                        __syncthreads();
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

            template <int level, bool sendAll, int rank, typename T_func, typename T_tuple, typename T_cRange, typename... T_others>
            DEVICEPREFIX INLINE void forEachHIPCore(const T_func &func, T_tuple tuple, N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> chunkTuple, T_cRange cRange, T_others... others)
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
                        forEachHIPCore<level + 1, sendAll, rank>(func, tuple, chunkTuple, others...);
                    }
                }
                else
                {
                    UNSIGNED_INDEX_TYPE chunkOffset=0;
                    for (size_t ichunk = 0; ichunk < GET<rlevel>(chunkTuple); ++ichunk)
                    {
                        SIGNED_INDEX_TYPE thread_local_index = [lower_bound, ichunk, upper_bound, &chunkOffset]()
                        {
                            SIGNED_INDEX_TYPE offset;
                            if constexpr (level == rank-portableWrapper::max(3,rank)+0)
                            {
                                offset = blockIdx.z * blockDim.z + threadIdx.z + lower_bound + chunkOffset;
                                chunkOffset+= blockDim.z * gridDim.z;
                            }
                            else if constexpr (level == rank-portableWrapper::max(3,rank)+1)
                            {
                                offset = blockIdx.y * blockDim.y + threadIdx.y + lower_bound + chunkOffset;
                                chunkOffset+= blockDim.y * gridDim.y;
                            }
                            else if constexpr (level == rank-portableWrapper::max(3,rank)+2)
                            {
                                offset = blockIdx.x * blockDim.x + threadIdx.x + lower_bound + chunkOffset;
                                chunkOffset+= blockDim.x * gridDim.x;
                            }
                            return offset;
                        }();
                        constexpr int loopedIndices = portableWrapper::max(0, rank-3);
                        GET<level+loopedIndices+tupleOffset>(tuple) = thread_local_index;
                        if (thread_local_index <= upper_bound || sendAll){
                            if constexpr(sendAll) GET<0>(tuple).active &= (thread_local_index <= upper_bound); 

                            if constexpr (sizeof...(T_others) > 0) {
                                forEachHIPCore<level + 1, sendAll, rank>(func, tuple, chunkTuple, others...);
                            } else {
                                applyToData(func, tuple);
                            }
                        } else {
                            break;
                        }
                    }
                }
            }

            template <bool sendAll=false, typename T_func=void, typename... T_ranges>
            __global__ void forEachHIP(T_func func, dim3 chunks, T_ranges... ranges)
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
                    forEachHIPCore<0, sendAll, rank>(func, tuple, chunkTuple, ranges...);
                    if constexpr (hasStartFinish<T_func>::value) func.finish();
                }
                else
                {
                    N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple;
                    forEachHIPCore<0, sendAll, rank>(func, tuple, chunkTuple, ranges...);
                }
            }

            template<bool sendAll=false, typename T_func=void, typename... T_ranges>
            __global__ void forEachHIPPtr(T_func* func, dim3 chunks, T_ranges... ranges)
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
                    forEachHIPCore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
                    if constexpr (hasStartFinish<T_func>::value) func->finish();
                }
                else
                {
                    N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple;
                    forEachHIPCore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
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
                int warpWidth = 0;
            };

            /**
             * Get the device properties. Queried on the first call only
             */
            inline const deviceLimits& getDeviceLimits()
            {
                static const deviceLimits limits = [](){
                    deviceLimits queried;
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.maxThreadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.maxBlockDimX, hipDeviceAttributeMaxBlockDimX, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.maxBlockDimY, hipDeviceAttributeMaxBlockDimY, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.maxBlockDimZ, hipDeviceAttributeMaxBlockDimZ, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.maxGridDimX, hipDeviceAttributeMaxGridDimX, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.maxGridDimY, hipDeviceAttributeMaxGridDimY, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.maxGridDimZ, hipDeviceAttributeMaxGridDimZ, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.smCount, hipDeviceAttributeMultiprocessorCount, 0));
                    HIP_ERROR_CHECK(hipDeviceGetAttribute(&queried.warpWidth, hipDeviceAttributeWarpSize, 0));
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
                    hipFuncAttributes attr;
                    if constexpr (impl::getKernelArgSize<T_func, dim3, T_ranges...>() > maxFormalArgSize)
                        HIP_ERROR_CHECK(hipFuncGetAttributes(&attr, forEachHIPPtr<sendAll, T_func, T_ranges...>));
                    else
                        HIP_ERROR_CHECK(hipFuncGetAttributes(&attr, forEachHIP<sendAll, T_func, T_ranges...>));
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
                        kernelPtr = (void*)forEachHIPPtr<sendAll, T_func, T_ranges...>;
                    else
                        kernelPtr = (void*)forEachHIP<sendAll, T_func, T_ranges...>;
                    hipOccupancyMaxActiveBlocksPerMultiprocessor(&activePerSM, kernelPtr, threadsPerBlock, sharedMemory);
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
            const int warpWidth = limits.warpWidth;
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
            auto ceilDiv = [](unsigned long long a, unsigned long long b){ return (a + b - 1ULL)/b; };
            unsigned long long needX = ceilDiv(parallelRanges.x, threads.x ? threads.x : 1);
            unsigned long long needY = (sizeof...(ranges)>1)? ceilDiv(parallelRanges.y, threads.y ? threads.y : 1) : 1ULL;
            unsigned long long needZ = (sizeof...(ranges)>2)? ceilDiv(parallelRanges.z, threads.z ? threads.z : 1) : 1ULL;

            int limX = limits.maxGridDimX;
            int limY = (sizeof...(ranges)>1) ? limits.maxGridDimY : 1;
            int limZ = (sizeof...(ranges)>2) ? limits.maxGridDimZ : 1;

            unsigned long long useX = std::min<unsigned long long>(needX, limX);
            unsigned long long useY = std::min<unsigned long long>(needY, limY);
            unsigned long long useZ = std::min<unsigned long long>(needZ, limZ);

            int smCount = limits.smCount;
            int activePerSM = getKernelActiveBlocksPerSM<sendAll, T_func, T_ranges...>(threads.x * threads.y * threads.z, shared_memory*threads.x * threads.y * threads.z);

            unsigned long long inflightLimit = (unsigned long long)activePerSM * (unsigned long long)smCount;

            //Only reductions trim the grid, because phase 2 is a single block
            auto trim = [&](unsigned long long &d, [[maybe_unused]] unsigned long long &needOther){
                while (useX * useY * useZ > inflightLimit && d > 1) {
                    d = (d + 1) / 2;
                }
            };
            if constexpr (sendAll) {
                trim(useX, needX); trim(useY, needY); trim(useZ, needZ);
            }

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

        template <bool sendAll = false, typename T_func=void, typename... T_ranges>
        UNREPEATED void applyKernelCore(T_func func, dim3 threads, dim3 blocks, dim3 chunks, size_t shared_memory, T_ranges... ranges){

            if constexpr (impl::getKernelArgSize<T_func, dim3, T_ranges...>() > impl::maxFormalArgSize)
            {
                //Large functors are copied to a device buffer, one per kernel, reused on the same stream
                static T_func* data=nullptr;
                if (!data) HIP_ERROR_CHECK(hipMalloc(&data, sizeof(T_func)));
                HIP_ERROR_CHECK(hipMemcpyAsync(data, &func, sizeof(T_func), hipMemcpyHostToDevice));
                hipLaunchKernelGGL(impl::forEachHIPPtr<sendAll,T_func>, blocks, threads, shared_memory, 0, data, chunks, ranges...);
                HIP_ERROR_CHECK(hipGetLastError());
            }
            else
            {
                hipLaunchKernelGGL(impl::forEachHIP<sendAll,T_func>, blocks, threads, shared_memory, 0, func, chunks, ranges...);
                HIP_ERROR_CHECK(hipGetLastError());
            }
        }

        template <typename T_func=void, typename... T_ranges>
        UNREPEATED void applyKernel(T_func func, T_ranges... ranges){
            dim3 threads, blocks, chunks;
            impl::buildDecompLogic<false,T_func>(threads, blocks, chunks, 0, ranges...);
            applyKernelCore<false>(func, threads, blocks, chunks, 0, ranges...);
        }

        template <typename T_ret=void, typename T_data_in=void, typename T_mapper=void, typename T_reducer=void, typename... T_ranges>
        UNREPEATED auto applyReduction(T_mapper mapper, T_reducer reducer, T_data_in initialValue, T_ranges... ranges)
        {
            using T_data = std::conditional_t<std::is_void_v<T_ret>,typename far::callableTraits<T_mapper>::type, T_ret>;

            dim3 threads, blocks, chunks;
            impl::buildDecompLogic<true,impl::reductionPhase1<T_mapper, T_reducer, T_data, sizeof...(T_ranges)>>(threads, blocks, chunks, sizeof(T_data), ranges...);

            size_t nBlocks = blocks.x * blocks.y * blocks.z;
            //One result per phase 1 block, plus the final result
            T_data *reductionSites;
            HIP_ERROR_CHECK(hipMalloc(&reductionSites, (nBlocks + 1) * sizeof(T_data)));

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
            //hipMemcpy waits for both phases to finish
            HIP_ERROR_CHECK(hipMemcpy(&result, reductionSites + nBlocks, sizeof(T_data), hipMemcpyDeviceToHost));
            HIP_ERROR_CHECK(hipFree(reductionSites));

            return result;
        }

       template<typename T_data, int rankS, int rankD, arrayTags tagS, arrayTags tagD>
        UNREPEATED void copyData(portableArray<T_data, rankD, tagD> &destination, const portableArray<T_data, rankS, tagS> &source) {
            HIP_ERROR_CHECK(hipMemcpy(destination.data(), source.data(), source.getElements() * sizeof(T_data), hipMemcpyDefault));
        }

        template<typename T>
        UNREPEATED T* allocate(size_t elements) {
            T* data;
            HIP_ERROR_CHECK(hipMalloc(&data, elements * sizeof(T)));
            return data;
        }

        template<typename T>
        UNREPEATED T* allocateShared(size_t elements) {
            T* data;
            HIP_ERROR_CHECK(hipMallocManaged(&data, elements * sizeof(T), hipMemAttachGlobal));
            return data;
        }

        template<typename T>
        UNREPEATED void deallocate(T* data) {
            HIP_ERROR_CHECK(hipFree(data));
        }

        UNREPEATED void fence() {
            hipDeviceSynchronize();
        }

        UNREPEATED void printInfo(){
            SAMS::cout << "HIP" << std::endl;
            int device;
            HIP_ERROR_CHECK(hipGetDevice(&device));
            hipDeviceProp_t prop;
            HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, device));
            SAMS::cout << "HIP device: " << prop.name << std::endl;
            SAMS::cout << "HIP device compute capability: " << prop.major << "." << prop.minor << std::endl;
            SAMS::cout << "HIP device total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        }

        UNREPEATED void initialize(int &argc, char *argv[])
        {
            int deviceCount;
            HIP_ERROR_CHECK(hipGetDeviceCount(&deviceCount));
            if (deviceCount == 0)
            {
                std::cerr << "No HIP devices found. Exiting." << std::endl;
                exit(EXIT_FAILURE);
            }
            const char *hipDeviceEnv = std::getenv("PW_HIP_DEVICE");
            if (hipDeviceEnv != nullptr)
            {
                int device = std::atoi(hipDeviceEnv);
                if (device < 0 || device >= deviceCount)
                {
                    std::cerr << "Invalid HIP device specified in PW_HIP_DEVICE: " << device << "\n";
                    exit(EXIT_FAILURE);
                }
                HIP_ERROR_CHECK(hipSetDevice(device));
            } else {
                HIP_ERROR_CHECK(hipSetDevice(0));
            }
        }

        //HIP atomic operations
        namespace atomic
        {
            /**
             * Atomic addition function for HIP
             * @param data Data to perform the atomic addition on
             * @param val Value to add atomically
             */
            template<typename T1, typename T2>
            DEVICEPREFIX void Add(T1& data, T2 val) {
                ::atomicAdd(&data, val);
            }

            /**
             * Atomic AND function for HIP
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
             * Atomic DECREMENT function for HIP
             * @param data Data to perform the atomic DECREMENT on
             */
            template<typename T1>
            DEVICEPREFIX void Dec(T1& data) {
                ::atomicSub(&data, T1(1));
            }

            /**
             * Atomic INCREMENT function for HIP
             * @param data Data to perform the atomic INCREMENT on
             */
            template<typename T1>
            DEVICEPREFIX void Inc(T1& data) {
                ::atomicAdd(&data, T1(1));
            }

            /**
             * Atomic MAX function for HIP
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
             * Atomic MIN function for HIP
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
             * Atomic OR function for HIP
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
             * Atomic Subtraction function for HIP
             * @param data Data to perform the atomic Subtraction on
             * @param val Value to subtract atomically
             */
            template<typename T1, typename T2>
            DEVICEPREFIX void Sub(T1& data, T2 val) {
                ::atomicSub(&data, val);
            }

        } // namespace atomic

    } // namespace hip
} // namespace portableWrapper
#endif // USE_HIP

#endif
