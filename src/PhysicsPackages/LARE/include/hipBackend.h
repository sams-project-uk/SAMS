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

            #if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
            constexpr size_t maxFormalArgSize = 32768; // HIP doesn't have the same restriction, but keep for compatibility
            #else
            constexpr size_t maxFormalArgSize=4096;
            #endif

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
                bool active;
                bool firstCallThisBlock;
            };

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
                    
                    if (threadIndex == 0)
                    {
                        reducer(data[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x], sharedReduction[0]);
                    }
                }
            
                reductionPhase1(T_mapper m, T_reducer r, T_data *d, T_data init=T_data())
                    : mapper(m), reducer(r), data(d), initialValue(init){}
            };

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
                    size_t lBlocks = portableWrapper::min(nBlocks, blockDim.x);

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
                        delta<<=1;
                        lBlocks >>= 1;
                        __syncthreads();
                    }
                }
                reductionPhase2(T_reducer r, T_data *d, size_t n)
                    : reducer(r), data(d), nBlocks(n) {}
            };

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
                    forEachHIPCore<0, sendAll, rank>(func, tuple, chunkTuple, ranges...);
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
                    forEachHIPCore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
                }
                else
                {
                    N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> tuple;
                    forEachHIPCore<0, sendAll, rank>(*func, tuple, chunkTuple, ranges...);
                }
            }

            template<bool sendAll, typename T_func, typename... T_ranges>
            void buildDecompLogic(dim3 &threads, dim3 &blocks, dim3& chunks, size_t shared_memory, const T_ranges... ranges) {

            dim3 parallelRanges = impl::getLast3Ranges(ranges...);

            int maxThreadsPerBlock = 0;
            HIP_ERROR_CHECK(hipDeviceGetAttribute(&maxThreadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, 0));

            hipFuncAttributes attr;
            if constexpr (sizeof(T_func) > maxFormalArgSize)
                HIP_ERROR_CHECK(hipFuncGetAttributes(&attr, forEachHIPPtr<sendAll, T_func, T_ranges...>));
            else    
                HIP_ERROR_CHECK(hipFuncGetAttributes(&attr, forEachHIP<sendAll, T_func, T_ranges...>));
            
            maxThreadsPerBlock = std::min(maxThreadsPerBlock, attr.maxThreadsPerBlock);

            int maxThreadsPerBlockX = 0, maxThreadsPerBlockY = 0, maxThreadsPerBlockZ = 0;
            HIP_ERROR_CHECK(hipDeviceGetAttribute(&maxThreadsPerBlockX, hipDeviceAttributeMaxBlockDimX, 0));
            HIP_ERROR_CHECK(hipDeviceGetAttribute(&maxThreadsPerBlockY, hipDeviceAttributeMaxBlockDimY, 0));
            HIP_ERROR_CHECK(hipDeviceGetAttribute(&maxThreadsPerBlockZ, hipDeviceAttributeMaxBlockDimZ, 0));

            int maxThreadsPerBlockPerDim = std::floor(std::pow(maxThreadsPerBlock, 1.0 / std::min(size_t(3), sizeof...(ranges))));

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
                threads = dim3(threadsPerBlockX, 1, 1);
                blocks = dim3((parallelRanges.x + threadsPerBlockX - 1) / threadsPerBlockX, 1, 1);
            }
            else if constexpr (sizeof...(ranges) == 2)
            {
                int threadsPerBlockX = std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockX);
                int threadsPerBlockY = std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockY);
                threads = dim3(threadsPerBlockX, threadsPerBlockY, 1);
                blocks = dim3((parallelRanges.x + threadsPerBlockX - 1) / threadsPerBlockX,
                            (parallelRanges.y + threadsPerBlockY - 1) / threadsPerBlockY, 1);
            }
            else if constexpr (sizeof...(ranges) >= 3)
            {
                int threadsPerBlockX = std::max(std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockX),1);
                int threadsPerBlockY = std::max(std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockY),1);
                int threadsPerBlockZ = std::max(std::min(maxThreadsPerBlockPerDim, maxThreadsPerBlockZ),1);

                threads = dim3(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
                blocks = dim3((parallelRanges.x + threadsPerBlockX - 1) / threadsPerBlockX,
                            (parallelRanges.y + threadsPerBlockY - 1) / threadsPerBlockY,
                            (parallelRanges.z + threadsPerBlockZ - 1) / threadsPerBlockZ);
            }
            auto ceilDiv = [](unsigned long long a, unsigned long long b){ return (a + b - 1ULL)/b; };
            unsigned long long needX = ceilDiv(parallelRanges.x, threads.x ? threads.x : 1);
            unsigned long long needY = (sizeof...(ranges)>1)? ceilDiv(parallelRanges.y, threads.y ? threads.y : 1) : 1ULL;
            unsigned long long needZ = (sizeof...(ranges)>2)? ceilDiv(parallelRanges.z, threads.z ? threads.z : 1) : 1ULL;

            int limX=1, limY=1, limZ=1;
            hipDeviceGetAttribute(&limX, hipDeviceAttributeMaxGridDimX, 0);
            if constexpr (sizeof...(ranges)>1) hipDeviceGetAttribute(&limY, hipDeviceAttributeMaxGridDimY, 0);
            if constexpr (sizeof...(ranges)>2) hipDeviceGetAttribute(&limZ, hipDeviceAttributeMaxGridDimZ, 0);

            unsigned long long useX = std::min<unsigned long long>(needX, limX);
            unsigned long long useY = std::min<unsigned long long>(needY, limY);
            unsigned long long useZ = std::min<unsigned long long>(needZ, limZ);

            int smCount=0; hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, 0);

            void *kernelPtr;
            if constexpr (sizeof(T_func) > maxFormalArgSize)
                kernelPtr = (void*)forEachHIPPtr<sendAll, T_func, T_ranges...>;
            else
                kernelPtr = (void*)forEachHIP<sendAll, T_func, T_ranges...>;

            int activePerSM=0;
            hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &activePerSM,
                kernelPtr,
                threads.x * threads.y * threads.z,
                shared_memory*threads.x * threads.y * threads.z);

            unsigned long long inflightLimit = (unsigned long long)activePerSM * (unsigned long long)smCount;

            auto trim = [&](unsigned long long &d, unsigned long long &needOther){
                while (useX * useY * useZ > inflightLimit && d > 1) {
                    d = (d + 1) / 2;
                }
            };
            trim(useX, needX); trim(useY, needY); trim(useZ, needZ);

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

            if constexpr (sizeof(T_func) > impl::maxFormalArgSize)
            {
                T_func* data=nullptr;
                HIP_ERROR_CHECK(hipMalloc(&data, sizeof(T_func)));
                HIP_ERROR_CHECK(hipMemcpy(data, &func, sizeof(T_func), hipMemcpyHostToDevice));
                hipLaunchKernelGGL(impl::forEachHIPPtr<sendAll,T_func>, blocks, threads, shared_memory, 0, data, chunks, ranges...);
                HIP_ERROR_CHECK(hipGetLastError());
                HIP_ERROR_CHECK(hipFree(data));
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
            T_data *reductionSites;
            HIP_ERROR_CHECK(hipMallocManaged(&reductionSites, nBlocks * sizeof(T_data)));

            for (int i=0; i < nBlocks; ++i)
            {
                reductionSites[i] = initialValue;
            }

            impl::reductionPhase1<T_mapper, T_reducer, T_data, sizeof...(T_ranges)> rp1(mapper, reducer, reductionSites, initialValue);

            size_t shared_memory = threads.x * threads.y * threads.z * sizeof(T_data);

            applyKernelCore<true>(rp1, threads, blocks, chunks, shared_memory, ranges...);
            hipDeviceSynchronize();

            {
                impl::reductionPhase2<T_reducer, T_data> rp2(reducer, reductionSites, nBlocks);
                impl::buildDecompLogic<false, decltype(rp2)>(threads, blocks, chunks, 0, Range(0,nBlocks-1));
                applyKernelCore(impl::reductionPhase2<T_reducer, T_data>(reducer, reductionSites, nBlocks), threads, blocks, chunks, 0 ,Range(0,nBlocks-1));
                hipDeviceSynchronize();
                nBlocks/=2;
            }

            T_data result;
            hipMemcpy(&result, reductionSites, sizeof(T_data), hipMemcpyDeviceToHost);
            hipFree(reductionSites);

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
            std::cout << "HIP" << std::endl;
            int device;
            HIP_ERROR_CHECK(hipGetDevice(&device));
            hipDeviceProp_t prop;
            HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, device));
            std::cout << "HIP device: " << prop.name << std::endl;
            std::cout << "HIP device compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "HIP device total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
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

    } // namespace hip
} // namespace portableWrapper
#endif // USE_HIP

#endif
