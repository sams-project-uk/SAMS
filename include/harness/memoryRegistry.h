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
#ifndef SAMS_MEMORYREGISTRY_H
#define SAMS_MEMORYREGISTRY_H

#include <unordered_map>
#include "harnessDef.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace SAMS{

    /**
     * Enumeration of memory spaces
     * NONE: No memory allocated
     * DEFAULT: Default memory space (usually HOST)
     * HOST: Host memory (CPU)
     * DEVICE: Device memory (GPU) (or CPU memory if no GPU is available)
     */
    enum class memorySpace {
        NONE,
        HOST,
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_KOKKOS)
        DEVICE,
        DEFAULT=DEVICE
#else
        DEVICE=HOST,
        DEFAULT=HOST
#endif

    };

    class memoryRegistry{
        friend memoryRegistry& getmemoryRegistry();
        private:
        /**
         * This class represents a single block of memory allocated on either the CPU or GPU
         * Uses macros to determine which backend to use and then wraps the memory with RAII semantics
         */
            class memoryBlock{
                private:
                    void* dataPtr=nullptr;
                    size_t size=0;
                    memorySpace memSpace=memorySpace::NONE;

                    //Allocators
                    void allocateCPU(size_t size){
                        dataPtr = malloc(size);
                    }
#ifdef USE_CUDA //CUDA backend
                    void allocateGPU(size_t size){
                        cudaMalloc(&dataPtr, size);
                    }
#elif defined(USE_HIP) //HIP backend
                    void allocateGPU(size_t size){
                        hipMalloc(&dataPtr, size);
                    }
#elif defined(USE_KOKKOS) //Kokkos backend
                    void allocateGPU(size_t size){
                        dataPtr = Kokkos::kokkos_malloc<KOKKOS_EXECUTION_SPACE>(size);
                    }
#else //Fallback to full CPU
                    void allocateGPU(size_t size){
                        dataPtr = malloc(size);
                    }
#endif
                    //Deallocators
                    void deallocateCPU(){
                        //nullptr is safe here
                        free(dataPtr);
                    }
#ifdef USE_CUDA //CUDA backend
                    void deallocateGPU(){
                        if (!dataPtr) return;
                        cudaFree(dataPtr);
                    }
#elif defined(USE_HIP) //HIP backend
                    void deallocateGPU(){
                        if (!dataPtr) return;
                        hipFree(dataPtr);
                    }
#elif defined(USE_KOKKOS) //Kokkos backend
                    void deallocateGPU(){
                        if (!dataPtr) return;
                        Kokkos::kokkos_free<KOKKOS_EXECUTION_SPACE>(dataPtr);
                    }
#else //Fallback to full CPU
                    void deallocateGPU(){
                        if (!dataPtr) return;
                        free(dataPtr);
                    }
#endif

                public:
                    /**
                     * Constructor allocates memory in the given memory space
                     */
                    memoryBlock(size_t size, memorySpace memSpace)
                        : size(size), memSpace(memSpace) {
                        if(memSpace==memorySpace::HOST){
                            allocateCPU(size);
                        }
                        else if(memSpace==memorySpace::DEVICE){
                            allocateGPU(size);
                        }
                    }
                    /**
                     * Destructor deallocates memory
                     */
                    ~memoryBlock(){
                        if (!dataPtr) return;
                        if(memSpace==memorySpace::HOST){
                            deallocateCPU();
                        }
                        else if(memSpace==memorySpace::DEVICE){
                            deallocateGPU();
                        }
                    }
                    /**
                     * Disable copy semantics, enable move semantics
                     */
                    memoryBlock(const memoryBlock&) = delete; //disable copy constructor
                    memoryBlock& operator=(const memoryBlock&) = delete; //disable copy assignment
                    memoryBlock(memoryBlock&& other) noexcept //move constructor
                        : dataPtr(other.dataPtr), size(other.size), memSpace(other.memSpace) {
                        other.dataPtr = nullptr;
                        other.size = 0;
                        other.memSpace = memorySpace::NONE;
                    }

                    /**
                     * Returns a pointer to the data
                     */
                    void* getDataPtr(){
                        return dataPtr;
                    }

            };
            std::unordered_map<void*, memoryBlock> blocks;
            memoryRegistry() = default;
        public:

        /**
         * Allocates a block of memory in a given memory space with a given byte size
         */
        void* allocate(size_t size, memorySpace memSpace){
            //Relies on move semantics of memoryBlock
            memoryBlock block(size, memSpace);
            void* ptr = block.getDataPtr();
            if (!ptr) return nullptr; //allocation failed
            blocks.emplace(ptr, std::move(block));
            return ptr;
        }

        /**
         * Deallocates a previously allocated block of memory
         */
        void deallocate(void* ptr){
            auto it = blocks.find(ptr);
            if(it != blocks.end()){
                blocks.erase(it);
            }
        }
    };


    /**
     * Returns the singleton instance of the memoryRegistry
     */
    inline memoryRegistry& getmemoryRegistry(){
        static memoryRegistry instance;
        return instance;
    }

};

#endif //SAMS_MEMORYREGISTRY_H