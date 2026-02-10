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

#include "pp/manager.h"
#include "pp/array.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(USE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace SAMS{

    class memoryRegistry{
        //Array manager from portableWrapper to handle allocations
        portableWrapper::portableArrayManager arrayManager; 
        public:

        memoryRegistry() = default;

        /**
         * Returns the portable array manager used by the memory registry
         */
        portableWrapper::portableArrayManager& getArrayManager(){
            return arrayManager;
        }

        /**
         * Allocates a block of memory in a given memory space with a given byte size
         */
        void* allocate(size_t size, portableWrapper::arrayTags tag) {
            //Just provide a wrapper around the portableArrayManager
            void* data;
            if (tag == portableWrapper::arrayTags::host) {
                auto array = arrayManager.allocate<double, portableWrapper::arrayTags::host>(size/sizeof(double));
                data = array.data();
            }
            else if (tag == portableWrapper::arrayTags::accelerated) {
                auto array = arrayManager.allocate<double, portableWrapper::arrayTags::accelerated>(size/sizeof(double));
                data = array.data();
            }
            else {
                throw std::runtime_error("Unknown memory space.");
            }
            if (!data) {
                throw std::runtime_error("Memory allocation failed.");
            }
            return data;
        }

        /**
         * Deallocates a previously allocated block of memory
         */
        void deallocate(void* ptr){
            // Use arrayManager to deallocate
            arrayManager.deallocate(ptr);
        }

        /**
         * Finalize the memory registry, deallocating all managed memory
         */
        void finalize(){
            arrayManager.finalize();
        }

    };

};

#endif //SAMS_MEMORYREGISTRY_H
