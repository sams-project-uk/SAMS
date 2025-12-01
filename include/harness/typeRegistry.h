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
#ifndef SAMS_TYPEREGISTERY_H
#define SAMS_TYPEREGISTERY_H

#include <vector>
#include <unordered_map>
#include <typeindex>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "mpiDefaultTypes.h"

namespace SAMS {

    /**
     * Built in types supported by the typeRegistery
     */
    enum class typeID {
        TYPE_DOUBLE = 0,
        TYPE_FLOAT = 1,
        TYPE_INT32 = 2,
        TYPE_INT64 = 3,
        TYPE_USER_BASE = 1000
    };

    /**
     * This class is unneeded until we implement MPI support
     */
    class typeRegistry {
        private:
#ifdef USE_MPI
/** Map from a harness typeID to an MPI_Datatype */
        std::vector<MPI_Datatype> mpiTypes{MPI_DOUBLE, MPI_FLOAT, MPI_INT, MPI_LONG};
#endif
        std::vector<size_t> typeSizes{sizeof(double), sizeof(float), sizeof(int32_t), sizeof(int64_t)};
        std::vector<std::string> typeNames{"double", "float", "int32_t", "int64_t"};

        std::unordered_map<std::type_index, typeID> typeMap{
            {std::type_index(typeid(double)), typeID::TYPE_DOUBLE},
            {std::type_index(typeid(float)), typeID::TYPE_FLOAT},
            {std::type_index(typeid(int32_t)), typeID::TYPE_INT32},
            {std::type_index(typeid(int64_t)), typeID::TYPE_INT64}
        };

        typeRegistry() = default;

        friend typeRegistry& gettypeRegistry();

        public:

        /**
         * Get the size in bytes of a typeID
         */
        size_t getSize(typeID t) {
            return typeSizes[static_cast<int>(t)];
        }

        /**
         * Get the MPI_Datatype corresponding to a typeID
         */
        MPI_Datatype getMPIType(typeID t) {
            #ifdef USE_MPI
            return mpiTypes[static_cast<int>(t)];
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Get the name of a typeID
         */
        std::string getTypeName(typeID t) {
            return typeNames[static_cast<int>(t)];
        }

        /**
         * Get a type name from an MPI_Datatype
         */
        std::string getTypeName(MPI_Datatype mpiType) {
            if (mpiType == MPI_DATATYPE_NULL){
                return "Null MPI_Datatype";
            }
            #ifdef USE_MPI
            for (size_t i = 0; i < mpiTypes.size(); i++) {
                if (mpiTypes[i] == mpiType) {
                    return typeNames[i];
                }
            }
            #endif
            throw std::runtime_error("MPI_Datatype not registered in typeRegistry.");
        }

        /**
         * Get the harness data type corresponding to a C++ type
         * @throws std::runtime_error if the type is not registered
         * @tparam T The C++ type to look up
         * @return The corresponding harness typeID
         */
        template<typename T>
        typeID getTypeID() {
            auto it = typeMap.find(std::type_index(typeid(T)));
            if (it == typeMap.end()) throw std::runtime_error("Type not registered with typeRegistry.");
            return it->second;
        }

    };//class typeRegistry

    /**
     * Returns the singleton instance of the typeRegistry
     */
    inline typeRegistry& gettypeRegistry() {
        static typeRegistry instance;
        return instance;
    }
} //namespace SAMS

#endif //HARNESS_TYPEREGISTERY_H