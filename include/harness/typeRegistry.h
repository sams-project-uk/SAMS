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

namespace SAMS {

    /**
     * Built in types supported by the typeRegistery
     */
    enum class typeID {
        TYPE_DOUBLE = 0,
        TYPE_FLOAT = 1,
        TYPE_INT32 = 2,
        TYPE_INT64 = 3
    };

    /**
     * This class is unneeded until we implement MPI support
     */
    class typeRegistry {
        private:
        //std::vector<MPI_Datatype> mpiTypes{MPI_DOUBLE, MPI_FLOAT, MPI_INT, MPI_LONG};
        std::vector<size_t> typeSizes{sizeof(double), sizeof(float), sizeof(int32_t), sizeof(int64_t)};
        std::unordered_map<std::type_index, typeID> typeMap{
            {std::type_index(typeid(double)), typeID::TYPE_DOUBLE},
            {std::type_index(typeid(float)), typeID::TYPE_FLOAT},
            {std::type_index(typeid(int32_t)), typeID::TYPE_INT32},
            {std::type_index(typeid(int64_t)), typeID::TYPE_INT64}
        };

        typeRegistry() = default;

        friend typeRegistry& gettypeRegistry();

        public:

        size_t getSize(typeID t) {
            return typeSizes[static_cast<int>(t)];
        }

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