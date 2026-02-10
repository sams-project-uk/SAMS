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
#include <complex>

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
        TYPE_UINT32 = 4,
        TYPE_UINT64 = 5,
        TYPE_COMPLEX_DOUBLE = 6,
        TYPE_COMPLEX_FLOAT = 7
    };

    struct typeHandle{
        int ID;
        typeHandle() : ID(-1) {}
        //typeHandle(int id) : ID(id) {}
        typeHandle(typeID id) : ID(static_cast<int>(id)) {}
        typeHandle(int id) : ID(id) {}
        operator int() const { return ID; }
    };

    /**
     * This class is unneeded until we implement MPI support
     */
    class typeRegistry {
        private:
#ifdef USE_MPI
/** Map from a harness typeID to an MPI_Datatype
 * C++ standard requires that std::complex be binary compatible with C complex types, so use the same MPI types
 */
        std::vector<MPI_Datatype> mpiTypes{MPI_DOUBLE, MPI_FLOAT, MPI_INT32_T, MPI_INT64_T, MPI_UINT32_T, MPI_UINT64_T, MPI_C_DOUBLE_COMPLEX, MPI_C_FLOAT_COMPLEX};
#endif
        std::vector<size_t> typeSizes{sizeof(double), sizeof(float), sizeof(int32_t), sizeof(int64_t), sizeof(uint32_t), sizeof(uint64_t), sizeof(std::complex<double>), sizeof(std::complex<float>)};
        std::vector<std::string> typeNames{"double", "float", "int32_t", "int64_t", "uint32_t", "uint64_t", "complex<double>", "complex<float>"};

        std::unordered_map<std::type_index, typeID> typeMap{
            {std::type_index(typeid(double)), typeID::TYPE_DOUBLE},
            {std::type_index(typeid(float)), typeID::TYPE_FLOAT},
            {std::type_index(typeid(int32_t)), typeID::TYPE_INT32},
            {std::type_index(typeid(int64_t)), typeID::TYPE_INT64},
            {std::type_index(typeid(uint32_t)), typeID::TYPE_UINT32},
            {std::type_index(typeid(uint64_t)), typeID::TYPE_UINT64},
            {std::type_index(typeid(std::complex<double>)), typeID::TYPE_COMPLEX_DOUBLE},
            {std::type_index(typeid(std::complex<float>)), typeID::TYPE_COMPLEX_FLOAT}
        };

        typeRegistry() = default;

        friend typeRegistry& gettypeRegistry();

        template<typename T>
        auto getTypeIterator(bool allowArrays=true) {
            if (allowArrays){
                return typeMap.find(std::type_index(typeid(std::decay_t<std::remove_extent_t<T>>)));
            } else {
                return typeMap.find(std::type_index(typeid(std::decay_t<T>)));
            }
        }

        template<typename T>
        std::string getDemangledName() {
            #if defined (HAS_DEMANGLE) && !defined(STRICT_STANDARDS)
            return demangle<T>();
            #else
            //std::type_info.name is NOT guaranteed to have any properties, so we have to use the hash_code if we can't demangle
            return std::to_string(typeid(T).hash_code());
            #endif
        }

        public:

        /**
         * Get the size in bytes of a typeID
         */
        size_t getSize(typeHandle t) {
            return typeSizes[static_cast<int>(t)];
        }

        /**
         * Get the size in bytes of a C++ type
         */
        template<typename T>
        size_t getSize() {
            auto it = getTypeIterator<T>();
            if (it == typeMap.end()) throw std::runtime_error("Type not registered with typeRegistry.");
            typeID t = it->second;
            return typeSizes[static_cast<int>(t)];
        }



        /**
         * Get the MPI_Datatype corresponding to a typeID
         */
        MPI_Datatype getMPIType([[maybe_unused]] typeHandle t) {
            #ifdef USE_MPI
            return mpiTypes[static_cast<int>(t)];
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Get the MPI_Datatype corresponding to a C++ type
         * @tparam T The C++ type to look up
         * @throws std::runtime_error if the type is not registered
         * @return The corresponding MPI_Datatype
         */
        template<typename T>
        MPI_Datatype getMPIType() {
            #ifdef USE_MPI
            auto it = getTypeIterator<T>();
            if (it == typeMap.end()) throw std::runtime_error("Type not registered with typeRegistry MPIT.");
            typeID t = it->second;
            return mpiTypes[static_cast<int>(t)];
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Get the name of a typeID
         */
        std::string getTypeName([[maybe_unused]] typeHandle t) {
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
        typeHandle getTypeID() {
            auto it = typeMap.find(std::type_index(typeid(T)));
            if (it == typeMap.end()) throw std::runtime_error("Type not registered with typeRegistry.");
            return it->second;
        }

        /**
         * Register a new type with the typeRegistry, returning a typeHandle
         * @tparam T The C++ type to register
         * @param typeName The name of the type
         * @throw std::runtime_error if the type is already registered
         * @return The corresponding harness typeID
         */
        template<typename T>
        typeHandle registerType(const std::string& typeName) {
            auto it = typeMap.find(std::type_index(typeid(T)));
            if (it != typeMap.end()) throw std::runtime_error("Type already registered with typeRegistry.");
            int newID = static_cast<int>(typeSizes.size());
            typeMap[std::type_index(typeid(T))] = static_cast<typeID>(newID);
            typeSizes.push_back(sizeof(T));
            typeNames.push_back(typeName);
            return typeHandle(newID);
        }

        /**
         * Register a new type with the typeRegistry, returning a typeHandle but using an automatic name
         * @tparam T The C++ type to register
         * @throw std::runtime_error if the type is already registered
         * @return The corresponding harness typeID
         */
        template<typename T>
        typeHandle registerType() {
            std::string typeName = getDemangledName<T>();
            return registerType<T>(typeName);
        }

        /**
         * Register a new type including an MPI_Datatype with the typeRegistry, returning a typeHandle
         * @tparam T The C++ type to register
         * @param typeName The name of the type
         * @param mpiType The MPI_Datatype corresponding to the type
         * @throw std::runtime_error if the type is already registered
         * @return The corresponding harness typeID
         */
        template<typename T>
        typeHandle registerType(const std::string& typeName, [[maybe_unused]] MPI_Datatype mpiType) {
            auto it = typeMap.find(std::type_index(typeid(T)));
            if (it != typeMap.end()) throw std::runtime_error("Type already registered with typeRegistry.");
            int newID = static_cast<int>(typeSizes.size());
            typeMap[std::type_index(typeid(T))] = static_cast<typeID>(newID);
            typeSizes.push_back(sizeof(T));
            typeNames.push_back(typeName);
            #ifdef USE_MPI
            mpiTypes.push_back(mpiType);
            #endif
            return typeHandle(newID);
        }

        /**
         * Register a new type including an MPI_Datatype with the typeRegistry, returning a typeHandle but using an automatic name
         * @tparam T The C++ type to register
         * @param mpiType The MPI_Datatype corresponding to the type
         * @throw std::runtime_error if the type is already registered
         * @return The corresponding harness typeID
         */
        template<typename T>
        typeHandle registerType(MPI_Datatype mpiType) {
            std::string typeName = getDemangledName<T>();
            return registerType<T>(typeName, mpiType);
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