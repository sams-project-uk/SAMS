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
#ifndef SAMS_VARIABLEDEF_H
#define SAMS_VARIABLEDEF_H

#include <array>
#include <stdexcept>
#include <string>
#include "typeRegistry.h"
#include "harnessDef.h"
#include "memoryRegistry.h"
#include "staggerRegistry.h"
#include "axisRegistry.h"
#include "mpiManager.h"

namespace SAMS {

    /**
     * Class representing a variable definition
     */
    class variableDef{
        private:
        template<int i>
        friend struct MPIManager;

        /**
         * To allow future expansion with different MPI managers
         */
        MPIManager<MPI_DECOMPOSITION_RANK>&mpiMgr = getMPIManager<MPI_DECOMPOSITION_RANK>();

        std::array<dimension, MAX_RANK> dimensions;
        typeID varType;
        MPI_Datatype mpiType;
        MPI_Datatype mpiSend[2*MAX_RANK]; //Array of MPI_Datatypes for sending in each dimension (lower and upper)
        MPI_Datatype mpiRecv[2*MAX_RANK]; //Array of MPI_Datatypes for receiving in each dimension (lower and upper)
        int rank=0;
        memorySpace memSpace=memorySpace::DEFAULT;

        /**
         * Pointer to the data. This is a void pointer because the type is not known at compile time.
         */
        void* dataPtr=nullptr;

        /**
         * Recursive parameter pack to set dimensions
         */
        template<int level, typename T, typename... Args>
        void setDimensions(T arg, Args... args){
            static_assert(std::is_same_v<T, dimension>, "Error: variableDef setDimensions arguments must be of type dimension");
            dimensions[level] = arg;
            if constexpr(sizeof...(args) > 0){
                setDimensions<level+1>(args...);
            }
        }

        /**
         * Create the MPI datatypes for this variable
         */
        void buildMPITypes(){
            mpiMgr.assignVariableMPITypes(rank, dimensions.data(), mpiSend, mpiRecv, mpiType);
        }

        /**
         * Clear the MPI type arrays
         */
        void clearMPITypeArrays(){
            mpiMgr.releaseVariableMPITypes(rank, mpiSend, mpiRecv);     
        }

        /**
         * Initialize the MPI type arrays to MPI_DATATYPE_NULL
         */
        void initMPITypeArrays(){
            for(int i=0; i<2*MAX_RANK; i++){
                mpiSend[i] = MPI_DATATYPE_NULL;
                mpiRecv[i] = MPI_DATATYPE_NULL;
            }
        }

        /**
         * Get the array of dimensions
         */
        std::array<dimension, MAX_RANK>& getDimensions() {
            return dimensions;
        }

        /**
         * Get a single dimension
         * @param dim The dimension to get (0 to rank-1)
         */
        dimension& getDimension(int dim) {
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef getDimension dimension out of range\n");
            }
            return dimensions[dim];
        }

    public:
    /**
     * Constructor
     * @param rank The rank of the variable (1-MAX_RANK)
     * @param varType The type of the variable (typeID)
     * @param memSpace The memory space of the variable (memorySpace)
     */
        variableDef(int rank, typeID varType, memorySpace memSpace)
            : varType(varType), rank(rank), memSpace(memSpace) {
            mpiType = gettypeRegistry().getMPIType(varType);
            if(rank<1 || rank>MAX_RANK){
                throw std::runtime_error("Error: variableDef rank must be between 1 and " + std::to_string(MAX_RANK) + "\n");
            }
            initMPITypeArrays();
        }

        template<typename... Args>
        variableDef(typeID varType, memorySpace memSpace, Args... args)
            : variableDef(sizeof...(args), varType, memSpace) {
            static_assert(sizeof...(args) <= MAX_RANK, "Error: variableDef rank exceeds MAX_RANK");
            setDimensions<0>(args...);
        }

        /**
         * This function makes two variableDefs consistent by taking the maximum ghost cells in each dimension. This produces the largest number of ghost cells required to hold both variableDefs.
         */
        void makeConsistent(const variableDef &other){
            if (other.rank != rank) {
                throw std::runtime_error("Error: variableDef ranks do not match\n");
            }
            if (other.varType != varType) {
                throw std::runtime_error("Error: variableDef types do not match\n");
            }
            if (other.memSpace != memSpace) {
                throw std::runtime_error("Error: variableDef memory spaces do not match\n");
            }
            for(int i=0; i<rank; i++){
                dimensions[i].lowerGhosts = std::max(dimensions[i].lowerGhosts, other.dimensions[i].lowerGhosts);
                dimensions[i].upperGhosts = std::max(dimensions[i].upperGhosts, other.dimensions[i].upperGhosts);

                //Tell the axis about the lower and upper ghost cells
                if (other.dimensions[i].axisName != "") {
                    getaxisRegistry().setMaxGhosts(other.dimensions[i].axisName, other.dimensions[i].lowerGhosts, other.dimensions[i].upperGhosts);
                }

                //Stagger must match
                if (other.dimensions[i].stagger != dimensions[i].stagger) {
                    throw std::runtime_error("Error: variableDef staggers do not match\n");
                }
                //If number of zones is set in both, they must match
                if (other.dimensions[i].zones != 0 && dimensions[i].zones != 0 && other.dimensions[i].zones != dimensions[i].zones) {
                    throw std::runtime_error("Error: variableDef zones do not match\n");
                }
                //If axis names are set in both, they must match
                if (other.dimensions[i].axisName != "" && dimensions[i].axisName != "" && other.dimensions[i].axisName != dimensions[i].axisName) {
                    throw std::runtime_error("Error: variableDef axis names do not match\n");
                }
            }
        }

        /**
         * This function sets the number of ghost cells in a given dimension
         * @param dim The dimension to set (0 to rank-1)
         * @param lowerGhosts The number of ghost cells on the lower side of the dimension
         * @param upperGhosts The number of ghost cells on the upper side of the dimension
         */
        void setGhosts(int dim, size_t lowerGhosts, size_t upperGhosts){
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef setGhosts dimension out of range\n");
            }
            dimensions[dim].lowerGhosts = lowerGhosts;
            dimensions[dim].upperGhosts = upperGhosts;
        }

        /**
         * This function sets the stagger type in a given dimension
         * @param dim The dimension to set (0 to rank-1)
         * @param stagger The stagger type to set (staggerType)
         */
        void setStagger(int dim, staggerType stagger){
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef setStagger dimension out of range\n");
            }
            dimensions[dim].stagger = stagger;
        }

        /**
         * This function sets the axis name in a given dimension
         * @param dim The dimension to set (0 to rank-1)
         * @param axisName The axis name to set (string)
         */
        void setAxisName(int dim, const std::string& axisName){
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef setAxisName dimension out of range\n");
            }
            dimensions[dim].axisName = axisName;
        }

        /**
         * This function sets the number of actual zones in a given dimension
         * @param dim The dimension to set (0 to rank-1)
         * @param zones The number of zones in that dimension
         */
        void setZones(int dim, size_t zones){
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef setZones dimension out of range\n");
            }
            dimensions[dim].zones = zones;
            dimensions[dim].zonesLocal = zones;
        }

        /**
         * Allocate memory for the variable
         */
        void allocate(){
            if(dataPtr){
                throw std::runtime_error("Error: variableDef already allocated\n");
            }
            size_t totalSize = gettypeRegistry().getSize(varType);
            for(int i=0; i<rank; i++){
                if (dimensions[i].zones == 0) {
                    const dimension& src = getaxisRegistry().getDimension(dimensions[i].axisName);
                    getaxisRegistry().setMaxGhosts(dimensions[i].axisName, dimensions[i].lowerGhosts, dimensions[i].upperGhosts);
                    //Update the dimension from the Canonical axis. This updates the number of zones
                    //and the MPI decomposition info if applicable
                    dimensions[i].getInfoFrom(src);
                }
                totalSize *= (dimensions[i].getLocalNativeDomainElements() + dimensions[i].lowerGhosts + dimensions[i].upperGhosts);
            }
            memoryRegistry &memHandler = getmemoryRegistry();
            dataPtr = memHandler.allocate(totalSize, memSpace);
            if(!dataPtr){
                throw std::runtime_error("Error: variableDef allocation failed\n");
            }
            //Request the MPI handler to build the MPI types
            buildMPITypes();
        }

        /**
         * Deallocate memory for the variable
         */
        void deallocate(){
            memoryRegistry &memHandler = getmemoryRegistry();
            memHandler.deallocate(dataPtr);
            dataPtr = nullptr;
            //Clear the MPI types
            clearMPITypeArrays();
        }

        void haloExchange(){
            MPIManager<MPI_DECOMPOSITION_RANK> &mpiMgr = getMPIManager<MPI_DECOMPOSITION_RANK>();
            mpiMgr.haloExchange(dataPtr, rank, mpiSend, mpiRecv);
        }

        /**
         * Get a pointer to the data
         */
        void* getDataPtr() const {
            return dataPtr;
        }

        /**
         * Get the rank of the variable
         */
        int getRank() const {
            return rank;
        }

        /**
         * Get the type of the variable
         */
        typeID getType() const {
            return varType;
        }

        MPI_Datatype getMPIType() const {
            return mpiType;
        }

        /**
         * Get the memory space of the variable
         */
        memorySpace getMemorySpace() const {
            return memSpace;
        }

        /**
         * Get the array of dimensions
         */
        const std::array<dimension, MAX_RANK>& getDimensions() const {
            return dimensions;
        }

        /**
         * Get a single dimension
         * @param dim The dimension to get (0 to rank-1)
         */
        const dimension& getDimension(int dim) const {
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef getDimension dimension out of range\n");
            }
            return dimensions[dim];
        }

        /**
         * Fill a portable array wrapping the variable data
         * @param array The portable array to fill
         */
        template<typename T, int Arank , portableWrapper::arrayTags tag>
        void fillPPArray(portableWrapper::portableArray<T, Arank, tag> & array) const {
            if (Arank != rank) {
                throw std::runtime_error("Error: variableDef buildArray rank does not match\n");
            }
            std::array<SIGNED_INDEX_TYPE, MAX_RANK> lowerBounds;
            std::array<SIGNED_INDEX_TYPE, MAX_RANK> upperBounds;
            for (int i=0; i<rank; i++) {
                lowerBounds[i] = dimensions[i].getLocalLB();
                upperBounds[i] = dimensions[i].getLocalUB();
            }
            auto &manager = getmemoryRegistry().getArrayManager();
            manager.wrap(array, static_cast<T*>(dataPtr), lowerBounds.data(), upperBounds.data());
        }

        /**
         * Build a portable array wrapping the variable data
         * @return The portable array
         */
        template<typename T, int Arank , portableWrapper::arrayTags tag>
        portableWrapper::portableArray<T, Arank, tag> buildPPArray() const {
            portableWrapper::portableArray<T, Arank, tag> array;
            fillPPArray(array);
            return array;
        }

        template<typename T, int Arank , portableWrapper::arrayTags tag>
        operator portableWrapper::portableArray<T, Arank, tag>() const {
            return buildPPArray<T, Arank, tag>();
        }

    };

} //namespace SAMS

#endif //SAMS_VARIABLEDEF_H
