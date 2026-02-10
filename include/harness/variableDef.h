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
#include <memory>
#include <vector>
#include "typeRegistry.h"
#include "harnessDef.h"
#include "memoryRegistry.h"
#include "staggerRegistry.h"
#include "axisRegistry.h"
#include "mpiManager.h"
#include "boundaryConditions.h"

namespace SAMS {

    /**
     * Class representing a variable definition
     * Note that this class does not manage the data itself, only the metadata and MPI types
     * Also, note that this class does not deep copy the data pointer on copy, so it also does not manage the data memory
     */
    class variableDef{
        private:
        template<int i>
        friend class MPIManager;

        typeHandle varType;
        int rank=0;
        MPIManager<MPI_DECOMPOSITION_RANK>& mpiMgr;
        portableWrapper::arrayTags memSpace;
        axisRegistry &axisReg;
        memoryRegistry &memReg;

        std::array<dimension, MAX_RANK> dimensions;
        std::array<MPIAxis, MAX_RANK> mpiAxes;
        std::array<bool, MAX_RANK*2> isEdge;
        MPI_Datatype mpiType;
        MPI_Datatype mpiSend[2*MAX_RANK]; //Array of MPI_Datatypes for sending in each dimension (lower and upper)
        MPI_Datatype mpiRecv[2*MAX_RANK]; //Array of MPI_Datatypes for receiving in each dimension (lower and upper)
        std::array<std::array<std::vector<std::shared_ptr<boundaryConditions>>, 2>, MAX_RANK> boundaryConditionList;

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

    public:
    /**
     * Constructor
     * @param rank The rank of the variable (1-MAX_RANK)
     * @param varType The type of the variable (typeHandle)
     * @param memSpace The memory space of the variable (portableWrapper::arrayTags)
     */
        variableDef(SAMS::MPIManager<MPI_DECOMPOSITION_RANK>& mpiManager, axisRegistry &axisRegistry,
            SAMS::memoryRegistry &memRegistry,
            int rank, typeHandle varType, portableWrapper::arrayTags memSpace)
            : varType(varType), rank(rank), mpiMgr(mpiManager), memSpace(memSpace), axisReg(axisRegistry), memReg(memRegistry)
            {
            mpiType = gettypeRegistry().getMPIType(varType);
            if(rank<1 || rank>MAX_RANK){
                throw std::runtime_error("Error: variableDef rank must be between 1 and " + std::to_string(MAX_RANK) + "\n");
            }
            initMPITypeArrays();
        }

        template<typename... Args>
        variableDef(
            SAMS::MPIManager<MPI_DECOMPOSITION_RANK>& mpiManager, axisRegistry &axisRegistry,
            SAMS::memoryRegistry &memRegistry,
            typeHandle varType, portableWrapper::arrayTags memSpace, Args... args)
            : variableDef(mpiManager, axisRegistry, memRegistry, sizeof...(args), varType, memSpace) {
            static_assert(sizeof...(args) <= MAX_RANK, "Error: variableDef rank exceeds MAX_RANK");
            setDimensions<0>(args...);
        }

        // No copy assignment or move assignment because of reference members
        variableDef& operator=(const variableDef&) = delete;
        variableDef& operator=(variableDef&&) = delete;

        //Both copy and move constructors are defaulted
        variableDef(const variableDef&) = default;
        variableDef(variableDef&&) = default;

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
                    axisReg.setMaxGhosts(other.dimensions[i].axisName, other.dimensions[i].lowerGhosts, other.dimensions[i].upperGhosts);
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
                    const dimension& src = axisReg.getDimension(dimensions[i].axisName);
                    axisReg.setMaxGhosts(dimensions[i].axisName, dimensions[i].lowerGhosts, dimensions[i].upperGhosts);
                    //Update the dimension from the Canonical axis. This updates the number of zones
                    //and the MPI decomposition info if applicable
                    dimensions[i].getInfoFrom(src);
                    isEdge[i*2] = mpiMgr.isEdge(dimensions[i].mpiAxis, SAMS::domain::edges::lower);
                    isEdge[i*2+1] = mpiMgr.isEdge(dimensions[i].mpiAxis, SAMS::domain::edges::upper);
                }
                totalSize *= (dimensions[i].getLocalNativeDomainElements() + dimensions[i].lowerGhosts + dimensions[i].upperGhosts);
            }
            memoryRegistry &memHandler = memReg;
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
            memoryRegistry &memHandler = memReg;
            memHandler.deallocate(dataPtr);
            dataPtr = nullptr;
            //Clear the MPI types
            clearMPITypeArrays();
        }

        void haloExchange(){
            mpiMgr.haloExchange(dataPtr, rank, mpiSend, mpiRecv);
        }

        void haloExchange(int axis){
            mpiMgr.haloExchange(dataPtr, rank, mpiSend, mpiRecv, axis);
        }

        void haloExchange(std::string axisName){
            int axis = axisReg.getMPIAxis(axisName);
            mpiMgr.haloExchange(dataPtr, rank, mpiSend, mpiRecv, axis);
        }

        void haloExchange(int axis, SAMS::domain::edges edgeType){
            mpiMgr.haloExchange(dataPtr, rank, mpiSend, mpiRecv, axis, edgeType);
        }

        void haloExchange(std::string axisName, SAMS::domain::edges edgeType){
            int axis = axisReg.getMPIAxis(axisName);
            mpiMgr.haloExchange(dataPtr, rank, mpiSend, mpiRecv, axis, edgeType);
        }

        void haloExchange(SAMS::MPIManager<MPI_DECOMPOSITION_RANK>& customMPIManager){
            customMPIManager.haloExchange(dataPtr, rank, mpiSend, mpiRecv);
        }



        /**
         * Get a pointer to the data
         */
        template<typename T>
        T* getDataPtr() const {
            return static_cast<T*>(dataPtr);
        }

        /**
         * Get a pointer to data using the normal STL function name
         */
        void* data() const {
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
        typeHandle getType() const {
            return varType;
        }

        MPI_Datatype getMPIType() const {
            return mpiType;
        }

        /**
         * Get the memory space of the variable
         */
        portableWrapper::arrayTags getMemorySpace() const {
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
         * Get a single dimension by name
         * @param axisName The name of the axis to get the dimension for
         */
        const dimension& getDimension(const std::string& axisName) const {
            for(int i=0; i<rank; i++){
                if(dimensions[i].axisName == axisName){
                    return dimensions[i];
                }
            }
            throw std::runtime_error("Error: variableDef getDimension axis name not found\n");
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

        /**
         * Get a single dimension by name
         * @param axisName The name of the axis to get the dimension for
         */
        dimension& getDimension(const std::string& axisName) {
            for(int i=0; i<rank; i++){
                if(dimensions[i].axisName == axisName){
                    return dimensions[i];
                }
            }
            throw std::runtime_error("Error: variableDef getDimension axis name not found\n");
        }

        /**
         * Add a boundary condition to a specified edge of a specified dimension
         * @param dim The dimension to add the boundary condition to (0 to rank-1)
         * @param edge The edge to add the boundary condition to (SAMS::domain::edges)
         * @param bc The boundary condition to add (shared_ptr to boundaryConditions or derived class)
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         */
        template<typename T>
        std::shared_ptr<boundaryConditions> addBoundaryCondition(int dim, SAMS::domain::edges edge, std::shared_ptr<T> bc){
            static_assert(std::is_base_of<boundaryConditions, T>::value, "Error: variableDef addBoundaryCondition bc must be derived from boundaryConditions");
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef addBoundaryCondition dimension out of range\n");
            }
            boundaryConditionList[dim][static_cast<int>(edge)].push_back(bc);
            return bc;
        }

        /**
         * Add a boundary condition to both edges of a specified dimension
         * @param dim The dimension to add the boundary condition to (0 to rank-1)
         * @param bc The boundary condition to add (shared_ptr to boundaryConditions or derived class)
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         * @note This adds the same boundary condition instance to both edges
         */
        template<typename T>
        std::shared_ptr<boundaryConditions> addBoundaryCondition(int dim, std::shared_ptr<T> bc){
            static_assert(std::is_base_of<boundaryConditions, T>::value, "Error: variableDef addBoundaryCondition bc must be derived from boundaryConditions");
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef addBoundaryCondition dimension out of range\n");
            }
            std::shared_ptr<boundaryConditions> bc_ptr = addBoundaryCondition(dim, SAMS::domain::edges::lower, bc);
            addBoundaryCondition(dim, SAMS::domain::edges::upper, bc);
            return bc_ptr;
        }

        /**
         * Add a boundary condition to both edges of all dimensions
         * @param bc The boundary condition to add (shared_ptr to boundaryConditions or derived class)
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         * @note This adds the same boundary condition instance to all edges of all dimensions
         */
        template<typename T>
        std::shared_ptr<boundaryConditions> addBoundaryCondition(std::shared_ptr<T> bc){
            static_assert(std::is_base_of<boundaryConditions, T>::value, "Error: variableDef addBoundaryCondition bc must be derived from boundaryConditions");
            std::shared_ptr<boundaryConditions> bc_ptr = bc;
            for(int dim=0; dim<rank; dim++){
                addBoundaryCondition(dim, bc_ptr);
            }
            return bc_ptr;
        }


        /** 
         * Add a boundary condition specified as an object (not a shared_ptr) to a specific edge of a specified dimension
         * @param dim The dimension to add the boundary condition to (0 to rank-1)
         * @param edge The edge to add the boundary condition to (SAMS::domain::edges)
         * @param bc The boundary condition to add (boundaryConditions or derived class)
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         */
        template<typename T>
        std::shared_ptr<boundaryConditions> addBoundaryCondition(int dim, SAMS::domain::edges edge, const T& bc){
            static_assert(std::is_base_of<boundaryConditions, T>::value, "Error: variableDef addBoundaryCondition bc must be derived from boundaryConditions");
            return addBoundaryCondition(dim, edge, std::make_shared<T>(bc));
        }

        /**
         * Add a boundary condition specified as an object (not a shared_ptr) to both edges of a specified dimension
         * @param dim The dimension to add the boundary condition to (0 to rank-1)
         * @param bc The boundary condition to add (boundaryConditions or derived class)
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         * @note This adds the same boundary condition instance to both edges
         */
        template<typename T>
        std::shared_ptr<boundaryConditions> addBoundaryCondition(int dim, const T& bc){
            static_assert(std::is_base_of<boundaryConditions, T>::value, "Error: variableDef addBoundaryCondition bc must be derived from boundaryConditions");
            return addBoundaryCondition(dim, std::make_shared<T>(bc));
        }

        /**
         * Add a boundary condition specified as an object (not a shared_ptr) to both edges of all dimensions
         * @param bc The boundary condition to add (boundaryConditions or derived class)
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         * @note This adds the same boundary condition instance to all edges of all dimensions
         */
        template<typename T>
        std::shared_ptr<boundaryConditions> addBoundaryCondition(const T& bc){
            static_assert(std::is_base_of<boundaryConditions, T>::value, "Error: variableDef addBoundaryCondition bc must be derived from boundaryConditions");
            return addBoundaryCondition(std::make_shared<T>(bc));
        }

        /**
         * Emplace a specified boundary condition to a specific edge of a specified dimension
         * @param dim The dimension to add the boundary condition to (0 to rank-1)
         * @param edge The edge to add the boundary condition to (SAMS::domain::edges)
         * @param args The arguments to construct the boundary condition
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         */
        template<typename T, typename... Args>
        std::shared_ptr<boundaryConditions> emplaceBoundaryCondition(int dim, SAMS::domain::edges edge, Args&&... args){
            static_assert(std::is_base_of<boundaryConditions, T>::value, "Error: variableDef emplaceBoundaryCondition bc must be derived from boundaryConditions");
            if constexpr (std::is_constructible_v<T, Args...>) {
                //Can construct just from the args
                return addBoundaryCondition(dim, edge, std::make_shared<T>(args...));
            } else if constexpr (std::is_constructible_v<T, variableDef&, Args...>) {
                //Need to pass variableDef as first argument
                return addBoundaryCondition(dim, edge, std::make_shared<T>(*this, std::forward<Args>(args)...));
            } else {
                static_assert(portableWrapper::alwaysFalse<T>::value, "Error: variableDef emplaceBoundaryCondition cannot construct boundary condition with given arguments");
            }
        }

        /**
         * Emplace a specified boundary condition to both edges of a specified dimension
         * @param dim The dimension to add the boundary condition to (0 to rank-1)
         * @param args The arguments to construct the boundary condition
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         * @note This adds the same boundary condition instance to both edges
         */
        template<typename T, typename... Args>
        std::shared_ptr<boundaryConditions> emplaceBoundaryCondition(int dim, Args&&... args){
            std::shared_ptr<boundaryConditions> bc_ptr = emplaceBoundaryCondition<T>(dim, SAMS::domain::edges::lower, std::forward<Args>(args)...);
            addBoundaryCondition<T>(dim, SAMS::domain::edges::upper, bc_ptr);
            return bc_ptr;
        }

        /**
         * Emplace a specified boundary condition to both edges of all dimensions
         * @param args The arguments to construct the boundary condition
         * @return The boundary condition added (shared_ptr to boundaryConditions)
         * @note This adds the same boundary condition instance to all edges of all dimensions
         */
        template<typename T, typename... Args>
        std::shared_ptr<boundaryConditions> emplaceBoundaryCondition(Args&&... args){
            std::shared_ptr<boundaryConditions> bc_ptr = emplaceBoundaryCondition<T>(0, std::forward<Args>(args)...);
            for(int dim=1; dim<rank; dim++){
                addBoundaryCondition<T>(dim, bc_ptr);
            }
            return bc_ptr;
        }

        /**
         * Call all boundary conditions on an edge and dimension
         * @param dim The dimension to call the boundary conditions on (0 to rank-1)
         * @param edge The edge to call the boundary conditions on (SAMS::domain::edges)
         */
        void applyBoundaryConditions(int dim, SAMS::domain::edges edge){
            haloExchange(dim, edge);
            //Check if we are on a real boundary
            //Get the MPI axis for this axis
            if (!isEdge[dim*2 + static_cast<int>(edge)]){
                //Not on a real boundary, so nothing to do
                return;
            }
            for(auto &bc : boundaryConditionList[dim][static_cast<int>(edge)]){
                bc->apply(dim, edge);
            }
        }

        /**
         * Call all boundary conditions on an edge and dimension passing the dimension by name
         * @param axisName The name of the axis to call the boundary conditions on
         * @param edge The edge to call the boundary conditions on (SAMS::domain::edges)
         */
        void applyBoundaryConditions(const std::string& axisName, SAMS::domain::edges edge){
            int dim = -1;
            for(int i=0; i<rank; i++){
                if(dimensions[i].axisName == axisName){
                    dim = i;
                    break;
                }
            }
            if(dim == -1){
                throw std::runtime_error("Error: variableDef applyBoundaryConditions axis name not found\n");
            }
            applyBoundaryConditions(dim, edge);
        }

        /**
         * Call all boundary conditions on a specified dimension
         * @param dim The dimension to call the boundary conditions on (0 to rank-1
         */
        void applyBoundaryConditions(int dim){
            applyBoundaryConditions(dim, SAMS::domain::edges::lower);
            applyBoundaryConditions(dim, SAMS::domain::edges::upper);
        }

        /**
         * Call all boundary conditions on a dimension specified by name
         * @param axisName The name of the axis to call the boundary conditions on
         */
        void applyBoundaryConditions(const std::string& axisName){
            int dim = -1;
            for(int i=0; i<rank; i++){
                if(dimensions[i].axisName == axisName){
                    dim = i;
                    break;
                }
            }
            if(dim == -1){
                throw std::runtime_error("Error: variableDef applyBoundaryConditions axis name not found\n");
            }
            applyBoundaryConditions(dim);
        }

        /**
         * Call all boundary conditions on all dimensions
         */
        void applyBoundaryConditions(){
            for(int dim=0; dim<rank; dim++){
                applyBoundaryConditions(dim);
            }
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
            auto &manager = memReg.getArrayManager();
            manager.wrapArray(array, static_cast<T*>(dataPtr), lowerBounds.data(), upperBounds.data());
        }

        /**
         * Build a portable array wrapping the variable data
         * @return The portable array
         */
        template<typename T, int Arank , portableWrapper::arrayTags tag>
        portableWrapper::portableArray<T, Arank, tag> getPPArray() const {
            portableWrapper::portableArray<T, Arank, tag> array;
            fillPPArray(array);
            return array;
        }

        template<typename T, int Arank , portableWrapper::arrayTags tag>
        operator portableWrapper::portableArray<T, Arank, tag>() const {
            return getPPArray<T, Arank, tag>();
        }

    };

} //namespace SAMS

#endif //SAMS_VARIABLEDEF_H
