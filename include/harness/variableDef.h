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

namespace SAMS {

    /**
     * Struct representing a single dimension of a variable (zones and ghost cells)
     * lowerGhosts: number of ghost cells on the lower side of the dimension
     * upperGhosts: number of ghost cells on the upper side of the dimension
     */
    struct dimension{
        size_t lowerGhosts=0;
        size_t upperGhosts=0;
        size_t zones=0;
        staggerType stagger=staggerType::CENTRED;
        std::string axisName="";

        dimension() = default;
        //Single equal numbers of ghost cells, with and without axis name
        dimension(size_t ghosts)
            : lowerGhosts(ghosts), upperGhosts(ghosts) {}
        dimension(const std::string& axisName, size_t ghosts)
            : lowerGhosts(ghosts), upperGhosts(ghosts), axisName(axisName) {}

        dimension(size_t ghosts, staggerType stagger)
            : lowerGhosts(ghosts), upperGhosts(ghosts), stagger(stagger) {}
        dimension(const std::string& axisName, size_t ghosts, staggerType stagger)
            : lowerGhosts(ghosts), upperGhosts(ghosts), stagger(stagger), axisName(axisName) {}

        //Separate numbers of ghost cells, with and without axis name
        dimension(size_t lowerGhosts, size_t upperGhosts)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts) {}
        dimension(const std::string& axisName, size_t lowerGhosts, size_t upperGhosts)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), axisName(axisName) {}

        //With staggering, with and without axis name
        dimension(size_t lowerGhosts, size_t upperGhosts, staggerType stagger)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), stagger(stagger) {}
        dimension(const std::string& axisName, size_t lowerGhosts, size_t upperGhosts, staggerType stagger)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), stagger(stagger), axisName(axisName) {}

        //With zones. Adding axis name would be redundant
        dimension(size_t lowerGhosts, size_t upperGhosts, size_t zones, staggerType stagger)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), zones(zones), stagger(stagger) {}

        size_t getZones() const {
            if(zones == 0){
                throw std::runtime_error("Error: dimension zones not set\n");
            }
            return zones;
        }

        size_t getCells() const {
            if(zones == 0){
                throw std::runtime_error("Error: dimension zones not set\n");
            }
            const auto& sreg = getstaggerRegistry();
            size_t cells = sreg.getExtraCells(stagger);
            return zones + cells;
        }

        size_t getTotalCells() const {
            if(zones == 0){
                throw std::runtime_error("Error: dimension zones not set\n");
            }
            const auto& sreg = getstaggerRegistry();
            size_t cells = sreg.getExtraCells(stagger);
            return zones + cells + lowerGhosts + upperGhosts;
        }

        size_t getLowerGhosts() const {
            return lowerGhosts;
        }

        size_t getUpperGhosts() const {
            return upperGhosts;
        }
        
    };

    /**
     * Class representing a variable definition
     */
    class variableDef{
        private:

        std::array<dimension, MAX_RANK> dimensions;
        typeID varType;
        int rank=0;
        memorySpace memSpace=memorySpace::DEFAULT;

        /**
         * Pointer to the data. This is a void pointer because the type is not known at compile time.
         */
        void* dataPtr=nullptr;

        template<int level, typename T, typename... Args>
        void setDimensions(T arg, Args... args){
            static_assert(std::is_same_v<T, dimension>, "Error: variableDef setDimensions arguments must be of type dimension");
            dimensions[level] = arg;
            if constexpr(sizeof...(args) > 0){
                setDimensions<level+1>(args...);
            }
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
            if(rank<1 || rank>MAX_RANK){
                throw std::runtime_error("Error: variableDef rank must be between 1 and " + std::to_string(MAX_RANK) + "\n");
            }
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

        void setStagger(int dim, staggerType stagger){
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef setStagger dimension out of range\n");
            }
            dimensions[dim].stagger = stagger;
        }

        void setAxisName(int dim, const std::string& axisName){
            if(dim<0 || dim>=rank){
                throw std::runtime_error("Error: variableDef setAxisName dimension out of range\n");
            }
            dimensions[dim].axisName = axisName;
            getaxisRegistry().registerAxis(axisName);
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
                    dimensions[i].zones = getaxisRegistry().getElements(dimensions[i].axisName);
                }
                size_t staggerExtra = getstaggerRegistry().getExtraCells(dimensions[i].stagger);
                totalSize *= (dimensions[i].zones + dimensions[i].lowerGhosts + dimensions[i].upperGhosts + staggerExtra);
            }
            memoryRegistry &memHandler = getmemoryRegistry();
            dataPtr = memHandler.allocate(totalSize, memSpace);
            if(!dataPtr){
                throw std::runtime_error("Error: variableDef allocation failed\n");
            }
        }

        /**
         * Deallocate memory for the variable
         */
        void deallocate(){
            memoryRegistry &memHandler = getmemoryRegistry();
            memHandler.deallocate(dataPtr);
            dataPtr = nullptr;
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

    };

} //namespace SAMS

#endif //SAMS_VARIABLEDEF_H