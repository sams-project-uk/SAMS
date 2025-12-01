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
#ifndef SAMS_AXISREGISTRY_H
#define SAMS_AXISREGISTRY_H
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

#include "handles.h"
#include "memoryRegistry.h"
#include "staggerRegistry.h"
#include "pp/parallelWrapper.h"
#include "dimension.h"
#include "harnessDef.h"

namespace SAMS
{

    /**
     * Registry for managing axes in the SAMS framework
     * Number of elements is defined as number of EDGES in the axis. This
     * is one more than the number of ZONES.
     * The axis starts and ends are the EDGES of the first and last zones respectively
     */
    class axisRegistry
    {
        friend axisRegistry &getaxisRegistry();

    public:

        struct axisInfo
        {
            using lineArray = portableWrapper::acceleratedArray<T_dataType, 1>;
            /**
             * Simple class to tie together axis and delta arrays
             */
            struct axisValues
            {
                lineArray axis;
                lineArray delta;
            };

            /**
             * Create an axis based on a provided function that specifies the location of each axis point
             * @param stagger The staggering type of the axis
             * @param memSpace The memory space to allocate the axis in
             * @param fn The function to compute the axis values
             */
            template <typename dataFn>
            void createPositionAxis(staggerType stagger, memorySpace memSpace, dataFn fn)
            {
                //If axis already exists, return
                if (axisData.find(stagger) != axisData.end() &&
                    axisData[stagger].find(memSpace) != axisData[stagger].end())
                {
                    return;
                }

                auto& axisInfo = axisData[stagger][memSpace];
                
                SIGNED_INDEX_TYPE lowerBound = dim.getLB(stagger);
                SIGNED_INDEX_TYPE upperBound = dim.getUB(stagger);

                // Now initialize the axis
                portableWrapper::Range range(lowerBound, upperBound);
                manager.allocate(axisInfo.axis, range);
                auto axisArray = axisInfo.axis; //Copy for lambda capture without capturing the whole class
                portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) { axisArray(i) = fn(i); }, range);
                portableWrapper::fence();

                /*Move the upper bound back by one to compute delta (i.e. delta is the distance between edges,
                so by fencepost rule there is one less delta than edges)*/
                manager.allocate(axisInfo.delta, range);
                auto axisDelta = axisInfo.delta;
                //Can only calculate a delta for i>lowerBound
                range.lower_bound+=1;
                portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {axisDelta(i) = axisArray(i)-axisArray(i-1);}, range);
                //Now apply a boundary condition to get the first delta
                if (dim.periodic){
                    portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {
                            //Copy round the delta from the upper bound
                            axisDelta(i) = axisDelta(upperBound);
                    }, portableWrapper::Range(lowerBound, lowerBound));
                } else {
                    portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {
                            //Assume uniform spacing for first delta
                            axisDelta(i) = axisDelta(i+1);
                    }, portableWrapper::Range(lowerBound, lowerBound));
                }
                portableWrapper::fence();
            }

            /**
             * Create a delta axis based on a provided function that specifies the delta between each axis point and a lower boundary value
             * @param stagger The staggering type of the axis
             * @param memSpace The memory space to allocate the axis in
             * @param fn The function to compute the delta values
             * @param initialValue The initial value at the lower boundary of the axis
             */
            template<typename deltaFn>
            void createDeltaAxis(staggerType stagger, memorySpace memSpace, deltaFn fn, T_dataType initialValue)
            {
                //If axis already exists, return
                if (axisData.find(stagger) != axisData.end() &&
                    axisData[stagger].find(memSpace) != axisData[stagger].end())
                {
                    return;
                }
                //If not, create it
                auto axisInfo = axisData[stagger][memSpace];
                SIGNED_INDEX_TYPE lowerBound = dim.getLB(stagger);
                SIGNED_INDEX_TYPE upperBound = dim.getUB(stagger);
                SIGNED_INDEX_TYPE domainLower = dim.getDomainLB(stagger);

                // Now initialize the delta axis1
                portableWrapper::Range range(lowerBound, upperBound);
                manager.allocate(axisInfo.delta, range);
                auto axisDelta = axisInfo.delta; //Copy for lambda capture without capturing the whole class
                range.lower_bound+=1; //Delta only defined for i>lowerBound
                portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) { axisDelta(i) = fn(i); }, range);
                range.lower_bound-=1;
                //Now apply a boundary condition to get the first delta
                //Has to be done like this to be consistent with LARE expectations
                if (dim.periodic){
                    portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {
                        if (i == lowerBound) {
                            //Copy round the delta from the upper bound
                            axisDelta(i) = axisDelta(upperBound);
                        }
                    }, portableWrapper::Range(lowerBound-1, lowerBound-1));
                } else {
                    portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {
                        if (i == lowerBound) {
                            //Assume uniform spacing for first delta
                            axisDelta(i) = axisDelta(i+1);
                        }
                    }, portableWrapper::Range(lowerBound-1, lowerBound-1));
                }
                portableWrapper::fence();

                //Now calculate the position axis by integrating the delta
                manager.allocate(axisInfo.axis, range);
                auto axisArray = axisInfo.axis;
                portableWrapper::assign(axisArray, T_dataType(0));
                portableWrapper::fence();
                //Set lower boundary value
                portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {axisArray(i) = initialValue;}, portableWrapper::Range(domainLower, domainLower));
                //Integrate upwards
                portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {
                    for (SIGNED_INDEX_TYPE j = lowerBound; j <= i; ++j)
                    {
                        axisArray(i) += axisDelta(j);
                    }
                }, portableWrapper::Range(domainLower+1, upperBound));
                //Integrate downwards
                portableWrapper::applyKernel(LAMBDA(SIGNED_INDEX_TYPE i) {
                    for (SIGNED_INDEX_TYPE j = i + 1; j <= upperBound; ++j)
                    {
                        axisArray(i) -= axisDelta(j);
                    }
                }, portableWrapper::Range(lowerBound, domainLower-1));
                portableWrapper::fence();
            }

            /**
             * Convenience function to create a linear axis from a given lower bound and uniform grid spacing
             * @param stagger The staggering type of the axis
             * @param memSpace The memory space to allocate the axis in
             * @param delta The uniform grid spacing
             * @param lowerBound The lower bound of the axis (default 0.0)
             */
            void createLinearAxis(staggerType stagger, memorySpace memSpace, T_dataType delta, T_dataType lowerBound=0.0)
            {
                auto linearFn = LAMBDA(SIGNED_INDEX_TYPE i)->T_dataType
                {
                    return lowerBound + delta * static_cast<T_dataType>(i);
                };
                createPositionAxis(stagger, memSpace, linearFn);
            }

            void createCentredAxis(memorySpace memSpace)
            {
                auto& halfCellAxis = axisData[staggerType::HALF_CELL][memorySpace::HOST].axis;
                auto linearFn = LAMBDA(SIGNED_INDEX_TYPE i)->T_dataType
                {
                    return 0.5 * (halfCellAxis(i) + halfCellAxis(i-1));
                };
                createPositionAxis(staggerType::CENTRED, memSpace, linearFn);
            }


            void generateAxis(staggerType stagger, memorySpace memSpace)
            {
                if (memSpace == memorySpace::HOST) {
                    throw std::runtime_error("Error: HOST axis should have been created at domain setup\n");
                } else if (memSpace == memorySpace::DEVICE){
                    axisData[stagger][memSpace].axis = manager.makeDeviceAvailable(axisData[stagger][memorySpace::HOST].axis);
                    axisData[stagger][memSpace].delta = manager.makeDeviceAvailable(axisData[stagger][memorySpace::HOST].delta);
                } else {
                    throw std::runtime_error("Error: Unsupported memory space for axis generation\n");
                }
            }

        public:
            dimension dim{SAMS::staggerType::HALF_CELL};
            portableWrapper::portableArrayManager &manager;
            SIGNED_INDEX_TYPE localLB = 0;
            SIGNED_INDEX_TYPE localUB = 0;
            COUNT_TYPE maxLowerGhosts = 0;
            COUNT_TYPE maxUpperGhosts = 0;
            T_dataType axisMinVal = 0.0;
            T_dataType axisMaxVal = 1.0;
            bool periodic = false;
            MPIAxis MPIAxisIndex;

            axisInfo() = delete; // Delete default constructor
            axisInfo(portableWrapper::portableArrayManager &mgr)
                : manager(mgr) {}
            axisInfo(portableWrapper::portableArrayManager &mgr, MPIAxis mpiAxis)
                : manager(mgr), MPIAxisIndex(mpiAxis) {}

            axisInfo(const axisInfo &) = delete;
            axisInfo &operator=(const axisInfo &) = delete;
            axisInfo(axisInfo &&) = default;
            axisInfo &operator=(axisInfo &&) = default;
            std::unordered_map<staggerType, std::unordered_map<memorySpace, axisValues>> axisData;

            bool hasAxis(staggerType stagger, memorySpace memSpace)
            {
                return (axisData.find(stagger) != axisData.end() &&
                        axisData[stagger].find(memSpace) != axisData[stagger].end());
            }

            lineArray getPPAxis(staggerType stagger, memorySpace memSpace)
            {
                if (!hasAxis(stagger, memSpace))
                {
                    generateAxis(stagger, memSpace);
                }
                return axisData[stagger][memSpace].axis;
            }

            lineArray getPPDelta(staggerType stagger, memorySpace memSpace)
            {
                if (!hasAxis(stagger, memSpace))
                {
                    generateAxis(stagger, memSpace);
                }
                return axisData[stagger][memSpace].delta;
            }
        }; // struct axisInfo

        portableWrapper::portableArrayManager manager;
        std::unordered_map<std::string, axisInfo> axisMap;
        axisRegistry() = default;

        axisInfo& getAxis(const std::string &name)
        {
            auto it = axisMap.find(name);
            if (it == axisMap.end())
            {
                throw std::runtime_error("Error: axis " + name + " not found in registry\n");
            }
            return it->second;
        }

        const axisInfo& getAxis(const std::string &name) const
        {
            auto it = axisMap.find(name);
            if (it == axisMap.end())
            {
                throw std::runtime_error("Error: axis " + name + " not found in registry\n");
            }
            return it->second;
        }

    public:
        /**
         * Register a new axis with the given name
         * @param name The name of the axis to register
         */
        void registerAxis(const std::string &name)
        {
            axisMap.try_emplace(name, manager);
        }

        /** Register a new axis with a given name and associate it with a given MPI axis
         * @param name The name of the axis to register
         * @param mpiAxis The MPI axis index to associate with the axis
         */

        void registerAxis(const std::string &name, MPIAxis mpiAxis)
        {
            axisMap.try_emplace(name, manager, mpiAxis);
        }

        /**
         * Get the number of global elements for the given axis
         * Global means the total number of elements across all MPI ranks
         * @param name The name of the axis
         * @return The number of global elements
         */
        COUNT_TYPE getDomainElements(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getDomainElements();
        }

        COUNT_TYPE getDomainElements(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getDomainElements(stagger);
        }

        /**
         * Get the number of local elements for the given axis
         * Local means the number of elements on the current MPI rank
         * @param name The name of the axis
         * @return The number of local elements
         */
        COUNT_TYPE getLocalDomainElements(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalNativeDomainElements();
        }

        COUNT_TYPE getLocalDomainElements(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalDomainElements(stagger);
        }

        /**
         * @brief
         * Set the number of global elements for the given axis
         * @param name The name of the axis
         * @param elements The number of global elements
         * @param stagger The stagger type (default CENTRED)
         * @details
         * You can specify the number of points with any stagger type, but they are
         * always stored as the number of EDGES in the axis. Therefore, the number of elements
         * stored is the number of points minus the extra cells required for the given stagger type.
         */
        void setDomainElements(const std::string &name, COUNT_TYPE elements)
        {
            auto &ax = getAxis(name);
            ax.dim.setDomainElements(elements, staggerType::CENTRED);
            ax.dim.setLocalDomainElements(elements, staggerType::CENTRED);
        }

        /**
         * Set the number of local elements for the given axis
         * @param name The name of the axis
         * @param elementsLocal The number of local elements
         */
        void setLocalDomainElements(const std::string &name, COUNT_TYPE elementsLocal)
        {
            auto &ax = getAxis(name);
            ax.dim.setLocalDomainElements(elementsLocal, staggerType::HALF_CELL);
        }

        /**
         * Set the number of global elements for the given axis with specified staggering type
         * @param name The name of the axis
         * @param elements The number of global elements
         * @param stagger The staggering type
         */
        void setLocalDomainElements(const std::string &name, COUNT_TYPE elementsLocal, staggerType stagger)
        {
            auto &ax = getAxis(name);
            ax.dim.setLocalDomainElements(elementsLocal, stagger);
        }

        /**
         * Set the global Range for the given axis
         */
        void setGlobalBounds(const std::string &name, SIGNED_INDEX_TYPE lb, SIGNED_INDEX_TYPE ub)
        {
            auto &ax = getAxis(name);
            ax.dim.globalLowerIndex = static_cast<SIGNED_INDEX_TYPE>(lb);
            ax.dim.globalUpperIndex = static_cast<SIGNED_INDEX_TYPE>(ub);
            ax.localLB = lb;
            ax.localUB = ub;
        }

        /**
         * @brief This gets the full global Range for the given axis
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the global domain
         * Both Range are inclusive. The first cell of the real domain will be at 0 for a edge-based axis
         * and 1 for a cell-based axis. Lower ghost cells will be below this.
         * This version uses the native staggering of the axis.
         * @param name The name of the axis
         * @return A Range object
         */
        portableWrapper::Range getFullRange(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getRange();
        }

        /**
         * @brief This gets the full global Range for the given axis
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the global domain
         * Both Range are inclusive. The first cell of the real domain will be at 0 for a edge-based axis
         * and 1 for a cell-based axis. Lower ghost cells will be below this.
         * This version uses a specified staggering type.
         * @param name The name of the axis
         * @param stagger The staggering type
         * @return A Range object
         */
         portableWrapper::Range getFullRange(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getRange(stagger);
        }

        /**
         * @brief This gets the full global Range for the given axis assuming zero-based indexing
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the global domain
         * Both Range are inclusive. The first ghost cell is at zero and all cells count up from there.
         * This version uses the native staggering of the axis.
         * @param name The name of the axis
         * @return A Range object
         */
         portableWrapper::Range getFullRangeZeroBase(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getRangeZeroBase();
        }

        /**
         * @brief This gets the full global Range for the given axis assuming zero-based indexing
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the global domain
         * Both Range are inclusive. The first ghost cell is at zero and all cells count up from there.
         * This version uses a specified staggering type.
         * @param name The name of the axis
         * @param stagger The staggering type
         * @return A Range object
         */
         portableWrapper::Range getFullRangeZeroBase(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getRangeZeroBase(stagger);
        }

        /**
         * @brief This gets the local Range for the given axis(i.e. for the current MPI rank)
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the local domain
         * Both Range are inclusive. The first cell of the real domain will be at 0 for a edge-based axis
         * and 1 for a cell-based axis. Lower ghost cells will be below this.
         * This version uses the native staggering of the axis.
         * @param name The name of the axis
         * @return A Range object
         */
         portableWrapper::Range getLocalRange(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalRange();
        }

        /**
         * @brief This gets the local Range for the given axis(i.e. for the current MPI rank)
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the local domain
         * Both Range are inclusive. The first cell of the real domain will be at 0 for a edge-based axis
         * and 1 for a cell-based axis. Lower ghost cells will be below this.
         * This version uses a specified staggering type.
         * @param name The name of the axis
         * @param stagger The staggering type
         * @return A Range object
         */
         portableWrapper::Range getLocalRange(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalRange(stagger);
        }

        /**
         * @brief This gets the local Range for the given axis(i.e. for the current MPI rank) assuming zero-based indexing
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the local domain
         * Both Range are inclusive. The first ghost cell is at zero and all cells count up from there.
         * This version uses the native staggering of the axis.
         * @param name The name of the axis
         * @return A Range object
         */
        portableWrapper::Range getLocalRangeZeroBase(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalRangeZeroBase();
        }

        /**
         * @brief This gets the local Range for the given axis(i.e. for the current MPI rank) assuming zero-based indexing
         * @details
         * This gives you the upper and lower ghost cells, and all of the real data in the local domain
         * Both Range are inclusive. The first ghost cell is at zero and all cells count up from there.
         * This version uses a specified staggering type.
         * @param name The name of the axis
         * @param stagger The staggering type
         * @return A Range object
         */
        portableWrapper::Range getLocalRangeZeroBase(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalRangeZeroBase(stagger);
        }

        /**
         * @brief This gets the domain Range for the given axis, that is the Range of this processors
         * array in the global domain. This includes the real data only, no ghost cells.
         */
        portableWrapper::Range getDomainRange(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getDomainRange();
        }

        /**
         * @brief This gets the domain Range for the given axis, that is the Range of this processors
         * array in the global domain. This includes the real data only, no ghost cells.
         * This version uses a specified staggering type.
         */
        portableWrapper::Range getDomainRange(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getDomainRange(stagger);
        }

        /**Set the domain (i.e. number of cells and the range of values) for the given axis
         * @param name The name of the axis
         * @param elements The number of elements in the axis
         * @param minVal The minimum value of the axis
         * @param maxVal The maximum value of the axis
         */
        void setDomain(const std::string &name, COUNT_TYPE elements, T_dataType minVal, T_dataType maxVal)
        {
            auto &ax = getAxis(name);
            setDomainElements(name, elements);
            T_dataType delta = (maxVal - minVal) / static_cast<T_dataType>(ax.dim.getDomainElements(staggerType::HALF_CELL) - 1);
            //Always create the HALF_CELL axis on the host
            ax.createLinearAxis(staggerType::HALF_CELL, memorySpace::HOST, delta, minVal);
            ax.axisMinVal = minVal-0.5*delta;
            ax.axisMaxVal = maxVal+0.5*delta;
            ax.createCentredAxis(memorySpace::HOST);
        }

        /*template<typename deltaFn>
        void setDomainFromDeltaFunction(const std::string &name, COUNT_TYPE elements, deltaFn fn, T_dataType initialValue)
        {
            auto &ax = getAxis(name);
            if (it == axisMap.end())
        }*/

        /**
         * Get the MPI axis index for the given axis
         * @param name The name of the axis
         * @return The MPI axis index
         */
        MPIAxis getMPIAxis(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.MPIAxisIndex;
        }

        /**
         * Set the MPI axis index for the given axis
         * @param name The name of the axis
         * @param mpiAxis The MPI axis index
         */
        void setMPIAxis(const std::string &name, MPIAxis mpiAxis)
        {
            auto &ax = getAxis(name);
            ax.MPIAxisIndex = mpiAxis;
        }

        /**
         * Set the maximum number of lower ghost cells for the given axis
         * NOTE: This only sets the maximum, it does not reduce it if the value is lower
         * @param name The name of the axis
         * @param ghosts The number of ghost cells
         */
        void setMaxLowerGhosts(const std::string &name, COUNT_TYPE ghosts)
        {
            auto &ax = getAxis(name);
            ax.dim.lowerGhosts = std::max(ax.dim.lowerGhosts, ghosts);
        }

        /** 
         * Set the maximum number of upper ghost cells for the given axis
         * NOTE: This only sets the maximum, it does not reduce it if the value is lower
         * @param name The name of the axis
         * @param ghosts The number of ghost cells
         */
        void setMaxUpperGhosts(const std::string &name, COUNT_TYPE ghosts)
        {
            auto &ax = getAxis(name);
            ax.dim.upperGhosts = std::max(ax.dim.upperGhosts, ghosts);
        }

        /**
         * Convenience function to set both maximum lower and upper ghost cells for the given axis
         * NOTE: This only sets the maximum, it does not reduce it if the value is lower
         * @param name The name of the axis
         * @param lowerGhosts The number of lower ghost cells
         * @param upperGhosts The number of upper ghost cells
         */
        void setMaxGhosts(const std::string &name, COUNT_TYPE lowerGhosts, COUNT_TYPE upperGhosts)
        {
            setMaxLowerGhosts(name, lowerGhosts);
            setMaxUpperGhosts(name, upperGhosts);
        }

        /**
         * Get the axis map
         */
        const std::unordered_map<std::string, axisInfo> &getAxisMap() const
        {
            return axisMap;
        }

        /**
         * Get the dimension info for a given axis (const version)
         * @param name The name of the axis
         * @return The dimension info
         */
        const dimension& getDimension(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim;
        }

        /**
         * Get the dimension info for a given axis
         * @param name The name of the axis
         * @return The dimension info
         */
        dimension& getDimension(const std::string &name)
        {
            auto &ax = getAxis(name);
            return ax.dim;
        }

        /**
         * Get the local range for a given axis
         * @param name The name of the axis
         * @param stagger The staggering type
         * @return The local range
         */
        portableWrapper::Range getLocalRange(const std::string &name, staggerType stagger)
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalRange(stagger);
        }

        /**
         * Get the local domain range for a given axis
         * @param name The name of the axis
         * @param stagger The staggering type
         * @return The local domain range
         */
        portableWrapper::Range getLocalDomainRange(const std::string &name, staggerType stagger)
        {
            auto &ax = getAxis(name);
            return ax.dim.getDomainRange(stagger);
        }

        /**
         * Get a portable array wrapping the axis data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */
        auto getPPAxis(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            auto &data = getAxis(axisname);
            return data.getPPAxis(stagger, memSpace);
        }

        /**
         * Get a portable array wrapping the axis data for the local domain only
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */
        auto getPPLocalAxis(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            //Get the full global axis data
            auto &data = getAxis(axisname);
            auto globalAxis = data.getPPAxis(stagger, memSpace);
            //Now get the part of the global axis that corresponds to the local domain
            auto globalRange = data.dim.getGlobalRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getLocalRange(stagger);
            //Create an array for the local domain as the correct slice of the global axis
            portableWrapper::portableArray<T_dataType, 1> localAxis = globalAxis(globalRange);
            //Rebase it to the local range
            localAxis.rebase(localRange);
            return localAxis;
        }

        /**
         * Get a portable array wrapping just the axis domain data on this local rank (i.e. excluding ghost cells)
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */
        auto getPPLocalDomainAxis(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            //Get the full global axis data
            auto &data = getAxis(axisname);
            auto globalAxis = data.getPPAxis(stagger, memSpace);
            //Now get the part of the global axis that corresponds to the local domain
            auto domainRange = data.dim.getGlobalDomainRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getDomainRange(stagger);
            //Create an array for the local domain as the correct slice of the global axis
            portableWrapper::portableArray<T_dataType, 1> localAxis = globalAxis(domainRange);
            //Rebase it to the local range
            localAxis.rebase(localRange);
            return localAxis;
        }

        /**
         * Get a portable array wrapping the axis delta data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */
        auto getPPDelta(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            auto &data = getAxis(axisname);
            return data.getPPDelta(stagger, memSpace);
        }

        /**
         * Get a portable array wrapping the axis delta data for the local domain only
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */
        auto getPPLocalDelta(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            //Get the full global axis delta data
            auto &data = getAxis(axisname);
            auto globalDelta = data.getPPDelta(stagger, memSpace);
            //Now get the part of the global delta that corresponds to the local domain
            auto globalRange = data.dim.getGlobalRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getLocalRange(stagger);
            //Create an array for the local domain as the correct slice of the global delta
            portableWrapper::portableArray<T_dataType, 1> localDelta = globalDelta(globalRange);
            //Rebase it to the local range
            localDelta.rebase(localRange);
            return localDelta;
        }

        /**
         * Get a raw pointer to the axis data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */
        T_dataType* getRawAxis(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            auto &data = getAxis(axisname);
            auto axisArray = data.getPPAxis(stagger, memSpace);
            return axisArray.data();
        }

        /**
         * Get a raw pointer to the axis delta data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */
        T_dataType* getRawDelta(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            auto &data = getAxis(axisname);
            auto deltaArray = data.getPPDelta(stagger, memSpace);
            return deltaArray.data();
        }

        /**
         * Get a raw pointer to the local axis data on this MPI rank
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param memSpace The memory space (default DEFAULT)
         */

        T_dataType* getRawLocalAxis(const std::string axisname, staggerType stagger, memorySpace memSpace = memorySpace::DEFAULT)
        {
            auto localAxis = getPPLocalAxis(axisname, stagger, memSpace);
            return localAxis.data();
        }

        void finalize()
        {
            axisMap.clear();
            manager.clear();
        }
    };

    inline axisRegistry &getaxisRegistry()
    {
        static axisRegistry instance;
        return instance;
    }

};

#endif // SAMS_AXISREGISTRY_H
