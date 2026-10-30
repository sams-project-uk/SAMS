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
#include <algorithm>
#include <any>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>
#include <variant>

#include "handles.h"
#include "memoryRegistry.h"
#include "staggerRegistry.h"
#include "pp/parallelWrapper.h"
#include "dimension.h"
#include "harnessDef.h"
#include "mpiManager.h"
#include "utils.h"
#include "pp/callableTraits.h"
#include "pp/demangle.h"
#include "typeRegistry.h"

//Managed

namespace SAMS
{

    namespace detail{
		/**
		 * Functor to return elements from a source array
		 */
		template <typename T>
		struct assignStaggeredValues
		{
            T* values;
            staggerType stagger;

			FUNCTORMETHODPREFIX INLINE assignStaggeredValues(T* vals, staggerType stag)
                : stagger(stag), values(vals)
            { 
            }

			FUNCTORMETHODPREFIX INLINE T operator()(UNSIGNED_INDEX_TYPE i) const
			{
				return values[static_cast<COUNT_TYPE>(i-(stagger==staggerType::CENTRED ? 1 : 0))];
			}
		};

        //Construct assignStaggeredValues functor based on array tags
        template<portableWrapper::arrayTags destTag, typename T, portableWrapper::arrayTags tag>
        auto make_assignStaggeredValues(portableWrapper::portableArray<T, 1, tag> &array, staggerType stag, portableWrapper::portableArrayManager &manager){
            if constexpr (destTag == tag){
                return detail::assignStaggeredValues<T>(array.data(), stag);
            } else if constexpr (destTag == portableWrapper::arrayTags::host){
                //Copy to host and return
                portableWrapper::hostArray<T, 1> hostArray;
                hostArray = manager.makeHostAvailable(array);
                return detail::assignStaggeredValues<T>(hostArray.data(), stag);
            } else {
                //Copy to device and return
                portableWrapper::acceleratedArray<T, 1> deviceArray;
                deviceArray = manager.makeDeviceAvailable(array);
                return detail::assignStaggeredValues<T>(deviceArray.data(), stag);
            }
        }

        template<portableWrapper::arrayTags destTag, portableWrapper::arrayTags srcTag, typename T>
        auto make_assignStaggeredValues(T* data, COUNT_TYPE elements, staggerType stag, portableWrapper::portableArrayManager &manager){
            if constexpr (destTag == srcTag){
                return detail::assignStaggeredValues<T>(data, stag);
            } else if constexpr (destTag == portableWrapper::arrayTags::host){
                //Copy to host and return
                portableWrapper::acceleratedArray<T, 1> srcArray;
                portableWrapper::hostArray<T, 1> hostArray;
                srcArray.bind(data, elements);
                hostArray=manager.makeHostAvailable(srcArray);
                return detail::assignStaggeredValues<T>(hostArray.data(), stag);
            } else {
                //Copy to device and return
                portableWrapper::hostArray<T, 1> srcArray;
                portableWrapper::acceleratedArray<T, 1> deviceArray;
                srcArray.bind(data, elements);
                deviceArray = manager.makeDeviceAvailable(srcArray);
                return detail::assignStaggeredValues<T>(deviceArray.data(), stag);
            }
        }
    }

    /**
     * The decomposition of an axis over the ranks along it
     */
    class axisDecomposition
    {
    public:
        /**
         * The position of a cell along the axis
         */
        struct location
        {
            /**
             * The rank along the axis that owns the cell
             */
            int axisRank;
            /**
             * The index of the cell in the block of that rank, counted from 1
             */
            SIGNED_INDEX_TYPE local;
        };

        /**
         * Set the first global index of the block of each rank along the axis, counted from 1, followed by one past the
         * last cell of the axis
         * @param starts The first global index of each block, then one past the last cell
         */
        void setBlockStarts(std::vector<SIGNED_INDEX_TYPE> starts)
        {
            if (starts.size() < 2 || starts.front() != 1)
            {
                throw std::runtime_error("Error: axis decomposition must have at least one block and start at 1\n");
            }
            for (std::size_t i = 1; i < starts.size(); ++i)
            {
                if (starts[i] < starts[i - 1])
                {
                    throw std::runtime_error("Error: axis decomposition blocks must not decrease\n");
                }
            }
            blockStarts = std::move(starts);
        }

        /**
         * Has the decomposition been set?
         */
        bool isSet() const { return !blockStarts.empty(); }

        /**
         * The number of ranks along the axis
         */
        int ranks() const { return static_cast<int>(blockStarts.size()) - 1; }

        /**
         * The number of cells in the whole axis
         */
        SIGNED_INDEX_TYPE globalCells() const { return blockStarts.back() - 1; }

        /**
         * The number of cells on the rank at a given position along the axis
         * @param axisRank The rank along the axis
         */
        SIGNED_INDEX_TYPE cellsOnRank(int axisRank) const { return blockStarts[axisRank + 1] - blockStarts[axisRank]; }

        /**
         * The global index, counted from 1, of the first cell on the rank at a given position along the axis
         * @param axisRank The rank along the axis
         */
        SIGNED_INDEX_TYPE firstIndexOnRank(int axisRank) const { return blockStarts[axisRank]; }

        /**
         * Find which rank along the axis owns a global index, and the index of the cell within the block of that rank
         * @param globalIndex The global index of the cell, counted from 1. It must be inside the axis
         */
        location locate(SIGNED_INDEX_TYPE globalIndex) const
        {
            // The block is the last one that starts at or before the index. The final entry does not start a block
            const auto it = std::upper_bound(blockStarts.begin(), blockStarts.end() - 1, globalIndex);
            const int axisRank = static_cast<int>(it - blockStarts.begin()) - 1;
            return {axisRank, globalIndex - blockStarts[axisRank] + 1};
        }

    private:
        std::vector<SIGNED_INDEX_TYPE> blockStarts;
    };

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
            using hostLineArray = portableWrapper::hostArray<T_dataType, 1>;

            portableWrapper::portableArrayManager &manager;
            bool logicalAxis = false; //Is this axis a logical axis (i.e. it does not correspond to physical data, but instead logical indices);
            /**
             * Simple class to tie together axis and delta arrays
             */
            struct axisValues
            {
                SAMS::typeHandle axisType;
                std::any axis;
                std::any delta;
            };

            /**
             * Create a position axis at core. Templates on the memory tag of the underlying PP array
             */

             template<typename dataFn, portableWrapper::arrayTags tag>
             void createPositionAxisCore(staggerType stagger, dataFn &fn){
                //Infer type of the axis from the callable
                using T_array = std::remove_const_t<typename far::callableTraits<dataFn>::type>;
                portableWrapper::arrayTags tagLocal = tag;
                //If axis already exists, return
                if (axisData.find(stagger) != axisData.end() &&
                    axisData[stagger].find(tagLocal) != axisData[stagger].end())
                {
                    return;
                }

                auto applyFn =[](auto &&kernel, auto &&...ranges){
                    if constexpr(tag == portableWrapper::arrayTags::host){
                        portableWrapper::applyKernelHost(std::forward<decltype(kernel)>(kernel), std::forward<decltype(ranges)>(ranges)...);
                    } else {
                        portableWrapper::applyKernel(std::forward<decltype(kernel)>(kernel), std::forward<decltype(ranges)>(ranges)...);
                    }
                };

                auto fenceFn = [](){
                    if constexpr(tag == portableWrapper::arrayTags::host){
                        portableWrapper::fenceHost();
                    } else {
                        portableWrapper::fence();
                    }
                };

                auto &axisInfo = axisData[stagger][tagLocal];
                axisInfo.axis.emplace<portableWrapper::portableArray<T_array, 1, tag>>();
                axisInfo.axisType = SAMS::gettypeRegistry().getTypeID<T_array>();
                auto &lineAxis = std::any_cast<portableWrapper::portableArray<T_array, 1, tag>&>(axisInfo.axis);
                SIGNED_INDEX_TYPE lowerBound = dim.getLB(stagger);
                SIGNED_INDEX_TYPE upperBound = dim.getUB(stagger);

                // Now initialize the axis
                portableWrapper::Range range = dim.getRange(stagger);
                manager.allocate(lineAxis, range);
                auto axisArray = lineAxis; //Copy for lambda capture without capturing the whole class
                applyFn(LAMBDA(SIGNED_INDEX_TYPE i) { axisArray(i) = fn(i); }, range);
                fenceFn();

                if constexpr(SAMS::has_subtraction_v<T_array, T_array>){
                    static_assert(std::is_trivially_copyable_v<SAMS::has_subtraction_t<T_array, T_array>>, "Error: axis types must be trivially copyable\n");
                    using T_delta = SAMS::has_subtraction_t<T_array, T_array>;
                    axisInfo.delta.emplace<portableWrapper::portableArray<T_delta, 1, tag>>();
                    //Move the upper bound back by one to compute delta (i.e. delta is the distance between edges,
                    //so by fencepost rule there is one less delta than edges)
                    auto &lineDelta = std::any_cast<portableWrapper::portableArray<T_delta, 1, tag>&>(axisInfo.delta);
                    manager.allocate(lineDelta, range);
                    auto deltaArray = lineDelta; //Copy for lambda capture without capturing the whole class
                    //Can only calculate a delta for i>lowerBound
                    range.lower_bound+=1;
                    applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {deltaArray(i) = axisArray(i)-axisArray(i-1);}, range);
                    fenceFn();
                    //Now apply a boundary condition to get the first delta
                    if (dim.periodic){
                        applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {
                                //Copy round the delta from the upper bound
                                deltaArray(i) = deltaArray(upperBound);
                        }, portableWrapper::Range(lowerBound, lowerBound));
                    } else {
                        applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {
                                //Assume uniform spacing for first delta
                                deltaArray(i) = deltaArray(i+1);
                        }, portableWrapper::Range(lowerBound, lowerBound));
                    }
                    fenceFn();
                } else {
                    //Cannot compute delta for non-arithmetic types
                    std::cout <<" Warning: axisRegistry cannot compute delta axis for non-arithmetic axis type\n";
                    axisInfo.delta = std::any();
                }
             }

            /**
             * Create an axis based on a provided function that specifies the location of each axis point
             * @param stagger The staggering type of the axis
             * @param tag The memory space tag to allocate the axis in
             * @param fn The function to compute the axis values
             */
            template <typename dataFn>
            void createPositionAxis(staggerType stagger, portableWrapper::arrayTags tag, dataFn &fn)
            {
                /*using T_dataType = typename far::callableTraits<dataFn>::type;
                static_assert(std::is_trivially_copyable_v<T_dataType>, "Error: axisRegistry createPositionAxis function must return a trivially copyable type\n");*/
                if (tag == portableWrapper::arrayTags::host) {
                    createPositionAxisCore<dataFn, portableWrapper::arrayTags::host>(stagger, fn);
                } else if (tag == portableWrapper::arrayTags::accelerated){
                    createPositionAxisCore<dataFn, portableWrapper::arrayTags::accelerated>(stagger, fn);
                } else {
                    throw std::runtime_error("Error: axisRegistry createPositionAxis unknown memory space\n");
                }
            }


            template<typename deltaFn, typename T_initial, portableWrapper::arrayTags tag, auto applyFn, auto fenceFn>
            void createDeltaAxisCore(staggerType stagger, deltaFn fn, T_initial initialValue){
                //Check what the delta function returns
                using T_delta = typename far::callableTraits<deltaFn>::type;
                //Must be trivially copyable for portable arrays
                static_assert(std::is_trivially_copyable_v<T_delta>, "Error: axisRegistry createDeltaAxis function must return a trivially copyable type\n");
                //Can I add the results to the initial value type?
                static_assert(has_addition_v<T_initial, T_delta>, "Error: axisRegistry createDeltaAxis function return type and initial value type must support addition\n");
                //Can I add the results to themselves?
                static_assert(has_addition_v<T_delta, T_delta>, "Error: axisRegistry createDeltaAxis function return type must support addition\n");
                //Must also be able to add the result of adding delta to initial value to the result of adding delta to itself
                static_assert(has_addition_v<has_addition_t<T_initial, T_delta>, has_addition_t<T_delta, T_delta>>, "Error: unable to add axisRegistry createDeltaAxis function return type and initial value type\n");
                //Assume that the result of adding delta to itself is the type of the axis
                using T_array = has_addition_t<T_delta, T_delta>;
                portableWrapper::arrayTags tagLocal = tag;
                //If axis already exists, return
                if (axisData.find(stagger) != axisData.end() &&
                    axisData[stagger].find(tagLocal) != axisData[stagger].end())
                {
                    return;
                }

                auto &axisInfo = axisData[stagger][tag];
                SIGNED_INDEX_TYPE lowerBound = dim.getLB(stagger);
                SIGNED_INDEX_TYPE upperBound = dim.getUB(stagger);
                SIGNED_INDEX_TYPE domainLower = dim.getDomainLB(stagger);

                // Now initialize the delta axis
                portableWrapper::Range range(lowerBound, upperBound);
                //Fill the std::any with the correct type
                axisInfo.delta.emplace<portableWrapper::portableArray<T_delta, 1, tag>>();
                //Get back the actual array reference
                portableWrapper::portableArray<T_array, 1, tag> &axisDelta = std::any_cast<portableWrapper::portableArray<T_array, 1, tag>&>(axisInfo.delta);
                //Allocate it
                manager.allocateManaged(axisDelta, range);


                range.lower_bound+=1; //Delta only defined for i>lowerBound
                //Use function to build the delta axis
                applyFn(LAMBDA(SIGNED_INDEX_TYPE i) { axisDelta(i) = fn(i); }, range);
                fenceFn();
                range.lower_bound-=1;
                //Now apply a boundary condition to get the first delta
                //Has to be done like this to be consistent with LARE expectations
                if (dim.periodic){
                    applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {
                        if (i == lowerBound) {
                            //Copy round the delta from the upper bound
                            axisDelta(i) = axisDelta(upperBound);
                        }}, portableWrapper::Range(lowerBound-1, lowerBound-1));
                } else {
                    applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {
                        if (i == lowerBound) {
                            //Assume uniform spacing for first delta
                            axisDelta(i) = axisDelta(i+1);
                        }}, portableWrapper::Range(lowerBound-1, lowerBound-1));
                }
                fenceFn();
                //Now calculate the position axis by integrating the delta
                axisInfo.axis.emplace<portableWrapper::portableArray<T_array, 1, tag>>();
                auto & axisArray = std::any_cast<portableWrapper::portableArray<T_array, 1, tag>&>(axisInfo.axis); 
                manager.allocateManaged(axisArray, range);
                portableWrapper::fence();
                //Set lower boundary value
                applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {axisArray(i) = initialValue;}, portableWrapper::Range(domainLower, domainLower));
                fenceFn();
                //Integrate upwards
                applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {
                    for (SIGNED_INDEX_TYPE j = lowerBound; j <= i; ++j)
                    {
                        axisArray(i) += axisDelta(j);
                    }
                }, portableWrapper::Range(domainLower+1, upperBound));
                //Integrate downwards
                applyFn(LAMBDA(SIGNED_INDEX_TYPE i) {
                    for (SIGNED_INDEX_TYPE j = i + 1; j <= upperBound; ++j)
                    {
                        axisArray(i) -= axisDelta(j);
                    }
                }, portableWrapper::Range(lowerBound, domainLower-1));
                fenceFn();
            }

            /**
             * Create a delta axis based on a provided function that specifies the delta between each axis point and a lower boundary value
             * @param stagger The staggering type of the axis
             * @param tag The memory space tag to allocate the axis in
             * @param fn The function to compute the delta values
             * @param initialValue The initial value at the lower boundary of the axis
             */
            /*template<typename deltaFn>
            void createDeltaAxis(staggerType stagger, portableWrapper::arrayTags tag, deltaFn fn, T_dataType initialValue)
            {
                //If axis already exists, return
                if (axisData.find(stagger) != axisData.end() &&
                    axisData[stagger].find(tag) != axisData[stagger].end())
                {
                    return;
                }
                //If not, create it
                auto axisInfo = axisData[stagger][tag];
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
            }*/

            /**
             * Convenience function to create a linear axis from a given lower bound and uniform grid spacing
             * @param stagger The staggering type of the axis
             * @param tag The memory space tag to allocate the axis in
             * @param delta The uniform grid spacing
             * @param lowerBound The lower bound of the axis (default 0.0)
             */
            template<typename T_range>
            void createLinearAxis(staggerType stagger, portableWrapper::arrayTags tag, T_range delta, T_range lowerBound=0.0)
            {
                auto linearFn = LAMBDA(SIGNED_INDEX_TYPE i)->T_range
                {
                    return lowerBound + delta * static_cast<T_range>(i);
                };
                createPositionAxis(stagger, tag, linearFn);
            }

            /**
             * Convert an edge-centred (half-cell) axis to a centred (cell-centred) axis
             * @param tag The memory space tag to allocate the axis in
             * @tparam T_array The data type of the array (default T_dataType) - Must match the type used to create the half-cell axis
             */
            template<typename T_array=T_dataType>
            void createCentredAxis(portableWrapper::arrayTags tag)
            {
                if constexpr(has_addition_v<T_array, T_array>) {
                    using T_add_result = has_addition_t<T_array, T_array>;
                    if constexpr(has_multiplication_v<double, T_add_result>){
                        using T_final_result = has_multiplication_t<double, T_add_result>;
                            if (tag == portableWrapper::arrayTags::host) {
                                auto halfCellAxis = std::any_cast<portableWrapper::portableArray<T_array, 1, portableWrapper::arrayTags::host>&>(axisData[staggerType::HALF_CELL][portableWrapper::arrayTags::host].axis);
                                auto linearFn = LAMBDA(SIGNED_INDEX_TYPE i)->T_final_result
                                {
                                    return 0.5 * (halfCellAxis(i) + halfCellAxis(i-1));
                                };
                                createPositionAxis(staggerType::CENTRED, tag, linearFn);
                            } else {
                                auto halfCellAxis = std::any_cast<portableWrapper::portableArray<T_array, 1, portableWrapper::arrayTags::accelerated>&>(axisData[staggerType::HALF_CELL][portableWrapper::arrayTags::accelerated].axis);
                                auto linearFn = LAMBDA(SIGNED_INDEX_TYPE i)->T_final_result
                                {
                                    return 0.5 * (halfCellAxis(i) + halfCellAxis(i-1));
                                };
                                createPositionAxis(staggerType::CENTRED, tag, linearFn);
                        }
                    }
                }
            }


            /**
             * This function generates the axis in the requested memory space if it does not already exist
             * It assumes that the host versions of the axis and delta have already been created
             */
            template<typename T_array=T_dataType, portableWrapper::arrayTags tag>
            void generateAxis(staggerType stagger)
            {
                if (tag == portableWrapper::arrayTags::host){
                    throw std::runtime_error("Error: HOST axis should have been created at domain setup\n");
                } else if (tag == portableWrapper::arrayTags::accelerated){
                    using deviceArray = portableWrapper::acceleratedArray<T_array, 1>;
                    using hostArray = portableWrapper::hostArray<T_array, 1>;
                    axisData[stagger][tag].axis.emplace<deviceArray>();
                    std::any_cast<deviceArray&>(axisData[stagger][tag].axis) = manager.makeDeviceAvailable(std::any_cast<hostArray&>(axisData[stagger][portableWrapper::arrayTags::host].axis));

                    //If the delta axis exists on host, create it on device too
                    if (axisData[stagger][portableWrapper::arrayTags::host].delta.has_value() == true) {
                        axisData[stagger][tag].delta.emplace<deviceArray>();
                        std::any_cast<deviceArray&>(axisData[stagger][tag].delta) = manager.makeDeviceAvailable(std::any_cast<hostArray&>(axisData[stagger][portableWrapper::arrayTags::host].delta));
                    }
                } else {
                    throw std::runtime_error("Error: Unsupported memory space for axis generation\n");
                }
            }

        public:
            dimension dim{SAMS::staggerType::HALF_CELL};
            axisDecomposition decomposition;
            /** For a derived axis, the name of the axis that it is derived from. It is empty for any other axis */
            std::string sourceAxis;
            SIGNED_INDEX_TYPE localLB = 0;
            SIGNED_INDEX_TYPE localUB = 0;
            COUNT_TYPE maxLowerGhosts = 0;
            COUNT_TYPE maxUpperGhosts = 0;
            bool periodic = false;

            axisInfo() = delete; // Delete default constructor
            axisInfo(portableWrapper::portableArrayManager &mgr)
                : manager(mgr) {}
            axisInfo(portableWrapper::portableArrayManager &mgr, MPIAxis mpiAxis)
                : manager(mgr) {dim.attachMPIAxis(mpiAxis);}
            axisInfo(portableWrapper::portableArrayManager &mgr, bool isLogical)
                : manager(mgr), logicalAxis(isLogical) {}
            axisInfo(portableWrapper::portableArrayManager &mgr, MPIAxis mpiAxis, bool isLogical)
                : manager(mgr), logicalAxis(isLogical) {dim.attachMPIAxis(mpiAxis);}
            axisInfo(const axisInfo &) = delete;
            axisInfo &operator=(const axisInfo &) = delete;
            axisInfo(axisInfo &&) = default;
            axisInfo &operator=(axisInfo &&) = delete;
            std::unordered_map<staggerType, std::unordered_map<portableWrapper::arrayTags, axisValues>> axisData;

            /**
             * Does a given axis exist in the registry?
             * @param stagger The staggering type of the axis
             * @param tag The memory tag of the axis
             * @return True if the axis exists, false otherwise
             */
            bool hasAxis(staggerType stagger, portableWrapper::arrayTags tag)
            {
                return (axisData.find(stagger) != axisData.end() &&
                        axisData[stagger].find(tag) != axisData[stagger].end());
            }

            /**
             * Return a performance portable array representing the axis
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array (default T_dataType) - Must match the type used to create the axis
             * @tparam tag The memory tag of the array (default accelerated)
             * @return The performance portable array representing the axis
             */
            template<typename T_array=T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
            portableWrapper::portableArray<T_array,1,tag> getPPAxis(staggerType stagger)
            {
                if (!hasAxis(stagger, tag))
                {
                    generateAxis<T_array,tag>(stagger);
                }
                return std::any_cast<portableWrapper::portableArray<T_array,1,tag>&>(axisData[stagger][tag].axis);
            }

            /**
             * Fill a performance portable array with an axis
             * @param array The performance portable array to fill
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array (default T_dataType) - Must match the type used to create the axis
             */
            template<typename T_array, portableWrapper::arrayTags tag>
            void fillPPAxis(portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
            {
                if (!hasAxis(stagger, tag))
                {
                    generateAxis<T_array,tag>(stagger);
                }
                if (!axisData[stagger][tag].axis.has_value()) {
                    throw std::runtime_error("Error: axis data not available in fillPPAxis");
                }
                try {
                    array = std::any_cast<portableWrapper::portableArray<T_array, 1, tag>&>(axisData[stagger][tag].axis);
                } catch (const std::bad_any_cast &e) {
                    throw std::runtime_error("Error: incorrect type used in fillPPAxis");
                }
            }

            #ifdef USE_KOKKOS
            /*template<typename T_array, portableWrapper::arrayTags tag>
            auto getKokkosView(staggerType stagger){
                return portableWrapper::kokkos::toView(getPPAxis<T_array, tag>(stagger));
            }*/
            #endif


            /**
             * Return a performance portable array representing the delta axis
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array (default T_dataType) - Must match the type used to create the axis
             * @tparam tag The memory tag of the array (default accelerated)
             * @return The performance portable array representing the delta axis
             */
            template<typename T_array=T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
            portableWrapper::portableArray<T_array,1,tag> getPPDelta(staggerType stagger)
            {
                if (!hasAxis(stagger, tag))
                {
                    generateAxis<T_array,tag>(stagger);
                }
                return std::any_cast<portableWrapper::portableArray<T_array,1,tag>&>(axisData[stagger][tag].delta);
            }

            /**
             * Fill a specified array with the delta axis
             * @param array The performance portable array to fill
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array - Must match the type used to create the axis
             * @tparam tag The memory tag of the array
             */
            template<typename T_array, portableWrapper::arrayTags tag>
            void fillPPDelta(portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
            {
                if (!hasAxis(stagger, tag))
                {
                    generateAxis<T_array,tag>(stagger);
                }
                if (!axisData[stagger][tag].delta.has_value()) {
                    throw std::runtime_error("Error: delta data not available in fillPPDelta");
                }
                try {
                    array = std::any_cast<portableWrapper::portableArray<T_array, 1, tag>&>(axisData[stagger][tag].delta);
                } catch (const std::bad_any_cast &e) {
                    throw std::runtime_error("Error: incorrect type used in fillPPDelta");
                }
            }

            /**
             * Return a performance portable array representing the local portion of the axis
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array (default T_dataType) - Must match the type used to create the axis
             * @tparam tag The memory tag of the array (default accelerated)
             * @return The performance portable array representing the local portion of the axis
             */
            template<typename T_array = T_dataType, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
            portableWrapper::portableArray<T_array, 1, tag> getPPLocalAxis(staggerType stagger)
            {
                auto globalAxis = getPPAxis<T_array, tag>(stagger);
                //Now get the part of the global axis that corresponds to the local domain
                auto globalRange = dim.getGlobalRange(stagger);
                //And the range of the local version of that
                auto localRange = dim.getLocalRange(stagger);
                //Create an array for the local domain as the correct slice of the global axis
                portableWrapper::portableArray<T_array, 1, tag> localAxis = globalAxis(globalRange);
                //Rebase it to the local range
                localAxis.rebase(localRange);
                return localAxis;
            }

            /**
             * Fill a specified array with the local portion of the axis
             * @param array The performance portable array to fill
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array - Must match the type used to create the axis
             */
            template<typename T_array, portableWrapper::arrayTags tag>
            void fillPPLocalAxis(portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
            {
                array = getPPLocalAxis<T_array, tag>(stagger);
            }

            /**
             * Return a performance portable array representing the local portion of the delta axis
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array (default T_dataType) - Must match the type used to create the axis
             * @tparam tag The memory tag of the array (default accelerated)
             * @return The performance portable array representing the local portion of the delta axis
             */
            template<typename T_array = T_dataType, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
            portableWrapper::portableArray<T_array, 1, tag> getPPLocalDelta(staggerType stagger)
            {
                auto globalDelta = getPPDelta<T_array, tag>(stagger);
                //Now get the part of the global delta that corresponds to the local domain
                auto globalRange = dim.getGlobalRange(stagger);
                //And the range of the local version of that
                auto localRange = dim.getLocalRange(stagger);
                //Create an array for the local domain as the correct slice of the global delta
                portableWrapper::portableArray<T_array, 1, tag> localDelta = globalDelta(globalRange);
                //Rebase it to the local range
                localDelta.rebase(localRange);
                return localDelta;
            }

            /**
             * Fill a specified array with the local portion of the delta axis
             * @param array The performance portable array to fill
             * @param stagger The staggering type of the axis
             * @tparam T_array The data type of the array - Must match the type used to create the axis
             * @tparam tag The memory tag of the array 
             */
            template<typename T_array, portableWrapper::arrayTags tag>
            void fillPPLocalDelta(portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
            {
                array = getPPLocalDelta<T_array, tag>(stagger);
            }
        }; // struct axisInfo

        portableWrapper::portableArrayManager &manager;
        std::unordered_map<std::string, axisInfo> axisMap;
        /** The MPI layer, which gives the topology that the axes are decomposed over */
        MPIManager<MPI_DECOMPOSITION_RANK> &mpiMgr;

    public:

        axisRegistry(SAMS::memoryRegistry &memRegistry, MPIManager<MPI_DECOMPOSITION_RANK> &mpiManager)
            : manager(memRegistry.getArrayManager()), mpiMgr(mpiManager)
        {
        }

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
         * Register a new logical axis with the given name
         * @param name The name of the axis to register
         */
        void registerLogicalAxis(const std::string &name)
        {
            axisMap.try_emplace(name, manager, true);
        }

        /**
         * Register a new logical axis with a given name and associate it with a given MPI axis
         * @param name The name of the axis to register
         * @param mpiAxis The MPI axis index to associate with the axis
         */
        void registerLogicalAxis(const std::string &name, MPIAxis mpiAxis)
        {
            axisMap.try_emplace(name, manager, mpiAxis, true);
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

        void setPeriodic(const std::string &name,  bool isPeriodic)
        {
            auto &ax = getAxis(name);
            ax.dim.periodic = isPeriodic;
            ax.periodic = isPeriodic;
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
        portableWrapper::Range getLocalDomainRange(const std::string &name) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalDomainRange();
        }

        /**
         * @brief This gets the domain Range for the given axis, that is the Range of this processors
         * array in the global domain. This includes the real data only, no ghost cells.
         * This version uses a specified staggering type.
         */
        /*portableWrapper::Range getDomainRange(const std::string &name, staggerType stagger) const
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalDomainRange(stagger);
        }*/

        /**Set the domain (i.e. number of cells and the range of values) for the given axis
         * @param name The name of the axis
         * @param elements The number of elements in the axis
         * @param minVal The minimum value of the axis
         * @param maxVal The maximum value of the axis
         */
        template<typename T_range>
        void setDomain(const std::string &name, COUNT_TYPE elements, T_range minVal, T_range maxVal)
        {
            //It isn't meaningful to call this on a logical axis
            if (getAxis(name).logicalAxis) {
                throw std::runtime_error("Error: Cannot set physical domain on logical axis " + name + "\n");
            }
            auto &ax = getAxis(name);
            setDomainElements(name, elements);
            T_range delta = (maxVal - minVal) / static_cast<T_range>(ax.dim.getDomainElements(staggerType::HALF_CELL) - 1);
            ax.createLinearAxis(staggerType::HALF_CELL, portableWrapper::arrayTags::host, delta, minVal);
            ax.createCentredAxis<T_range>(portableWrapper::arrayTags::host);
        }

        /**
         * Set the domain by specifying the values directly. This is intended for logical axes
         */
        template<typename T>
        void setDomainValues(const std::string &name, const std::vector<T> &values, staggerType stagger = staggerType::CENTRED)
        {
            auto &ax = getAxis(name);
            COUNT_TYPE elements = static_cast<COUNT_TYPE>(values.size());
            setDomainElements(name, elements);
            //Use a local manager because we want any temporaries cleaned up straight away
            portableWrapper::portableArrayManager mgr;
            auto asv = detail::make_assignStaggeredValues<portableWrapper::arrayTags::host, portableWrapper::arrayTags::host>(values.data(), elements, stagger, mgr);

            ax.createPositionAxis(stagger, portableWrapper::arrayTags::host, asv);
            //If the axis is half-cell staggered and capable of being averaged to cell centres, do so
            if (stagger == staggerType::HALF_CELL) {
                ax.createCentredAxis(portableWrapper::arrayTags::host);
            }
        }

        template<typename T, COUNT_TYPE N>
        void setDomainValues(const std::string &name, const std::array<T, N> &values, staggerType stagger = staggerType::CENTRED)
        {
            auto &ax = getAxis(name);
            COUNT_TYPE elements = static_cast<COUNT_TYPE>(values.size());
            setDomainElements(name, elements);
            ax.createPositionAxis(stagger, portableWrapper::arrayTags::host,
                [values,stagger](SIGNED_INDEX_TYPE i) -> T {
                    return values[static_cast<COUNT_TYPE>(i-(stagger==staggerType::CENTRED ? 1 : 0))];
                });
            if (stagger == staggerType::HALF_CELL) {
                ax.createCentredAxis(portableWrapper::arrayTags::host);
            }
        }

        template<far::callable T_func>
        void setDomainValues(const std::string &name, COUNT_TYPE elements, T_func fn, staggerType stagger = staggerType::CENTRED)
        {
            auto &ax = getAxis(name);
            setDomainElements(name, elements);
            ax.createPositionAxis(stagger, portableWrapper::arrayTags::host, fn);
            //If the axis is half-cell staggered and capable of being averaged to cell centres, do so
            if (stagger == staggerType::HALF_CELL) {
                ax.createCentredAxis(portableWrapper::arrayTags::host);
            }
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
            return ax.dim.mpiAxis;
        }

        /**
         * Set the MPI axis index for the given axis
         * @param name The name of the axis
         * @param mpiAxis The MPI axis index
         */
        void setMPIAxis(const std::string &name, MPIAxis mpiAxis)
        {
            auto &ax = getAxis(name);
            ax.dim.attachMPIAxis(mpiAxis);
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
            return ax.dim.getLocalDomainRange(stagger);
        }


        /**
         * Get the local non domain(i.e. ghost cell) range for a given axis
         * @param name The name of the axis
         * @param stagger The staggering type
         * @return The local non domain range
         */
        portableWrapper::Range getLocalNonDomainRange(const std::string &name, staggerType stagger, SAMS::domain::edges edge)
        {
            auto &ax = getAxis(name);
            return ax.dim.getLocalNonDomainRange(stagger, edge);
        }


        /**
         * Get a portable array wrapping the axis data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
        auto getPPAxis(const std::string axisname, staggerType stagger)
        {
            auto &data = getAxis(axisname);
            return data.getPPAxis<T_array, tag>(stagger);
        }

        /**
         * Fill an existing array with the axis data
         * @param axisname The name of the axis
         * @param array The array to fill
         * @param stagger The staggering type
         */
        template<typename T_array, portableWrapper::arrayTags tag>
        void fillPPAxis(const std::string axisname, portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
        {
            auto &data = getAxis(axisname);
            if (data.logicalAxis && stagger == staggerType::HALF_CELL){
                throw std::runtime_error("Error: Logical axis " + axisname + " cannot have HALF_CELL staggering");
            }
            data.fillPPAxis(array, stagger);
        }

#ifdef USE_KOKKOS
        /**
         * Get a Kokkos View wrapping the axis data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        /*template<typename T_array = T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
        auto getKokkosView(const std::string axisname, staggerType stagger)
        {
            auto &data = getAxis(axisname);
            return data.getKokkosView<T_array, tag>(stagger);
        }*/
#endif

        /**
         * Get a portable array wrapping the axis data for the local domain only
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
        auto getPPLocalAxis(const std::string axisname, staggerType stagger)
        {
            //Get the full global axis data
            auto &data = getAxis(axisname);
            auto globalAxis = data.getPPAxis<T_array, tag>(stagger);
            //Now get the part of the global axis that corresponds to the local domain
            auto globalRange = data.dim.getGlobalRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getLocalRange(stagger);
            //Create an array for the local domain as the correct slice of the global axis
            portableWrapper::portableArray<T_array, 1, tag> localAxis = globalAxis(globalRange);
            //Rebase it to the local range
            localAxis.rebase(localRange);
            return localAxis;
        }

        /**
         * Fill an existing array with the axis data for the local domain only
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param array The array to fill
         * @param stagger The staggering type
         */
        template<typename T_array, portableWrapper::arrayTags tag>
        void fillPPLocalAxis(const std::string axisname, portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
        {
            //Get the full global axis data
            auto &data = getAxis(axisname);
            portableWrapper::portableArray<T_dataType, 1, tag> globalAxis;
            data.fillPPAxis(globalAxis, stagger);
            //Now get the part of the global axis that corresponds to the local domain
            auto globalRange = data.dim.getGlobalRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getLocalRange(stagger);
            //Create an array for the local domain as the correct slice of the global axis
            array = globalAxis(globalRange);
            //Rebase it to the local range
            array.rebase(localRange);
        }

        /**
         * Get a portable array wrapping just the axis domain data on this local rank (i.e. excluding ghost cells)
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
        auto getPPLocalDomainAxis(const std::string axisname, staggerType stagger)
        {
            //Get the full global axis data
            auto &data = getAxis(axisname);
            auto globalAxis = data.getPPAxis<T_array, tag>(stagger);
            //Now get the part of the global axis that corresponds to the local domain
            auto domainRange = data.dim.getGlobalDomainRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getLocalDomainRange(stagger);
            //Create an array for the local domain as the correct slice of the global axis
            portableWrapper::portableArray<T_array, 1, tag> localAxis = globalAxis(domainRange);
            //Rebase it to the local range
            localAxis.rebase(localRange);
            return localAxis;
        }

        /**
         * Get a portable array wrapping the axis delta data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
        auto getPPDelta(const std::string axisname, staggerType stagger)
        {
            auto &data = getAxis(axisname);
            return data.getPPDelta<T_array, tag>(stagger);
        }

        /**
         * Fill an existing array with the axis delta data
         * @param axisname The name of the axis
         * @param array The array to fill
         * @param stagger The staggering type
         */
        template<typename T_array, portableWrapper::arrayTags tag>
        void fillPPDelta(const std::string axisname, portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
        {
            auto &data = getAxis(axisname);
            data.fillPPDelta(array, stagger);
        }

        /**
         * Get a portable array wrapping the axis delta data for the local domain only
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated>
        auto getPPLocalDelta(const std::string axisname, staggerType stagger)
        {
            //Get the full global axis delta data
            auto &data = getAxis(axisname);
            auto globalDelta = data.getPPDelta<T_array, tag>(stagger);
            //Now get the part of the global delta that corresponds to the local domain
            auto globalRange = data.dim.getGlobalRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getLocalRange(stagger);
            //Create an array for the local domain as the correct slice of the global delta
            portableWrapper::portableArray<T_array, 1, tag> localDelta = globalDelta(globalRange);
            //Rebase it to the local range
            localDelta.rebase(localRange);
            return localDelta;
        }

        /**
         * Fill an existing array with the axis delta data for the local domain only
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param array The array to fill
         * @param stagger The staggering type
         */
        template<typename T_array, portableWrapper::arrayTags tag>
        void fillPPLocalDelta(const std::string axisname, portableWrapper::portableArray<T_array, 1, tag> &array, staggerType stagger)
        {
            //Get the full global axis delta data
            auto &data = getAxis(axisname);
            portableWrapper::portableArray<T_array, 1, tag> globalDelta;
            data.fillPPDelta<T_array,tag>(globalDelta, stagger);
            //Now get the part of the global delta that corresponds to the local domain
            auto globalRange = data.dim.getGlobalRange(stagger);
            //And the range of the local version of that
            auto localRange = data.dim.getLocalRange(stagger);
            //Create an array for the local domain as the correct slice of the global delta
            array = globalDelta(globalRange);
            //Rebase it to the local range
            array.rebase(localRange);
        }

#ifdef USE_KOKKOS
        /**
         * Fill a Kokkos offset view (KOV) with the axis data. The element type is that of the view
         * @param axisname The name of the axis
         * @param view The offset view to fill
         * @param stagger The staggering type
         */
        template<typename T_view>
        void fillKOVAxis(const std::string axisname, T_view &view, staggerType stagger)
        {
            auto array = getPPAxis<typename T_view::non_const_value_type>(axisname, stagger);
            view = portableWrapper::kokkos::toOffsetView(array);
        }

        /**
         * Fill a Kokkos offset view (KOV) with the axis data for the local domain only, which includes ghost cells
         * @param axisname The name of the axis
         * @param view The offset view to fill
         * @param stagger The staggering type
         */
        template<typename T_view>
        void fillKOVLocalAxis(const std::string axisname, T_view &view, staggerType stagger)
        {
            auto array = getPPLocalAxis<typename T_view::non_const_value_type>(axisname, stagger);
            view = portableWrapper::kokkos::toOffsetView(array);
        }

        /**
         * Fill a Kokkos offset view (KOV) with the axis delta data for the local domain only, which includes ghost cells
         * @param axisname The name of the axis
         * @param view The offset view to fill
         * @param stagger The staggering type
         */
        template<typename T_view>
        void fillKOVLocalDelta(const std::string axisname, T_view &view, staggerType stagger)
        {
            auto array = getPPLocalDelta<typename T_view::non_const_value_type>(axisname, stagger);
            view = portableWrapper::kokkos::toOffsetView(array);
        }
#endif

        /**
         * Get a raw pointer to the axis data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType>
        T_array* getRawAxis(const std::string axisname, staggerType stagger, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated)
        {
            auto &data = getAxis(axisname);
            if (tag == portableWrapper::arrayTags::host){
                auto axisArray = data.getPPAxis<T_array, portableWrapper::arrayTags::host>(stagger);
                return axisArray.data();
            } else {
                auto axisArray = data.getPPAxis<T_array, portableWrapper::arrayTags::accelerated>(stagger);
                return axisArray.data();
            }
        }

        /**
         * Get a raw pointer to the axis delta data
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType>
        T_array* getRawDelta(const std::string axisname, staggerType stagger, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated)
        {
            auto &data = getAxis(axisname);
            if (tag == portableWrapper::arrayTags::host){
                auto deltaArray = data.getPPDelta<T_array, portableWrapper::arrayTags::host>(stagger);
                return deltaArray.data();
            } else {
                auto deltaArray = data.getPPDelta<T_array, portableWrapper::arrayTags::accelerated>(stagger);
                return deltaArray.data();
            }
        }

        /**
         * Get a raw pointer to the local axis data on this MPI rank
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param stagger The staggering type
         * @param tag The memory space tag (default accelerated)
         */
        template<typename T_array = T_dataType>
        T_dataType* getRawLocalAxis(const std::string axisname, staggerType stagger, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated)
        {
            auto &data = getAxis(axisname);
            if (tag == portableWrapper::arrayTags::host){
                auto axisArray = data.getPPLocalAxis<T_array, portableWrapper::arrayTags::host>(stagger);
                return axisArray.data();
            } else {
                auto axisArray = data.getPPLocalAxis<T_array, portableWrapper::arrayTags::accelerated>(stagger);
                return axisArray.data();
            }
        }

        /**
         * Get a raw pointer to the local axis delta data on this MPI rank
         * Note! This will include ghost cells if they are present in the local domain
         * so there will be overlap between MPI ranks
         * @param axisname The name of the axis
         * @param stagger The staggering type
         */
        template<typename T_array = T_dataType>
        T_dataType* getRawLocalDelta(const std::string axisname, staggerType stagger, portableWrapper::arrayTags tag=portableWrapper::arrayTags::accelerated)
        {
            auto &data = getAxis(axisname);
            if (tag == portableWrapper::arrayTags::host){
                auto deltaArray = data.getPPLocalDelta<T_array, portableWrapper::arrayTags::host>(stagger);
                return deltaArray.data();
            } else {
                auto deltaArray = data.getPPLocalDelta<T_array, portableWrapper::arrayTags::accelerated>(stagger);
                return deltaArray.data();
            }
        }

        /**
         * Register an axis that is derived from another axis, with a different number of cells over the same range
         * @param name The name of the new axis
         * @param sourceName The name of the axis that it is derived from
         * @param cells The number of cells of the new axis
         */
        void deriveAxis(const std::string &name, const std::string &sourceName, COUNT_TYPE cells)
        {
            if (cells < 1)
            {
                throw std::runtime_error("Error: a derived axis must have at least one cell\n");
            }
            const auto existing = axisMap.find(name);
            if (existing != axisMap.end())
            {
                // An axis that already exists with the same number of cells is left as it is
                if (existing->second.dim.getDomainElements(staggerType::CENTRED) == cells)
                {
                    return;
                }
                throw std::runtime_error("Error: axis " + name + " already exists in registry with a different number of cells\n");
            }
            const axisInfo &source = getAxis(sourceName);
            const SIGNED_INDEX_TYPE sourceCells = static_cast<SIGNED_INDEX_TYPE>(source.dim.getDomainElements(staggerType::CENTRED));
            const SIGNED_INDEX_TYPE derivedCells = static_cast<SIGNED_INDEX_TYPE>(cells);

            registerAxis(name, source.dim.mpiAxis);
            setDomainElements(name, cells);
            setPeriodic(name, source.periodic);
            auto &derived = getAxis(name);
            derived.sourceAxis = sourceName;

            // Edge i is at i * sourceCells / derivedCells along the source, by linear interpolation
            portableWrapper::hostArray<T_dataType, 1> sourceEdges;
            fillPPAxis(sourceName, sourceEdges, staggerType::HALF_CELL);
            auto edge = LAMBDA(SIGNED_INDEX_TYPE i)->T_dataType
            {
                const SIGNED_INDEX_TYPE scaled = i * sourceCells;
                const SIGNED_INDEX_TYPE lower = scaled / derivedCells;
                if (lower >= sourceCells)
                {
                    return sourceEdges(sourceCells);
                }
                const T_dataType fraction = static_cast<T_dataType>(scaled % derivedCells) / static_cast<T_dataType>(derivedCells);
                return (static_cast<T_dataType>(1) - fraction) * sourceEdges(lower) + fraction * sourceEdges(lower + 1);
            };
            derived.createPositionAxis(staggerType::HALF_CELL, portableWrapper::arrayTags::host, edge);
            derived.createCentredAxis<T_dataType>(portableWrapper::arrayTags::host);
        }

        /**
         * Decompose a single axis by setting the local elements for that axis based on the global elements and the MPI decomposition
         * A derived axis follows the decomposition of its source axis
         * @param axisName The name of the axis to decompose
         */
        void decomposeAxis(const std::string &axisName)
        {
            const int axis = getMPIAxis(axisName);
            const size_t globalElements = getDomainElements(axisName, staggerType::CENTRED);
            auto &axisRef = getAxis(axisName);
            std::vector<SIGNED_INDEX_TYPE> starts;
            if (axisRef.sourceAxis.empty())
            {
                // Decompose the axis, distributing any remainder to the first few ranks. An axis that is not decomposed is one block
                const int ranksAlongAxis = axis >= 0 ? mpiMgr.getDims()[axis] : 1;
                const size_t base = globalElements / ranksAlongAxis;
                const size_t remainder = globalElements % ranksAlongAxis;
                starts.assign(ranksAlongAxis + 1, 1);
                for (int p = 0; p < ranksAlongAxis; ++p)
                {
                    starts[p + 1] = starts[p] + static_cast<SIGNED_INDEX_TYPE>(base + (static_cast<size_t>(p) < remainder ? 1 : 0));
                }
            }
            else
            {
                // A derived axis follows its source: a rank owns the cells whose first edge lies in its source cells
                if (!getAxis(axisRef.sourceAxis).decomposition.isSet())
                {
                    decomposeAxis(axisRef.sourceAxis);
                }
                const axisDecomposition &sourceDecomposition = getAxis(axisRef.sourceAxis).decomposition;
                const SIGNED_INDEX_TYPE sourceCells = sourceDecomposition.globalCells();
                const SIGNED_INDEX_TYPE cells = static_cast<SIGNED_INDEX_TYPE>(globalElements);
                starts.resize(sourceDecomposition.ranks() + 1);
                for (int p = 0; p < sourceDecomposition.ranks(); ++p)
                {
                    starts[p] = ((sourceDecomposition.firstIndexOnRank(p) - 1) * cells + sourceCells - 1) / sourceCells + 1;
                }
                starts.back() = cells + 1;
            }
            axisRef.decomposition.setBlockStarts(starts);

            // Now set the part of the global axis that is on this processor
            const int myAxisRank = axis >= 0 ? mpiMgr.getCoords()[axis] : 0;
            const SIGNED_INDEX_TYPE localElements = axisRef.decomposition.cellsOnRank(myAxisRank);
            setLocalDomainElements(axisName, static_cast<COUNT_TYPE>(localElements), staggerType::CENTRED);
            const SIGNED_INDEX_TYPE localLB = axisRef.decomposition.firstIndexOnRank(myAxisRank) - 1;
            axisRef.dim.setGlobalBounds(localLB, localLB + localElements);
        }

        /**
         * Decompose all registered axes
         */
        void decomposeAllAxes()
        {
            for (const auto &pair : axisMap)
            {
                // An axis that has been decomposed already is left as it is
                if (pair.second.decomposition.isSet())
                {
                    continue;
                }
                SAMS::debug3 << "Decomposing axis: " << pair.first << std::endl;
                decomposeAxis(pair.first);
            }
        }

        /**
         * Get the number of cells on all of the ranks before a given rank, in the order of the ranks of the Cartesian
         * communicator. This is the first row that the rank owns in a numbering in which each rank's cells are contiguous
         * @tparam N The number of axes
         * @param axisNames The names of the axes
         * @param owner The rank in the Cartesian communicator
         */
        template<std::size_t N>
        SIGNED_INDEX_TYPE getBlockOffset(const std::array<std::string, N> &axisNames, int owner) const
        {
            SIGNED_INDEX_TYPE offset = 0;
            for (int r = 0; r < owner; ++r)
            {
                const std::array<int, MAX_RANK> axisRanks = mpiMgr.getCoordsOfRank(r);
                SIGNED_INDEX_TYPE cells = 1;
                for (std::size_t a = 0; a < N; ++a)
                {
                    const int axis = getMPIAxis(axisNames[a]);
                    cells *= getAxis(axisNames[a]).decomposition.cellsOnRank(axis >= 0 ? axisRanks[axis] : 0);
                }
                offset += cells;
            }
            return offset;
        }

        /**
         * Get the row of a cell in a numbering in which the cells of each rank are contiguous, in the order of the ranks of
         * the Cartesian communicator, with the first axis fastest inside a rank.
         * A cell beyond the domain has the row of the cell it wraps to if its axis is periodic, and the row -1 if it is not
         * @tparam N The number of axes
         * @param axisNames The names of the axes
         * @param globalIndex The global index of the cell along each axis, counted from 1
         */
        template<std::size_t N>
        SIGNED_INDEX_TYPE getBlockedIndex(const std::array<std::string, N> &axisNames, std::array<SIGNED_INDEX_TYPE, N> globalIndex) const
        {
            std::array<int, MAX_RANK> ownerAxisRanks{};
            std::array<SIGNED_INDEX_TYPE, N> ownerCells;
            std::array<SIGNED_INDEX_TYPE, N> local;
            for (std::size_t a = 0; a < N; ++a)
            {
                const int axis = getMPIAxis(axisNames[a]);
                const axisDecomposition &decomposition = getAxis(axisNames[a]).decomposition;
                const SIGNED_INDEX_TYPE global = decomposition.globalCells();
                if (globalIndex[a] < 1 || globalIndex[a] > global)
                {
                    if (axis < 0 || !mpiMgr.isPeriodic(axis)) return -1;
                    globalIndex[a] = ((globalIndex[a] - 1) % global + global) % global + 1;
                }
                const axisDecomposition::location where = decomposition.locate(globalIndex[a]);
                if (axis >= 0) ownerAxisRanks[axis] = where.axisRank;
                local[a] = where.local;
                ownerCells[a] = decomposition.cellsOnRank(where.axisRank);
            }
            const int owner = mpiMgr.getRankFromCoords(ownerAxisRanks);
            SIGNED_INDEX_TYPE inBlock = 0, stride = 1;
            for (std::size_t a = 0; a < N; ++a)
            {
                inBlock += (local[a] - 1) * stride;
                stride *= ownerCells[a];
            }
            return getBlockOffset(axisNames, owner) + inBlock;
        }

        void finalize()
        {
            axisMap.clear();
            manager.clear();
        }
    };

};

#endif // SAMS_AXISREGISTRY_H

