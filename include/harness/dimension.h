#ifndef SAMS_DIMENSION_H
#define SAMS_DIMENSION_H

#include "handles.h"
#include "memoryRegistry.h"
#include "staggerRegistry.h"
#include "pp/parallelWrapper.h"

namespace SAMS{

    /**
     * Struct representing a single dimension of a variable or an axis (zones and ghost cells)
     * lowerGhosts: number of ghost cells on the lower side of the dimension
     * upperGhosts: number of ghost cells on the upper side of the dimension
     */
    struct dimension
    {
        /**
         * Number of ghost cells on the lower side of the dimension
         */
        COUNT_TYPE lowerGhosts = 0;
        /**
         * Number of ghost cells on the upper side of the dimension
         */
        COUNT_TYPE upperGhosts = 0;
        /**
         * Number of actual zones in this dimension
         */
        COUNT_TYPE zones = 0;
        /**
         * Number of zones on this MPI rank
         */
        COUNT_TYPE zonesLocal = 0;
        /**
         * The index in the virtual global array of the lower bound of this dimension
         * i.e. if the dimension wasn't MPI decomposed what indices would this rank hold
         */
        SIGNED_INDEX_TYPE globalLowerIndex = 0;
        /**
         * The index in the virtual global array of the upper bound of this dimension
         * i.e. if the dimension wasn't MPI decomposed what indices would this rank hold
         */
        SIGNED_INDEX_TYPE globalUpperIndex = 0;
        /**
         * How is this dimension staggered?
         */
        staggerType stagger = staggerType::CENTRED;

        /**
         * Is this dimension periodic?
         */
        bool periodic = false;

        /**
         * What geometry does this dimension represent?
         */
        //geometryType geometry = geometryType::CARTESIAN;

        /**
         * The name of the axis this dimension is associated with
         * Empty string if no axis is associated and this is a custom dimension
         */
        std::string axisName = "";

        dimension() = default;

        dimension(staggerType stag)
            : stagger(stag) {}

        // Single equal numbers of ghost cells, with and without axis name
        dimension(COUNT_TYPE ghosts)
            : lowerGhosts(ghosts), upperGhosts(ghosts) {}
        dimension(const std::string &axisName, COUNT_TYPE ghosts)
            : lowerGhosts(ghosts), upperGhosts(ghosts), axisName(axisName) {
            }

        dimension(COUNT_TYPE ghosts, staggerType stagger)
            : lowerGhosts(ghosts), upperGhosts(ghosts), stagger(stagger) {}
        dimension(const std::string &axisName, COUNT_TYPE ghosts, staggerType stagger)
            : lowerGhosts(ghosts), upperGhosts(ghosts), stagger(stagger), axisName(axisName) {
            }

        // Separate numbers of ghost cells, with and without axis name
        dimension(COUNT_TYPE lowerGhosts, COUNT_TYPE upperGhosts)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts) {}
        dimension(const std::string &axisName, COUNT_TYPE lowerGhosts, COUNT_TYPE upperGhosts)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), axisName(axisName) {
            }

        // With staggering, with and without axis name
        dimension(COUNT_TYPE lowerGhosts, COUNT_TYPE upperGhosts, staggerType stagger)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), stagger(stagger) {}
        dimension(const std::string &axisName, COUNT_TYPE lowerGhosts, COUNT_TYPE upperGhosts, staggerType stagger)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), stagger(stagger), axisName(axisName) {
            }

        // With zones. Adding axis name would be redundant
        dimension(COUNT_TYPE lowerGhosts, COUNT_TYPE upperGhosts, COUNT_TYPE zones, staggerType stagger)
            : lowerGhosts(lowerGhosts), upperGhosts(upperGhosts), zones(zones), zonesLocal(zones), stagger(stagger) {}

        /**
         * Get the total number of cells different between the current staggering and another staggering
         * @param s The other staggering type
         */
        SIGNED_INDEX_TYPE getCellDelta(staggerType s) const
        {
            const auto &sreg = getstaggerRegistry();
            return sreg.getExtraCells(s) - sreg.getExtraCells(stagger);
        }

        /**
         * Get the number of elements in this dimension for a given staggering type
         */
        COUNT_TYPE getDomainElements(staggerType s) const
        {
            if (zones == 0)
            {
                throw std::runtime_error("Error: dimension zones not set\n");
            }
            return zones + getCellDelta(s);
        }

        /**
         * Get the number of elements in this dimension for the native staggering type
         */
        COUNT_TYPE getDomainElements() const
        {
            if (zones == 0)
            {
                throw std::runtime_error("Error: dimension zones not set\n");
            }
            return zones;
        }

        /**
         * Set the number of elements in this dimension specifying the number of elements for a given staggering type
         */
        void setDomainElements(COUNT_TYPE elements, staggerType s)
        {
            zones = elements - getCellDelta(s);
            zonesLocal = elements - getCellDelta(s);
            globalLowerIndex = getLB(s);
            globalUpperIndex = getUB(s);
        }

        /**
         * Set the number of elements in this dimension specifying the number of elements for the native staggering type
         */
        void setDomainElements(COUNT_TYPE elements)
        {
            setDomainElements(elements, stagger);
        }

        /**
         * Get the number of local elements in this dimension for a given staggering type
         * @param s The staggering type
         */
        COUNT_TYPE getLocalDomainElements(staggerType s) const
        {
            if (zonesLocal == 0)
            {
                throw std::runtime_error("Error: dimension local zones not set\n");
            }
            return zonesLocal + getCellDelta(s);
        }

        /**
         * Get the number of local elements in this dimension for the native staggering type
         * @param elements The number of local elements
         */
        void setLocalDomainElements(COUNT_TYPE elements)
        {
            zonesLocal = elements;
        }

        /**
         * Set the number of local elements in this dimension specifying the number of elements for a given staggering type
         * @param elements The number of local elements
         * @param s The staggering type
         */
        void setLocalDomainElements(COUNT_TYPE elements, staggerType s)
        {
            zonesLocal = elements - getCellDelta(s);
        }

        /**
         * Set the global bounds for this dimension
         * @param lb The global lower bound
         * @param ub The global upper bound
         * @param s The staggering type
         */
        void setGlobalBounds(SIGNED_INDEX_TYPE lb, SIGNED_INDEX_TYPE ub, staggerType s)
        {
            globalLowerIndex = lb - getstaggerRegistry().getLowerAdjust(s) + getstaggerRegistry().getLowerAdjust(stagger);
            globalUpperIndex = ub - getstaggerRegistry().getUpperAdjust(s) + getstaggerRegistry().getUpperAdjust(stagger);
        }

        /**
         * Set the global bounds for this dimension for its native staggering type
         * @param lb The global lower bound
         * @param ub The global upper bound
         */
        void setGlobalBounds(SIGNED_INDEX_TYPE lb, SIGNED_INDEX_TYPE ub)
        {
            setGlobalBounds(lb, ub, stagger);
        }

        /**
         * Get the number of elements in this dimension for its native staggering type
         */
        COUNT_TYPE getNativeDomainElements() const
        {
            return getDomainElements(stagger);
        }

        /**
         * Get the number of local elements in this dimension for its native staggering type
         */
        COUNT_TYPE getLocalNativeDomainElements() const
        {
            return getLocalDomainElements(stagger);
        }

        /**
         * Get the number of cell centres in this dimension
         */
        COUNT_TYPE getDomainCells() const
        {
            return getDomainElements(staggerType::CENTRED);
        }

        /**
         * Get the number of local cell centres in this dimension
         */
        COUNT_TYPE getLocalDomainCells() const
        {
            return getLocalDomainElements(staggerType::CENTRED);
        }

        /**
         * Get the number of edges in this dimension
         */
        COUNT_TYPE getDomainEdges() const
        {
            return getDomainElements(staggerType::HALF_CELL);
        }

        /**
         * Get the number of local edges in this dimension
         */
        COUNT_TYPE getLocalDomainEdges() const
        {
            return getLocalDomainElements(staggerType::HALF_CELL);
        }

        /**
         * Get the lower bound for the dimension for the local part of the MPI decomposition
         */
        SIGNED_INDEX_TYPE getLocalLB() const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(stagger) - static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
        * Get the lower bound for the dimension for the local part of the MPI decomposition, assuming zero-based indexing
        * @note Always returns zero, but just used to avoid magic numbers in code
         */
        SIGNED_INDEX_TYPE getLocalLBZeroBase() const
        {
            //Default index starts at 0, no adjustment needed
            return 0;
        }

        /**
         * Get the lower bound for the domain i.e. the first index of the real data for the local part of the MPI decomposition
         */

        /**
         * Get the lower bound for the dimension for the local part of the MPI decomposition
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getLocalLB(staggerType s) const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(s) - static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
        * Get the lower bound for the dimension for the local part of the MPI decomposition, assuming zero-based indexing
        * @note Always returns zero, but just used to avoid magic numbers in code
        @param s The staggering type
        */
        SIGNED_INDEX_TYPE getLocalLBZeroBase(staggerType s) const
        {
            //Default index starts at 0, no adjustment needed
            return 0;
        }

        /**
         * Get the global lower bound for the dimension
         */
        SIGNED_INDEX_TYPE getLB() const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(stagger) - static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
        * Get the global lower bound for the dimension, assuming zero-based indexing
        * @note Always returns zero, but just used to avoid magic numbers in code
        */
        SIGNED_INDEX_TYPE getLBZeroBase() const
        {
            //Default index starts at 0, no adjustment needed
            return 0;
        }

        /**
         * Get the global lower bound for the dimension
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getLB(staggerType s) const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(s) - static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
         * Get the global lower bound for the dimension, assuming zero-based indexing
         * @note Always returns zero, but just used to avoid magic numbers in code
         */
        SIGNED_INDEX_TYPE getLBZeroBase(staggerType s) const
        {
            //Default index starts at 0, no adjustment needed
            return 0;
        }

        /**
         * Get the global lower bound for the actual DOMAIN i.e. the first index of the real data
         */
        SIGNED_INDEX_TYPE getDomainLB() const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(stagger);
        }

        /**
         * Get the global lower bound for the actual DOMAIN i.e. the first index of the real data, assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getDomainLBZeroBase() const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
         * Get the global lower bound for the actual DOMAIN i.e. the first index of the real data for a given staggering type
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getDomainLB(staggerType s) const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(s);
        }

        /**
         * Get the global lower bound for the actual DOMAIN i.e. the first index of the real data for a given staggering type, assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getDomainLBZeroBase(staggerType s) const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
         * Get the lower bound for the actual DOMAIN i.e. the first index of the real data
         */
        SIGNED_INDEX_TYPE getLocalDomainLB() const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(stagger);
        }

        /**
         * Get the lower bound for the actual DOMAIN i.e. the first index of the real data for a given staggering type
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getLocalDomainLB(staggerType s) const
        {
            //Default index starts at 1, but ask the staggerRegistry for adjustment
            return 1 + getstaggerRegistry().getLowerAdjust(s);
        }

        /**
        * Get the lower bound for the actual DOMAIN i.e. the first index of the real data for a given staggering type, assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getLocalDomainLBZeroBase(staggerType s) const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
         * Get the upper bound for the dimension for the local part of the MPI decomposition
         */
        SIGNED_INDEX_TYPE getLocalUB() const
        {
            //Start the lowerbound at the correct place for the staggering, then add the local elements and ghost cells
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(stagger);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getLocalNativeDomainElements()) + getstaggerRegistry().getUpperAdjust(stagger) + static_cast<SIGNED_INDEX_TYPE>(upperGhosts) -1;
        }

        /**
        * Get the upper bound for the dimension for the local part of the MPI decomposition, assuming zero-based indexing
        */
        SIGNED_INDEX_TYPE getLocalUBZeroBase() const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return static_cast<SIGNED_INDEX_TYPE>(getLocalNativeDomainElements() + upperGhosts + lowerGhosts) -1;
        }

        /**
         * Get the upper bound for the dimension for the local part of the MPI decomposition
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getLocalUB(staggerType s) const
        {
            //Start the lowerbound at the correct place for the staggering, then add the local elements and ghost cells
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(s);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getLocalDomainElements(s)) + getstaggerRegistry().getUpperAdjust(s) + static_cast<SIGNED_INDEX_TYPE>(upperGhosts) -1;
        }

        /**
        * Get the upper bound for the dimension for the local part of the MPI decomposition, assuming zero-based indexing
        * @param s The staggering type
        */
        SIGNED_INDEX_TYPE getLocalUBZeroBase(staggerType s) const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return static_cast<SIGNED_INDEX_TYPE>(getLocalDomainElements(s) + upperGhosts + lowerGhosts) -1;
        }

        /**
         * Get the global upper bound for the dimension
         */
        SIGNED_INDEX_TYPE getUB() const
        {
            //Start the lowerbound at the correct place for the staggering, then add the elements and ghost cells
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(stagger);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getLocalNativeDomainElements()) + getstaggerRegistry().getUpperAdjust(stagger) + static_cast<SIGNED_INDEX_TYPE>(upperGhosts) -1;
        }

        /**
         * Get the global upper bound for the dimension, assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getUBZeroBase() const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return static_cast<SIGNED_INDEX_TYPE>(getNativeDomainElements() + upperGhosts + lowerGhosts) -1;
        }

        /**
         * Get the global upper bound for the dimension
         */
        SIGNED_INDEX_TYPE getUB(staggerType s) const
        {
            //Start the lowerbound at the correct place for the staggering, then add the elements and ghost cells
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(s);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getDomainElements(s)) + getstaggerRegistry().getUpperAdjust(s) + static_cast<SIGNED_INDEX_TYPE>(upperGhosts) -1;
        }

        /**
         * Get the global upper bound for the dimension, assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getUBZeroBase(staggerType s) const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return static_cast<SIGNED_INDEX_TYPE>(getDomainElements(s) + upperGhosts + lowerGhosts) -1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data)
         */
        SIGNED_INDEX_TYPE getDomainUB() const
        {
            //Start the lowerbound at the correct place for the staggering, then add the elements
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(stagger);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getNativeDomainElements()) + getstaggerRegistry().getUpperAdjust(stagger) -1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data), assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getDomainUBZeroBase() const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return getDomainLBZeroBase() + static_cast<SIGNED_INDEX_TYPE>(getNativeDomainElements())-1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data) for a given staggering type
         */
        SIGNED_INDEX_TYPE getDomainUB(staggerType s) const
        {
            //Start the lowerbound at the correct place for the staggering, then add the elements
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(s);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getDomainElements(s)) + getstaggerRegistry().getUpperAdjust(s) -1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data) for a given staggering type, assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getDomainUBZeroBase(staggerType s) const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return getDomainLBZeroBase(s) + static_cast<SIGNED_INDEX_TYPE>(getDomainElements(s))-1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data)
         */
        SIGNED_INDEX_TYPE getLocalDomainUB() const
        {
            //Start the lowerbound at the correct place for the staggering, then add the elements
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(stagger);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getLocalNativeDomainElements()) + getstaggerRegistry().getUpperAdjust(stagger) -1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data), assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getLocalDomainUBZeroBase() const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return getDomainLBZeroBase() + static_cast<SIGNED_INDEX_TYPE>(getLocalNativeDomainElements())-1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data) for a given staggering type
         */
        SIGNED_INDEX_TYPE getLocalDomainUB(staggerType s) const
        {
            //Start the lowerbound at the correct place for the staggering, then add the elements
            SIGNED_INDEX_TYPE LB = 1+getstaggerRegistry().getLowerAdjust(s);
            return LB + static_cast<SIGNED_INDEX_TYPE>(getLocalDomainElements(s)) + getstaggerRegistry().getUpperAdjust(s) -1;
        }

        /**
         * Get the upper bound for the actual DOMAIN (i.e. the last index of the real data) for a given staggering type, assuming zero-based indexing
         */
        SIGNED_INDEX_TYPE getLocalDomainUBZeroBase(staggerType s) const
        {
            //Now we have to add the number of ghost cells to get to zero-based indexing
            return getLocalDomainLBZeroBase(s) + static_cast<SIGNED_INDEX_TYPE>(getLocalDomainElements(s))-1;
        }

        /**
         * Get the global lower bound for the dimension. i.e. the index of the first domain cell in the "effective" global grid
         * on the local MPI rank
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getGlobalDomainLB(SAMS::staggerType s) const
        {
            return globalLowerIndex + getstaggerRegistry().getLowerAdjust(s) - getstaggerRegistry().getLowerAdjust(stagger);
        }

        /**
         * Get the global lower bound for the dimension. i.e. the index of the first domain cell in the "effective" global grid
         * on the local MPI rank. This uses the native staggering type
         */
        SIGNED_INDEX_TYPE getGlobalDomainLB() const
        {
            return getGlobalDomainLB(stagger);
        }

        /**
         * Get the global lower bound for the dimension. i.e. the index of the first domain cell in the "effective" global grid including ghost cells
         */
        SIGNED_INDEX_TYPE getGlobalLB() const
        {
            return getGlobalDomainLB() - static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
         * Get the global upper bound for the dimension. i.e. the index of the last domain cell in the "effective" global grid including ghost cells
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getGlobalLB(staggerType s) const
        {
            return getGlobalDomainLB(s) - static_cast<SIGNED_INDEX_TYPE>(lowerGhosts);
        }

        /**
         * Get the global upper bound for the dimension. i.e. the index of the last domain cell in the "effective" global grid including ghost cells
         */
        SIGNED_INDEX_TYPE getGlobalUB() const
        {
            return getGlobalDomainUB() + static_cast<SIGNED_INDEX_TYPE>(upperGhosts);
        }



        /**
         * Get the global upper bound for the dimension. i.e. the index of the last domain cell in the "effective" global grid
         * on the local MPI rank. This uses the native staggering type
         */
        SIGNED_INDEX_TYPE getGlobalDomainUB() const
        {
            return getGlobalDomainUB(stagger);
        }

        /**
         * Get the global upper bound for the dimension. i.e. the index of the last domain cell in the "effective" global grid
         * on the local MPI rank
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getGlobalDomainUB(SAMS::staggerType s) const
        {
            return globalUpperIndex + getstaggerRegistry().getUpperAdjust(s) - getstaggerRegistry().getUpperAdjust(stagger);
        }

        /**
         * Get the global upper bound for the dimension. i.e. the index of the last domain cell in the "effective" global grid including ghost cells
         * @param s The staggering type
         */
        SIGNED_INDEX_TYPE getGlobalUB(staggerType s) const
        {
            return getGlobalDomainUB(s) + static_cast<SIGNED_INDEX_TYPE>(upperGhosts);
        }


        /**
         * Get the range for the local part of the MPI decomposition
         * @return A portableWrapper::Range representing the local range
         */
        portableWrapper::Range getLocalRange() const
        {
            return portableWrapper::Range(getLocalLB(), getLocalUB());
        }

        /**
         * Get the range for the local part of the MPI decomposition
         * @param s The staggering type
         * @return A portableWrapper::Range representing the local range
         */
        portableWrapper::Range getLocalRange(staggerType s) const
        {
            return portableWrapper::Range(getLocalLB(s), getLocalUB(s));
        }

        /**
         * Get the range for the local part of the MPI decomposition
         * @return A portableWrapper::Range representing the local range
         */
        portableWrapper::Range getLocalRangeZeroBase() const
        {
            return portableWrapper::Range(getLocalLBZeroBase(), getLocalUBZeroBase());
        }

        /**
         * Get the range for the local part of the MPI decomposition
         * @param s The staggering type
         * @return A portableWrapper::Range representing the local range
         */
        portableWrapper::Range getLocalRangeZeroBase(staggerType s) const
        {
            return portableWrapper::Range(getLocalLBZeroBase(s), getLocalUBZeroBase(s));
        }

        /**
         * Get the range for the actual domain (i.e. the real data only, no ghost cells)
         * @return A portableWrapper::Range representing the domain range
         */
        portableWrapper::Range getDomainRange() const
        {
            return portableWrapper::Range(getLocalDomainLB(), getLocalDomainUB());
        }

        /**
         * Get the range for the actual domain (i.e. the real data only, no ghost cells)
         * @param s The staggering type
         * @return A portableWrapper::Range representing the domain range
         */
        portableWrapper::Range getDomainRange(staggerType s) const
        {
            return portableWrapper::Range(getLocalDomainLB(s), getLocalDomainUB(s));
        }

        /**
         * Get the global range for the dimension
         * @return A portableWrapper::Range representing the global range
         */
        portableWrapper::Range getRange() const
        {
            return portableWrapper::Range(getLB(), getUB());
        }

        /**
         * Get the global range for the dimension
         * @param s The staggering type
         * @return A portableWrapper::Range representing the global range
         */
        portableWrapper::Range getRange(staggerType s) const
        {
            return portableWrapper::Range(getLB(s), getUB(s));
        }

        /**
         * Get the global range for the dimension
         * @return A portableWrapper::Range representing the global range
         */
        portableWrapper::Range getRangeZeroBase() const
        {
            return portableWrapper::Range(getLBZeroBase(), getUBZeroBase());
        }

        /**
         * Get the global range for the dimension
         * @param s The staggering type
         * @return A portableWrapper::Range representing the global range
         */
        portableWrapper::Range getRangeZeroBase(staggerType s) const
        {
            return portableWrapper::Range(getLBZeroBase(s), getUBZeroBase(s));
        }

        /**
         * Get the global range for the local domain (i.e. the real data only, no ghost cells)
         * @return A portableWrapper::Range representing the global domain range
         */
        portableWrapper::Range getGlobalDomainRange() const 
        {
            return portableWrapper::Range(getGlobalDomainLB(), getGlobalDomainUB());
        }

        /**
         * Get the global range for the local domain (i.e. the real data only, no ghost cells)
         * @param s The staggering type
         * @return A portableWrapper::Range representing the global domain range
         */
        portableWrapper::Range getGlobalDomainRange(staggerType s) const
        {
            return portableWrapper::Range(getGlobalDomainLB(s), getGlobalDomainUB(s));
        }

        /**
         * Get the global range for the whole dimension including ghost cells
         * @return A portableWrapper::Range representing the global range including ghost cells
         */
        portableWrapper::Range getGlobalRange() const
        {
            return portableWrapper::Range(getGlobalLB() , getGlobalUB());
        }

        /**
         * Get the global range for the whole dimension including ghost cells
         * @param s The staggering type
         * @return A portableWrapper::Range representing the global range including ghost cells
         */
        portableWrapper::Range getGlobalRange(staggerType s) const
        {
            return portableWrapper::Range(getGlobalLB(s), getGlobalUB(s));
        }

        /**
         * Make another dimension consistent with this one (i.e. copy over zones, zonesLocal, globalLowerIndex, globalUpperIndex). Leave staggering and ghost cells unchanged.
         * This is to allow a variable to specialise its dimensions while using the axis
         * registry for shared information
         */
        void getInfoFrom(const dimension &src)
        {
            zones = src.getDomainElements(stagger);
            zonesLocal = src.getLocalDomainElements(stagger);
            globalLowerIndex = src.getGlobalDomainLB(stagger);
            globalUpperIndex = src.getGlobalDomainUB(stagger);
        }

    }; // struct dimension

};

#endif