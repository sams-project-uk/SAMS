#ifndef SAMS_BUILTINBOUNDARYCONDITIONS_H
#define SAMS_BUILTINBOUNDARYCONDITIONS_H

#include "boundaryConditions.h"
#include "pp/parallelWrapper.h"
#include "variableDef.h"
#include <array>

namespace SAMS
{

    /**
     * General boundary condition class for a single variable. Calculates the boundary ranges automatically.
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class singleVariableBC : public boundaryConditions
    {
    public:
        portableWrapper::portableArray<T, rank, tag> variable; // The variable to apply the boundary condition to
        variableDef varDef;
        //Inner array is the ranges that correspond to the boundary on each edge
        //The outer array is over all edges (lower and upper for each dimension)
        std::array<portableWrapper::std_N_ary_tuple_type_t<portableWrapper::Range, rank>, 2*rank> boundaryRanges;
    public:
        singleVariableBC(const SAMS::variableDef &varDef)
            :
            variable(varDef.template getPPArray<T, rank, tag>()), varDef(varDef)
        {
            for (int iface = 0; iface < rank; iface++){
                for (int i=0; i<rank; i++)
                {
                    auto &dim = varDef.getDimension(i);
                    if (i != iface){
                        portableWrapper::getTupleElement(boundaryRanges[iface*2], i) = dim.getLocalRange();
                        portableWrapper::getTupleElement(boundaryRanges[iface*2+1], i) = dim.getLocalRange();
                    } else {
                        portableWrapper::getTupleElement(boundaryRanges[iface*2], i) = dim.getLocalNonDomainRange(SAMS::domain::edges::lower);
                        portableWrapper::getTupleElement(boundaryRanges[iface*2+1], i) = dim.getLocalNonDomainRange(SAMS::domain::edges::upper);
                    }
                }
            }
        }
    };

    /**
     * A simple boundary condition that clamps the variable to a specified value at the boundary.
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class simpleClamp : public singleVariableBC<T, rank, tag>
    {
        T clampValue;   
    public:
    /**
     * Constructor
     * @param varDef The variable definition
     * @param clampValue The value to clamp the variable to at the boundary
     */
        simpleClamp(const SAMS::variableDef &varDef, T clampValue)
            : singleVariableBC<T, rank, tag>(varDef), clampValue(clampValue)
        {
        }

        /**
         * Apply the boundary condition (implements base class method)
         * @param dimension The dimension to apply the boundary condition on
         * @param edge The edge to apply the boundary condition on (lower or upper)
         */
        virtual void apply(int dimension, SAMS::domain::edges edge)
        {
            int edgeIndex = (edge == SAMS::domain::edges::lower) ? 0 : 1;
            auto &ranges = this->boundaryRanges[dimension * 2 + edgeIndex];
            portableWrapper::portableArray<T, rank, tag> slice = std::apply([this](auto... rangeArgs){
                return this->variable(rangeArgs...);
            }, ranges);
            portableWrapper::assign(slice, clampValue);
        }

    };

    /**
     * Helper functor to perform mirror boundary conditions
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class mirrorHelper 
    {
        private:
            portableWrapper::portableArray<T, rank, tag> Array;
            int dim;
            T_indexType lastDomainPoint, firstGhostPoint;
            T mirrorValue;
        public:

        /**
         * Constructor
         * @param Array The array to apply the mirror boundary condition to
         * @param dim The dimension to apply the boundary condition on
         * @param lastDomainPoint The last point in the domain (to mirror from)
         * @param firstGhostPoint The first ghost point (to mirror to)
         * @note The first ghost point and the last domain point may be the same point for some staggers
         * This does lead to extra work since it assigns the cell to itself, but it simplifies the code
         * consider optimizing this in the future if performance is an issue
         * @note This implements mirroring around the value on the boundary itself (this either uses the last domain point for half cell stagger or averages the last two domain points for cell centered stagger)
         */
        mirrorHelper(portableWrapper::portableArray<T, rank, tag> &Array, int dim, T_indexType lastDomainPoint, T_indexType firstGhostPoint)
            : Array(Array), dim(dim), lastDomainPoint(lastDomainPoint), firstGhostPoint(firstGhostPoint)
        {
        }

        template<typename... Params>
        FUNCTORMETHODPREFIX void operator()(Params... params) const {

            TUPLE<Params...> dst(params...), firstGhostTuple(params...);
            TUPLE<Params...> src(params...), lastDomainTuple(params...);

            //Find the boundary value to mirror around
            portableWrapper::getTupleElement(firstGhostTuple, dim) = firstGhostPoint;
            portableWrapper::getTupleElement(lastDomainTuple, dim) = lastDomainPoint;
            T boundaryValue = (APPLY(Array, lastDomainTuple) + APPLY(Array, lastDomainTuple))/static_cast<T>(2);

            //Now find your distance from the boundary and mirror the value around it
            T_indexType distance = portableWrapper::getTupleElement(dst, dim) - firstGhostPoint;
            portableWrapper::getTupleElement(src, dim) = lastDomainPoint - distance;
            T delta = APPLY(Array, src) - boundaryValue;
            APPLY(Array, dst) = boundaryValue - delta;
        }
    };

    /**
     * A simple mirror boundary condition that reflects the variable at the boundary.
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     * @note uses the mirrorHelper functor
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class simpleMirrorBC : public singleVariableBC<T, rank, tag>
    {
    public:
        simpleMirrorBC(const SAMS::variableDef &varDef)
            : singleVariableBC<T, rank, tag>(varDef)
        {
        }

        void apply(int dimension, SAMS::domain::edges edge) override
        {
            int edgeIndex = (edge == SAMS::domain::edges::lower) ? 0 : 1;
            auto &ranges = this->boundaryRanges[dimension * 2 + edgeIndex];
            auto &var = this->variable;
            T_indexType lastDomainPoint, firstGhostPoint;
            auto &dimInfo = this->varDef.getDimension(dimension);
            if (edge == SAMS::domain::edges::lower){
                lastDomainPoint = dimInfo.getLocalDomainLB();
                firstGhostPoint = dimInfo.getLocalNonDomainUB(SAMS::domain::edges::lower);
            } else {
                lastDomainPoint = dimInfo.getLocalDomainUB();
                firstGhostPoint = dimInfo.getLocalNonDomainLB(SAMS::domain::edges::upper);
            }
            mirrorHelper<T, rank, tag> helper(var, dimension, lastDomainPoint, firstGhostPoint);
            std::apply([&helper](auto... params){
                portableWrapper::applyKernel(helper, params...);
            }, ranges);
        }
    };


    /**
     * Helper functor for reflectionBC: sets a ghost point to parity * (its mirror image across the
     * symmetry plane). The mirror image of index g is (indexSum - g); see reflectionBC.
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class reflectionHelper
    {
        private:
            portableWrapper::portableArray<T, rank, tag> Array;
            int dim;
            T_indexType indexSum;
            T parity;
        public:
        reflectionHelper(portableWrapper::portableArray<T, rank, tag> &Array, int dim, T_indexType indexSum, T parity)
            : Array(Array), dim(dim), indexSum(indexSum), parity(parity)
        {
        }

        template<typename... Params>
        FUNCTORMETHODPREFIX void operator()(Params... params) const {
            TUPLE<Params...> dst(params...), src(params...);
            portableWrapper::getTupleElement(src, dim) = indexSum - portableWrapper::getTupleElement(dst, dim);
            APPLY(Array, dst) = parity * APPLY(Array, src);
        }
    };

    /**
     * Symmetry-plane (reflection) boundary condition: the boundary is a plane of mirror symmetry and the
     * variable is even (parity = +1) or odd (parity = -1) under the reflection. Each ghost point is set to
     * parity times its mirror image inside the domain:
     *   - cell-centred data (the plane lies on the face between the last ghost and the first domain cell):
     *     lower edge ghost g <- parity * A(2 LB - 1 - g), upper edge ghost g <- parity * A(2 UB + 1 - g);
     *   - half-cell (vertex / face) data, or data marked face-centred in this dimension (the plane passes
     *     through the boundary point P = first/last face): ghost g <- parity * A(2 P - g), and for odd
     *     variables the value ON the plane is set to zero (e.g. the normal velocity or normal mass flux).
     * faceCentredDim marks a dimension in which the data are face values although the variable is
     * registered as cell-centred (LARE's remap mass flux dm, which holds fluxes through x faces during the
     * x remap). For such data the face on the boundary has index LB - 1.
     * This differs from simpleMirrorBC, which reflects the deviation from the boundary value
     * (ghost = 2 A(boundary) - A(interior)), i.e. a linear extrapolation rather than a symmetry plane.
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class reflectionBC : public singleVariableBC<T, rank, tag>
    {
        T parity;
        int faceCentredDim;
    public:
        reflectionBC(const SAMS::variableDef &varDef, T parity, int faceCentredDim = -1)
            : singleVariableBC<T, rank, tag>(varDef), parity(parity), faceCentredDim(faceCentredDim)
        {
        }

        void apply(int dimension, SAMS::domain::edges edge) override
        {
            const bool lower = (edge == SAMS::domain::edges::lower);
            auto &ranges = this->boundaryRanges[dimension * 2 + (lower ? 0 : 1)];
            auto &dimInfo = this->varDef.getDimension(dimension);
            const T_indexType LB = dimInfo.getLocalDomainLB();
            const T_indexType UB = dimInfo.getLocalDomainUB();
            const bool cellCentred = (LB == dimInfo.getLocalDomainLB(SAMS::staggerType::CENTRED));
            const bool faceData = (dimension == faceCentredDim);

            // Mirror index sum, and the index of the point lying on the plane (if any)
            T_indexType indexSum, planeIndex = 0;
            bool pointOnPlane = true;
            if (faceData && cellCentred) {             // face values stored with cell-centred indexing
                planeIndex = lower ? LB - 1 : UB;
                indexSum = 2 * planeIndex;
            } else if (cellCentred) {                  // plane between cells
                indexSum = lower ? 2 * LB - 1 : 2 * UB + 1;
                pointOnPlane = false;
            } else {                                   // half-cell: plane through the boundary point
                planeIndex = lower ? LB : UB;
                indexSum = 2 * planeIndex;
            }

            reflectionHelper<T, rank, tag> helper(this->variable, dimension, indexSum, parity);
            std::apply([&helper](auto... params){
                portableWrapper::applyKernel(helper, params...);
            }, ranges);

            // An odd variable vanishes on the symmetry plane
            if (pointOnPlane && parity < static_cast<T>(0)) {
                auto planeRanges = ranges;
                portableWrapper::getTupleElement(planeRanges, dimension) = portableWrapper::Range(planeIndex, planeIndex);
                portableWrapper::portableArray<T, rank, tag> slice = std::apply([this](auto... rangeArgs){
                    return this->variable(rangeArgs...);
                }, planeRanges);
                portableWrapper::assign(slice, static_cast<T>(0));
            }
        }
    };

   /**
     * Helper functor to perform zero gradient boundary conditions
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class zeroGradientHelper 
    {
        private:
            portableWrapper::portableArray<T, rank, tag> Array;
            int dim;
            T_indexType lastDomainPoint, firstGhostPoint;
            T mirrorValue;
        public:

        /**
         * Constructor
         * @param Array The array to apply the mirror boundary condition to
         * @param dim The dimension to apply the boundary condition on
         * @param lastDomainPoint The last point in the domain (to mirror from)
         * @note This implements mirroring around the value on the boundary itself (this either uses the last domain point for half cell stagger or averages the last two domain points for cell centered stagger)
         */
        zeroGradientHelper(portableWrapper::portableArray<T, rank, tag> &Array, int dim, T_indexType lastDomainPoint)
            : Array(Array), dim(dim), lastDomainPoint(lastDomainPoint)
        {
        }

        template<typename... Params>
        FUNCTORMETHODPREFIX void operator()(Params... params) const {

            TUPLE<Params...> dst(params...), src(params...);

            //Zero gradient: just copy the last domain value to the ghost cells

            //Now find your distance from the boundary and mirror the value around it
            portableWrapper::getTupleElement(src, dim) = lastDomainPoint;
            APPLY(Array, dst) = APPLY(Array, src);
        }
    };

    /**
     * A simple zero gradient boundary condition that reflects the variable at the boundary.
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     * @note uses the mirrorHelper functor
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class simpleZeroGradientBC : public singleVariableBC<T, rank, tag>
    {
    public:
        simpleZeroGradientBC(const SAMS::variableDef &varDef)
            : singleVariableBC<T, rank, tag>(varDef)
        {
        }

        void apply(int dimension, SAMS::domain::edges edge) override
        {
            int edgeIndex = (edge == SAMS::domain::edges::lower) ? 0 : 1;
            auto &ranges = this->boundaryRanges[dimension * 2 + edgeIndex];
            auto &var = this->variable;
            T_indexType lastDomainPoint;
            auto &dimInfo = this->varDef.getDimension(dimension);
            if (edge == SAMS::domain::edges::lower){
                lastDomainPoint = dimInfo.getLocalDomainLB();
            } else {
                lastDomainPoint = dimInfo.getLocalDomainUB();
            }
            zeroGradientHelper<T, rank, tag> helper(var, dimension, lastDomainPoint);
            std::apply([&helper](auto... params){
                portableWrapper::applyKernel(helper, params...);
            }, ranges);
        }
    };

    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class simplePeriodicBC : public singleVariableBC<T, rank, tag>
    {
        std::array<portableWrapper::std_N_ary_tuple_type_t<portableWrapper::Range, rank>, 2*rank> sourceRanges;
        std::array<portableWrapper::portableArray<T, rank, tag>, 2*rank> sourceSlices;
        std::array<portableWrapper::portableArray<T, rank, tag>, 2*rank> destSlices;
    public:
        simplePeriodicBC(const SAMS::variableDef &varDef)
            : singleVariableBC<T, rank, tag>(varDef)
        {
            using Range = portableWrapper::Range;
            for (int iface = 0; iface < rank; iface++){
                for (int i=0; i<rank; i++)
                {
                    auto &dim = varDef.getDimension(i);
                    if (i != iface){
                        portableWrapper::getTupleElement(sourceRanges[iface*2], i) = dim.getLocalRange();
                        portableWrapper::getTupleElement(sourceRanges[iface*2+1], i) = dim.getLocalRange();
                    } else {
                        portableWrapper::getTupleElement(sourceRanges[iface*2], i) = 
                            Range(dim.getLocalDomainUB() - dim.getLocalNonDomainCount(domain::edges::lower) + 1, dim.getLocalDomainUB());
                        portableWrapper::getTupleElement(sourceRanges[iface*2+1], i) = 
                            Range(dim.getLocalDomainLB(), dim.getLocalDomainLB() + dim.getLocalNonDomainCount(domain::edges::upper) - 1);
                    }
                }

                //Grab the source and destination slices and just cache them
                sourceSlices[iface*2] = std::apply([this](auto... rangeArgs){
                    return this->variable(rangeArgs...);
                }, sourceRanges[iface*2]);
                sourceSlices[iface*2+1] = std::apply([this](auto... rangeArgs){
                    return this->variable(rangeArgs...);
                }, sourceRanges[iface*2+1]);
                destSlices[iface*2] = std::apply([this](auto... rangeArgs){
                    return this->variable(rangeArgs...);
                }, this->boundaryRanges[iface*2]);
                destSlices[iface*2+1] = std::apply([this](auto... rangeArgs){
                    return this->variable(rangeArgs...);
                }, this->boundaryRanges[iface*2+1]);

            }
        }
        void apply(int dimension, SAMS::domain::edges edge) override
        {
            int edgeIndex = (edge == SAMS::domain::edges::lower) ? 0 : 1;
            auto &sourceSlice = sourceSlices[dimension * 2 + edgeIndex];
            auto &destSlice = destSlices[dimension * 2 + edgeIndex];
            portableWrapper::assign(destSlice, sourceSlice);
        }
    };

} // namespace SAMS

#endif // SAMS_BUILTINBOUNDARYCONDITIONS_H
