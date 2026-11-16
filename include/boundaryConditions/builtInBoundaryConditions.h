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
     * A clamp boundary condition that sets the ghost cells of an edge to a value for each cell
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     * @note The values of an edge are set through values(dimension, edge) before the boundary condition is applied, and can be changed between applications
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class arrayClampBC : public singleVariableBC<T, rank, tag>
    {
        portableWrapper::portableArrayManager *manager;
        //Per edge, indexed as boundaryRanges
        std::array<portableWrapper::portableArray<T, rank, tag>, 2*rank> edgeValues;
        std::array<bool, 2*rank> allocated{};
    public:
        /**
         * Constructor
         * @param varDef The variable definition
         * @param manager The array manager that allocates the values of the edges
         */
        arrayClampBC(const SAMS::variableDef &varDef, portableWrapper::portableArrayManager &manager)
            : singleVariableBC<T, rank, tag>(varDef), manager(&manager)
        {
        }

        /**
         * The values of the ghost cells of an edge, with the indices of the variable. Allocated the first time it is asked for
         * @param dimension The dimension of the edge
         * @param edge The edge (lower or upper)
         */
        portableWrapper::portableArray<T, rank, tag> &values(int dimension, SAMS::domain::edges edge)
        {
            int index = dimension * 2 + ((edge == SAMS::domain::edges::lower) ? 0 : 1);
            if (!allocated[index]){
                std::apply([&](auto... rangeArgs){ manager->allocate(edgeValues[index], rangeArgs...); }, this->boundaryRanges[index]);
                allocated[index] = true;
            }
            return edgeValues[index];
        }

        /**
         * Apply the boundary condition (implements base class method)
         * @param dimension The dimension to apply the boundary condition on
         * @param edge The edge to apply the boundary condition on (lower or upper)
         */
        void apply(int dimension, SAMS::domain::edges edge) override
        {
            int index = dimension * 2 + ((edge == SAMS::domain::edges::lower) ? 0 : 1);
            portableWrapper::portableArray<T, rank, tag> slice = std::apply([this](auto... rangeArgs){
                return this->variable(rangeArgs...);
            }, this->boundaryRanges[index]);
            portableWrapper::assign(slice, edgeValues[index]);
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
         * @param mirrorValue The value on the boundary
         * @note The first ghost point and the last domain point may be the same point for some staggers
         * @note Reflects about mirrorValue. A boundary on a grid point is set to mirrorValue, otherwise the average across the edge is mirrorValue
         */
        mirrorHelper(portableWrapper::portableArray<T, rank, tag> &Array, int dim, T_indexType lastDomainPoint, T_indexType firstGhostPoint, T mirrorValue)
            : Array(Array), dim(dim), lastDomainPoint(lastDomainPoint), firstGhostPoint(firstGhostPoint), mirrorValue(mirrorValue)
        {
        }

        template<typename... Params>
        FUNCTORMETHODPREFIX void operator()(Params... params) const {

            TUPLE<Params...> dst(params...), src(params...);

            T_indexType distance = portableWrapper::getTupleElement(dst, dim) - firstGhostPoint;
            //Boundary on a grid point
            if (distance == 0 && lastDomainPoint == firstGhostPoint) {
                APPLY(Array, dst) = mirrorValue;
                return;
            }
            portableWrapper::getTupleElement(src, dim) = lastDomainPoint - distance;
            APPLY(Array, dst) = static_cast<T>(2) * mirrorValue - APPLY(Array, src);
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
    class fixedMirrorBC : public singleVariableBC<T, rank, tag>
    {
        T mirrorValue;
    public:
        /**
         * Constructor
         * @param varDef The variable definition
         * @param mirrorValue The value on the boundary
         */
        fixedMirrorBC(const SAMS::variableDef &varDef, T mirrorValue)
            : singleVariableBC<T, rank, tag>(varDef), mirrorValue(mirrorValue)
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
            mirrorHelper<T, rank, tag> helper(var, dimension, lastDomainPoint, firstGhostPoint, mirrorValue);
            std::apply([&helper](auto... params){
                portableWrapper::applyKernel(helper, params...);
            }, ranges);
        }
    };

    /**
     * Helper functor to perform mirror boundary conditions about a value for each ghost cell
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class arrayMirrorHelper
    {
        private:
            portableWrapper::portableArray<T, rank, tag> Array, mirrorValues;
            int dim;
            T_indexType lastDomainPoint, firstGhostPoint;
        public:

        /**
         * Constructor
         * @param Array The array to apply the mirror boundary condition to
         * @param mirrorValues The values on the boundary, at the indices of the ghost cells
         * @param dim The dimension to apply the boundary condition on
         * @param lastDomainPoint The last point in the domain (to mirror from)
         * @param firstGhostPoint The first ghost point (to mirror to)
         * @note The first ghost point and the last domain point may be the same point for some staggers
         * @note Reflects about the value of each ghost cell, as mirrorHelper does about mirrorValue
         */
        arrayMirrorHelper(portableWrapper::portableArray<T, rank, tag> &Array, portableWrapper::portableArray<T, rank, tag> &mirrorValues, int dim, T_indexType lastDomainPoint, T_indexType firstGhostPoint)
            : Array(Array), mirrorValues(mirrorValues), dim(dim), lastDomainPoint(lastDomainPoint), firstGhostPoint(firstGhostPoint)
        {
        }

        template<typename... Params>
        FUNCTORMETHODPREFIX void operator()(Params... params) const {

            TUPLE<Params...> dst(params...), src(params...);

            T_indexType distance = portableWrapper::getTupleElement(dst, dim) - firstGhostPoint;
            //Boundary on a grid point
            if (distance == 0 && lastDomainPoint == firstGhostPoint) {
                APPLY(Array, dst) = APPLY(mirrorValues, dst);
                return;
            }
            portableWrapper::getTupleElement(src, dim) = lastDomainPoint - distance;
            APPLY(Array, dst) = static_cast<T>(2) * APPLY(mirrorValues, dst) - APPLY(Array, src);
        }
    };

    /**
     * A mirror boundary condition that reflects the variable about a value for each ghost cell
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     * @note uses the arrayMirrorHelper functor. The values of an edge are set through values(dimension, edge) before the boundary condition is applied, and can be changed between applications
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class arrayMirrorBC : public singleVariableBC<T, rank, tag>
    {
        portableWrapper::portableArrayManager *manager;
        //Per edge, indexed as boundaryRanges
        std::array<portableWrapper::portableArray<T, rank, tag>, 2*rank> edgeValues;
        std::array<bool, 2*rank> allocated{};
    public:
        /**
         * Constructor
         * @param varDef The variable definition
         * @param manager The array manager that allocates the values of the edges
         */
        arrayMirrorBC(const SAMS::variableDef &varDef, portableWrapper::portableArrayManager &manager)
            : singleVariableBC<T, rank, tag>(varDef), manager(&manager)
        {
        }

        /**
         * The values on the boundary for the ghost cells of an edge, with the indices of the variable. Allocated the first time it is asked for
         * @param dimension The dimension of the edge
         * @param edge The edge (lower or upper)
         */
        portableWrapper::portableArray<T, rank, tag> &values(int dimension, SAMS::domain::edges edge)
        {
            int index = dimension * 2 + ((edge == SAMS::domain::edges::lower) ? 0 : 1);
            if (!allocated[index]){
                std::apply([&](auto... rangeArgs){ manager->allocate(edgeValues[index], rangeArgs...); }, this->boundaryRanges[index]);
                allocated[index] = true;
            }
            return edgeValues[index];
        }

        void apply(int dimension, SAMS::domain::edges edge) override
        {
            int index = dimension * 2 + ((edge == SAMS::domain::edges::lower) ? 0 : 1);
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
            arrayMirrorHelper<T, rank, tag> helper(var, edgeValues[index], dimension, lastDomainPoint, firstGhostPoint);
            std::apply([&helper](auto... params){
                portableWrapper::applyKernel(helper, params...);
            }, this->boundaryRanges[index]);
        }
    };

    /**
     * Helper functor to perform floating mirror boundary conditions
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class floatingMirrorHelper
    {
        private:
            portableWrapper::portableArray<T, rank, tag> Array;
            int dim;
            T_indexType lastDomainPoint, firstGhostPoint, innerPoint;
            T lastWeight, innerWeight;
        public:

        /**
         * Constructor
         * @param Array The array to apply the mirror boundary condition to
         * @param dim The dimension to apply the boundary condition on
         * @param lastDomainPoint The last point in the domain (to mirror from)
         * @param firstGhostPoint The first ghost point (to mirror to)
         * @param innerPoint The domain point next to the last point in the domain
         * @param lastWeight The weight of the last domain point in the value on the boundary
         * @param innerWeight The weight of the inner point in the value on the boundary
         * @note The first ghost point and the last domain point may be the same point for some staggers
         * @note Reflects about the value on the boundary, which is lastWeight * Array(lastDomainPoint) + innerWeight * Array(innerPoint)
         */
        floatingMirrorHelper(portableWrapper::portableArray<T, rank, tag> &Array, int dim, T_indexType lastDomainPoint, T_indexType firstGhostPoint, T_indexType innerPoint, T lastWeight, T innerWeight)
            : Array(Array), dim(dim), lastDomainPoint(lastDomainPoint), firstGhostPoint(firstGhostPoint), innerPoint(innerPoint), lastWeight(lastWeight), innerWeight(innerWeight)
        {
        }

        template<typename... Params>
        FUNCTORMETHODPREFIX void operator()(Params... params) const {

            TUPLE<Params...> dst(params...), src(params...), last(params...), inner(params...);

            T_indexType distance = portableWrapper::getTupleElement(dst, dim) - firstGhostPoint;
            //Boundary on a grid point keeps its value
            if (distance == 0 && lastDomainPoint == firstGhostPoint) return;
            portableWrapper::getTupleElement(src, dim) = lastDomainPoint - distance;
            portableWrapper::getTupleElement(last, dim) = lastDomainPoint;
            portableWrapper::getTupleElement(inner, dim) = innerPoint;
            const T edgeValue = lastWeight * APPLY(Array, last) + innerWeight * APPLY(Array, inner);
            APPLY(Array, dst) = static_cast<T>(2) * edgeValue - APPLY(Array, src);
        }
    };

    /**
     * A mirror boundary condition that reflects the variable about its value on the boundary, found from the domain
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     * @note uses the floatingMirrorHelper functor. A boundary on a grid point keeps its value, otherwise the value on the boundary is extrapolated linearly from the last two domain points
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class floatingMirrorBC : public singleVariableBC<T, rank, tag>
    {
    public:
        floatingMirrorBC(const SAMS::variableDef &varDef)
            : singleVariableBC<T, rank, tag>(varDef)
        {
        }

        void apply(int dimension, SAMS::domain::edges edge) override
        {
            int edgeIndex = (edge == SAMS::domain::edges::lower) ? 0 : 1;
            auto &ranges = this->boundaryRanges[dimension * 2 + edgeIndex];
            auto &var = this->variable;
            T_indexType lastDomainPoint, firstGhostPoint, innerPoint;
            auto &dimInfo = this->varDef.getDimension(dimension);
            if (edge == SAMS::domain::edges::lower){
                lastDomainPoint = dimInfo.getLocalDomainLB();
                firstGhostPoint = dimInfo.getLocalNonDomainUB(SAMS::domain::edges::lower);
                innerPoint = lastDomainPoint + 1;
            } else {
                lastDomainPoint = dimInfo.getLocalDomainUB();
                firstGhostPoint = dimInfo.getLocalNonDomainLB(SAMS::domain::edges::upper);
                innerPoint = lastDomainPoint - 1;
            }
            const bool onGridPoint = lastDomainPoint == firstGhostPoint;
            const T lastWeight = onGridPoint ? static_cast<T>(1) : static_cast<T>(1.5);
            const T innerWeight = onGridPoint ? static_cast<T>(0) : static_cast<T>(-0.5);
            floatingMirrorHelper<T, rank, tag> helper(var, dimension, lastDomainPoint, firstGhostPoint, innerPoint, lastWeight, innerWeight);
            std::apply([&helper](auto... params){
                portableWrapper::applyKernel(helper, params...);
            }, ranges);
        }
    };

    /**
     * Helper functor to perform symmetric reflecting boundary conditions
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class symmetricHelper
    {
        private:
            portableWrapper::portableArray<T, rank, tag> Array;
            int dim;
            T_indexType lastDomainPoint, firstGhostPoint;
        public:

        /**
         * Constructor
         * @param Array The array to apply the boundary condition to
         * @param dim The dimension to apply the boundary condition on
         * @param lastDomainPoint The last point in the domain (to reflect from)
         * @param firstGhostPoint The first ghost point (to reflect to)
         * @note The first ghost point and the last domain point may be the same point for some staggers
         */
        symmetricHelper(portableWrapper::portableArray<T, rank, tag> &Array, int dim, T_indexType lastDomainPoint, T_indexType firstGhostPoint)
            : Array(Array), dim(dim), lastDomainPoint(lastDomainPoint), firstGhostPoint(firstGhostPoint)
        {
        }

        template<typename... Params>
        FUNCTORMETHODPREFIX void operator()(Params... params) const {

            TUPLE<Params...> dst(params...), src(params...);

            T_indexType distance = portableWrapper::getTupleElement(dst, dim) - firstGhostPoint;
            portableWrapper::getTupleElement(src, dim) = lastDomainPoint - distance;
            APPLY(Array, dst) = APPLY(Array, src);
        }
    };

    /**
     * A reflecting boundary condition that copies the variable symmetrically across the boundary
     * @tparam T The data type of the variable
     * @tparam rank The rank of the variable
     * @tparam tag The memory space of the variable
     * @note uses the symmetricHelper functor
     */
    template<typename T, int rank, portableWrapper::arrayTags tag = portableWrapper::arrayTags::accelerated>
    class simpleSymmetricBC : public singleVariableBC<T, rank, tag>
    {
    public:
        simpleSymmetricBC(const SAMS::variableDef &varDef)
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
            symmetricHelper<T, rank, tag> helper(var, dimension, lastDomainPoint, firstGhostPoint);
            std::apply([&helper](auto... params){
                portableWrapper::applyKernel(helper, params...);
            }, ranges);
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
