#ifndef LARE_BOUNDARY_HELPERS_H
#define LARE_BOUNDARY_HELPERS_H

#include "lareic.h"
#include <tuple>

/**
 * Operations on the ghost cells at one edge of the domain, shared by the classes of boundary condition.
 * Cell centred dimensions have real cells 1 to n and ghost cells -1, 0 and n+1, n+2. Node centred dimensions have
 * boundary nodes 0 and n and ghost nodes -2, -1 and n+1, n+2
 */
namespace LARE::boundaryHelpers
{
    namespace pw = portableWrapper;

    /**
     * Functor to assign the negative of one array to another
     */
    template<typename T, int rank, pw::arrayTags tag>
    struct assignNegatedArray
    {
        using pa = pw::portableArray<T, rank, tag>;
        pa dest;
        pa src;

        FUNCTORMETHODPREFIX INLINE assignNegatedArray(pa &dest, const pa &src)
            : dest(dest), src(src) {}

        template<typename... T_indices>
        FUNCTORMETHODPREFIX INLINE void operator()(T_indices... indices) const
        {
            dest.getZB(indices...) = -src.getZB(indices...);
        }
    };

    /**
     * Assign the negative of one array to another
     */
    template<typename T, int rank, pw::arrayTags tag>
    void assignNegated(pw::portableArray<T, rank, tag> dest, const pw::portableArray<T, rank, tag> &src)
    {
        using namespace portableWrapper;
        std_N_ary_tuple_type_t<Range, rank> ranges;
        pw::detail::arrayToRangesZB<0, T, rank, tag>(src, ranges);
        auto tpl = std::tuple_cat(std::make_tuple(assignNegatedArray<T, rank, tag>(dest, src)), ranges);
        std::apply([](auto&&... args) {
            constexpr executionSpace space = getDefaultExecutionSpace(tag);
            applyKernel<space>(args...);
        }, tpl);
    }

    /**
     * The part of an array at an index, or a range of indices, in one dimension
     */
    template<typename T_array, typename T_index>
    auto slab(T_array &array, int dimension, T_index index)
    {
        if (dimension == 0) return array(index, pw::Range(), pw::Range());
        if (dimension == 1) return array(pw::Range(), index, pw::Range());
        return array(pw::Range(), pw::Range(), index);
    }

    /**
     * The number of real cells in a dimension
     */
    template<typename T_data>
    SIGNED_INDEX_TYPE cells(const T_data &data, int dimension)
    {
        if (dimension == 0) return data.nx;
        if (dimension == 1) return data.ny;
        return data.nz;
    }

    /**
     * The inflow state at an edge
     */
    inline const inflowState &inflow(const icData &icd, int dimension, SAMS::domain::edges edge)
    {
        return icd.inflow[dimension][edge == SAMS::domain::edges::upper ? 1 : 0];
    }

    /**
     * Mirror a cell centred dimension about the boundary
     */
    template<typename T_array>
    void mirrorCell(T_array &array, int dimension, SAMS::domain::edges edge, SIGNED_INDEX_TYPE n)
    {
        if (edge == SAMS::domain::edges::lower){
            pw::assign(slab(array, dimension, 0), slab(array, dimension, 1));
            pw::assign(slab(array, dimension, -1), slab(array, dimension, 2));
        } else {
            pw::assign(slab(array, dimension, n + 1), slab(array, dimension, n));
            pw::assign(slab(array, dimension, n + 2), slab(array, dimension, n - 1));
        }
    }

    /**
     * Mirror a node centred dimension about the boundary node
     */
    template<typename T_array>
    void mirrorNode(T_array &array, int dimension, SAMS::domain::edges edge, SIGNED_INDEX_TYPE n)
    {
        if (edge == SAMS::domain::edges::lower){
            pw::assign(slab(array, dimension, -1), slab(array, dimension, 1));
            pw::assign(slab(array, dimension, -2), slab(array, dimension, 2));
        } else {
            pw::assign(slab(array, dimension, n + 1), slab(array, dimension, n - 1));
            pw::assign(slab(array, dimension, n + 2), slab(array, dimension, n - 2));
        }
    }

    /**
     * Mirror a node centred dimension about the boundary node with a change of sign, so that the boundary node is zero
     */
    template<typename T_array>
    void mirrorNodeNegated(T_array &array, int dimension, SAMS::domain::edges edge, SIGNED_INDEX_TYPE n)
    {
        if (edge == SAMS::domain::edges::lower){
            pw::assign(slab(array, dimension, 0), 0.0);
            assignNegated(slab(array, dimension, -1), slab(array, dimension, 1));
            assignNegated(slab(array, dimension, -2), slab(array, dimension, 2));
        } else {
            pw::assign(slab(array, dimension, n), 0.0);
            assignNegated(slab(array, dimension, n + 1), slab(array, dimension, n - 1));
            assignNegated(slab(array, dimension, n + 2), slab(array, dimension, n - 2));
        }
    }

    /**
     * Copy the last real cell of a cell centred dimension into the ghost cells
     */
    template<typename T_array>
    void copyCell(T_array &array, int dimension, SAMS::domain::edges edge, SIGNED_INDEX_TYPE n)
    {
        if (edge == SAMS::domain::edges::lower){
            pw::assign(slab(array, dimension, 0), slab(array, dimension, 1));
            pw::assign(slab(array, dimension, -1), slab(array, dimension, 1));
        } else {
            pw::assign(slab(array, dimension, n + 1), slab(array, dimension, n));
            pw::assign(slab(array, dimension, n + 2), slab(array, dimension, n));
        }
    }

    /**
     * Copy the boundary node of a node centred dimension into the ghost nodes
     */
    template<typename T_array>
    void copyNode(T_array &array, int dimension, SAMS::domain::edges edge, SIGNED_INDEX_TYPE n)
    {
        if (edge == SAMS::domain::edges::lower){
            pw::assign(slab(array, dimension, -1), slab(array, dimension, 0));
            pw::assign(slab(array, dimension, -2), slab(array, dimension, 0));
        } else {
            pw::assign(slab(array, dimension, n + 1), slab(array, dimension, n));
            pw::assign(slab(array, dimension, n + 2), slab(array, dimension, n));
        }
    }

    /**
     * Set the ghost cells of a cell centred dimension to a value
     */
    template<typename T_array>
    void fillCell(T_array &array, int dimension, SAMS::domain::edges edge, SIGNED_INDEX_TYPE n, SAMS::T_dataType value)
    {
        if (edge == SAMS::domain::edges::lower){
            pw::assign(slab(array, dimension, pw::Range(-1, 0)), value);
        } else {
            pw::assign(slab(array, dimension, pw::Range(n + 1, n + 2)), value);
        }
    }

    /**
     * Set the boundary node and the ghost nodes of a node centred dimension to a value
     */
    template<typename T_array>
    void fillNode(T_array &array, int dimension, SAMS::domain::edges edge, SIGNED_INDEX_TYPE n, SAMS::T_dataType value)
    {
        if (edge == SAMS::domain::edges::lower){
            pw::assign(slab(array, dimension, pw::Range(-2, 0)), value);
        } else {
            pw::assign(slab(array, dimension, pw::Range(n, n + 2)), value);
        }
    }

}

#endif
