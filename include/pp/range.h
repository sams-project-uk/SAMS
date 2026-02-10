#ifndef RANGE_H
#define RANGE_H

#include "defs.h"
#include <stdint.h>
#include <tuple>
#include <type_traits>

namespace portableWrapper
{
    /**
     * Class to represent a range of indices.
     * It can be used to specify the bounds for parallel operations
     * and also the range of indices for the parallelWrapper.
     * Range implements a tuple-like interface to allow structured bindings.
     * This means that we can return Range objects even to users who don't want
     * to use portableWrapper, and they can unpack them using structured bindings.
     */
    struct Range
    {
        SIGNED_INDEX_TYPE lower_bound;
        SIGNED_INDEX_TYPE upper_bound;
        DEVICEPREFIX Range(SIGNED_INDEX_TYPE lb, SIGNED_INDEX_TYPE ub)
            : lower_bound(lb), upper_bound(ub){}
        DEVICEPREFIX Range(SIGNED_INDEX_TYPE nels)
            : lower_bound(1), upper_bound(nels){}
        DEVICEPREFIX Range() 
            : lower_bound(INT64_MIN), 
              upper_bound(INT64_MAX){}
    };

    //This code implements a tuple-like interface for Range. This allows a range object to
    //be unpacked using structured bindings.
    template<std::size_t I>
    DEVICEPREFIX SIGNED_INDEX_TYPE & get(Range &r) noexcept
    {
        static_assert(I < 2, "Index out of bounds in tuple get for Range");
        if constexpr (I == 0)
            return r.lower_bound;
        else
            return r.upper_bound;
    }

    template<std::size_t I>
    DEVICEPREFIX const SIGNED_INDEX_TYPE & get(const Range &r) noexcept
    {
        static_assert(I < 2, "Index out of bounds in tuple get for Range");
        if constexpr (I == 0)
            return r.lower_bound;
        else
            return r.upper_bound;
    }

    template<std::size_t I>
    DEVICEPREFIX SIGNED_INDEX_TYPE&& get(Range &&r) noexcept
    {
        static_assert(I < 2, "Index out of bounds in tuple get for Range");
        if constexpr (I == 0)
            return std::move(r.lower_bound);
        else
            return std::move(r.upper_bound);
    }
}

/**
 * Overload the output stream operator for Range
 */
UNREPEATED std::ostream& operator<<(std::ostream& os, const portableWrapper::Range& range)
{
    os << "[" << range.lower_bound << ", " << range.upper_bound << "]";
    return os;
}

//Implement tuple_size and tuple_element for Range
//This allows structured bindings to be used with Range
namespace std {
    template<> struct tuple_size<portableWrapper::Range> : std::integral_constant<std::size_t, 2> {};
    template<> struct tuple_size<const portableWrapper::Range> : std::integral_constant<std::size_t, 2> {};
    template<> struct tuple_size<volatile portableWrapper::Range> : std::integral_constant<std::size_t, 2> {};
    template<> struct tuple_size<const volatile portableWrapper::Range> : std::integral_constant<std::size_t, 2> {};
    template<> struct tuple_element<0, portableWrapper::Range> {using type = SIGNED_INDEX_TYPE;};
    template<> struct tuple_element<1, portableWrapper::Range> {using type = SIGNED_INDEX_TYPE;};
    template<> struct tuple_element<0, const portableWrapper::Range> {using type = const SIGNED_INDEX_TYPE;};
    template<> struct tuple_element<1, const portableWrapper::Range> {using type = const SIGNED_INDEX_TYPE;};
}



#endif