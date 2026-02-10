#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "array.h"

namespace portableWrapper
{
  namespace detail{
    /**
     * Functor to assign a value to a specific element in the array.
     * Has to be a functor because HIP/CUDA don't support variadic lambdas
     */
    template<typename T, int rank, arrayTags arrayTag, typename T2>
      struct assignValue{
        using pa = portableArray<T, rank, arrayTag>;
        pa array;
        T2 value;
        FUNCTORMETHODPREFIX INLINE assignValue(pa arr, T value)
          : array(arr), value(value) {}

        template<typename... T_indices>
          FUNCTORMETHODPREFIX INLINE void operator()(T_indices... indices) const
          {
            array.getZB(indices...) = value;
          }
      };

    /**
     * Functor to assign one portableArray to another.
     */
    template<typename T, int rank, arrayTags arrayTag>
      struct assignArray{
        using pa = portableArray<T, rank, arrayTag>;
        pa dest;
        pa src;

        FUNCTORMETHODPREFIX INLINE assignArray(pa &dest, const pa &src)
          : dest(dest), src(src) {}

        template<typename... T_indices>
          FUNCTORMETHODPREFIX INLINE void operator()(T_indices... indices) const
          {
            dest.getZB(indices...) = src.getZB(indices...); // Use the overloaded operator() to assign the value
          }
      };

    /**
     * Functor for returning a value from an array
     * Used for reductions like minval, maxval, sum, etc.
     */
    template<typename T, int rank, arrayTags arrayTag>
      struct returnArrayValue{
        using pa = portableArray<T, rank, arrayTag>;
        pa src;

        FUNCTORMETHODPREFIX INLINE returnArrayValue(const pa &src)
          : src(src) {}

        template<typename... T_indices>
          FUNCTORMETHODPREFIX INLINE T operator()(T_indices... indices) const
          {
            return src.getZB(indices...);
          }
      };
    }

  /**
   * Assigns one portableArray to another.
   * This function checks if the source and destination arrays have the same number of elements
   * and then applies the assignment in parallel using the provided ranges.
   */
  template< bool fence=false, typename T=void, int rank=0, arrayTags tag=arrayTags::accelerated>
    HOSTFLATTEN void assign(portableArray<T, rank, tag> dest, const portableArray<T, rank, tag> &src) {
      if ((&dest) != (&src))
      {
        if (dest.getElements() != src.getElements()) {
          throw std::runtime_error("Source and destination arrays must have the same number of elements.");
        }
        std_N_ary_tuple_type_t<Range,rank> ranges;
        detail::arrayToRangesZB<0,T,rank,tag>(src, ranges);
        auto tpl = std::tuple_cat(
            std::make_tuple(detail::assignArray(dest, src)),
            ranges
            );
        std::apply([](auto&&... args) {
            constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
            portableWrapper::applyKernel<space>(args...);
         }, tpl);
        if constexpr(fence) portableWrapper::fence();
      }
    }

  template< typename T, int rank, arrayTags tag, typename T2>
    HOSTFLATTEN void assign(portableArray<T, rank, tag> dest, const T2 &src) {
      static_assert(std::is_convertible_v<T,T2>,"Source type must be convertible to destination type in assignment");
      std_N_ary_tuple_type_t<Range,rank> ranges;
      detail::arrayToRangesZB(dest, ranges);

      auto tpl = std::tuple_cat(
          std::make_tuple(detail::assignValue<T,rank,tag,T2>(dest, src)),
          ranges
          );
      std::apply([](auto&&... args) {
          constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
          portableWrapper::applyKernel<space>(args...);
          }, tpl);
    }


  /**
   * Returns the maximum value in the portableArray.
   */
  template<typename T, int rank, arrayTags tag>
    UNREPEATED T maxval(const portableWrapper::portableArray<T, rank, tag> &array) {
      std_N_ary_tuple_type_t<Range, rank> ranges;
      detail::arrayToRangesZB(array, ranges);

      auto maxValFunc = LAMBDA(T &a, const T &b) {
        a = portableWrapper::max(a, b);
      };

      auto tpl = std::tuple_cat(
          std::make_tuple(detail::returnArrayValue<T, rank, tag>(array)),
          std::make_tuple(maxValFunc),
          std::make_tuple(std::numeric_limits<T>::lowest()),
          ranges
          );

      return std::apply([](auto&&... args) {
          constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
          return applyReduction<space>(args...);
          }, tpl);
    }

  /**
   * Returns the minimum value in the portableArray.
   */
  template<typename T, int rank, arrayTags tag>
    UNREPEATED T minval(const portableWrapper::portableArray<T, rank, tag> &array) {
      std_N_ary_tuple_type_t<Range, rank> ranges;
      detail::arrayToRangesZB(array, ranges);

      auto minValFunc = LAMBDA(T &a, const T &b) {
        a = portableWrapper::min(a, b);
      };

      auto tpl = std::tuple_cat(
          std::make_tuple(detail::returnArrayValue<T, rank, tag>(array)),
          std::make_tuple(minValFunc),
          std::make_tuple(std::numeric_limits<T>::max()),
          ranges
          );

      return std::apply([](auto&&... args) {
          constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
          return applyReduction<space>(args...);
          }, tpl);
    }

  /**
   * Returns the sum of all elements in the portableArray.
   */
  template<typename T, int rank, arrayTags tag>
    UNREPEATED T sum(const portableWrapper::portableArray<T, rank, tag> &array) {
      std_N_ary_tuple_type_t<Range, rank> ranges;
      detail::arrayToRangesZB(array, ranges);

      auto sumValFunc = LAMBDA(T &a, const T &b) {
        a +=b;
      };

      auto tpl = std::tuple_cat(
          std::make_tuple(detail::returnArrayValue<T, rank, tag>(array)),
          std::make_tuple(sumValFunc),
          std::make_tuple(T{}),
          ranges
          );

      return std::apply([](auto&&... args) {
          constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
          return applyReduction<space>(args...);
          }, tpl);
    }

}

#endif // ALGORITHM_H