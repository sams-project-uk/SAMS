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
#ifndef UTILS_H
#define UTILS_H

#include "range.h"
#include "defs.h"
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace portableWrapper
{

namespace{
   /**
    * Count the number of Range obects in a parameter pack
    */
    template<typename... Args>
      inline DEVICEPREFIX constexpr UNSIGNED_INDEX_TYPE countRanges() {
        return (... + std::is_same_v<Args, Range>);
      }

      /**
       * Portable versions of the std::tuple features.
       */
    namespace portableTuple{

      /**
       *  Partial tuple implementation that can be used on all backends
       */
      struct tuple_empty {};

      /**
       *  Tuple class template that can hold multiple values of different types.
       *  It is a recursive structure that builds a tuple from its head and tail.
       */
      template <typename Head, typename... Tail>
        struct tuple : tuple<Tail...> {
          Head value;
          FUNCTORMETHODPREFIX tuple(Head v, Tail... tail) : tuple<Tail...>(tail...), value(v) {}
          FUNCTORMETHODPREFIX tuple() : tuple<Tail...>(), value() {}
        };


      /**
       *   Specialization of tuple for the case with only one element.
       */
      template <typename Head>
        struct tuple<Head> : tuple_empty {
          Head value;
          FUNCTORMETHODPREFIX tuple(Head v) : value(v) {}
          FUNCTORMETHODPREFIX tuple() : value() {}
        };

      /**
       * Implementation of make_tuple function that creates a tuple from given arguments.
       * Mostly just syntactic sugar to use type inference to make the tuple type.
       */
      template <typename... Args>
        FUNCTORMETHODPREFIX tuple<Args...> make_tuple(Args... args) {
          return tuple<Args...>(args...);
        }

      /**
       * Reimplementation of std::get for our tuple type.
       */
      template <std::size_t N, typename Head, typename... Tail>
        inline FUNCTORMETHODPREFIX auto& get(tuple<Head, Tail...>& t) {
          if constexpr (N == 0) {
            return t.value;
          } else {
            return get<N - 1>(static_cast<tuple<Tail...>&>(t));
          }
        }

      /**
       * Const version of get for our tuple type.
       */
      template <std::size_t N, typename Head, typename... Tail>
        inline FUNCTORMETHODPREFIX const auto& get(const tuple<Head, Tail...>& t) {
          if constexpr (N == 0) {
            return t.value;
          } else {
            return get<N - 1>(static_cast<const tuple<Tail...>&>(t));
          }
        }

      /**
       * Reimplementation of std::tuple_element
       */
      template <std::size_t N, typename Tuple>
        struct tuple_element;

      template <std::size_t N, typename Head, typename... Tail>
        struct tuple_element<N, tuple<Head, Tail...>> {
          using type = typename tuple_element<N - 1, tuple<Tail...>>::type;
        };

      template <typename Head, typename... Tail>
        struct tuple_element<0, tuple<Head, Tail...>> {
          using type = Head;
        };

      template <std::size_t N, typename Tuple>
        using tuple_element_t = typename tuple_element<N, Tuple>::type;

      /**
       * Base case for tuple_size specialization.
       */
      template <typename Tuple>
        struct tuple_size;

      /**
       * Specialization of tuple_size for our tuple type.
       * It calculates the size of the tuple at compile time.
       */
      template <typename... Args>
        struct tuple_size<tuple<Args...>> {
          static constexpr std::size_t value = sizeof...(Args);
        };

			template<typename T>
			constexpr size_t tuple_size_v=tuple_size<T>::value;

      //Implementation of std::declval

      template <typename T>
        inline FUNCTORMETHODPREFIX T&& declval() noexcept {
          return static_cast<T&&>(*(T*)nullptr);
        }

      /**
       * std::apply only works in CUDA with the --expt-relaxed-constexpr flag
       * This does the same job as std::apply but is a device function
       * that can be used in CUDA kernels.
       */
      template <typename F, typename Tuple, std::size_t... I>
        inline FUNCTORMETHODPREFIX auto apply_impl(F &&f, Tuple &&t, std::index_sequence<I...>)
        {
          return f(get<I>(t)...);
        }

      template <typename F, typename Tuple>
        inline FUNCTORMETHODPREFIX auto apply(F &&f, Tuple &&t)
        {
          constexpr std::size_t N = tuple_size<std::remove_reference_t<Tuple>>::value;
          return apply_impl(
              std::forward<F>(f), t, std::make_index_sequence<N>{});
        }

        /**
         * Forward declaration of tuple_cat_impl.
         */
        template <typename... Tuples>
        struct tuple_cat_impl;

        /**
         * Specialization of tuple_cat_impl for two tuples.
         * This concatenates two tuples into one.
         * Ultimately all concatenations are done by this specialization
         */
        template <typename... Args1, typename... Args2>
        struct tuple_cat_impl<tuple<Args1...>, tuple<Args2...>> {
          public:
          //Construct type for two tuples concatenated together
          using type = tuple<Args1..., Args2...>;
          static FUNCTORMETHODPREFIX type cat(const tuple<Args1...>& t1, const tuple<Args2...>& t2) {
            return cat_impl(t1, t2, std::make_index_sequence<sizeof...(Args1)>(), std::make_index_sequence<sizeof...(Args2)>());
          }
        private:
        //Normally would put private above public, but here want type to be public
        //Use a fold operation on std::get with the two index sequences
          template <std::size_t... I1, std::size_t... I2>
          static FUNCTORMETHODPREFIX type cat_impl(const tuple<Args1...>& t1, const tuple<Args2...>& t2,
                  std::index_sequence<I1...>, std::index_sequence<I2...>) 
          {
           //This uses a fold to expand all of the elements of t1 and then all of the elements of t2                                       
            return type(get<I1>(t1)..., get<I2>(t2)...);
          }
        };

        /**
         *  Recursive implementation of tuple_cat for multiple tuples.
         */
        template <typename Tuple1, typename Tuple2, typename... Rest>
        struct tuple_cat_impl<Tuple1, Tuple2, Rest...> {
          using type = typename tuple_cat_impl<typename tuple_cat_impl<Tuple1, Tuple2>::type, Rest...>::type;
          static FUNCTORMETHODPREFIX type cat(const Tuple1& t1, const Tuple2& t2, const Rest&... rest) {
            auto t12 = tuple_cat_impl<Tuple1, Tuple2>::cat(t1, t2);
            return tuple_cat_impl<decltype(t12), Rest...>::cat(t12, rest...);
          }
        };

        /**
         * Concatenate multiple tuples into one tuple.
         * This is the public interface for tuple concatenation.
         * It uses the tuple_cat_impl to perform the actual concatenation.
         */
        template <typename... Tuples>
        inline FUNCTORMETHODPREFIX auto tuple_cat(const Tuples&... tuples) {
          return tuple_cat_impl<Tuples...>::cat(tuples...);
        }

      /** Return a tuple with it's last element removed
       */
      template <int level=0, typename... Args>
        inline FUNCTORMETHODPREFIX auto tupleRemoveLast(const tuple<Args...>& t) {
          if constexpr (level < sizeof...(Args) - 2) {
            return tuple_cat(
                make_tuple(get<level>(t)),
                tupleRemoveLast<level + 1>(t));
          } else {
            return make_tuple(get<level>(t));
          }
        }

    /**
     * Helper to create a tuple type from two tuples types
     */
    template <typename T1, typename T2, typename... Rest>
      using tuple_cat_type_t = typename portableTuple::tuple_cat_impl<T1, T2, Rest...>::type;


    } // namespace portableTuple

    /**
     * Portable version of std::max
     * @brief Returns the maximum of two values.
     * @param v1 First value.
     * @param v2 Second value.
     */
    template<typename T1, typename T2>
    FUNCTORMETHODPREFIX constexpr auto max(const T1&v1, const T2&v2){
      return  v1>v2?v1:v2;
    }

    /**
     * Portable version of std::max that takes an initializer list
     * @brief Returns the maximum of a list of values.
     * @param values An initializer list of values.
     * @return The maximum value from the list.
     */
    template<typename T>
    FUNCTORMETHODPREFIX T max(std::initializer_list<T> values) {
      T maxValue = *values.begin();
      for (const T& value : values) {
        if (value > maxValue) {
          maxValue = value;
        }
      }
      return maxValue;
    }
    
    /**
     * Portable version of std::min
     * @brief Returns the minimum of two values.
     * @param v1 First value.
     * @param v2 Second value.
     */
    template<typename T1, typename T2>
    FUNCTORMETHODPREFIX constexpr auto min(const T1&v1, const T2&v2){
      return  v1<v2?v1:v2;
    }

    /**
     * Portable version of std::min that takes an initializer list
     * @brief Returns the minimum of a list of values.
     * @param values An initializer list of values.
     * @return The minimum value from the list.
     */
    template<typename T>
    FUNCTORMETHODPREFIX T min(std::initializer_list<T> values) {
      T minValue = *values.begin();
      for (const T& value : values) {
        if (value < minValue) {
          minValue = value;
        }
      }
      return minValue;
    }

    /**
     * Call a specified callable with the specified arguments in reverse order.
     */
    template <typename F, typename Tuple, std::size_t... Is>
      inline FUNCTORMETHODPREFIX auto reverseAndCall(F&& f, Tuple&& tup, std::index_sequence<Is...>) {
        return f(GET<sizeof...(Is) - 1 - Is>(tup)...);
      }


      /**
       * Functor that provides a callable that reverses the order of arguments
       * before calling the original function.
       * This is needed because Kokkos orders parallel_for strangely
       * and we need to reverse the order of the arguments to match the
       * original function's expectations when using a CUDA or Thrust Kokkos backend.
       */
    template <typename F>
      class reverseFunctor {
        F f;
        public:
        FUNCTORMETHODPREFIX reverseFunctor(F func) : f(func) {}

        template <typename... Args>
          FUNCTORMETHODPREFIX auto operator()(Args&&... args) const {
            auto tup = MAKETUPLE(std::forward<Args>(args)...);
            return reverseAndCall(f, tup, std::index_sequence_for<Args...>{});
          }
      };

    // Helper to create the functor
    template <typename F>
      inline FUNCTORMETHODPREFIX reverseFunctor<F> makeReverseFunctor(F f) {
        return reverseFunctor<F>(f);
      }
    /**
     * Create an N element tuple of a given type
     * use as N_ary_tuple_type<int,4> to create std::tuple<int,int,int,int>
     */
    template <typename T, int N, typename... REST>
      struct N_ary_tuple_type
      {
        typedef typename N_ary_tuple_type<T, N - 1, T, REST...>::type type;
      };
    template <typename T, typename... REST>
      struct N_ary_tuple_type<T, 0, REST...>
      {
        typedef TUPLE<REST...> type;
      };

    /**
     * Type alias for N_ary_tuple_type that can be used to create a tuple type
     * with a given number of elements of a specific type.
     */
    template <typename T, int N>
      using N_ary_tuple_type_t = typename N_ary_tuple_type<T, N>::type;


      //STD::TUPLE version of N_ary_tuple_type
    template <typename T, int N, typename... REST>
      struct std_N_ary_tuple_type
      {
        typedef typename std_N_ary_tuple_type<T, N - 1, T, REST...>::type type;
      };
    template <typename T, typename... REST>
      struct std_N_ary_tuple_type<T, 0, REST...>
      {
        typedef std::tuple<REST...> type;
      };

    /**
     * Type alias for N_ary_tuple_type that can be used to create a tuple type
     * with a given number of elements of a specific type.
     */
    template <typename T, int N>
      using std_N_ary_tuple_type_t = typename std_N_ary_tuple_type<T, N>::type;      

    // Dummy stubs for OpenMP functions when not using OpenMP
    UNSIGNED_INDEX_TYPE INLINE getOMPMaxThreads()
    {
#ifdef _OPENMP
      return omp_get_max_threads();
#else
      return 1;
#endif
    }
    UNSIGNED_INDEX_TYPE INLINE getOMPThreads()
    {
#ifdef _OPENMP
      return omp_get_num_threads();
#else
      return 1;
#endif
    }
    UNSIGNED_INDEX_TYPE INLINE getOMPThreadID()
    {
#ifdef _OPENMP
      return omp_get_thread_num();
#else
      return 0;
#endif
    }

  };
};

#endif
