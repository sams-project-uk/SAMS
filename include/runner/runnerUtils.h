#ifndef SAMS_RUNNERUTILS_H
#define SAMS_RUNNERUTILS_H

#include <type_traits>
#include <tuple>
#include <utility>
#include "pp/callableTraits.h"
#include "pp/utils.h"

namespace SAMS{

        namespace dataPacks{
        template <typename... dataPacks>
        struct multiPack{
            using type = std::tuple<dataPacks...>;
        };

        /**
         * Holds references to data packs that live in the runner. A hook that takes this as a parameter gets it filled from the runner by type
         */
        template <typename... Ts>
        struct packRefs{
            std::tuple<Ts&...> refs;

            packRefs(Ts&... packs) : refs(packs...) {}

            /**
             * Get one of the packs by type
             */
            template <typename T>
            T& get() const {
                return std::get<T&>(refs);
            }

            /**
             * Get one of the packs by index
             */
            template <std::size_t N>
            decltype(auto) get() const {
                return std::get<N>(refs);
            }
        };
    }

    /**
     * Helper class to find out if a class has a member variable called dataPack. If so get it's type. If not return void.
     */
    /**
     * Void class when T does not have a dataPack member
     */
    template<typename T, typename = void>
    struct getDataPackType {
        using type = void;
    };
    /**
     * Specialisation when T has a dataPack member
     */
    template<typename T>
    struct getDataPackType<T, std::void_t<typename T::dataPack>> {
        using type = typename T::dataPack;
    };

    /**
     * Alias for getDataPackType::type
     */
    template<typename T>
    using getDataPackType_t = typename getDataPackType<T>::type;

    /**
     * A std::tuple of the types in a list, with dataPacks::multiPack unpacked and void types ignored
     */
    template<typename... T>
    struct flattenPacks;

    /**
     * Base case for flattenPacks = empty tuple
     */
    template<>
    struct flattenPacks<> {
        using type = std::tuple<>;
    };

    /**
     * Recursive case for flattenPacks
     */
    template<typename T, typename... Rest>
    struct flattenPacks<T, Rest...> {
        using type = decltype(std::tuple_cat(std::declval<std::conditional_t<std::is_void_v<T>, std::tuple<>, std::tuple<T>>>(),
                                             std::declval<typename flattenPacks<Rest...>::type>()));
    };

    /**
     * Specialisation to unpack a tuple and process its types into the flattenPacks
     */
    template<typename... T, typename... Rest>
    struct flattenPacks<std::tuple<T...>, Rest...> {
        static_assert(portableWrapper::alwaysFalse<T...>::value, "Error: Unpacking a tuple directly into uniqueTuple is not allowed. Use dataPacks::multiPack to wrap the tuple instead.");
        using type = typename flattenPacks<T..., Rest...>::type;
    };

    /**
     * Specialisation to unpack dataPacks::multiPack and process its types into the flattenPacks
     */
    template<typename... T, typename... Rest>
    struct flattenPacks<dataPacks::multiPack<T...>, Rest...> {
        using type = typename flattenPacks<T..., Rest...>::type;
    };

    /**
     * The place of a type in the hierarchy used when types that differ only by reference and const are merged.
     * A reference to non-const is highest, then a value, then a reference to const
     */
    template<typename T>
    struct uniqueHierarchy {
        static constexpr int value = std::is_lvalue_reference_v<T> ? (std::is_const_v<std::remove_reference_t<T>> ? 0 : 2) : 1;
    };

    /**
     * A type as it is stored in a unique tuple. A const value becomes a value, and a reference is unchanged
     */
    template<typename T>
    using uniqueStored_t = std::conditional_t<std::is_reference_v<T>, T, std::remove_const_t<T>>;

    /**
     * Is the type at index J of a list higher in the hierarchy than the type at index I? Only types that are the same
     * apart from reference and const are compared. If they are equal in the hierarchy, the later type is higher
     */
    template<typename T_list, std::size_t I, std::size_t J>
    constexpr bool uniqueHigher()
    {
        using Ti = std::tuple_element_t<I, T_list>;
        using Tj = std::tuple_element_t<J, T_list>;
        if constexpr (I == J || !std::is_same_v<std::remove_cvref_t<Ti>, std::remove_cvref_t<Tj>>) {
            return false;
        } else {
            return uniqueHierarchy<Tj>::value > uniqueHierarchy<Ti>::value || (uniqueHierarchy<Tj>::value == uniqueHierarchy<Ti>::value && J > I);
        }
    }

    /**
     * Is any type in a list higher in the hierarchy than the type at index I
     */
    template<typename T_list, std::size_t I, std::size_t... Js>
    constexpr bool uniqueIsLower(std::index_sequence<Js...>)
    {
        return (uniqueHigher<T_list, I, Js>() || ...);
    }

    /**
     * A tuple of the type at index I of a list, stored as it is kept, or an empty tuple if another type is higher
     */
    template<typename T_list, std::size_t I, typename T_all>
    using uniqueKeep_t = std::conditional_t<uniqueIsLower<T_list, I>(T_all{}), std::tuple<>, std::tuple<uniqueStored_t<std::tuple_element_t<I, T_list>>>>;

    /**
     * The tuple of the types of a list that no other type is higher than, in the order of the list
     */
    template<typename T_list, std::size_t... Is>
    auto uniqueBuild(std::index_sequence<Is...>) -> decltype(std::tuple_cat(std::declval<uniqueKeep_t<T_list, Is, std::make_index_sequence<sizeof...(Is)>>>()...));

    /**
     * Remove the types of a list that are lower in the hierarchy than a type that is the same apart from reference and const
     */
    template<typename T_list>
    struct uniqueFilter;

    template<typename... Ts>
    struct uniqueFilter<std::tuple<Ts...>> {
        using type = decltype(uniqueBuild<std::tuple<Ts...>>(std::make_index_sequence<sizeof...(Ts)>{}));
    };

    /**
     * A tuple of the unique types in a list, with dataPacks::multiPack unpacked and void types ignored.
     * Types that are the same apart from reference and const are merged. A reference to non-const is kept in
     * preference to a value, and a value in preference to a reference to const. A const value is stored as a value.
     * Of equal types the last is kept, and the kept types keep the order of the list.
     */
    template<typename... T>
    struct uniqueTuple {
        using type = typename uniqueFilter<typename flattenPacks<T...>::type>::type;
    };

    /**
     * Alias for uniqueTuple::type
     */
    template<typename... T>
    using uniqueTuple_t = typename uniqueTuple<T...>::type;

    /**
     * The packRefs of the data packs in a std::tuple
     */
    template<typename T_tuple>
    struct packRefsFromTuple;

    template<typename... Ts>
    struct packRefsFromTuple<std::tuple<Ts...>> {
        using type = dataPacks::packRefs<Ts...>;
    };

    /**
     * The packRefs of a dataPack, which is unpacked in the same way as the runner unpacks it
     */
    template<typename T_pack>
    using packRefsOf_t = typename packRefsFromTuple<uniqueTuple_t<T_pack>>::type;

    /**
     * Helper class to get the datapacks from a list of simulations and create a unique tuple of datapack types
     */
    template<typename... Sims>
    struct dataPackTuple {
        public:
        using type = uniqueTuple_t<getDataPackType_t<Sims>...>;
    };

    template<typename... Sims>
    using dataPackTuple_t = typename dataPackTuple<Sims...>::type;

    /**
     * Helper class to return a tuple type lacking the first N elements
     */
    template<std::size_t N, typename T_tuple, typename=void>
    struct tupleTail;

    /**
     * Specialisation for N=0, return the full tuple
     */
    template<typename... Ts>
    struct tupleTail<0, std::tuple<Ts...>,void> {
        using type = std::tuple<Ts...>;
    };

    /**
     * Recursive case to remove the first element
     */
    template<std::size_t N, typename T_first, typename... T_rest>
    struct tupleTail<N, std::tuple<T_first, T_rest...>, std::enable_if_t<(N>0), void>> {
        static_assert(N <= sizeof...(T_rest)+1, "Error: N is larger than the tuple size in tupleTail");
        using type = typename tupleTail<N-1, std::tuple<T_rest...>>::type;
    };


    template<std::size_t N, typename T_tuple>
    using tupleTail_t = typename tupleTail<N, T_tuple>::type;

    /**
     * Get the index of a type in a tuple
     * If the type is not found, returns -1
     */
    template<typename T, typename T_tuple, std::size_t level=0>
    struct tupleTypeIndex {
        static constexpr int64_t value = []() {
            if constexpr (level >= std::tuple_size_v<T_tuple>) {
                return -1; // Not found
            } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::remove_cvref_t<std::tuple_element_t<level, T_tuple>>>) {
                return level;
            } else {
                return tupleTypeIndex<T, T_tuple, level + 1>::value;
            }
        }();
    };

    template<typename T, typename T_tuple>
    static constexpr std::int64_t tupleTypeIndex_v = tupleTypeIndex<T, T_tuple>::value;


    /**
     * Get a type from the first tuple that contains it
     */
    template<typename T, typename... T_tuples>
    struct tupleTypeFinder;

    template<typename T, typename T_first, typename... T_rest>
    struct tupleTypeFinder<T, T_first, T_rest...> {
        using type = std::conditional_t<
            (tupleTypeIndex_v<T, T_first> >= 0),
            T,
            typename tupleTypeFinder<T, T_rest...>::type
        >;
    };

    template<typename T>
    struct tupleTypeFinder<T> {
        using type = void;
    };

    template<typename T, typename... T_tuples>
    using tupleTypeFinder_t = typename tupleTypeFinder<T, T_tuples...>::type;


    /**
     * Get a reference to a tuple element by type
     * Find the first instance of that type in the source tuple(s)
     */
    template<typename T, typename T_src, typename... T_others>
    decltype(auto) getItemFromTuple(T_src& src, T_others&&... others)
    {
        using Tvalue = std::remove_const_t<std::remove_reference_t<T>>;
        using Tref = std::add_lvalue_reference_t<Tvalue>;

        constexpr int64_t indexref = tupleTypeIndex_v<Tref, T_src>;
        constexpr int64_t indexvalue = tupleTypeIndex_v<Tvalue, T_src>;
        if constexpr (indexref >= 0){
            return std::get<static_cast<std::size_t>(indexref)>(src);
        } else if constexpr (indexvalue >= 0){
            return std::get<static_cast<std::size_t>(indexvalue)>(src);
        } else {
            if constexpr (sizeof...(T_others) > 0){
                return getItemFromTuple<T>(std::forward<T_others>(others)...);
            } else {
                static_assert(portableWrapper::alwaysFalse<T>::value, "Error: Type not found in any source tuple in getItem.");
            }
        }
    }


    /**
     * Is T a packRefs
     */
    template<typename T>
    struct isPackRefs : std::false_type {};

    template<typename... Ts>
    struct isPackRefs<dataPacks::packRefs<Ts...>> : std::true_type {};

    /**
     * Build a packRefs by looking up each of its types in the source tuple(s)
     */
    template<typename... Ts, typename... T_tuples>
    dataPacks::packRefs<Ts...> makePackRefs(dataPacks::packRefs<Ts...>*, T_tuples&... src)
    {
        return dataPacks::packRefs<Ts...>(getItemFromTuple<Ts>(src...)...);
    }

    /**
     * Get one function parameter from the source tuple(s). A packRefs is built, and anything else is a reference to an element
     */
    template<typename T, typename... T_tuples>
    decltype(auto) getParameter(T_tuples&&... src)
    {
        if constexpr (isPackRefs<std::remove_cvref_t<T>>::value){
            return makePackRefs(static_cast<std::remove_cvref_t<T>*>(nullptr), src...);
        } else {
            return getItemFromTuple<T>(src...);
        }
    }

    /**
     * Wrap a parameter in a tuple, holding a reference to an element and a copy of a packRefs
     */
    template<typename T>
    auto parameterTuple(T&& value)
    {
        if constexpr (std::is_lvalue_reference_v<T>){
            return std::tuple<T>(value);
        } else {
            return std::tuple<std::remove_cvref_t<T>>(std::move(value));
        }
    }

    /**
     * Fill a destination tuple selecting elements from a source by type
     */
    template<typename T_dest, int level=0, typename... T_tuples>
    auto getFunctionParametersCore(T_tuples&& ... src)
    {        
        decltype(auto) current = getParameter<std::tuple_element_t<level, T_dest>>(std::forward<T_tuples>(src)...);
        if constexpr (level < std::tuple_size_v<T_dest>-1)
        {
            return std::tuple_cat(parameterTuple(std::forward<decltype(current)>(current)),
                                  getFunctionParametersCore<T_dest,level+1>(std::forward<T_tuples>(src)...));
        } else {
            return parameterTuple(std::forward<decltype(current)>(current));
        }
    }

    /**
     * Get a tuple of function parameters from a function type and a source tuple
     */
    template<typename T_func, typename T_tuple>
    auto getFunctionParameters(T_tuple& tuple)
    {
        if constexpr (std::tuple_size_v<T_tuple> == 0){
            return std::tuple<>();
        } else {
            using T_call = typename far::callableTraits<T_func>::params;
            if constexpr (std::tuple_size_v<T_call> == 0){
                return std::tuple<>();
            } else {
                return getFunctionParametersCore<T_call>(tuple);
            }
        }
    }

    /**
     * Get a tuple of elements from a source tuple by type
     */
    template<typename T_dest, typename T_src>
    auto getCallTupleElements(T_src& src)
    {
        if constexpr (std::tuple_size_v<T_dest> == 0 || std::tuple_size_v<T_src> == 0){
            return std::tuple<>();
        } else {
            return getFunctionParametersCore<T_dest>(src);
        }
    }

    /**
     * Call a member function with a fixed list of leading arguments, and then the remaining parameters
     * filled from a source tuple by type
     * @param obj The object that the function is called on
     * @param func The member function, which cannot be overloaded or a template
     * @param src The source tuple that the remaining parameters are taken from
     * @param fixed The leading arguments, which are passed as they are
     */
    template<typename T_obj, typename T_func, typename T_src, typename... T_fixed>
    decltype(auto) callWithFixedAndInferred(T_obj& obj, T_func func, T_src& src, T_fixed&&... fixed)
    {
        using T_params = typename far::callableTraits<T_func>::params;
        using T_rest = tupleTail_t<sizeof...(T_fixed), T_params>;
        auto inferred = getCallTupleElements<T_rest>(src);
        return std::apply([&](auto&&... rest) -> decltype(auto)
        {
            return (obj.*func)(std::forward<T_fixed>(fixed)..., std::forward<decltype(rest)>(rest)...);
        }, inferred);
    }

    /**
     * The same as above, with the remaining parameters taken from the data packs in a packRefs
     */
    template<typename T_obj, typename T_func, typename... Ts, typename... T_fixed>
    decltype(auto) callWithFixedAndInferred(T_obj& obj, T_func func, dataPacks::packRefs<Ts...>& packs, T_fixed&&... fixed)
    {
        return callWithFixedAndInferred(obj, func, packs.refs, std::forward<T_fixed>(fixed)...);
    }

    //Get a type the consists of the union of two tuples
    template<typename ...Types>
    struct tupleUnion;

    template<typename... T1s, typename... T2s, typename... Rest>
    struct tupleUnion<std::tuple<T1s...>, std::tuple<T2s...>, Rest...> {
        using type = typename tupleUnion<uniqueTuple_t<T1s..., T2s...>, Rest...>::type;
    };

    template<typename... T1s, typename... T2s>
    struct tupleUnion<std::tuple<T1s...>, std::tuple<T2s...>> {
        using type = uniqueTuple_t<T1s..., T2s...>;
    };

    template<typename... T1s>
    struct tupleUnion<std::tuple<T1s...>> {
        using type = uniqueTuple_t<T1s...>;
    };

    template<typename... T>
    using tupleUnion_t = typename tupleUnion<T...>::type;

} // namespace SAMS

#endif // SAMS_RUNNERUTILS_H
