#ifndef SAMS_HARNESS_UTILS_H
#define SAMS_HARNESS_UTILS_H

#include <type_traits>
#include <utility>
#include <functional>

namespace SAMS{


    //Utility class to find if a class has a binary operator defined
    //Operator should be something like std::plus<>, std::minus<>, std::multiplies<>, std::divides<>
    //Relies on a void specialisation that goes to a fully templated operator() of the binary operator
    template<typename T, typename U, template<typename...> typename Op, typename = void>
    struct has_binary_operator : std::false_type {
        using type = void;
    };
    template<typename T, typename U, template<typename...> typename Op>
    struct has_binary_operator<
        T, U, Op,
        std::void_t<decltype(std::declval<Op<void>&>()(std::declval<T>(), std::declval<U>()))>
    > : std::true_type {
        using type = decltype(std::declval<Op<void>&>()(std::declval<T>(), std::declval<U>()));
    };

    template<typename T, typename U, template<typename...> typename Op>
    constexpr bool has_binary_operator_v = has_binary_operator<T, U, Op>::value;

    template<typename T, typename U, template<typename...> typename Op>
    using has_binary_operator_t = typename has_binary_operator<T, U, Op>::type;

    //Now alises for + and - and * and /
    template<typename T, typename U>
    constexpr bool has_addition_v = has_binary_operator_v<T, U, std::plus>;
    template<typename T, typename U>
    using has_addition_t = has_binary_operator_t<T, U, std::plus>;
    template<typename T, typename U>
    constexpr bool has_subtraction_v = has_binary_operator_v<T, U, std::minus>;
    template<typename T, typename U>
    using has_subtraction_t = has_binary_operator_t<T, U, std::minus>;
    template<typename T, typename U>
    constexpr bool has_multiplication_v = has_binary_operator_v<T, U, std::multiplies>;
    template<typename T, typename U>
    using has_multiplication_t = has_binary_operator_t<T, U, std::multiplies>;
    template<typename T, typename U>
    constexpr bool has_division_v = has_binary_operator_v<T, U, std::divides>;
    template<typename T, typename U>
    using has_division_t = has_binary_operator_t<T, U, std::divides>;
}

#endif // SAMS_HARNESS_UTILS_H