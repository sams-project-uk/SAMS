#ifndef SAMS_HARNESS_UTILS_H
#define SAMS_HARNESS_UTILS_H

#include <type_traits>
#include <utility>
#include <functional>

namespace SAMS{

    /**
     * Wrapper around the idea of a constexpr string. We use it for composable names
     */
    template <std::size_t N>
    struct constexprName {
        std::array<char, N> data;
        constexpr constexprName(const char (&s)[N]) {
            for (std::size_t i = 0; i < N; ++i) data[i] = s[i];
        }
        // ...added constructor to accept array results from operator+...
        constexpr constexprName(const std::array<char, N> &a) : data(a) {}
        constexpr std::size_t size() const noexcept { return N - 1; }
        // constexpr conversion operator — works on constexpr instances
        constexpr operator std::string_view() const noexcept { return {data.data(), size()}; }
    };

    /**
     * Addition operator for constexprName to allow concatenation of names at compile time.
     */
    template <std::size_t N, std::size_t M>
    constexpr auto operator+(const constexprName<N>& a, const constexprName<M>& b) {
        std::array<char, N + M - 1> out{};
        for (std::size_t i = 0; i < N - 1; ++i) out[i] = a.data[i];
        for (std::size_t j = 0; j < M; ++j)     out[N - 1 + j] = b.data[j]; // copy b including '\0'
        return constexprName<N + M - 1>(out);
    }

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