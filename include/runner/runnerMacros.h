#ifndef SAMS_RUNNERMACROS_H
#define SAMS_RUNNERMACROS_H

#define STR(x) #x
#define TOSTRING(x) std::string(STR(x))

#define HAS_X(X) \
private: \
template<typename U, typename... T_spec> \
struct hasMethodHelper_##X { \
private: \
    /* helper: two concrete implementations, pick one via conditional to avoid in-class specialization */ \
    struct det_template_no_spec { \
        template<typename V, typename = void> struct det { static constexpr bool value = false; using params = void; }; \
    }; \
    struct det_template_with_spec { \
        template<typename V, typename = void> struct det { static constexpr bool value = false; using params = void; }; \
        template<typename V> struct det<V, std::void_t<decltype(&V::template X<T_spec...>)>> { \
            static constexpr bool value = true; \
            using params = typename far::callableTraits<decltype(&V::template X<T_spec...>)>::params; \
        }; \
    }; \
    template<typename V, typename = void> \
    using det_template = typename std::conditional_t<(sizeof...(T_spec) > 0), det_template_with_spec, det_template_no_spec>::template det<V, void>; \
    /* detector for non-template member X */ \
    template<typename V, typename = void> struct det_non_template { static constexpr bool value = false; using params = void; }; \
    template<typename V> struct det_non_template<V, std::void_t<decltype(&V::X)>> { \
        static constexpr bool value = true; \
        using params = typename far::callableTraits<decltype(&V::X)>::params; \
    }; \
public: \
    static constexpr bool value = det_template<U>::value || det_non_template<U>::value; \
    using params = std::conditional_t< \
        !std::is_void_v<typename det_template<U>::params>, \
        typename det_template<U>::params, \
        typename det_non_template<U>::params \
    >; \
}; \
    template<int level, typename... T_spec> \
    struct hasMethod_##X { \
        using type = std::tuple_element_t<level, T_combined>; \
        using helper = hasMethodHelper_##X<type, T_spec...>; \
        static constexpr bool value = helper::value; \
        using params = typename helper::params; \
    }; 

     /**
     * Macro to get a type and value for a named parameter which may or may not be defined
     * If it is not defined then the type is void and the value is an integer with value 0
     */
    #define HAS_X_PARAM(X)\
    private: \
        template<typename T, typename = void>\
        struct hasParam_##X {\
            using type = void;\
            static constexpr int value = 0;\
        };\
        template<typename T>\
        struct hasParam_##X<T, std::void_t<decltype(T::X)>> {\
            using type = decltype(T::X);\
            static constexpr type value = T::X;\
        }; \
        template<bool ifActive = true, bool recursive = true, int level=0>\
        T_sizeType countParam_##X(){\
            T_sizeType count;\
            if constexpr(!std::is_void_v<typename hasParam_##X<std::tuple_element_t<level, T_combined>>::type>){\
                using type = typename hasParamType_##X<std::tuple_element_t<level, T_combined>>::type;\
                static_assert(std::is_convertible_v<type, REQ>, "Error: Parameter " #X " does not match the required type."); \
                T_sizeType shouldCount = ifActive ? simulationActiveFlags[level] : true;\
                count = shouldCount ? 1:0;\
            }\
            if constexpr (recursive && level < sizeof...(Package)-1){\
                return count + countParam_##X<ifActive, recursive, level+1>();\
            } else {\
                return count;\
            }\
        }


    /**
     * Macro to get a type with a specified type requirement and value for a named parameter which may or may not be defined
     * If it is not defined then the type is void and the value is an integer with value 0
     * The type requirement is enforced using a static_assert
     */
    #define HAS_X_PARAM_WITH_TYPE(X, REQ)\
    private: \
        template<typename T, typename = void>\
        struct hasParamType_##X {\
            using type = void;\
            static constexpr int value = 0;\
        };\
        template<typename T>\
        struct hasParamType_##X<T, std::void_t<decltype(T::X)>> {\
            using type = decltype(T::X);\
            static constexpr type value = T::X;\
        }; \
        template<bool ifActive = true, bool recursive = true, int level=0>\
        T_sizeType countParam_##X(){\
            T_sizeType count;\
            if constexpr(!std::is_void_v<typename hasParamType_##X<std::tuple_element_t<level, T_combined>>::type>){\
                using type = typename hasParamType_##X<std::tuple_element_t<level, T_combined>>::type;\
                static_assert(std::is_convertible_v<type, REQ>, "Error: Parameter " #X " does not match the required type."); \
                T_sizeType shouldCount = ifActive ? simulationActiveFlags[level] : true;\
                count = shouldCount ? 1:0;\
            }\
            if constexpr (recursive && level < sizeof...(Packages)-1){\
                return count + countParam_##X<ifActive, recursive, level+1>();\
            } else {\
                return count;\
            }\
        }

#ifdef USE_TIMER
    #define TIMER_X(X)\
        template<int level=0>\
        auto nameTimers_##X(){\
            std::string name = static_cast<std::string>(hasParamType_name<std::tuple_element_t<level, T_combined>>::value); \
            if constexpr(level < sizeof...(Packages)-1){\
                return std::tuple_cat( \
                    std::make_tuple(timer(TOSTRING(X) + " " + name + "_" + std::to_string(level))), \
                    nameTimers_##X<level+1>() \
                ); \
            } else {\
                return std::make_tuple(timer(TOSTRING(X) + " " + name + "_" + std::to_string(level))); \
            }\
        } \
        portableWrapper::std_N_ary_tuple_type_t<timer, sizeof...(Packages)> X##_timers;\
        template<int level>\
        void toggleTimer_##X(){\
            std::get<level>(X##_timers).toggle();\
        }\
        template<int level=0>\
        void printTimer_##X(){\
            timer& t = std::get<level>(X##_timers);\
            if (simulationActiveFlags[level] && t.everRun()) {\
                std::string name = static_cast<std::string>(hasParamType_name<std::tuple_element_t<level, T_combined>>::value); \
                SAMS::cout << "Timer " << TOSTRING(X) << " for simulation " << name << " (level " << level << "): " << t.end_silent() << " seconds." << std::endl;\
            }\
            if constexpr(level < sizeof...(Packages)-1){\
                printTimer_##X<level+1>();\
            }\
        }
#else
//No timer support
    #define TIMER_X(X) \
    template<int level>\
    void toggleTimer_##X(){}\
    template<int level=0>\
    void printTimer_##X(){}
#endif

   #define CALL_X(X)\
        HAS_X(X) \
        TIMER_X(X) \
        public: \
        template<bool handleExceptions = false, bool ifActive = true, bool recursive = true, int level = 0, typename... T_spec, typename... T>\
        auto callCore_##X(T&&... args){\
            if constexpr(hasMethod_##X<level, T_spec...>::value){\
                using rtype = typename hasMethod_##X<level, T_spec...>::params; \
                using outerParams = tupleTail_t<sizeof...(T), rtype>; \
                bool shouldRun = ifActive ? simulationActiveFlags[level] : true;\
                if (shouldRun) {\
                    toggleTimer_##X<level>();\
                    auto cTuple = std::tuple_cat( \
                        std::forward_as_tuple(std::forward<T>(args)...), \
                        getCallTupleElements<outerParams>(runnerData) \
                    ); \
                    if constexpr(handleExceptions){\
                        try {\
                            if constexpr(sizeof...(T_spec)>0){\
                                std::apply([&](auto&&... cargs) {\
                                    std::get<level>(runnerData).template X<T_spec...>(std::forward<decltype(cargs)>(cargs)...);\
                                }, cTuple);\
                            } else {\
                                std::apply([&](auto&&... cargs) {\
                                    std::get<level>(runnerData).X(std::forward<decltype(cargs)>(cargs)...);\
                                }, cTuple);\
                            }\
                        } catch (const std::exception& e) {\
                            std::string name = static_cast<std::string>(hasParamType_name<std::tuple_element_t<level, T_combined>>::value); \
                            std::stringstream ss;\
                            ss << "Error in simulation " << name << " (level " << level << ") during call to " << TOSTRING(X) << ": " << e.what() << std::endl;\
                            abort(ss.str(), false);\
                        }\
                    } else {\
                        if constexpr(sizeof...(T_spec)>0){\
                            std::apply([&](auto&&... cargs) {\
                                std::get<level>(runnerData).template X<T_spec...>(std::forward<decltype(cargs)>(cargs)...);\
                            }, cTuple);\
                        } else {\
                            std::apply([&](auto&&... cargs) {\
                                std::get<level>(runnerData).X(std::forward<decltype(cargs)>(cargs)...);\
                            }, cTuple);\
                        }\
                    }\
                    toggleTimer_##X<level>();\
                } \
            }\
            if constexpr (recursive && level < sizeof...(Packages)-1){\
                callCore_##X<handleExceptions,ifActive, recursive, (level+1), T_spec...>(std::forward<T>(args)...);\
            }\
        }


    #define FULL_CALL_X(X) \
        CALL_X(X) \
        template<typename... T_spec, typename... T> \
        void X(T&&... args){\
            callCore_##X<false,true,true,0,T_spec...>(std::forward<T>(args)...);\
        }

    #define FULL_CALL_X_WITH_EXCEPTIONS(X) \
        CALL_X(X) \
        template<typename... T_spec, typename... T> \
        void X##_with_exceptions(T&&... args){\
            callCore_##X<true,true,true,0,T_spec...>(std::forward<T>(args)...);\
        }

#endif // SAMS_RUNNERMACROS_H