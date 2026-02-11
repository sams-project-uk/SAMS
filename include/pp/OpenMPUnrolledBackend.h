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
#ifndef OPENMPUNROLLEDBACKEND_H
#define OPENMPUNROLLEDBACKEND_H

#include "defs.h"
#include "range.h"
#include "utils.h"
#include <tuple>

namespace portableWrapper
{
    /*
    Yes, this looks stupid, but some GCC versions seem to have trouble with
    inlining the templated recursive version of forEachParallel. This 
    is a very unpleasant workaround, but it does work
    */
    namespace openmp {

        /**
         * Maximum rank for which unrolled OpenMP loops are implemented
         * Ranks above this will use the standard recursive implementation
         * NOTE: Only increase this value if you have implemented the corresponding
         * unrolled forEachParallel function templates below.
         */
        static constexpr int UNROLL_LIMIT = 8;

        /**
         * 1D parallel forEach using OpenMP unrolling
         * @param func The function to call for each element in the range
         * @param cRange The range to iterate over
         */
        template <typename T_func, typename T_cRange>
        HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange cRange)
        {
            auto range = getRange(cRange);
            T_indexType lower_bound1 = range.lower_bound; T_indexType upper_bound1 = range.upper_bound;
#pragma omp parallel for schedule(dynamic)
            for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
            {
                func(i1);
            }
        }

        /**
         * 2D parallel forEach using OpenMP unrolling
         * @param func The function to call for each element in the range
         * @param cRange1 The first range to iterate over
         * @param cRange2 The second range to iterate over
         */
        template <typename T_func, typename T_cRange1, typename T_cRange2>
        HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange1 cRange1, T_cRange2 cRange2)
        {
            //Yes, this could be done with a structured binding, but it causes
            //CLANG frontend to crash when parsing
            auto range1 = getRange(cRange1);
            auto range2 = getRange(cRange2);
            T_indexType lower_bound1 = range1.lower_bound; T_indexType upper_bound1 = range1.upper_bound;
            T_indexType lower_bound2 = range2.lower_bound; T_indexType upper_bound2 = range2.upper_bound;
#if defined(NATIVE_LOOP_COLLAPSE) && NATIVE_LOOP_COLLAPSE <= 2
#pragma omp parallel for collapse(2) schedule(dynamic) 
#else
#pragma omp parallel for schedule(dynamic)
#endif
            for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
            {
                for (SIGNED_INDEX_TYPE i2 = lower_bound2; i2 <= upper_bound2; ++i2)
                {
                    func(i1, i2);
                }
            }
        }

        /**
         * 3D parallel forEach using OpenMP unrolling
         * @param func The function to call for each element in the range
         * @param cRange1 The first range to iterate over
         * @param cRange2 The second range to iterate over
         * @param cRange3 The third range to iterate over
         */
        template <typename T_func, typename T_cRange1, typename T_cRange2, typename T_cRange3>
        HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange1 cRange1, T_cRange2 cRange2, T_cRange3 cRange3)
        {
            auto range1 = getRange(cRange1);
            auto range2 = getRange(cRange2);
            auto range3 = getRange(cRange3);
            T_indexType lower_bound1 = range1.lower_bound; T_indexType upper_bound1 = range1.upper_bound;
            T_indexType lower_bound2 = range2.lower_bound; T_indexType upper_bound2 = range2.upper_bound;
            T_indexType lower_bound3 = range3.lower_bound; T_indexType upper_bound3 = range3.upper_bound;
#if defined(NATIVE_LOOP_COLLAPSE) && NATIVE_LOOP_COLLAPSE <= 3
#pragma omp parallel for collapse(3) schedule(dynamic)
#else
#pragma omp parallel for schedule(dynamic)
#endif
            for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
            {
                for (SIGNED_INDEX_TYPE i2 = lower_bound2; i2 <= upper_bound2; ++i2)
                {
                    for (SIGNED_INDEX_TYPE i3 = lower_bound3; i3 <= upper_bound3; ++i3)
                    {
                        func(i1, i2, i3);
                    }
                }
            }
        }

        /**
         * 4D parallel forEach using OpenMP unrolling
         * @param func The function to call for each element in the range
         * @param cRange1 The first range to iterate over
         * @param cRange2 The second range to iterate over
         * @param cRange3 The third range to iterate over
         * @param cRange4 The fourth range to iterate over
         */
        template <typename T_func, typename T_cRange1, typename T_cRange2, typename T_cRange3, typename T_cRange4>
        HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange1 cRange1, T_cRange2 cRange2, T_cRange3 cRange3, T_cRange4 cRange4)
        {
                auto range1 = getRange(cRange1); 
                auto range2 = getRange(cRange2);
                auto range3 = getRange(cRange3);
                auto range4 = getRange(cRange4);
                T_indexType lower_bound1 = range1.lower_bound; T_indexType upper_bound1 = range1.upper_bound;
                T_indexType lower_bound2 = range2.lower_bound; T_indexType upper_bound2 = range2.upper_bound;
                T_indexType lower_bound3 = range3.lower_bound; T_indexType upper_bound3 = range3.upper_bound;
                T_indexType lower_bound4 = range4.lower_bound; T_indexType upper_bound4 = range4.upper_bound;
#if defined(NATIVE_LOOP_COLLAPSE) && NATIVE_LOOP_COLLAPSE <= 4
#pragma omp parallel for collapse(4) schedule(dynamic)
#else
#pragma omp parallel for schedule(dynamic)
#endif
            for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
            {
                for (SIGNED_INDEX_TYPE i2 = lower_bound2; i2 <= upper_bound2; ++i2)
                {
                    for (SIGNED_INDEX_TYPE i3 = lower_bound3; i3 <= upper_bound3; ++i3)
                    {
                        for (SIGNED_INDEX_TYPE i4 = lower_bound4; i4 <= upper_bound4; ++i4)
                        {
                            func(i1, i2, i3, i4);
                        }
                    }
                }
            }
        }

        /**
         * 5D parallel forEach using OpenMP unrolling
         * @param func The function to call for each element in the range
         * @param cRange1 The first range to iterate over
         * @param cRange2 The second range to iterate over
         * @param cRange3 The third range to iterate over
         * @param cRange4 The fourth range to iterate over
         * @param cRange5 The fifth range to iterate over
         */
        template <typename T_func, typename T_cRange1, typename T_cRange2, typename T_cRange3, typename T_cRange4, typename T_cRange5>
        HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange1 cRange1, T_cRange2 cRange2, T_cRange3 cRange3, T_cRange4 cRange4, T_cRange5 cRange5)
        {
            auto range1 = getRange(cRange1);
            auto range2 = getRange(cRange2);
            auto range3 = getRange(cRange3);
            auto range4 = getRange(cRange4);
            auto range5 = getRange(cRange5);
            T_indexType lower_bound1 = range1.lower_bound; T_indexType upper_bound1 = range1.upper_bound;
            T_indexType lower_bound2 = range2.lower_bound; T_indexType upper_bound2 = range2.upper_bound;
            T_indexType lower_bound3 = range3.lower_bound; T_indexType upper_bound3 = range3.upper_bound;
            T_indexType lower_bound4 = range4.lower_bound; T_indexType upper_bound4 = range4.upper_bound;
            T_indexType lower_bound5 = range5.lower_bound; T_indexType upper_bound5 = range5.upper_bound;
#if defined(NATIVE_LOOP_COLLAPSE) && NATIVE_LOOP_COLLAPSE <= 5
#pragma omp parallel for collapse(5) schedule(dynamic)
#else
#pragma omp parallel for schedule(dynamic)
#endif
            for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
            {
                for (SIGNED_INDEX_TYPE i2 = lower_bound2; i2 <= upper_bound2; ++i2)
                {
                    for (SIGNED_INDEX_TYPE i3 = lower_bound3; i3 <= upper_bound3; ++i3)
                    {
                        for (SIGNED_INDEX_TYPE i4 = lower_bound4; i4 <= upper_bound4; ++i4)
                        {
                            for (SIGNED_INDEX_TYPE i5 = lower_bound5; i5 <= upper_bound5; ++i5)
                            {
                                func(i1, i2, i3, i4, i5);
                            }
                        }
                    }
                }
            }
        }

    /**
     * 6D parallel forEach using OpenMP unrolling
     * @param func The function to call for each element in the range
     * @param cRange1 The first range to iterate over
     * @param cRange2 The second range to iterate over
     * @param cRange3 The third range to iterate over
     * @param cRange4 The fourth range to iterate over
     * @param cRange5 The fifth range to iterate over
     * @param cRange6 The sixth range to iterate over
     */
    template <typename T_func, typename T_cRange1, typename T_cRange2, typename T_cRange3, typename T_cRange4, typename T_cRange5, typename T_cRange6>
    HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange1 cRange1, T_cRange2 cRange2, T_cRange3 cRange3, T_cRange4 cRange4, T_cRange5 cRange5, T_cRange6 cRange6)
    {
        auto range1 = getRange(cRange1);
        auto range2 = getRange(cRange2);
        auto range3 = getRange(cRange3);
        auto range4 = getRange(cRange4);
        auto range5 = getRange(cRange5);
        auto range6 = getRange(cRange6);
        T_indexType lower_bound1 = range1.lower_bound; T_indexType upper_bound1 = range1.upper_bound;
        T_indexType lower_bound2 = range2.lower_bound; T_indexType upper_bound2 = range2.upper_bound;
        T_indexType lower_bound3 = range3.lower_bound; T_indexType upper_bound3 = range3.upper_bound;
        T_indexType lower_bound4 = range4.lower_bound; T_indexType upper_bound4 = range4.upper_bound;
        T_indexType lower_bound5 = range5.lower_bound; T_indexType upper_bound5 = range5.upper_bound;
        T_indexType lower_bound6 = range6.lower_bound; T_indexType upper_bound6 = range6.upper_bound;
#if defined(NATIVE_LOOP_COLLAPSE) && NATIVE_LOOP_COLLAPSE <= 6
#pragma omp parallel for collapse(6) schedule(dynamic)
#else
#pragma omp parallel for schedule(dynamic)
#endif
        for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
        {
            for (SIGNED_INDEX_TYPE i2 = lower_bound2; i2 <= upper_bound2; ++i2)
            {
                for (SIGNED_INDEX_TYPE i3 = lower_bound3; i3 <= upper_bound3; ++i3)
                {
                    for (SIGNED_INDEX_TYPE i4 = lower_bound4; i4 <= upper_bound4; ++i4)
                    {
                        for (SIGNED_INDEX_TYPE i5 = lower_bound5; i5 <= upper_bound5; ++i5)
                        {
                            for (SIGNED_INDEX_TYPE i6 = lower_bound6; i6 <= upper_bound6; ++i6)
                            {
                                func(i1, i2, i3, i4, i5, i6);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * 7D parallel forEach using OpenMP unrolling
     * @param func The function to call for each element in the range
     * @param cRange1 The first range to iterate over
     * @param cRange2 The second range to iterate over
     * @param cRange3 The third range to iterate over
     * @param cRange4 The fourth range to iterate over
     * @param cRange5 The fifth range to iterate over
     * @param cRange6 The sixth range to iterate over
     * @param cRange7 The seventh range to iterate over
     */
    template <typename T_func, typename T_cRange1, typename T_cRange2, typename T_cRange3, typename T_cRange4, typename T_cRange5, typename T_cRange6, typename T_cRange7>
    HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange1 cRange1, T_cRange2 cRange2, T_cRange3 cRange3, T_cRange4 cRange4, T_cRange5 cRange5, T_cRange6 cRange6, T_cRange7 cRange7)
    {
        auto range1 = getRange(cRange1);
        auto range2 = getRange(cRange2);
        auto range3 = getRange(cRange3);
        auto range4 = getRange(cRange4);
        auto range5 = getRange(cRange5);
        auto range6 = getRange(cRange6);
        auto range7 = getRange(cRange7);
        T_indexType lower_bound1 = range1.lower_bound; T_indexType upper_bound1 = range1.upper_bound;
        T_indexType lower_bound2 = range2.lower_bound; T_indexType upper_bound2 = range2.upper_bound;
        T_indexType lower_bound3 = range3.lower_bound; T_indexType upper_bound3 = range3.upper_bound;
        T_indexType lower_bound4 = range4.lower_bound; T_indexType upper_bound4 = range4.upper_bound;
        T_indexType lower_bound5 = range5.lower_bound; T_indexType upper_bound5 = range5.upper_bound;
        T_indexType lower_bound6 = range6.lower_bound; T_indexType upper_bound6 = range6.upper_bound;
        T_indexType lower_bound7 = range7.lower_bound; T_indexType upper_bound7 = range7.upper_bound;
#if defined(NATIVE_LOOP_COLLAPSE) && NATIVE_LOOP_COLLAPSE <= 7
#pragma omp parallel for collapse(7) schedule(dynamic)
#else
#pragma omp parallel for schedule(dynamic)
#endif
        for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
        {
            for (SIGNED_INDEX_TYPE i2 = lower_bound2; i2 <= upper_bound2; ++i2)
            {
                for (SIGNED_INDEX_TYPE i3 = lower_bound3; i3 <= upper_bound3; ++i3)
                {
                    for (SIGNED_INDEX_TYPE i4 = lower_bound4; i4 <= upper_bound4; ++i4)
                    {
                        for (SIGNED_INDEX_TYPE i5 = lower_bound5; i5 <= upper_bound5; ++i5)
                        {
                            for (SIGNED_INDEX_TYPE i6 = lower_bound6; i6 <= upper_bound6; ++i6)
                            {
                                for (SIGNED_INDEX_TYPE i7 = lower_bound7; i7 <= upper_bound7; ++i7)
                                {
                                    func(i1, i2, i3, i4, i5, i6, i7);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * 8D parallel forEach using OpenMP unrolling
     * @param func The function to call for each element in the range
     * @param cRange1 The first range to iterate over
     * @param cRange2 The second range to iterate over
     * @param cRange3 The third range to iterate over
     * @param cRange4 The fourth range to iterate over
     * @param cRange5 The fifth range to iterate over
     * @param cRange6 The sixth range to iterate over
     * @param cRange7 The seventh range to iterate over
     * @param cRange8 The eighth range to iterate over
     */
    template <typename T_func, typename T_cRange1, typename T_cRange2, typename T_cRange3, typename T_cRange4, typename T_cRange5, typename T_cRange6, typename T_cRange7, typename T_cRange8>
    HOSTDEVICEPREFIX HOSTINLINE void forEachParallel(T_func func,T_cRange1 cRange1, T_cRange2 cRange2, T_cRange3 cRange3, T_cRange4 cRange4, T_cRange5 cRange5, T_cRange6 cRange6, T_cRange7 cRange7, T_cRange8 cRange8)
    {
        auto range1 = getRange(cRange1);
        auto range2 = getRange(cRange2);
        auto range3 = getRange(cRange3);
        auto range4 = getRange(cRange4);
        auto range5 = getRange(cRange5);
        auto range6 = getRange(cRange6);
        auto range7 = getRange(cRange7);
        auto range8 = getRange(cRange8);
        T_indexType lower_bound1 = range1.lower_bound; T_indexType upper_bound1 = range1.upper_bound;
        T_indexType lower_bound2 = range2.lower_bound; T_indexType upper_bound2 = range2.upper_bound;
        T_indexType lower_bound3 = range3.lower_bound; T_indexType upper_bound3 = range3.upper_bound;
        T_indexType lower_bound4 = range4.lower_bound; T_indexType upper_bound4 = range4.upper_bound;
        T_indexType lower_bound5 = range5.lower_bound; T_indexType upper_bound5 = range5.upper_bound;
        T_indexType lower_bound6 = range6.lower_bound; T_indexType upper_bound6 = range6.upper_bound;
        T_indexType lower_bound7 = range7.lower_bound; T_indexType upper_bound7 = range7.upper_bound;
        T_indexType lower_bound8 = range8.lower_bound; T_indexType upper_bound8 = range8.upper_bound;
#if defined(NATIVE_LOOP_COLLAPSE) && NATIVE_LOOP_COLLAPSE <= 8
#pragma omp parallel for collapse(8) schedule(dynamic)
#else
#pragma omp parallel for schedule(dynamic)
#endif
        for (SIGNED_INDEX_TYPE i1 = lower_bound1; i1 <= upper_bound1; ++i1)
        {
            for (SIGNED_INDEX_TYPE i2 = lower_bound2; i2 <= upper_bound2; ++i2)
            {
                for (SIGNED_INDEX_TYPE i3 = lower_bound3; i3 <= upper_bound3; ++i3)
                {
                    for (SIGNED_INDEX_TYPE i4 = lower_bound4; i4 <= upper_bound4; ++i4)
                    {
                        for (SIGNED_INDEX_TYPE i5 = lower_bound5; i5 <= upper_bound5; ++i5)
                        {
                            for (SIGNED_INDEX_TYPE i6 = lower_bound6; i6 <= upper_bound6; ++i6)
                            {
                                for (SIGNED_INDEX_TYPE i7 = lower_bound7; i7 <= upper_bound7; ++i7)
                                {
                                    for (SIGNED_INDEX_TYPE i8 = lower_bound8; i8 <= upper_bound8; ++i8)
                                    {
                                        func(i1, i2, i3, i4, i5, i6 , i7, i8);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    } // namespace openmp
} // namespace portableWrapper


#endif //OPENMPUNROLLEDBACKEND_H