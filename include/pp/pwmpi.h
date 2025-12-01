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
#ifndef PW_MPI_H
#define PW_MPI_H

#include "defs.h"
#include "utils.h"
#include "range.h"

#if defined(MPI_VERSION) || defined(USE_MPI)
#include <mpi.h>

namespace portableWrapper
{
    namespace mpi
    {
        template <int level = 0, typename T_lb, typename T_cRange, typename... T_ranges>
        inline void populateSubarrayRanges(int *starts, int *subsizes, const T_lb *arrayLBounds, T_cRange cRange, T_ranges... oRanges)
        {
            if constexpr (std::is_same_v<T_cRange, Range>)
            {
                starts[level] = cRange.lower_bound - arrayLBounds[level];
                subsizes[level] = cRange.upper_bound - cRange.lower_bound + 1;
            }
            else
            {
                starts[level] = cRange - arrayLBounds[level];
                subsizes[level] = 1;
            }
            if constexpr (sizeof...(oRanges) > 0)
            {
                populateSubarrayRanges<level + 1>(starts, subsizes, arrayLBounds, oRanges...);
            }
        }

        /**
         * Function to map from a C++ type to an MPI datatype
         */

        template <typename T>
        inline constexpr MPI_Datatype getMPIDatatype()
        {
            if constexpr (std::is_same_v<T, int>)
            {
                return MPI_INT;
            }
            else if constexpr (std::is_same_v<T, unsigned int>)
            {
                return MPI_UNSIGNED;
            }
            else if constexpr (std::is_same_v<T, long>)
            {
                return MPI_LONG;
            }
            else if constexpr (std::is_same_v<T, unsigned long>)
            {
                return MPI_UNSIGNED_LONG;
            }
            else if constexpr (std::is_same_v<T, long long>)
            {
                return MPI_LONG_LONG;
            }
            else if constexpr (std::is_same_v<T, unsigned long long>)
            {
                return MPI_UNSIGNED_LONG_LONG;
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                return MPI_FLOAT;
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return MPI_DOUBLE;
            }
            else if constexpr (std::is_same_v<T, long double>)
            {
                return MPI_LONG_DOUBLE;
            }
            else if constexpr (std::is_same_v<T, char>)
            {
                return MPI_CHAR;
            }
            else
            {
                static_assert(false, "Unsupported type for MPI datatype conversion.");
            }
        }


        inline void initialize(int &argc, char *argv[])
        {
                int initialized;
                MPI_Initialized(&initialized);
                if (!initialized){
                    std::cerr << "MPI is not initialized. Please initialize MPI before using the portable wrapper." << std::endl;
                    exit(EXIT_FAILURE);
                }
        }

        /**
         * Function to get an MPI_Datatype for a specified slice of a portableArray
         */
        template <typename T_data, int rank, portableWrapper::arrayTags tag, typename... T_ranges>
        inline MPI_Datatype getMPISliceType(const portableArray<T_data, rank, tag> &array, T_ranges... ranges)
        {
            static_assert(sizeof...(ranges) == rank, "Number of ranges must match the rank of the portable array.");
            int starts[rank];
            int subsizes[rank];
            int sizes[rank];
            for (int i = 0; i < rank; ++i)
                sizes[i] = array.getSize(i);
            mpi::populateSubarrayRanges(starts, subsizes, array.getLowerBounds(), ranges...);
            MPI_Datatype mpiType;
            MPI_Type_create_subarray(rank, sizes, subsizes, starts, MPI_ORDER_C, mpi::getMPIDatatype<T_data>(), &mpiType);
            MPI_Type_commit(&mpiType);
            return mpiType;
        }
    } // namespace mpi

} // namespace portableWrapper

#endif // defined(MPI_VERSION) || defined(USE_MPI)
#endif // PW_MPI_H