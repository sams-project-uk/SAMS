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
#ifndef RANGE_H
#define RANGE_H

#include "defs.h"
#include <stdint.h>

namespace portableWrapper
{
    /**
     * Class to represent a range of indices.
     * It can be used to specify the bounds for parallel operations
     * and also the range of indices for the parallelWrapper.
     */
    struct Range
    {
        SIGNED_INDEX_TYPE lower_bound;
        SIGNED_INDEX_TYPE upper_bound;
        SIGNED_INDEX_TYPE stride;
        DEVICEPREFIX Range(SIGNED_INDEX_TYPE lb, SIGNED_INDEX_TYPE ub)
            : lower_bound(lb), upper_bound(ub), stride(1) {}
        DEVICEPREFIX Range(SIGNED_INDEX_TYPE nels)
            : lower_bound(1), upper_bound(nels), stride(1) {}
        DEVICEPREFIX Range(SIGNED_INDEX_TYPE lb, SIGNED_INDEX_TYPE ub, SIGNED_INDEX_TYPE stride)
            : lower_bound(lb), upper_bound(ub), stride(stride) {}
        DEVICEPREFIX Range() 
            : lower_bound(INT64_MIN), 
              upper_bound(INT64_MAX), 
              stride(1) {}
    };
}

#endif