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
#ifndef AGNOSTICTOOLS_H
#define AGNOSTICTOOLS_H

#include "defs.h"
#include "utils.h"

namespace portableWrapper
{

        /**
        Call a kernel function with the specified tuple of arguments.
        */
        template <typename T_func, typename T_tuple>
        INLINE DEVICEPREFIX auto applyToData(const T_func &func, T_tuple &tuple)
        {
            return APPLY(func, tuple);
        }

}
#endif
