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
#ifndef PARALLELWRAPPER_H
#define PARALLELWRAPPER_H


#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <tuple>
#include <type_traits>
#include "callableTraits.h"

#include <map>
#include <vector>
#include <sstream>

#include "defs.h"
#include "range.h"
#include "utils.h"

//Agnostic tools are things like applyToData, which can be used
//to call a function/lambda/functor with a tuple of arguments.
//These tools are used by the backends to implement the forEach and reduction functions.
#include "agnosticTools.h"

//Now import the actual backends
#include "cudaBackend.h"
#include "hipBackend.h"
#include "KokkosBackend.h"
#include "OpenMPBackend.h"
#include "array.h"

//#include "pwmpi.h"

#include "manager.h"
#include "portableWrapperFrontend.h"

#endif
