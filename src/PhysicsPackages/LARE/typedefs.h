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
#ifndef TYPEDDEFS_H
#define TYPEDDEFS_H

#include "pp/parallelWrapper.h"

namespace LARE
{

  using T_dataType = SAMS::T_dataType;
  using T_sizeType = SAMS::T_sizeType;
  using T_indexType = SAMS::T_indexType;

  using volumeArray = portableWrapper::acceleratedArray<T_dataType, 3>;
  using hostVolumeArray = portableWrapper::hostArray<T_dataType, 3>;
  using planeArray = portableWrapper::acceleratedArray<T_dataType, 2>;
  using hostPlaneArray = portableWrapper::hostArray<T_dataType, 2>;
  using lineArray = portableWrapper::acceleratedArray<T_dataType, 1>;
  using hostLineArray = portableWrapper::hostArray<T_dataType, 1>;
}
#endif