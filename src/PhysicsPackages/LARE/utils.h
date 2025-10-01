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
#ifndef SAMS_UTILS_H
#define SAMS_UTILS_H
#include "shared_data.h"
#include "include/parallelWrapper.h"

namespace sams{

    template<typename T, int N>
    UNREPEATED void assignArray(portableWrapper::portableArray<T, N> &array, const T &value) {
        size_t elements = array.getElements();
        portableWrapper::portableArray<T,1> tempArray;
        tempArray.bind(array.data(), portableWrapper::Range(0, elements - 1));
        portableWrapper::applyKernel(LAMBDA(size_t i) {
            tempArray(i) = value;
        }, portableWrapper::Range(0, elements - 1));
    }
    template<typename T, int N1, int N2>
    UNREPEATED void copyArray(portableWrapper::portableArray<T, N1> &dest, const portableWrapper::portableArray<T, N2> &src) {
        if (dest.getElements() != src.getElements()) {
            throw std::runtime_error("Source and destination arrays must have the same number of elements.");
        }
        portableWrapper::portableArray<T,1> tempS, tempD;
        tempS.bind(src.data(), portableWrapper::Range(0, src.getElements() - 1));
        tempD.bind(dest.data(), portableWrapper::Range(0, dest.getElements() - 1));
        portableWrapper::applyKernel(LAMBDA(size_t i) {
            tempD(i) = tempS(i);
        }, portableWrapper::Range(0, dest.getElements() - 1));
    }
}

#endif