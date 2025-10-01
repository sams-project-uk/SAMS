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
#ifndef SAMS_STAGGERREGISTRY_H
#define SAMS_STAGGERREGISTRY_H
#include <array>

namespace SAMS{

    /**
     * Enumeration of variable staggering types
     * This is separately for each dimension
     * CENTRED: Variable is centred in the cell
     * HALF_CELL: Variable is staggered by half a cell
     */
    enum class staggerType{
        CENTRED=0,
        HALF_CELL=1
    };

    class staggerRegistry{

        friend const staggerRegistry& getstaggerRegistry();

        private:
        /**
         * How many extra cells are needed for each staggering type
         * CENTRED: 0 extra cells
         * HALF_CELL: 1 extra cell
         */
        std::array<size_t, 2> extraCells{0,1};

        staggerRegistry() = default;
        public:

        size_t getExtraCells(staggerType s) const {
            return extraCells[static_cast<int>(s)];
        }

    };

    inline const staggerRegistry& getstaggerRegistry(){
        static staggerRegistry instance;
        return instance;
    }

};

#endif //SAMS_STAGGERREGISTRY_H