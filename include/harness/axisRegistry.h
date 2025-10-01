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
#ifndef SAMS_AXISREGISTRY_H
#define SAMS_AXISREGISTRY_H
#include <unordered_map>
#include <string>

namespace SAMS{

    class axisRegistry{
        friend axisRegistry& getaxisRegistry();
        private:
            struct axisInfo{
                size_t elements=0;
            };
            std::unordered_map<std::string, axisInfo> axisMap;
            axisRegistry() = default;
        public:

            void registerAxis(const std::string& name){
                if(axisMap.find(name) == axisMap.end()){
                    axisMap.emplace(name, axisInfo());
                }
            }

            size_t getElements(const std::string& name) const{
                auto it = axisMap.find(name);
                if(it == axisMap.end()){
                    throw std::runtime_error("Error: axis " + name + " not found in registry\n");
                }
                return it->second.elements;
            }

            void setElements(const std::string& name, size_t elements){
                auto it = axisMap.find(name);
                if(it == axisMap.end()){
                    throw std::runtime_error("Error: axis " + name + " not found in registry\n");
                }
                it->second.elements = elements;
            }
    };


    inline axisRegistry& getaxisRegistry(){
        static axisRegistry instance;
        return instance;
    }

};

#endif //SAMS_AXISREGISTRY_H