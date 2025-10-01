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
#ifndef SAMS_VARIABLEREGISTRY_H
#define SAMS_VARIABLEREGISTRY_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "variableDef.h"

namespace SAMS{

    class variableRegistry{
        friend variableRegistry& getvariableRegistry();
        private:
        std::unordered_map<std::string, variableDef> variables;
        std::vector<std::function<void(std::string)>> allocateCallbacks;
        variableRegistry() = default;
        public:

        /**
         * Register a variable definition with a given name. If the variable already exists, make the definitions consistent.
         * @param name The name of the variable (Must be from list of physically meaningful names)
         * @param varDef The variable definition to register
         */
        void registerVariable(const std::string& name, const variableDef& varDef){
            auto it = variables.find(name);
            if(it != variables.end()){
                //Variable already exists, make consistent
                it->second.makeConsistent(varDef);
            } else {
                //New variable, insert
                variables.emplace(name, varDef);
            }
        }

        /**
         * Syntatic sugar for registering a variable definition with a given name and dimensions. If the variable already exists, make the definitions consistent.
         * @param name The name of the variable (Must be from list of physically meaningful names)
         * @param varType The type of the variable (typeID)
         * @param memSpace The memory space of the variable (memorySpace)
         * @param args The dimensions of the variable (dimension...)
         */
        template<typename... Args>
        void registerVariable(const std::string& name, typeID varType, memorySpace memSpace, Args... args){
            static_assert(sizeof...(args) <= MAX_RANK, "Error: variableDef rank exceeds MAX_RANK");
            variableDef varDef(varType, memSpace, args...);
            registerVariable(name, varDef);
        }

        /**
         * Get a variable definition by name. Throws an error if the variable does not exist.
         * @param name The name of the variable
         * @return The variable definition
         */
        const variableDef& getVariable(const std::string& name) {
            auto it = variables.find(name);
            if(it == variables.end()){
                throw std::runtime_error("Error: variable " + name + " not found in registry\n");
            }
            return it->second;
        }

        /**
         * Allocate memory for a specific variable
         * @param name The name of the variable
         */
        void allocate(const std::string& name){
            auto it = variables.find(name);
            if(it == variables.end()){
                throw std::runtime_error("Error: variable " + name + " not found in registry\n");
            }
            it->second.allocate();
            for(auto& callback : allocateCallbacks){
                callback(name);
            }
        }

        /**
         * Allocate memory for all registered variables
         */
        void allocateAll(){
            for(auto& [name, var] : variables){
                allocate(name);
            }
        }


        /**
         * Deallocate memory for a specific variable
         * @param name The name of the variable
         */
        void deallocate(const std::string& name){
            auto it = variables.find(name);
            if(it == variables.end()){
                throw std::runtime_error("Error: variable " + name + " not found in registry\n");
            }
            it->second.deallocate();
        }

        /**
         * Deallocate memory for all registered variables
         */
        void deallocateAll(){
            for(auto& [name, var] : variables){
                deallocate(name);
            }
        }

        void addAllocateCallback(std::function<void(std::string)> callback){
            allocateCallbacks.push_back(callback);
        }

    };

    /**
     * Returns the singleton instance of the variableRegistry
     */
    inline variableRegistry& getvariableRegistry(){
        static variableRegistry instance;
        return instance;
    }

};

#endif //SAMS_VARIABLEREGISTRY_H