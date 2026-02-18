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
#ifndef WRITERPROTO_H
#define WRITERPROTO_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <variant>
#include <vector>

enum class meshDataOrder{
	dataFirst,
	meshFirst,
	noOrder
};

//Class enum for data state
enum class dataState{
	notFound,
	foundUnique,
	foundMultiple
};

class primitive {
public:
using allowedMeshTypes_impl = std::tuple<float, double>;
};


	/*
	Some macro magic to automatically generate the get_<name> functions
	It uses SFINAE to check if the type has the member variable <name>
	and if it does, it uses that value. If not, it uses the default value
	*/
	#define DEFAULTVALUE(name, dataType, defaultValue) \
    template<typename T, typename = void> \
    struct get_##name { \
        static constexpr dataType value = defaultValue; \
    }; \
    template<typename T> \
    struct get_##name<T, std::void_t<decltype(T::name##_impl)>> { \
        static constexpr dataType value = T::name##_impl; \
    }; \
	constexpr static dataType name = get_##name<T_core>::value;

	//This is for a class having a "using name = type" statement
	#define DEFAULTUSE(name, defaultUse) \
	template<typename T, typename = void> \
	struct use_##name { \
		using type = defaultUse; \
	}; \
	template<typename T> \
	struct use_##name<T, std::void_t<typename T::name##_impl>> { \
		using type = typename T::name##_impl; \
	}; \
	//using name = typename use_##name<T_core>::type;

	#define DEFAULTUSE2(name, defaultUse) \
	template<typename T> \
	struct usex_##name { \
		using type = typename T_core::name##_impl; \
	};

	#define HASMEMBER(member) \
	template<typename, typename T, typename... Args> \
	struct hasFunction_##member : std::false_type {}; \
	template<typename T, typename... Args> \
	struct hasFunction_##member<std::void_t<decltype(std::declval<T>().member(std::declval<Args>()...))>, T, Args...> : std::true_type {}; \
	template<typename T, typename... Args> \
	constexpr static bool has_##member = hasFunction_##member<void, T, Args...>::value;

	DEFAULTUSE(allowedMeshTypes, primitive::allowedMeshTypes_impl);

	/*
	end macro magic
	*/

	/*
	 * @brief Base class for all writers. This class provides the interface for writing data to a file.
	 * @tparam T_core The type of the writer. This should be a subclass of writer.
	 * @details This uses CRTP to allow the writer to be a subclass of writer. This allows the writer to implement the interface without having to use virtual functions.
	 */
template<typename T_core>
class writer {

	using defaultAllowedMeshTypes = std::tuple<float, double, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;
	using defaultAllowedDataTypes = std::tuple<float, double, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;

	template<typename T, typename T_Tuple, int I>
	constexpr static bool typeInTuple() {
		if constexpr(std::tuple_size_v<T_Tuple> == 0) return true;//Empty tuple is "Accept All"
		if constexpr (I >= std::tuple_size_v<T_Tuple>) return true; //Out of elements is true
		else 
		{
			return std::is_same_v<T, std::tuple_element_t<I, T_Tuple>> || typeInTuple<T, T_Tuple, I + 1>();
		}
	}

	public:

	//This public section describes the capabilities of the writer
	//It should be used by the ouput code if it needs to write data
	//differently for different writers. THIS SHOULD BE AN UNUSUAL CASE!

	//What is the name of the writer. This is not given a
	//default value because it is an error to not have it
	constexpr static std::string_view writerName = T_core::writerName;

	//Can you register more variables after starting writing
	DEFAULTVALUE(registerAfterWrite, bool, false);
	//Are dataset names unique to the mesh or globally unique
	DEFAULTVALUE(supportsPerMeshDataNames, bool, false);
	//Can you have multiple timesteps in the file
	DEFAULTVALUE(supportsMultipleTimesteps, bool, false);
	//What is the maximum rank that the writer supports
	DEFAULTVALUE(supportsMultipleTimesteps2, bool, false);
	//Do you have to register the mesh before the data, vice versa, or does it not matter
	DEFAULTVALUE(registerOrder, meshDataOrder, meshDataOrder::noOrder);
	//Do you have to write the mesh before the data, vice versa, or does it not matter
	DEFAULTVALUE(writeOrder, meshDataOrder, meshDataOrder::noOrder);
	constexpr static size_t maximumRank = 8;
	//Does the writer support key-value pairs globally
	DEFAULTVALUE(supportsKeyValue, bool, false);
	//Does the writer support setting the key-value pairs for a specific mesh
	DEFAULTVALUE(supportsMeshKeyValue, bool, false);
	//Does the writer support setting the key-value pairs for a specific data
	DEFAULTVALUE(supportsDataKeyValue, bool, false);

	private:

	DEFAULTUSE(allowedMeshTypes, defaultAllowedMeshTypes);
	DEFAULTUSE2(allowedMeshTypes, defaultAllowedDataTypes);
	DEFAULTUSE(allowedDataTypes, defaultAllowedDataTypes);

	//Implementation for start of registration
	HASMEMBER(startRegisterImpl);
	//Implementation for start of writing
	HASMEMBER(startWriteImpl);
	//Implementation for opening a file
	HASMEMBER(openFileImpl);
	/*
	Implementation for setting a timestep/iteration
	Writers that don't support multiple timesteps only need to support this being called once
	before the first register. Writers that support multiple timesteps must allow it to be called
	at any time and must start a new timestep when it is called.
	*/
	HASMEMBER(setTimeImpl);
	//Implementation for setting a job name, completely optional and a writer must not require it
	HASMEMBER(setJobName);
	//Implementation for setting a key-value pair in the global scope.
	HASMEMBER(setKeyValue);
	//Implementation for setting a key-value pair on a specific mesh
	HASMEMBER(setMeshKeyValue);
	//Implementation for setting a key-value pair on a specific dataset
	HASMEMBER(setDataKeyValue);
	//Implementation for registering a rectilinear mesh
	HASMEMBER(registerRectilinearMeshImpl);
	//Implementation for registering an arbitrary mesh
	HASMEMBER(registerMeshImpl);
	//Implementation for writing a rectilinear mesh
	HASMEMBER(writeRectilinearMeshImpl);
	//Implementation for registering a dataset
	HASMEMBER(registerDataImpl);
	//Implementation for writing a dataset. This is the ONLY compulsory function
	//that a writer MUST implement, so there is no HASMEMBER for it
	//HASMEMBER(writeDataImpl);
	//Implementation for flushing the file to disk
	HASMEMBER(flushFileImpl);
	//Implementation for closing the file
	HASMEMBER(closeFileImpl);

	friend T_core;
	private:

	/**
	 * @brief Structure to hold infor about a rectilinear mesh
	 * @param rank The rank of the mesh
	 * @param sizes The size of the mesh in each dimension
	 * 
	 */
	struct rlMeshInfo{
		size_t rank;
		std::array<size_t, maximumRank> sizes;
		size_t zones;
		template<typename ...Args>
		rlMeshInfo(Args... args) : rank{sizeof...(args)}, sizes{args...}, zones{std::reduce(sizes.begin(), sizes.end(), size_t(1), std::multiplies<>())} {
			static_assert(sizeof...(args) <= maximumRank, "Too many dimensions for rectilinear mesh");
		}
		rlMeshInfo() = default;
	};

	struct meshInfo {
		bool written = false;
		size_t zones;
		const std::type_info* type = nullptr;
		std::variant<rlMeshInfo> specificInfo;
	};

	//Structure to hold data info
	struct dataInfo{
		std::string meshName;
		size_t elementSize;		
		const std::type_info* type = nullptr;
		std::array<size_t, maximumRank> staggers;
		bool written = false;
		dataInfo() = default;
		dataInfo(const char* meshName, const size_t elementSize, const std::type_info* type) : meshName(meshName), elementSize(elementSize), type(type){};
	};

	//Map from mesh name to mesh info
	std::unordered_map<std::string, meshInfo> meshMap;
	//Map from data name to data info
	std::unordered_map<std::string, std::unordered_map<std::string,dataInfo>> dataMap;
	//For writers that have globally unique names, map from data name to mesh name
	std::unordered_map<std::string, std::pair<std::string,size_t>> dataNameToMeshNameMap;
	//List of acceptable types for meshes
	//std::vector<const std::type_info*> allowedMeshTypes;
	//List of acceptable types for data
	//std::vector<const std::type_info*> allowedDataTypes;

	std::string filename = "";
	bool isOpen=false; //Is there a file open
	bool registerStarted=false; // Has there been a call to register yet
	bool writeStarted=false; // Has the first call to write happened

	bool meshRegistered=false; // Has any mesh been registered
	bool dataRegistered=false; // Has any data been registered
	bool meshWritten=false; // Has any mesh been written
	bool dataWritten=false; // Has any data been written
	bool timeSet=false; // Has the time been set
	bool jobNameSet=false; // Has the job name been set

	bool verbose=false; // Should the writer be verbose

	/**
		Reset the state of the writer. This is called when the file is closed.
		*/
	void clear(){
		meshMap.clear();
		dataMap.clear();
		dataNameToMeshNameMap.clear();
		isOpen=false;
		registerStarted=false;
		writeStarted=false;
		meshRegistered=false;
		dataRegistered=false;
		meshWritten=false;
		dataWritten=false;
		timeSet=false;
		jobNameSet=false;
	}

	/**
	 * @brief Register a type as acceptable as a mesh type
	 * @tparam T The type of the mesh
	 * @throws std::invalid_argument if the type is not allowed by the core writer
	 */
	template<typename T>
	void registerMeshType(){
		//allowedMeshTypes.push_back(&typeid(T));
	}

	/**
	 * @brief Convenience function to register the "typical" mesh types
	 */
	void registerStandardMeshTypes(){
		registerMeshType<float>();
		registerMeshType<double>();
		registerMeshType<int8_t>();
		registerMeshType<uint8_t>();
		registerMeshType<int16_t>();
		registerMeshType<uint16_t>();
		registerMeshType<int32_t>();
		registerMeshType<uint32_t>();
		registerMeshType<int64_t>();
		registerMeshType<uint64_t>();
	}

	/**
	 * @brief Register a type as acceptable as a data type
	 * @tparam T The type of the data
	 * @throws std::invalid_argument if the type is not allowed by the core writer
	 */
	template<typename T>
	void registerDataType(){
		//allowedDataTypes.push_back(&typeid(T));
	}

	/**
	 * @brief Convenience function to register the "typical" data types
	 */
	void registerStandardDataTypes(){
		registerDataType<float>();
		registerDataType<double>();
		registerDataType<int8_t>();
		registerDataType<uint8_t>();
		registerDataType<int16_t>();
		registerDataType<uint16_t>();
		registerDataType<int32_t>();
		registerDataType<uint32_t>();
		registerDataType<int64_t>();
		registerDataType<uint64_t>();
	}

	

	/**
	 * @brief Start the registration process. This should not be called directly, but should be called by startMeshRegister and startDataRegister.
	 * @throws std::runtime_error if the writer is already writing if that is not allowed
	 */
	void startRegister(){
		if (writeStarted && ! registerAfterWrite) throw std::runtime_error("Trying to register a new item after writing has begun");
		if constexpr (has_startRegisterImpl<T_core>) {
			if (!registerStarted) static_cast<T_core*>(this)->startRegisterImpl();
		}

		registerStarted=true;
	}

	/**
	 * @brief Start the mesh registration process. This should be called when a mesh is being registered
	 * @throws std::runtime_error if you are trying to register a mesh after data has been registered
	 */
	void startMeshRegister(){
		startRegister();
		if (registerOrder==meshDataOrder::meshFirst && dataRegistered) throw std::runtime_error("Trying to register a mesh after data has been registered");
		meshRegistered=true;
	}

	/**
	 * @brief Start the data registration process. This should be called when data is being registered
	 */
	void startDataRegister(){
		startRegister();
		if (registerOrder==meshDataOrder::dataFirst && meshRegistered) throw std::runtime_error("Trying to register data after a mesh has been registered");
		dataRegistered=true;
	}

	/**
	 * @brief Start the writing process. This should not be called directly, but should be called by startMeshWrite and startDataWrite.
	 * */
	void startWrite(){
		if constexpr (has_startWriteImpl<T_core>) {
			if (!writeStarted) static_cast<T_core*>(this)->startWriteImpl();
		}
		writeStarted=true;
	}

	/**
	 * @brief Start the writing process. This should be called when a mesh is being written
	 */
	void startMeshWrite(){
		startWrite();
		if (writeOrder==meshDataOrder::meshFirst && dataWritten) throw std::runtime_error("Trying to write a mesh after data has been written");
		meshWritten=true;
	}

	/**
	 * @brief Start the writing process. This should be called when data is being written
	 */
	void startDataWrite(){
		startWrite();
		if (writeOrder==meshDataOrder::dataFirst && meshWritten) throw std::runtime_error("Trying to write data after a mesh has been written");
		dataWritten=true;
	}


	/**
	 * @brief common actions for registering any mesh
	 * @tparam T The type of the mesh
	 * @param name The name of the mesh
	 * @throws std::invalid_argument if a mesh with that name is already registered
	 * @throws std::invalid_argument if the type is not allowed
	 */
	template<typename T>
	void registerMeshCommon(const char* name){
		//using p = typename use_allowedMeshTypes<primitive>::type;
		//static_assert(typeInTuple<T, allowedMeshTypes, 0>(), "Type not allowed");
		if (meshMap.find(name)!=meshMap.end()){
			throw std::invalid_argument("Mesh already registered");
		}
		//Check if the type is allowed
		/*if (!allowedMeshTypes.empty()){ //Not registering any types means all types are allowed
			if (!std::any_of(allowedMeshTypes.begin(), allowedMeshTypes.end(), [](const std::type_info* type){return typeid(T)==*type;})){
				throw std::invalid_argument("Type not allowed");
			}
		}*/
		
		//Create the empty data entry for this mesh
		if (dataMap.find(name)==dataMap.end()){
			dataMap[name] = std::unordered_map<std::string, dataInfo>();
		}
	}

	template<typename T>
	void registerDataCommon(const char* name, const char* meshName){
		testMeshRegistered(meshName);	
		if (dataMap[meshName].find(name)!=dataMap[meshName].end()){
			throw std::invalid_argument("Data already registered");
		}
		//Check if the type is allowed
		/*if (!allowedDataTypes.empty()){ //Not registering any types means all types are allowed
			if (!std::any_of(allowedDataTypes.begin(), allowedDataTypes.end(), [](const std::type_info* type){return typeid(T)==*type;})){
				throw std::invalid_argument("Type not allowed");
			}
		}*/
	}

	/**
	 * @brief Check if a mesh is registered.
	 * @param name The name of the mesh
	 * @throws std::invalid_argument if the mesh is not registered
	 */
	bool testMeshRegistered(const char* name){
		return meshMap.find(name)!=meshMap.end();
	}

	/**
	 * @brief Check if a dataset is registered to a specific mesh.
	 * @param meshName The name of the mesh
	 * @param dataName The name of the data
	 * @throws std::invalid_argument if the data is not registered
	 */
	void testDataOnMesh(const char* meshName, const char* dataName){
		testMeshRegistered(meshName);
		if (dataMap[meshName].find(dataName)==dataMap[meshName].end()){
			throw std::invalid_argument("Data not registered");
		}
	}

	/**
	 * @brief Check if a dataset is uniquely registered.
	 * @param dataName The name of the dataset
	 * @throws std::invalid_argument if the data is not registered
	 * @throws std::logic_error if the data is not uniquely registered
	 */
	void testDataUnique(const char* dataName){
		if (dataNameToMeshNameMap.find(dataName)==dataNameToMeshNameMap.end()){
			throw std::invalid_argument("Data not registered");
		}
		auto &meshInfo = dataNameToMeshNameMap[dataName];
		if (meshInfo.second>1){
			throw std::logic_error("Data is not uniquely registered");
		}
	}	

	protected:
	writer(){}

	public:
	~writer(){
		if (isOpen) closeFile();
	}
	/**
	 * @brief Open a file for writing. This should be called before any other functions are called.
	 */
	void openFile(const char* filename) {
		if (isOpen) throw std::runtime_error("Writer is already connected to a file");
		clear();
		this->filename = filename;
		//Should this be optional? Is it meaningful to not have an openFileImpl?
		if constexpr (has_openFileImpl<T_core, decltype(filename)>) {
			static_cast<T_core*>(this)->openFileImpl(filename);
		}
		isOpen=true;
	}

	/**
	 * @brief Set the timestep/iteration. This should be called before any data or mesh is registered unless a writer supports multiple timesteps in a single file. If it supports multiple timesteps then this function being called should mean that a new timestep is being started.
	 * @param time The time to set
	 * @param iteration The iteration to set. This is optional and defaults to 0.
	 */
	void setTime(double time, size_t iteration=0){
		if constexpr (!supportsMultipleTimesteps) {
			if (meshRegistered || dataRegistered) throw std::runtime_error("Trying to set time after mesh or data has been registered");
			if (timeSet) throw std::runtime_error("Trying to set time more than once");
		}
		timeSet=true;
		if constexpr (has_setTimeImpl<T_core, decltype(time), decltype(iteration)>) {
			static_cast<T_core*>(this)->setTimeImpl(time, iteration);
		}
	}

	/**
	 * @brief Set the job name. This is completely optional and a writer must not require it.
	 * @param jobName The job name to set
	 */
	void setJobName(const char* jobName){
		if (jobNameSet) throw std::runtime_error("Trying to set job name more than once");
		jobNameSet=true;
		if constexpr (has_setJobName<T_core, decltype(jobName)>) {
			static_cast<T_core*>(this)->setJobName(jobName);
		}
	}

	/**
	 * @brief Set a key-value pair in the global scope. This is optional and a writer must not require it.
	 * @param key The key to set
	 * @param value The value to set
	 */
	void setKeyValue(const char* key, const char* value){
		if (verbose && !supportsKeyValue) std::cout << "Warning: Writer does not support key-value pairs, ignoring setKeyValue call\n";
		if constexpr (has_setKeyValue<T_core, decltype(key), decltype(value)>) {
			static_cast<T_core*>(this)->setKeyValue(key, value);
		}
	}

	/**
	 * @brief Set a key-value pair on a specific mesh. This is optional and a writer must not require it.
	 * @param meshName The name of the mesh
	 * @param key The key to set
	 * @param value The value to set
	 * @throws std::runtime_error if the mesh is not registered
	 */
	void setMeshKeyValue(const char* meshName, const char* key, const char* value){
		if (verbose && !supportsMeshKeyValue) std::cout << "Warning: Writer does not support mesh key-value pairs, ignoring setMeshKeyValue call\n";
		testMeshRegistered(meshName);
		if constexpr (has_setMeshKeyValue<T_core, decltype(meshName), decltype(key), decltype(value)>) {
			static_cast<T_core*>(this)->setMeshKeyValue(meshName, key, value);
		}
	}

	/**
	 * @brief Set a key-value pair on a specific dataset. This is optional and a writer must not require it.
	 * @param meshName The name of the mesh
	 * @param dataName The name of the data
	 * @param key The key to set
	 * @param value The value to set
	 */
	void setDataKeyValue(const char* meshName, const char* dataName, const char* key, const char* value){
		if (verbose && !supportsDataKeyValue) std::cout << "Warning: Writer does not support data key-value pairs, ignoring setDataKeyValue call\n";
		testDataOnMesh(meshName, dataName);
		if constexpr (has_setDataKeyValue<T_core, decltype(meshName), decltype(dataName), decltype(key), decltype(value)>) {
			static_cast<T_core*>(this)->setDataKeyValue(meshName, dataName, key, value);
		}
	}

	/**
	 * @brief Register an ND rectilinear mesh. This should be called before any data is written.
	 * @param name The name of the mesh
	 * @param args The number of points in each dimension (must be of integral type)
	 * @throws std::invalid_argument if the mesh is already registered
	 */
	template<typename T_mesh = double, typename... Args>
	void registerRectilinearMesh(const char* name, Args... args){
		startMeshRegister();
		registerMeshCommon<T_mesh>(name);
		meshMap[name].specificInfo = rlMeshInfo(static_cast<size_t>(args)...);
		meshMap[name].type = &typeid(T_mesh);
		auto &info = meshMap[name];
		auto &specificInfo = std::get<rlMeshInfo>(info.specificInfo);
		meshMap[name].zones = specificInfo.zones;

		//This is the specific call for rectilinear meshes
		/*if constexpr (has_registerRectilinearMeshImpl<T_core, decltype(name), decltype(specificInfo), decltype(static_cast<size_t>(args))...>) {*/
			static_cast<T_core*>(this)->registerRectilinearMeshImpl(name, specificInfo, 
				static_cast<size_t>(args)...);
		//}
		//This is a generic call that will be called for any mesh
		if constexpr(has_registerMeshImpl<T_core, decltype(name), decltype(info)>) {
			static_cast<T_core*>(this)->registerMeshImpl(name, info, static_cast<size_t>(args)...);
		}
	}

	/**
	 * @brief Write a ND rectilinear mesh. This should be called after the mesh has been registered.
	 * This it to support writers like NetCDF that require the mesh to be defined before it is written.
	 * @param name The name of the mesh
	 * @param args The coordinates of the mesh in each dimension (must be of pointer type). Must be as many arguments as the mesh rank. Must be of arithmetic type and the same type as the mesh when it was registered.
	 * @throws std::runtime_error if the mesh is not registered
	 * @throws std::runtime_error if the mesh has already been written
	 * @throws std::runtime_error if the mesh rank does not match the rank of the mesh when it was registered
	 */

	template <typename... T_data>
		void writeRectilinearMesh(const char* name, const T_data... args){
			static_assert(sizeof...(args) <= maximumRank, "Too many dimensions for rectilinear mesh");
			static_assert((std::is_arithmetic_v<std::remove_pointer_t<T_data>> && ...), "All arguments must be arithmetic types");
			static_assert((std::is_pointer_v<T_data> && ...), "All arguments must be pointers");
			testMeshRegistered(name);
			auto &meshInfo = meshMap[name];
			auto &meshSpecificInfo = [&meshInfo]() -> auto& {
				try {
					return std::get<rlMeshInfo>(meshInfo.specificInfo);
				} catch (const std::bad_variant_access&) {
					throw std::runtime_error("Trying to write a mesh as a different type than at registration");
				}
			}();
			if ((... || (&typeid(std::decay_t<std::remove_pointer_t<T_data>>) != meshInfo.type))) throw std::invalid_argument("Type of mesh when writing does not match type of mesh when registering");
			if (meshInfo.written) throw std::runtime_error("Mesh has already been written!");
			if (meshSpecificInfo.rank != sizeof...(args)) throw std::runtime_error("Mesh defined rank and written rank do not match");
			startMeshWrite();
			static_cast<T_core*>(this)->writeRectilinearMeshImpl(name, meshInfo, args...);
			meshInfo.written=true;
		}

	/**
	 * @brief Register data against a mesh. This should be called before any data is written.
	 * @template T_data The type of the data. This must be a type that the underlying writer can handle.
	 * Stick to basic types like int, float, double, etc. and you'll be fine.
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @throws std::invalid_argument if the data is already registered
	 * @throws std::runtime_error if the mesh is not registered
	 * @throws std::runtime_error if the data type does not match the registered type
	 * @throws std::runtime_error if the mesh is not registered
	 * @throws std::runtime_error if the data is not registered
	 */
	template<typename T_data=double>
		void registerData(const char* name, const char* meshName){
			registerDataCommon<T_data>(name, meshName);
			if (dataMap[meshName].find(name)!=dataMap[meshName].end()) throw std::invalid_argument("Data already registered");
			if constexpr (!supportsPerMeshDataNames) {
				//If we don't have support for per mesh data names, we need to check that the name is unique across all meshes and throw an exception if it is not
				if (dataNameToMeshNameMap.find(name)!=dataNameToMeshNameMap.end()) throw std::invalid_argument("Data name must be globally unique across all meshes");
				dataNameToMeshNameMap[name]=std::make_pair(meshName,1);
			} else {
				//If we do have support for per mesh data names then we need to keep a count of the number of uses of the name. If > 1 then trying to with
				auto & pair = dataNameToMeshNameMap[name];
				pair.first = meshName;
				pair.second++;
			} 
			dataMap[meshName][name] = dataInfo(meshName, sizeof(T_data), &typeid(std::decay_t<T_data>));
			auto &dataInfo = dataMap[meshName][name];
			auto &meshInfo = meshMap[meshName];
			startDataRegister();
			static_cast<T_core*>(this)->registerDataImpl(name, meshName, dataInfo, meshInfo);
		}

	/**
	 * @brief Register data with type info provided by a pointer to the data. This should be called before any data is written. DOES NOT WRITE DATA
	 * @template T_data The type of the data. This must be a type that the underlying writer can handle.
	 * Stick to basic types like int, float, double, etc. and you'll be fine.
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param data A data pointer having the type of the data. This is only used to get the type info for the data.
	 * @details This is a convenience function that allows you to register data without having to specify the type. The type is determined from the pointer. This is because the "object.template method" syntax is not very clear
	 */
	template<typename T>
	void registerData(const char* name, const char* meshName, [[maybe_unused]] T* data)
	{
		registerData<T>(name, meshName);
	}

	/**
	 * @brief Write data against a mesh. This should be called after the data has been registered.
	 * @template T_data The type of the data. This must be a type that the underlying writer can handle.
	 * Stick to basic types like int, float, double, etc. and you'll be fine.
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param data The data to be written
	 * @throws std::invalid_argument if the data is not registered
	 * @throws std::runtime_error if the data type does not match the registered type
	 */
	template<typename T_data>
		void writeData(const char* name, const char* meshName, const T_data *data) {
			testDataOnMesh(meshName, name);
			auto &dataInfo = dataMap[meshName][name];
			auto &meshInfo = meshMap[dataInfo.meshName];
			if (dataInfo.written) {
				std::cerr << "Data item " << name << " on mesh " << meshName << " has already been written and is being written again!\n";
			}
			dataInfo.written=true;
			if (&typeid(std::decay_t<T_data>)!=dataInfo.type) throw std::invalid_argument("Type of data when writing does not match type of data when registering");
			startDataWrite();
			static_cast<T_core*>(this)->writeDataImpl(name, data, dataInfo, meshInfo);
		}

	/**
	 * @brief Write data against a mesh. This should be called after the data has been registered.
	 * This is a convenience function that allows you to write data without having to specify the mesh name. The mesh name is determined from the data name. This is only valid if the data name is unique across all meshes.
	 * @template T_data The type of the data. This must be a type that the underlying writer can handle.
	 * Stick to basic types like int, float, double, etc. and you'll be fine.
	 * This version is a convenience function where the mesh name is not required. It only works if the data name is unique
	 * @param name The name of the data
	 * @param data The data to be written
	 * @throws std::invalid_argument if the data is not registered
	 * @throws std::runtime_error if the data type does not match the registered type
	 */
	template<typename T_data>
		void writeData(const char* name, const T_data *data) {
			//Find the mesh name from the data name
			testDataUnique(name);
			const auto& pair = dataNameToMeshNameMap[name];
			std::string meshName = pair.first;
			writeData(name, meshName.c_str(), data);
		}

	//Flush file to disk
	void flushFile() {
		static_cast<T_core*>(this)->flushFileImpl();
	}

	//Close file
	void closeFile() {

		bool errorState = false;

		//Loop over all meshes
		for (auto &pair : meshMap) {
			auto &meshInfo = pair.second;
			if (!meshInfo.written) {
				if (!errorState) {
					errorState = true;
					std::cerr << "\n******************************\n**** ERRORS ON FILE CLOSE ****\n******************************\n\n";
				}
				std::cerr << "Mesh " << pair.first << " was defined but has not been written!\n";
			}
		}
		//Loop over all data
		for (auto &meshPair : dataMap) {
			auto &meshName = meshPair.first;
			auto &dataMap = meshPair.second;
			for (auto &dataPair : dataMap) {
				auto &dataInfo = dataPair.second;
				if (!dataInfo.written) {
					if (!errorState) {
						errorState = true;
						std::cerr << "\n******************************\n**** ERRORS ON FILE CLOSE ****\n******************************\n\n";
					}
					std::cerr << "Dataset " << dataPair.first << " on mesh " << meshName << " was defined but has not been written!\n";
				}
			}
		}
	
		static_cast<T_core*>(this)->closeFileImpl();
		clear();
	}
};

class demoFile : public writer<demoFile> {
	public:
	constexpr static const char* writerName = "demoFile";
	constexpr static bool registerAfterWrite=false;
	const static meshDataOrder registerOrder=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder=meshDataOrder::noOrder;
	//private:
	friend class writer<demoFile>;
	using writerMeshInfo = writer<demoFile>::meshInfo;
	using writerRLMeshInfo  = writer<demoFile>::rlMeshInfo;
	using writerDataInfo = writer<demoFile>::dataInfo;

	using allowedMeshTypes_impl = std::tuple<float, double>;

	void openFileImpl(const char* filename){std::cout << "Opening " << filename << "\n";}

	void startRegisterImpl(){}
	void startWriteImpl(){}

	template<typename... Args>
	void registerRectilinearMeshImpl(const char* name,[[maybe_unused]] writerRLMeshInfo &info, Args... args){
		std::cout << "Rectilinear mesh registration called\n";
		std::cout << "Registering mesh \"" << name << "\" with rank " << sizeof...(args) << "\n";
		//Print the sizes using a fold
		std::cout << "Sizes: ";
		((std::cout << args << " "), ...);
		std::cout << "\n";
	}

	template<typename... Args>
	void registerMeshImpl(const char* name, [[maybe_unused]] writerMeshInfo &info, Args... args){
		std::cout << "Generic registerMeshImpl called\n";
		std::cout << "Registering mesh \"" << name << "\" with rank " << sizeof...(args) << "\n";
		//Print the sizes using a fold
		std::cout << "Sizes: ";
		((std::cout << args << " "), ...);
		std::cout << "\n";
	}

	//Write 3d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name, [[maybe_unused]] writerMeshInfo &info, [[maybe_unused]] const T_data *x, [[maybe_unused]] const T_data *y, [[maybe_unused]] const T_data *z){
			auto specificInfo = std::get<rlMeshInfo>(info.specificInfo);
			std::cout << "Writing mesh \"" << name << "\" " << specificInfo.sizes[0] << "x"<<specificInfo.sizes[1]<<"x"<<specificInfo.sizes[2]<<"\n";
		}
	//Write 2d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name, [[maybe_unused]] writerMeshInfo &info, [[maybe_unused]] const T_data *x, [[maybe_unused]] const T_data *y) {
			auto specificInfo = std::get<rlMeshInfo>(info.specificInfo);
			std::cout << "Writing mesh \"" << name << "\" " << specificInfo.sizes[0] << "x"<<specificInfo.sizes[1]<<"\n";
		}
	//Write 1d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name, [[maybe_unused]] writerMeshInfo &info, [[maybe_unused]] const T_data *x){
			auto specificInfo = std::get<rlMeshInfo>(info.specificInfo);
			std::cout << "Writing mesh \"" << name << "\" " << specificInfo.sizes[0] << "\n";
		}

	void registerDataImpl(const char* name, const char* meshName, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo){
		auto specificMeshInfo = std::get<rlMeshInfo>(meshInfo.specificInfo);
		std::cout << "Registering data \"" << name << "\" on mesh \"" << meshName << "\" with size " << specificMeshInfo.sizes[0];
		if (specificMeshInfo.rank > 1) {
			std::cout << "x" << specificMeshInfo.sizes[1];
		}
		if (specificMeshInfo.rank > 2) {
			std::cout << "x" << specificMeshInfo.sizes[2];
		}
		std::cout << "\n";
	}

	//Write data against a mesh
	template<typename T_data>
		void writeDataImpl(const char* name, [[maybe_unused]] const T_data *data, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
			auto specificMeshInfo = std::get<rlMeshInfo>(meshInfo.specificInfo);
			std::cout << "Writing data \"" << name << "\" on mesh \"" << dataInfo.meshName << "\" with size " << specificMeshInfo.sizes[0];
			if (specificMeshInfo.rank > 1) {
				std::cout << "x" << specificMeshInfo.sizes[1];
			}
			if (specificMeshInfo.rank > 2) {
				std::cout << "x" << specificMeshInfo.sizes[2];
			}
			std::cout << ". Total count of " << meshInfo.zones << " zones\n";
		}

	//Flush file to disk
	void flushFileImpl() {
		std::cout << "Flushing file\n";
	}

	//Close file
	void closeFileImpl() {
		std::cout << "Closing file " << this->filename << "\n";
	}
};

#endif
