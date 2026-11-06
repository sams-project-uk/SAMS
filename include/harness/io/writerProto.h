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

#include "harness.h"

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

//Name and staggering of one axis of a rectilinear mesh. An empty name means that the dimension is not used
struct meshAxis{
	std::string name;
	SAMS::staggerType stagger = SAMS::staggerType::CENTRED;
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

	SAMS::harness &h;

	private:

	/**
	 * @brief Structure to hold infor about a rectilinear mesh
	 * @param rank The rank of the mesh
	 * @param sizes The global size of the mesh in each dimension
	 * @param axisNames The name of the axis in each dimension
	 * @param staggers The staggering of the mesh in each dimension
	 */
	struct rlMeshInfo{
		size_t rank = 0;
		std::array<size_t, maximumRank> sizes{};
		std::array<std::string, maximumRank> axisNames;
		std::array<SAMS::staggerType, maximumRank> staggers{};
		size_t zones = 1;
	};

	struct meshInfo {
		bool written = false;
		size_t zones;
		const std::type_info* type = nullptr;
		std::variant<rlMeshInfo> specificInfo;
		//Function that writes the mesh with the type of its axes
		void (writer::*writeMeshFn)(const char*, meshInfo&) = nullptr;
	};

	//Structure to hold data info
	struct dataInfo{
		std::string meshName;
		//Name of the variable in the variable registry that the data is taken from
		std::string varName;
		size_t elementSize;
		const std::type_info* type = nullptr;
		std::array<size_t, maximumRank> staggers;
		bool written = false;
		//Function that writes the data with the type of its variable
		void (writer::*writeFn)(const char*, dataInfo&, meshInfo&) = nullptr;
		dataInfo() = default;
		dataInfo(const char* meshName, const char* varName) : meshName(meshName), varName(varName){};
	};

	/**
	 * @brief Structure to hold the part of a variable that this rank writes
	 * @param rank The rank of the variable
	 * @param localSizes The size of the local array, including ghost cells, in each dimension
	 * @param localStarts The first element of the local array that is written in each dimension
	 * @param counts The number of elements written in each dimension
	 * @param globalSizes The size of the whole variable across all ranks in each dimension
	 * @param globalStarts The position of the written elements in the whole variable in each dimension
	 */
	struct blockInfo{
		size_t rank = 0;
		std::array<size_t, maximumRank> localSizes{};
		std::array<size_t, maximumRank> localStarts{};
		std::array<size_t, maximumRank> counts{};
		std::array<size_t, maximumRank> globalSizes{};
		std::array<size_t, maximumRank> globalStarts{};
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
	 * @param name The name of the mesh
	 * @throws std::invalid_argument if a mesh with that name is already registered
	 * @throws std::invalid_argument if the type is not allowed
	 */
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

	/**
	 * @brief Write a registered mesh, taking the coordinates from the axis registry
	 * @tparam T_mesh The type of the axes of the mesh
	 * @param name The name of the mesh
	 * @param info The information about the mesh
	 */
	template<typename T_mesh>
	void writeRectilinearMeshTyped(const char* name, meshInfo &info){
		const rlMeshInfo &specificInfo = std::get<rlMeshInfo>(info.specificInfo);
		//Every rank holds the whole of each axis
		std::array<portableWrapper::hostArray<T_mesh, 1>, maximumRank> axisArrays;
		std::array<const T_mesh*, maximumRank> axes{};
		for (size_t i = 0; i < specificInfo.rank; ++i){
			h.axisRegistry.fillPPAxis(specificInfo.axisNames[i], axisArrays[i], specificInfo.staggers[i]);
			axes[i] = &axisArrays[i](h.axisRegistry.getDimension(specificInfo.axisNames[i]).getDomainLB(specificInfo.staggers[i]));
		}
		static_cast<T_core*>(this)->writeRectilinearMeshImpl(name, info, axes);
	}

	/**
	 * @brief Set the type of a mesh and the function that writes it from the type handle of its axes
	 * @tparam I The index in the list of allowed mesh types of the type to test
	 * @param info The information about the mesh
	 * @param type The type handle of the axes
	 * @throws std::invalid_argument if the type of the axes cannot be written
	 */
	template<size_t I = 0>
	void setMeshType(meshInfo &info, SAMS::typeHandle type){
		if constexpr (I >= std::tuple_size_v<defaultAllowedMeshTypes>) {
			throw std::invalid_argument("Type of mesh axes cannot be written");
		} else {
			using T = std::tuple_element_t<I, defaultAllowedMeshTypes>;
			if (type == SAMS::gettypeRegistry().getTypeID<T>()) {
				info.type = &typeid(T);
				info.writeMeshFn = &writer::template writeRectilinearMeshTyped<T>;
				return;
			}
			setMeshType<I + 1>(info, type);
		}
	}

	/**
	 * @brief Set the type of a dataset and the function that writes it from the type handle of its variable
	 * @tparam I The index in the list of allowed data types of the type to test
	 * @param info The information about the data
	 * @param type The type handle of the variable
	 * @throws std::invalid_argument if the type of the variable cannot be written
	 */
	template<size_t I = 0>
	void setDataType(dataInfo &info, SAMS::typeHandle type){
		if constexpr (I >= std::tuple_size_v<defaultAllowedDataTypes>) {
			throw std::invalid_argument("Type of variable cannot be written");
		} else {
			using T = std::tuple_element_t<I, defaultAllowedDataTypes>;
			if (type == SAMS::gettypeRegistry().getTypeID<T>()) {
				info.elementSize = sizeof(T);
				info.type = &typeid(T);
				info.writeFn = &writer::template writeDataTyped<T>;
				return;
			}
			setDataType<I + 1>(info, type);
		}
	}

	/**
	 * @brief Write a registered dataset from its variable in the variable registry
	 * @tparam T_data The type of the variable
	 * @param name The name of the data
	 * @param dInfo The information about the data
	 * @param mInfo The information about the mesh that the data is on
	 */
	template<typename T_data>
	void writeDataTyped(const char* name, dataInfo &dInfo, meshInfo &mInfo){
		const SAMS::variableDef &varDef = h.variableRegistry.getVariable(dInfo.varName);
		blockInfo block;
		block.rank = static_cast<size_t>(varDef.getRank());
		for (size_t i = 0; i < block.rank; ++i){
			const SAMS::dimension &dim = varDef.getDimension(static_cast<int>(i));
			//The local array is all of the memory of the variable on this rank, including ghost cells
			block.localSizes[i] = static_cast<size_t>(dim.getLocalUB() - dim.getLocalLB() + 1);
			block.localStarts[i] = static_cast<size_t>(dim.getLocalDomainLB() - dim.getLocalLB());
			block.counts[i] = dim.getLocalDomainCount();
			block.globalSizes[i] = dim.getDomainCount();
			block.globalStarts[i] = static_cast<size_t>(dim.getGlobalDomainLB() - dim.getDomainLB());
		}
		writeDataOnHost<T_data, 1>(name, varDef, block, dInfo, mInfo);
	}

	/**
	 * @brief Copy the data of a variable to the host if it is not already there and write it. The copy is released as soon as it is written
	 * @tparam T_data The type of the data
	 * @tparam R The rank to test the variable against
	 * @param name The name of the data
	 * @param varDef The variable to write
	 * @param block The part of the variable that this rank writes
	 * @param dInfo The information about the data
	 * @param mInfo The information about the mesh that the data is on
	 */
	template<typename T_data, int R>
	void writeDataOnHost(const char* name, const SAMS::variableDef &varDef, const blockInfo &block, dataInfo &dInfo, meshInfo &mInfo){
		if constexpr (R > SAMS::MAX_RANK) {
			throw std::runtime_error("Rank of variable is too high to write");
		} else {
			if (varDef.getRank() != R) {
				writeDataOnHost<T_data, R + 1>(name, varDef, block, dInfo, mInfo);
				return;
			}
			if (varDef.getMemorySpace() == portableWrapper::arrayTags::host) {
				static_cast<T_core*>(this)->writeDataImpl(name, static_cast<const T_data*>(varDef.getDataPtr<T_data>()), block, dInfo, mInfo);
			} else {
				portableWrapper::portableArrayManager manager;
				portableWrapper::portableArray<T_data, R, portableWrapper::arrayTags::accelerated> device;
				varDef.fillPPArray(device);
				auto host = manager.makeHostAvailable(device);
				static_cast<T_core*>(this)->writeDataImpl(name, static_cast<const T_data*>(host.data()), block, dInfo, mInfo);
				manager.deallocate(host);
			}
		}
	}

	protected:
	writer(SAMS::harness &h) : h(h) {
		registerStandardMeshTypes();
		registerStandardDataTypes();
	}

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
	 * @param axis0 The name and staggering of the axis in the first dimension
	 * @param axis1 - axis7 The name and staggering of the axis in each further dimension. Dimensions that are not used are left with an empty name
	 * @details The global size of the mesh in each dimension is taken from the axis registry. The type of the mesh is the type of its axes
	 * @throws std::invalid_argument if the mesh is already registered
	 * @throws std::invalid_argument if the axes have different types or their type cannot be written
	 */
	void registerRectilinearMesh(const char* name, const meshAxis &axis0, const meshAxis &axis1 = {}, const meshAxis &axis2 = {}, const meshAxis &axis3 = {},
		const meshAxis &axis4 = {}, const meshAxis &axis5 = {}, const meshAxis &axis6 = {}, const meshAxis &axis7 = {}){
		startMeshRegister();
		registerMeshCommon(name);
		const std::array<const meshAxis*, maximumRank> axes = {&axis0, &axis1, &axis2, &axis3, &axis4, &axis5, &axis6, &axis7};
		rlMeshInfo newInfo;
		SAMS::typeHandle meshType;
		for (const meshAxis *axis : axes){
			if (axis->name.empty()) break;
			const SAMS::dimension &dim = h.axisRegistry.getDimension(axis->name);
			//Every axis of the mesh must have the same type
			const SAMS::typeHandle axisType = h.axisRegistry.getAxis(axis->name).axisData.at(axis->stagger).at(portableWrapper::arrayTags::host).axisType;
			if (newInfo.rank == 0) {
				meshType = axisType;
			} else if (axisType != meshType) {
				throw std::invalid_argument("Axes of mesh have different types");
			}
			newInfo.axisNames[newInfo.rank] = axis->name;
			newInfo.staggers[newInfo.rank] = axis->stagger;
			newInfo.sizes[newInfo.rank] = dim.getDomainCount(axis->stagger);
			newInfo.zones *= newInfo.sizes[newInfo.rank];
			newInfo.rank++;
		}
		meshMap[name].specificInfo = newInfo;
		setMeshType(meshMap[name], meshType);
		auto &info = meshMap[name];
		auto &specificInfo = std::get<rlMeshInfo>(info.specificInfo);
		meshMap[name].zones = specificInfo.zones;

		//This is the specific call for rectilinear meshes
		static_cast<T_core*>(this)->registerRectilinearMeshImpl(name, specificInfo);
		//This is a generic call that will be called for any mesh
		if constexpr(has_registerMeshImpl<T_core, decltype(name), decltype(info)>) {
			static_cast<T_core*>(this)->registerMeshImpl(name, info);
		}
	}

	/**
	 * @brief Write a ND rectilinear mesh. This should be called after the mesh has been registered.
	 * This it to support writers like NetCDF that require the mesh to be defined before it is written.
	 * The coordinates of the mesh are taken from the axis registry
	 * @param name The name of the mesh
	 * @throws std::runtime_error if the mesh is not registered
	 * @throws std::runtime_error if the mesh has already been written
	 */
	void writeRectilinearMesh(const char* name){
		if (!testMeshRegistered(name)) throw std::runtime_error("Mesh is not registered");
		auto &meshInfo = meshMap[name];
		if (meshInfo.written) throw std::runtime_error("Mesh has already been written!");
		startMeshWrite();
		(this->*meshInfo.writeMeshFn)(name, meshInfo);
		meshInfo.written=true;
	}

	/**
	 * @brief Register data against a mesh. This should be called before any data is written.
	 * The data is taken from a variable in the variable registry, which also gives its type
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param varName The name of the variable in the variable registry
	 * @throws std::invalid_argument if the data is already registered
	 * @throws std::invalid_argument if the variable does not lie on the axes of the mesh
	 * @throws std::invalid_argument if the type of the variable cannot be written
	 */
		void registerData(const char* name, const char* meshName, const char* varName){
			registerDataCommon(name, meshName);
			//The variable must lie on the axes of the mesh with the same staggering
			const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
			const rlMeshInfo &specificInfo = std::get<rlMeshInfo>(meshMap[meshName].specificInfo);
			if (static_cast<size_t>(varDef.getRank()) != specificInfo.rank) throw std::invalid_argument("Rank of variable does not match rank of mesh");
			for (size_t i = 0; i < specificInfo.rank; ++i){
				const SAMS::dimension &dim = varDef.getDimension(static_cast<int>(i));
				if (dim.axisName != specificInfo.axisNames[i] || dim.stagger != specificInfo.staggers[i]) throw std::invalid_argument("Axes of variable do not match axes of mesh");
			}
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
			dataMap[meshName][name] = dataInfo(meshName, varName);
			auto &dataInfo = dataMap[meshName][name];
			auto &meshInfo = meshMap[meshName];
			//The type of the data is the type of the variable
			setDataType(dataInfo, varDef.getType());
			startDataRegister();
			static_cast<T_core*>(this)->registerDataImpl(name, meshName, dataInfo, meshInfo);
		}

	/**
	 * @brief Write data against a mesh. This should be called after the data has been registered.
	 * The data is taken from the variable that it was registered with. Each rank writes its own part of the variable
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @throws std::invalid_argument if the data is not registered
	 */
	void writeData(const char* name, const char* meshName) {
		testDataOnMesh(meshName, name);
		auto &dataInfo = dataMap[meshName][name];
		auto &meshInfo = meshMap[dataInfo.meshName];
		if (dataInfo.written) {
			std::cerr << "Data item " << name << " on mesh " << meshName << " has already been written and is being written again!\n";
		}
		dataInfo.written=true;
		startDataWrite();
		(this->*dataInfo.writeFn)(name, dataInfo, meshInfo);
	}

	/**
	 * @brief Write data against a mesh. This should be called after the data has been registered.
	 * This is a convenience function that allows you to write data without having to specify the mesh name. The mesh name is determined from the data name. This is only valid if the data name is unique across all meshes.
	 * @param name The name of the data
	 * @throws std::invalid_argument if the data is not registered
	 */
	void writeData(const char* name) {
		//Find the mesh name from the data name
		testDataUnique(name);
		const auto& pair = dataNameToMeshNameMap[name];
		std::string meshName = pair.first;
		writeData(name, meshName.c_str());
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
	using writerBlockInfo = writer<demoFile>::blockInfo;

	using allowedMeshTypes_impl = std::tuple<float, double>;

	void openFileImpl(const char* filename){std::cout << "Opening " << filename << "\n";}

	void startRegisterImpl(){}
	void startWriteImpl(){}

	void registerRectilinearMeshImpl(const char* name, const writerRLMeshInfo &info){
		std::cout << "Rectilinear mesh registration called\n";
		std::cout << "Registering mesh \"" << name << "\" with rank " << info.rank << "\n";
		//Print the sizes
		std::cout << "Sizes: ";
		for (size_t i = 0; i < info.rank; ++i) std::cout << info.sizes[i] << " ";
		std::cout << "\n";
	}

	void registerMeshImpl(const char* name, writerMeshInfo &info){
		const writerRLMeshInfo &specificInfo = std::get<writerRLMeshInfo>(info.specificInfo);
		std::cout << "Generic registerMeshImpl called\n";
		std::cout << "Registering mesh \"" << name << "\" with rank " << specificInfo.rank << "\n";
		//Print the sizes
		std::cout << "Sizes: ";
		for (size_t i = 0; i < specificInfo.rank; ++i) std::cout << specificInfo.sizes[i] << " ";
		std::cout << "\n";
	}

	//Write rectilinear mesh of any rank
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name, writerMeshInfo &info, [[maybe_unused]] const std::array<const T_data*, maximumRank> &axes){
			auto specificInfo = std::get<rlMeshInfo>(info.specificInfo);
			std::cout << "Writing mesh \"" << name << "\" ";
			for (size_t i = 0; i < specificInfo.rank; ++i) std::cout << (i > 0 ? "x" : "") << specificInfo.sizes[i];
			std::cout << "\n";
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
		void writeDataImpl(const char* name, [[maybe_unused]] const T_data *data, [[maybe_unused]] const writerBlockInfo &block, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
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
