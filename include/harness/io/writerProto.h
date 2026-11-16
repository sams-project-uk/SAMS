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
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "harness.h"
#include "types.h"
#include "KVS/kvs.h"

namespace SAMS::IO
{

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

//The part of a distributed dataset that this rank holds, in each dimension. The size of the whole dataset is the size of its mesh
struct decomposition{
	//The size of the local array, including ghost cells
	std::array<size_t, SAMS::MAX_RANK> localSizes{};
	//The first element of the local array that is written
	std::array<size_t, SAMS::MAX_RANK> localStarts{};
	//The number of elements that are written
	std::array<size_t, SAMS::MAX_RANK> counts{};
	//The position of the written elements in the whole dataset
	std::array<size_t, SAMS::MAX_RANK> globalStarts{};
	//The communicator of the ranks that write. It is MPI_COMM_NULL if this rank does not write
	SAMSMPI::comm comm = SAMSMPI::commNull;
};

//A dump that a writer opens. Writers build its file name from it, and can use the rest to decide how to write it
struct dumpInfo{
	//The output directory
	std::string directory;
	//The suffix of the files of the dump. Empty for the main diagnostics files
	std::string suffix;
	//The number of the dump among the dumps with its suffix
	size_t number = 0;
	//The name of a named dump, relative to the output directory. Empty if the dump is not named
	std::string name;
	//Whether the dump is a restart dump
	bool restart = false;
	//The simulation time of the dump
	T_dataType time = 0.0;
	//The simulation step of the dump
	T_indexType step = 0;
	//Whether a writer that can combine dumps into one file with multiple timesteps may try to. A file is always written the same way
	bool attemptTimeCoalescence = false;
};

//The types that a key-value pair in the global scope can have
template<typename T>
inline constexpr bool isKeyValueType = std::is_same_v<T, double> || std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, std::string>;

//The type that a step is stored as, which has the signedness of T_indexType
using stepType = std::conditional_t<std::is_signed_v<T_indexType>, int64_t, uint64_t>;

/**
 * @brief The name of a dump without the extension that its writer adds
 * @param info The dump
 */
inline std::string baseName(const dumpInfo &info){
	if (!info.name.empty()) return info.directory + "/" + info.name;
	std::ostringstream number;
	number << std::setw(5) << std::setfill('0') << info.number;
	return info.directory + "/diagnostics_" + (info.suffix.empty() ? "" : info.suffix + "_") + number.str();
}

//Values of the metavariables in a file name
struct metavariables{
	//N
	size_t dumpNumber = 0;
};

//Which metavariables to remove from a file name
struct metavariableFlags{
	//N
	bool dumpNumber = false;
};

//Expands and removes the {X...} metavariables in file names
class metavariableName{
	//The member for a metavariable character, or nullptr if the character is not a metavariable
	template<typename T>
	static auto member(T &s, char c) -> decltype(&s.dumpNumber){
		switch (c){
			case 'N': return &s.dumpNumber;
			default: return nullptr;
		}
	}

public:
	/**
	 * @brief Expand the {X...} metavariables in a file name
	 * @param name The file name
	 * @param values The metavariable values
	 */
	static std::string expand(const std::string &name, const metavariables &values){
		std::ostringstream expanded;
		size_t pos = 0;
		for (size_t open = name.find('{'); open != std::string::npos; open = name.find('{', pos)){
			const size_t close = name.find('}', open);
			expanded << name.substr(pos, open - pos);
			const std::string sequence = name.substr(open + 1, close - open - 1);
			for (size_t start = 0, end; start < sequence.size(); start = end){
				end = sequence.find_first_not_of(sequence[start], start);
				if (end == std::string::npos) end = sequence.size();
				const size_t *value = member(values, sequence[start]);
				if (value) expanded << std::setw(static_cast<int>(end - start)) << std::setfill('0') << *value;
				else expanded << sequence.substr(start, end - start);
			}
			pos = close + 1;
		}
		expanded << name.substr(pos);
		return expanded.str();
	}

	/**
	 * @brief Remove the flagged {X...} metavariables from a file name. A {} left with no metavariables is removed
	 * @param name The file name
	 * @param flags The metavariables to remove
	 */
	static std::string removeMetadata(const std::string &name, const metavariableFlags &flags){
		std::string result;
		size_t pos = 0;
		for (size_t open = name.find('{'); open != std::string::npos; open = name.find('{', pos)){
			const size_t close = name.find('}', open);
			result += name.substr(pos, open - pos);
			std::string kept;
			bool special = false;
			for (size_t i = open + 1; i < close; ++i){
				const bool *flag = member(flags, name[i]);
				if (flag && *flag) continue;
				kept += name[i];
				if (flag) special = true;
			}
			if (special) result += "{" + kept + "}";
			pos = close + 1;
		}
		result += name.substr(pos);
		return result;
	}
};

/**
 * @brief Split a name into its namespace and the name in that namespace. Namespaces are separated by "/"
 * @param name The name
 * @return The namespace, which is empty if the name has none, and the name in that namespace
 */
inline std::pair<std::string, std::string> splitNamespace(const std::string &name){
	const size_t split = name.rfind('/');
	if (split == std::string::npos) return {"", name};
	return {name.substr(0, split), name.substr(split + 1)};
}

/**
 * @brief The decomposition of the part of a distributed array that a localVariablePart describes
 * @param part The part of the array, and where it is in the whole array
 */
template<typename T_data, int R>
decomposition partDecomposition(const localVariablePart<T_data, R> &part){
	decomposition decomp;
	for (int i = 0; i < R; ++i){
		const SIGNED_INDEX_TYPE lb = part.array.getLowerBound(i);
		decomp.localSizes[i] = static_cast<size_t>(part.array.getUpperBound(i) - lb + 1);
		decomp.localStarts[i] = static_cast<size_t>(part.sourceLower[i] - lb);
		decomp.counts[i] = static_cast<size_t>(part.sourceUpper[i] - part.sourceLower[i] + 1);
		decomp.globalStarts[i] = part.globalStarts[i];
	}
	return decomp;
}

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


	//This public section describes the capabilities of the writer
	//It should be used by the ouput code if it needs to write data
	//differently for different writers. THIS SHOULD BE AN UNUSUAL CASE!

	//What is the name of the writer. This is not given a
	//default value because it is an error to not have it
	constexpr static std::string_view writerName = T_core::writerName;
	//The extension of the files that the writer writes
	constexpr static std::string_view extension = T_core::extension;

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
	//Does the writer support true namespaces, which are the parts of names separated by "/". A writer that does not support them flattens the names
	DEFAULTVALUE(supportsTrueNamespaces, bool, false);

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
	//Implementation for setting a job name, completely optional and a writer must not require it
	HASMEMBER(setJobName);
	//Implementation for setting a key-value pair in the global scope.
	HASMEMBER(setKeyValueImpl);
	//Implementation for writing the deck
	HASMEMBER(writeDeckImpl);
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
	 * @brief Structure to hold info about a rectilinear mesh. Each dimension of the mesh is a range of cells of an axis
	 * @param rank The rank of the mesh
	 * @param axes The name of the axis in each dimension
	 * @param starts The core index of the first plotted cell of the axis in each dimension
	 * @param sizes The number of cells of the axis in each dimension, which is the size of the mesh in that dimension
	 */
	struct rlMeshInfo{
		size_t rank = 0;
		std::array<std::string, maximumRank> axes;
		std::array<SIGNED_INDEX_TYPE, maximumRank> starts{};
		std::array<size_t, maximumRank> sizes{};
		size_t zones = 0;
	};

	struct meshInfo {
		bool written = false;
		size_t zones;
		std::variant<rlMeshInfo> specificInfo;
		//Key-value pairs that are written with the mesh
		std::map<std::string, std::string> keyValues;
	};

	//Structure to hold data info
	struct dataInfo{
		std::string meshName;
		size_t elementSize;		
		const std::type_info* type = nullptr;
		std::array<size_t, maximumRank> staggers;
		//Rank, and the core index of the first element and number of elements in each dimension
		size_t rank = 0;
		std::array<SIGNED_INDEX_TYPE, maximumRank> starts{};
		std::array<size_t, maximumRank> sizes{};
		bool written = false;
		//Key-value pairs that are written with the data
		std::map<std::string, std::string> keyValues;
		dataInfo() = default;
		dataInfo(const char* meshName, const size_t elementSize, const std::type_info* type) : meshName(meshName), elementSize(elementSize), type(type){};
	};

	//Map from mesh name to mesh info
	std::unordered_map<std::string, meshInfo> meshMap;
	//Map from data name to data info
	std::unordered_map<std::string, std::unordered_map<std::string,dataInfo>> dataMap;
	//For writers that have globally unique names, map from data name to mesh name
	std::unordered_map<std::string, std::pair<std::string,size_t>> dataNameToMeshNameMap;

	//Structure to hold blob info. A blob is a distributed array that is not associated with a mesh
	struct blobInfo{
		size_t rank;
		std::array<size_t, maximumRank> sizes;
		size_t elementSize;
		const std::type_info* type = nullptr;
		bool written = false;
		//Key-value pairs that are written with the blob
		std::map<std::string, std::string> keyValues;
	};

	//Map from blob name to blob info. Blob names are their own namespace
	std::unordered_map<std::string, blobInfo> blobMap;

	//Structure to hold axis info. An axis is a whole one dimensional array, which meshes refer to by name
	struct axisInfo{
		//Core index of the first element
		SIGNED_INDEX_TYPE start = 0;
		size_t size = 0;
		//The number of ghost cells below and above the domain
		size_t lowerGhosts = 0;
		size_t upperGhosts = 0;
		size_t elementSize = 0;
		const std::type_info* type = nullptr;
		bool written = false;
		//Key-value pairs that are written with the axis
		std::map<std::string, std::string> keyValues;
	};

	//Map from axis name to axis info
	std::unordered_map<std::string, axisInfo> axisMap;

	//Structure to hold info about a file name
	struct nameInfo{
		size_t dumpNumber = 0;
	};

	//Map from file name, relative to the output directory and with its metavariables, to name info. Kept between files
	std::map<std::string, nameInfo> nameMap;
	//Whether the run was restarted from a restart dump
	bool restartedRun = false;
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
	bool jobNameSet=false; // Has the job name been set

	bool verbose=false; // Should the writer be verbose

	/**
		Reset the state of the writer. This is called when the file is closed.
		*/
	void clear(){
		meshMap.clear();
		dataMap.clear();
		dataNameToMeshNameMap.clear();
		blobMap.clear();
		axisMap.clear();
		isOpen=false;
		registerStarted=false;
		writeStarted=false;
		meshRegistered=false;
		dataRegistered=false;
		meshWritten=false;
		dataWritten=false;
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
	 */
	void registerMeshCommon(const char* name){
		if (meshMap.find(name)!=meshMap.end()){
			throw std::invalid_argument("Mesh already registered");
		}
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

	/**
	 * @brief Build the decomposition of the part of a distributed array that this rank holds, and the communicator of the ranks that write it
	 * @param part The part of the array that this rank holds, and where it is in the whole array
	 * @param decomp The decomposition to fill. Its communicator is MPI_COMM_NULL if this rank does not write, and is kept by the MPIManager
	 */
	template<typename T_data, int R>
	void makeDecomposition(const localVariablePart<T_data, R> &part, decomposition &decomp){
		decomp = partDecomposition(part);
		//Ranks along an MPI direction that the array is not split across hold the same part, so only the first of them writes
		auto &mpi = h.MPIManager;
		const auto dims = mpi.getDims();
		[[maybe_unused]] bool split = false;
		std::array<bool, SAMS::MPI_DECOMPOSITION_RANK> keepDirections;
		for (int d = 0; d < mpi.Dims; ++d){
			keepDirections[d] = part.splitDirections[d] || dims[d] == 1;
			if (!keepDirections[d]) split = true;
		}
		//The communicator of the ranks that write. It is the main communicator unless some ranks do not write
		decomp.comm = mpi.getComm();
		#ifdef USE_MPI
		//Every rank asks for the communicator of the ranks that write, because making it takes every rank. It is MPI_COMM_NULL on the other ranks
		if (split) decomp.comm = mpi.getWriterComm(keepDirections);
		#endif
	}

	protected:
	writer(SAMS::harness &h) : h(h) {
		registerStandardMeshTypes();
		registerStandardDataTypes();
	}

	public:
	/**
	 * @brief Open a file for writing. This should be called before any other functions are called.
	 * @param info The dump. Its name without the writer's extension is filename
	 */
	void openFile(const dumpInfo &info) {
		if (isOpen) throw std::runtime_error("Writer is already connected to a file");
		clear();
		const std::string name = baseName(info);
		metavariables values;
		values.dumpNumber = nameMap[info.name].dumpNumber++;
		//Time coalesced output goes to one file, so it has no dump number
		metavariableFlags flags;
		flags.dumpNumber = info.attemptTimeCoalescence && supportsMultipleTimesteps;
		this->filename = metavariableName::expand(metavariableName::removeMetadata(name, flags), values) + std::string(extension);
		//Should this be optional? Is it meaningful to not have an openFileImpl?
		if constexpr (has_openFileImpl<T_core, const dumpInfo&>) {
			static_cast<T_core*>(this)->openFileImpl(info);
		}
		isOpen=true;
	}

	/**
	 * @brief Build the decomposition of a distributed output that is not a variable in the variable registry. The local array holds exactly this rank's part of the output
	 * @details A simulationAxis dimension is decomposed like that axis of the axis registry at its staggering. An integral dimension is that many elements, which every rank holds.
	 * Ranks along an MPI direction that no dimension is decomposed along hold the same part, so only the first of them writes
	 * @param dimensions The dimensions of the output
	 */
	template<typename... T_dimensions>
	decomposition makeDecomposition(const T_dimensions&... dimensions){
		static_assert(sizeof...(T_dimensions) >= 1 && sizeof...(T_dimensions) <= maximumRank, "Rank must be between 1 and the maximum rank");
		decomposition decomp;
		//The MPI directions that the output is split across
		std::array<bool, SAMS::MPI_DECOMPOSITION_RANK> splitDirections{};
		size_t i = 0;
		auto addDimension = [&](const auto &dimension){
			using T_dimension = std::decay_t<decltype(dimension)>;
			static_assert(std::is_same_v<T_dimension, simulationAxis> || std::is_integral_v<T_dimension>, "A dimension must be a simulationAxis or an integral size");
			if constexpr (std::is_same_v<T_dimension, simulationAxis>){
				const SAMS::dimension &dim = h.axisRegistry.getDimension(dimension.name);
				decomp.counts[i] = static_cast<size_t>(dim.getLocalDomainUB(dimension.stagger) - dim.getLocalDomainLB(dimension.stagger) + 1);
				decomp.globalStarts[i] = static_cast<size_t>(dim.getGlobalDomainLB(dimension.stagger) - dim.getDomainLB(dimension.stagger));
				const int mpiAxis = static_cast<int>(dim.mpiAxis);
				if (mpiAxis >= 0) splitDirections[mpiAxis] = true;
			} else {
				decomp.counts[i] = static_cast<size_t>(dimension);
			}
			decomp.localSizes[i] = decomp.counts[i];
			++i;
		};
		(addDimension(dimensions), ...);
		//Ranks along an MPI direction that the output is not split across hold the same part, so only the first of them writes
		auto &mpi = h.MPIManager;
		const auto dims = mpi.getDims();
		[[maybe_unused]] bool split = false;
		std::array<bool, SAMS::MPI_DECOMPOSITION_RANK> keepDirections;
		for (int d = 0; d < mpi.Dims; ++d){
			keepDirections[d] = splitDirections[d] || dims[d] == 1;
			if (!keepDirections[d]) split = true;
		}
		//The communicator of the ranks that write. It is the main communicator unless some ranks do not write
		decomp.comm = mpi.getComm();
		#ifdef USE_MPI
		//Every rank asks for the communicator of the ranks that write, because making it takes every rank. It is MPI_COMM_NULL on the other ranks
		if (split) decomp.comm = mpi.getWriterComm(keepDirections);
		#endif
		return decomp;
	}

	/**
	 * @brief Write the state of the writer to a restart dump
	 * @param target The writer of the restart dump
	 */
	template<typename T_target>
	void writeRestart(T_target &target) const
	{
		const std::string prefix = "writers/" + std::string(writerName) + "/";
		target.template setKeyValue<uint64_t>((prefix + "count").c_str(), nameMap.size());
		size_t i = 0;
		for (const auto &[name, entry] : nameMap)
		{
			target.template setKeyValue<std::string>((prefix + "name_" + std::to_string(i)).c_str(), name);
			target.template setKeyValue<uint64_t>((prefix + "number_" + std::to_string(i)).c_str(), entry.dumpNumber);
			++i;
		}
	}

	/**
	 * @brief Read the state of the writer from a restart dump
	 * @param source The reader of the restart dump
	 */
	template<typename T_source>
	void restart(T_source &source)
	{
		restartedRun = true;
		const std::string prefix = "writers/" + std::string(writerName) + "/";
		//A writer that wrote nothing before the restart has no state
		if (!source.hasKeyValue((prefix + "count").c_str())) return;
		const size_t count = source.template readKeyValue<uint64_t>((prefix + "count").c_str());
		for (size_t i = 0; i < count; ++i)
		{
			const std::string name = source.template readKeyValue<std::string>((prefix + "name_" + std::to_string(i)).c_str());
			nameMap[name].dumpNumber = source.template readKeyValue<uint64_t>((prefix + "number_" + std::to_string(i)).c_str());
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
	 * @tparam T The type that the value is stored as: double, int64_t, uint64_t or std::string. It is always given, and the value is converted to it
	 * @param key The key to set
	 * @param value The value to set
	 */
	template<typename T>
	void setKeyValue(const char* key, const std::type_identity_t<T> &value){
		static_assert(isKeyValueType<T>, "A key-value pair must be a double, int64_t, uint64_t or std::string");
		if (verbose && !supportsKeyValue) std::cout << "Warning: Writer does not support key-value pairs, ignoring setKeyValue call\n";
		if constexpr (has_setKeyValueImpl<T_core, const char*, const T&>) {
			static_cast<T_core*>(this)->template setKeyValueImpl<T>(key, value);
		}
	}

	/**
	 * @brief Set a key-value pair on a specific mesh. The pair is kept until the mesh is written, and is written with it. This is optional and a writer must not require it.
	 * @param meshName The name of the mesh
	 * @param key The key to set
	 * @param value The value to set
	 * @throws std::invalid_argument if the mesh is not registered
	 * @throws std::runtime_error if the mesh has already been written
	 */
	void setMeshKeyValue(const char* meshName, const char* key, const char* value){
		if (verbose && !supportsMeshKeyValue) std::cout << "Warning: Writer does not support mesh key-value pairs, ignoring setMeshKeyValue call\n";
		if (std::string(key).find('/') != std::string::npos) throw std::invalid_argument(std::string("Key ") + key + " of a mesh cannot be in a namespace");
		if (!testMeshRegistered(meshName)) throw std::invalid_argument(std::string("Mesh ") + meshName + " is not registered");
		meshInfo &info = meshMap[meshName];
		if (info.written) throw std::runtime_error(std::string("Mesh ") + meshName + " has already been written, so a key-value pair cannot be set on it");
		info.keyValues[key] = value;
	}

	/**
	 * @brief Set a key-value pair on a specific dataset. The pair is kept until the data is written, and is written with it. This is optional and a writer must not require it.
	 * @param meshName The name of the mesh
	 * @param dataName The name of the data
	 * @param key The key to set
	 * @param value The value to set
	 * @throws std::invalid_argument if the data is not registered
	 * @throws std::runtime_error if the data has already been written
	 */
	void setDataKeyValue(const char* meshName, const char* dataName, const char* key, const char* value){
		if (verbose && !supportsDataKeyValue) std::cout << "Warning: Writer does not support data key-value pairs, ignoring setDataKeyValue call\n";
		if (std::string(key).find('/') != std::string::npos) throw std::invalid_argument(std::string("Key ") + key + " of a dataset cannot be in a namespace");
		testDataOnMesh(meshName, dataName);
		dataInfo &info = dataMap[meshName][dataName];
		if (info.written) throw std::runtime_error(std::string("Data ") + dataName + " on mesh " + meshName + " has already been written, so a key-value pair cannot be set on it");
		info.keyValues[key] = value;
	}

	/**
	 * @brief Set a key-value pair on a specific blob. The pair is kept until the blob is written, and is written with it
	 * @param blobName The name of the blob
	 * @param key The key to set
	 * @param value The value to set
	 * @throws std::invalid_argument if the key is in a namespace
	 */
	void setBlobKeyValue(const char* blobName, const char* key, const char* value){
		if (std::string(key).find('/') != std::string::npos) throw std::invalid_argument(std::string("Key ") + key + " of a blob cannot be in a namespace");
		blobMap.at(blobName).keyValues[key] = value;
	}

	/**
	 * @brief Set a key-value pair on a specific axis. The pair is kept until the axis is written, and is written with it
	 * @param axisName The name of the axis
	 * @param key The key to set
	 * @param value The value to set
	 * @throws std::invalid_argument if the key is in a namespace
	 */
	void setAxisKeyValue(const char* axisName, const char* key, const char* value){
		if (std::string(key).find('/') != std::string::npos) throw std::invalid_argument(std::string("Key ") + key + " of an axis cannot be in a namespace");
		axisMap.at(axisName).keyValues[key] = value;
	}

	/**
	 * @brief Write the deck, as the text of an EPOCH deck
	 * @param deck The deck
	 */
	void writeDeck(const SAMS::KVS &deck){
		if constexpr (has_writeDeckImpl<T_core, const std::string&>) {
			static_cast<T_core*>(this)->writeDeckImpl(SAMS::KVSConversion::toEPOCHString(deck));
		}
	}

	/**
	 * @brief Register an axis. An axis is a whole one dimensional array that meshes refer to by name. It is written by rank 0
	 * @tparam T_axis The type of the axis
	 * @param name The name of the axis, which can be in a namespace
	 * @param start The core index of the first element of the axis
	 * @param size The number of elements of the axis
	 * @param lowerGhosts The number of ghost cells below the domain
	 * @param upperGhosts The number of ghost cells above the domain
	 * @throws std::invalid_argument if the axis is already registered
	 */
	template<typename T_axis>
		void registerAxis(const char* name, SIGNED_INDEX_TYPE start, size_t size, size_t lowerGhosts, size_t upperGhosts){
			if (axisMap.find(name)!=axisMap.end()) throw std::invalid_argument("Axis already registered");
			startRegister();
			axisInfo &info = axisMap[name];
			info.start = start;
			info.size = size;
			info.lowerGhosts = lowerGhosts;
			info.upperGhosts = upperGhosts;
			info.elementSize = sizeof(T_axis);
			info.type = &typeid(std::decay_t<T_axis>);
			static_cast<T_core*>(this)->registerAxisImpl(name, info);
		}

	/**
	 * @brief Write an axis. Every rank holds the whole axis, and rank 0 writes it
	 * @tparam T_axis The type of the axis
	 * @param name The name of the axis
	 * @param data The whole axis
	 */
	template<typename T_axis>
		void writeAxis(const char* name, const T_axis *data){
			axisInfo &info = axisMap.at(name);
			info.written=true;
			startWrite();
			static_cast<T_core*>(this)->writeAxisImpl(name, data, info);
		}

	/**
	 * @brief Register an ND rectilinear mesh. Each dimension of the mesh is a range of cells of an axis. This should be called before any data is written.
	 * @param name The name of the mesh
	 * @param axes The name of the axis in each dimension
	 * @param starts The core index of the first plotted cell of the axis in each dimension
	 * @param sizes The number of plotted cells of the axis in each dimension
	 * @throws std::invalid_argument if the mesh is already registered
	 */
	template<size_t R>
		void registerRectilinearMesh(const char* name, const std::array<std::string, R> &axes, const std::array<SIGNED_INDEX_TYPE, R> &starts, const std::array<size_t, R> &sizes){
			static_assert(R >= 1 && R <= maximumRank, "Mesh rank must be between 1 and the maximum rank");
			startMeshRegister();
			registerMeshCommon(name);
			meshInfo &info = meshMap[name];
			rlMeshInfo specificInfo;
			specificInfo.rank = R;
			specificInfo.zones = 1;
			for (size_t i = 0; i < R; ++i){
				specificInfo.axes[i] = axes[i];
				specificInfo.starts[i] = starts[i];
				specificInfo.sizes[i] = sizes[i];
				specificInfo.zones *= sizes[i];
			}
			info.specificInfo = specificInfo;
			info.zones = specificInfo.zones;
			static_cast<T_core*>(this)->registerRectilinearMeshImpl(name, std::get<rlMeshInfo>(info.specificInfo));
			//This is a generic call that will be called for any mesh
			if constexpr(has_registerMeshImpl<T_core, decltype(name), decltype(info)>) {
				static_cast<T_core*>(this)->registerMeshImpl(name, info);
			}
		}

	/**
	 * @brief Write an ND rectilinear mesh. A mesh holds no values of its own, so this writes which range of which axis is each dimension of the mesh, and the key-value pairs of the mesh
	 * @param name The name of the mesh
	 * @throws std::runtime_error if the mesh has already been written
	 */
	void writeRectilinearMesh(const char* name){
		meshInfo &info = meshMap.at(name);
		if (info.written) throw std::runtime_error("Mesh has already been written!");
		startMeshWrite();
		static_cast<T_core*>(this)->writeRectilinearMeshImpl(name, info);
		info.written=true;
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
			//The data matches its mesh exactly
			const rlMeshInfo &rlInfo = std::get<rlMeshInfo>(meshMap.at(meshName).specificInfo);
			registerDataWithSizes<T_data>(name, meshName, rlInfo.rank, rlInfo.starts.data(), rlInfo.sizes.data());
		}

	/**
	 * @brief Register data against a mesh that it does not match exactly. This should be called before any data is written.
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param starts The core index of the first element in each dimension
	 * @param sizes The number of elements in each dimension
	 */
	template<typename T_data, size_t R>
		void registerData(const char* name, const char* meshName, const std::array<SIGNED_INDEX_TYPE, R> &starts, const std::array<size_t, R> &sizes){
			registerDataWithSizes<T_data>(name, meshName, R, starts.data(), sizes.data());
		}

	/**
	 * @brief Register data with its own starts and sizes
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param rank The rank of the data
	 * @param starts The core index of the first element in each dimension
	 * @param sizes The number of elements in each dimension
	 */
	template<typename T_data>
		void registerDataWithSizes(const char* name, const char* meshName, size_t rank, const SIGNED_INDEX_TYPE *starts, const size_t *sizes){
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
			dataInfo.rank = rank;
			std::copy(starts, starts + rank, dataInfo.starts.begin());
			std::copy(sizes, sizes + rank, dataInfo.sizes.begin());
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
	 * @brief Write the part of a distributed dataset that this rank holds. This should be called after the data has been registered.
	 * @template T_data The type of the data. This must be a type that the underlying writer can handle.
	 * Stick to basic types like int, float, double, etc. and you'll be fine.
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param data The local array that holds the part of the data on this rank
	 * @param part The part of the data that this rank holds, where it is in the local array, and the communicator of the ranks that write
	 * @throws std::invalid_argument if the data is not registered
	 * @throws std::runtime_error if the data type does not match the registered type
	 */
	template<typename T_data>
		void writeData(const char* name, const char* meshName, const T_data *data, const decomposition &part) {
			testDataOnMesh(meshName, name);
			auto &dataInfo = dataMap[meshName][name];
			auto &meshInfo = meshMap[dataInfo.meshName];
			if (dataInfo.written) {
				std::cerr << "Data item " << name << " on mesh " << meshName << " has already been written and is being written again!\n";
			}
			dataInfo.written=true;
			if (&typeid(std::decay_t<T_data>)!=dataInfo.type) throw std::invalid_argument("Type of data when writing does not match type of data when registering");
			startDataWrite();
			static_cast<T_core*>(this)->writeDataImpl(name, data, part, dataInfo, meshInfo);
		}

	/**
	 * @brief Write the part of a distributed dataset that this rank holds. This should be called after the data has been registered.
	 * This is a convenience function that allows you to write data without having to specify the mesh name. The mesh name is determined from the data name. This is only valid if the data name is unique across all meshes.
	 * @template T_data The type of the data. This must be a type that the underlying writer can handle.
	 * Stick to basic types like int, float, double, etc. and you'll be fine.
	 * This version is a convenience function where the mesh name is not required. It only works if the data name is unique
	 * @param name The name of the data
	 * @param data The local array that holds the part of the data on this rank
	 * @param part The part of the data that this rank holds, where it is in the local array, and the communicator of the ranks that write
	 * @throws std::invalid_argument if the data is not registered
	 * @throws std::runtime_error if the data type does not match the registered type
	 */
	template<typename T_data>
		void writeData(const char* name, const T_data *data, const decomposition &part) {
			//Find the mesh name from the data name
			testDataUnique(name);
			const auto& pair = dataNameToMeshNameMap[name];
			std::string meshName = pair.first;
			writeData(name, meshName.c_str(), data, part);
		}

	/**
	 * @brief Write the part of a variable that this rank holds. This should be called after the data has been registered.
	 * The mesh name is determined from the data name, so this is only valid if the data name is unique across all meshes
	 * @param name The name of the data
	 * @param part The part of the variable that this rank holds, and where it is in the whole variable
	 * @throws std::invalid_argument if the rank of the part does not match the rank of the mesh
	 */
	template<typename T_data, int R>
		void writeData(const char* name, const localVariablePart<T_data, R> &part){
			//Find the mesh name from the data name
			testDataUnique(name);
			const std::string meshName = dataNameToMeshNameMap[name].first;
			if (static_cast<size_t>(R) != dataMap[meshName][name].rank) throw std::invalid_argument("Rank of data does not match registered rank");
			decomposition decomp;
			makeDecomposition(part, decomp);
			writeData(name, meshName.c_str(), part.array.data(), decomp);
		}

	/**
	 * @brief Register a blob. A blob is a distributed array that is not associated with a mesh
	 * @tparam T_data The type of the blob
	 * @param name The name of the blob
	 * @param sizes The global size of the blob in each dimension
	 * @throws std::invalid_argument if the blob is already registered
	 */
	template<typename T_data, size_t R>
		void registerBlob(const char* name, const std::array<size_t, R> &sizes){
			static_assert(R >= 1 && R <= maximumRank, "Blob rank must be between 1 and the maximum rank");
			if (blobMap.find(name)!=blobMap.end()) throw std::invalid_argument("Blob already registered");
			startRegister();
			blobInfo &info = blobMap[name];
			info.rank = R;
			std::copy(sizes.begin(), sizes.end(), info.sizes.begin());
			info.elementSize = sizeof(T_data);
			info.type = &typeid(std::decay_t<T_data>);
			static_cast<T_core*>(this)->registerBlobImpl(name, info);
		}

	/**
	 * @brief Write the part of a blob that this rank holds. This should be called after the blob has been registered
	 * @param name The name of the blob
	 * @param part The part of the blob that this rank holds, and where it is in the whole blob
	 * @throws std::invalid_argument if the blob is not registered
	 * @throws std::invalid_argument if the rank or type of the part does not match the registered blob
	 */
	template<typename T_data, int R>
		void writeBlob(const char* name, const localVariablePart<T_data, R> &part){
			auto it = blobMap.find(name);
			if (it==blobMap.end()) throw std::invalid_argument("Blob not registered");
			blobInfo &info = it->second;
			if (static_cast<size_t>(R) != info.rank) throw std::invalid_argument("Rank of blob does not match registered rank");
			if (&typeid(std::decay_t<T_data>)!=info.type) throw std::invalid_argument("Type of blob when writing does not match type of blob when registering");
			if (info.written) {
				std::cerr << "Blob " << name << " has already been written and is being written again!\n";
			}
			info.written=true;
			decomposition decomp;
			makeDecomposition(part, decomp);
			startWrite();
			static_cast<T_core*>(this)->writeBlobImpl(name, part.array.data(), decomp, info);
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
		//Loop over all blobs
		for (auto &pair : blobMap) {
			if (!pair.second.written) {
				if (!errorState) {
					errorState = true;
					std::cerr << "\n******************************\n**** ERRORS ON FILE CLOSE ****\n******************************\n\n";
				}
				std::cerr << "Blob " << pair.first << " was defined but has not been written!\n";
			}
		}

		static_cast<T_core*>(this)->closeFileImpl();
		clear();
	}
};

class demoFile : public writer<demoFile> {
	public:
	constexpr static const char* writerName = "demoFile";
	constexpr static const char* extension = "";
	constexpr static bool registerAfterWrite=false;
	const static meshDataOrder registerOrder=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder=meshDataOrder::noOrder;
	//private:
	friend class writer<demoFile>;
	using writerMeshInfo = writer<demoFile>::meshInfo;
	using writerRLMeshInfo  = writer<demoFile>::rlMeshInfo;
	using writerDataInfo = writer<demoFile>::dataInfo;

	using allowedMeshTypes_impl = std::tuple<float, double>;

	void openFileImpl([[maybe_unused]] const dumpInfo &info){std::cout << "Opening " << this->filename << "\n";}

	void startRegisterImpl(){}
	void startWriteImpl(){}

	void registerAxisImpl(const char* name, [[maybe_unused]] writer<demoFile>::axisInfo &info){
		std::cout << "Registering axis \"" << name << "\" with size " << info.size << "\n";
	}

	template <typename T_data>
		void writeAxisImpl(const char* name, [[maybe_unused]] const T_data *data, [[maybe_unused]] writer<demoFile>::axisInfo &info){
			std::cout << "Writing axis \"" << name << "\" with size " << info.size << "\n";
		}

	void registerRectilinearMeshImpl(const char* name, const writerRLMeshInfo &info){
		std::cout << "Registering mesh \"" << name << "\" with rank " << info.rank << "\n";
		for (size_t i = 0; i < info.rank; ++i){
			std::cout << "Axis " << info.axes[i] << " from cell " << info.starts[i] << " for " << info.sizes[i] << " cells\n";
		}
	}

	void writeRectilinearMeshImpl(const char* name, [[maybe_unused]] writerMeshInfo &info){
		std::cout << "Writing mesh \"" << name << "\"\n";
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
		void writeDataImpl(const char* name, [[maybe_unused]] const T_data *data, [[maybe_unused]] const decomposition &part, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
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

	//Close the file while the members that closing it uses still exist
	~demoFile(){
		if (this->isOpen) this->closeFile();
	}
};

} // namespace SAMS::IO

#endif
