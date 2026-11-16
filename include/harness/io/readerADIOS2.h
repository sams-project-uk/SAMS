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
#ifndef READER_ADIOS2_H
#define READER_ADIOS2_H
#include <array>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include "readerProto.h"
#if __has_include(<adios2.h>)
#include <adios2.h>
#include "harnessDef.h"
#include "mpiManager.h"

namespace SAMS::IO
{

/**
 * @brief Reads the files that ADIOS2File writes
 */
class ADIOS2Reader : public reader<ADIOS2Reader> {
	friend class reader<ADIOS2Reader>;

	//The ADIOS2 objects of the open file
	std::unique_ptr<adios2::ADIOS> adios;
	adios2::IO io;
	adios2::Engine engine;
	//The step of the file that is read, by its position in the file
	size_t selectedStep = 0;

	void openFileImpl(const char* filename){
		const std::string bpFilename = std::string(filename) + ".bp";
#ifdef USE_MPI
		//Every rank opens the file
		adios = std::make_unique<adios2::ADIOS>(this->h.MPIManager.getComm());
#else
		adios = std::make_unique<adios2::ADIOS>();
#endif
		io = adios->DeclareIO("SAMSRead");
		//Any step can be read
		engine = io.Open(bpFilename, adios2::Mode::ReadRandomAccess);
		selectedStep = 0;
	}

	void closeFileImpl(){
		engine.Close();
		adios.reset();
	}

	//Whether the file has a variable
	bool hasVariable(const std::string &name){
		return !io.VariableType(name).empty();
	}

	//Read a string value. The metadata and key-value pairs are values named in their item
	std::string readString(const std::string &item, const std::string &name){
		adios2::Variable<std::string> variable = io.InquireVariable<std::string>(name);
		if (!variable) {
			throw std::invalid_argument(item + " does not have the key " + name);
		}
		std::string value;
		variable.SetStepSelection({selectedStep, 1});
		engine.Get(variable, value, adios2::Mode::Sync);
		return value;
	}

	//Read a value or an array of metadata
	template<typename T>
	std::vector<T> readValues(const std::string &name){
		adios2::Variable<T> variable = io.InquireVariable<T>(name);
		if (!variable) {
			throw std::invalid_argument("File does not have " + name);
		}
		std::vector<T> values;
		variable.SetStepSelection({selectedStep, 1});
		engine.Get(variable, values, adios2::Mode::Sync);
		return values;
	}

	//The type in the type registry of an ADIOS2 type
	SAMS::typeHandle typeFromADIOS2(const std::string &type){
		if (type == "float") return SAMS::typeID::TYPE_FLOAT;
		if (type == "double") return SAMS::typeID::TYPE_DOUBLE;
		if (type == "int8_t") return SAMS::typeID::TYPE_INT8;
		if (type == "int16_t") return SAMS::typeID::TYPE_INT16;
		if (type == "int32_t") return SAMS::typeID::TYPE_INT32;
		if (type == "int64_t") return SAMS::typeID::TYPE_INT64;
		if (type == "uint8_t") return SAMS::typeID::TYPE_UINT8;
		if (type == "uint16_t") return SAMS::typeID::TYPE_UINT16;
		if (type == "uint32_t") return SAMS::typeID::TYPE_UINT32;
		if (type == "uint64_t") return SAMS::typeID::TYPE_UINT64;
		if (type == "float complex") return SAMS::typeID::TYPE_COMPLEX_FLOAT;
		if (type == "double complex") return SAMS::typeID::TYPE_COMPLEX_DOUBLE;
		return SAMS::typeHandle();
	}

	//Describe a variable: its type, rank and size, from the parameters that ADIOS2 lists for it
	itemInfo variableInfo(const std::string &name){
		const auto variables = io.AvailableVariables();
		const auto it = variables.find(name);
		if (it == variables.end()) {
			throw std::runtime_error("File does not have the variable " + name);
		}
		itemInfo info;
		info.type = typeFromADIOS2(it->second.at("Type"));
		//The shape is the sizes separated by commas
		std::istringstream shape(it->second.at("Shape"));
		std::string size;
		while (std::getline(shape, size, ',')) {
			if (info.rank >= info.sizes.size()) {
				throw std::runtime_error("Variable " + name + " has a rank that cannot be read");
			}
			info.sizes[info.rank++] = std::stoul(size);
		}
		return info;
	}

	//Read the part of a variable that this rank holds
	template<typename T_data>
	void readPart(const std::string &name, T_data *data, const decomposition &part, const itemInfo &info){
		adios2::Variable<T_data> variable = io.InquireVariable<T_data>(name);
		if (!variable) {
			throw std::runtime_error("Could not open variable " + name);
		}
		adios2::Dims start(info.rank), count(info.rank), memoryStart(info.rank), memoryCount(info.rank);
		for (size_t i = 0; i < info.rank; ++i) {
			start[i] = part.globalStarts[i];
			count[i] = part.counts[i];
			memoryStart[i] = part.localStarts[i];
			memoryCount[i] = part.localSizes[i];
		}
		variable.SetSelection({start, count});
		//The part without the ghost cells is selected in the local array
		variable.SetMemorySelection({memoryStart, memoryCount});
		variable.SetStepSelection({selectedStep, 1});
		engine.Get(variable, data, adios2::Mode::Sync);
	}

	//A key-value pair in the global scope is a value with the type of the value at its key
	bool hasKeyValueImpl(const char* key){
		return hasVariable(key);
	}

	template<typename T>
	T readKeyValueImpl(const char* key){
		if constexpr (std::is_same_v<T, std::string>) {
			return readString("File", key);
		} else {
			return readValues<T>(key)[0];
		}
	}

	//The deck is the string value restart/deck
	std::string readDeckImpl(){
		return readString("File", "restart/deck");
	}

	//An axis is a variable with a start
	std::vector<std::string> axisNamesImpl(){
		std::vector<std::string> names;
		for (const auto &variable : io.AvailableVariables()) {
			if (hasVariable(variable.first + "/start")) names.push_back(variable.first);
		}
		return names;
	}

	//A mesh is the values axis<dimension>, starts and sizes in the meshes namespace
	std::vector<std::string> meshNamesImpl(){
		std::vector<std::string> names;
		const std::string prefix = "meshes/";
		const std::string suffix = "/sizes";
		for (const auto &variable : io.AvailableVariables()) {
			const std::string &name = variable.first;
			if (name.size() > prefix.size() + suffix.size() && name.compare(0, prefix.size(), prefix) == 0 && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
				names.push_back(name.substr(prefix.size(), name.size() - prefix.size() - suffix.size()));
			}
		}
		return names;
	}

	//A data item is a variable with a mesh name
	std::vector<std::string> dataNamesImpl(){
		std::vector<std::string> names;
		for (const auto &variable : io.AvailableVariables()) {
			if (hasVariable(variable.first + "/meshName")) names.push_back(variable.first);
		}
		return names;
	}

	//A blob is a variable in the blobs namespace that is not one of their key-value pairs, which are strings
	std::vector<std::string> blobNamesImpl(){
		std::vector<std::string> names;
		const std::string prefix = "blobs/";
		for (const auto &variable : io.AvailableVariables()) {
			if (variable.first.compare(0, prefix.size(), prefix) == 0 && io.VariableType(variable.first) != "string") names.push_back(variable.first.substr(prefix.size()));
		}
		return names;
	}

	itemInfo axisInfoImpl(const char* name){
		if (!hasVariable(name)) {
			throw std::invalid_argument("File does not have the axis " + std::string(name));
		}
		itemInfo info = variableInfo(name);
		info.starts[0] = readValues<int64_t>(std::string(name) + "/start")[0];
		const std::vector<int64_t> ghosts = readValues<int64_t>(std::string(name) + "/ghosts");
		info.lowerGhosts = static_cast<size_t>(ghosts[0]);
		info.upperGhosts = static_cast<size_t>(ghosts[1]);
		return info;
	}

	//A mesh is the values axis<dimension>, starts and sizes in the meshes namespace
	itemInfo meshInfoImpl(const char* name){
		const std::string meshName = "meshes/" + std::string(name);
		if (!hasVariable(meshName + "/sizes")) {
			throw std::invalid_argument("File does not have the mesh " + std::string(name));
		}
		const std::vector<int64_t> startData = readValues<int64_t>(meshName + "/starts");
		const std::vector<uint64_t> sizeData = readValues<uint64_t>(meshName + "/sizes");
		itemInfo info;
		info.rank = sizeData.size();
		for (size_t i = 0; i < info.rank; ++i) {
			info.axes[i] = readString("Mesh " + std::string(name), meshName + "/axis" + std::to_string(i));
			info.starts[i] = startData[i];
			info.sizes[i] = sizeData[i];
		}
		return info;
	}

	itemInfo dataInfoImpl(const char* name){
		if (!hasVariable(name) || !hasVariable(std::string(name) + "/meshName")) {
			throw std::invalid_argument("File does not have the data " + std::string(name));
		}
		itemInfo info = variableInfo(name);
		info.meshName = readString("Data " + std::string(name), std::string(name) + "/meshName");
		const std::vector<int64_t> starts = readValues<int64_t>(std::string(name) + "/starts");
		for (size_t i = 0; i < info.rank; ++i) info.starts[i] = starts[i];
		return info;
	}

	itemInfo blobInfoImpl(const char* name){
		const std::string blobName = "blobs/" + std::string(name);
		if (!hasVariable(blobName)) {
			throw std::invalid_argument("File does not have the blob " + std::string(name));
		}
		return variableInfo(blobName);
	}

	//The key-value pairs of a mesh are string values named in the mesh, in the meshes namespace
	std::string readMeshKeyValueImpl(const char* meshName, const char* key){
		return readString("Mesh " + std::string(meshName), "meshes/" + std::string(meshName) + "/" + key);
	}

	//The key-value pairs of a data item are string values named in it
	std::string readDataKeyValueImpl(const char* dataName, const char* key){
		return readString("Data " + std::string(dataName), std::string(dataName) + "/" + key);
	}

	//The key-value pairs of a blob are string values named in it
	std::string readBlobKeyValueImpl(const char* blobName, const char* key){
		return readString("Blob " + std::string(blobName), "blobs/" + std::string(blobName) + "/" + key);
	}

	//The key-value pairs of an axis are string values named in it
	std::string readAxisKeyValueImpl(const char* axisName, const char* key){
		return readString("Axis " + std::string(axisName), std::string(axisName) + "/" + key);
	}

	//Every rank reads the whole axis
	template<typename T_axis>
	void readAxisImpl(const char* name, T_axis *data, const itemInfo &info){
		decomposition part;
		part.localSizes[0] = info.sizes[0];
		part.counts[0] = info.sizes[0];
		readPart(name, data, part, info);
	}

	template<typename T_data>
	void readDataImpl(const char* name, T_data *data, const decomposition &part, const itemInfo &info){
		readPart(name, data, part, info);
	}

	template<typename T_data>
	void readBlobImpl(const char* name, T_data *data, const decomposition &part, const itemInfo &info){
		readPart("blobs/" + std::string(name), data, part, info);
	}

	public:

	//The name of the input format, which is the name of the writer of the same format
	constexpr static std::string_view readerName = "adios2";

	ADIOS2Reader(SAMS::harness &h) : reader<ADIOS2Reader>(h) {
	}

	/**
	 * @brief Select the step of the file that is read. Step 0 is selected when the file is opened
	 * @param step The step, by its position in the file
	 */
	void selectStep(size_t step){
		selectedStep = step;
	}

	/**
	 * @brief The number of steps in the file
	 */
	size_t getStepCount(){
		return engine.Steps();
	}

	//Close the file while the members that closing it uses still exist
	~ADIOS2Reader(){
		if (this->isOpen) this->closeFile();
	}
};

} // namespace SAMS::IO
#endif
#endif
