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
#ifndef READER_SIMPLE_H
#define READER_SIMPLE_H
#include <array>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include "readerProto.h"
#include "mpiManager.h"
#include "simpleHelpers.h"

namespace SAMS::IO
{

/**
 * @brief Reads the directories that simpleFile writes
 */
class simpleReader : public reader<simpleReader> {
	friend class reader<simpleReader>;

	//The header of a file: the lines after the number of header lines, and where the data after the header starts
	struct fileHeader{
		std::vector<std::string> lines;
		std::streamoff dataStart = 0;
	};

	//Rank 0 reads everything except the data, and broadcasts it. A failure throws on rank 0, which the caller aborts
	bool isRoot(){
		return this->h.MPIManager.getRank() == 0;
	}

	void broadcast([[maybe_unused]] std::string &value){
		#ifdef USE_MPI
		broadcastFromRoot(this->h.MPIManager.getComm(), value);
		#endif
	}

	bool rootExists(const std::string &name){
		#ifdef USE_MPI
		return existsOnRoot(this->h.MPIManager.getComm(), name);
		#else
		return existsOnRoot(name);
		#endif
	}

	bool rootIsDirectory(const std::string &name){
		#ifdef USE_MPI
		return isDirectoryOnRoot(this->h.MPIManager.getComm(), name);
		#else
		return isDirectoryOnRoot(name);
		#endif
	}

	std::string rootRead(const std::string &name){
		#ifdef USE_MPI
		return readOnRoot(this->h.MPIManager.getComm(), name);
		#else
		return readOnRoot(name);
		#endif
	}

	void openFileImpl(const char* filename){
		if (!rootIsDirectory(filename)) {
			throw std::runtime_error("Could not open directory " + std::string(filename));
		}
	}

	void closeFileImpl(){
	}

	//The name of the file of an item in the directory
	std::string itemFileName(const std::string &name, const std::string &extension){
		return std::string(this->filename) + "/" + name + extension;
	}

	//The name of the key-value file of a namespace in the directory
	std::string keyValueFileName(const std::string &space){
		return std::string(this->filename) + "/" + (space.empty() ? "" : space + "/") + "keyValues";
	}

	//Read the header of a file
	fileHeader readHeader(const std::string &fileName){
		//Broadcast as the start of the data, the number of lines, then the lines, separated by newlines
		std::string packed;
		if (isRoot()) {
			std::ifstream file(fileName, std::ios::binary);
			if (!file.is_open()) {
				throw std::runtime_error("Could not open file " + fileName);
			}
			std::string line;
			std::getline(file, line);
			const size_t count = std::stoul(line);
			std::string lines;
			for (size_t i = 0; i < count; ++i) {
				std::getline(file, line);
				lines += "\n" + line;
			}
			packed = std::to_string(static_cast<long long>(file.tellg())) + "\n" + std::to_string(count) + lines;
		}
		broadcast(packed);
		fileHeader header;
		std::istringstream stream(packed);
		std::string line;
		std::getline(stream, line);
		header.dataStart = static_cast<std::streamoff>(std::stoll(line));
		std::getline(stream, line);
		const size_t count = std::stoul(line);
		for (size_t i = 0; i < count; ++i) {
			std::getline(stream, line);
			header.lines.push_back(line);
		}
		return header;
	}

	//Find a key in the header lines from a given line on, which are of the form "key value"
	bool findKeyValue(const std::vector<std::string> &lines, size_t first, const std::string &key, std::string &value){
		for (size_t i = first; i < lines.size(); ++i) {
			const size_t split = lines[i].find(' ');
			if (lines[i].substr(0, split) == key) {
				value = split == std::string::npos ? "" : lines[i].substr(split + 1);
				return true;
			}
		}
		return false;
	}

	//Read a key from the header lines from a given line on
	std::string readKeyValueLines(const std::vector<std::string> &lines, size_t first, const std::string &item, const std::string &key){
		std::string value;
		if (!findKeyValue(lines, first, key, value)) {
			throw std::invalid_argument(item + " does not have the key " + key);
		}
		return value;
	}

	//The type of an item from its header line of the form "category size"
	SAMS::typeHandle typeFromHeader(const std::string &line){
		std::istringstream stream(line);
		std::string category;
		size_t size = 0;
		stream >> category >> size;
		if (category == "float") {
			if (size == 4) return SAMS::typeID::TYPE_FLOAT;
			if (size == 8) return SAMS::typeID::TYPE_DOUBLE;
		} else if (category == "int") {
			if (size == 1) return SAMS::typeID::TYPE_INT8;
			if (size == 2) return SAMS::typeID::TYPE_INT16;
			if (size == 4) return SAMS::typeID::TYPE_INT32;
			if (size == 8) return SAMS::typeID::TYPE_INT64;
		} else if (category == "uint") {
			if (size == 1) return SAMS::typeID::TYPE_UINT8;
			if (size == 2) return SAMS::typeID::TYPE_UINT16;
			if (size == 4) return SAMS::typeID::TYPE_UINT32;
			if (size == 8) return SAMS::typeID::TYPE_UINT64;
		} else if (category == "complex") {
			if (size == 8) return SAMS::typeID::TYPE_COMPLEX_FLOAT;
			if (size == 16) return SAMS::typeID::TYPE_COMPLEX_DOUBLE;
		}
		throw std::runtime_error("Type " + line + " cannot be read");
	}

	//Read the sizes of an item from its header line of the sizes separated by spaces
	void sizesFromHeader(const std::string &line, itemInfo &info){
		std::istringstream stream(line);
		size_t size;
		while (stream >> size) {
			if (info.rank >= info.sizes.size()) {
				throw std::runtime_error("Item has a rank that cannot be read");
			}
			info.sizes[info.rank++] = size;
		}
	}

	//The names of the items with an extension in the directory, including those in namespaces
	std::vector<std::string> itemNames(const std::string &extension){
		//Broadcast as the names separated by newlines
		std::string packed;
		if (isRoot()) {
			const std::filesystem::path directory(this->filename);
			for (const auto &entry : std::filesystem::recursive_directory_iterator(directory)) {
				if (!entry.is_regular_file() || entry.path().extension() != extension) continue;
				const std::filesystem::path relative = std::filesystem::relative(entry.path(), directory);
				packed += (relative.parent_path() / relative.stem()).generic_string() + "\n";
			}
		}
		broadcast(packed);
		std::vector<std::string> names;
		std::istringstream stream(packed);
		std::string name;
		while (std::getline(stream, name)) names.push_back(name);
		return names;
	}

	/*
	Internal function to read the part of an item that this rank holds, from the data after its header
	With MPI every rank reads its own part of the file. Without MPI each contiguous run of the last dimension is read in turn
	*/
	template<typename T_data>
	void readPart(const std::string &fileName, std::streamoff dataStart, T_data *data, const decomposition &part, const itemInfo &info){
		#ifdef USE_MPI
		SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> &mpi = this->h.MPIManager;
		MPI_File file;
		if (MPI_File_open(mpi.getComm(), fileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
			throw std::runtime_error("Could not open file " + fileName);
		}
		std::array<int, SAMS::MAX_RANK> globalSizes, globalStarts, counts, localSizes, localStarts;
		for (size_t i = 0; i < info.rank; ++i) {
			globalSizes[i] = static_cast<int>(info.sizes[i]);
			globalStarts[i] = static_cast<int>(part.globalStarts[i]);
			counts[i] = static_cast<int>(part.counts[i]);
			localSizes[i] = static_cast<int>(part.localSizes[i]);
			localStarts[i] = static_cast<int>(part.localStarts[i]);
		}
		SAMSMPI::datatype baseType = SAMS::gettypeRegistry().getMPIType<T_data>();
		SAMSMPI::datatype fileType = mpi.createMPISubarrayType(static_cast<int>(info.rank), globalSizes.data(), counts.data(), globalStarts.data(), baseType);
		SAMSMPI::datatype memoryType = mpi.createMPISubarrayType(static_cast<int>(info.rank), localSizes.data(), counts.data(), localStarts.data(), baseType);
		if (MPI_File_set_view(file, static_cast<MPI_Offset>(dataStart), baseType, fileType, "native", MPI_INFO_NULL) != MPI_SUCCESS) {
			throw std::runtime_error("Could not set view of file " + fileName);
		}
		if (MPI_File_read_all(file, data, 1, memoryType, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			throw std::runtime_error("Could not read data from file " + fileName);
		}
		if (MPI_File_close(&file) != MPI_SUCCESS) {
			throw std::runtime_error("Could not close file " + fileName);
		}
		mpi.checkinMPIType(fileType);
		mpi.checkinMPIType(memoryType);
		#else
		std::ifstream file(fileName, std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file " + fileName);
		}
		const size_t last = info.rank - 1;
		size_t runs = 1;
		for (size_t i = 0; i < last; ++i) runs *= part.counts[i];
		std::array<size_t, SAMS::MAX_RANK> index{};
		for (size_t n = 0; n < runs; ++n) {
			//The start of the run in the file and in the local array, with the last dimension fastest
			size_t fileOffset = 0, memoryOffset = 0;
			for (size_t i = 0; i < info.rank; ++i) {
				fileOffset = fileOffset * info.sizes[i] + part.globalStarts[i] + index[i];
				memoryOffset = memoryOffset * part.localSizes[i] + part.localStarts[i] + index[i];
			}
			file.seekg(dataStart + static_cast<std::streamoff>(fileOffset * sizeof(T_data)));
			file.read(reinterpret_cast<char*>(data + memoryOffset), static_cast<std::streamsize>(part.counts[last] * sizeof(T_data)));
			if (!file) {
				throw std::runtime_error("Could not read data from file " + fileName);
			}
			for (size_t i = last; i-- > 0;) {
				if (++index[i] < part.counts[i]) break;
				index[i] = 0;
			}
		}
		#endif
	}

	bool hasKeyValueImpl(const char* key){
		const auto [space, name] = splitNamespace(key);
		const std::string fileName = keyValueFileName(space);
		if (!rootExists(fileName)) return false;
		std::string value;
		return findKeyValue(readHeader(fileName).lines, 0, name, value);
	}

	//A key-value pair in the global scope is text. A real number is in hexadecimal floating point
	template<typename T>
	T readKeyValueImpl(const char* key){
		const auto [space, name] = splitNamespace(key);
		const std::string text = readKeyValueLines(readHeader(keyValueFileName(space)).lines, 0, "File", name);
		if constexpr (std::is_same_v<T, std::string>) {
			return text;
		} else if constexpr (std::is_same_v<T, double>) {
			return std::strtod(text.c_str(), nullptr);
		} else if constexpr (std::is_same_v<T, int64_t>) {
			return static_cast<int64_t>(std::stoll(text));
		} else {
			return static_cast<uint64_t>(std::stoull(text));
		}
	}

	//The deck is the file input.deck in the directory
	std::string readDeckImpl(){
		return rootRead(std::string(this->filename) + "/input.deck");
	}

	std::vector<std::string> axisNamesImpl(){
		return itemNames(".axis");
	}

	std::vector<std::string> meshNamesImpl(){
		return itemNames(".mesh");
	}

	std::vector<std::string> dataNamesImpl(){
		return itemNames(".data");
	}

	std::vector<std::string> blobNamesImpl(){
		return itemNames(".blob");
	}

	//The header of an axis is the type, the number of elements, the core index of the first element, then the numbers of ghost cells below and above the domain
	itemInfo axisInfoImpl(const char* name){
		const std::string fileName = itemFileName(name, ".axis");
		if (!rootExists(fileName)) {
			throw std::invalid_argument("File does not have the axis " + std::string(name));
		}
		const fileHeader header = readHeader(fileName);
		itemInfo info;
		info.type = typeFromHeader(header.lines[0]);
		sizesFromHeader(header.lines[1], info);
		info.starts[0] = std::stoll(header.lines[2]);
		std::istringstream ghosts(header.lines[3]);
		ghosts >> info.lowerGhosts >> info.upperGhosts;
		return info;
	}

	//The header of a mesh is the rank, then for each dimension the axis, the first cell and the number of cells, then the key-value pairs. A mesh holds no values
	itemInfo meshInfoImpl(const char* name){
		const std::string fileName = itemFileName(name, ".mesh");
		if (!rootExists(fileName)) {
			throw std::invalid_argument("File does not have the mesh " + std::string(name));
		}
		const fileHeader header = readHeader(fileName);
		itemInfo info;
		info.rank = std::stoul(header.lines[0]);
		for (size_t i = 0; i < info.rank; ++i) {
			std::istringstream stream(header.lines[1 + i]);
			stream >> info.axes[i] >> info.starts[i] >> info.sizes[i];
		}
		return info;
	}

	//The header of data is the mesh name, the type, the sizes, the core index of the first element in each dimension, then the key-value pairs
	itemInfo dataInfoImpl(const char* name){
		const std::string fileName = itemFileName(name, ".data");
		if (!rootExists(fileName)) {
			throw std::invalid_argument("File does not have the data " + std::string(name));
		}
		const fileHeader header = readHeader(fileName);
		itemInfo info;
		info.meshName = header.lines[0];
		info.type = typeFromHeader(header.lines[1]);
		sizesFromHeader(header.lines[2], info);
		std::istringstream starts(header.lines[3]);
		for (size_t i = 0; i < info.rank; ++i) starts >> info.starts[i];
		return info;
	}

	//The header of a blob is the type, then the sizes, then the key-value pairs
	itemInfo blobInfoImpl(const char* name){
		const std::string fileName = itemFileName(name, ".blob");
		if (!rootExists(fileName)) {
			throw std::invalid_argument("File does not have the blob " + std::string(name));
		}
		const fileHeader header = readHeader(fileName);
		itemInfo info;
		info.type = typeFromHeader(header.lines[0]);
		sizesFromHeader(header.lines[1], info);
		return info;
	}

	//The header line of each kind of item that its key-value pairs start at. They follow the mesh name, type, sizes and starts of data, the type and sizes of a blob, and the type, size, start and ghost cells of an axis
	//The key-value pairs of a mesh follow its rank and a line for each dimension, so where they start depends on the rank
	static constexpr size_t dataKeyValueStart = 4;
	static constexpr size_t blobKeyValueStart = 2;
	static constexpr size_t axisKeyValueStart = 4;

	//Read the header of an item in the directory
	fileHeader readItemHeader(const std::string &name, const std::string &extension){
		return readHeader(itemFileName(name, extension));
	}

	std::string readMeshKeyValueImpl(const char* meshName, const char* key){
		const fileHeader header = readItemHeader(meshName, ".mesh");
		return readKeyValueLines(header.lines, 1 + std::stoul(header.lines[0]), "Mesh " + std::string(meshName), key);
	}

	std::string readDataKeyValueImpl(const char* dataName, const char* key){
		const fileHeader header = readItemHeader(dataName, ".data");
		return readKeyValueLines(header.lines, dataKeyValueStart, "Data " + std::string(dataName), key);
	}

	std::string readBlobKeyValueImpl(const char* blobName, const char* key){
		const fileHeader header = readItemHeader(blobName, ".blob");
		return readKeyValueLines(header.lines, blobKeyValueStart, "Blob " + std::string(blobName), key);
	}

	std::string readAxisKeyValueImpl(const char* axisName, const char* key){
		const fileHeader header = readItemHeader(axisName, ".axis");
		return readKeyValueLines(header.lines, axisKeyValueStart, "Axis " + std::string(axisName), key);
	}

	//Every rank reads the whole axis, which follows its header
	template<typename T_axis>
	void readAxisImpl(const char* name, T_axis *data, const itemInfo &info){
		const std::string fileName = itemFileName(name, ".axis");
		decomposition part;
		part.localSizes[0] = info.sizes[0];
		part.counts[0] = info.sizes[0];
		readPart(fileName, readHeader(fileName).dataStart, data, part, info);
	}

	template<typename T_data>
	void readDataImpl(const char* name, T_data *data, const decomposition &part, const itemInfo &info){
		const std::string fileName = itemFileName(name, ".data");
		readPart(fileName, readHeader(fileName).dataStart, data, part, info);
	}

	template<typename T_data>
	void readBlobImpl(const char* name, T_data *data, const decomposition &part, const itemInfo &info){
		const std::string fileName = itemFileName(name, ".blob");
		readPart(fileName, readHeader(fileName).dataStart, data, part, info);
	}

	public:

	//The name of the input format, which is the name of the writer of the same format
	constexpr static std::string_view readerName = "simple";

	simpleReader(SAMS::harness &h) : reader<simpleReader>(h) {
	}

	//Close the file while the members that closing it uses still exist
	~simpleReader(){
		if (this->isOpen) this->closeFile();
	}
};

} // namespace SAMS::IO

#endif
