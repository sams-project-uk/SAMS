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
#ifndef READERPROTO_H
#define READERPROTO_H

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "harness.h"
#include "types.h"
#include "writerProto.h"

namespace SAMS::IO
{

/**
 * @brief Description of an axis, mesh, data item or blob in a file
 */
struct itemInfo{
	//The type of the item. It is not valid for a mesh, which holds no values
	SAMS::typeHandle type;
	//The rank of the item
	size_t rank = 0;
	//The global size of the item in each dimension
	std::array<size_t, SAMS::MAX_RANK> sizes{};
	//The mesh that a data item is on. Empty for anything else
	std::string meshName;
	//The axis of each dimension of a mesh. Empty for anything else
	std::array<std::string, SAMS::MAX_RANK> axes;
	//Core index of the first element in each dimension of a mesh, data item or axis
	std::array<SIGNED_INDEX_TYPE, SAMS::MAX_RANK> starts{};
	//The number of ghost cells below and above the domain of an axis. Zero for anything else
	size_t lowerGhosts = 0;
	size_t upperGhosts = 0;
};

	/*
	 * @brief Base class for all readers. This class provides the interface for reading data from a file.
	 * @tparam T_core The type of the reader. This should be a subclass of reader.
	 * @details This uses CRTP in the same way as writer. A reader reads everything that the writer of the same format writes
	 */
template<typename T_core>
class reader {

	public:

	//The name of the input format, which is the name of the writer of the same format
	constexpr static std::string_view readerName = T_core::readerName;

	private:

	friend T_core;

	SAMS::harness &h;

	std::string filename = "";
	bool isOpen=false; //Is there a file open

	/**
	 * @brief Check that a file is open
	 * @throws std::runtime_error if no file is open
	 */
	void testOpen() const {
		if (!isOpen) throw std::runtime_error("Reader is not connected to a file");
	}

	/**
	 * @brief Check that an item in the file can be read as a type
	 * @tparam T_data The type that the item is read as
	 * @param item The description of the item, for errors
	 * @param info The item
	 * @throws std::invalid_argument if the type of the item is not the type that it is read as
	 */
	template<typename T_data>
	void checkType(const std::string &item, const itemInfo &info){
		//A file that does not record the type of an item cannot be checked
		if (info.type.ID < 0) return;
		const SAMS::typeHandle type = SAMS::gettypeRegistry().getTypeID<T_data>();
		if (info.type == type) return;
		throw std::invalid_argument("Type of " + item + " in the file is " + SAMS::gettypeRegistry().getTypeName(info.type) + ", not " + SAMS::gettypeRegistry().getTypeName(type));
	}

	/**
	 * @brief Check that a part of a distributed item can be read, and get its decomposition
	 * @param item The description of the item, for errors
	 * @param part The part that is read
	 * @param info The item
	 * @throws std::invalid_argument if the rank or type of the item does not match, or the part is outside it
	 */
	template<typename T_data, int R>
	decomposition readDecomposition(const std::string &item, const localVariablePart<T_data, R> &part, const itemInfo &info){
		if (info.rank != static_cast<size_t>(R)) throw std::invalid_argument("Rank of " + item + " in the file is " + std::to_string(info.rank) + ", not " + std::to_string(R));
		checkType<T_data>(item, info);
		decomposition decomp = partDecomposition(part);
		for (int i = 0; i < R; ++i){
			if (decomp.globalStarts[i] + decomp.counts[i] > info.sizes[i]) throw std::invalid_argument("The part of " + item + " that is read is outside it");
		}
		return decomp;
	}

	protected:
	reader(SAMS::harness &h) : h(h) {}

	public:

	/**
	 * @brief Open a file for reading. This should be called before any other functions are called.
	 * @param filename The name of the file, without the extension that the format adds
	 * @throws std::runtime_error if a file is already open
	 */
	void openFile(const char* filename){
		if (isOpen) throw std::runtime_error("Reader is already connected to a file");
		this->filename = filename;
		static_cast<T_core*>(this)->openFileImpl(filename);
		isOpen=true;
	}

	/**
	 * @brief Close the file
	 */
	void closeFile(){
		testOpen();
		static_cast<T_core*>(this)->closeFileImpl();
		isOpen=false;
	}

	/**
	 * @brief Whether the file has a key-value pair in the global scope
	 * @param key The key, which can be in a namespace
	 */
	bool hasKeyValue(const char* key){
		testOpen();
		return static_cast<T_core*>(this)->hasKeyValueImpl(key);
	}

	/**
	 * @brief Read a key-value pair in the global scope
	 * @tparam T The type that the value was stored as: double, int64_t, uint64_t or std::string
	 * @param key The key, which can be in a namespace
	 * @return The value
	 * @throws std::invalid_argument if the file does not have the key
	 */
	template<typename T>
	T readKeyValue(const char* key){
		static_assert(isKeyValueType<T>, "A key-value pair must be a double, int64_t, uint64_t or std::string");
		if (!hasKeyValue(key)) throw std::invalid_argument(std::string("File does not have the key ") + key);
		return static_cast<T_core*>(this)->template readKeyValueImpl<T>(key);
	}

	/**
	 * @brief Read the deck that writeDeck wrote, as the text of an EPOCH deck
	 */
	std::string readDeck(){
		testOpen();
		return static_cast<T_core*>(this)->readDeckImpl();
	}

	/**
	 * @brief The names of the axes in the file
	 */
	std::vector<std::string> axisNames(){
		testOpen();
		return static_cast<T_core*>(this)->axisNamesImpl();
	}

	/**
	 * @brief The names of the meshes in the file
	 */
	std::vector<std::string> meshNames(){
		testOpen();
		return static_cast<T_core*>(this)->meshNamesImpl();
	}

	/**
	 * @brief The names of the data items in the file
	 */
	std::vector<std::string> dataNames(){
		testOpen();
		return static_cast<T_core*>(this)->dataNamesImpl();
	}

	/**
	 * @brief The names of the blobs in the file
	 */
	std::vector<std::string> blobNames(){
		testOpen();
		return static_cast<T_core*>(this)->blobNamesImpl();
	}

	/**
	 * @brief Describe an axis in the file. An axis has rank 1
	 * @param name The name of the axis
	 * @throws std::invalid_argument if the file does not have the axis
	 */
	itemInfo axisInfo(const char* name){
		testOpen();
		return static_cast<T_core*>(this)->axisInfoImpl(name);
	}

	/**
	 * @brief Describe a mesh in the file. The sizes are the numbers of cells of the axes, and the axes and starts say which axis and which range of it is each dimension
	 * @param name The name of the mesh
	 * @throws std::invalid_argument if the file does not have the mesh
	 */
	itemInfo meshInfo(const char* name){
		testOpen();
		return static_cast<T_core*>(this)->meshInfoImpl(name);
	}

	/**
	 * @brief Describe a data item in the file
	 * @param name The name of the data
	 * @throws std::invalid_argument if the file does not have the data
	 */
	itemInfo dataInfo(const char* name){
		testOpen();
		return static_cast<T_core*>(this)->dataInfoImpl(name);
	}

	/**
	 * @brief Describe a blob in the file
	 * @param name The name of the blob
	 * @throws std::invalid_argument if the file does not have the blob
	 */
	itemInfo blobInfo(const char* name){
		testOpen();
		return static_cast<T_core*>(this)->blobInfoImpl(name);
	}

	/**
	 * @brief Read a key-value pair on a mesh
	 * @param meshName The name of the mesh
	 * @param key The key
	 * @throws std::invalid_argument if the mesh does not have the key
	 */
	std::string readMeshKeyValue(const char* meshName, const char* key){
		testOpen();
		return static_cast<T_core*>(this)->readMeshKeyValueImpl(meshName, key);
	}

	/**
	 * @brief Read a key-value pair on a data item
	 * @param dataName The name of the data
	 * @param key The key
	 * @throws std::invalid_argument if the data does not have the key
	 */
	std::string readDataKeyValue(const char* dataName, const char* key){
		testOpen();
		return static_cast<T_core*>(this)->readDataKeyValueImpl(dataName, key);
	}

	/**
	 * @brief Read a key-value pair on a blob
	 * @param blobName The name of the blob
	 * @param key The key
	 * @throws std::invalid_argument if the blob does not have the key
	 */
	std::string readBlobKeyValue(const char* blobName, const char* key){
		testOpen();
		return static_cast<T_core*>(this)->readBlobKeyValueImpl(blobName, key);
	}

	/**
	 * @brief Read a key-value pair on an axis
	 * @param axisName The name of the axis
	 * @param key The key
	 * @throws std::invalid_argument if the axis does not have the key
	 */
	std::string readAxisKeyValue(const char* axisName, const char* key){
		testOpen();
		return static_cast<T_core*>(this)->readAxisKeyValueImpl(axisName, key);
	}

	/**
	 * @brief Read the whole of an axis. Every rank reads the whole axis
	 * @param name The name of the axis
	 * @param data The array to read the axis into, which must be as long as the axis
	 * @throws std::invalid_argument if the axis does not have the type that it is read as
	 */
	template<typename T_axis>
	void readAxis(const char* name, T_axis *data){
		const itemInfo info = axisInfo(name);
		checkType<T_axis>(std::string("axis ") + name, info);
		static_cast<T_core*>(this)->readAxisImpl(name, data, info);
	}

	/**
	 * @brief Read the part of a data item that this rank holds. Every rank reads its own part, so the decomposition can be different from the one that it was written with
	 * @param name The name of the data
	 * @param part The part to read, and where it is in the whole data. Its array must already be allocated
	 * @throws std::invalid_argument if the data does not have the rank or type that it is read as, or the part is outside it
	 */
	template<typename T_data, int R>
	void readData(const char* name, localVariablePart<T_data, R> &part){
		const itemInfo info = dataInfo(name);
		const decomposition decomp = readDecomposition(std::string("data ") + name, part, info);
		static_cast<T_core*>(this)->readDataImpl(name, part.array.data(), decomp, info);
	}

	/**
	 * @brief Read the part of a blob that this rank holds. Every rank reads its own part, so the decomposition can be different from the one that it was written with
	 * @param name The name of the blob
	 * @param part The part to read, and where it is in the whole blob. Its array must already be allocated
	 * @throws std::invalid_argument if the blob does not have the rank or type that it is read as, or the part is outside it
	 */
	template<typename T_data, int R>
	void readBlob(const char* name, localVariablePart<T_data, R> &part){
		const itemInfo info = blobInfo(name);
		const decomposition decomp = readDecomposition(std::string("blob ") + name, part, info);
		static_cast<T_core*>(this)->readBlobImpl(name, part.array.data(), decomp, info);
	}
};

} // namespace SAMS::IO

#endif
