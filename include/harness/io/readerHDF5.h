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
#ifndef READER_HDF5_H
#define READER_HDF5_H
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include "readerProto.h"
#if __has_include(<hdf5.h>)
#include <hdf5.h>
#include "hdf5Types.h"
#include "harnessDef.h"
#include "mpiManager.h"

namespace SAMS::IO
{

namespace detail
{
	//The objects in an HDF5 file, found by visiting its links
	struct hdf5Objects{
		std::vector<std::string> datasets;
		std::vector<std::string> groups;
	};

	/**
	 * @brief Record an object in an HDF5 file as a dataset or a group
	 */
	inline herr_t visitHDF5Object(hid_t group, const char* name, [[maybe_unused]] const H5L_info_t *info, void *objects){
		hid_t object = H5Oopen(group, name, H5P_DEFAULT);
		if (object < 0) return 0;
		hdf5Objects &found = *static_cast<hdf5Objects*>(objects);
		const H5I_type_t type = H5Iget_type(object);
		if (type == H5I_DATASET) found.datasets.push_back(name);
		if (type == H5I_GROUP) found.groups.push_back(name);
		H5Oclose(object);
		return 0;
	}
}

/**
 * @brief Reads the files that HDF5File writes
 */
class HDF5Reader : public reader<HDF5Reader> {
	friend class reader<HDF5Reader>;

	//HDF5 file handle
	hid_t fileHandle = H5I_INVALID_HID;
	//The HDF5 types of the C++ types
	hdf5Types types;

	void openFileImpl(const char* filename){
		std::string h5filename = std::string(filename) + ".h5";
		#ifdef USE_MPI
		//Every rank opens the file for parallel access
		hid_t accessList = H5Pcreate(H5P_FILE_ACCESS);
		if (accessList < 0) {
			throw std::runtime_error("Could not create file access property list for " + h5filename);
		}
		if (H5Pset_fapl_mpio(accessList, this->h.MPIManager.getComm(), MPI_INFO_NULL) < 0) {
			throw std::runtime_error("Could not set parallel access for " + h5filename);
		}
		fileHandle = H5Fopen(h5filename.c_str(), H5F_ACC_RDONLY, accessList);
		if (H5Pclose(accessList) < 0) {
			throw std::runtime_error("Could not close file access property list for " + h5filename);
		}
		#else
		fileHandle = H5Fopen(h5filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		#endif
		if (fileHandle < 0) {
			throw std::runtime_error("Could not open file " + h5filename);
		}
	}

	void closeFileImpl(){
		types.close();
		if (H5Fclose(fileHandle) < 0) {
			throw std::runtime_error("Could not close file");
		}
		fileHandle = H5I_INVALID_HID;
	}

	/*
	Internal function to check whether a path exists in the file
	Each level is checked in turn, because checking a link below one that does not exist is an error
	*/
	bool pathExists(const std::string &path){
		for (size_t pos = path.find('/'); ; pos = path.find('/', pos + 1)) {
			if (H5Lexists(fileHandle, path.substr(0, pos).c_str(), H5P_DEFAULT) <= 0) return false;
			if (pos == std::string::npos) return true;
		}
	}

	/*
	Internal function to check whether an object has an attribute
	*/
	bool attributeExists(const std::string &objectPath, const std::string &key){
		if (objectPath != "/" && !pathExists(objectPath)) return false;
		return H5Aexists_by_name(fileHandle, objectPath.c_str(), key.c_str(), H5P_DEFAULT) > 0;
	}

	/*
	Internal function to read a string attribute of an object
	*/
	std::string readStringAttribute(const std::string &objectPath, const std::string &key){
		if (!attributeExists(objectPath, key)) {
			throw std::invalid_argument("Object " + objectPath + " does not have the key " + key);
		}
		hid_t attr = H5Aopen_by_name(fileHandle, objectPath.c_str(), key.c_str(), H5P_DEFAULT, H5P_DEFAULT);
		if (attr < 0) {
			throw std::runtime_error("Could not open attribute " + key);
		}
		hid_t strType = H5Tcopy(H5T_C_S1);
		if (strType < 0) {
			throw std::runtime_error("Could not create string datatype for attribute " + key);
		}
		if (H5Tset_size(strType, H5T_VARIABLE) < 0) {
			throw std::runtime_error("Could not set size of string datatype for attribute " + key);
		}
		char *valueCStr = nullptr;
		if (H5Aread(attr, strType, &valueCStr) < 0) {
			throw std::runtime_error("Could not read attribute " + key);
		}
		std::string value = valueCStr ? valueCStr : "";
		if (H5free_memory(valueCStr) < 0) {
			throw std::runtime_error("Could not free the value of attribute " + key);
		}
		if (H5Tclose(strType) < 0) {
			throw std::runtime_error("Could not close string datatype for attribute " + key);
		}
		if (H5Aclose(attr) < 0) {
			throw std::runtime_error("Could not close attribute " + key);
		}
		return value;
	}

	/*
	Internal function to describe a dataset: its type, rank and size
	*/
	itemInfo datasetInfo(const std::string &path){
		hid_t dataset = H5Dopen(fileHandle, path.c_str(), H5P_DEFAULT);
		if (dataset < 0) {
			throw std::runtime_error("Could not open dataset " + path);
		}
		itemInfo info;
		hid_t type = H5Dget_type(dataset);
		if (type < 0) {
			throw std::runtime_error("Could not get type of dataset " + path);
		}
		info.type = typeHandleFromHDF5(type);
		if (H5Tclose(type) < 0) {
			throw std::runtime_error("Could not close type of dataset " + path);
		}
		hid_t dataspace = H5Dget_space(dataset);
		if (dataspace < 0) {
			throw std::runtime_error("Could not get dataspace of dataset " + path);
		}
		const int rank = H5Sget_simple_extent_ndims(dataspace);
		if (rank < 0 || static_cast<size_t>(rank) > info.sizes.size()) {
			throw std::runtime_error("Dataset " + path + " has a rank that cannot be read");
		}
		hsize_t dims[SAMS::MAX_RANK];
		if (H5Sget_simple_extent_dims(dataspace, dims, NULL) < 0) {
			throw std::runtime_error("Could not get sizes of dataset " + path);
		}
		info.rank = static_cast<size_t>(rank);
		for (size_t i = 0; i < info.rank; ++i) info.sizes[i] = dims[i];
		if (H5Sclose(dataspace) < 0) {
			throw std::runtime_error("Could not close dataspace of dataset " + path);
		}
		if (H5Dclose(dataset) < 0) {
			throw std::runtime_error("Could not close dataset " + path);
		}
		return info;
	}

	/*
	Internal function to find the objects in the file
	*/
	detail::hdf5Objects findObjects(){
		detail::hdf5Objects objects;
		if (H5Lvisit(fileHandle, H5_INDEX_NAME, H5_ITER_NATIVE, detail::visitHDF5Object, &objects) < 0) {
			throw std::runtime_error("Could not list the objects in the file");
		}
		return objects;
	}

	/*
	Internal function to read the part of a dataset that this rank holds. Every rank of the file takes part, because the calls are collective
	*/
	template<typename T_data>
	void readPart(const std::string &path, T_data *data, const decomposition &part, const itemInfo &info){
		hsize_t globalStarts[SAMS::MAX_RANK], counts[SAMS::MAX_RANK], localSizes[SAMS::MAX_RANK], localStarts[SAMS::MAX_RANK];
		for (size_t i = 0; i < info.rank; ++i) {
			globalStarts[i] = part.globalStarts[i];
			counts[i] = part.counts[i];
			localSizes[i] = part.localSizes[i];
			localStarts[i] = part.localStarts[i];
		}
		hid_t dataset = H5Dopen(fileHandle, path.c_str(), H5P_DEFAULT);
		if (dataset < 0) {
			throw std::runtime_error("Could not open dataset " + path);
		}
		//Select the part in the file, and the part of the local array without the ghost cells
		hid_t dataspace = H5Dget_space(dataset);
		if (dataspace < 0) {
			throw std::runtime_error("Could not get dataspace of dataset " + path);
		}
		if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, globalStarts, NULL, counts, NULL) < 0) {
			throw std::runtime_error("Could not select part in dataspace for " + path);
		}
		hid_t memspace = H5Screate_simple(static_cast<int>(info.rank), localSizes, NULL);
		if (memspace < 0) {
			throw std::runtime_error("Could not create memory dataspace for " + path);
		}
		if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET, localStarts, NULL, counts, NULL) < 0) {
			throw std::runtime_error("Could not select part in memory dataspace for " + path);
		}
		#ifdef USE_MPI
		hid_t transferList = H5Pcreate(H5P_DATASET_XFER);
		if (transferList < 0 || H5Pset_dxpl_mpio(transferList, H5FD_MPIO_COLLECTIVE) < 0) {
			throw std::runtime_error("Could not set collective transfer");
		}
		#else
		hid_t transferList = H5P_DEFAULT;
		#endif
		if (H5Dread(dataset, types.get<T_data>(), memspace, dataspace, transferList, data) < 0) {
			throw std::runtime_error("Could not read " + path);
		}
		#ifdef USE_MPI
		if (H5Pclose(transferList) < 0) {
			throw std::runtime_error("Could not close dataset transfer property list");
		}
		#endif
		if (H5Sclose(memspace) < 0) {
			throw std::runtime_error("Could not close memory dataspace for " + path);
		}
		if (H5Sclose(dataspace) < 0) {
			throw std::runtime_error("Could not close dataspace for " + path);
		}
		if (H5Dclose(dataset) < 0) {
			throw std::runtime_error("Could not close dataset " + path);
		}
	}

	bool hasKeyValueImpl(const char* key){
		const auto [space, name] = splitNamespace(key);
		return attributeExists(space.empty() ? "/" : space, name);
	}

	//A key-value pair in the global scope is an attribute with the type of its value on the group of its namespace, or of the root group if it has none
	template<typename T>
	T readKeyValueImpl(const char* key){
		const auto [space, name] = splitNamespace(key);
		const std::string objectPath = space.empty() ? "/" : space;
		if constexpr (std::is_same_v<T, std::string>) {
			return readStringAttribute(objectPath, name);
		} else {
			hid_t type = H5T_NATIVE_DOUBLE;
			if constexpr (std::is_same_v<T, int64_t>) type = H5T_NATIVE_INT64;
			if constexpr (std::is_same_v<T, uint64_t>) type = H5T_NATIVE_UINT64;
			hid_t attr = H5Aopen_by_name(fileHandle, objectPath.c_str(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT);
			if (attr < 0) {
				throw std::runtime_error("Could not open attribute " + name + " of " + objectPath);
			}
			T value;
			if (H5Aread(attr, type, &value) < 0) {
				throw std::runtime_error("Could not read attribute " + name + " of " + objectPath);
			}
			if (H5Aclose(attr) < 0) {
				throw std::runtime_error("Could not close attribute " + name + " of " + objectPath);
			}
			return value;
		}
	}

	//The deck is the string attribute restart/deck
	std::string readDeckImpl(){
		return readStringAttribute("restart", "deck");
	}

	/*
	Internal function to read an array of sizes that is an attribute of an object
	*/
	std::vector<size_t> readSizeArrayAttribute(const std::string &objectPath, const std::string &key){
		hid_t attr = H5Aopen_by_name(fileHandle, objectPath.c_str(), key.c_str(), H5P_DEFAULT, H5P_DEFAULT);
		if (attr < 0) {
			throw std::runtime_error("Could not open attribute " + key + " of " + objectPath);
		}
		hid_t dataspace = H5Aget_space(attr);
		if (dataspace < 0) {
			throw std::runtime_error("Could not get dataspace of attribute " + key + " of " + objectPath);
		}
		const hssize_t points = H5Sget_simple_extent_npoints(dataspace);
		if (points < 0) {
			throw std::runtime_error("Could not get size of attribute " + key + " of " + objectPath);
		}
		std::vector<uint64_t> buffer(static_cast<size_t>(points));
		if (H5Aread(attr, H5T_NATIVE_UINT64, buffer.data()) < 0) {
			throw std::runtime_error("Could not read attribute " + key + " of " + objectPath);
		}
		if (H5Sclose(dataspace) < 0) {
			throw std::runtime_error("Could not close dataspace of attribute " + key + " of " + objectPath);
		}
		if (H5Aclose(attr) < 0) {
			throw std::runtime_error("Could not close attribute " + key + " of " + objectPath);
		}
		return std::vector<size_t>(buffer.begin(), buffer.end());
	}

	/*
	Internal function to read an array of core indices that is an attribute of an object
	*/
	std::vector<SIGNED_INDEX_TYPE> readIndexArrayAttribute(const std::string &objectPath, const std::string &key){
		hid_t attr = H5Aopen_by_name(fileHandle, objectPath.c_str(), key.c_str(), H5P_DEFAULT, H5P_DEFAULT);
		if (attr < 0) {
			throw std::runtime_error("Could not open attribute " + key + " of " + objectPath);
		}
		hid_t dataspace = H5Aget_space(attr);
		if (dataspace < 0) {
			throw std::runtime_error("Could not get dataspace of attribute " + key + " of " + objectPath);
		}
		const hssize_t points = H5Sget_simple_extent_npoints(dataspace);
		if (points < 0) {
			throw std::runtime_error("Could not get size of attribute " + key + " of " + objectPath);
		}
		std::vector<int64_t> buffer(static_cast<size_t>(points));
		if (H5Aread(attr, H5T_NATIVE_INT64, buffer.data()) < 0) {
			throw std::runtime_error("Could not read attribute " + key + " of " + objectPath);
		}
		if (H5Sclose(dataspace) < 0) {
			throw std::runtime_error("Could not close dataspace of attribute " + key + " of " + objectPath);
		}
		if (H5Aclose(attr) < 0) {
			throw std::runtime_error("Could not close attribute " + key + " of " + objectPath);
		}
		return std::vector<SIGNED_INDEX_TYPE>(buffer.begin(), buffer.end());
	}

	//An axis is a dataset that is not a data item or a blob
	std::vector<std::string> axisNamesImpl(){
		std::vector<std::string> names;
		const std::string prefix = "blobs/";
		for (const std::string &dataset : findObjects().datasets) {
			if (dataset.compare(0, prefix.size(), prefix) != 0 && !attributeExists(dataset, "meshName")) names.push_back(dataset);
		}
		return names;
	}

	//A mesh is a group in the meshes namespace
	std::vector<std::string> meshNamesImpl(){
		std::vector<std::string> names;
		const std::string prefix = "meshes/";
		for (const std::string &group : findObjects().groups) {
			if (group.compare(0, prefix.size(), prefix) == 0 && attributeExists(group, "sizes")) names.push_back(group.substr(prefix.size()));
		}
		return names;
	}

	//A data item is a dataset with a mesh name
	std::vector<std::string> dataNamesImpl(){
		std::vector<std::string> names;
		for (const std::string &dataset : findObjects().datasets) {
			if (attributeExists(dataset, "meshName")) names.push_back(dataset);
		}
		return names;
	}

	//A blob is a dataset in the blobs group
	std::vector<std::string> blobNamesImpl(){
		std::vector<std::string> names;
		const std::string prefix = "blobs/";
		for (const std::string &dataset : findObjects().datasets) {
			if (dataset.compare(0, prefix.size(), prefix) == 0) names.push_back(dataset.substr(prefix.size()));
		}
		return names;
	}

	itemInfo axisInfoImpl(const char* name){
		if (!pathExists(name)) {
			throw std::invalid_argument("File does not have the axis " + std::string(name));
		}
		itemInfo info = datasetInfo(name);
		info.starts[0] = readIndexArrayAttribute(name, "start")[0];
		const std::vector<SIGNED_INDEX_TYPE> ghosts = readIndexArrayAttribute(name, "ghosts");
		info.lowerGhosts = static_cast<size_t>(ghosts[0]);
		info.upperGhosts = static_cast<size_t>(ghosts[1]);
		return info;
	}

	//A mesh is a group in the meshes namespace. The axis of each dimension is the attribute axis<dimension>, and the first cells and numbers of cells are the attributes starts and sizes
	itemInfo meshInfoImpl(const char* name){
		const std::string groupName = "meshes/" + std::string(name);
		if (!pathExists(groupName)) {
			throw std::invalid_argument("File does not have the mesh " + std::string(name));
		}
		const std::vector<SIGNED_INDEX_TYPE> starts = readIndexArrayAttribute(groupName, "starts");
		const std::vector<size_t> sizes = readSizeArrayAttribute(groupName, "sizes");
		itemInfo info;
		info.rank = sizes.size();
		for (size_t i = 0; i < info.rank; ++i) {
			info.axes[i] = readStringAttribute(groupName, "axis" + std::to_string(i));
			info.starts[i] = starts[i];
			info.sizes[i] = sizes[i];
		}
		return info;
	}

	itemInfo dataInfoImpl(const char* name){
		if (!pathExists(name) || !attributeExists(name, "meshName")) {
			throw std::invalid_argument("File does not have the data " + std::string(name));
		}
		itemInfo info = datasetInfo(name);
		info.meshName = readStringAttribute(name, "meshName");
		const std::vector<SIGNED_INDEX_TYPE> starts = readIndexArrayAttribute(name, "starts");
		for (size_t i = 0; i < info.rank; ++i) info.starts[i] = starts[i];
		return info;
	}

	itemInfo blobInfoImpl(const char* name){
		const std::string blobName = "blobs/" + std::string(name);
		if (!pathExists(blobName)) {
			throw std::invalid_argument("File does not have the blob " + std::string(name));
		}
		return datasetInfo(blobName);
	}

	std::string readMeshKeyValueImpl(const char* meshName, const char* key){
		return readStringAttribute("meshes/" + std::string(meshName), key);
	}

	std::string readDataKeyValueImpl(const char* dataName, const char* key){
		return readStringAttribute(dataName, key);
	}

	std::string readBlobKeyValueImpl(const char* blobName, const char* key){
		return readStringAttribute("blobs/" + std::string(blobName), key);
	}

	std::string readAxisKeyValueImpl(const char* axisName, const char* key){
		return readStringAttribute(axisName, key);
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
	constexpr static std::string_view readerName = "hdf5";

	HDF5Reader(SAMS::harness &h) : reader<HDF5Reader>(h) {
	}

	//Close the file while the members that closing it uses still exist
	~HDF5Reader(){
		if (this->isOpen) this->closeFile();
	}
};

} // namespace SAMS::IO
#endif
#endif
