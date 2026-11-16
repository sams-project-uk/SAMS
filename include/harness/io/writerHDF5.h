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
#ifndef WRITER_HDF5_H
#define WRITER_HDF5_H
#include <cstddef>
#include <iostream>
#include "writerProto.h"
//Filesystem functions
#include <filesystem>
#include <fstream>
#if __has_include(<hdf5.h>)
#include <hdf5.h>
#include "hdf5Types.h"
#include "harnessDef.h"
#include "mpiManager.h"

namespace SAMS::IO
{

class HDF5File : public writer<HDF5File> {
	friend class writer<HDF5File>;
	using writerMeshInfo = writer<HDF5File>::meshInfo;
  using writerRLMeshInfo = writer<HDF5File>::rlMeshInfo;
	using writerDataInfo = writer<HDF5File>::dataInfo;

	const static bool registerAfterWrite_impl=true;
	const static bool supportsKeyValue_impl=true;
	const static bool supportsMeshKeyValue_impl=true;
	const static bool supportsDataKeyValue_impl=true;
	const static bool supportsTrueNamespaces_impl=true;
	const static meshDataOrder registerOrder_impl=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder_impl=meshDataOrder::noOrder;


  //HDF5 file handle
  hid_t fileHandle;
  //The HDF5 types of the C++ types
  hdf5Types types;

	void openFileImpl([[maybe_unused]] const dumpInfo &info){
    std::string h5filename = filename;
    #ifdef USE_MPI
    SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> &mpi = this->h.MPIManager;
    //If the file exists rank 0 deletes it before any rank creates it
    if (mpi.getRank() == 0 && std::filesystem::exists(h5filename)) {
      SAMS::debug3 << "File " << h5filename << " already exists. Deleting it." << std::endl;
      std::filesystem::remove(h5filename);
    }
    MPI_Barrier(mpi.getComm());
    //Create the file for parallel access by all ranks
    hid_t accessList = H5Pcreate(H5P_FILE_ACCESS);
    if (accessList < 0) {
      throw std::runtime_error("Could not create file access property list for " + h5filename);
    }
    if (H5Pset_fapl_mpio(accessList, mpi.getComm(), MPI_INFO_NULL) < 0) {
      throw std::runtime_error("Could not set parallel access for " + h5filename);
    }
    fileHandle = H5Fcreate(h5filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, accessList);
    if (H5Pclose(accessList) < 0) {
      throw std::runtime_error("Could not close file access property list for " + h5filename);
    }
    #else
    //If the file exists delete it
    if (std::filesystem::exists(h5filename)) {
      SAMS::debug3 << "File " << h5filename << " already exists. Deleting it." << std::endl;
      std::filesystem::remove(h5filename);
    }
    //Create the file
    fileHandle = H5Fcreate(h5filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    #endif
    if (fileHandle < 0) {
      throw std::runtime_error("Could not create file " + std::string(filename));
    }
    //Blobs are written into their own group, so that their names are separate from data names
    hid_t blobGroup = H5Gcreate(fileHandle, "blobs", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (blobGroup < 0) {
      throw std::runtime_error("Could not create group blobs");
    }
    if (H5Gclose(blobGroup) < 0) {
      throw std::runtime_error("Could not close group blobs");
    }
    //The simulation time of the dump
    setKeyValueImpl<double>("time", static_cast<double>(info.time));
    //The simulation step of the dump
    setKeyValueImpl<stepType>("step", static_cast<stepType>(info.step));
  }

  /*
  None of these functions are needed for HDF5, but they are here to satisfy the interface
  */
	void startRegisterImpl(){}
	void startWriteImpl(){}
	void registerRectilinearMeshImpl([[maybe_unused]] const char* name, [[maybe_unused]] const writerRLMeshInfo &info){
	}
	void registerAxisImpl([[maybe_unused]] const char* name, [[maybe_unused]] writer<HDF5File>::axisInfo &info){
	}

  /*
  Internal function to create the transfer property list for writing datasets
  Writes are collective when running with MPI
  */
  hid_t createTransferList(){
    #ifdef USE_MPI
    hid_t transferList = H5Pcreate(H5P_DATASET_XFER);
    if (transferList < 0) {
      throw std::runtime_error("Could not create dataset transfer property list");
    }
    if (H5Pset_dxpl_mpio(transferList, H5FD_MPIO_COLLECTIVE) < 0) {
      throw std::runtime_error("Could not set collective transfer");
    }
    return transferList;
    #else
    return H5P_DEFAULT;
    #endif
  }

  /*
  Internal function to close the transfer property list for writing datasets
  */
  void closeTransferList([[maybe_unused]] hid_t transferList){
    #ifdef USE_MPI
    if (H5Pclose(transferList) < 0) {
      throw std::runtime_error("Could not close dataset transfer property list");
    }
    #endif
  }

  /*
  Internal function to write a string attribute on an object
  Every rank of the file writes it, because creating attributes is collective
  */
  void writeStringAttribute(hid_t object, const std::string &key, const std::string &value){
    hid_t strType = H5Tcopy(H5T_C_S1);
    if (strType < 0) {
      throw std::runtime_error("Could not create string datatype for attribute " + key);
    }
    if (H5Tset_size(strType, H5T_VARIABLE) < 0) {
      throw std::runtime_error("Could not set size of string datatype for attribute " + key);
    }
    hid_t scalar = H5Screate(H5S_SCALAR);
    if (scalar < 0) {
      throw std::runtime_error("Could not create dataspace for attribute " + key);
    }
    hid_t attr = H5Acreate(object, key.c_str(), strType, scalar, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
      throw std::runtime_error("Could not create attribute " + key);
    }
    const char* valueCStr = value.c_str();
    if (H5Awrite(attr, strType, &valueCStr) < 0) {
      throw std::runtime_error("Could not write attribute " + key);
    }
    if (H5Aclose(attr) < 0) {
      throw std::runtime_error("Could not close attribute " + key);
    }
    if (H5Sclose(scalar) < 0) {
      throw std::runtime_error("Could not close dataspace for attribute " + key);
    }
    if (H5Tclose(strType) < 0) {
      throw std::runtime_error("Could not close string datatype for attribute " + key);
    }
  }

  /*
  Internal function to write an attribute on an object with the type of its value: a string, or a scalar double, int64_t or uint64_t
  Every rank of the file writes it, because creating attributes is collective
  */
  template<typename T>
  void writeAttribute(hid_t object, const std::string &key, const T &value){
    if constexpr (std::is_same_v<T, std::string>) {
      writeStringAttribute(object, key, value);
    } else {
      hid_t type = H5T_NATIVE_DOUBLE;
      if constexpr (std::is_same_v<T, int64_t>) type = H5T_NATIVE_INT64;
      if constexpr (std::is_same_v<T, uint64_t>) type = H5T_NATIVE_UINT64;
      hid_t scalar = H5Screate(H5S_SCALAR);
      if (scalar < 0) {
        throw std::runtime_error("Could not create dataspace for attribute " + key);
      }
      hid_t attr = H5Acreate(object, key.c_str(), type, scalar, H5P_DEFAULT, H5P_DEFAULT);
      if (attr < 0) {
        throw std::runtime_error("Could not create attribute " + key);
      }
      if (H5Awrite(attr, type, &value) < 0) {
        throw std::runtime_error("Could not write attribute " + key);
      }
      if (H5Aclose(attr) < 0) {
        throw std::runtime_error("Could not close attribute " + key);
      }
      if (H5Sclose(scalar) < 0) {
        throw std::runtime_error("Could not close dataspace for attribute " + key);
      }
    }
  }

  /*
  Internal function to create the link creation property list that creates the groups of the namespaces of a name as they are needed
  */
  hid_t createLinkList(){
    hid_t linkList = H5Pcreate(H5P_LINK_CREATE);
    if (linkList < 0) {
      throw std::runtime_error("Could not create link creation property list");
    }
    if (H5Pset_create_intermediate_group(linkList, 1) < 0) {
      throw std::runtime_error("Could not set creation of intermediate groups");
    }
    return linkList;
  }

  /*
  Internal function to close the link creation property list
  */
  void closeLinkList(hid_t linkList){
    if (H5Pclose(linkList) < 0) {
      throw std::runtime_error("Could not close link creation property list");
    }
  }

  /*
  Internal function to create a group, creating the groups of its namespaces as they are needed
  */
  hid_t createGroup(const std::string &name){
    hid_t linkList = createLinkList();
    hid_t group = H5Gcreate(fileHandle, name.c_str(), linkList, H5P_DEFAULT, H5P_DEFAULT);
    closeLinkList(linkList);
    return group;
  }

  /*
  Internal function to create a dataset, creating the groups of its namespaces as they are needed
  */
  hid_t createDataset(const std::string &name, hid_t type, hid_t dataspace){
    hid_t linkList = createLinkList();
    hid_t dataset = H5Dcreate(fileHandle, name.c_str(), type, dataspace, linkList, H5P_DEFAULT, H5P_DEFAULT);
    closeLinkList(linkList);
    return dataset;
  }

  /*
  Write a key-value pair in the global scope, as an attribute with the type of its value on the group of its namespace, or of the root group if it has none
  The group is created, with the groups of the namespaces above it, if it does not exist
  Every rank of the file writes it, because creating groups and attributes is collective
  */
  template<typename T>
  void setKeyValueImpl(const char* key, const T &value){
    const auto [groupName, attributeName] = splitNamespace(key);
    if (groupName.empty()) {
      writeAttribute(fileHandle, attributeName, value);
      return;
    }
    //Check each level of the namespace in turn, because checking a link below one that does not exist is an error
    bool exists = true;
    for (size_t pos = groupName.find('/'); ; pos = groupName.find('/', pos + 1)) {
      if (H5Lexists(fileHandle, groupName.substr(0, pos).c_str(), H5P_DEFAULT) <= 0) {
        exists = false;
        break;
      }
      if (pos == std::string::npos) break;
    }
    hid_t group;
    if (exists) {
      group = H5Gopen(fileHandle, groupName.c_str(), H5P_DEFAULT);
    } else {
      group = createGroup(groupName);
    }
    if (group < 0) {
      throw std::runtime_error("Could not open group " + groupName);
    }
    writeAttribute(group, attributeName, value);
    if (H5Gclose(group) < 0) {
      throw std::runtime_error("Could not close group " + groupName);
    }
  }

  /*
  Write the deck as the string attribute restart/deck
  */
  void writeDeckImpl(const std::string &text){
    setKeyValueImpl<std::string>("restart/deck", text);
  }

  /*
  Write an axis as a dataset at its name
  Every rank holds the whole axis and creates the dataset, but only rank 0 writes the values
  */
  template <typename T_data>
    void writeAxisImpl(const char* name, const T_data *data, writer<HDF5File>::axisInfo &info){
      //Create the dataspaces for the axis in the file and in memory
      hsize_t dims[1] = {info.size};
      hid_t dataspace = H5Screate_simple(1, dims, NULL);
      if (dataspace < 0) {
        throw std::runtime_error("Could not create dataspace for axis " + std::string(name));
      }
      hid_t memspace = H5Screate_simple(1, dims, NULL);
      if (memspace < 0) {
        throw std::runtime_error("Could not create memory dataspace for axis " + std::string(name));
      }
      //Ranks other than 0 take part in the write without writing anything
      if (this->h.MPIManager.getRank() != 0) {
        if (H5Sselect_none(dataspace) < 0 || H5Sselect_none(memspace) < 0) {
          throw std::runtime_error("Could not clear selection for axis " + std::string(name));
        }
      }
      //Create the dataset for the axis
      hid_t dataset = createDataset(name, types.get<T_data>(), dataspace);
      if (dataset < 0) {
        throw std::runtime_error("Could not create dataset for axis " + std::string(name));
      }
      //Write the axis to the dataset
      hid_t transferList = createTransferList();
      if (H5Dwrite(dataset, types.get<T_data>(), memspace, dataspace, transferList, data) < 0) {
        throw std::runtime_error("Could not write axis " + std::string(name) + " to dataset");
      }
      closeTransferList(transferList);
      //The core index of the first element of the axis
      writeIndexArrayAttribute(dataset, "start", &info.start, 1);
      //The number of ghost cells below and above the domain
      const SIGNED_INDEX_TYPE ghosts[2] = {static_cast<SIGNED_INDEX_TYPE>(info.lowerGhosts), static_cast<SIGNED_INDEX_TYPE>(info.upperGhosts)};
      writeIndexArrayAttribute(dataset, "ghosts", ghosts, 2);
      //The key-value pairs of the axis are attributes of its dataset
      for (const auto &pair : info.keyValues) {
        writeStringAttribute(dataset, pair.first, pair.second);
      }
      //Close the dataset
      if (H5Dclose(dataset) < 0) {
        throw std::runtime_error("Could not close dataset for axis " + std::string(name));
      }
      //Close the dataspaces
      if (H5Sclose(dataspace) < 0) {
        throw std::runtime_error("Could not close dataspace for axis " + std::string(name));
      }
      if (H5Sclose(memspace) < 0) {
        throw std::runtime_error("Could not close memory dataspace for axis " + std::string(name));
      }
    }

  /*
  Internal function to write an array of sizes as an attribute of an object
  Every rank of the file writes it, because creating attributes is collective
  */
  void writeSizeArrayAttribute(hid_t object, const std::string &key, const size_t *values, size_t count){
    std::vector<uint64_t> buffer(values, values + count);
    hsize_t dims[1] = {count};
    hid_t dataspace = H5Screate_simple(1, dims, NULL);
    if (dataspace < 0) {
      throw std::runtime_error("Could not create dataspace for attribute " + key);
    }
    hid_t attr = H5Acreate(object, key.c_str(), H5T_NATIVE_UINT64, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
      throw std::runtime_error("Could not create attribute " + key);
    }
    if (H5Awrite(attr, H5T_NATIVE_UINT64, buffer.data()) < 0) {
      throw std::runtime_error("Could not write attribute " + key);
    }
    if (H5Aclose(attr) < 0) {
      throw std::runtime_error("Could not close attribute " + key);
    }
    if (H5Sclose(dataspace) < 0) {
      throw std::runtime_error("Could not close dataspace for attribute " + key);
    }
  }

  /*
  Internal function to write an array of core indices as an attribute of an object
  Every rank of the file writes it, because creating attributes is collective
  */
  void writeIndexArrayAttribute(hid_t object, const std::string &key, const SIGNED_INDEX_TYPE *values, size_t count){
    std::vector<int64_t> buffer(values, values + count);
    hsize_t dims[1] = {count};
    hid_t dataspace = H5Screate_simple(1, dims, NULL);
    if (dataspace < 0) {
      throw std::runtime_error("Could not create dataspace for attribute " + key);
    }
    hid_t attr = H5Acreate(object, key.c_str(), H5T_NATIVE_INT64, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
      throw std::runtime_error("Could not create attribute " + key);
    }
    if (H5Awrite(attr, H5T_NATIVE_INT64, buffer.data()) < 0) {
      throw std::runtime_error("Could not write attribute " + key);
    }
    if (H5Aclose(attr) < 0) {
      throw std::runtime_error("Could not close attribute " + key);
    }
    if (H5Sclose(dataspace) < 0) {
      throw std::runtime_error("Could not close dataspace for attribute " + key);
    }
  }

  /*
  Write a mesh as a group in the meshes namespace. A mesh holds no values of its own
  The axis of each dimension is the attribute axis<dimension>, and the first cell and the number of cells of the axis in each dimension are the attributes starts and sizes
  */
  void writeRectilinearMeshImpl(const char* name, const writerMeshInfo &info){
    const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
    const std::string groupName = "meshes/" + std::string(name);
    hid_t group = createGroup(groupName);
    if (group < 0) {
      throw std::runtime_error("Could not create group " + groupName);
    }
    for (size_t i = 0; i < rlInfo.rank; ++i) {
      writeStringAttribute(group, "axis" + std::to_string(i), rlInfo.axes[i]);
    }
    writeIndexArrayAttribute(group, "starts", rlInfo.starts.data(), rlInfo.rank);
    writeSizeArrayAttribute(group, "sizes", rlInfo.sizes.data(), rlInfo.rank);
    //The key-value pairs of the mesh are attributes of its group
    for (const auto &pair : info.keyValues) {
      writeStringAttribute(group, pair.first, pair.second);
    }
    //Close the group
    if (H5Gclose(group) < 0) {
      throw std::runtime_error("Could not close group " + std::string(name));
    }
  }


	void registerDataImpl([[maybe_unused]] const char* name, [[maybe_unused]] const char* meshName, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo){
	}

	//Write data against a mest. Each rank that writes writes its own part into the whole dataset. Every rank of the file takes part, because the calls are collective
	template<typename T_data>
		void writeDataImpl(const char* name, const T_data *data, const decomposition &part, writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
      [[maybe_unused]] const SAMSMPI::comm comm = part.comm;
      //Just write a dataset into the root group and add the mesh name as an attribute
      hsize_t globalSizes[maximumRank], globalStarts[maximumRank], counts[maximumRank], localSizes[maximumRank], localStarts[maximumRank];
      for (size_t i = 0; i < dataInfo.rank; ++i) {
        globalSizes[i] = dataInfo.sizes[i];
        globalStarts[i] = part.globalStarts[i];
        counts[i] = part.counts[i];
        localSizes[i] = part.localSizes[i];
        localStarts[i] = part.localStarts[i];
      }
      //Create the dataspace for the whole data and select the part that this rank writes
      hid_t dataspace = H5Screate_simple(static_cast<int>(dataInfo.rank), globalSizes, NULL);
      if (dataspace < 0) {
        throw std::runtime_error("Could not create dataspace for " + std::string(name) + " data");
      }
      if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, globalStarts, NULL, counts, NULL) < 0) {
        throw std::runtime_error("Could not select part in dataspace for " + std::string(name) + " data");
      }
      //Create the dataspace for the local array and select the part without the ghost cells
      hid_t memspace = H5Screate_simple(static_cast<int>(dataInfo.rank), localSizes, NULL);
      if (memspace < 0) {
        throw std::runtime_error("Could not create memory dataspace for " + std::string(name) + " data");
      }
      if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET, localStarts, NULL, counts, NULL) < 0) {
        throw std::runtime_error("Could not select part in memory dataspace for " + std::string(name) + " data");
      }
      #ifdef USE_MPI
      //A rank that does not write takes part with nothing selected
      if (comm == SAMSMPI::commNull) {
        if (H5Sselect_none(dataspace) < 0 || H5Sselect_none(memspace) < 0) {
          throw std::runtime_error("Could not clear selection for " + std::string(name) + " data");
        }
      }
      #endif
      //Create the dataset for the data
      hid_t dataset = createDataset(name, types.get<T_data>(), dataspace);
      if (dataset < 0) {
        throw std::runtime_error("Could not create dataset for " + std::string(name) + " data");
      }
      //Write the data to the dataset
      hid_t transferList = createTransferList();
      if (H5Dwrite(dataset, types.get<T_data>(), memspace, dataspace, transferList, data) < 0) {
        throw std::runtime_error("Could not write " + std::string(name) + " data to dataset");
      }
      closeTransferList(transferList);
      //The mesh name is an attribute of the data
      writeStringAttribute(dataset, "meshName", dataInfo.meshName);
      //The first cell of the data on the mesh's axis in each dimension. Its sizes are the shape of the dataset
      writeIndexArrayAttribute(dataset, "starts", dataInfo.starts.data(), dataInfo.rank);
      //The key-value pairs of the data are attributes of its dataset
      for (const auto &pair : dataInfo.keyValues) {
        writeStringAttribute(dataset, pair.first, pair.second);
      }
      //Close the dataset
      if (H5Dclose(dataset) < 0) {
        throw std::runtime_error("Could not close dataset for " + std::string(name) + " data");
      }
      //Close the dataspaces
      if (H5Sclose(dataspace) < 0) {
        throw std::runtime_error("Could not close dataspace for " + std::string(name) + " data");
      }
      if (H5Sclose(memspace) < 0) {
        throw std::runtime_error("Could not close memory dataspace for " + std::string(name) + " data");
      }
		}

	void registerBlobImpl([[maybe_unused]] const char* name, [[maybe_unused]] writer<HDF5File>::blobInfo &info){
	}

	//Write a blob into the blobs group. Each rank that writes writes its own part into the whole blob. Every rank of the file takes part, because the calls are collective
	template<typename T_data>
		void writeBlobImpl(const char* name, const T_data *data, const decomposition &part, writer<HDF5File>::blobInfo &info) {
      [[maybe_unused]] const SAMSMPI::comm comm = part.comm;
      const std::string blobName = "blobs/" + std::string(name);
      hsize_t globalSizes[maximumRank], globalStarts[maximumRank], counts[maximumRank], localSizes[maximumRank], localStarts[maximumRank];
      for (size_t i = 0; i < info.rank; ++i) {
        globalSizes[i] = info.sizes[i];
        globalStarts[i] = part.globalStarts[i];
        counts[i] = part.counts[i];
        localSizes[i] = part.localSizes[i];
        localStarts[i] = part.localStarts[i];
      }
      //Create the dataspace for the whole blob and select the part that this rank writes
      hid_t dataspace = H5Screate_simple(static_cast<int>(info.rank), globalSizes, NULL);
      if (dataspace < 0) {
        throw std::runtime_error("Could not create dataspace for " + blobName + " blob");
      }
      if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, globalStarts, NULL, counts, NULL) < 0) {
        throw std::runtime_error("Could not select part in dataspace for " + blobName + " blob");
      }
      //Create the dataspace for the local array and select the part without the ghost cells
      hid_t memspace = H5Screate_simple(static_cast<int>(info.rank), localSizes, NULL);
      if (memspace < 0) {
        throw std::runtime_error("Could not create memory dataspace for " + blobName + " blob");
      }
      if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET, localStarts, NULL, counts, NULL) < 0) {
        throw std::runtime_error("Could not select part in memory dataspace for " + blobName + " blob");
      }
      #ifdef USE_MPI
      //A rank that does not write takes part with nothing selected
      if (comm == SAMSMPI::commNull) {
        if (H5Sselect_none(dataspace) < 0 || H5Sselect_none(memspace) < 0) {
          throw std::runtime_error("Could not clear selection for " + blobName + " blob");
        }
      }
      #endif
      //Create the dataset for the blob
      hid_t dataset = createDataset(blobName, types.get<T_data>(), dataspace);
      if (dataset < 0) {
        throw std::runtime_error("Could not create dataset for " + blobName + " blob");
      }
      //Write the blob to the dataset
      hid_t transferList = createTransferList();
      if (H5Dwrite(dataset, types.get<T_data>(), memspace, dataspace, transferList, data) < 0) {
        throw std::runtime_error("Could not write " + blobName + " blob to dataset");
      }
      closeTransferList(transferList);
      //The key-value pairs of the blob are attributes of its dataset
      for (const auto &pair : info.keyValues) {
        writeStringAttribute(dataset, pair.first, pair.second);
      }
      //Close the dataset
      if (H5Dclose(dataset) < 0) {
        throw std::runtime_error("Could not close dataset for " + blobName + " blob");
      }
      //Close the dataspaces
      if (H5Sclose(dataspace) < 0) {
        throw std::runtime_error("Could not close dataspace for " + blobName + " blob");
      }
      if (H5Sclose(memspace) < 0) {
        throw std::runtime_error("Could not close memory dataspace for " + blobName + " blob");
      }
		}

	//Flush file to disk
	void flushFileImpl() {
    if (H5Fflush(fileHandle, H5F_SCOPE_GLOBAL) < 0) {
      throw std::runtime_error("Could not flush file to disk");
    }
	}

	//Close file
	void closeFileImpl() {
    if (H5Fclose(fileHandle) < 0) {
      throw std::runtime_error("Could not close file");
    }
	}

  public:

  //The name of the output format, which selects this writer in the deck
  constexpr static std::string_view writerName = "hdf5";

  //The extension of the files that the writer writes
  constexpr static std::string_view extension = ".h5";

  HDF5File(SAMS::harness &h) : writer<HDF5File>(h) {
  }

  //Close the file while the members that closing it uses still exist
  ~HDF5File(){
    if (this->isOpen) this->closeFile();
    //Close the types that were created for the files
    types.close();
  }
};

} // namespace SAMS::IO
#endif
#endif