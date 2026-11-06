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
#include "harnessDef.h"
#include "mpiManager.h"

class HDF5File : public writer<HDF5File> {
	friend class writer<HDF5File>;
	using writerMeshInfo = writer<HDF5File>::meshInfo;
  using writerRLMeshInfo = writer<HDF5File>::rlMeshInfo;
	using writerDataInfo = writer<HDF5File>::dataInfo;
	using writerBlockInfo = writer<HDF5File>::blockInfo;

	const static bool registerAfterWrite=true;
	const static meshDataOrder registerOrder=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder=meshDataOrder::noOrder;


  //HDF5 file handle
  hid_t fileHandle;

	void openFileImpl(const char* filename){
    std::string h5filename = std::string(filename) + ".h5";
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
  }

  /*
  None of these functions are needed for HDF5, but they are here to satisfy the interface
  */
	void startRegisterImpl(){}
	void startWriteImpl(){}
	void registerRectilinearMeshImpl([[maybe_unused]] const char* name, [[maybe_unused]] const writerRLMeshInfo &info){
	}

  /**
   * Internal function used to get HDF5 type from a C++ type
   */
  template<typename T_data>
  hid_t getHDF5Type(){
    if constexpr (std::is_same_v<T_data, float>) {
      return H5T_NATIVE_FLOAT;
    } else if constexpr (std::is_same_v<T_data, double>) {
      return H5T_NATIVE_DOUBLE;
    } else if constexpr (std::is_same_v<T_data, int8_t>) {
      return H5T_NATIVE_INT8;
    } else if constexpr (std::is_same_v<T_data, uint8_t>) {
      return H5T_NATIVE_UINT8;
    } else if constexpr (std::is_same_v<T_data, int16_t>) {
      return H5T_NATIVE_INT16;
    } else if constexpr (std::is_same_v<T_data, uint16_t>) {
      return H5T_NATIVE_UINT16;
    } else if constexpr (std::is_same_v<T_data, int32_t>) {
      return H5T_NATIVE_INT32;
    } else if constexpr (std::is_same_v<T_data, uint32_t>) {
      return H5T_NATIVE_UINT32;
    } else if constexpr (std::is_same_v<T_data, int64_t>) {
      return H5T_NATIVE_INT64;
    } else if constexpr (std::is_same_v<T_data, uint64_t>) {
      return H5T_NATIVE_UINT64;
    }
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
  Internal function to write a single axis of a rectilinear mesh
  Every rank creates the dataset, but only rank 0 writes the coordinates
  */
  template <typename T_data>
    void writeSingleAxis(const char* axisName, const std::string &registryName, const T_data *data, hid_t group, size_t elements, hid_t transferList){
      //Create the dataspaces for the coordinates in the file and in memory
      hsize_t dims[1] = {elements};
      hid_t dataspace = H5Screate_simple(1, dims, NULL);
      if (dataspace < 0) {
        throw std::runtime_error("Could not create dataspace for " + std::string(axisName) + " coordinates");
      }
      hid_t memspace = H5Screate_simple(1, dims, NULL);
      if (memspace < 0) {
        throw std::runtime_error("Could not create memory dataspace for " + std::string(axisName) + " coordinates");
      }
      //Ranks other than 0 take part in the write without writing anything
      if (this->h.MPIManager.getRank() != 0) {
        if (H5Sselect_none(dataspace) < 0 || H5Sselect_none(memspace) < 0) {
          throw std::runtime_error("Could not clear selection for " + std::string(axisName) + " coordinates");
        }
      }
      //Create the dataset for the coordinates
      hid_t dataset = H5Dcreate(group, axisName, getHDF5Type<T_data>(), dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (dataset < 0) {
        throw std::runtime_error("Could not create dataset for " + std::string(axisName) + " coordinates");
      }
      //Write the coordinates to the dataset
      if (H5Dwrite(dataset, getHDF5Type<T_data>(), memspace, dataspace, transferList, data) < 0) {
        throw std::runtime_error("Could not write " + std::string(axisName) + " coordinates to dataset");
      }
      //Store the name of the axis in the axis registry, so that readers do not have to work it out from the position
      hid_t strType = H5Tcopy(H5T_C_S1);
      H5Tset_size(strType, H5T_VARIABLE);
      hid_t attrSpace = H5Screate(H5S_SCALAR);
      hid_t attr = H5Acreate(dataset, "axisName", strType, attrSpace, H5P_DEFAULT, H5P_DEFAULT);
      if (attr < 0) {
        throw std::runtime_error("Could not create name attribute for " + std::string(axisName) + " coordinates");
      }
      const char* registryNameCStr = registryName.c_str();
      if (H5Awrite(attr, strType, &registryNameCStr) < 0) {
        throw std::runtime_error("Could not write name attribute for " + std::string(axisName) + " coordinates");
      }
      if (H5Aclose(attr) < 0 || H5Sclose(attrSpace) < 0 || H5Tclose(strType) < 0) {
        throw std::runtime_error("Could not close name attribute for " + std::string(axisName) + " coordinates");
      }
      //Close the dataset
      if (H5Dclose(dataset) < 0) {
        throw std::runtime_error("Could not close dataset for " + std::string(axisName) + " coordinates");
      }
      //Close the dataspaces
      if (H5Sclose(dataspace) < 0) {
        throw std::runtime_error("Could not close dataspace for " + std::string(axisName) + " coordinates");
      }
      if (H5Sclose(memspace) < 0) {
        throw std::runtime_error("Could not close memory dataspace for " + std::string(axisName) + " coordinates");
      }
    }

  //Write arbitary rank mesh data
  template<typename T_data>
  void writeRectilinearMeshImpl(const char* name, const writerMeshInfo &info, const std::array<const T_data*, maximumRank> &axes){
    static constexpr std::array<std::string_view, 8> axisNames = {"x", "y", "z", "a", "b", "c", "d", "e"};
    const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
    hid_t group = H5Gcreate(fileHandle, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group < 0) {
      throw std::runtime_error("Could not create group " + std::string(name));
    }
    hid_t transferList = createTransferList();
    for (size_t i = 0; i < rlInfo.rank; ++i) {
      writeSingleAxis(axisNames[i].data(), rlInfo.axisNames[i], axes[i], group, rlInfo.sizes[i], transferList);
    }
    closeTransferList(transferList);
    //Close the group
    if (H5Gclose(group) < 0) {
      throw std::runtime_error("Could not close group " + std::string(name));
    }
  }


	void registerDataImpl([[maybe_unused]] const char* name, [[maybe_unused]] const char* meshName, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo){
	}

	//Write data against a mest. Each rank writes its own block into the whole dataset
	template<typename T_data>
		void writeDataImpl(const char* name, const T_data *data, const writerBlockInfo &block, writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
      //Just write a dataset into the root group and add the mesh name as an attribute
      hsize_t globalSizes[maximumRank], globalStarts[maximumRank], counts[maximumRank], localSizes[maximumRank], localStarts[maximumRank];
      for (size_t i = 0; i < block.rank; ++i) {
        globalSizes[i] = block.globalSizes[i];
        globalStarts[i] = block.globalStarts[i];
        counts[i] = block.counts[i];
        localSizes[i] = block.localSizes[i];
        localStarts[i] = block.localStarts[i];
      }
      //Create the dataspace for the whole data and select the block that this rank writes
      hid_t dataspace = H5Screate_simple(static_cast<int>(block.rank), globalSizes, NULL);
      if (dataspace < 0) {
        throw std::runtime_error("Could not create dataspace for " + std::string(name) + " data");
      }
      if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, globalStarts, NULL, counts, NULL) < 0) {
        throw std::runtime_error("Could not select block in dataspace for " + std::string(name) + " data");
      }
      //Create the dataspace for the local array and select the block without the ghost cells
      hid_t memspace = H5Screate_simple(static_cast<int>(block.rank), localSizes, NULL);
      if (memspace < 0) {
        throw std::runtime_error("Could not create memory dataspace for " + std::string(name) + " data");
      }
      if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET, localStarts, NULL, counts, NULL) < 0) {
        throw std::runtime_error("Could not select block in memory dataspace for " + std::string(name) + " data");
      }
      //Create the dataset for the data
      hid_t dataset = H5Dcreate(fileHandle, name, getHDF5Type<T_data>(), dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (dataset < 0) {
        throw std::runtime_error("Could not create dataset for " + std::string(name) + " data");
      }
      //Write the data to the dataset
      hid_t transferList = createTransferList();
      if (H5Dwrite(dataset, getHDF5Type<T_data>(), memspace, dataspace, transferList, data) < 0) {
        throw std::runtime_error("Could not write " + std::string(name) + " data to dataset");
      }
      closeTransferList(transferList);
        // Create a string datatype for the attribute
        hid_t strType = H5Tcopy(H5T_C_S1);
        H5Tset_size(strType, H5T_VARIABLE); // Use variable-length string

        // Create the attribute
        hid_t attr = H5Acreate(dataset, "meshName", strType, H5Screate(H5S_SCALAR), H5P_DEFAULT, H5P_DEFAULT);
        if (attr < 0) {
            throw std::runtime_error("Could not create attribute for " + std::string(name) + " data");
        }

        // Write the string to the attribute
        const char* meshNameCStr = dataInfo.meshName.c_str();
        if (H5Awrite(attr, strType, &meshNameCStr) < 0) {
            throw std::runtime_error("Could not write attribute for " + std::string(name) + " data");
        }

        // Close the attribute and datatype
        if (H5Aclose(attr) < 0) {
            throw std::runtime_error("Could not close attribute for " + std::string(name) + " data");
        }
        if (H5Tclose(strType) < 0) {
            throw std::runtime_error("Could not close string datatype for " + std::string(name) + " data");
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

  HDF5File(SAMS::harness &h) : writer<HDF5File>(h) {
  }
};
#endif
#endif