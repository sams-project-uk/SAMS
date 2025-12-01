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

	const static bool registerAfterWrite=true;
	const static meshDataOrder registerOrder=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder=meshDataOrder::noOrder;


  //HDF5 file handle
  hid_t fileHandle;

	void openFileImpl(const char* filename){
    std::string h5filename = filename;
    //If the file exists delete it
    if (std::filesystem::exists(h5filename)) {
      SAMS::cout << "File " << h5filename << " already exists. Deleting it." << std::endl;
      std::filesystem::remove(h5filename);
    }
    #ifdef USE_MPI
    SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> &mpi = SAMS::getMPIManager<SAMS::MPI_DECOMPOSITION_RANK>();
    int rank = mpi.getRank();
    h5filename += "_rank" + std::to_string(rank);
    #endif

    h5filename += ".h5";
    //Create the file
    fileHandle = H5Fcreate(h5filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fileHandle < 0) {
      throw std::runtime_error("Could not create file " + std::string(filename));
    }
  }

  /*
  None of these functions are needed for HDF5, but they are here to satisfy the interface
  */
	void startRegisterImpl(){}
	void startWriteImpl(){}
	void registerRectilinearMeshImpl(const char* name, const writerRLMeshInfo &info, const size_t nx, const size_t ny, const size_t nz){
	}
	void registerRectilinearMeshImpl(const char* name, const writerRLMeshInfo &info, const size_t nx, const size_t ny){
	}
	void registerRectilinearMeshImpl(const char* name, const writerRLMeshInfo &info, const size_t nx){
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
  Internal function to write a single axis of a rectilinear mesh
  Used in 1D, 2D, and 3D meshes
  */
  template <typename T_data>
    void writeSingleAxis(const char* axisName, const T_data *data, hid_t group, size_t elements){
      //Create the dataspace for the x, y, and z coordinates
      hsize_t dims[1] = {elements};
      hid_t dataspace = H5Screate_simple(1, dims, NULL);
      if (dataspace < 0) {
        throw std::runtime_error("Could not create dataspace for " + std::string(axisName) + " coordinates");
      }
      //Create the dataset for the x coordinates
      hid_t dataset = H5Dcreate(group, axisName, getHDF5Type<T_data>(), dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (dataset < 0) {
        throw std::runtime_error("Could not create dataset for " + std::string(axisName) + " coordinates");
      }
      //Write the x coordinates to the dataset
      if (H5Dwrite(dataset, getHDF5Type<T_data>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        throw std::runtime_error("Could not write " + std::string(axisName) + " coordinates to dataset");
      }
      //Close the dataset
      if (H5Dclose(dataset) < 0) {
        throw std::runtime_error("Could not close dataset for " + std::string(axisName) + " coordinates");
      }
      //Close the dataspace
      if (H5Sclose(dataspace) < 0) {
        throw std::runtime_error("Could not close dataspace for " + std::string(axisName) + " coordinates");
      }
    }

	//Write 3d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name, const writerMeshInfo &info, const T_data *x, const T_data *y, const T_data *z){

      hid_t group = H5Gcreate(fileHandle, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (group < 0) {
        throw std::runtime_error("Could not create group " + std::string(name));
      }
      const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
      //Use the writeSingleAxis function to write the x, y, and z coordinates
      writeSingleAxis("x", x, group, rlInfo.sizes[0]);
      writeSingleAxis("y", y, group, rlInfo.sizes[1]);
      writeSingleAxis("z", z, group, rlInfo.sizes[2]);
      //Close the group
      if (H5Gclose(group) < 0) {
        throw std::runtime_error("Could not close group " + std::string(name));
      }

		}
	//Write 2d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name, const writerMeshInfo &info, const T_data *x, const T_data *y) {
      //Write the mesh data as a set of 2 1D arrays
      //File is already open
      //Create the group for the mesh
      hid_t group = H5Gcreate(fileHandle, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (group < 0) {
        throw std::runtime_error("Could not create group " + std::string(name));
      }
      const writerRLMeshInfo &rlInfo = info.specificInfo;
      //Use the writeSingleAxis function to write the x and y coordinates
      writeSingleAxis("x", x, group, rlInfo.sizes[0]);
      writeSingleAxis("y", y, group, rlInfo.sizes[1]);
      //Close the group
      if (H5Gclose(group) < 0) {
        throw std::runtime_error("Could not close group " + std::string(name));
      }
		}
	//Write 1d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name, const writerMeshInfo &info, const T_data *x){
      //Write the mesh data as a set of 1 1D arrays
      //File is already open
      //Create the group for the mesh
      hid_t group = H5Gcreate(fileHandle, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (group < 0) {
        throw std::runtime_error("Could not create group " + std::string(name));
      }
      const writerRLMeshInfo &rlInfo = info.specificInfo;
      //Use the writeSingleAxis function to write the x coordinates
      writeSingleAxis("x", x, group, rlInfo.sizes[0]);
      //Close the group
      if (H5Gclose(group) < 0) {
        throw std::runtime_error("Could not close group " + std::string(name));
      }
		}

	void registerDataImpl(const char* name, const char* meshName, writerDataInfo &dataInfo, writerMeshInfo &meshInfo){
	}

	//Write data against a mest
	template<typename T_data>
		void writeDataImpl(const char* name, const T_data *data, writerDataInfo &dataInfo, writerMeshInfo &meshInfo) {
      const writerRLMeshInfo &rlMeshInfo = std::get<writerRLMeshInfo>(meshInfo.specificInfo);
      //Just write a dataset into the root group and add the mesh name as an attribute
      //Create the dataspace for the data
      hsize_t dims[3] = {rlMeshInfo.sizes[0], rlMeshInfo.sizes[1], rlMeshInfo.sizes[2]};
      hid_t dataspace = H5Screate_simple(rlMeshInfo.rank, dims, NULL);
      if (dataspace < 0) {
        throw std::runtime_error("Could not create dataspace for " + std::string(name) + " data");
      }
      //Create the dataset for the data
      hid_t dataset = H5Dcreate(fileHandle, name, getHDF5Type<T_data>(), dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (dataset < 0) {
        throw std::runtime_error("Could not create dataset for " + std::string(name) + " data");
      }
      //Write the data to the dataset
      if (H5Dwrite(dataset, getHDF5Type<T_data>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        throw std::runtime_error("Could not write " + std::string(name) + " data to dataset");
      }
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
      //Close the dataspace
      if (H5Sclose(dataspace) < 0) {
        throw std::runtime_error("Could not close dataspace for " + std::string(name) + " data");
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
};
#endif
#endif
