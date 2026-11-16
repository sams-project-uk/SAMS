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
#ifndef IO_RAWIO_H
#define IO_RAWIO_H
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "writerProto.h"
#include "harnessDef.h"
#include "mpiManager.h"
#if defined(USE_HDF5)
#include <hdf5.h>
#include "hdf5Types.h"
#endif
#if defined(USE_ADIOS2)
#include <memory>
#include <sstream>
#include <adios2.h>
#endif
#ifndef USE_MPI
#include <fstream>
#endif

namespace SAMS::IO::raw
{

//The order of the elements of an array in a file
enum class arrayOrder{
	//The last index varies fastest
	C
};

//How an array is laid out in a file
struct arrayLayout{
	arrayOrder order = arrayOrder::C;
};

//The block of an array that this rank reads, and how the array is laid out in the file. The block is always read in C order
struct blockRequest{
	//The part of the array that is read, and where it goes in the local array
	decomposition part;
	arrayLayout layout;
};

//Where an array is in a binary file, which does not describe it
struct binaryLocation{
	//The position of the first element in the file, in bytes
	size_t byteOffset = 0;
	//The rank of the array
	size_t rank = 0;
	//The size of the array in each dimension
	std::array<size_t, SAMS::MAX_RANK> globalSizes{};
};

//The description of an array in a file
struct arrayInfo{
	SAMS::typeHandle type;
	size_t rank = 0;
	std::array<size_t, SAMS::MAX_RANK> sizes{};
};

/**
 * @brief Check that an array in a file has exactly the type that it is read as
 * @tparam T_data The type that the array is read as
 * @param name The name of the array, for errors
 * @param info The array
 * @throws std::invalid_argument if the type does not match
 */
template<typename T_data>
void checkType(const std::string &name, const arrayInfo &info){
	const SAMS::typeHandle type = SAMS::gettypeRegistry().getTypeID<T_data>();
	if (info.type == type) return;
	throw std::invalid_argument("Type of " + name + " in the file is not " + SAMS::gettypeRegistry().getTypeName(type));
}

#if defined(USE_HDF5)
/**
 * @brief An HDF5 file, opened for reading by every rank of a communicator
 */
class hdf5File{
	hid_t fileHandle = H5I_INVALID_HID;
	//The HDF5 types of the C++ types
	hdf5Types types;

	public:
	/**
	 * @brief Open a file
	 * @param filename The name of the file
	 * @param comm The communicator of the ranks that read it
	 */
	hdf5File(const std::string &filename, [[maybe_unused]] SAMSMPI::comm comm){
		#ifdef USE_MPI
		hid_t accessList = H5Pcreate(H5P_FILE_ACCESS);
		if (accessList < 0) {
			throw std::runtime_error("Could not create file access property list for " + filename);
		}
		if (H5Pset_fapl_mpio(accessList, comm, MPI_INFO_NULL) < 0) {
			throw std::runtime_error("Could not set parallel access for " + filename);
		}
		fileHandle = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, accessList);
		if (H5Pclose(accessList) < 0) {
			throw std::runtime_error("Could not close file access property list for " + filename);
		}
		#else
		fileHandle = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		#endif
		if (fileHandle < 0) {
			throw std::runtime_error("Could not open file " + filename);
		}
	}

	hdf5File(const hdf5File&) = delete;
	hdf5File& operator=(const hdf5File&) = delete;

	~hdf5File(){
		types.close();
		H5Fclose(fileHandle);
	}

	/**
	 * @brief Describe a dataset: its type, rank and size
	 * @param name The name of the dataset
	 */
	arrayInfo describe(const std::string &name){
		arrayInfo info;
		hid_t dataset = H5Dopen(fileHandle, name.c_str(), H5P_DEFAULT);
		if (dataset < 0) {
			throw std::runtime_error("Could not open dataset " + name);
		}
		hid_t type = H5Dget_type(dataset);
		if (type < 0) {
			throw std::runtime_error("Could not get type of dataset " + name);
		}
		info.type = typeHandleFromHDF5(type);
		if (H5Tclose(type) < 0) {
			throw std::runtime_error("Could not close type of dataset " + name);
		}
		hid_t dataspace = H5Dget_space(dataset);
		if (dataspace < 0) {
			throw std::runtime_error("Could not get dataspace of dataset " + name);
		}
		const int rank = H5Sget_simple_extent_ndims(dataspace);
		if (rank < 0 || static_cast<size_t>(rank) > info.sizes.size()) {
			throw std::runtime_error("Dataset " + name + " has a rank that cannot be read");
		}
		hsize_t dims[SAMS::MAX_RANK];
		if (H5Sget_simple_extent_dims(dataspace, dims, NULL) < 0) {
			throw std::runtime_error("Could not get sizes of dataset " + name);
		}
		info.rank = static_cast<size_t>(rank);
		for (size_t i = 0; i < info.rank; ++i) info.sizes[i] = dims[i];
		if (H5Sclose(dataspace) < 0) {
			throw std::runtime_error("Could not close dataspace of dataset " + name);
		}
		if (H5Dclose(dataset) < 0) {
			throw std::runtime_error("Could not close dataset " + name);
		}
		return info;
	}

	/**
	 * @brief Read the block of a dataset that this rank holds. Every rank of the file takes part, because the read is collective
	 * @tparam T_data The type of the dataset
	 * @param name The name of the dataset
	 * @param data The local array
	 * @param request The block, and where it goes in the local array
	 * @throws std::invalid_argument if the type of the dataset is not T_data
	 */
	template<typename T_data>
	void readBlock(const std::string &name, T_data *data, const blockRequest &request){
		const arrayInfo info = describe(name);
		checkType<T_data>(name, info);
		const decomposition &part = request.part;
		hsize_t globalStarts[SAMS::MAX_RANK], counts[SAMS::MAX_RANK], localSizes[SAMS::MAX_RANK], localStarts[SAMS::MAX_RANK];
		for (size_t i = 0; i < info.rank; ++i) {
			globalStarts[i] = part.globalStarts[i];
			counts[i] = part.counts[i];
			localSizes[i] = part.localSizes[i];
			localStarts[i] = part.localStarts[i];
		}
		hid_t dataset = H5Dopen(fileHandle, name.c_str(), H5P_DEFAULT);
		if (dataset < 0) {
			throw std::runtime_error("Could not open dataset " + name);
		}
		//Select the block in the file, and where it goes in the local array
		hid_t dataspace = H5Dget_space(dataset);
		if (dataspace < 0) {
			throw std::runtime_error("Could not get dataspace of dataset " + name);
		}
		if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, globalStarts, NULL, counts, NULL) < 0) {
			throw std::runtime_error("Could not select block in dataspace for " + name);
		}
		hid_t memspace = H5Screate_simple(static_cast<int>(info.rank), localSizes, NULL);
		if (memspace < 0) {
			throw std::runtime_error("Could not create memory dataspace for " + name);
		}
		if (H5Sselect_hyperslab(memspace, H5S_SELECT_SET, localStarts, NULL, counts, NULL) < 0) {
			throw std::runtime_error("Could not select block in memory dataspace for " + name);
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
			throw std::runtime_error("Could not read " + name);
		}
		#ifdef USE_MPI
		if (H5Pclose(transferList) < 0) {
			throw std::runtime_error("Could not close dataset transfer property list");
		}
		#endif
		if (H5Sclose(memspace) < 0) {
			throw std::runtime_error("Could not close memory dataspace for " + name);
		}
		if (H5Sclose(dataspace) < 0) {
			throw std::runtime_error("Could not close dataspace for " + name);
		}
		if (H5Dclose(dataset) < 0) {
			throw std::runtime_error("Could not close dataset " + name);
		}
	}
};
#endif

#if defined(USE_ADIOS2)
/**
 * @brief An ADIOS2 file, opened for reading by every rank of a communicator. Any step can be read
 */
class adios2File{
	std::unique_ptr<adios2::ADIOS> adios;
	adios2::IO io;
	adios2::Engine engine;
	//The step that is read, by its position in the file
	size_t selectedStep = 0;

	//The type in the type registry of an ADIOS2 type
	static SAMS::typeHandle typeFromADIOS2(const std::string &type){
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

	public:
	/**
	 * @brief Open a file. Step 0 is selected
	 * @param filename The name of the file
	 * @param comm The communicator of the ranks that read it
	 */
	adios2File(const std::string &filename, [[maybe_unused]] SAMSMPI::comm comm){
		#ifdef USE_MPI
		adios = std::make_unique<adios2::ADIOS>(comm);
		#else
		adios = std::make_unique<adios2::ADIOS>();
		#endif
		io = adios->DeclareIO("SAMSRawRead");
		engine = io.Open(filename, adios2::Mode::ReadRandomAccess);
	}

	adios2File(const adios2File&) = delete;
	adios2File& operator=(const adios2File&) = delete;

	~adios2File(){
		engine.Close();
	}

	/**
	 * @brief The number of steps in the file
	 */
	size_t stepCount(){
		return engine.Steps();
	}

	/**
	 * @brief Select the step that is read
	 * @param step The step, by its position in the file
	 */
	void selectStep(size_t step){
		selectedStep = step;
	}

	/**
	 * @brief Describe a variable: its type, rank and size
	 * @param name The name of the variable
	 */
	arrayInfo describe(const std::string &name){
		const auto variables = io.AvailableVariables();
		const auto it = variables.find(name);
		if (it == variables.end()) {
			throw std::runtime_error("File does not have the variable " + name);
		}
		arrayInfo info;
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

	/**
	 * @brief Read the block of a variable that this rank holds, from the selected step
	 * @tparam T_data The type of the variable
	 * @param name The name of the variable
	 * @param data The local array
	 * @param request The block, and where it goes in the local array
	 * @throws std::invalid_argument if the type of the variable is not T_data
	 */
	template<typename T_data>
	void readBlock(const std::string &name, T_data *data, const blockRequest &request){
		const arrayInfo info = describe(name);
		checkType<T_data>(name, info);
		const decomposition &part = request.part;
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
		variable.SetMemorySelection({memoryStart, memoryCount});
		variable.SetStepSelection({selectedStep, 1});
		engine.Get(variable, data, adios2::Mode::Sync);
	}
};
#endif

/**
 * @brief A file of bare bytes, opened for reading by every rank of the harness's communicator. It is read with MPI-IO under MPI
 */
class binaryFile{
	[[maybe_unused]] SAMS::harness &h;
	#ifdef USE_MPI
	MPI_File file;
	#else
	std::ifstream file;
	#endif
	std::string filename;

	public:
	/**
	 * @brief Open a file
	 * @param h The harness
	 * @param filename The name of the file
	 */
	binaryFile(SAMS::harness &h, const std::string &filename) : h(h), filename(filename){
		#ifdef USE_MPI
		if (MPI_File_open(h.MPIManager.getComm(), filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
			throw std::runtime_error("Could not open file " + filename);
		}
		#else
		file.open(filename, std::ios::binary);
		if (!file) {
			throw std::runtime_error("Could not open file " + filename);
		}
		#endif
	}

	binaryFile(const binaryFile&) = delete;
	binaryFile& operator=(const binaryFile&) = delete;

	~binaryFile(){
		#ifdef USE_MPI
		MPI_File_close(&file);
		#endif
	}

	/**
	 * @brief Read the block of an array that this rank holds. Under MPI every rank of the file takes part, because the read is collective
	 * @tparam T_data The type of the array
	 * @param data The local array
	 * @param request The block, and where it goes in the local array
	 * @param location Where the array is in the file, and its size
	 */
	template<typename T_data>
	void readBlock(T_data *data, const blockRequest &request, const binaryLocation &location){
		const decomposition &part = request.part;
		#ifdef USE_MPI
		int globalSizes[SAMS::MAX_RANK], counts[SAMS::MAX_RANK], globalStarts[SAMS::MAX_RANK], localSizes[SAMS::MAX_RANK], localStarts[SAMS::MAX_RANK];
		for (size_t i = 0; i < location.rank; ++i) {
			globalSizes[i] = static_cast<int>(location.globalSizes[i]);
			counts[i] = static_cast<int>(part.counts[i]);
			globalStarts[i] = static_cast<int>(part.globalStarts[i]);
			localSizes[i] = static_cast<int>(part.localSizes[i]);
			localStarts[i] = static_cast<int>(part.localStarts[i]);
		}
		const int rank = static_cast<int>(location.rank);
		//The block in the file, and where it goes in the local array
		SAMSMPI::datatype element = SAMS::gettypeRegistry().getMPIType<T_data>();
		SAMSMPI::datatype fileType = h.MPIManager.createMPISubarrayType(rank, globalSizes, counts, globalStarts, element);
		SAMSMPI::datatype memoryType = h.MPIManager.createMPISubarrayType(rank, localSizes, counts, localStarts, element);
		if (MPI_File_set_view(file, static_cast<MPI_Offset>(location.byteOffset), element, fileType, "native", MPI_INFO_NULL) != MPI_SUCCESS) {
			throw std::runtime_error("Could not set view of " + filename);
		}
		MPI_Status status;
		if (MPI_File_read_all(file, data, 1, memoryType, &status) != MPI_SUCCESS) {
			throw std::runtime_error("Could not read " + filename);
		}
		//A file that is too short reads less than the whole block
		int read = 0;
		if (MPI_Get_count(&status, memoryType, &read) != MPI_SUCCESS || read != 1) {
			throw std::runtime_error("Could not read the whole block from " + filename);
		}
		h.MPIManager.checkinMPIType(memoryType);
		h.MPIManager.checkinMPIType(fileType);
		#else
		//Each row along the last dimension is contiguous in the file and in the local array
		const size_t last = location.rank - 1;
		std::array<size_t, SAMS::MAX_RANK> index{};
		for (bool more = true; more; ){
			size_t fileOffset = 0, localOffset = 0;
			for (size_t i = 0; i < location.rank; ++i) {
				fileOffset = fileOffset * location.globalSizes[i] + part.globalStarts[i] + index[i];
				localOffset = localOffset * part.localSizes[i] + part.localStarts[i] + index[i];
			}
			file.seekg(static_cast<std::streamoff>(location.byteOffset + fileOffset * sizeof(T_data)));
			file.read(reinterpret_cast<char*>(data + localOffset), static_cast<std::streamsize>(part.counts[last] * sizeof(T_data)));
			if (!file) {
				throw std::runtime_error("Could not read the whole block from " + filename);
			}
			//Move to the next row
			more = false;
			for (size_t i = last; i-- > 0; ) {
				if (++index[i] < part.counts[i]) {
					more = true;
					break;
				}
				index[i] = 0;
			}
		}
		#endif
	}
};

} // namespace SAMS::IO::raw
#endif
