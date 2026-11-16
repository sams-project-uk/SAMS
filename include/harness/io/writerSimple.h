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
#ifndef WRITER_SIMPLE_H
#define WRITER_SIMPLE_H
#include <complex>
#include <cstddef>
#include <iostream>
#include "writerProto.h"
//Filesystem functions
#include <filesystem>
#include <fstream>
#include <sstream>
#include "mpiManager.h"

namespace SAMS::IO
{

class simpleFile : public writer<simpleFile> {
	friend class writer<simpleFile>;
	using writerMeshInfo = writer<simpleFile>::meshInfo;
  using writerRLMeshInfo = writer<simpleFile>::rlMeshInfo;
	using writerDataInfo = writer<simpleFile>::dataInfo;

	const static bool registerAfterWrite_impl=true;
	const static bool supportsKeyValue_impl=true;
	const static bool supportsMeshKeyValue_impl=true;
	const static bool supportsDataKeyValue_impl=true;
	const static bool supportsTrueNamespaces_impl=true;
	const static meshDataOrder registerOrder_impl=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder_impl=meshDataOrder::noOrder;

	//Key-value pairs in the global scope, which are written when the file is closed
	std::map<std::string, std::string> globalKeyValues;

	//Store a key-value pair in the global scope as text. A real number is written in hexadecimal floating point, so that it is read back exactly
	template<typename T>
	void setKeyValueImpl(const char* key, const T &value){
		if constexpr (std::is_same_v<T, std::string>) {
			globalKeyValues[key] = value;
		} else if constexpr (std::is_same_v<T, double>) {
			std::ostringstream text;
			text << std::hexfloat << value;
			globalKeyValues[key] = text.str();
		} else {
			globalKeyValues[key] = std::to_string(value);
		}
	}

	//Write the deck to input.deck in the directory. Only rank 0 writes it
	void writeDeckImpl(const std::string &text){
		if (this->h.MPIManager.getRank() != 0) return;
		const std::string deckFileName = std::string(this->filename) + "/input.deck";
		auto file = std::ofstream(deckFileName);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file " + deckFileName);
		}
		file << text;
		if (!file) {
			throw std::runtime_error("Could not write file " + deckFileName);
		}
		file.close();
	}

	//Create the directory of the namespace of a file, with the directories of the namespaces above it
	void createNamespaceDirectory(const std::string &fileName){
		std::filesystem::create_directories(std::filesystem::path(fileName).parent_path());
	}

	//Write key-value pairs to a file in the directory of its namespace: the number of pairs, then the pairs
	void writeKeyValueFile(const std::string &fileName, const std::map<std::string, std::string> &pairs){
		createNamespaceDirectory(fileName);
		auto file = std::ofstream(fileName);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file " + fileName);
		}
		file << pairs.size() << "\n";
		for (const auto &pair : pairs){
			file << pair.first << " " << pair.second << "\n";
		}
		if (!file) {
			throw std::runtime_error("Could not write file " + fileName);
		}
		file.close();
	}

	//Copy the part of a local array that is written, without the ghost cells, into a contiguous buffer with the last dimension fastest
	template<typename T_data>
	std::vector<T_data> gatherPart(const T_data *data, const decomposition &part, size_t rank){
		size_t total = 1;
		for (size_t i=0;i<rank;++i) total *= part.counts[i];
		std::vector<T_data> buffer(total);
		std::array<size_t, maximumRank> index{};
		for (size_t n=0;n<total;++n){
			size_t offset = 0;
			for (size_t i=0;i<rank;++i) offset = offset * part.localSizes[i] + part.localStarts[i] + index[i];
			buffer[n] = data[offset];
			for (size_t i=rank;i-- > 0;){
				if (++index[i] < part.counts[i]) break;
				index[i] = 0;
			}
		}
		return buffer;
	}

#ifdef USE_MPI
	//The first rank of a communicator creates the directory of the namespace of a file, and every rank of it waits until the directory exists
	void createNamespaceDirectory(const std::string &fileName, SAMSMPI::comm comm){
		int commRank;
		if (MPI_Comm_rank(comm, &commRank) != MPI_SUCCESS) {
			throw std::runtime_error("Could not get rank in communicator for " + fileName);
		}
		if (commRank == 0) createNamespaceDirectory(fileName);
		if (MPI_Barrier(comm) != MPI_SUCCESS) {
			throw std::runtime_error("Could not wait for the directory of " + fileName);
		}
	}
#endif


	//filename is the directory of the dump
	void openFileImpl([[maybe_unused]] const dumpInfo &info){
        //Only rank 0 creates the directory
        if (this->h.MPIManager.getRank() == 0) {
            //Check if there is a directory with the specified filename
            if (std::filesystem::exists(filename)) {
                SAMS::debug3 << "Directory " << filename << " already exists. Deleting.\n";
                //Delete the directory and its contents
                std::filesystem::remove_all(filename);
            }
            SAMS::debug3 << "Creating directory " << filename << "\n";
            std::filesystem::create_directory(filename);
        }
        #ifdef USE_MPI
        //Every rank waits for the directory to exist
        if (MPI_Barrier(this->h.MPIManager.getComm()) != MPI_SUCCESS) {
            throw std::runtime_error("Could not wait for directory " + std::string(filename));
        }
        #endif
        //The simulation time of the dump
        setKeyValueImpl<double>("time", static_cast<double>(info.time));
        //The simulation step of the dump
        setKeyValueImpl<stepType>("step", static_cast<stepType>(info.step));
    }

	void startRegisterImpl(){}
	void startWriteImpl(){}

	void registerRectilinearMeshImpl([[maybe_unused]] const char* name, [[maybe_unused]] const writerRLMeshInfo &info){
	}

	void registerAxisImpl([[maybe_unused]] const char* name, [[maybe_unused]] writer<simpleFile>::axisInfo &info){
	}

	//Write the line of a header that gives the type of an item: its category, then its size in bytes
	template<typename T_data>
	void writeTypeLine(std::ostream &stream){
		if constexpr (std::is_integral<T_data>::value && std::is_unsigned<T_data>::value) {
			stream << "uint ";
		} else if constexpr (std::is_integral<T_data>::value) {
			stream << "int ";
		} else if constexpr (std::is_floating_point<T_data>::value) {
			stream << "float ";
		} else if constexpr (std::is_same_v<T_data, std::complex<float>> || std::is_same_v<T_data, std::complex<double>>) {
			stream << "complex ";
		} else {
			static_assert("Unsupported data type");
		}
		stream << sizeof(T_data) << "\n";
	}

	//Write an axis. Every rank holds the whole axis, so only rank 0 writes it
	//The header is the number of header lines that follow, the type and size, the number of elements, the core index of the first element, the numbers of ghost cells below and above the domain, then the key-value pairs of the axis. The values follow the header
	template<typename T_data>
	void writeAxisImpl(const char* name, const T_data *data, writer<simpleFile>::axisInfo &info){
		if (this->h.MPIManager.getRank() != 0) return;
		//Combine the "filename" (which is actually the directory) with the axis name and ".axis"
		std::string axisFileName = std::string(this->filename) + "/" + std::string(name) + ".axis";
		//Create the directory of the namespace of the axis, then open the file for writing
		createNamespaceDirectory(axisFileName);
		auto file = std::ofstream(axisFileName, std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file " + axisFileName);
		}
		std::ostringstream headerStream;
		headerStream << 4 + info.keyValues.size() << "\n";
		writeTypeLine<T_data>(headerStream);
		headerStream << info.size << "\n";
		headerStream << info.start << "\n";
		headerStream << info.lowerGhosts << " " << info.upperGhosts << "\n";
		for (const auto &pair : info.keyValues){
			headerStream << pair.first << " " << pair.second << "\n";
		}
		file << headerStream.str();
		file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(info.size * sizeof(T_data)));
		if (!file) {
			throw std::runtime_error("Could not write file " + axisFileName);
		}
		file.close();
	}

	//Write a mesh. A mesh holds no values of its own. Every rank holds the whole mesh, so only rank 0 writes it
	//The header is the number of header lines that follow, the rank, then for each dimension the axis, the first cell and the number of cells, then the key-value pairs of the mesh
	void writeRectilinearMeshImpl(const char* name, const writerMeshInfo &info){
		if (this->h.MPIManager.getRank() != 0) return;
		const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
		//Combine the "filename" (which is actually the directory) with the mesh name and ".mesh"
		std::string meshFileName = std::string(this->filename) + "/" + std::string(name) + ".mesh";
		//Create the directory of the namespace of the mesh, then open the file for writing
		createNamespaceDirectory(meshFileName);
		auto file = std::ofstream(meshFileName);
		if (!file.is_open()) {
			throw std::runtime_error("Could not open file " + meshFileName);
		}
		file << 1 + rlInfo.rank + info.keyValues.size() << "\n";
		file << rlInfo.rank << "\n";
		for (size_t i=0;i<rlInfo.rank;++i){
			file << rlInfo.axes[i] << " " << rlInfo.starts[i] << " " << rlInfo.sizes[i] << "\n";
		}
		for (const auto &pair : info.keyValues){
			file << pair.first << " " << pair.second << "\n";
		}
		if (!file) {
			throw std::runtime_error("Could not write file " + meshFileName);
		}
		file.close();
	}

	void registerDataImpl([[maybe_unused]] const char* name, [[maybe_unused]] const char* meshName, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo){
	}

	//Write data against a mest. Each rank that writes writes its own part into the whole data
	template<typename T_data>
		void writeDataImpl(const char* name, const T_data *data, const decomposition &part, writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
            [[maybe_unused]] const SAMSMPI::comm comm = part.comm;
            //Open the file. Combine the "filename" (which is actually the directory) with the mesh name and ".data"
            std::string dataFileName = std::string(this->filename) + "/" + std::string(name) + ".data";
            //Build the header. Every rank builds it so that every rank knows where the data starts
            //It is the number of header lines that follow, the mesh name, the type and size, the sizes, the first cells on the mesh's axes, then the key-value pairs of the data
            std::ostringstream headerStream;
            headerStream << 4 + dataInfo.keyValues.size() << "\n";
            headerStream << dataInfo.meshName << "\n";
            writeTypeLine<T_data>(headerStream);
            for (size_t i=0;i<dataInfo.rank;++i){
                headerStream << dataInfo.sizes[i] << (i + 1 < dataInfo.rank ? " " : "\n");
            }
            for (size_t i=0;i<dataInfo.rank;++i){
                headerStream << dataInfo.starts[i] << (i + 1 < dataInfo.rank ? " " : "\n");
            }
            for (const auto &pair : dataInfo.keyValues){
                headerStream << pair.first << " " << pair.second << "\n";
            }
            const std::string header = headerStream.str();
            #ifdef USE_MPI
            SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> &mpi = this->h.MPIManager;
            //A rank that does not write takes no part
            if (comm == SAMSMPI::commNull) return;
            //The ranks that write create the directory of the namespace of the data
            createNamespaceDirectory(dataFileName, comm);
            //Open the file for writing by the ranks that write
            MPI_File file;
            if (MPI_File_open(comm, dataFileName.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
                throw std::runtime_error("Could not open file " + dataFileName);
            }
            //The first rank that writes writes the header
            int commRank;
            if (MPI_Comm_rank(comm, &commRank) != MPI_SUCCESS) {
                throw std::runtime_error("Could not get rank in communicator");
            }
            if (commRank == 0) {
                if (MPI_File_write_at(file, 0, header.data(), static_cast<int>(header.size()), MPI_CHAR, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                    throw std::runtime_error("Could not write header to file " + dataFileName);
                }
            }
            //Each rank sees only its own part of the whole data in the file, and sends only the part without the ghost cells from memory
            std::array<int, maximumRank> globalSizes, globalStarts, counts, localSizes, localStarts;
            for (size_t i=0;i<dataInfo.rank;++i){
                globalSizes[i] = static_cast<int>(dataInfo.sizes[i]);
                globalStarts[i] = static_cast<int>(part.globalStarts[i]);
                counts[i] = static_cast<int>(part.counts[i]);
                localSizes[i] = static_cast<int>(part.localSizes[i]);
                localStarts[i] = static_cast<int>(part.localStarts[i]);
            }
            SAMSMPI::datatype baseType = SAMS::gettypeRegistry().getMPIType<T_data>();
            SAMSMPI::datatype fileType = mpi.createMPISubarrayType(static_cast<int>(dataInfo.rank), globalSizes.data(), counts.data(), globalStarts.data(), baseType);
            SAMSMPI::datatype memoryType = mpi.createMPISubarrayType(static_cast<int>(dataInfo.rank), localSizes.data(), counts.data(), localStarts.data(), baseType);
            if (MPI_File_set_view(file, static_cast<MPI_Offset>(header.size()), baseType, fileType, "native", MPI_INFO_NULL) != MPI_SUCCESS) {
                throw std::runtime_error("Could not set view of file " + dataFileName);
            }
            if (MPI_File_write_all(file, data, 1, memoryType, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                throw std::runtime_error("Could not write data to file " + dataFileName);
            }
            if (MPI_File_close(&file) != MPI_SUCCESS) {
                throw std::runtime_error("Could not close file " + dataFileName);
            }
            mpi.checkinMPIType(fileType);
            mpi.checkinMPIType(memoryType);
            #else
            //Create the directory of the namespace of the data, then open the file for writing
            createNamespaceDirectory(dataFileName);
            auto file = std::ofstream(dataFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + dataFileName);
            }
            //Write the data to the file
            file << header;
            //Write the part without the ghost cells, with the last dimension fastest
            const std::vector<T_data> buffer = gatherPart(data, part, dataInfo.rank);
            file.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(T_data)));
            if (!file) {
                throw std::runtime_error("Could not write file " + dataFileName);
            }
            file.close();
            #endif
		}

	void registerBlobImpl([[maybe_unused]] const char* name, [[maybe_unused]] writer<simpleFile>::blobInfo &info){
	}

	//Write a blob. Each rank that writes writes its own part into the whole blob
	template<typename T_data>
		void writeBlobImpl(const char* name, const T_data *data, const decomposition &part, writer<simpleFile>::blobInfo &info) {
            [[maybe_unused]] const SAMSMPI::comm comm = part.comm;
            //Open the file. Combine the "filename" (which is actually the directory) with the blob name and ".blob"
            std::string blobFileName = std::string(this->filename) + "/" + std::string(name) + ".blob";
            //Build the header. Every rank builds it so that every rank knows where the data starts
            //It is the number of header lines that follow, the type and size, the size in each dimension, then the key-value pairs of the blob
            std::ostringstream headerStream;
            headerStream << 2 + info.keyValues.size() << "\n";
            writeTypeLine<T_data>(headerStream);
            for (size_t i=0;i<info.rank;++i){
                headerStream << info.sizes[i] << (i + 1 < info.rank ? " " : "\n");
            }
            for (const auto &pair : info.keyValues){
                headerStream << pair.first << " " << pair.second << "\n";
            }
            const std::string header = headerStream.str();
            #ifdef USE_MPI
            SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> &mpi = this->h.MPIManager;
            //A rank that does not write takes no part
            if (comm == SAMSMPI::commNull) return;
            //The ranks that write create the directory of the namespace of the blob
            createNamespaceDirectory(blobFileName, comm);
            //Open the file for writing by the ranks that write
            MPI_File file;
            if (MPI_File_open(comm, blobFileName.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
                throw std::runtime_error("Could not open file " + blobFileName);
            }
            //The first rank that writes writes the header
            int commRank;
            if (MPI_Comm_rank(comm, &commRank) != MPI_SUCCESS) {
                throw std::runtime_error("Could not get rank in communicator");
            }
            if (commRank == 0) {
                if (MPI_File_write_at(file, 0, header.data(), static_cast<int>(header.size()), MPI_CHAR, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                    throw std::runtime_error("Could not write header to file " + blobFileName);
                }
            }
            //Each rank sees only its own part of the whole blob in the file, and sends only the part without the ghost cells from memory
            std::array<int, maximumRank> globalSizes, globalStarts, counts, localSizes, localStarts;
            for (size_t i=0;i<info.rank;++i){
                globalSizes[i] = static_cast<int>(info.sizes[i]);
                globalStarts[i] = static_cast<int>(part.globalStarts[i]);
                counts[i] = static_cast<int>(part.counts[i]);
                localSizes[i] = static_cast<int>(part.localSizes[i]);
                localStarts[i] = static_cast<int>(part.localStarts[i]);
            }
            SAMSMPI::datatype baseType = SAMS::gettypeRegistry().getMPIType<T_data>();
            SAMSMPI::datatype fileType = mpi.createMPISubarrayType(static_cast<int>(info.rank), globalSizes.data(), counts.data(), globalStarts.data(), baseType);
            SAMSMPI::datatype memoryType = mpi.createMPISubarrayType(static_cast<int>(info.rank), localSizes.data(), counts.data(), localStarts.data(), baseType);
            if (MPI_File_set_view(file, static_cast<MPI_Offset>(header.size()), baseType, fileType, "native", MPI_INFO_NULL) != MPI_SUCCESS) {
                throw std::runtime_error("Could not set view of file " + blobFileName);
            }
            if (MPI_File_write_all(file, data, 1, memoryType, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                throw std::runtime_error("Could not write blob to file " + blobFileName);
            }
            if (MPI_File_close(&file) != MPI_SUCCESS) {
                throw std::runtime_error("Could not close file " + blobFileName);
            }
            mpi.checkinMPIType(fileType);
            mpi.checkinMPIType(memoryType);
            #else
            //Create the directory of the namespace of the blob, then open the file for writing
            createNamespaceDirectory(blobFileName);
            auto file = std::ofstream(blobFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + blobFileName);
            }
            //Write the blob to the file
            file << header;
            //Write the part without the ghost cells, with the last dimension fastest
            const std::vector<T_data> buffer = gatherPart(data, part, info.rank);
            file.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(T_data)));
            if (!file) {
                throw std::runtime_error("Could not write file " + blobFileName);
            }
            file.close();
            #endif
		}

	//Flush file to disk
	void flushFileImpl() {
	}

	//Close file. Rank 0 writes the key-value pairs in the global scope to a keyValues file in the directory of each namespace: the number of pairs, then the pairs
	void closeFileImpl() {
		if (this->h.MPIManager.getRank() == 0) {
			//Group the pairs by namespace. The pairs with no namespace are in the top directory
			std::map<std::string, std::map<std::string, std::string>> namespaces;
			for (const auto &pair : globalKeyValues){
				const auto [space, key] = splitNamespace(pair.first);
				namespaces[space][key] = pair.second;
			}
			for (const auto &space : namespaces){
				writeKeyValueFile(std::string(this->filename) + "/" + (space.first.empty() ? "" : space.first + "/") + "keyValues", space.second);
			}
		}
		globalKeyValues.clear();
	}

  public:

  //The name of the output format, which selects this writer in the deck
  constexpr static std::string_view writerName = "simple";

  //The extension of the files that the writer writes
  constexpr static std::string_view extension = "";

  simpleFile(SAMS::harness &h) : writer<simpleFile>(h) {
  }

  //Close the file while the members that closing it uses still exist
  ~simpleFile(){
    if (this->isOpen) this->closeFile();
  }

};

} // namespace SAMS::IO

#endif
