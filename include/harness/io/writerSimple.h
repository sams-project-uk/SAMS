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
#include <cstddef>
#include <iostream>
#include "writerProto.h"
//Filesystem functions
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include "mpiManager.h"
class simpleFile : public writer<simpleFile> {
	friend class writer<simpleFile>;
	using writerMeshInfo = writer<simpleFile>::meshInfo;
  using writerRLMeshInfo = writer<simpleFile>::rlMeshInfo;
	using writerDataInfo = writer<simpleFile>::dataInfo;
	using writerBlockInfo = writer<simpleFile>::blockInfo;

	const static bool registerAfterWrite=true;
	const static meshDataOrder registerOrder=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder=meshDataOrder::noOrder;


	void openFileImpl(const char* filename){
        //Only rank 0 creates the directory
        if (this->h.MPIManager.getRank() == 0) {
            //Check if there is a directory with the specified filename
            if (std::filesystem::exists(filename)) {
                std::cout << "Directory " << filename << " already exists. Deleting.\n";
                //Delete the directory and its contents
                std::filesystem::remove_all(filename);
            }
            std::cout << "Creating directory " << filename << "\n";
            std::filesystem::create_directory(filename);
        }
        #ifdef USE_MPI
        //Every rank waits for the directory to exist
        MPI_Barrier(this->h.MPIManager.getComm());
        #endif
    }

	void startRegisterImpl(){}
	void startWriteImpl(){}

	void registerRectilinearMeshImpl([[maybe_unused]] const char* name, [[maybe_unused]] const writerRLMeshInfo &info){
	}

	//Write rectilinear mesh of any rank. Only rank 0 writes the mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name,  const writerMeshInfo &info, const std::array<const T_data*, maximumRank> &axes){
            if (this->h.MPIManager.getRank() != 0) return;
            const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
            //Open the file. Combine the "filename" (which is actually the directory) with the mesh name and ".mesh"
            std::string meshFileName = std::string(this->filename) + "/" + std::string(name) + ".mesh";
            //Open the file for writing
            auto file = std::ofstream(meshFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + meshFileName);
            }
            //Write the mesh data to the file
            for (size_t i=0;i<rlInfo.rank;++i){
              file << rlInfo.sizes[i] << (i + 1 < rlInfo.rank ? " " : "\n");
            }
            for (size_t i=0;i<rlInfo.rank;++i){
              file << "#" << rlInfo.axisNames[i] << "\n";
              for (size_t j=0;j<rlInfo.sizes[i];++j){
                file << axes[i][j] << "\n";
              }
            }
            file.close();
		}

	void registerDataImpl([[maybe_unused]] const char* name, [[maybe_unused]] const char* meshName, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo){
	}

	//Write data against a mest. Each rank writes its own block into the whole data
	template<typename T_data>
		void writeDataImpl(const char* name, const T_data *data, const writerBlockInfo &block, writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
            //Open the file. Combine the "filename" (which is actually the directory) with the mesh name and ".data"
            std::string dataFileName = std::string(this->filename) + "/" + std::string(name) + ".data";
            //Build the header. Every rank builds it so that every rank knows where the data starts
            std::ostringstream headerStream;
            headerStream << dataInfo.meshName << "\n";
            if constexpr (std::is_integral<T_data>::value) {
                headerStream << "int ";
            } else if constexpr (std::is_floating_point<T_data>::value) {
                headerStream << "float ";
            } else {
                static_assert("Unsupported data type");
            }
            headerStream << sizeof(T_data) << "\n";
            const std::string header = headerStream.str();
            #ifdef USE_MPI
            SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> &mpi = this->h.MPIManager;
            //Open the file for writing by all ranks
            MPI_File file;
            if (MPI_File_open(mpi.getComm(), dataFileName.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
                throw std::runtime_error("Could not open file " + dataFileName);
            }
            //Rank 0 writes the header
            if (mpi.getRank() == 0) {
                if (MPI_File_write_at(file, 0, header.data(), static_cast<int>(header.size()), MPI_CHAR, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                    throw std::runtime_error("Could not write header to file " + dataFileName);
                }
            }
            //Each rank sees only its own block of the whole data in the file, and sends only the block without the ghost cells from memory
            std::array<int, maximumRank> globalSizes, globalStarts, counts, localSizes, localStarts;
            for (size_t i=0;i<block.rank;++i){
                globalSizes[i] = static_cast<int>(block.globalSizes[i]);
                globalStarts[i] = static_cast<int>(block.globalStarts[i]);
                counts[i] = static_cast<int>(block.counts[i]);
                localSizes[i] = static_cast<int>(block.localSizes[i]);
                localStarts[i] = static_cast<int>(block.localStarts[i]);
            }
            MPI_Datatype baseType = SAMS::gettypeRegistry().getMPIType<T_data>();
            MPI_Datatype fileType = mpi.createMPISubarrayType(static_cast<int>(block.rank), globalSizes.data(), counts.data(), globalStarts.data(), baseType);
            MPI_Datatype memoryType = mpi.createMPISubarrayType(static_cast<int>(block.rank), localSizes.data(), counts.data(), localStarts.data(), baseType);
            if (MPI_File_set_view(file, static_cast<MPI_Offset>(header.size()), baseType, fileType, "native", MPI_INFO_NULL) != MPI_SUCCESS) {
                throw std::runtime_error("Could not set view of file " + dataFileName);
            }
            if (MPI_File_write_all(file, data, 1, memoryType, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
                throw std::runtime_error("Could not write data to file " + dataFileName);
            }
            MPI_File_close(&file);
            mpi.checkinMPIType(fileType);
            mpi.checkinMPIType(memoryType);
            #else
            //Open the file for writing
            auto file = std::ofstream(dataFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + dataFileName);
            }
            //Write the data to the file
            file << header;
            //Write the block without the ghost cells, with the last dimension fastest
            size_t total = 1;
            for (size_t i=0;i<block.rank;++i) total *= block.counts[i];
            std::array<size_t, maximumRank> index{};
            for (size_t n=0;n<total;++n){
                size_t offset = 0;
                for (size_t i=0;i<block.rank;++i) offset = offset * block.localSizes[i] + block.localStarts[i] + index[i];
                file.write(reinterpret_cast<const char*>(data + offset), sizeof(T_data));
                for (size_t i=block.rank;i-- > 0;){
                    if (++index[i] < block.counts[i]) break;
                    index[i] = 0;
                }
            }
            file.close();
            #endif
		}

	//Flush file to disk
	void flushFileImpl() {
	}

	//Close file
	void closeFileImpl() {
	}

  simpleFile(SAMS::harness &h) : writer<simpleFile>(h) {
  }

};

#endif
