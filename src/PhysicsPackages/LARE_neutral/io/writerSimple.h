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
class simpleFile : public writer<simpleFile> {
	friend class writer<simpleFile>;
	using writerMeshInfo = writer<simpleFile>::meshInfo;
  using writerRLMeshInfo = writer<simpleFile>::rlMeshInfo;
	using writerDataInfo = writer<simpleFile>::dataInfo;

	const static bool registerAfterWrite=true;
	const static meshDataOrder registerOrder=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder=meshDataOrder::noOrder;


	void openFileImpl(const char* filename){
        //Check if there is a directory with the specified filename
        if (std::filesystem::exists(filename)) {
            std::cout << "Directory " << filename << " already exists. Deleting.\n";
            //Delete the directory and its contents
            std::filesystem::remove_all(filename);
        }
        std::cout << "Creating directory " << filename << "\n";
        std::filesystem::create_directory(filename);
    }

	void startRegisterImpl(){}
	void startWriteImpl(){}

	void registerRectilinearMeshImpl([[maybe_unused]] const char* name, [[maybe_unused]] const writerRLMeshInfo &info, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz){
	}

	void registerRectilinearMeshImpl([[maybe_unused]] const char* name,  [[maybe_unused]] const writerRLMeshInfo &info, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny){
	}

	void registerRectilinearMeshImpl([[maybe_unused]] const char* name,  [[maybe_unused]] const writerRLMeshInfo &info, [[maybe_unused]] const size_t nx){
	}

	//Write 3d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name,  const writerMeshInfo &info, const T_data *x, const T_data *y, const T_data *z){
            const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
            //Open the file. Combine the "filename" (which is actually the directory) with the mesh name and ".mesh"
            std::string meshFileName = std::string(this->filename) + "/" + std::string(name) + ".mesh";
            //Open the file for writing
            auto file = std::ofstream(meshFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + meshFileName);
            }
            //Write the mesh data to the file
            file << rlInfo.sizes[0] << " " << rlInfo.sizes[1] << " " << rlInfo.sizes[2] << "\n";
						file << "#x\n";
						for ( size_t i=0;i<rlInfo.sizes[0];++i){
							file << x[i] << "\n";
						}
						file << "#y\n";
            for (size_t j=0;j<rlInfo.sizes[1];++j){
              file << y[j] << "\n";
            }
						file << "#z\n";
            for (size_t k=0;k<rlInfo.sizes[2];++k){
              file << z[k] << "\n";
            }
            file.close();

		}
	//Write 2d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name,  const writerMeshInfo &info, const T_data *x, const T_data *y) {
            const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
            //Open the file. Combine the "filename" (which is actually the directory) with the mesh name and ".mesh"
            std::string meshFileName = std::string(this->filename) + "/" + std::string(name) + ".mesh";
            //Open the file for writing
            auto file = std::ofstream(meshFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + meshFileName);
            }
            //Write the mesh data to the file
            file << rlInfo.sizes[0] << " " << rlInfo.sizes[1] << " " << "\n";
            file << "#x\n";
            for (size_t i=0;i<rlInfo.sizes[0];++i){
              file << x[i] << "\n";
            }
						file << "#y\n";
            for (size_t j=0;j<rlInfo.sizes[1];++j){
              file << y[j] << "\n";
            }
            file.close();
		}
	//Write 1d rectilinear mesh
	template <typename T_data>
		void writeRectilinearMeshImpl(const char* name,  const writerMeshInfo &info, const T_data *x){
            const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
            //Open the file. Combine the "filename" (which is actually the directory) with the mesh name and ".mesh"
            std::string meshFileName = std::string(this->filename) + "/" + std::string(name) + ".mesh";
            //Open the file for writing
            auto file = std::ofstream(meshFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + meshFileName);
            }
            //Write the mesh data to the file
            file << rlInfo.sizes[0] << " " << rlInfo.sizes[1] << " " << "\n";
            file << "#x\n";
            for (size_t i=0;i<rlInfo.sizes[0];++i){
              file << x[i] << "\n";
            }
            file.close();
		}

	void registerDataImpl([[maybe_unused]] const char* name, [[maybe_unused]] const char* meshName, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo){
	}

	//Write data against a mest
	template<typename T_data>
		void writeDataImpl(const char* name, const T_data *data, writerDataInfo &dataInfo, writerMeshInfo &meshInfo) {
            //Open the file. Combine the "filename" (which is actually the directory) with the mesh name and ".data"
            std::string dataFileName = std::string(this->filename) + "/" + std::string(name) + ".data";
            //Open the file for writing
            auto file = std::ofstream(dataFileName);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file " + dataFileName);
            }
            //Write the data to the file
            file << dataInfo.meshName << "\n";
            if constexpr (std::is_integral<T_data>::value) {
                file << "int ";
            } else if constexpr (std::is_floating_point<T_data>::value) {
                file << "float ";
            } else {
                static_assert("Unsupported data type");
            }
            file << sizeof(T_data) << "\n";
            file.write(reinterpret_cast<const char*>(data), meshInfo.zones * sizeof(T_data));
            file.close();
		}

	//Flush file to disk
	void flushFileImpl() {
	}

	//Close file
	void closeFileImpl() {
	}
};

#endif
