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
#ifndef WRITER_ADIOS2_H
#define WRITER_ADIOS2_H
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include "writerProto.h"
#if __has_include(<adios2.h>)
#include <adios2.h>
#include "harnessDef.h"
#include "mpiManager.h"

namespace SAMS::IO
{

class ADIOS2File : public writer<ADIOS2File> {
	friend class writer<ADIOS2File>;
	using writerMeshInfo = writer<ADIOS2File>::meshInfo;
	using writerRLMeshInfo = writer<ADIOS2File>::rlMeshInfo;
	using writerDataInfo = writer<ADIOS2File>::dataInfo;

	const static bool registerAfterWrite_impl=true;
	const static bool supportsKeyValue_impl=true;
	const static bool supportsMeshKeyValue_impl=true;
	const static bool supportsDataKeyValue_impl=true;
	const static bool supportsTrueNamespaces_impl=true;
	const static bool supportsMultipleTimesteps_impl=true;
	const static meshDataOrder registerOrder_impl=meshDataOrder::noOrder;
	const static meshDataOrder writeOrder_impl=meshDataOrder::noOrder;

	//An open file and its ADIOS2 objects
	struct openStream{
		std::unique_ptr<adios2::ADIOS> adios;
		adios2::IO io;
		adios2::Engine engine;
	};

	//Files that dumps share, with a step for each dump, by name without extension. They stay open until the writer is destroyed
	std::map<std::string, openStream> sharedStreams;
	//Whether the dumps to each file are separate, by the name of the file that they would share. It cannot change once the file is opened
	std::map<std::string, bool> fileModes;
	//The file of a separate dump
	openStream separateStream;
	//The file of the open dump
	openStream *current = nullptr;

	/*
	Open the file of a dump and start its step. Everything in a step, including the metadata, is a variable, so that it can change from step to step
	Unnamed dumps share a file for their suffix, and named dumps a file of their name, unless they are separate
	*/
	void openFileImpl(const dumpInfo &info){
		const auto mode = fileModes.emplace(filename, !info.attemptTimeCoalescence).first;
		if (mode->second != !info.attemptTimeCoalescence) {
			throw std::runtime_error("Dumps to " + filename + " cannot be both separate and shared");
		}
		current = !info.attemptTimeCoalescence ? &separateStream : &sharedStreams[filename];
		if (!current->adios) {
#ifdef USE_MPI
			//Every rank of the file opens it
			current->adios = std::make_unique<adios2::ADIOS>(this->h.MPIManager.getComm());
#else
			current->adios = std::make_unique<adios2::ADIOS>();
#endif
			current->io = current->adios->DeclareIO("SAMS");
			//A restarted run adds its steps to the files that dumps share
			current->engine = current->io.Open(filename, this->restartedRun && current != &separateStream ? adios2::Mode::Append : adios2::Mode::Write);
		}
		current->engine.BeginStep();
		//The simulation time of the dump
		putValue<double>("time", static_cast<double>(info.time));
		//The simulation step of the dump
		putValue<stepType>("step", static_cast<stepType>(info.step));
	}

	//The variable of an array, defined in the first step that writes it and given the shape and selection of each later step
	template<typename T_data>
	adios2::Variable<T_data> arrayVariable(const std::string &name, const adios2::Dims &shape, const adios2::Dims &start, const adios2::Dims &count){
		adios2::Variable<T_data> variable = current->io.InquireVariable<T_data>(name);
		if (!variable) return current->io.DefineVariable<T_data>(name, shape, start, count);
		variable.SetShape(shape);
		variable.SetSelection({start, count});
		return variable;
	}

	//Write a value that every rank holds. Only rank 0 writes it
	template<typename T_data>
	void putValue(const std::string &name, const T_data &value){
		adios2::Variable<T_data> variable = current->io.InquireVariable<T_data>(name);
		if (!variable) variable = current->io.DefineVariable<T_data>(name);
		if (this->h.MPIManager.getRank() == 0) current->engine.Put(variable, value, adios2::Mode::Sync);
	}

	//Write an array that every rank holds. Only rank 0 writes it
	template<typename T_data>
	void putArray(const std::string &name, const std::vector<T_data> &values){
		const bool writes = this->h.MPIManager.getRank() == 0;
		adios2::Variable<T_data> variable = arrayVariable<T_data>(name, {values.size()}, {0}, {writes ? values.size() : 0});
		if (writes) current->engine.Put(variable, values.data(), adios2::Mode::Sync);
	}

	/*
	None of these functions are needed for ADIOS2, because variables are defined when they are written, but they are here to satisfy the interface
	*/
	void startRegisterImpl(){}
	void startWriteImpl(){}
	void registerRectilinearMeshImpl([[maybe_unused]] const char* name, [[maybe_unused]] const writerRLMeshInfo &info){
	}
	void registerAxisImpl([[maybe_unused]] const char* name, [[maybe_unused]] writer<ADIOS2File>::axisInfo &info){
	}
	void registerDataImpl([[maybe_unused]] const char* name, [[maybe_unused]] const char* meshName, [[maybe_unused]] writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo){
	}

	//Write a key-value pair in the global scope, as a value with the type of the value at its key
	template<typename T>
	void setKeyValueImpl(const char* key, const T &value){
		putValue<T>(key, value);
	}

	//Write the deck as the string value restart/deck
	void writeDeckImpl(const std::string &text){
		setKeyValueImpl<std::string>("restart/deck", text);
	}

	/*
	Write an axis as a variable at its name, with its metadata as values named in it
	Every rank holds the whole axis, but only rank 0 writes the values
	The buffers that the writer is given are released when the write returns, so every put is synchronous
	*/
	template <typename T_data>
	void writeAxisImpl(const char* name, const T_data *data, writer<ADIOS2File>::axisInfo &info){
		const std::string axisName = name;
		const size_t count = this->h.MPIManager.getRank() == 0 ? info.size : 0;
		adios2::Variable<T_data> variable = arrayVariable<T_data>(axisName, {info.size}, {0}, {count});
		//The core index of the first element of the axis
		putValue<int64_t>(axisName + "/start", static_cast<int64_t>(info.start));
		//The number of ghost cells below and above the domain
		putArray<int64_t>(axisName + "/ghosts", {static_cast<int64_t>(info.lowerGhosts), static_cast<int64_t>(info.upperGhosts)});
		for (const auto &pair : info.keyValues){
			putValue<std::string>(axisName + "/" + pair.first, pair.second);
		}
		if (count > 0) current->engine.Put(variable, data, adios2::Mode::Sync);
	}

	/*
	Write a mesh as values in the meshes namespace. A mesh holds no values of its own
	The axis of each dimension is axis<dimension>, and the first cell and the number of cells of the axis in each dimension are starts and sizes
	*/
	void writeRectilinearMeshImpl(const char* name, const writerMeshInfo &info){
		const writerRLMeshInfo &rlInfo = std::get<writerRLMeshInfo>(info.specificInfo);
		const std::string meshName = "meshes/" + std::string(name);
		for (size_t i = 0; i < rlInfo.rank; ++i){
			putValue<std::string>(meshName + "/axis" + std::to_string(i), rlInfo.axes[i]);
		}
		putArray<int64_t>(meshName + "/starts", std::vector<int64_t>(rlInfo.starts.begin(), rlInfo.starts.begin() + rlInfo.rank));
		putArray<uint64_t>(meshName + "/sizes", std::vector<uint64_t>(rlInfo.sizes.begin(), rlInfo.sizes.begin() + rlInfo.rank));
		for (const auto &pair : info.keyValues){
			putValue<std::string>(meshName + "/" + pair.first, pair.second);
		}
	}

	//Write data against a mesh. Each rank that writes writes its own part into the whole data. A rank that does not write puts nothing
	template<typename T_data>
	void writeDataImpl(const char* name, const T_data *data, const decomposition &part, writerDataInfo &dataInfo, [[maybe_unused]] writerMeshInfo &meshInfo) {
		[[maybe_unused]] const SAMSMPI::comm comm = part.comm;
		adios2::Dims shape(dataInfo.rank), start(dataInfo.rank), count(dataInfo.rank), memoryStart(dataInfo.rank), memoryCount(dataInfo.rank);
		for (size_t i = 0; i < dataInfo.rank; ++i) {
			shape[i] = dataInfo.sizes[i];
			start[i] = part.globalStarts[i];
			count[i] = part.counts[i];
			memoryStart[i] = part.localStarts[i];
			memoryCount[i] = part.localSizes[i];
		}
		const std::string dataName = name;
		adios2::Variable<T_data> variable = arrayVariable<T_data>(dataName, shape, start, count);
		//The mesh name of the data
		putValue<std::string>(dataName + "/meshName", dataInfo.meshName);
		//The first cell of the data on the mesh's axis in each dimension. Its sizes are the shape of the variable
		putArray<int64_t>(dataName + "/starts", std::vector<int64_t>(dataInfo.starts.begin(), dataInfo.starts.begin() + dataInfo.rank));
		for (const auto &pair : dataInfo.keyValues){
			putValue<std::string>(dataName + "/" + pair.first, pair.second);
		}
#ifdef USE_MPI
		if (comm == SAMSMPI::commNull) return;
#endif
		//The part without the ghost cells is selected from the local array
		variable.SetMemorySelection({memoryStart, memoryCount});
		current->engine.Put(variable, data, adios2::Mode::Sync);
	}

	void registerBlobImpl([[maybe_unused]] const char* name, [[maybe_unused]] writer<ADIOS2File>::blobInfo &info){
	}

	//Write a blob as a variable in the blobs namespace. Each rank that writes writes its own part into the whole blob. A rank that does not write puts nothing
	template<typename T_data>
	void writeBlobImpl(const char* name, const T_data *data, const decomposition &part, writer<ADIOS2File>::blobInfo &info) {
		[[maybe_unused]] const SAMSMPI::comm comm = part.comm;
		adios2::Dims shape(info.rank), start(info.rank), count(info.rank), memoryStart(info.rank), memoryCount(info.rank);
		for (size_t i = 0; i < info.rank; ++i) {
			shape[i] = info.sizes[i];
			start[i] = part.globalStarts[i];
			count[i] = part.counts[i];
			memoryStart[i] = part.localStarts[i];
			memoryCount[i] = part.localSizes[i];
		}
		const std::string blobName = "blobs/" + std::string(name);
		adios2::Variable<T_data> variable = arrayVariable<T_data>(blobName, shape, start, count);
		for (const auto &pair : info.keyValues){
			putValue<std::string>(blobName + "/" + pair.first, pair.second);
		}
#ifdef USE_MPI
		if (comm == SAMSMPI::commNull) return;
#endif
		//The part without the ghost cells is selected from the local array
		variable.SetMemorySelection({memoryStart, memoryCount});
		current->engine.Put(variable, data, adios2::Mode::Sync);
	}

	//Flush file to disk
	void flushFileImpl() {
		current->engine.PerformPuts();
	}

	//End the step of the dump. The file of a separate dump is closed. A shared file stays open until the writer is destroyed
	void closeFileImpl() {
		current->engine.EndStep();
		if (current == &separateStream) {
			separateStream.engine.Close();
			separateStream.adios.reset();
		}
		current = nullptr;
	}

	public:

	//The name of the output format, which selects this writer in the deck
	constexpr static std::string_view writerName = "adios2";

	//The extension of the files that the writer writes
	constexpr static std::string_view extension = ".bp";

	ADIOS2File(SAMS::harness &h) : writer<ADIOS2File>(h) {
	}

	//Close the file, and the files that dumps share, while the members that closing them uses still exist
	~ADIOS2File(){
		if (this->isOpen) this->closeFile();
		for (auto &pair : sharedStreams) pair.second.engine.Close();
	}
};

} // namespace SAMS::IO
#endif
#endif
