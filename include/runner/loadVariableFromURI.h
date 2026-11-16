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
#ifndef RUNNER_LOADVARIABLEFROMURI_H
#define RUNNER_LOADVARIABLEFROMURI_H
#include <array>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include "harness.h"
#include "io/types.h"
#include "io/rawIO.h"
#include "io/readers.h"
#include "KVS/URI.h"

namespace SAMS
{

namespace detail
{
	/**
	 * @brief Split the query of a URI into its key=value pairs, which are separated by &
	 * @param query The query
	 */
	inline std::map<std::string, std::string> parseURIQuery(const std::string &query){
		std::map<std::string, std::string> values;
		for (size_t start = 0; start < query.size(); ){
			size_t end = query.find('&', start);
			if (end == std::string::npos) end = query.size();
			const std::string pair = query.substr(start, end - start);
			const size_t equals = pair.find('=');
			values[pair.substr(0, equals)] = equals == std::string::npos ? "" : pair.substr(equals + 1);
			start = end + 1;
		}
		return values;
	}

	/**
	 * @brief Read the domain of a data item in a SAMS dump into the part of a variable that this rank holds
	 * @details The domain of the item is the range of its mesh, which must have the same size as the domain of the variable. Any ghost cells of the item are not read
	 * @param reader The reader of the dump, which is open
	 * @param arrayName The name of the data item
	 * @param part The part of the variable that this rank holds, which is read into
	 * @param domainSizes The size of the domain of the variable in each dimension
	 * @param uri The URI of the dump, for errors
	 * @throws std::invalid_argument if the domain of the item does not have the size of the domain of the variable
	 */
	template<typename T_reader, typename T_data, int R>
	void readSAMSData(T_reader &reader, const std::string &arrayName, IO::localVariablePart<T_data, R> &part, const std::array<size_t, SAMS::MAX_RANK> &domainSizes, const std::string &uri){
		const IO::itemInfo data = reader.dataInfo(arrayName.c_str());
		const IO::itemInfo mesh = reader.meshInfo(data.meshName.c_str());
		bool same = data.rank == static_cast<size_t>(R) && mesh.rank == static_cast<size_t>(R);
		for (int i = 0; i < R && same; ++i) same = mesh.sizes[i] == domainSizes[i];
		if (!same) throw std::invalid_argument("The domain of " + arrayName + " in " + uri + " does not have the size of the domain of the variable");
		//The domain starts at the first cell of the mesh, which is after any ghost cells of the item
		const std::array<size_t, R> globalStarts = part.globalStarts;
		for (int i = 0; i < R; ++i) part.globalStarts[i] += static_cast<size_t>(mesh.starts[i] - data.starts[i]);
		reader.readData(arrayName.c_str(), part);
		part.globalStarts = globalStarts;
	}

	//What a scheme needs to load a variable from a URI
	template<typename T_data, int R>
	struct uriRequest{
		SAMS::harness &h;
		//The name of the variable in the variable registry
		const std::string &varName;
		//The URI, for errors
		const std::string &uri;
		//The path of the file, which is joined to the directory if it is relative
		std::string path;
		//The keys and values of the query
		std::map<std::string, std::string> query;
		//The name of the array in the file
		std::string arrayName;
		//The part of the variable that this rank holds, which is read into
		IO::localVariablePart<T_data, R> &part;
		//The block that this rank reads, and how the array is laid out in the file
		IO::raw::blockRequest block;
		//The size of the domain of the variable in each dimension
		std::array<size_t, SAMS::MAX_RANK> domainSizes{};
	};

	/**
	 * @brief Check that an array in a file has the size of the domain of the variable
	 * @param request The request
	 * @param info The array
	 * @throws std::invalid_argument if the array does not have the size of the domain
	 */
	template<typename T_data, int R>
	void checkShape(const uriRequest<T_data, R> &request, const IO::raw::arrayInfo &info){
		bool same = info.rank == static_cast<size_t>(R);
		for (int i = 0; i < R && same; ++i) same = info.sizes[i] == request.domainSizes[i];
		if (!same) throw std::invalid_argument("The array in " + request.uri + " does not have the size of the domain of " + request.varName);
	}

#if defined(USE_HDF5)
	/**
	 * @brief Load from an HDF5 file
	 * @param request The request
	 */
	template<typename T_data, int R>
	void loadHDF5(uriRequest<T_data, R> &request){
		IO::raw::hdf5File file(request.path, request.h.MPIManager.getComm());
		checkShape(request, file.describe(request.arrayName));
		file.readBlock(request.arrayName, request.part.array.data(), request.block);
	}
#endif

#if defined(USE_ADIOS2)
	/**
	 * @brief Load from an ADIOS2 file
	 * @param request The request
	 */
	template<typename T_data, int R>
	void loadADIOS2(uriRequest<T_data, R> &request){
		IO::raw::adios2File file(request.path, request.h.MPIManager.getComm());
		if (request.query.count("step") != 0) file.selectStep(std::stoull(request.query["step"]));
		checkShape(request, file.describe(request.arrayName));
		file.readBlock(request.arrayName, request.part.array.data(), request.block);
	}
#endif

	/**
	 * @brief Load from a binary file
	 * @param request The request
	 */
	template<typename T_data, int R>
	void loadBinary(uriRequest<T_data, R> &request){
		IO::raw::binaryFile file(request.h, request.path);
		IO::raw::binaryLocation location;
		if (request.query.count("byte_offset") != 0) location.byteOffset = std::stoull(request.query["byte_offset"]);
		location.rank = R;
		location.globalSizes = request.domainSizes;
		file.readBlock(request.part.array.data(), request.block, location);
	}

	/**
	 * @brief Load from a data item in a SAMS dump. The format of the dump is hdf5 if it ends in .h5, adios2 if it ends in .bp and simple if it is a directory
	 * @param request The request
	 * @throws std::invalid_argument if the path is not a dump that can be read
	 */
	template<typename T_data, int R>
	void loadSAMS(uriRequest<T_data, R> &request){
		const std::filesystem::path dump(request.path);
#if defined(USE_HDF5)
		if (dump.extension() == ".h5"){
			IO::HDF5Reader reader(request.h);
			reader.openFile((dump.parent_path() / dump.stem()).string().c_str());
			readSAMSData(reader, request.arrayName, request.part, request.domainSizes, request.uri);
			return;
		}
#endif
#if defined(USE_ADIOS2)
		if (dump.extension() == ".bp"){
			IO::ADIOS2Reader reader(request.h);
			reader.openFile((dump.parent_path() / dump.stem()).string().c_str());
			if (request.query.count("step") != 0) reader.selectStep(std::stoull(request.query["step"]));
			readSAMSData(reader, request.arrayName, request.part, request.domainSizes, request.uri);
			return;
		}
#endif
		if (std::filesystem::is_directory(dump)){
			IO::simpleReader reader(request.h);
			reader.openFile(dump.string().c_str());
			readSAMSData(reader, request.arrayName, request.part, request.domainSizes, request.uri);
			return;
		}
		throw std::invalid_argument(request.path + " in " + request.uri + " is not a dump that can be read");
	}
}

/**
 * @brief Load a variable in the variable registry from a file given by a URI
 * @details The scheme is the format of the file: hdf5, adios2 or binary, or sams for a data item in a SAMS dump. The format of a SAMS dump is hdf5 if it ends in .h5,
 * adios2 if it ends in .bp and simple if it is a directory, and only the domain of the item is read. These schemes only name local files, so an authority is the start
 * of a relative path, and scheme://data/file.h5 is the same as scheme:data/file.h5. A relative path is relative to the directory.
 * The name of the array in an HDF5 or ADIOS2 file is the query key variable or the fragment, and it is an error to give both. The other query keys are step, the step of an ADIOS2 file;
 * byte_offset, the position of the array in a binary file; and order, the order of the array in the file, which can only be C.
 * The array must have exactly the type of the variable and the size of its domain
 * @tparam T_data The type of the variable
 * @tparam R The rank of the variable
 * @param h The harness
 * @param varName The name of the variable in the variable registry
 * @param uri The URI of the file
 * @param directory The directory that a relative path is relative to
 * @throws std::invalid_argument if the scheme or order is not known, or the array does not match the variable
 */
template<typename T_data, int R>
void loadVariableFromURI(SAMS::harness &h, const std::string &varName, const std::string &uri, const std::string &directory){
	//The function that loads from each scheme
	static const std::map<std::string, void(*)(detail::uriRequest<T_data, R>&)> schemes{
#if defined(USE_HDF5)
		{"hdf5", detail::loadHDF5<T_data, R>},
#endif
#if defined(USE_ADIOS2)
		{"adios2", detail::loadADIOS2<T_data, R>},
#endif
		{"binary", detail::loadBinary<T_data, R>},
		{"sams", detail::loadSAMS<T_data, R>}
	};

	const URI parsed = URI::parse(uri);
	const std::string filePath = parsed.buildPath();
	IO::localVariablePart<T_data, R> part = IO::getLocalVariablePart<T_data, R>(h, varName);
	detail::uriRequest<T_data, R> request{h, varName, uri, (!filePath.empty() && filePath[0] == '/') ? filePath : directory + "/" + filePath,
		detail::parseURIQuery(parsed.query), "", part, {}, {}};
	std::map<std::string, std::string> &query = request.query;
	//The array is named by the variable key or by the fragment, but not both
	if (query.count("variable") != 0 && !parsed.fragment.empty()) throw std::invalid_argument("The array in " + uri + " is named by both the variable key and the fragment");
	request.arrayName = query.count("variable") != 0 ? query["variable"] : parsed.fragment;
	if (query.count("order") != 0 && query["order"] != "C") throw std::invalid_argument("Order " + query["order"] + " of " + uri + " is not known");

	request.block.part = IO::partDecomposition(part);
	request.block.layout.order = IO::raw::arrayOrder::C;

	//The array is the domain of the variable
	const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
	for (int i = 0; i < R; ++i){
		const SAMS::dimension &varDim = varDef.getDimension(i);
		request.domainSizes[i] = static_cast<size_t>(h.axisRegistry.getDimension(varDim.axisName).getDomainCount(varDim.stagger));
	}

	const auto scheme = schemes.find(parsed.scheme);
	if (scheme == schemes.end()) throw std::invalid_argument("Scheme " + parsed.scheme + " of " + uri + " is not known");
	scheme->second(request);
	IO::setLocalVariablePart(h, varName, part);
}

} // namespace SAMS
#endif
