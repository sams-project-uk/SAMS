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
#ifndef IO_TYPES_H
#define IO_TYPES_H

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "harness.h"

namespace SAMS::IO
{

/**
 * @brief The part of a variable that this rank holds, in a host array, and where it is in the whole variable
 * @tparam T The type of the variable
 * @tparam rank The rank of the variable
 */
template<typename T, int rank>
struct localVariablePart{
	//The array that holds the data
	portableWrapper::portableArray<T, rank, portableWrapper::arrayTags::host> array;
	//The first and last index of the part of the array that is written, in the array's own indices
	std::array<SIGNED_INDEX_TYPE, rank> sourceLower{}, sourceUpper{};
	//The position in the whole variable of the first element that is written
	std::array<size_t, rank> globalStarts{};
	//Whether the whole variable is split across each MPI decomposition direction
	std::array<bool, SAMS::MPI_DECOMPOSITION_RANK> splitDirections{};
	//Owns the array if it is a copy, so that the copy is released with the part
	portableWrapper::portableArrayManager manager;
};

/**
 * @brief Get the part of a variable in the variable registry that this rank holds, in a contiguous host array
 * @tparam T The type of the variable
 * @tparam rank The rank of the variable
 * @param h The harness
 * @param varName The name of the variable in the variable registry
 * @throws std::invalid_argument if the type or rank of the variable does not match
 */
template<typename T, int rank>
localVariablePart<T, rank> getLocalVariablePart(SAMS::harness &h, const std::string &varName){
	const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
	if (varDef.getType() != SAMS::gettypeRegistry().getTypeID<T>()) throw std::invalid_argument("Type of variable " + varName + " does not match");
	if (varDef.getRank() != rank) throw std::invalid_argument("Rank of variable " + varName + " does not match");
	localVariablePart<T, rank> part;
	//The variable is only copied if it is not already contiguous on the host
	if (varDef.getMemorySpace() == portableWrapper::arrayTags::host) {
		portableWrapper::portableArray<T, rank, portableWrapper::arrayTags::host> array;
		varDef.fillPPArray(array);
		part.array = part.manager.makeHostAvailableContiguous(array);
	} else {
		portableWrapper::portableArray<T, rank, portableWrapper::arrayTags::accelerated> array;
		varDef.fillPPArray(array);
		part.array = part.manager.makeHostAvailableContiguous(array);
	}
	for (int i = 0; i < rank; ++i){
		const SAMS::dimension &varDim = varDef.getDimension(i);
		const SAMS::dimension &dim = h.axisRegistry.getDimension(varDim.axisName);
		const SAMS::staggerType stagger = varDim.stagger;
		part.sourceLower[i] = dim.getLocalDomainLB(stagger);
		part.sourceUpper[i] = dim.getLocalDomainUB(stagger);
		part.globalStarts[i] = static_cast<size_t>(dim.getGlobalDomainLB(stagger) - dim.getDomainLB(stagger));
		const int mpiAxis = static_cast<int>(dim.mpiAxis);
		if (mpiAxis >= 0) part.splitDirections[mpiAxis] = true;
	}
	return part;
}

/**
 * @brief Get the part of a variable in the variable registry that this rank holds, including the variable's ghost cells on the edges of the global domain, in a contiguous host array
 * @details The whole variable starts at the first lower ghost cell of the global domain
 * @tparam T The type of the variable
 * @tparam rank The rank of the variable
 * @param h The harness
 * @param varName The name of the variable in the variable registry
 * @throws std::invalid_argument if the type or rank of the variable does not match
 */
template<typename T, int rank>
localVariablePart<T, rank> getLocalVariablePartWithGhosts(SAMS::harness &h, const std::string &varName){
	localVariablePart<T, rank> part = getLocalVariablePart<T, rank>(h, varName);
	const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
	for (int i = 0; i < rank; ++i){
		const SAMS::dimension &varDim = varDef.getDimension(i);
		const SAMS::dimension &dim = h.axisRegistry.getDimension(varDim.axisName);
		const SAMS::staggerType stagger = varDim.stagger;
		const SIGNED_INDEX_TYPE lowerGhosts = static_cast<SIGNED_INDEX_TYPE>(varDim.lowerGhosts);
		const SIGNED_INDEX_TYPE upperGhosts = static_cast<SIGNED_INDEX_TYPE>(varDim.upperGhosts);
		//Ranks on an edge of the global domain hold the ghost cells on that edge
		if (dim.getGlobalDomainLB(stagger) == dim.getDomainLB(stagger)) {
			part.sourceLower[i] -= lowerGhosts;
		} else {
			part.globalStarts[i] += static_cast<size_t>(lowerGhosts);
		}
		if (dim.getGlobalDomainUB(stagger) == dim.getDomainUB(stagger)) part.sourceUpper[i] += upperGhosts;
	}
	return part;
}

/**
 * @brief Copy a part of a variable back into the variable in the variable registry, in the variable's own memory
 * @details The whole local array is copied, so the elements outside the part keep the values that they had when the part was got
 * @tparam T The type of the variable
 * @tparam rank The rank of the variable
 * @param h The harness
 * @param varName The name of the variable in the variable registry
 * @param part The part of the variable, which was got from the same variable
 */
template<typename T, int rank>
void setLocalVariablePart(SAMS::harness &h, const std::string &varName, localVariablePart<T, rank> &part){
	const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
	//The copy does nothing if the part is the variable's own array
	if (varDef.getMemorySpace() == portableWrapper::arrayTags::host) {
		portableWrapper::portableArray<T, rank, portableWrapper::arrayTags::host> array;
		varDef.fillPPArray(array);
		part.manager.copyData(array, part.array);
	} else {
		portableWrapper::portableArray<T, rank, portableWrapper::arrayTags::accelerated> array;
		varDef.fillPPArray(array);
		part.manager.copyData(array, part.array);
	}
}

/**
 * @brief Name and staggering of an axis in the axis registry
 */
struct simulationAxis{
	std::string name;
	SAMS::staggerType stagger = SAMS::staggerType::CENTRED;
};

/**
 * @brief Writes meshes and variables straight from the axis and variable registries through a writer
 * @tparam T_writer The type of the writer
 */
template<typename T_writer>
class registryWriter{
	//The axes of a mesh
	struct meshAxes{
		std::array<simulationAxis, SAMS::MAX_RANK> axes{};
		size_t rank = 0;
	};

	SAMS::harness &h;
	T_writer &writer;
	//The axes of each mesh, to check the variables against
	std::unordered_map<std::string, meshAxes> meshes;
	//The names in the file of the axes that have been registered, so that each is only registered once
	std::set<std::string> registeredAxes;
	//Write each axis with the type that it was registered with, and each mesh
	std::vector<std::function<void()>> meshWriters;
	//Write each variable with the type and rank that it was registered with
	std::vector<std::function<void()>> variableWriters;

	/**
	 * @brief The name in the file of a staggering of an axis in the axis registry. It is Axes/c/<axis> for the cell centres and Axes/b/<axis> for the cell edges
	 * @param axis The name and staggering of the axis
	 */
	std::string axisItemName(const simulationAxis &axis){
		return std::string("Axes/") + (axis.stagger == SAMS::staggerType::CENTRED ? "c" : "b") + "/" + axis.name;
	}

	/**
	 * @brief Register the whole of a staggering of an axis in the axis registry, including its ghost cells
	 * @tparam T_axis The type of the axis
	 * @param itemName The name of the axis in the file
	 * @param axis The name and staggering of the axis
	 */
	template<typename T_axis>
	void registerAxisTyped(const std::string &itemName, const simulationAxis &axis){
		//The core indices of the axis are the bounds of its array
		portableWrapper::hostArray<T_axis, 1> axisArray;
		h.axisRegistry.fillPPAxis(axis.name, axisArray, axis.stagger);
		const SIGNED_INDEX_TYPE start = axisArray.getLowerBound(0);
		const SAMS::dimension &dim = h.axisRegistry.getDimension(axis.name);
		writer.template registerAxis<T_axis>(itemName.c_str(), start, static_cast<size_t>(axisArray.getUpperBound(0) - start + 1), dim.getLowerGhosts(axis.stagger), dim.getUpperGhosts(axis.stagger));
		//Every rank holds the whole of the axis
		meshWriters.push_back([this, itemName, axis](){
			portableWrapper::hostArray<T_axis, 1> axisValues;
			h.axisRegistry.fillPPAxis(axis.name, axisValues, axis.stagger);
			writer.writeAxis(itemName.c_str(), &axisValues(axisValues.getLowerBound(0)));
		});
	}

	/**
	 * @brief Register a staggering of an axis in the axis registry with the type that it has in the axis registry, unless it is already registered
	 * @param axis The name and staggering of the axis
	 * @return The name of the axis in the file
	 * @throws std::invalid_argument if the axis has a type that is not built in to the type registry
	 */
	std::string registerAxis(const simulationAxis &axis){
		const std::string itemName = axisItemName(axis);
		if (registeredAxes.count(itemName) != 0) return itemName;
		registeredAxes.insert(itemName);
		switch (static_cast<SAMS::typeID>(h.axisRegistry.getAxisType(axis.name, axis.stagger).ID)){
			case SAMS::typeID::TYPE_DOUBLE: registerAxisTyped<double>(itemName, axis); break;
			case SAMS::typeID::TYPE_FLOAT: registerAxisTyped<float>(itemName, axis); break;
			case SAMS::typeID::TYPE_INT32: registerAxisTyped<int32_t>(itemName, axis); break;
			case SAMS::typeID::TYPE_INT64: registerAxisTyped<int64_t>(itemName, axis); break;
			case SAMS::typeID::TYPE_UINT32: registerAxisTyped<uint32_t>(itemName, axis); break;
			case SAMS::typeID::TYPE_UINT64: registerAxisTyped<uint64_t>(itemName, axis); break;
			case SAMS::typeID::TYPE_INT8: registerAxisTyped<int8_t>(itemName, axis); break;
			case SAMS::typeID::TYPE_INT16: registerAxisTyped<int16_t>(itemName, axis); break;
			case SAMS::typeID::TYPE_UINT8: registerAxisTyped<uint8_t>(itemName, axis); break;
			case SAMS::typeID::TYPE_UINT16: registerAxisTyped<uint16_t>(itemName, axis); break;
			default: throw std::invalid_argument("Axis " + axis.name + " has a type that is not built in to the type registry, so it cannot be written");
		}
		return itemName;
	}

	/**
	 * @brief Register a blob of the whole of an axis in the axis registry, including its ghost cells
	 * @tparam T_axis The type of the axis
	 * @param blobName The name of the blob
	 * @param axisName The name of the axis in the axis registry
	 * @param stagger The staggering of the axis that is written
	 */
	template<typename T_axis>
	void registerAxisBlob(const std::string &blobName, const std::string &axisName, SAMS::staggerType stagger){
		const SAMS::dimension &dim = h.axisRegistry.getDimension(axisName);
		writer.template registerBlob<T_axis>(blobName.c_str(), std::array<size_t, 1>{static_cast<size_t>(dim.getCount(stagger))});
		//Every rank holds the whole of the axis, so the part is the whole axis and it is not split across any MPI direction
		variableWriters.push_back([this, blobName, axisName, stagger](){
			const SAMS::dimension &axisDim = h.axisRegistry.getDimension(axisName);
			localVariablePart<T_axis, 1> part;
			h.axisRegistry.fillPPAxis(axisName, part.array, stagger);
			part.sourceLower[0] = axisDim.getLB(stagger);
			part.sourceUpper[0] = axisDim.getUB(stagger);
			writer.writeBlob(blobName.c_str(), part);
		});
	}

	public:
	/**
	 * @brief Constructor
	 * @param h The harness
	 * @param writer The writer that the meshes and variables are written through
	 */
	registryWriter(SAMS::harness &h, T_writer &writer) : h(h), writer(writer) {}

	/**
	 * @brief Register a rectilinear mesh of the domains of axes in the axis registry. The axes that it uses are registered with it
	 * @param name The name of the mesh
	 * @param axes The name and staggering of the axis in each dimension
	 * @throws std::invalid_argument if a mesh with the same name is already registered with different axes
	 */
	template<size_t R>
	void registerRectilinearMesh(const char* name, const simulationAxis (&axes)[R]){
		//A mesh is only registered once. Registering it again must give the same axes
		if (meshes.count(name) != 0){
			const meshAxes &existing = meshes[name];
			bool same = existing.rank == R;
			for (size_t i = 0; i < R && same; ++i){
				same = existing.axes[i].name == axes[i].name && existing.axes[i].stagger == axes[i].stagger;
			}
			if (!same) throw std::invalid_argument(std::string("Mesh ") + name + " is already registered with different axes");
			return;
		}
		std::array<std::string, R> axisNames;
		std::array<SIGNED_INDEX_TYPE, R> starts;
		std::array<size_t, R> sizes;
		meshAxes &mesh = meshes[name];
		for (size_t i = 0; i < R; ++i){
			const SAMS::dimension &dim = h.axisRegistry.getDimension(axes[i].name);
			axisNames[i] = registerAxis(axes[i]);
			//The plotted area is the domain
			starts[i] = dim.getDomainLB(axes[i].stagger);
			sizes[i] = dim.getDomainCount(axes[i].stagger);
			mesh.axes[i] = axes[i];
		}
		mesh.rank = R;
		writer.registerRectilinearMesh(name, axisNames, starts, sizes);
		meshWriters.push_back([this, meshName = std::string(name)](){ writer.writeRectilinearMesh(meshName.c_str()); });
	}

	/**
	 * @brief Register data from a variable in the variable registry against a mesh registered from the axis registry
	 * @tparam T_data The type of the variable
	 * @tparam R The rank of the variable
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param varName The name of the variable in the variable registry
	 * @throws std::invalid_argument if the variable does not lie on the axes of the mesh
	 */
	template<typename T_data, int R>
	void registerData(const char* name, const char* meshName, const char* varName){
		//The variable must lie on the axes of the mesh with the same staggering
		const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
		const meshAxes &mesh = meshes[meshName];
		if (varDef.getRank() != R || mesh.rank != static_cast<size_t>(R)) throw std::invalid_argument("Rank of variable does not match rank of mesh");
		for (int i = 0; i < R; ++i){
			const SAMS::dimension &dim = varDef.getDimension(i);
			if (dim.axisName != mesh.axes[i].name || dim.stagger != mesh.axes[i].stagger) throw std::invalid_argument("Axes of variable do not match axes of mesh");
		}
		writer.template registerData<T_data>(name, meshName);
		variableWriters.push_back([this, dataName = std::string(name), variable = std::string(varName)](){ writer.writeData(dataName.c_str(), getLocalVariablePart<T_data, R>(h, variable)); });
	}

	/**
	 * @brief Register a blob from a variable in the variable registry, including the variable's ghost cells on the edges of the global domain
	 * @details The blob is written with the variables
	 * @tparam T_data The type of the variable
	 * @tparam R The rank of the variable
	 * @param name The name of the blob
	 * @param varName The name of the variable in the variable registry
	 * @throws std::invalid_argument if the rank of the variable does not match
	 */
	template<typename T_data, int R>
	void registerBlob(const char* name, const char* varName){
		const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
		if (varDef.getRank() != R) throw std::invalid_argument("Rank of variable does not match rank of blob");
		std::array<size_t, R> sizes;
		for (int i = 0; i < R; ++i){
			const SAMS::dimension &varDim = varDef.getDimension(i);
			const SAMS::dimension &dim = h.axisRegistry.getDimension(varDim.axisName);
			sizes[i] = static_cast<size_t>(dim.getDomainCount(varDim.stagger) + varDim.lowerGhosts + varDim.upperGhosts);
		}
		writer.template registerBlob<T_data>(name, sizes);
		variableWriters.push_back([this, blobName = std::string(name), variable = std::string(varName)](){ writer.writeBlob(blobName.c_str(), getLocalVariablePartWithGhosts<T_data, R>(h, variable)); });
	}

	/**
	 * @brief Register data from a variable in the variable registry including the variable's ghost cells on the edges of the global domain, against a mesh registered from the axis registry
	 * @details The data is larger than its mesh. It records the core index of its first element from the variable's dimensions
	 * @tparam T_data The type of the variable
	 * @tparam R The rank of the variable
	 * @param name The name of the data
	 * @param meshName The name of the mesh that the data is associated with
	 * @param varName The name of the variable in the variable registry
	 */
	template<typename T_data, int R>
	void registerDataWithGhosts(const char* name, const char* meshName, const char* varName){
		const SAMS::variableDef &varDef = h.variableRegistry.getVariable(varName);
		std::array<SIGNED_INDEX_TYPE, R> starts;
		std::array<size_t, R> sizes;
		for (int i = 0; i < R; ++i){
			const SAMS::dimension &varDim = varDef.getDimension(i);
			const SAMS::dimension &dim = h.axisRegistry.getDimension(varDim.axisName);
			starts[i] = varDim.getLB(varDim.stagger);
			sizes[i] = dim.getDomainCount(varDim.stagger) + varDim.lowerGhosts + varDim.upperGhosts;
		}
		writer.template registerData<T_data, R>(name, meshName, starts, sizes);
		variableWriters.push_back([this, dataName = std::string(name), variable = std::string(varName)](){ writer.writeData(dataName.c_str(), getLocalVariablePartWithGhosts<T_data, R>(h, variable)); });
	}

	/**
	 * @brief Register every axis in the axis registry. A logical axis only has cell centres, and every other axis has edges, from which
	 * its other staggerings are derived, so the centres of a logical axis and the edges of any other axis are registered
	 * @throws std::invalid_argument if an axis has a type that is not built in to the type registry
	 */
	void registerAllAxes(){
		for (const auto &entry : h.axisRegistry.getAxisMap()){
			registerAxis({entry.first, entry.second.logicalAxis ? SAMS::staggerType::CENTRED : SAMS::staggerType::HALF_CELL});
		}
	}

	/**
	 * @brief Register a blob of the whole of every axis in the axis registry, including its ghost cells, named SAMS/axes/<axis>
	 * @details The blobs are written with the variables. A logical axis only has cell centres, and every other axis has edges, from which
	 * its other staggerings are derived, so only the centres of a logical axis and the edges of any other axis are written
	 * @throws std::invalid_argument if an axis has a type that is not built in to the type registry
	 */
	void registerAxisBlobs(){
		for (const auto &entry : h.axisRegistry.getAxisMap()){
			const std::string &axisName = entry.first;
			const SAMS::staggerType stagger = entry.second.logicalAxis ? SAMS::staggerType::CENTRED : SAMS::staggerType::HALF_CELL;
			const std::string blobName = "SAMS/axes/" + axisName;
			switch (static_cast<SAMS::typeID>(h.axisRegistry.getAxisType(axisName, stagger).ID)){
				case SAMS::typeID::TYPE_DOUBLE: registerAxisBlob<double>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_FLOAT: registerAxisBlob<float>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_INT32: registerAxisBlob<int32_t>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_INT64: registerAxisBlob<int64_t>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_UINT32: registerAxisBlob<uint32_t>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_UINT64: registerAxisBlob<uint64_t>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_COMPLEX_DOUBLE: registerAxisBlob<std::complex<double>>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_COMPLEX_FLOAT: registerAxisBlob<std::complex<float>>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_INT8: registerAxisBlob<int8_t>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_INT16: registerAxisBlob<int16_t>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_UINT8: registerAxisBlob<uint8_t>(blobName, axisName, stagger); break;
				case SAMS::typeID::TYPE_UINT16: registerAxisBlob<uint16_t>(blobName, axisName, stagger); break;
				default: throw std::invalid_argument("Axis " + axisName + " has a type that is not built in to the type registry, so it cannot be written");
			}
		}
	}

	/**
	 * @brief Write every registered mesh
	 */
	void writeMeshes(){ for (auto &write : meshWriters) write(); }

	/**
	 * @brief Write every registered variable
	 */
	void writeVariables(){ for (auto &write : variableWriters) write(); }
};

/**
 * @brief Reads variables in the variable registry straight from a reader
 * @details Reading needs no registration, so each item is read when it is asked for
 * @tparam T_reader The type of the reader
 */
template<typename T_reader>
class registryReader{
	SAMS::harness &h;
	T_reader &reader;

	public:
	/**
	 * @brief Constructor
	 * @param h The harness
	 * @param reader The reader that the variables are read through
	 */
	registryReader(SAMS::harness &h, T_reader &reader) : h(h), reader(reader) {}

	/**
	 * @brief Read a data item into the part of a variable in the variable registry that this rank holds, without its ghost cells
	 * @tparam T_data The type of the variable
	 * @tparam R The rank of the variable
	 * @param name The name of the data
	 * @param varName The name of the variable in the variable registry
	 */
	template<typename T_data, int R>
	void readData(const char* name, const char* varName){
		localVariablePart<T_data, R> part = getLocalVariablePart<T_data, R>(h, varName);
		reader.readData(name, part);
		setLocalVariablePart(h, varName, part);
	}

	/**
	 * @brief Read a data item into the part of a variable in the variable registry that this rank holds, including the variable's ghost cells on the edges of the global domain
	 * @tparam T_data The type of the variable
	 * @tparam R The rank of the variable
	 * @param name The name of the data
	 * @param varName The name of the variable in the variable registry
	 */
	template<typename T_data, int R>
	void readDataWithGhosts(const char* name, const char* varName){
		localVariablePart<T_data, R> part = getLocalVariablePartWithGhosts<T_data, R>(h, varName);
		reader.readData(name, part);
		setLocalVariablePart(h, varName, part);
	}

	/**
	 * @brief Read a blob into the part of a variable in the variable registry that this rank holds, including the variable's ghost cells on the edges of the global domain
	 * @tparam T_data The type of the variable
	 * @tparam R The rank of the variable
	 * @param name The name of the blob
	 * @param varName The name of the variable in the variable registry
	 */
	template<typename T_data, int R>
	void readBlob(const char* name, const char* varName){
		localVariablePart<T_data, R> part = getLocalVariablePartWithGhosts<T_data, R>(h, varName);
		reader.readBlob(name, part);
		setLocalVariablePart(h, varName, part);
	}
};
} // namespace SAMS::IO

#endif
