#ifndef DECKUTILS_H
#define DECKUTILS_H
#include "harness.h"
#include "KVS/kvs.h"
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace SAMS
{
	namespace pw = portableWrapper;

	/**
	 * The name of a deck condition or value without its argument, such as file for file(uri)
	 * @param text The deck text
	 */
	inline std::string conditionType(const std::string &text)
	{
		const std::string type = text.substr(0, text.find('('));
		const std::string::size_type first = type.find_first_not_of(" \t");
		if (first == std::string::npos) return "";
		return type.substr(first, type.find_last_not_of(" \t") - first + 1);
	}

	/**
	 * The names of the directions in deck keys
	 */
	inline constexpr const char *directionNames[3] = {"x", "y", "z"};

	/**
	 * Deck text for one edge, from the block's edge key, block's direction key, the block itself if it is a value, top level edge key or top level direction key. Empty if none
	 * @param kvs The key value store
	 * @param root The path of the boundaries block, such as /boundaries
	 * @param block The name of the block in the boundaries block
	 * @param dimension The dimension normal to the edge
	 * @param suffix The suffix of the edge key, _min or _max
	 * @param path Set to the key used
	 */
	inline std::string edgeText(KVS &kvs, const std::string &root, const std::string &block, int dimension, const std::string &suffix, std::string &path)
	{
		const std::string direction = directionNames[dimension];
		for (const std::string &candidate : {root + "/" + block + "/" + direction + suffix, root + "/" + block + "/" + direction, root + "/" + block,
				root + "/" + direction + suffix, root + "/" + direction}){
			KVS::Element element = kvs.getPath(candidate, KVS::Element());
			//A block of keys is not a condition
			if (element.isType(KVS::Type::None) || element.isType(KVS::Type::Map) || element.isType(KVS::Type::Ordered)) continue;
			path = candidate;
			return element.asString();
		}
		return "";
	}

	/**
	 * Host copies of variables in the registry, by name, for setting the variables on the host. Variables on the device
	 * are copied back by copyToDevice
	 */
	template <T_smallIndexType rank>
	class deckHostVariables
	{
		struct namedArray
		{
			std::string name;
			//True if the host array is a copy of the device array
			bool copies = false;
			pw::acceleratedArray<T_dataType, rank> device;
			pw::hostArray<T_dataType, rank> array;
		};
		std::vector<namedArray> variables;

		namedArray &find(std::string_view name)
		{
			return *std::find_if(variables.begin(), variables.end(), [name](const namedArray &v){ return v.name == name; });
		}

	public:
		/**
		 * Add a variable. It is made available on the host
		 * @param harnessRef The harness, which holds the variable
		 * @param name The name of the variable in the registry
		 * @param copyBack If true the host array is copied from the device, and if false it is set to zero
		 */
		void addVariable(harness &harnessRef, const std::string &name, bool copyBack = false)
		{
			const variableDef &varDef = harnessRef.variableRegistry.getVariable(name);
			auto &manager = harnessRef.memoryRegistry.getArrayManager();
			namedArray &v = variables.emplace_back();
			v.name = name;
			if (varDef.getMemorySpace() == pw::arrayTags::host) {
				varDef.fillPPArray(v.array);
			} else {
				varDef.fillPPArray(v.device);
				if (copyBack) v.array = manager.makeHostAvailable(v.device);
				else manager.mold(v.array, v.device);
				v.copies = v.array.data() != v.device.data();
			}
			if (!copyBack) pw::assign(v.array, 0.0);
		}

		/**
		 * Copy the host array of a variable off the device
		 * @param harnessRef The harness, which holds the array manager
		 * @param name The name of the variable in the registry
		 */
		void copyToHost(harness &harnessRef, std::string_view name)
		{
			namedArray &v = find(name);
			if (v.copies) harnessRef.memoryRegistry.getArrayManager().copyData(v.array, v.device);
		}

		/**
		 * Copy the host arrays of the variables back onto the device
		 * @param harnessRef The harness, which holds the array manager
		 */
		void copyToDevice(harness &harnessRef)
		{
			auto &manager = harnessRef.memoryRegistry.getArrayManager();
			for (namedArray &v : variables){
				if (v.copies) manager.copyData(v.device, v.array);
			}
		}

		/**
		 * Release the host copies of the variables and remove the variables
		 * @param harnessRef The harness, which holds the array manager
		 */
		void clearVariables(harness &harnessRef)
		{
			auto &manager = harnessRef.memoryRegistry.getArrayManager();
			for (namedArray &v : variables){
				if (v.copies) manager.deallocate(v.array);
			}
			variables.clear();
		}

		/**
		 * The host array of a variable
		 * @param name The name of the variable in the registry
		 */
		pw::hostArray<T_dataType, rank> &variable(std::string_view name)
		{
			return find(name).array;
		}

		/**
		 * The host array of a variable
		 * @param name The name of the variable in the registry
		 */
		const pw::hostArray<T_dataType, rank> &variable(std::string_view name) const
		{
			return std::find_if(variables.begin(), variables.end(), [name](const namedArray &v){ return v.name == name; })->array;
		}
	};
}

#endif
