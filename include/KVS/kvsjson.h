#ifndef KVSJSON_H
#define KVSJSON_H

#include "kvs.h"
#include "json.hpp"
#include <sstream>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <fstream>

namespace SAMS
{

	namespace KVSConversion
	{

		namespace
		{

			nlohmann::json mapToJSON(std::shared_ptr<KVS::Map> map);
			nlohmann::json orderedToJSON(std::shared_ptr<KVS::Ordered> ordered);
			nlohmann::json arrayToJSON(std::shared_ptr<KVS::Array> array);
			nlohmann::json elementToJSON(const KVS::Element &element);

			KVS::Element arrayFromJSON(const nlohmann::json &j);
			KVS::Element mapFromJSON(const nlohmann::json &j);
			KVS::Element orderedFromJSON(const nlohmann::json &j);
			KVS::Element rangedFromJSON(const nlohmann::json &j);
			KVS::Element elementFromJSON(const nlohmann::json &j);

			/**
			 * Function to convert a Key-Value Store (KVS) element to a JSON representation.
			 * The function takes a KVS::Element as input and returns a nlohmann::json object that represents the element in JSON format.
			 * The conversion is based on the type of the KVS::Element, which can be
			 * an integer, float, string, array, map, or ordered collection.
			 * Arrays and maps are converted directly to JSON arrays and objects.
			 * Ordered and ranged types are converted to JSON objects with specific fields to represent their structure.
			 * @param element The KVS::Element to be converted to JSON.
			 * @return A nlohmann::json object representing the KVS::Element in JSON format.
			 * @throws std::runtime_error If the KVS::Element type is unsupported for JSON conversion.
			 */
			inline nlohmann::json elementToJSON(const KVS::Element &element)
			{

				nlohmann::json j;

				if (element.type == KVS::Type::Int32)
					j = std::get<int32_t>(element.value);
				else if (element.type == KVS::Type::Int64)
					j = std::get<int64_t>(element.value);
				else if (element.type == KVS::Type::Float)
					j = std::get<float>(element.value);
				else if (element.type == KVS::Type::Double)
					j = std::get<double>(element.value);
				else if (element.type == KVS::Type::String)
					j = std::get<std::string>(element.value);
				else if (element.type == KVS::Type::Bool)
					j = std::get<bool>(element.value);
				else if (element.type == KVS::Type::Int32Range)
				{
					const auto &range = std::get<KVS::ranged<int32_t>>(element.value);
					j = {{"Value", range.value}, {"LB", range.begin}, {"UB", range.end}};
				}
				else if (element.type == KVS::Type::Int64Range)
				{
					const auto &range = std::get<KVS::ranged<int64_t>>(element.value);
					j = {{"Value", range.value}, {"LB", range.begin}, {"UB", range.end}};
				}
				else if (element.type == KVS::Type::FloatRange)
				{
					const auto &range = std::get<KVS::ranged<float>>(element.value);
					j = {{"Value", range.value}, {"LB", range.begin}, {"UB", range.end}};
				}
				else if (element.type == KVS::Type::DoubleRange)
				{
					const auto &range = std::get<KVS::ranged<double>>(element.value);
					j = {{"Value", range.value}, {"LB", range.begin}, {"UB", range.end}};
				}
				else if (element.type == KVS::Type::Array)
				{
					// build an array value
					j = arrayToJSON(std::get<std::shared_ptr<KVS::Array>>(element.value));
				}
				else if (element.type == KVS::Type::Map)
				{
					j = mapToJSON(std::get<std::shared_ptr<KVS::Map>>(element.value));
				}
				else if (element.type == KVS::Type::Ordered)
				{
					j = orderedToJSON(std::get<std::shared_ptr<KVS::Ordered>>(element.value));
				}
				else if (element.type == KVS::Type::None)
					j = nullptr;
				else
					throw std::runtime_error("Unsupported element type for JSON conversion");

				return j;
			}

			/**
			 * Helper function to convert a KVS::Array to a JSON array.
			 * @param array A shared pointer to a KVS::Array that needs to be converted to JSON.
			 * @return A nlohmann::json object representing the KVS::Array as a JSON array.
			 * @throws std::runtime_error If any element in the KVS::Array is of an unsupported type for JSON conversion.
			 */
			inline nlohmann::json arrayToJSON(std::shared_ptr<KVS::Array> array)
			{
				nlohmann::json arr = nlohmann::json::array();
				for (const auto &element : *array)
				{
					nlohmann::json elemJson;
					elemJson = elementToJSON(element);
					arr.push_back(std::move(elemJson));
				}
				return arr;
			}

			/**
			 * Helper function to convert a KVS::Map to a JSON object.
			 * @param map A shared pointer to a KVS::Map that needs to be converted to JSON.
			 * @return A nlohmann::json object representing the KVS::Map as a JSON object.
			 * @throws std::runtime_error If any value in the KVS::Map is of an unsupported type for JSON conversion.
			 */
			inline nlohmann::json mapToJSON(std::shared_ptr<KVS::Map> map)
			{
				nlohmann::json obj = nlohmann::json::object();
				for (const auto &pair : *map)
				{
					// convert Element key to a string for JSON object key
					std::ostringstream os;
					os << pair.first;
					std::string key = os.str();

					nlohmann::json valueJson;
					valueJson = elementToJSON(pair.second);
					obj[key] = std::move(valueJson);
				}
				return obj;
			}

			/**
			 * Helper function to convert a KVS::Ordered collection to a JSON array of objects,
			 * where each object has "Name" and "Value" fields.
			 * @param ordered A shared pointer to a KVS::Ordered collection that needs to be converted to JSON.
			 * @return A nlohmann::json object representing the KVS::Ordered collection as a JSON array of objects.
			 * @throws std::runtime_error If any value in the KVS::Ordered collection is of an unsupported type for JSON conversion.
			 */
			inline nlohmann::json orderedToJSON(std::shared_ptr<KVS::Ordered> ordered)
			{
				nlohmann::json obj = nlohmann::json::array();
				for (const auto &pair : *ordered)
				{
					nlohmann::json subObject = nlohmann::json::object();
					nlohmann::json valueJson;
					subObject.push_back({pair.first, elementToJSON(pair.second)});
					obj.push_back(std::move(subObject));
				}
				return obj;
			}

			// From JSON section

			/**
			 * Convert a JSON value to a KVS:: Element
			 * Direct conversions are performed for JSON numbers (integers and floats) and strings.
			 * Calls to other functions check for array, objects, ranges and ordered collections.
			 * @param j The JSON value to be converted to a KVS::Element.
			 * @return A KVS::Element representing the JSON value.
			 * @throws std::runtime_error If the JSON value type is unsupported for conversion to KVS::Element,
			 *  or if the structure of the JSON value does not match the expected format for structured types
			 * (e.g., arrays, maps, ordered collections, ranges).
			 */
			inline KVS::Element elementFromJSON(const nlohmann::json &j)
			{
				if (j.is_boolean())
				{
					return KVS::Element(j.get<bool>());
				}
				else if (j.is_number_integer())
				{
					// Check if it fits in int32_t
					int64_t value = j.get<int64_t>();
					if (value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max())
						return KVS::Element(static_cast<int32_t>(value));
					else
						return KVS::Element(value);
				}
				else if (j.is_number_float())
				{
					/*
						 To avoid precision issues just stick with double
						 */
					double value = j.get<double>();
					return KVS::Element(value);
				}
				else if (j.is_string())
				{
					return KVS::Element(j.get<std::string>());
				}
				else if (j.is_array())
				{

					//Spin through all elements of the array. If they are all single element objects thiis is an ordered collection, otherwise it is a normal array
					bool isOrdered = true;
					for (const auto &elem : j)
					{
						if (!elem.is_object() || elem.size() != 1)
						{
							isOrdered = false;
							break;
						}
					}

					if (isOrdered)
						return orderedFromJSON(j);
					else
						return arrayFromJSON(j);
				}
				else if (j.is_object())
				{
					// If it has the three keys "Value", "LB" and "UB" and no other keys, we will interpret this as a ranged value instead of a map
					bool hasValue = false;
					bool hasLB = false;
					bool hasUB = false;
					bool hasOtherKeys = false;
					for (auto it = j.begin(); it != j.end(); ++it)
					{
						if (it.key() == "Value")
							hasValue = true;
						else if (it.key() == "LB")
							hasLB = true;
						else if (it.key() == "UB")
							hasUB = true;
						else
						{
							hasOtherKeys = true;
							break; // Has other keys is already enough to disqualify this as a ranged value, so we can break out of the loop early
						}
					}

					if (hasValue && hasLB && hasUB && !hasOtherKeys)
					{
						return rangedFromJSON(j);
					}
					else
					{
						return mapFromJSON(j);
					}
				}
				else
					throw std::runtime_error("Unsupported JSON type for conversion to Element");
			}

			/**
			 * Construct a KVS array from a JSON array
			 * @param j The JSON array to be converted to a KVS::Element containing a KVS::Array.
			 * @return A KVS::Element containing a KVS::Array.
			 */
			inline KVS::Element arrayFromJSON(const nlohmann::json &j)
			{
				if (!j.is_array())
					throw std::runtime_error("Expected JSON array for conversion to KVS::Array");

				KVS::Array array;
				for (const auto &elem : j)
				{
					array.push_back(elementFromJSON(elem));
				}
				return KVS::Element(array);
			}

			/**
			 * Construct a KVS map from a JSON object
			 * @param j The JSON object to be converted to a KVS::Element containing a KVS::Map.
			 * @return A KVS::Element containing a KVS::Map.
			 */
			inline KVS::Element mapFromJSON(const nlohmann::json &j)
			{
				if (!j.is_object())
					throw std::runtime_error("Expected JSON object for conversion to KVS::Map");

				KVS::Map map;
				for (auto it = j.begin(); it != j.end(); ++it)
				{
					// Convert JSON key to Element key (using string representation)
					std::string key(it.key());
					KVS::Element value = elementFromJSON(it.value());
					map.upsert(key) = value;
				}
				return KVS::Element(map);
			}

			/**
			 * Construct a KVS ranged value from a JSON object with "Value", "LB" and "UB" fields
			 * @param j The JSON object to be converted to a KVS::Element containing a KVS::ranged value.
			 * @return A KVS::Element containing a KVS::ranged value
			 */
			inline KVS::Element rangedFromJSON(const nlohmann::json &j)
			{
				if (!j.is_object() || !j.contains("Value") || !j.contains("LB") || !j.contains("UB"))
					throw std::runtime_error("Expected JSON object with 'Value', 'LB' and 'UB' for conversion to KVS::ranged");

				// We will try to infer the type of the range based on the types of the Value, LB and UB fields. They all have to be of the same type for this to work, otherwise we will throw an error
				auto valueElement = elementFromJSON(j["Value"]);
				auto lbElement = elementFromJSON(j["LB"]);
				auto ubElement = elementFromJSON(j["UB"]);

				if (valueElement.type != lbElement.type || valueElement.type != ubElement.type)
					throw std::runtime_error("The 'Value', 'LB' and 'UB' fields must be of the same type for conversion to KVS::ranged");

				switch (valueElement.type)
				{
					case KVS::Type::Int32:
						return KVS::Element(KVS::int32Range{valueElement.get<int32_t>(), lbElement.get<int32_t>(), ubElement.get<int32_t>()});
					case KVS::Type::Int64:
						return KVS::Element(KVS::int64Range{valueElement.get<int64_t>(), lbElement.get<int64_t>(), ubElement.get<int64_t>()});
					case KVS::Type::Float:
						return KVS::Element(KVS::floatRange{valueElement.get<float>(), lbElement.get<float>(), ubElement.get<float>()});
					case KVS::Type::Double:
						return KVS::Element(KVS::doubleRange{valueElement.get<double>(), lbElement.get<double>(), ubElement.get<double>()});
					default:
						throw std::runtime_error("Unsupported type for ranged value in JSON conversion");
				}
			}

			/**
			 * Construct a KVS ordered collection from a JSON array of objects with "Name" and "Value" fields
			 * @param j The JSON array to be converted to a KVS::Element containing a KVS::Ordered collection.
			 * @return A KVS::Element containing a KVS::Ordered collection.
			 */
			inline KVS::Element orderedFromJSON(const nlohmann::json &j)
			{
				if (!j.is_array())
					throw std::runtime_error("Expected JSON array for conversion to KVS::Ordered");

				KVS::Ordered ordered;
				for (const auto &elem : j)
				{
					if (!elem.is_object() || elem.size() != 1)
						throw std::runtime_error("Expected JSON object of a single key-value pair for conversion to KVS::Ordered");

					auto it = elem.begin();
					std::string key = it.key();
					KVS::Element value = elementFromJSON(it.value());
					ordered.upsert(key) = value;
				}
				return KVS::Element(ordered);
			}

		}
		/**
		 * Function to convert a Key-Value Store (KVS) to a JSON representation.
		 * The function takes a KVS object as input and returns a nlohmann::json object that represents the entire KVS in JSON format.
		 * @param kvs The KVS object to be converted to JSON.
		 */
		inline nlohmann::json toJSON(const KVS &kvs)
		{
			return elementToJSON(kvs.root);
		}

		/**
		 * Function to convert a Key-Value Store (KVS) to a JSON string representation.
		 * The function takes a KVS object as input and returns a std::string that represents the entire KVS in JSON format.
		 * @param kvs The KVS object to be converted to a JSON string.
		 * @return A std::string containing the JSON representation of the KVS.
		 */
		inline std::string toJSONString(const KVS &kvs, int indent = -1)
		{
			return toJSON(kvs).dump(indent);
		}

		inline void toJSONFile(const KVS &kvs, const std::string &filename, int indent = -1)
		{
			std::ofstream file(filename);
			if (!file.is_open())
				throw std::runtime_error("Could not open file for writing: " + filename);
			file << toJSON(kvs).dump(indent);
			file.close();
		}

		/**
		 * Convert a JSON representation of a KVS back to a KVS object.
		 * @param j The JSON object to be converted to a KVS object.
		 * @return A KVS object.
		 */
		inline KVS fromJSON(const nlohmann::json &j)
		{
			return KVS(elementFromJSON(j));
		}

		inline KVS fromJSONString(const std::string &jsonString)
		{
			nlohmann::json j = nlohmann::json::parse(jsonString);
			return fromJSON(j);
		}

		inline KVS fromJSONFile(const std::string &filename)
		{
			nlohmann::json j = nlohmann::json::parse(std::ifstream(filename));
			return fromJSON(j);
		}

	}

}
#endif // KVSJSON_H
