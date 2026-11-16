#ifndef KVSYAML_H
#define KVSYAML_H

#include "kvs.h"
#include <sstream>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <fstream>
#include <string>
#include <cstring>

/*#define RYML_SINGLE_HDR_DEFINE_NOW
#include "yaml.hpp"*/

    #include <ryml.hpp>
    // <ryml_std.hpp> is needed if interop with std containers is
    // desired; ryml itself does not use any STL container.
    // For this sample, we will be using std interop, so...
    #include <ryml_std.hpp> // optional header, provided for std:: interop
    #include <c4/format.hpp> // needed for the examples below
    // optional header, definitions for error utilities to implement
    // error callbacks:
    #include <c4/yml/error.def.hpp>
	

namespace SAMS
{

	namespace
	{
		void elementToYAML(ryml::NodeRef node, const KVS::Element &element);
		void arrayToYAML(ryml::NodeRef node, std::shared_ptr<KVS::Array> array);
		void mapToYAML(ryml::NodeRef node, std::shared_ptr<KVS::Map> map);
		void orderedToYAML(ryml::NodeRef node, std::shared_ptr<KVS::Ordered> ordered);

		KVS::Element elementFromYAML(ryml::ConstNodeRef node);
		KVS::Element arrayFromYAML(ryml::ConstNodeRef node);
		KVS::Element mapFromYAML(ryml::ConstNodeRef node);
		KVS::Element orderedFromYAML(ryml::ConstNodeRef node);
		KVS::Element rangedFromYAML(ryml::ConstNodeRef node);

		// Helper to convert a ryml::csubstr to std::string
		inline std::string to_std_string(ryml::csubstr s)
		{
			return std::string(s.str, s.len);
		}

		// Helper to set a scalar value on a node from a string
		inline void set_val(ryml::NodeRef node, const std::string &val)
		{
			// We need to copy the string into the tree's arena so it persists
			ryml::csubstr arena_str = node.tree()->to_arena(ryml::to_csubstr(val));
			node |= ryml::VAL;
			node.set_val(arena_str);
		}

		inline void set_key_val(ryml::NodeRef node, const std::string &key, const std::string &val)
		{
			ryml::csubstr arena_key = node.tree()->to_arena(ryml::to_csubstr(key));
			ryml::csubstr arena_val = node.tree()->to_arena(ryml::to_csubstr(val));
			node |= ryml::KEYVAL;
			node.set_key(arena_key);
			node.set_val(arena_val);
		}

		/**
		 * Function to convert a Key-Value Store (KVS) element into a YAML node using rapidyaml.
		 * The conversion is based on the type of the KVS::Element, which can be
		 * an integer, float, string, array, map, or ordered collection.
		 * Ordered and ranged types are converted to YAML structures with specific fields.
		 * @param node The ryml::NodeRef to populate with the element data.
		 * @param element The KVS::Element to be converted to YAML.
		 * @throws std::runtime_error If the KVS::Element type is unsupported for YAML conversion.
		 */
		inline void elementToYAML(ryml::NodeRef node, const KVS::Element &element)
		{
			if (element.type == KVS::Type::Int32)
			{
				std::string s = std::to_string(std::get<int32_t>(element.value));
				set_val(node, s);
			}
			else if (element.type == KVS::Type::Int64)
			{
				std::string s = std::to_string(std::get<int64_t>(element.value));
				set_val(node, s);
			}
			else if (element.type == KVS::Type::Float)
			{
				std::ostringstream os;
				os << std::get<float>(element.value);
				set_val(node, os.str());
			}
			else if (element.type == KVS::Type::Double)
			{
				std::ostringstream os;
				os << std::get<double>(element.value);
				set_val(node, os.str());
			}
			else if (element.type == KVS::Type::String)
			{
				set_val(node, std::get<std::string>(element.value));
			}
			else if (element.type == KVS::Type::Int32Range)
			{
				const auto &range = std::get<KVS::ranged<int32_t>>(element.value);
				node |= ryml::MAP;
				auto vn = node.append_child();
				set_key_val(vn, "Value", std::to_string(range.value));
				auto lb = node.append_child();
				set_key_val(lb, "LB", std::to_string(range.begin));
				auto ub = node.append_child();
				set_key_val(ub, "UB", std::to_string(range.end));
			}
			else if (element.type == KVS::Type::Int64Range)
			{
				const auto &range = std::get<KVS::ranged<int64_t>>(element.value);
				node |= ryml::MAP;
				auto vn = node.append_child();
				set_key_val(vn, "Value", std::to_string(range.value));
				auto lb = node.append_child();
				set_key_val(lb, "LB", std::to_string(range.begin));
				auto ub = node.append_child();
				set_key_val(ub, "UB", std::to_string(range.end));
			}
			else if (element.type == KVS::Type::FloatRange)
			{
				const auto &range = std::get<KVS::ranged<float>>(element.value);
				node |= ryml::MAP;
				std::ostringstream osv, oslb, osub;
				osv << range.value;
				oslb << range.begin;
				osub << range.end;
				auto vn = node.append_child();
				set_key_val(vn, "Value", osv.str());
				auto lb = node.append_child();
				set_key_val(lb, "LB", oslb.str());
				auto ub = node.append_child();
				set_key_val(ub, "UB", osub.str());
			}
			else if (element.type == KVS::Type::DoubleRange)
			{
				const auto &range = std::get<KVS::ranged<double>>(element.value);
				node |= ryml::MAP;
				std::ostringstream osv, oslb, osub;
				osv << range.value;
				oslb << range.begin;
				osub << range.end;
				auto vn = node.append_child();
				set_key_val(vn, "Value", osv.str());
				auto lb = node.append_child();
				set_key_val(lb, "LB", oslb.str());
				auto ub = node.append_child();
				set_key_val(ub, "UB", osub.str());
			}
			else if (element.type == KVS::Type::Array)
			{
				arrayToYAML(node, std::get<std::shared_ptr<KVS::Array>>(element.value));
			}
			else if (element.type == KVS::Type::Map)
			{
				mapToYAML(node, std::get<std::shared_ptr<KVS::Map>>(element.value));
			}
			else if (element.type == KVS::Type::Ordered)
			{
				orderedToYAML(node, std::get<std::shared_ptr<KVS::Ordered>>(element.value));
			}
			else if (element.type == KVS::Type::None)
			{
				node |= ryml::VAL;
				node.set_val("null");
			}
			else
				throw std::runtime_error("Unsupported element type for YAML conversion");
		}

		/**
		 * Helper function to convert a KVS::Array to a YAML sequence node.
		 * @param node The ryml::NodeRef to populate as a YAML sequence.
		 * @param array A shared pointer to a KVS::Array that needs to be converted to YAML.
		 */
		inline void arrayToYAML(ryml::NodeRef node, std::shared_ptr<KVS::Array> array)
		{
			node |= ryml::SEQ;
			for (const auto &element : *array)
			{
				auto child = node.append_child();
				elementToYAML(child, element);
			}
		}

		/**
		 * Helper function to convert a KVS::Map to a YAML mapping node.
		 * @param node The ryml::NodeRef to populate as a YAML mapping.
		 * @param map A shared pointer to a KVS::Map that needs to be converted to YAML.
		 */
		inline void mapToYAML(ryml::NodeRef node, std::shared_ptr<KVS::Map> map)
		{
			node |= ryml::MAP;
			for (const auto &pair : *map)
			{
				std::ostringstream os;
				os << pair.first;
				std::string key = os.str();

				auto child = node.append_child();
				ryml::csubstr arena_key = node.tree()->to_arena(ryml::to_csubstr(key));
				child.set_key(arena_key);
				elementToYAML(child, pair.second);
			}
		}

		/**
		 * Helper function to convert a KVS::Ordered collection to a YAML sequence of mappings,
		 * where each mapping has "Name" and "Value" fields.
		 * @param node The ryml::NodeRef to populate as a YAML sequence.
		 * @param ordered A shared pointer to a KVS::Ordered collection.
		 */
		inline void orderedToYAML(ryml::NodeRef node, std::shared_ptr<KVS::Ordered> ordered)
		{
			node |= ryml::SEQ;
			node.set_val_tag("!!omap"); // Use the omap tag to indicate ordered mapping
			for (const auto &pair : *ordered)
			{
				auto child = node.append_child();
				child |= ryml::MAP;

				auto kv = child.append_child();
				ryml::csubstr arena_key = node.tree()->to_arena(ryml::to_csubstr(pair.first));
				kv.set_key(arena_key);
				elementToYAML(kv, pair.second);
			}
		}

		// From YAML section

		// Helper: try to parse a string as int64, return true on success
		inline bool tryParseInt64(const std::string &s, int64_t &out)
		{
			if (s.empty()) return false;
			char *end = nullptr;
			errno = 0;
			long long val = std::strtoll(s.c_str(), &end, 10);
			if (errno != 0 || end != s.c_str() + s.size()) return false;
			out = static_cast<int64_t>(val);
			return true;
		}

		// Helper: try to parse a string as double, return true on success
		inline bool tryParseDouble(const std::string &s, double &out)
		{
			if (s.empty()) return false;
			char *end = nullptr;
			errno = 0;
			double val = std::strtod(s.c_str(), &end);
			if (errno != 0 || end != s.c_str() + s.size()) return false;
			out = val;
			return true;
		}

		/**
		 * Convert a YAML node to a KVS::Element.
		 * Direct conversions are performed for scalars (integers, floats, strings).
		 * Sequences and mappings are checked for array, ordered, map and ranged patterns.
		 * @param node The YAML node to convert.
		 * @return A KVS::Element representing the YAML node.
		 * @throws std::runtime_error If the YAML node type is unsupported.
		 */
		inline KVS::Element elementFromYAML(ryml::ConstNodeRef node)
		{
			if (node.is_seq())
			{
				// Check if this is an ordered collection (a sequence of single key-value pairs, preferably with the !!omap tag)
				bool isOrdered = node.has_val_tag() && node.val_tag() == "!!omap"; // If the sequence has the omap tag, treat as ordered

				//If the node has one child and that child is a map with exactly one key-value pair, we can also treat it as an ordered collection even if it doesn't have the omap tag
				isOrdered |= (node.num_children() > 0 && node[0].is_map() && node[0].num_children() == 1);

				if (isOrdered)
					return orderedFromYAML(node);
				else
					return arrayFromYAML(node);
			}
			else if (node.is_map())
			{
				// Check for ranged value pattern: exactly keys "Value", "LB", "UB"
				bool hasValue = false;
				bool hasLB = false;
				bool hasUB = false;
				bool hasOtherKeys = false;
				for (size_t i = 0; i < node.num_children(); ++i)
				{
					auto key = to_std_string(node[i].key());
					if (key == "Value")
						hasValue = true;
					else if (key == "LB")
						hasLB = true;
					else if (key == "UB")
						hasUB = true;
					else
					{
						hasOtherKeys = true;
						break;
					}
				}

				if (hasValue && hasLB && hasUB && !hasOtherKeys)
					return rangedFromYAML(node);
				else
					return mapFromYAML(node);
			}
			else if (node.has_val())
			{
				std::string val = to_std_string(node.val());

				// Try integer first
				int64_t ival;
				if (tryParseInt64(val, ival))
				{
					if (ival >= std::numeric_limits<int32_t>::min() && ival <= std::numeric_limits<int32_t>::max())
						return KVS::Element(static_cast<int32_t>(ival));
					else
						return KVS::Element(ival);
				}

				// Try floating point
				double dval;
				if (tryParseDouble(val, dval))
				{
					//To avoid numerical issues 
					return KVS::Element(dval);
				}

				// Otherwise treat as string
				return KVS::Element(val);
			}
			else
			{
				throw std::runtime_error("Unsupported YAML node type for conversion to Element");
			}
		}

		/**
		 * Construct a KVS array from a YAML sequence node.
		 * @param node The YAML sequence node.
		 * @return A KVS::Element containing a KVS::Array.
		 */
		inline KVS::Element arrayFromYAML(ryml::ConstNodeRef node)
		{
			if (!node.is_seq())
				throw std::runtime_error("Expected YAML sequence for conversion to KVS::Array");

			KVS::Array array;
			for (size_t i = 0; i < node.num_children(); ++i)
			{
				array.push_back(elementFromYAML(node[i]));
			}
			return KVS::Element(array);
		}

		/**
		 * Construct a KVS map from a YAML mapping node.
		 * @param node The YAML mapping node.
		 * @return A KVS::Element containing a KVS::Map.
		 */
		inline KVS::Element mapFromYAML(ryml::ConstNodeRef node)
		{
			if (!node.is_map())
				throw std::runtime_error("Expected YAML mapping for conversion to KVS::Map");

			KVS::Map map;
			for (size_t i = 0; i < node.num_children(); ++i)
			{
				auto child = node[i];
				std::string key = to_std_string(child.key());
				KVS::Element value = elementFromYAML(child);
				map.upsert(key) = value;
			}
			return KVS::Element(map);
		}

		/**
		 * Construct a KVS ranged value from a YAML mapping with "Value", "LB" and "UB" keys.
		 * @param node The YAML mapping node.
		 * @return A KVS::Element containing a KVS::ranged value.
		 */
		inline KVS::Element rangedFromYAML(ryml::ConstNodeRef node)
		{
			if (!node.is_map())
				throw std::runtime_error("Expected YAML mapping with 'Value', 'LB' and 'UB' for conversion to KVS::ranged");

			auto valueElement = elementFromYAML(node["Value"]);
			auto lbElement = elementFromYAML(node["LB"]);
			auto ubElement = elementFromYAML(node["UB"]);

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
					throw std::runtime_error("Unsupported type for ranged value in YAML conversion");
			}
		}

		/**
		 * Construct a KVS ordered collection from a YAML sequence of mappings with "Name" and "Value" keys.
		 * @param node The YAML sequence node.
		 * @return A KVS::Element containing a KVS::Ordered collection.
		 */
		inline KVS::Element orderedFromYAML(ryml::ConstNodeRef node)
		{
			if (!node.is_seq())
				throw std::runtime_error("Expected YAML sequence for conversion to KVS::Ordered");

			KVS::Ordered ordered;
			for (size_t i = 0; i < node.num_children(); ++i)
			{
				auto child = node[i];
				if (!child.is_map())
					throw std::runtime_error("Expected YAML mapping for conversion to KVS::Ordered");

				if (child.num_children() != 1)
					throw std::runtime_error("Expected exactly one key-value pair in each mapping for conversion to KVS::Ordered");

				std::string key = to_std_string(child[0].key());
				KVS::Element value = elementFromYAML(child[0]);
				ordered.upsert(key) = value;
			}
			return KVS::Element(ordered);
		}

	}

	namespace KVSConversion
	{
		/**
		 * Function to convert a Key-Value Store (KVS) to a YAML string representation.
		 * @param kvs The KVS object to be converted to YAML.
		 * @return A std::string containing the YAML representation of the KVS.
		 */
		inline std::string toYAMLString(const KVS &kvs)
		{
			ryml::Tree tree;
			ryml::NodeRef root = tree.rootref();
			elementToYAML(root, kvs.root);
			std::string output = ryml::emitrs_yaml<std::string>(tree);
			return output;
		}

		/**
		 * Convert a YAML string to a KVS object.
		 * @param yamlString The YAML string to be parsed and converted.
		 * @return A KVS object.
		 */
		inline KVS fromYAMLString(const std::string &yamlString)
		{
			ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yamlString));
			ryml::ConstNodeRef root = tree.rootref();
			return KVS(elementFromYAML(root));
		}

		/**
		 * Read a YAML file and convert it to a KVS object.
		 * @param filename The path to the YAML file.
		 * @return A KVS object.
		 */
		inline KVS fromYAMLFile(const std::string &filename)
		{
			std::ifstream ifs(filename);
			if (!ifs.is_open())
				throw std::runtime_error("Could not open file: " + filename);
			std::string content((std::istreambuf_iterator<char>(ifs)),
					std::istreambuf_iterator<char>());
			return fromYAMLString(content);
		}

		/**
		 * Write a KVS object to a YAML file.
		 * @param kvs The KVS object to write.
		 * @param filename The path to the output YAML file.
		 */
		inline void toYAMLFile(const KVS &kvs, const std::string &filename)
		{
			std::string yaml = toYAMLString(kvs);
			std::ofstream ofs(filename);
			if (!ofs.is_open())
				throw std::runtime_error("Could not open file for writing: " + filename);
			ofs << yaml;
		}
	}
}
#endif // KVSYAML_H
