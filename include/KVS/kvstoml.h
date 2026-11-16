#ifndef KVSTOML_H
#define KVSTOML_H

#include "kvs.h"
#include "toml.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace SAMS{

	namespace KVSConversion
	{
		namespace
		{
			using toml_node_ptr = std::unique_ptr<toml::node>;

			// Compatibility helpers for toml++ node construction
			template <typename T>
				inline toml_node_ptr make_toml_node_scalar(const T &v)
				{
					return std::make_unique<toml::value<T>>(v);
				}

			inline toml_node_ptr make_toml_node_table()
			{
				return std::make_unique<toml::table>();
			}

			inline toml_node_ptr make_toml_node_array()
			{
				return std::make_unique<toml::array>();
			}

			inline void push_toml_node(toml::array &arr, toml_node_ptr node)
			{
				if (!node)
					throw std::runtime_error("Null TOML node cannot be added to array");
				switch (node->type())
				{
					case toml::node_type::integer:
						arr.push_back(node->as_integer()->get());
						break;
					case toml::node_type::floating_point:
						arr.push_back(node->as_floating_point()->get());
						break;
					case toml::node_type::string:
						arr.push_back(std::string{node->as_string()->get()});
						break;
					case toml::node_type::table:
						arr.push_back(std::move(*node->as_table()));
						break;
					case toml::node_type::array:
						arr.push_back(std::move(*node->as_array()));
						break;
					default:
						throw std::runtime_error("Unsupported TOML node type for array insertion");
				}
			}

			inline void insert_toml_node(toml::table &tbl, const std::string &key, toml_node_ptr node)
			{
				if (!node)
					throw std::runtime_error("Null TOML node cannot be inserted into table");
				switch (node->type())
				{
					case toml::node_type::integer:
						tbl.insert(key, node->as_integer()->get());
						break;
					case toml::node_type::floating_point:
						tbl.insert(key, node->as_floating_point()->get());
						break;
					case toml::node_type::string:
						tbl.insert(key, std::string{node->as_string()->get()});
						break;
					case toml::node_type::table:
						tbl.insert(key, std::move(*node->as_table()));
						break;
					case toml::node_type::array:
						tbl.insert(key, std::move(*node->as_array()));
						break;
					default:
						throw std::runtime_error("Unsupported TOML node type for table insertion");
				}
			}

			inline KVS::Element scalarFromTOML(const toml::node &node)
			{
				if (const auto *i = node.as_integer())
				{
					const int64_t v = i->get();
					return KVS::Element(v);
				}

				if (const auto *f = node.as_floating_point())
				{
					const double v = f->get();
					return KVS::Element(v);
				}

				if (const auto *s = node.as_string())
					return KVS::Element(std::string{s->get()});

				throw std::runtime_error("Unsupported TOML scalar type for KVS::Element");
			}

			inline toml_node_ptr scalarToTOML(const KVS::Element &element)
			{
				switch (element.type)
				{
					case KVS::Type::Int32:
						return make_toml_node_scalar(static_cast<int64_t>(std::get<int32_t>(element.value)));
					case KVS::Type::Int64:
						return make_toml_node_scalar(std::get<int64_t>(element.value));
					case KVS::Type::Float:
						return make_toml_node_scalar(static_cast<double>(std::get<float>(element.value)));
					case KVS::Type::Double:
						return make_toml_node_scalar(std::get<double>(element.value));
					case KVS::Type::String:
						return make_toml_node_scalar(std::get<std::string>(element.value));
					case KVS::Type::None:
						return make_toml_node_scalar(std::string{}); // Represent None as an empty string in TOML
					default:
						break;
				}
				return nullptr;
			}

			toml_node_ptr mapToTOML(std::shared_ptr<KVS::Map> map);
			toml_node_ptr orderedToTOML(std::shared_ptr<KVS::Ordered> ordered);
			toml_node_ptr arrayToTOML(std::shared_ptr<KVS::Array> array);
			toml_node_ptr elementToTOML(const KVS::Element &element);

			KVS::Element arrayFromTOML(const toml::array &arr);
			KVS::Element mapFromTOML(const toml::table &tbl);
			KVS::Element orderedFromTOML(const toml::array &arr);
			KVS::Element rangedFromTOML(const toml::table &tbl);
			KVS::Element elementFromTOML(const toml::node &node);

			inline std::string keyToString(const KVS::Element &key)
			{
				std::ostringstream os;
				os << key;
				return os.str();
			}

			inline toml_node_ptr elementToTOML(const KVS::Element &element)
			{
				if (auto s = scalarToTOML(element))
					return s;

				if (element.type == KVS::Type::Int32Range)
				{
					const auto &r = std::get<KVS::ranged<int32_t>>(element.value);
					auto t = make_toml_node_table();
					auto &tbl = *t->as_table();
					tbl.insert("Value", r.value);
					tbl.insert("LB", r.begin);
					tbl.insert("UB", r.end);
					return t;
				}
				if (element.type == KVS::Type::Int64Range)
				{
					const auto &r = std::get<KVS::ranged<int64_t>>(element.value);
					auto t = make_toml_node_table();
					auto &tbl = *t->as_table();
					tbl.insert("Value", r.value);
					tbl.insert("LB", r.begin);
					tbl.insert("UB", r.end);
					return t;
				}
				if (element.type == KVS::Type::FloatRange)
				{
					const auto &r = std::get<KVS::ranged<float>>(element.value);
					auto t = make_toml_node_table();
					auto &tbl = *t->as_table();
					tbl.insert("Value", static_cast<double>(r.value));
					tbl.insert("LB", static_cast<double>(r.begin));
					tbl.insert("UB", static_cast<double>(r.end));
					return t;
				}
				if (element.type == KVS::Type::DoubleRange)
				{
					const auto &r = std::get<KVS::ranged<double>>(element.value);
					auto t = make_toml_node_table();
					auto &tbl = *t->as_table();
					tbl.insert("Value", r.value);
					tbl.insert("LB", r.begin);
					tbl.insert("UB", r.end);
					return t;
				}
				if (element.type == KVS::Type::Array)
					return arrayToTOML(std::get<std::shared_ptr<KVS::Array>>(element.value));
				if (element.type == KVS::Type::Map)
					return mapToTOML(std::get<std::shared_ptr<KVS::Map>>(element.value));
				if (element.type == KVS::Type::Ordered)
					return orderedToTOML(std::get<std::shared_ptr<KVS::Ordered>>(element.value));

				throw std::runtime_error("Unsupported element type for TOML conversion");
			}

			inline toml_node_ptr arrayToTOML(std::shared_ptr<KVS::Array> array)
			{
				auto out = make_toml_node_array();
				auto &arr = *out->as_array();

				for (const auto &element : *array)
				{
					auto node = elementToTOML(element);
					if (!node)
						throw std::runtime_error("Failed converting array element to TOML");
					push_toml_node(arr, std::move(node));
				}
				return out;
			}

			inline toml_node_ptr mapToTOML(std::shared_ptr<KVS::Map> map)
			{
				auto out = make_toml_node_table();
				auto &tbl = *out->as_table();

				for (const auto &pair : *map)
				{
					auto node = elementToTOML(pair.second);
					if (!node)
						throw std::runtime_error("Failed converting map value to TOML");
					insert_toml_node(tbl, keyToString(pair.first), std::move(node));
				}
				return out;
			}

			inline toml_node_ptr orderedToTOML(std::shared_ptr<KVS::Ordered> ordered)
			{
				if (ordered->hasDuplicateKeys())
				{
					auto out = make_toml_node_array();
					auto &arr = *out->as_array();

					for (const auto &pair : *ordered)
					{
						// Coalesce consecutive single-key pairs into a single table entry.
						auto out = make_toml_node_array();
						auto &arr = *out->as_array();

						toml_node_ptr current_entry;
						toml::table *cur_tbl = nullptr;
						std::unordered_set<std::string> seen_keys;

						for (const auto &pair : *ordered)
						{
							const std::string key = pair.first;

							// start a new entry if needed
							if (!current_entry)
							{
								current_entry = make_toml_node_table();
								cur_tbl = current_entry->as_table();
								seen_keys.clear();
							}

							// if the key would duplicate inside the current entry, close it and start a new one
							if (seen_keys.find(key) != seen_keys.end())
							{
								push_toml_node(arr, std::move(current_entry));
								current_entry = make_toml_node_table();
								cur_tbl = current_entry->as_table();
								seen_keys.clear();
							}

							auto valueNode = elementToTOML(pair.second);
							if (!valueNode)
								throw std::runtime_error("Failed converting ordered value to TOML");

							insert_toml_node(*cur_tbl, key, std::move(valueNode));
							seen_keys.insert(key);
						}

						if (current_entry)
							push_toml_node(arr, std::move(current_entry));

						return out;
					}
					return out;
				}
				else
				{
					auto out = make_toml_node_table();
					auto &tbl = *out->as_table();

					for (const auto &pair : *ordered)
					{
						if (tbl.contains(pair.first))
							throw std::runtime_error(
									"Cannot serialize KVS::Ordered with duplicate keys as a TOML table: " + pair.first);

						auto valueNode = elementToTOML(pair.second);
						if (!valueNode)
							throw std::runtime_error("Failed converting ordered value to TOML");

						insert_toml_node(tbl, pair.first, std::move(valueNode));
					}

					return out;
				}
			}

			inline KVS::Element elementFromTOML(const toml::node &node)
			{
				if (node.is_integer() || node.is_floating_point() || node.is_string())
					return scalarFromTOML(node);

				if (const auto *arr = node.as_array())
				{
					// Treat "array of tables" as Ordered (portable, preserves order).
					// Otherwise treat as a plain array.
					bool all_tables = true;
					for (const auto &elem : *arr)
					{
						if (!elem.is_table())
						{
							all_tables = false;
							break;
						}
					}
					return all_tables ? orderedFromTOML(*arr) : arrayFromTOML(*arr);
				}

				if (const auto *tbl = node.as_table())
				{
					const bool isRange =
						tbl->size() == 3 &&
						tbl->contains("Value") &&
						tbl->contains("LB") &&
						tbl->contains("UB");

					return isRange ? rangedFromTOML(*tbl) : mapFromTOML(*tbl);
				}

				throw std::runtime_error("Unsupported TOML type for conversion to KVS::Element");
			}

			inline KVS::Element arrayFromTOML(const toml::array &arr)
			{
				KVS::Array array;
				for (const auto &elem : arr)
					array.push_back(elementFromTOML(elem));
				return KVS::Element(array);
			}

			inline KVS::Element mapFromTOML(const toml::table &tbl)
			{
				KVS::Map map;
				for (const auto &kvp : tbl)
				{
					const std::string key{kvp.first.str()};
					KVS::Element value = elementFromTOML(kvp.second);
					map.upsert(key) = value;
				}
				return KVS::Element(map);
			}

			inline KVS::Element rangedFromTOML(const toml::table &tbl)
			{
				if (!tbl.contains("Value") || !tbl.contains("LB") || !tbl.contains("UB"))
					throw std::runtime_error("Expected TOML table with 'Value', 'LB', and 'UB'");

				auto valueElement = elementFromTOML(*tbl.get("Value"));
				auto lbElement = elementFromTOML(*tbl.get("LB"));
				auto ubElement = elementFromTOML(*tbl.get("UB"));

				if (valueElement.type != lbElement.type || valueElement.type != ubElement.type)
					throw std::runtime_error("Range fields 'Value', 'LB', 'UB' must have same type");

				switch (valueElement.type)
				{
					case KVS::Type::Int32:
						return KVS::Element(KVS::int32Range{
								valueElement.get<int32_t>(), lbElement.get<int32_t>(), ubElement.get<int32_t>()});
					case KVS::Type::Int64:
						return KVS::Element(KVS::int64Range{
								valueElement.get<int64_t>(), lbElement.get<int64_t>(), ubElement.get<int64_t>()});
					case KVS::Type::Float:
						return KVS::Element(KVS::floatRange{
								valueElement.get<float>(), lbElement.get<float>(), ubElement.get<float>()});
					case KVS::Type::Double:
						return KVS::Element(KVS::doubleRange{
								valueElement.get<double>(), lbElement.get<double>(), ubElement.get<double>()});
					default:
						throw std::runtime_error("Unsupported range type");
				}
			}

			inline KVS::Element orderedFromTOML(const toml::array &arr)
			{
				KVS::Ordered ordered;
				size_t idx = 0;

				for (const auto &elem : arr)
				{
					const auto *t = elem.as_table();
					if (!t)
						throw std::runtime_error("Expected array of tables for KVS::Ordered");

					// 1) explicit Key/Value pair form
					if (t->contains("Key") && t->contains("Value"))
					{
						const auto *keyNode = t->get("Key");
						if (!keyNode || !keyNode->is_string())
							throw std::runtime_error("Ordered Key must be a string");
						const std::string key{keyNode->as_string()->get()};

						const auto *valueNode = t->get("Value");
						if (!valueNode)
							throw std::runtime_error("Ordered entry missing Value");
						KVS::Element value = elementFromTOML(*valueNode);
						ordered.upsert(key) = value;
						continue;
					}

					// 2) named-entity form: use "name" (or "Name") as the entry key when present
					if (t->contains("name") || t->contains("Name"))
					{
						const std::string nameKey = t->contains("name") ? "name" : "Name";
						const auto *nameNode = t->get(nameKey);
						if (!nameNode || !nameNode->is_string())
							throw std::runtime_error("Ordered entity name must be a string");

						std::string key = nameNode->as_string()->get();

						// build a temporary table excluding the name field to become the value map
						toml::table tmp = *t;
						tmp.erase(nameKey);
						KVS::Element value = mapFromTOML(tmp);
						ordered.upsert(key) = value;
						continue;
					}

					// 3) legacy single-entry table: { k = v }
					if (t->size() == 1)
					{
						auto it = t->begin();
						const std::string key{it->first.str()};
						KVS::Element value = elementFromTOML(it->second);
						ordered.upsert(key) = value;
						continue;
					}

					// 4) multi-field anonymous entity: convert whole table to a Map value and assign a generated key
					{
						KVS::Element value = mapFromTOML(*t);
						const std::string genkey = std::to_string(idx++);
						ordered.upsert(genkey) = value;
					}
				}

				return KVS::Element(ordered);
			}
		}

		inline toml_node_ptr toTOML(const KVS &kvs)
		{
			return elementToTOML(kvs.root);
		}

		inline std::string toTOMLString(const KVS &kvs)
		{
			auto n = toTOML(kvs);
			if (!n)
				throw std::runtime_error("Failed converting KVS to TOML");

			std::ostringstream os;
			// Serialize the TOML node to a string
			if (n->type() == toml::node_type::table)
				os << *n->as_table();
			else if (n->type() == toml::node_type::array)
				os << *n->as_array();
			else
				throw std::runtime_error("Unexpected TOML node type at root");
			return os.str();
		}

		inline void toTOMLFile(const KVS &kvs, const std::string &filename)
		{
			auto n = toTOMLString(kvs);
			std::ofstream ofs(filename);
			if (!ofs)
				throw std::runtime_error("Failed opening file for writing: " + filename);

			ofs << n;
		}

		inline KVS fromTOML(const toml::node &node)
		{
			return KVS(elementFromTOML(node));
		}

		inline KVS fromTOMLString(const std::string &tomlString)
		{
			toml::table tbl = toml::parse(tomlString);
			return fromTOML(tbl);
		}

		inline KVS fromTOMLFile(const std::string &filename)
		{
			toml::table tbl = toml::parse_file(filename);
			return fromTOML(tbl);
		}
	}
}
#endif // KVSTOML_H
