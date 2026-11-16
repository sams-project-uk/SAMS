#ifndef KVSCORE_H
#define KVSCORE_H

#include <variant>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <utility>
#include <initializer_list>
#include <map>
#include <iostream>
#include <stack>
#include <unordered_set>
#include <functional>

#include "kvsutils.h"

namespace SAMS
{

	struct KVS
	{

		/**
		 * The Type enum defines the possible types of values that can be stored in the KVS. It includes:
		 * - Undefined: Represents an uninitialized or undefined value.
		 * - Array: Represents an array of elements.
		 * - Map: Represents a key-value map.
		 * - Int32: Represents a 32-bit integer value.
		 * - Int64: Represents a 64-bit integer value.
		 * - Float: Represents a single-precision floating-point value.
		 * - Double: Represents a double-precision floating-point value.
		 * - String: Represents a string value.
		 * - Bool: Represents a boolean value.
		 */
		enum class Type
		{
			None,
			Array,
			Map,
			Ordered,
			Int32,
			Int64,
			Float,
			Double,
			Int32Range,
			Int64Range,
			FloatRange,
			DoubleRange,
			String,
			Bool
		};

		using None = std::monostate;

		/**
		 * Converts the text of an element to a number when a number is asked for and the element holds a string. Null if strings are not converted
		 */
		inline static double (*stringToNumber)(void* context, const std::string &text) = nullptr;
		/**
		 * The context that is passed to stringToNumber
		 */
		inline static void* stringToNumberContext = nullptr;

		struct Array;
		struct Map;
		struct Ordered;

		/**
		 * Class representing a ranged value - a value that contains a value and a range that it must be within
		 * @tparam T The type of the value and range (e.g. int, float, etc.)
		 */
		template <typename T>
			struct ranged
			{
				T begin;
				T value;
				T end;

				/**
				 * Set the value of the ranged object, ensuring that it is within the specified range. If the value is out of range, an std::out_of_range exception is thrown.
				 */
				void set(T newValue)
				{
					if (newValue < begin || newValue > end)
						throw std::out_of_range("Value out of range");
					value = newValue;
				}

				/**
				 * Compare two ranged objects for equality. They are equal if only their value match, regardless of their range.
				 */
				bool operator==(const ranged<T> &other) const
				{
					return value == other.value;
				}

				ranged() = default;

				/**
				 * Constructor for the ranged object. It takes a value and a range (begin and end) and initializes the object. If the value is out of the specified range, an std::out_of_range exception is thrown.
				 * @param v The value to be stored in the ranged object.
				 * @param b The beginning of the range (inclusive).
				 * @param e The end of the range (inclusive).
				 * @throws std::out_of_range If the value is outside the specified range.
				 */
				ranged(T v, T b, T e) : begin(b), value(v), end(e)
				{
					if (v < b || v > e)
						throw std::out_of_range("Value out of range");
				}
			};

		// Type aliases for specific ranged types
		using int32Range = ranged<int32_t>;
		using int64Range = ranged<int64_t>;
		using floatRange = ranged<float>;
		using doubleRange = ranged<double>;

		/**
		 * Basic type erased element class that can hold any of the supported types.
		 */
		struct Element
		{

			private:
				template<typename T>
				T tryConvert()
				{
					//A boolean only converts to a boolean
					if (type == Type::Bool)
					{
						if constexpr (std::is_same_v<T, bool>)
							return std::get<bool>(value);
						throw std::bad_variant_access();
					}
					if constexpr (std::is_arithmetic_v<T>)
					{
						if (type == Type::Int32 && std::is_convertible_v<int32_t, T>)
							return std::get<int32_t>(value);
						else if (type == Type::Int64 && std::is_convertible_v<int64_t, T>)
							return std::get<int64_t>(value);
						else if (type == Type::Float && std::is_convertible_v<float, T>)
							return std::get<float>(value);
						else if (type == Type::Double && std::is_convertible_v<double, T>)
							return std::get<double>(value);
					}
					throw std::bad_variant_access();

				}

			public:

			std::variant<std::monostate, int32_t, int64_t, float, double, bool, ranged<int32_t>, ranged<int64_t>, ranged<float>, ranged<double>, std::string, std::shared_ptr<Array>, std::shared_ptr<Map>, std::shared_ptr<Ordered>> value;
			std::map<std::string, std::string> metadata; // Optional metadata associated with the element
			bool directive = false; // Flag to indicate if the element is a directive
			Type type = Type::None;

			Element() = default;
			Element(int32_t v) : value(v), type(Type::Int32) {}
			Element(int64_t v) : value(v), type(Type::Int64) {}
			Element(float v) : value(v), type(Type::Float) {}
			Element(double v) : value(v), type(Type::Double) {}
			Element(bool v) : value(v), type(Type::Bool) {}
			Element(ranged<int32_t> v) : value(v), type(Type::Int32Range) {}
			Element(ranged<int64_t> v) : value(v), type(Type::Int64Range) {}
			Element(ranged<float> v) : value(v), type(Type::FloatRange) {}
			Element(ranged<double> v) : value(v), type(Type::DoubleRange) {}
			Element(const std::string &v) : value(v), type(Type::String) {}
			Element(std::string &&v) : value(std::move(v)), type(Type::String) {}
			Element(const char *s) : value(std::string(s)), type(Type::String) {}
			Element(const Array &a) : value(std::make_shared<Array>(a)), type(Type::Array) {}
			Element(std::shared_ptr<Array> a) : value(std::move(a)), type(Type::Array) {}
			Element(const Map &m) : value(std::make_shared<Map>(m)), type(Type::Map) {}
			Element(std::shared_ptr<Map> m) : value(std::move(m)), type(Type::Map) {}
			Element(const Ordered &n) : value(std::make_shared<Ordered>(n)), type(Type::Ordered) {}
			Element(std::shared_ptr<Ordered> n) : value(std::move(n)), type(Type::Ordered) {}

			/*Has this key been accessed?*/
			mutable bool touched=false;

			/**
			 * Set the optional comment for the element
			 */
			void setComment(const std::string &c)
			{
				setMetadata("comment", c);
			}

			/**
			 * Get the optional comment for the element. If no comment is set, an empty string is returned.
			 */
			std::string getComment() const
			{
				auto it = metadata.find("comment");
				if (it != metadata.end())
					return it->second;
				return "";
			}

			/**
			 * Check if this element contains a given key. This is only applicable if the element is a Map or an Ordered collection. If the element is not a Map or Ordered collection, an std::runtime_error is thrown.
			 * @param key The key to check for in the element.
			 * @return true if the key exists in the element, false otherwise.
			 * @throws std::runtime_error If the element is not a Map or Ordered collection.
			 */
			bool hasElement(const std::string &key) const {
				if (type == Type::Map){
					auto &map = get<Map&>();
					for (const auto &pair : map.data){
						if (pair.first == key)
							return true;
					}
					return false;
				}
				else if (type == Type::Ordered){
					auto &ordered = get<Ordered&>();
					for (const auto &pair : ordered.data){
						if (pair.first == key)
							return true;
					}
					return false;
				}
				else if (type == Type ::None){
					//Empty is not invalid, just doesn't have the key
					return false;
				}
				else
					throw std::runtime_error("Element is not a map or ordered collection");
			}

			/**
			 * Get the size of the element. This is only applicable if the element is an Array, Map, or Ordered collection. If the element is not one of these types, an std::runtime_error is thrown.
			 * @return The size of the element.
			 * @throws std::runtime_error If the element is not an array, map, or ordered collection.
			 */
			size_t size() const {
				if (type == Type::Array)
					return get<Array&>().size();
				else if (type == Type::Map)
					return get<Map&>().data.size();
				else if (type == Type::Ordered)
					return get<Ordered&>().data.size();
				else
					throw std::runtime_error("Element is not an array, map, or ordered collection");
			}

			bool isType(Type t) const {
				return type == t;
			}

			template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
				Element &operator[](T index);
			Element &operator[](const std::string key);

			Element &getPath(const std::string &path);
			Element getPath(const std::string &path, const Element &defaultValue);
			Element &upsertPath(const std::string &path);
			Element &upsertPathUntouched(const std::string &path);
			template<typename T>
			Element &setDefaultPath(const std::string &path, const T &value);
			template<typename T>
			Element &setContentPath(const std::string &path, const T &value);
			template<typename T>
			Element &setDefault(const std::string &key, const T &value);
			template<typename T>
			Element &setContent(const std::string &key, const T &value);
			void merge(const Element &other);

			Element getElement(const std::string key);
			template<typename T>
			Element getElement(const std::string key, const T&value);
			Array getArray(const std::string &key);
			template<typename T>
			Array getArray(const std::string &key, const T &defaultValue);
			Ordered getOrdered(const std::string &key);
			template<typename T>
			Ordered getOrdered(const std::string &key, const T &defaultValue);
			Map getMap(const std::string &key);
			template<typename T>
			Map getMap(const std::string &key, const T &defaultValue);
			Element &operator[](const char *key);
			Element &operator[](const Element &key);

			Element &upsert(const std::string &key);
			Element &upsert(const char *key);

			bool operator==(const Element &other) const;
			//get<T&> gives a reference to the value, which must be of type T. get<T> gives a copy, and any scalar can be got as a std::string
			template <typename T>
				std::conditional_t<std::is_reference_v<T>, T, std::remove_cv_t<T>> get();

			template<typename T>
				T getValue() const;

			template<typename T>
				T getDefault(const T &defaultValue) const;

			template<typename T>
				std::conditional_t<std::is_reference_v<T>, const std::remove_reference_t<T> &, std::remove_cv_t<T>> get() const;

			void setMetadata(const std::string &key, const std::string &value)
			{
				metadata[key] = value;
			}

			bool hasMetadata(const std::string &key) const
			{
				return metadata.find(key) != metadata.end();
			}

			std::string getMetadata(const std::string &key) const
			{
				auto it = metadata.find(key);
				if (it != metadata.end())
					return it->second;
				throw std::out_of_range("Metadata key not found: " + key);
			}

			/**
			 * Convert a Map element to an Ordered collection. This is useful when you want to maintain the order of key-value pairs while still allowing access by key. If the element is not a Map, an std::runtime_error is thrown.
			 * @throws std::runtime_error If the element is not a Map and cannot be converted to an ordered collection.
			 */
			void toOrdered()
			{
				if (type == Type::Map)
				{
					auto newOrdered = std::make_shared<Ordered>();
					auto map = std::get<std::shared_ptr<Map>>(value);
					for (const auto &pair : map->data)
					{
						newOrdered->data.emplace_back(pair.first, pair.second);
					}
					value = newOrdered;
					type = Type::Ordered;
				}
				else if (type == Type::Ordered)
				{
					// Already an ordered collection, do nothing
				}
				else
				{
					throw std::runtime_error("Element is not a map and cannot be converted to an ordered collection");
				}
			}

			/**
			 * Convert an ordered collection to a map or silently pass through a map. Otherwise throw an exception.
			 */
			void toMap()
			{
				if (type == Type::Ordered)
				{
					auto newMap = std::make_shared<Map>();
					auto ordered = std::get<std::shared_ptr<Ordered>>(value);
					for (const auto &pair : ordered->data)
					{
						newMap->data.emplace_back(pair.first, pair.second);
					}
					value = newMap;
					type = Type::Map;
				}
				else if (type == Type::Map)
				{
					// Already a map, do nothing
				}
				else
				{
					throw std::runtime_error("Element is not an ordered collection and cannot be converted to a map");
				}
			}


			/**
			 * Convert an integer element to an int64_t. If the element is not an integer type, an std::runtime_error is thrown.
			 */
			int64_t asInt(){
				if (type == Type::Int32)
					return std::get<int32_t>(value);
				else if (type == Type::Int64)
					return std::get<int64_t>(value);
				else if (type == Type::String && stringToNumber)
				{
					//The text must evaluate to a whole number
					const double number = stringToNumber(stringToNumberContext, std::get<std::string>(value));
					if (number != std::trunc(number))
						throw std::runtime_error("Element is not an integer type");
					return static_cast<int64_t>(number);
				}
				else
					throw std::runtime_error("Element is not an integer type");
			}

			/**
			 * Convert the element to a boolean. Recognizes "true", "T" and 1 as true and "false", "F" and 0 as false. Anything else throws an std::runtime_error.
			 */
			bool asBool(){
				if (type == Type::Bool)
					return std::get<bool>(value);
				if (type == Type::String)
				{
					const std::string &s = std::get<std::string>(value);
					if (s == "true" || s == "T") return true;
					if (s == "false" || s == "F") return false;
				}
				else if (type == Type::Int32 || type == Type::Int64)
				{
					int64_t i = asInt();
					if (i == 1) return true;
					if (i == 0) return false;
				}
				throw std::runtime_error("Element is not a boolean");
			}

			/**
			 * Convert a real number element to a double. If the element is not a real number type or an integer type, an std::runtime_error is thrown.
			 */
			double asReal(){
				if (type == Type::Float)
					return std::get<float>(value);
				else if (type == Type::Double)
					return std::get<double>(value);
				else if (type == Type::Int32)
					return static_cast<double>(std::get<int32_t>(value));
				else if (type == Type::Int64)
					return static_cast<double>(std::get<int64_t>(value));
				else if (type == Type::String && stringToNumber)
					return stringToNumber(stringToNumberContext, std::get<std::string>(value));
				else
					throw std::runtime_error("Element is not a real number type");
			}

			/**
			 * Convert the element to a string representation. If the element is not a string, integer, or real number type, an std::runtime_error is thrown.
			 */
			std::string asString() const{
				if (type == Type::String)
					return std::get<std::string>(value);
				if (type == Type::Bool)
					return KVSConversion::boolToString(std::get<bool>(value));
				if (type == Type::Int32)
					return std::to_string(std::get<int32_t>(value));
				else if (type == Type::Int64)
					return std::to_string(std::get<int64_t>(value));
				else if (type == Type::Float)
					return KVSConversion::floatToString(std::get<float>(value));
				else if (type == Type::Double)
					return KVSConversion::doubleToString(std::get<double>(value));
				else
					throw std::runtime_error("Element cannot be converted to string");
			}

			void traverse(std::function<void(const std::string&, const Element&)> blockStart, std::function<void(const std::string&)> blockEnd, std::function<void(const std::string&, const Element&)> elementCallback) const
			{
				if (type == Type::Map)
				{
					auto map = std::get<std::shared_ptr<Map>>(value);
					for (const auto &pair : map->data)
					{
						if (pair.second.type != Type::Map && pair.second.type != Type::Ordered)
							elementCallback(pair.first, pair.second);
						else
						{
							blockStart(pair.first, pair.second); 
							pair.second.traverse(blockStart, blockEnd, elementCallback);
							blockEnd(pair.first);
						}
					}
				}
				else if (type == Type::Ordered)
				{
					auto ordered = std::get<std::shared_ptr<Ordered>>(value);
					for (const auto &pair : ordered->data)
					{
						if (pair.second.type != Type::Map && pair.second.type != Type::Ordered)
							elementCallback(pair.first, pair.second);
						else
						{
							blockStart(pair.first, pair.second);
							pair.second.traverse(blockStart, blockEnd, elementCallback);
							blockEnd(pair.first);
						}
					}
				}
				else
				{
					elementCallback("", *this);
				}
			}

			/**
			 * Go through the blocks and keys below this element as traverse does, except that a block is only entered if blockStart returns true
			 * @param blockStart Called at the start of each block. If it returns false the contents of the block and its blockEnd are skipped
			 * @param blockEnd Called at the end of each block that is entered
			 * @param elementCallback Called for each key
			 */
			void partialTraverse(std::function<bool(const std::string&, const Element&)> blockStart, std::function<void(const std::string&)> blockEnd, std::function<void(const std::string&, const Element&)> elementCallback) const
			{
				if (type == Type::Map)
				{
					auto map = std::get<std::shared_ptr<Map>>(value);
					for (const auto &pair : map->data)
					{
						if (pair.second.type != Type::Map && pair.second.type != Type::Ordered)
							elementCallback(pair.first, pair.second);
						else if (blockStart(pair.first, pair.second))
						{
							pair.second.partialTraverse(blockStart, blockEnd, elementCallback);
							blockEnd(pair.first);
						}
					}
				}
				else if (type == Type::Ordered)
				{
					auto ordered = std::get<std::shared_ptr<Ordered>>(value);
					for (const auto &pair : ordered->data)
					{
						if (pair.second.type != Type::Map && pair.second.type != Type::Ordered)
							elementCallback(pair.first, pair.second);
						else if (blockStart(pair.first, pair.second))
						{
							pair.second.partialTraverse(blockStart, blockEnd, elementCallback);
							blockEnd(pair.first);
						}
					}
				}
				else
				{
					elementCallback("", *this);
				}
			}

		};


		/**
		 * Report every block and key that has never been read. Those that the code set, which are marked as autoset, are reported as such
		 * @param stream The stream to report to
		 */
		template<typename T_stream>
		void checkUnused(T_stream &stream){
			//A block that has never been read is reported, but not entered, because nothing in it has been read either
			partialTraverse([&stream](const std::string& string, const Element&element){
				if (!element.touched){
					if (element.hasMetadata("autoset")) stream << "Block " << string << " was set by the code and has never been read\n";
					else stream << "Block " << string << " has never been read - this may indicate a block in the deck that is not intended to be there\n";
				}
				return element.touched;
				}, [](const std::string&){}, [&stream](const std::string& name, const Element&element){
				if (!element.touched){
					if (element.hasMetadata("autoset")) stream << "Element " << name << " was set by the code and has never been read\n";
					else stream << "Element " << name << " has never been read - this may indicate a key in the deck that is not intended to be there\n";
				}
				});
		}

		/**
		 * Custom hash function for Element
		 */
		struct ElementHash
		{
			size_t operator()(Element const &e) const noexcept
			{
				using K = KVS;
				size_t seed = std::hash<int>()(static_cast<int>(e.type));
				// Simple hash function based on a TEA inspired mix function
				auto mix = [](size_t a, size_t b)
				{
					return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
				};

				switch (e.type)
				{
					case K::Type::Int32:
						return mix(seed, std::hash<int32_t>{}(std::get<int32_t>(e.value)));
					case K::Type::Int64:
						return mix(seed, std::hash<int64_t>{}(std::get<int64_t>(e.value)));
					case K::Type::Float:
						return mix(seed, std::hash<float>{}(std::get<float>(e.value)));
					case K::Type::Double:
						return mix(seed, std::hash<double>{}(std::get<double>(e.value)));
					case K::Type::Int32Range:
						return mix(seed, std::hash<int32_t>{}(std::get<ranged<int32_t>>(e.value).value));
					case K::Type::Int64Range:
						return mix(seed, std::hash<int64_t>{}(std::get<ranged<int64_t>>(e.value).value));
					case K::Type::FloatRange:
						return mix(seed, std::hash<float>{}(std::get<ranged<float>>(e.value).value));
					case K::Type::DoubleRange:
						return mix(seed, std::hash<double>{}(std::get<ranged<double>>(e.value).value));
					case K::Type::String:
						return mix(seed, std::hash<std::string>{}(std::get<std::string>(e.value)));
					case K::Type::Bool:
						return mix(seed, std::hash<bool>{}(std::get<bool>(e.value)));
					case K::Type::Array:
						return mix(seed, std::hash<const void *>{}(std::get<std::shared_ptr<K::Array>>(e.value).get()));
					case K::Type::Map:
						return mix(seed, std::hash<const void *>{}(std::get<std::shared_ptr<K::Map>>(e.value).get()));
					case K::Type::Ordered:
						return mix(seed, std::hash<const void *>{}(std::get<std::shared_ptr<K::Ordered>>(e.value).get()));
					default:
						return seed;
				}
			}
		};

		// Comparison functor for std::map (must provide a strict weak ordering)
		struct ElementLess
		{
			bool operator()(Element const &a, Element const &b) const noexcept
			{
				if (a.type != b.type)
					return static_cast<int>(a.type) < static_cast<int>(b.type);

				switch (a.type)
				{
					case Type::Int32:
						return std::get<int32_t>(a.value) < std::get<int32_t>(b.value);
					case Type::Int64:
						return std::get<int64_t>(a.value) < std::get<int64_t>(b.value);
					case Type::Float:
						return std::get<float>(a.value) < std::get<float>(b.value);
					case Type::Double:
						return std::get<double>(a.value) < std::get<double>(b.value);
					case Type::Int32Range:
						return std::get<int32Range>(a.value).value < std::get<int32Range>(b.value).value;
					case Type::Int64Range:
						return std::get<int64Range>(a.value).value < std::get<int64Range>(b.value).value;
					case Type::FloatRange:
						return std::get<floatRange>(a.value).value < std::get<floatRange>(b.value).value;
					case Type::DoubleRange:
						return std::get<doubleRange>(a.value).value < std::get<doubleRange>(b.value).value;
					case Type::String:
						return std::get<std::string>(a.value) < std::get<std::string>(b.value);
					case Type::Bool:
						return std::get<bool>(a.value) < std::get<bool>(b.value);
					case Type::Array:
						return std::get<std::shared_ptr<Array>>(a.value).get() < std::get<std::shared_ptr<Array>>(b.value).get();
					case Type::Map:
						return std::get<std::shared_ptr<Map>>(a.value).get() < std::get<std::shared_ptr<Map>>(b.value).get();
					case Type::Ordered:
						return std::get<std::shared_ptr<Ordered>>(a.value).get() < std::get<std::shared_ptr<Ordered>>(b.value).get();
					default:
						return false;
				}
			}
		};


		// Output operator for Element
		friend std::ostream &operator<<(std::ostream &os, const Element &e)
		{
			switch (e.type)
			{
				case Type::Int32:
					os << std::get<int32_t>(e.value);
					break;
				case Type::Int64:
					os << std::get<int64_t>(e.value);
					break;
				case Type::Float:
					os << std::get<float>(e.value);
					break;
				case Type::Double:
					os << std::get<double>(e.value);
					break;

				case Type::Int32Range:
					os << std::get<ranged<int32_t>>(e.value).value << " (range: " << std::get<ranged<int32_t>>(e.value).begin << " - " << std::get<ranged<int32_t>>(e.value).end << ")";
					break;
				case Type::Int64Range:
					os << std::get<ranged<int64_t>>(e.value).value << " (range: " << std::get<ranged<int64_t>>(e.value).begin << " - " << std::get<ranged<int64_t>>(e.value).end << ")";
					break;
				case Type::FloatRange:
					os << std::get<ranged<float>>(e.value).value << " (range: " << std::get<ranged<float>>(e.value).begin << " - " << std::get<ranged<float>>(e.value).end << ")";
					break;
				case Type::DoubleRange:
					os << std::get<ranged<double>>(e.value).value << " (range: " << std::get<ranged<double>>(e.value).begin << " - " << std::get<ranged<double>>(e.value).end << ")";
					break;
				case Type::String:
					os << std::get<std::string>(e.value);
					break;
				case Type::Bool:
					os << KVSConversion::boolToString(std::get<bool>(e.value));
					break;
				case Type::Array:
					// Array has its own output operator, so we can just delegate to it
					os << *std::get<std::shared_ptr<Array>>(e.value);
					break;
				case Type::Map:
					// Map has its own output operator, so we can just delegate to it
					os << *std::get<std::shared_ptr<Map>>(e.value);
					break;
				case Type::Ordered:
					// Ordered has its own output operator, so we can just delegate to it
					os << *std::get<std::shared_ptr<Ordered>>(e.value);
					break;
				default:
					os << "None";
					break;
			}
			return os;
		}

		/**
		 * Class representing an array of elements
		 */
		struct Array
		{
			std::vector<Element> elements;
			Array() = default;
			Array(std::initializer_list<Element> il) : elements(il) {}
			/**
			 * Access an element of the array by index. Throws std::out_of_range if the index is out of bounds.
			 * @param index The index of the element to access.
			 * @return A reference to the element at the specified index.
			 * @throws std::out_of_range If the index is out of bounds.
			 */
			Element &operator[](size_t index)
			{
				return elements[index];
			}
			/**
			 * Add an element to the end of the array.
			 * @param element The element to be added to the array.
			 */
			void push_back(const Element &element)
			{
				elements.push_back(element);
			}
			/**
			 * Set the size of the array
			 * @param newSize The new size of the array
			 */
			void resize(size_t newSize)
			{
				elements.resize(newSize);
			}
			/**
			 * Reserve space for a certain number of elements in the array to optimize memory allocations when the number of elements is known in advance.
			 * @param newCapacity The number of elements to reserve space for.
			 */
			void reserve(size_t newCapacity)
			{
				elements.reserve(newCapacity);
			}
			/**
			 * Get the number of elements in the array
			 * @return The number of elements currently stored in the array.
			 */
			size_t size() const
			{
				return elements.size();
			}

			auto begin() { return elements.begin(); }
			auto end() { return elements.end(); }
			auto begin() const { return elements.begin(); }
			auto end() const { return elements.end(); }

			/**
			 * Go through each element calling the provided callback function with the element as an argument
			 * @param elementCallback The callback function to be called for each element in the array. It should take a single argument of type const Element&.
			 */
			void traverse(std::function<void(const Element&)> elementCallback){
				for (const auto &element : elements)
				{
					elementCallback(element);
				}
			}

			/**
			 * Get the element at the specified index as an Array. If the element at the index is not an Array, an empty Array is returned.
			 */
			Array getArray(size_t index)
			{
				try {
					Element &element = (*this)[index];
					return element.get<Array>();
				}
				catch (const std::runtime_error &e)
				{
					return Array{};
				}
				return Array{};
			}

			template<typename T>
			Array getArray(size_t index, const T &defaultValue)
			{
				try
				{
					return (*this)[index].get<Array>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Array>)
						return defaultValue;
					else
						return Array{Element(defaultValue)};
					return defaultValue;
				}
			}

			Ordered getOrdered(size_t index)
			{
				try
				{
					return (*this)[index].get<Ordered>();
				}
				catch (const std::runtime_error &e)
				{
					
					return Ordered{};
				}
			}

			template<typename T>
			Ordered getOrdered(size_t index, const T &defaultValue)
			{
				try
				{
					return (*this)[index].get<Ordered>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Ordered>)
						return defaultValue;
					else
						return Ordered{{"default", Element(defaultValue)}};
					return defaultValue;
				}
			}

			Map getMap(size_t index)
			{
				try
				{
					return (*this)[index].get<Map>();
				}
				catch (const std::runtime_error &e)
				{
					
					return Map{};
				}
			}

			template<typename T>
			Map getMap(size_t index, const T &defaultValue)
			{
				try
				{
					return (*this)[index].get<Map>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Map>)
						return defaultValue;
					else
						return Map{{"default", Element(defaultValue)}};
					return defaultValue;
				}
			}


		};

		// Output operator for array
		friend std::ostream &operator<<(std::ostream &os, const Array &a)
		{
			os << "[";
			for (size_t i = 0; i < a.elements.size(); ++i)
			{
				os << a.elements[i];
				if (i < a.elements.size() - 1)
					os << ", ";
			}
			os << "]";
			return os;
		}

		/**
		 * Class representing a key value pair. The Key
		 */
		struct Map
		{
			std::vector<std::pair<std::string, Element>> data;
			Map() = default;
			Map(std::initializer_list<std::pair<std::string, Element>> il) : data(il) {}

			Element &operator[](const std::string &key)
			{
				//Check for more than one key
				int count = 0;
				std::pair<std::string, Element>* lastMatch = nullptr;
				for (auto &pair : data)
				{
					if (pair.first == key){
						count++;
						lastMatch = &pair;
					}
					if (count > 1)
						break;
				}
				if (count == 1)
					return lastMatch->second;
				else if (count == 0){
					throw std::runtime_error("Key not found: " + key);
				}
				else
					throw std::runtime_error("Multiple elements with key: " + key);
			}


			Element &upsert(const std::string &key)
			{
				//Check for more than one key
				int count = 0;
				std::pair<std::string, Element>* lastMatch = nullptr;
				for (auto &pair : data)
				{
					if (pair.first == key){
						count++;
						lastMatch = &pair;
					}
					if (count > 1)
						break;
				}
				if (count == 1)
					return lastMatch->second;
				else if (count == 0){
					data.emplace_back(key, Element());
					return data.back().second;
				}
				else
					throw std::runtime_error("Multiple elements with key: " + key);
			}


			auto begin() { return data.begin(); }
			auto end() { return data.end(); }
			auto begin() const { return data.begin(); }
			auto end() const { return data.end(); }


			Array getArray(const std::string &key)
			{
				try
				{
					return (*this)[key].get<Array>();
				}
				catch (const std::runtime_error &e)
				{
					return Array{};
				}
			}

			template<typename T>
			Array getArray(const std::string &key, const T &defaultValue)
			{
				try
				{
					return (*this)[key].get<Array>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Array>)
						return defaultValue;
					else
						return Array{Element(defaultValue)};
					return defaultValue;
				}
			}
			
			Map getMap(const std::string &key)
			{
				try
				{
					return (*this)[key].get<Map>();
				}
				catch (const std::runtime_error &e)
				{
					return Map{};
				}
			}

			template<typename T>
			Map getMap(const std::string &key, const T &defaultValue)
			{
				try
				{
					return (*this)[key].get<Map>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Map>)
						return defaultValue;
					else
						return Map{{"default", Element(defaultValue)}};
					return defaultValue;
				}
			}

			Ordered getOrdered(const std::string &key)
			{
				try
				{
					return (*this)[key].get<Ordered>();
				}
				catch (const std::runtime_error &e)
				{
					return Ordered{};
				}
			}

			template<typename T>
			Ordered getOrdered(const std::string &key, const T &defaultValue)
			{
				try
				{
					return (*this)[key].get<Ordered>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Ordered>)
						return defaultValue;
					else
						return Ordered{{"default", Element(defaultValue)}};
				}
			}

			void traverse(std::function<void(const std::string&, const Element&)> elementCallback){
				for (const auto &pair : data)
				{
					elementCallback(pair.first, pair.second);
				}
			}
		};

		// Output operator for map
		friend std::ostream &operator<<(std::ostream &os, const Map &m)
		{
			os << "{";
			size_t count = 0;
			for (const auto &pair : m.data)
			{
				os << pair.first << ": " << pair.second;
				if (count < m.data.size() - 1)
					os << ", ";
				++count;
			}
			os << "}\n";
			return os;
		}

		/**
		 * Class representing an ordered collection of elements. It can be accessed by index
		 * or by key, and maintains the order of insertion.
		 */
		struct Ordered
		{
			std::vector<std::pair<std::string, Element>> data;
			Ordered() = default;
			Ordered(std::initializer_list<std::pair<std::string, Element>> il) : data(il) {}

			Element &operator[](const std::string &key)
			{
				//Check for more than one key
				int count = 0;
				std::pair<std::string, Element>* lastMatch = nullptr;
				for (auto &pair : data)
				{
					if (pair.first == key){
						count++;
						lastMatch = &pair;
					}
					if (count > 1)
						break;
				}
				if (count == 1)
					return lastMatch->second;
				else if (count == 0){
					throw std::runtime_error("Key not found: " + key);
				}
				else
					throw std::runtime_error("Multiple elements with key: " + key);
			}

			Element &upsert(const std::string &key)
			{
				//Check for more than one key
				int count = 0;
				std::pair<std::string, Element>* lastMatch = nullptr;
				for (auto &pair : data)
				{
					if (pair.first == key){
						count++;
						lastMatch = &pair;
					}
					if (count > 1)
						break;
				}
				if (count == 1)
					return lastMatch->second;
				else if (count == 0){
					data.emplace_back(key, Element());
					return data.back().second;
				}
				else
					throw std::runtime_error("Multiple elements with key: " + key);
			} 

			Element &operator[](size_t index)
			{
				return data[index].second;
			}

			Element getElement(size_t index)
			{
				return (*this)[index];
			}

			Element getElement(size_t index, const Element &defaultValue)
			{
				if (index < data.size())
					return (*this)[index];
				else
					return defaultValue;
			}

			Element getElement(const std::string &key)
			{
				return (*this)[key];
			}

			Element getElement(const std::string &key, const Element &defaultValue)
			{
				try {
					return (*this)[key];
				} catch (const std::runtime_error& e) {
					return defaultValue;
				}
			}

			Array getArray(size_t index)
			{
				try {
					return (*this)[index].get<Array>();
				}
				catch (const std::runtime_error &e)
				{
					return Array{};
				}
			}

			template<typename T>
			Array getArray(size_t index, const T &defaultValue)
			{
				try {
					return (*this)[index].get<Array>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Array>)
						return defaultValue;
					else
						return Array{Element(defaultValue)};
				}
			}

			Ordered getOrdered(size_t index)
			{
				try {
					return (*this)[index].get<Ordered>();
				}
				catch (const std::runtime_error &e)
				{
					return Ordered{};
				}
			}

			template<typename T>
			Ordered getOrdered(size_t index, const T &defaultValue)
			{
				try {
					return (*this)[index].get<Ordered>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Ordered>)
						return defaultValue;
					else
						return Ordered{{"default", Element(defaultValue)}};
				}
			}

			Map getMap(size_t index)
			{
				try {
					return (*this)[index].get<Map>();
				}
				catch (const std::runtime_error &e)
				{
					return Map{};
				}
			}

			template<typename T>
			Map getMap(size_t index, const T &defaultValue)
			{
				try {
					return (*this)[index].get<Map>();
				}
				catch (const std::runtime_error &e)
				{
					if constexpr (std::is_same_v<T, Map>)
						return defaultValue;
					else
						return Map{{"default", Element(defaultValue)}};
				}
			}

			std::string getKey(size_t index) const
			{
				return data[index].first;
			}

			std::vector<std::reference_wrapper<Element>> getAllByKey(const std::string &key)
			{
				std::vector<std::reference_wrapper<Element>> matches;
				for (auto &pair : data)
				{
					if (pair.first == key)
						matches.push_back(std::ref(pair.second));
				}
				return matches;
			}

			Element& push_back(const std::string &key, const Element &element)
			{
				data.emplace_back(key, element);
				return data.back().second;
			}

			bool hasDuplicateKeys() const
			{
				std::unordered_set<std::string> keys;
				for (const auto &pair : data)
				{
					if (keys.count(pair.first) > 0)
						return true;
					keys.insert(pair.first);
				}
				return false;
			}

			auto begin() { return data.begin(); }
			auto end() { return data.end(); }
			auto begin() const { return data.begin(); }
			auto end() const { return data.end(); }

			void traverse(std::function<void(const std::string&, const Element&)> elementCallback){
				for (const auto &pair : data)
				{
					elementCallback(pair.first, pair.second);
				}
			}
		};

		// Output operator for Ordered
		friend std::ostream &operator<<(std::ostream &os, const Ordered &n)
		{
			os << "{";
			for (size_t i = 0; i < n.data.size(); ++i)
			{
				os << n.data[i].first << ": " << n.data[i].second;
				if (i < n.data.size() - 1)
					os << ", ";
			}
			os << "}\n";
			return os;
		}

		Element root;

		/**
		 * Called with the message of an error that the KVS cannot recover from, before the error is thrown. If it is not set the error is only thrown
		 */
		std::function<void(const std::string&)> fatalErrorHandler;

		/**
		 * Pass the message of an error that the KVS cannot recover from to the fatal error handler, if there is one
		 */
		void reportFatalError(const std::string &message) const
		{
			if (fatalErrorHandler) fatalErrorHandler(message);
		}

		KVS() = default;
		KVS(const Element &element) : root(element) {}

		//Copying or moving into an existing KVS keeps its fatal error handler
		KVS(const KVS &other) = default;
		KVS(KVS &&other) = default;
		KVS &operator=(const KVS &other)
		{
			root = other.root;
			return *this;
		}
		KVS &operator=(KVS &&other)
		{
			root = std::move(other.root);
			return *this;
		}

		template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
			Element &operator[](T index)
			{
				try { return root[index]; }
				catch (const std::exception &e) { reportFatalError(e.what()); throw; }
			}

		Element &operator[](std::string key)
		{
			try { return root[key]; }
			catch (const std::exception &e) { reportFatalError(e.what()); throw; }
		}

		Element &operator[](const char *key)
		{
			try { return root[std::string(key)]; }
			catch (const std::exception &e) { reportFatalError(e.what()); throw; }
		}

		Element &upsert(const std::string &key)
		{
			//An empty root becomes a block
			if (root.isType(Type::None)) root = Ordered();
			root.touched = true;
			try { Element &element = root.upsert(key); element.touched = true; return element; }
			catch (const std::exception &e) { reportFatalError(e.what()); throw; }
		}

		Element &upsert(const char *key)
		{
			//An empty root becomes a block
			if (root.isType(Type::None)) root = Ordered();
			root.touched = true;
			try { Element &element = root.upsert(std::string(key)); element.touched = true; return element; }
			catch (const std::exception &e) { reportFatalError(e.what()); throw; }
		}

		void traverse(std::function<void(const std::string&, const Element&)> blockStart, std::function<void(const std::string&)> blockEnd, std::function<void(const std::string&, const Element&)> elementCallback) const
		{
			blockStart("{root}",  root);
			root.traverse(blockStart, blockEnd, elementCallback);
			blockEnd("{root}");
		}

		/**
		 * Go through the blocks and keys of the KVS as traverse does, except that a block is only entered if blockStart returns true
		 * @param blockStart Called at the start of each block. If it returns false the contents of the block and its blockEnd are skipped
		 * @param blockEnd Called at the end of each block that is entered
		 * @param elementCallback Called for each key
		 */
		void partialTraverse(std::function<bool(const std::string&, const Element&)> blockStart, std::function<void(const std::string&)> blockEnd, std::function<void(const std::string&, const Element&)> elementCallback) const
		{
			if (blockStart("{root}", root))
			{
				root.partialTraverse(blockStart, blockEnd, elementCallback);
				blockEnd("{root}");
			}
		}

	/**
	 * Get an element by key, returning a copy of the element. This means that if the element is a map or ordered collection then modifying the members of that element WILL modify the original element, but reassigning the element itself will not modify the original element. If the key does not exist, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve.
	 * @return A copy of the element associated with the specified key.
	 * @throws std::runtime_error If the key does not exist or if the root element is not a map or ordered collection.
	 */
	inline KVS::Element getElement(const std::string key)
	{
		try { return root[key]; }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	/**
	 * Get an element by key, returning a copy of the element. If the key does not exist then an element with the provided value is returned. This element is NOT added to the collection and modifying it will not modify the original collection. This means that if the element is a map, array or ordered collection then modifying the members of that element WILL modify the original element, but reassigning the element itself will not modify the original element. If the root element is not a map or ordered collection, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve.
	 * @param value The default value to return if the key does not exist.
	 * @return A copy of the element associated with the specified key, or an element with the provided value if the key does not exist.
	 * @throws std::runtime_error If the root element is not a map or ordered collection.
	 */
	template<typename T>
		KVS::Element getElement(const std::string key, const T&value) {
			try { return root.getElement(key, value); }
			catch (const std::exception &e) { reportFatalError(e.what()); throw; }
		}

	KVS::Array getArray(const std::string &key)
	{
		try { return root.getArray(key); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	template<typename T>
	KVS::Array getArray(const std::string &key, const T &defaultValue)
	{
		try { return root.getArray(key, defaultValue); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	KVS::Ordered getOrdered(const std::string &key)
	{
		try { return root.getOrdered(key); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	template<typename T>
	KVS::Ordered getOrdered(const std::string &key, const T &defaultValue)
	{
		try { return root.getOrdered(key, defaultValue); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	KVS::Map getMap(const std::string &key)
	{
		try { return root.getMap(key); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	template<typename T>
	KVS::Map getMap(const std::string &key, const T &defaultValue)
	{
		try { return root.getMap(key, defaultValue); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	bool hasElement(const std::string &key) const {
		try { return root.hasElement(key); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	size_t size() const {
		try { return root.size(); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	Element& getPath(const std::string &path){
		try { return root.getPath(path); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	Element getPath(const std::string &path, const Element &defaultValue){
		try { return root.getPath(path, defaultValue); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	Element& upsertPath(const std::string &path){
		try { return root.upsertPath(path); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	template<typename T>
	Element& setDefaultPath(const std::string &path, const T &value){
		try { return root.setDefaultPath(path, value); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	template<typename T>
	Element& setContentPath(const std::string &path, const T &value){
		try { return root.setContentPath(path, value); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	template<typename T>
	Element& setDefault(const std::string &key, const T &value){
		//An empty root becomes a block
		if (root.isType(Type::None)) root = Ordered();
		root.touched = true;
		try { return root.setDefault(key, value); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	template<typename T>
	Element& setContent(const std::string &key, const T &value){
		//An empty root becomes a block
		if (root.isType(Type::None)) root = Ordered();
		root.touched = true;
		try { return root.setContent(key, value); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	/**
	 * Merge another KVS into this one. Keys that this one already has keep their values, except that blocks that both have are merged in the same way. Keys that only the other has are added
	 * @param other The KVS to merge into this one
	 */
	void merge(const KVS &other){
		//An empty root becomes a block
		if (root.isType(Type::None)) root = Ordered();
		try { root.merge(other.root); }
		catch (const std::exception &e) { reportFatalError(e.what()); throw; }
	}

	/**
	 * Get a value by path, converted to a boolean, integer, real number or string. If a default is given, it is returned when the
	 * path does not exist. A value that exists but cannot be converted, or a missing path without a default, throws an std::runtime_error naming the path.
	 */
	bool getBool(const std::string &path, bool defaultValue){
		Element *element;
		try { element = &root.getPath(path); }
		catch (const std::exception &) { return defaultValue; }
		try { return element->asBool(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	bool getBool(const std::string &path){
		try { return root.getPath(path).asBool(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	int64_t getInt(const std::string &path, int64_t defaultValue){
		Element *element;
		try { element = &root.getPath(path); }
		catch (const std::exception &) { return defaultValue; }
		try { return element->asInt(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	int64_t getInt(const std::string &path){
		try { return root.getPath(path).asInt(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	double getReal(const std::string &path, double defaultValue){
		Element *element;
		try { element = &root.getPath(path); }
		catch (const std::exception &) { return defaultValue; }
		try { return element->asReal(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	double getReal(const std::string &path){
		try { return root.getPath(path).asReal(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	std::string getString(const std::string &path, const std::string &defaultValue){
		Element *element;
		try { element = &root.getPath(path); }
		catch (const std::exception &) { return defaultValue; }
		try { return element->asString(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	std::string getString(const std::string &path){
		try { return root.getPath(path).asString(); }
		catch (const std::exception &e) { reportFatalError(path + ": " + e.what()); throw std::runtime_error(path + ": " + e.what()); }
	}

	};

	template <typename T, std::enable_if_t<std::is_integral_v<T>, int>>
		KVS::Element &KVS::Element::operator[](T index)
		{
			if (type == Type::Array)
				return get<Array&>()[index];
			if (type == Type::Ordered)
				return get<Ordered&>()[index];
			throw std::runtime_error("Root element is not an array or ordered collection");
		}

	inline KVS::Element &KVS::Element::operator[](const std::string key)
	{
		if (type == Type::Map)
			return get<Map&>()[key];
		if (type == Type::Ordered)
			return get<Ordered&>()[key];
		throw std::runtime_error("Root element is not a map or ordered collection");
	}

	/**
	 * Get an element by a path of keys separated by /. For example, getPath("a/b/c") would return the element at key "c" in the map at key "b" in the map at key "a" in the root element. If any key in the path does not exist, an std::runtime_error is thrown.
	 * @param path The path of keys separated by / to the element to retrieve.
	 * @return A reference to the element at the specified path.
	 */
	inline KVS::Element &KVS::Element::getPath(const std::string &path){
		touched = true;
		//If the path is empty, return the current element
		if (path == ""){
			return *this;
		}

		//Find the first / in the path
		size_t pos = path.find('/');

		std::string pathElement, remainingPath;
		if (pos == std::string::npos || pos == path.size() - 1) {
			pathElement = path;
			remainingPath = "";
		} else { 
			pathElement = path.substr(0, pos);
			remainingPath = path.substr(pos + 1);
		}

		if (pathElement=="") return (*this).getPath(remainingPath);
		
		//If it is not surrounded by [] then it is a key, otherwise it is an index
		if (pathElement.size() >= 2 && pathElement.front() == '[' && pathElement.back() == ']')
		{
			//Check that it is an ordered or array element
			if (type == Type::Array || type == Type::Ordered)
			{
				//Get the index between the []
				std::string indexStr = pathElement.substr(1, pathElement.size() - 2);
				size_t index = std::stoul(indexStr);
				return (*this)[index].getPath(remainingPath);
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected array or ordered collection for index access.");
			}
		} else {
			//Key access
			if (type == Type::Map || type == Type::Ordered)
			{
				return (*this)[pathElement].getPath(remainingPath);
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected map or ordered collection for key access.");
			}
		}
	}

	//Version of getPath that returns an element instead of a reference and takes a default value
	inline KVS::Element KVS::Element::getPath(const std::string &path, const Element &defaultValue)
	{
		touched = true;
		if (path == ""){
			return *this;
		}

		//Find the first / in the path
		size_t pos = path.find('/');

		std::string pathElement, remainingPath;
		if (pos == std::string::npos || pos == path.size() - 1) {
			pathElement = path;
			remainingPath = "";
		} else { 
			pathElement = path.substr(0, pos);
			remainingPath = path.substr(pos + 1);
		}

    if (pathElement=="") return (*this).getPath(remainingPath, defaultValue);

		//If it is not surrounded by [] then it is a key, otherwise it is an index
		if (pathElement.size() >= 2 && pathElement.front() == '[' && pathElement.back() == ']')
		{
			//Check that it is an ordered or array element
			if (type == Type::Array || type == Type::Ordered)
			{
				//Get the index between the []
				std::string indexStr = pathElement.substr(1, pathElement.size() - 2);
				size_t index = std::stoul(indexStr);
				try {
					return (*this)[index].getPath(path.substr(pos + 1));
				} catch (const std::runtime_error&) {
					return defaultValue;
				}
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected array or ordered collection for index access.");
			}
		} else {
			//Key access
			if (type == Type::Map || type == Type::Ordered)
			{
				try {
					return (*this)[pathElement].getPath(remainingPath);
				} catch (const std::runtime_error&) {
					return defaultValue;
				}
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected map or ordered collection for key access.");
			}
		}
		return KVS::Element();
	}

	/**
	 * Get an element by a path of keys separated by /, creating any key in the path that does not exist. A created key holds an empty element, and an empty element that a key is looked up in becomes an ordered collection. Every element on the path is marked as touched. Index access does not create elements.
	 * @param path The path of keys separated by / to the element to retrieve or create.
	 * @return A reference to the element at the specified path.
	 */
	inline KVS::Element &KVS::Element::upsertPath(const std::string &path){
		touched = true;
		//If the path is empty, return the current element
		if (path == ""){
			return *this;
		}

		//Find the first / in the path
		size_t pos = path.find('/');

		std::string pathElement, remainingPath;
		if (pos == std::string::npos || pos == path.size() - 1) {
			pathElement = path;
			remainingPath = "";
		} else {
			pathElement = path.substr(0, pos);
			remainingPath = path.substr(pos + 1);
		}

		if (pathElement=="") return (*this).upsertPath(remainingPath);

		//If it is not surrounded by [] then it is a key, otherwise it is an index
		if (pathElement.size() >= 2 && pathElement.front() == '[' && pathElement.back() == ']')
		{
			//Check that it is an ordered or array element
			if (type == Type::Array || type == Type::Ordered)
			{
				//Get the index between the []
				std::string indexStr = pathElement.substr(1, pathElement.size() - 2);
				size_t index = std::stoul(indexStr);
				return (*this)[index].upsertPath(remainingPath);
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected array or ordered collection for index access.");
			}
		} else {
			//An empty element becomes a block
			if (type == Type::None)
			{
				*this = Ordered();
				touched = true;
			}
			//Key access
			if (type == Type::Map || type == Type::Ordered)
			{
				return upsert(pathElement).upsertPath(remainingPath);
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected map or ordered collection for key access.");
			}
		}
	}

	/**
	 * Get an element by a path of keys separated by /, creating any key in the path that does not exist, as upsertPath does, except that nothing is marked as touched.
	 * A key that is created is marked as autoset in its metadata. Index access does not create elements.
	 * @param path The path of keys separated by / to the element to retrieve or create.
	 * @return A reference to the element at the specified path.
	 */
	inline KVS::Element &KVS::Element::upsertPathUntouched(const std::string &path){
		//If the path is empty, return the current element
		if (path == ""){
			return *this;
		}

		//Find the first / in the path
		size_t pos = path.find('/');

		std::string pathElement, remainingPath;
		if (pos == std::string::npos || pos == path.size() - 1) {
			pathElement = path;
			remainingPath = "";
		} else {
			pathElement = path.substr(0, pos);
			remainingPath = path.substr(pos + 1);
		}

		if (pathElement=="") return (*this).upsertPathUntouched(remainingPath);

		//If it is not surrounded by [] then it is a key, otherwise it is an index
		if (pathElement.size() >= 2 && pathElement.front() == '[' && pathElement.back() == ']')
		{
			//Check that it is an ordered or array element
			if (type == Type::Array || type == Type::Ordered)
			{
				//Get the index between the []
				std::string indexStr = pathElement.substr(1, pathElement.size() - 2);
				size_t index = std::stoul(indexStr);
				return (*this)[index].upsertPathUntouched(remainingPath);
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected array or ordered collection for index access.");
			}
		} else {
			//An empty element becomes a block. Only the value and type change, so its metadata is kept
			if (type == Type::None)
			{
				value = std::make_shared<Ordered>();
				type = Type::Ordered;
			}
			//Key access
			if (type == Type::Map || type == Type::Ordered)
			{
				const bool created = !hasElement(pathElement);
				Element &element = upsert(pathElement);
				if (created) element.setMetadata("autoset", "true");
				return element.upsertPathUntouched(remainingPath);
			}
			else
			{
				throw std::runtime_error("Element of incorrect type. Expected map or ordered collection for key access.");
			}
		}
	}

	/**
	 * Set the element at a path of keys separated by / to a value, unless it already holds one. Any key in the path that does not exist is created, as in upsertPathUntouched.
	 * Nothing is marked as touched, and a value that is set is marked as autoset in its metadata
	 * @param path The path of keys separated by / to the element to set.
	 * @param value The value to set the element to if it does not already hold one.
	 * @return A reference to the element at the specified path.
	 */
	template<typename T>
	KVS::Element &KVS::Element::setDefaultPath(const std::string &path, const T &value)
	{
		Element &element = upsertPathUntouched(path);
		if (element.isType(Type::None))
		{
			element = Element(value);
			element.setMetadata("autoset", "true");
		}
		return element;
	}

	/**
	 * Set the content of the element at a path of keys separated by / to a value, whether or not it already holds one. Only the value and type are changed, so the metadata and directive flag of an existing element are kept. Any key in the path that does not exist is created, as in upsertPath, and the element is marked as touched.
	 * @param path The path of keys separated by / to the element to set.
	 * @param value The value to set the element to.
	 * @return A reference to the element at the specified path.
	 */
	template<typename T>
	KVS::Element &KVS::Element::setContentPath(const std::string &path, const T &value)
	{
		Element &element = upsertPath(path);
		Element content(value);
		element.value = std::move(content.value);
		element.type = content.type;
		return element;
	}

	/**
	 * Set the element at a key to a value, unless it already holds one. A key that does not exist is created, as in upsert.
	 * A value that is set is marked as autoset in its metadata and is not marked as touched
	 * @param key The key of the element to set.
	 * @param value The value to set the element to if it does not already hold one.
	 * @return A reference to the element at the specified key.
	 */
	template<typename T>
	KVS::Element &KVS::Element::setDefault(const std::string &key, const T &value)
	{
		Element &element = upsert(key);
		if (element.isType(Type::None))
		{
			element = Element(value);
			element.setMetadata("autoset", "true");
		}
		return element;
	}

	/**
	 * Set the content of the element at a key to a value, whether or not it already holds one. Only the value and type are changed, so the metadata and directive flag of an existing element are kept. A key that does not exist is created, as in upsert, and the element is marked as touched.
	 * @param key The key of the element to set.
	 * @param value The value to set the element to.
	 * @return A reference to the element at the specified key.
	 */
	template<typename T>
	KVS::Element &KVS::Element::setContent(const std::string &key, const T &value)
	{
		Element &element = upsert(key);
		Element content(value);
		element.value = std::move(content.value);
		element.type = content.type;
		element.touched = true;
		return element;
	}

	/**
	 * Merge another element into this one. Keys that this element already has keep their values, except that blocks that both have are merged in the same way.
	 * Keys that only the other element has are added at the end. Both elements must be maps or ordered collections
	 * @param other The element to merge into this one
	 */
	inline void KVS::Element::merge(const Element &other)
	{
		const std::vector<std::pair<std::string, Element>> &otherData = other.type == Type::Map ? std::get<std::shared_ptr<Map>>(other.value)->data : std::get<std::shared_ptr<Ordered>>(other.value)->data;
		std::vector<std::pair<std::string, Element>> &data = type == Type::Map ? std::get<std::shared_ptr<Map>>(value)->data : std::get<std::shared_ptr<Ordered>>(value)->data;
		//The keys that this element has before the merge
		std::unordered_set<std::string> keys;
		for (const auto &pair : data) keys.insert(pair.first);
		for (const auto &pair : otherData)
		{
			if (keys.count(pair.first) == 0)
			{
				data.push_back(pair);
				continue;
			}
			Element &mine = (*this)[pair.first];
			const bool mineBlock = mine.type == Type::Map || mine.type == Type::Ordered;
			const bool otherBlock = pair.second.type == Type::Map || pair.second.type == Type::Ordered;
			if (mineBlock && otherBlock) mine.merge(pair.second);
		}
	}

	/**
	 * Get an element by key, returning a copy of the element. This means that if the element is a map or ordered collection then modifying the members of that element WILL modify the original element, but reassigning the element itself will not modify the original element. If the key does not exist, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve.
	 * @return A copy of the element associated with the specified key.
	 * @throws std::runtime_error If the key does not exist or if the root element is not a map or ordered collection.
	 */
	inline KVS::Element KVS::Element::getElement(const std::string key)
	{
		(*this)[key].touched = true;
		return (*this)[key];
	}

	/**
	 * Get an element by key, returning a copy of the element. If the key does not exist then an element with the provided value is returned. This element is NOT added to the collection and modifying it will not modify the original collection. This means that if the element is a map, array or ordered collection then modifying the members of that element WILL modify the original element, but reassigning the element itself will not modify the original element. If the root element is not a map or ordered collection, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve.
	 * @param value The default value to return if the key does not exist.
	 * @return A copy of the element associated with the specified key, or an element with the provided value if the key does not exist.
	 * @throws std::runtime_error If the root element is not a map or ordered collection.
	 */
	template<typename T>
		KVS::Element KVS::Element::getElement(const std::string key, const T&value){
			if (!hasElement(key))
				return Element(value);
			return getElement(key);
		}

	/**
	 * Get an element as an array by key. Returns an empty array if the key does not exist
	 * and throws if the element is not an array
	 * @param key The key of the element to retrieve as an array.
	 * @return The element associated with the specified key as an array, or an empty array if the key does not exist.
	 * @throws std::runtime_error If the element associated with the specified key is not an array or if the root element is not a map or ordered collection.
	 */
	inline KVS::Array KVS::Element::getArray(const std::string &key)
	{
		return this->getElement(key,KVS::Array()).template get<Array>();
	}

	/**
	 * Get an element as an array by key with a default value. If the key does not exist, an element with the provided default value is returned. This element is NOT added to the collection and modifying it will not modify the original collection. If the key exists but the element is not an array, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve as an array.
	 * @param defaultValue The default value to return if the key does not exist.
	 * @return The element associated with the specified key as an array, or an element with the provided default value if the key does not exist.
	 * @throws std::runtime_error If the key exists but the element associated with the specified key is not an array, or if the root
	 */
	template<typename T>
	KVS::Array KVS::Element::getArray(const std::string &key, const T &defaultValue)
	{
		return this->getElement(key, defaultValue).template get<Array>();
	}

	/**
	 * Get an element as an ordered collection by key. Returns an empty ordered collection if the key does not exist and throws if the element is not an ordered collection
	 * @param key The key of the element to retrieve as an ordered collection.
	 * @return The element associated with the specified key as an ordered collection, or an empty ordered collection if the key does not exist.
	 * @throws std::runtime_error If the element associated with the specified key is not an ordered collection or if the root element is not a map or
	 */
	inline KVS::Ordered KVS::Element::getOrdered(const std::string &key)
	{
		return this->getElement(key, KVS::Ordered()).template get<Ordered>();
	}

	/**
	 * Get an element as an ordered collection by key with a default value. If the key does not exist, an element with the provided default value is returned. This element is NOT added to the collection and modifying it will not modify the original collection. If the key exists but the element is not an ordered collection, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve as an ordered collection.
	 * @param defaultValue The default value to return if the key does not exist.
	 * @return The element associated with the specified key as an ordered collection, or an element with the provided default value if the key does not exist.
	 * @throws std::runtime_error If the key exists but the element associated with the specified key is not an ordered collection, or if the root element is not a map or ordered collection.
	 */
	template<typename T>
	KVS::Ordered KVS::Element::getOrdered(const std::string &key, const T &defaultValue)
	{
		return this->getElement(key, defaultValue).template get<Ordered>();
	}

	/**
	 * Get an element as a map by key. Returns an empty map if the key does not exist and throws if the element is not a map
	 * @param key The key of the element to retrieve as a map.
	 * @return The element associated with the specified key as a map, or an empty map if the key does not exist.
	 * @throws std::runtime_error If the element associated with the specified key is not a map or if the root element is not a map or ordered collection.
	 */
	inline KVS::Map KVS::Element::getMap(const std::string &key)
	{
		return this->getElement(key, KVS::Map()).template get<Map>();
	}

	/**
	 * Get an element as a map by key with a default value. If the key does not exist, an element with the provided default value is returned. This element is NOT added to the collection and modifying it will not modify the original collection. If the key exists but the element is not a map, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve as a map.
	 * @param defaultValue The default value to return if the key does not exist.
	 * @return The element associated with the specified key as a map, or an element with the provided default value if the key does not exist.
	 * @throws std::runtime_error If the key exists but the element associated with the specified key is not a map, or if the root
	 */
	template<typename T>
	KVS::Map KVS::Element::getMap(const std::string &key, const T &defaultValue)
	{
		return this->getElement(key, defaultValue).template get<Map>();
	}

	/**
	 * Get an element as a map by key with a default value. If the key does not exist, an element with the provided default value is returned. This element is NOT added to the collection and modifying it will not modify the original collection. If the key exists but the element is not a map, an std::runtime_error is thrown.
	 * @param key The key of the element to retrieve as a map.
	 * @param defaultValue The default value to return if the key does not exist.
	 * @return The element associated with the specified key as a map, or an element with the provided default value if the key does not exist.
	 * @throws std::runtime_error If the key exists but the element associated with the specified key is not a map, or if the root element is not a map or ordered collection.
	 */

	inline KVS::Element &KVS::Element::operator[](const char *key)
	{
		return (*this)[std::string(key)];
	}

	inline KVS::Element &KVS::Element::upsert(const std::string &key)
	{
		if (type == Type::Map)
			return get<Map&>().upsert(key);
		if (type == Type::Ordered)
			return get<Ordered&>().upsert(key);
		throw std::runtime_error("Root element is not a map or ordered collection");
	}

	inline KVS::Element &KVS::Element::upsert(const char *key)
	{
		return upsert(std::string(key));
	}

	inline bool KVS::Element::operator==(const KVS::Element &other) const
	{
		if (type != other.type)
			return false;
		switch (type)
		{
			case Type::Int32:
				return std::get<int32_t>(value) == std::get<int32_t>(other.value);
			case Type::Int64:
				return std::get<int64_t>(value) == std::get<int64_t>(other.value);
			case Type::Float:
				return std::get<float>(value) == std::get<float>(other.value);
			case Type::Double:
				return std::get<double>(value) == std::get<double>(other.value);
			case Type::Int32Range:
				return std::get<ranged<int32_t>>(value) == std::get<ranged<int32_t>>(other.value);
			case Type::Int64Range:
				return std::get<ranged<int64_t>>(value) == std::get<ranged<int64_t>>(other.value);
			case Type::FloatRange:
				return std::get<ranged<float>>(value) == std::get<ranged<float>>(other.value);
			case Type::DoubleRange:
				return std::get<ranged<double>>(value) == std::get<ranged<double>>(other.value);
			case Type::String:
				return std::get<std::string>(value) == std::get<std::string>(other.value);
			case Type::Bool:
				return std::get<bool>(value) == std::get<bool>(other.value);
			case Type::Array:
				return std::get<std::shared_ptr<Array>>(value)->elements == std::get<std::shared_ptr<Array>>(other.value)->elements;
			case Type::Map:
				return std::get<std::shared_ptr<Map>>(value)->data == std::get<std::shared_ptr<Map>>(other.value)->data;
			default: // Undefined / monostate
				return true;
		}
	}

	template <typename T>
		std::conditional_t<std::is_reference_v<T>, T, std::remove_cv_t<T>> KVS::Element::get()
		{
			this->touched=true;
			if constexpr (!std::is_reference_v<T>)
			{
				//A copy of the value, except that any scalar can be got as a string
				if constexpr (std::is_same_v<std::remove_cv_t<T>, std::string>)
					return asString();
				else
					return get<T&>();
			}
			else
			{
				using U = std::remove_reference_t<T>;
				if constexpr (std::is_same_v<U, Array>)
				{
					return *std::get<std::shared_ptr<Array>>(value);
				}
				else if constexpr (std::is_same_v<U, Map>)
				{
					return *std::get<std::shared_ptr<Map>>(value);
				}
				else if constexpr (std::is_same_v<U, Ordered>)
				{
					return *std::get<std::shared_ptr<Ordered>>(value);
				}
				else if constexpr(std::is_same_v<U, std::string>)
				{
					return std::get<std::string>(value);
				}
				else if constexpr(std::is_same_v<U, ranged<int32_t>>)
				{
					return std::get<ranged<int32_t>>(value);
				}
				else if constexpr(std::is_same_v<U, ranged<int64_t>>)
				{
					return std::get<ranged<int64_t>>(value);
				}
				else if constexpr(std::is_same_v<U, ranged<float>>)
				{
					return std::get<ranged<float>>(value);
				}
				else if constexpr(std::is_same_v<U, ranged<double>>)
				{
					return std::get<ranged<double>>(value);
				}
				else if constexpr(std::is_arithmetic_v<U>)
				{
					return std::get<U>(value);
				}
				throw std::bad_variant_access();
			}
		}

	template<typename T>
	T KVS::Element::getValue() const
	{
		  this->touched=true;
				if constexpr (std::is_same_v<T, Array>)
			{
				return *std::get<std::shared_ptr<Array>>(value);
			}
			else if constexpr (std::is_same_v<T, Map>)
			{
				return *std::get<std::shared_ptr<Map>>(value);
			}
			else if constexpr (std::is_same_v<T, Ordered>)
			{
				return *std::get<std::shared_ptr<Ordered>>(value);
			}
			else if constexpr(std::is_same_v<T, std::string>)
			{
				return std::get<std::string>(value);
			}
			else if constexpr(std::is_same_v<T, ranged<int32_t>>)
			{
				return std::get<ranged<int32_t>>(value);
			}
			else if constexpr(std::is_same_v<T, ranged<int64_t>>)
			{
				return std::get<ranged<int64_t>>(value);
			}
			else if constexpr(std::is_same_v<T, ranged<float>>)
			{
				return std::get<ranged<float>>(value);
			}
			else if constexpr(std::is_same_v<T, ranged<double>>)
			{
				return std::get<ranged<double>>(value);
			}
			else if constexpr(std::is_arithmetic_v<T>)
			{
				return tryConvert<T>();
			}
			throw std::bad_variant_access();
	}

	template<typename T>
	T KVS::Element::getDefault(const T &defaultValue) const
		{
			try {
				return get<T>();
			} catch (const std::bad_variant_access&) {
				return defaultValue;
			} catch (const std::runtime_error&) {
				return defaultValue;
			}
		}

	template<typename T>
		std::conditional_t<std::is_reference_v<T>, const std::remove_reference_t<T> &, std::remove_cv_t<T>> KVS::Element::get() const
		{
			this->touched=true;
			if constexpr (!std::is_reference_v<T>)
			{
				//A copy of the value, except that any scalar can be got as a string
				if constexpr (std::is_same_v<std::remove_cv_t<T>, std::string>)
					return asString();
				else
					return get<T&>();
			}
			else
			{
				using U = std::remove_cv_t<std::remove_reference_t<T>>;
				if constexpr (std::is_same_v<U, Array>)
				{
					return *std::get<std::shared_ptr<Array>>(value);
				}
				else if constexpr (std::is_same_v<U, Map>)
				{
					return *std::get<std::shared_ptr<Map>>(value);
				}
				else if constexpr (std::is_same_v<U, Ordered>)
				{
					return *std::get<std::shared_ptr<Ordered>>(value);
				}
				else
				{
					return std::get<U>(value);
				}
			}
		}

	// Output operator for KVS
	inline std::ostream &operator<<(std::ostream &os, const KVS &kvs)
	{
		os << kvs.root;
		return os;
	}
}
#endif
