#ifndef SAMS_KVS_H
#define SAMS_KVS_H

#include <string>
#include <stdexcept>
#include <cstdint>
#include <unordered_map>

#include "URI.h"

namespace SAMS {

    enum class kvsValueHint{
        NONE,
        BOOL,
        INT,
        DOUBLE,
        STRING,
        URI
    };

    class kvsItem {
        private:
        //The value is stored as a string
        std::string value="";
        //To allow blocks within blocks, we have a map of sub-items
        std::unordered_map<std::string, kvsItem> subItems;
        //A hint to the type of the value
        kvsValueHint hint = kvsValueHint::NONE;

        void checkHint(kvsValueHint expected) const {
            if (hint != kvsValueHint::NONE && hint != expected) {
                #ifdef SAMS_KVS_STRICT_HINTS
                throw std::runtime_error("Error: Value hint mismatch. Expected " + std::to_string(static_cast<int>(expected)) + " but got " + std::to_string(static_cast<int>(hint)));
                #elif defined(SAMS_KVS_WARN_HINTS)
                std::cerr << "Warning: Value hint mismatch. Expected " + std::to_string(static_cast<int>(expected)) + " but got " + std::to_string(static_cast<int>(hint)) << std::endl;
                #endif
            }
        }

        kvsItem& getItem(const std::string& key) {
            auto it = subItems.find(key);
            if (it == subItems.end()) {
                throw std::runtime_error("Error: Key not found: " + key);
            }
            return it->second;
        }

        std::string operator=(const std::string& val) {
            return value = val;
        }

        void addHint(kvsValueHint h) {
            hint = h;
        }
        
        void setValue(const std::string& val, kvsValueHint h = kvsValueHint::NONE) {
            value = val;
            hint = h;
        }

        public:
        kvsItem() = default;
        kvsItem(const std::string& val, kvsValueHint h = kvsValueHint::NONE) : value(val), hint(h) {}
        kvsItem(const kvsItem& other) = default;
        kvsItem(kvsItem&& other) noexcept = default;
        kvsItem& operator=(const kvsItem& other) = default;
        kvsItem& operator=(kvsItem&& other) noexcept = default;
        ~kvsItem() = default;

        /**
         * Convert the stored string to a boolean.
         * Recognizes "true", "T", "1" as true and "false", "F", "0" as false.
         * Throws an exception for any other value.
         * @return The boolean representation of the stored string.
         * @throws std::runtime_error if the string cannot be converted to a boolean.
         */
        bool asBool() const {

            checkHint(kvsValueHint::BOOL);
            
            // Convert to lower case for comparison
            std::string lowerValue;
            lowerValue.resize(value.size());
            std::transform(value.begin(), value.end(), lowerValue.begin(), ::tolower);
            if (lowerValue == "true" || lowerValue == "t" || lowerValue == "1") {
                return true;
            } else if (lowerValue == "false" || lowerValue == "f" || lowerValue == "0") {
                return false;
            } else {
                throw std::runtime_error("Error: Cannot convert string to bool: " + value);
            }
        }

        /**
         * Convert the stored string to an integer.
         * Uses std::stoll to perform the conversion.
         * @return The integer representation of the stored string.
         * @throws std::runtime_error if the string cannot be converted to an integer.
         * @throws std::out_of_range if the converted value is out of the range of representable values for int64_t.
         */
        int64_t asInt() const {
            checkHint(kvsValueHint::INT);
            try {
                return std::stoll(value);
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Error: Cannot convert string to int: " + value);
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Error: Integer value out of range: " + value);
            }
        }

        /**
         * Convert the stored string to a double.
         * Uses std::stod to perform the conversion.
         * @return The double representation of the stored string.
         * @throws std::runtime_error if the string cannot be converted to a double.
         * @throws std::out_of_range if the converted value is out of the range of representable values for double.
         */
        double asDouble() const {
            checkHint(kvsValueHint::DOUBLE);
            try {
                return std::stod(value);
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Error: Cannot convert string to double: " + value);
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Error: Double value out of range: " + value);
            }
        }

        /**
         * Return the string as a string
         * @return The string representation of the stored value.
         */
        std::string asString() const {
            //Explicitly do not check the hint here as any type can be represented as a string
            //checkHint(kvsValueHint::STRING);
            return value;
        }

        /**
         * Return a URI representation of the stored string
         * @return The URI representation of the stored value.
         * @throws std::runtime_error if the string cannot be parsed as a URI.
         */
        URI asURI() const {
            checkHint(kvsValueHint::URI);
            return URI::parse(value);
        }

        /**
         * Access a sub-item by key.
         * Throws an exception if the key does not exist.
         * @param key The key of the sub-item to access.
         * @return A reference to the sub-item.
         * @throws std::runtime_error if the key does not exist.
         */
        const kvsItem& operator[](const std::string& key) const {
            auto it = subItems.find(key);
            if (it == subItems.end()) {
                throw std::runtime_error("Error: Key not found: " + key);
            }
            return it->second;
        }
    };


    class KVS {
        private:
        std::unordered_map<std::string, kvsItem> items;

        kvsItem& getItem(const std::string& key) {
            auto it = items.find(key);
            if (it == items.end()) {
                throw std::runtime_error("Error: Key not found: " + key);
            }
            return it->second;
        }
        public:

        const kvsItem& operator[](const std::string& key) const {
            auto it = items.find(key);
            if (it == items.end()) {
                throw std::runtime_error("Error: Key not found: " + key);
            }
            return it->second;
        }

    };

        
}

#endif //SAMS_KVS_H