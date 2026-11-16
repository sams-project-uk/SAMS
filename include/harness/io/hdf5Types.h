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
#ifndef IO_HDF5TYPES_H
#define IO_HDF5TYPES_H
#if __has_include(<hdf5.h>)
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <hdf5.h>
#include "typeRegistry.h"

namespace SAMS::IO
{

/**
 * @brief The HDF5 types of the C++ types that can be read and written
 * @details The compound types for complex data, with the real part r and the imaginary part i, are created when they are first used and closed by close
 */
class hdf5Types{
	hid_t complexFloatType = H5I_INVALID_HID;
	hid_t complexDoubleType = H5I_INVALID_HID;

	/**
	 * @brief Create the compound type for complex data
	 * @tparam T_component The type of the real and imaginary parts
	 */
	template<typename T_component>
	hid_t createComplexType(){
		hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<T_component>));
		if (type < 0) {
			throw std::runtime_error("Could not create complex datatype");
		}
		//std::complex holds the real part followed by the imaginary part
		if (H5Tinsert(type, "r", 0, get<T_component>()) < 0 || H5Tinsert(type, "i", sizeof(T_component), get<T_component>()) < 0) {
			throw std::runtime_error("Could not create the parts of complex datatype");
		}
		return type;
	}

	public:
	/**
	 * @brief Get the HDF5 type of a C++ type
	 * @tparam T_data The C++ type
	 */
	template<typename T_data>
	hid_t get(){
		if constexpr (std::is_same_v<T_data, float>) {
			return H5T_NATIVE_FLOAT;
		} else if constexpr (std::is_same_v<T_data, double>) {
			return H5T_NATIVE_DOUBLE;
		} else if constexpr (std::is_same_v<T_data, int8_t>) {
			return H5T_NATIVE_INT8;
		} else if constexpr (std::is_same_v<T_data, uint8_t>) {
			return H5T_NATIVE_UINT8;
		} else if constexpr (std::is_same_v<T_data, int16_t>) {
			return H5T_NATIVE_INT16;
		} else if constexpr (std::is_same_v<T_data, uint16_t>) {
			return H5T_NATIVE_UINT16;
		} else if constexpr (std::is_same_v<T_data, int32_t>) {
			return H5T_NATIVE_INT32;
		} else if constexpr (std::is_same_v<T_data, uint32_t>) {
			return H5T_NATIVE_UINT32;
		} else if constexpr (std::is_same_v<T_data, int64_t>) {
			return H5T_NATIVE_INT64;
		} else if constexpr (std::is_same_v<T_data, uint64_t>) {
			return H5T_NATIVE_UINT64;
		} else if constexpr (std::is_same_v<T_data, std::complex<float>>) {
			if (complexFloatType == H5I_INVALID_HID) complexFloatType = createComplexType<float>();
			return complexFloatType;
		} else if constexpr (std::is_same_v<T_data, std::complex<double>>) {
			if (complexDoubleType == H5I_INVALID_HID) complexDoubleType = createComplexType<double>();
			return complexDoubleType;
		} else {
			static_assert(std::is_same_v<T_data, float>, "Type cannot be read or written with HDF5");
		}
	}

	/**
	 * @brief Close the compound types that have been created
	 */
	void close(){
		for (hid_t *type : {&complexFloatType, &complexDoubleType}) {
			if (*type != H5I_INVALID_HID) {
				if (H5Tclose(*type) < 0) {
					throw std::runtime_error("Could not close complex datatype");
				}
				*type = H5I_INVALID_HID;
			}
		}
	}
};

/**
 * @brief Get the type in the type registry of an HDF5 type
 * @param type The HDF5 type
 * @return The type handle, which is not valid if the type is not one that can be read
 */
inline SAMS::typeHandle typeHandleFromHDF5(hid_t type){
	const size_t size = H5Tget_size(type);
	switch (H5Tget_class(type)) {
		case H5T_FLOAT:
			if (size == 4) return SAMS::typeID::TYPE_FLOAT;
			if (size == 8) return SAMS::typeID::TYPE_DOUBLE;
			break;
		case H5T_INTEGER:
			if (H5Tget_sign(type) == H5T_SGN_NONE) {
				if (size == 1) return SAMS::typeID::TYPE_UINT8;
				if (size == 2) return SAMS::typeID::TYPE_UINT16;
				if (size == 4) return SAMS::typeID::TYPE_UINT32;
				if (size == 8) return SAMS::typeID::TYPE_UINT64;
			} else {
				if (size == 1) return SAMS::typeID::TYPE_INT8;
				if (size == 2) return SAMS::typeID::TYPE_INT16;
				if (size == 4) return SAMS::typeID::TYPE_INT32;
				if (size == 8) return SAMS::typeID::TYPE_INT64;
			}
			break;
		case H5T_COMPOUND: {
			//Complex data is a compound of the real part r and the imaginary part i
			if (H5Tget_nmembers(type) != 2) break;
			char *real = H5Tget_member_name(type, 0);
			char *imaginary = H5Tget_member_name(type, 1);
			const bool parts = std::string(real) == "r" && std::string(imaginary) == "i";
			H5free_memory(real);
			H5free_memory(imaginary);
			if (parts && size == 8) return SAMS::typeID::TYPE_COMPLEX_FLOAT;
			if (parts && size == 16) return SAMS::typeID::TYPE_COMPLEX_DOUBLE;
			break;
		}
		default:
			break;
	}
	return SAMS::typeHandle();
}

} // namespace SAMS::IO
#endif
#endif
