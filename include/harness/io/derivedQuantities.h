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
#ifndef IO_DERIVEDQUANTITIES_H
#define IO_DERIVEDQUANTITIES_H

#include <array>
#include <tuple>
#include "harness.h"
#include "types.h"

namespace SAMS::IO
{

/**
 * @brief Sets each element of an array to the value of a calculator at its indices. It is outside derivedQuantity because a kernel cannot take a type that is private to a class
 * @tparam T_data The type of the array
 * @tparam R The rank of the array
 * @tparam T_calculator The type of the calculator
 */
template<typename T_data, int R, typename T_calculator>
struct derivedQuantityFill{
	portableWrapper::portableArray<T_data, R, portableWrapper::arrayTags::accelerated> array;
	T_calculator calculator;

	template<typename... T_indices>
	FUNCTORMETHODPREFIX INLINE void operator()(T_indices... indices) const
	{
		array(indices...) = calculator(indices...);
	}
};

/**
 * @brief Calculates a quantity on the device over the local domain of a set of axes and copies it to a host array for writing
 * @details The arrays are kept between calculations and reallocated when the local domain changes
 * @tparam T_data The type of the quantities
 * @tparam R The rank of the quantities
 */
template<typename T_data, int R>
class derivedQuantity{
	SAMS::harness &h;
	//Owns the device and host arrays
	portableWrapper::portableArrayManager manager;
	portableWrapper::portableArray<T_data, R, portableWrapper::arrayTags::accelerated> device;
	portableWrapper::portableArray<T_data, R, portableWrapper::arrayTags::host> host;
	//The local domain of each axis of the arrays
	std::array<SIGNED_INDEX_TYPE, R> lower{}, upper{};
	bool allocated = false;

	public:
	/**
	 * @brief Constructor
	 * @param h The harness
	 */
	derivedQuantity(SAMS::harness &h) : h(h) {}

	/**
	 * @brief Calculate a quantity at every point of the local domain of a set of axes on the device, and copy it to the host array
	 * @param axes The name and staggering of the axis of each dimension
	 * @param calculator Gives the value of the quantity at the indices of a point. It is called on the device
	 */
	template<typename T_calculator>
	void operator()(const simulationAxis (&axes)[R], T_calculator calculator){
		std::array<portableWrapper::Range, R> ranges;
		bool same = allocated;
		for (int i = 0; i < R; ++i){
			ranges[i] = h.axisRegistry.getLocalDomainRange(axes[i].name, axes[i].stagger);
			same = same && lower[i] == ranges[i].lower_bound && upper[i] == ranges[i].upper_bound;
			lower[i] = ranges[i].lower_bound;
			upper[i] = ranges[i].upper_bound;
		}
		//The arrays of the last quantity are reused if they have the same local domain
		if (!same){
			manager.allocate(device, lower.data(), upper.data());
			manager.allocate(host, lower.data(), upper.data());
			allocated = true;
		}
		std::apply([&](auto... range){ portableWrapper::applyKernel(derivedQuantityFill<T_data, R, T_calculator>{device, calculator}, range...); }, ranges);
		portableWrapper::fence();
		manager.copyData(host, device);
	}

	/**
	 * @brief The host array of the quantity, which holds only the local domain
	 */
	const T_data *data() const {
		return host.data();
	}
};

} // namespace SAMS::IO

#endif
