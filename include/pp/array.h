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
#ifndef ARRAY_H
#define ARRAY_H

#include "defs.h"
#include "utils.h"
#include "callableTraits.h"
#include <limits>

namespace portableWrapper
{

	//Maximum size of the Name for an array
	constexpr size_t MAX_ARRAY_NAME_SIZE = 256;

	class portableArrayManager;

	namespace detail{
		/**
		 * Get the ranges corresponding to the whole of a portableArray
		 */
		template <int level = 0, typename T, int rank, arrayTags T_tag>
		UNREPEATED void arrayToRanges(const portableArray<T, rank, T_tag> &array, std_N_ary_tuple_type_t<Range, rank> &tuple)
		{
			std::get<level>(tuple) = Range(array.lower_bound[level], array.upper_bound[level]);
			if constexpr (level + 1 < rank)
			{
				arrayToRanges<level + 1>(array, tuple);
			}
		}

		/**
		 * Get the ranges corresponding to the whole of a portableArray with zero-based indexing
		 */
		template <int level = 0, typename T, int rank, arrayTags T_tag>
		UNREPEATED void arrayToRangesZB(const portableArray<T, rank, T_tag> &array, std_N_ary_tuple_type_t<Range, rank> &tuple)
		{
			std::get<level>(tuple) = Range(0, array.size[level] - 1);
			if constexpr (level + 1 < rank)
			{
				arrayToRangesZB<level + 1>(array, tuple);
			}
		}	
	}

	/**
	 * Array wrapper class for parallel operations
	 * This class provides a way to wrap an array of type T with a specified rank.
	 * It allows for parallel operations on the array using the specified ranges.
	 * Because of CUDA requirements IT MUST BE TRIVIALLY COPYABLE
	 * and have a trivial destructor.
	 * This is why we later on have the factory class to create instances of this class, and
	 * delete them when they are no longer needed.
	 */
	template <typename T, int i_rank, arrayTags tag>
	class portableArray
	{
		public:
		DEVICEPREFIX INLINE static constexpr bool rowMajor()
		{
			if constexpr (tag == arrayTags::host)
			{
				return true; // Host arrays are row-major
			}
			else
			{
				#ifdef USE_KOKKOS
				#if defined(KOKKOS_CUDA) || defined(KOKKOS_HIP)
				// Column-major for CUDA and HIP
				//NB this is a Kokkos limitation, not a portableWrapper one
				//Our CUDA and HIP backends work with row-major arrays internally
				//But Kokkos looses performance for row-major arrays on these platforms
				//No idea why
				return false;
				#else
				return true; // Default to row-major
				#endif
				#endif
				return true;

			}
		}
		friend class portableArrayManager;
		template <int level2, typename T2, int rank2, arrayTags T_tag2>
		friend void detail::arrayToRanges(const portableArray<T2, rank2, T_tag2> &, std_N_ary_tuple_type_t<Range, rank2> &);
		template <int level2, typename T2, int rank2, arrayTags T_tag2>
		friend void detail::arrayToRangesZB(const portableArray<T2, rank2, T_tag2> &, std_N_ary_tuple_type_t<Range, rank2> &);

		template <typename T2, int rank2, arrayTags T_tag2>
		friend class portableArray;

	private:
		using type = T;
		using selectRefType = T &;
		static constexpr int rank = i_rank;
		static constexpr arrayTags arrayTag = tag;
		T *data_ = nullptr;
		portableArrayManager *manager = nullptr;
		bool ownsData = true; // Indicates if this wrapper owns the data pointer
		bool managed = false;
		bool contiguous = true;
		SIZE_TYPE elements = 0;
		SIZE_TYPE extent = 0;
		SIGNED_INDEX_TYPE lower_bound[rank];
		SIGNED_INDEX_TYPE upper_bound[rank];
		SIGNED_INDEX_TYPE stride[rank];
		SIGNED_INDEX_TYPE size[rank];
		SIGNED_INDEX_TYPE offset=0;
		char Name[MAX_ARRAY_NAME_SIZE] = "";

		void manage(portableArrayManager *mgr)
		{
			manager = mgr;
		}

		template<int level = 0, typename T_current, typename... T_otherRanges>
		void printIndex(T_current c, T_otherRanges... otherRanges) const
		{
			constexpr int rlevel = level;
			std::cout << "Level " << rlevel << ": c=" << c << ", lb=" << lower_bound[rlevel] << ", ub=" << upper_bound[rlevel] << ", stride=" << stride[rlevel] << "\n";
			if constexpr (sizeof...(otherRanges) > 0)
			{
				printIndex<level + 1>(otherRanges...);
			}
		}

		/**
		 * Builds the index into the underlying data array
		 */
		template <int level = 0, typename T_current, typename... T_others>
		DEVICEPREFIX INLINE SIGNED_INDEX_TYPE buildIndex(T_current c, T_others... r) const
		{
			static_assert(sizeof...(r) + level + 1 == rank, "Number of indices must match the rank of the portable array.");
			constexpr int rlevel = level;
			SIGNED_INDEX_TYPE index = c*stride[rlevel] ;
			#ifdef ARRAY_BOUNDS_CHECKING
			SIGNED_INDEX_TYPE lb = lower_bound[rlevel];
            if (c < lb || c > upper_bound[rlevel]) {
				if (Name[0] != '\0') {
					std::cout << "Index out of bounds in array " << Name << ": " << c << " not in [" << lb << ", " << upper_bound[rlevel] << "]\n";
				} else {
                	std::cout << "Index out of bounds: " << c << " not in [" << lb << ", " << upper_bound[rlevel] << "]\n";
				}
				throw std::out_of_range("Index out of bounds");
            }
			#endif
			if constexpr (sizeof...(r) > 0)
			{
				index += buildIndex<level + 1>(r...);
			}
			if constexpr(level == 0){
				index += offset;
				/*#ifdef ARRAY_BOUNDS_CHECKING
				if (index >= elements) {
					if (Name[0] != '\0') {
						std::cout << "Calculated index out of bounds in array " << Name << ": " << index << " not in [0, " << elements - 1 << "]\n";
					} else {
						std::cout << "Calculated index out of bounds: " << index << " not in [0, " << elements - 1 << "]\n";
					}
					throw std::out_of_range("Calculated index out of bounds");
				}
				#endif*/
			}
			return index;
		}

		template <int level = 0, typename T_current, typename... T_others>
		DEVICEPREFIX INLINE UNSIGNED_INDEX_TYPE buildIndexZB(T_current c, T_others... r) const
		{
			constexpr int rlevel = level;
			UNSIGNED_INDEX_TYPE index = c * stride[rlevel];
			#ifdef ARRAY_BOUNDS_CHECKING
			SIGNED_INDEX_TYPE lb = lower_bound[rlevel];
            if (c < 0 || c > size[rlevel]) {
				if (Name[0] != '\0') {
					std::cout << "Index out of bounds in array " << Name << ": " << c << " not in [" << 0 << ", " << size[rlevel] << "]\n";
				} else {
                	std::cout << "Index out of bounds: " << c << " not in [" << 0 << ", " << size[rlevel] << "]\n";
				}
				throw std::out_of_range("Index out of bounds");
            }
			#endif
			if constexpr (sizeof...(r) > 0)
			{
				index += buildIndexZB<level + 1>(r...);
			}
			#ifdef ARRAY_BOUNDS_CHECKING
			/*if (index >= elements) {
				if (Name[0] != '\0') {
					std::cout << "Calculated index out of bounds in array " << Name << ": " << index << " not in [0, " << elements - 1 << "]\n";
				} else {
					std::cout << "Calculated index out of bounds: " << index << " not in [0, " << elements - 1 << "]\n";
				}
				throw std::out_of_range("Calculated index out of bounds");
			}*/
			#endif
			return index;
		}

		/**
		 * Get the last element index in the array
		 */
		DEVICEPREFIX size_t getLastElement() const
		{
			//Because we might be a slice we can't just use elements-1
			size_t lastIndex = 0;
			for (int i = 0; i < rank; ++i)
			{
				lastIndex += (upper_bound[i] - lower_bound[i]) * stride[i];
			}
			return lastIndex;
		}

		template<int level = 0>
		DEVICEPREFIX void getUBTuple(portableWrapper::N_ary_tuple_type_t<SIGNED_INDEX_TYPE,rank> &tuple) const
		{
			get<level>(tuple) = upper_bound[level];
			if constexpr (level + 1 < rank)
			{
				getUBTuple<level + 1>(tuple);
			}
		}

		/**
		 * Is this array a pointer to Managed (host/device) memory?
		 * Set by the allocator
		 */
		void setManaged(bool state)
		{
			//Host arrays can never be managed
			managed = state && (tag != arrayTags::host);
		}

		/**
		 * Is this array contiguous in memory?
		 */
		void setContiguous(bool state)
		{
			contiguous = state;
		}

		/**
		 * During setup, calculates the strides for each dimension to
		 * get to the next element in a given dimension.
		 */
		template <int level = 0>
		DEVICEPREFIX void calculateStrides()
		{
			static constexpr int applyLevel = rowMajor() ? rank - level - 1 : level;
			static constexpr int delta = rowMajor() ? 1 : -1;
			if constexpr (level > 0)
			{
				stride[applyLevel] = stride[applyLevel + delta] * size[applyLevel + delta];
			}
			else
			{
				stride[applyLevel] = 1;
			}
			offset -= lower_bound[applyLevel] * stride[applyLevel];
			if constexpr (level + 1 < rank)
			{
				calculateStrides<level + 1>();
			}
		}

		/**
		 * Unpacks the ranges and calculates the sizes
		 * Doesn't calculate the stride because it is easier to do it
		 * in a separate function that can be called after all sizes are set.
		 */
		template <int level = 0, typename T_current, typename... T_others>
		DEVICEPREFIX void buildSizes(T_current c, T_others... r)
		{
			if constexpr (level == 0)
			{
				elements = 1; // Reset elements for each allocation
				extent = 1;
				offset = 0;
			}
			if constexpr (std::is_same_v<T_current, Range>)
			{
				lower_bound[level] = c.lower_bound;
				upper_bound[level] = c.upper_bound;
				size[level] = c.upper_bound - c.lower_bound + 1;
			}
			else
			{
				lower_bound[level] = 1;
				upper_bound[level] = c;
				size[level] = c;
			}
			elements *= size[level];
			extent *= size[level];
			if constexpr (sizeof...(r) > 0)
			{
				buildSizes<level + 1>(r...);
			}
			if constexpr (level == 0)
			{
				calculateStrides<0>();
				extent = getLastElement();
			}
		}

		template <int drank, int slevel = 0, int dlevel = 0, typename T_tuple>
		DEVICEPREFIX INLINE void sliceToImpl(portableArray<T, drank, arrayTag> &other, size_t &startOffset, T_tuple indices) const
		{
			constexpr int rlevel = drank - dlevel - 1;
			constexpr int rslevel = rank - slevel - 1;
			if constexpr (dlevel == 0)
			{
				other.elements = 1;
				other.extent = 1;
			}
			//Is this slice index an integer (i.e. sliceing out a dimension) or a Range?
			if constexpr (std::is_integral_v<std::remove_reference_t<TUPLEELEMENT<rslevel, T_tuple>>>)
			{
				//It is an integer (slice out this dimension)
				startOffset += (GET<rslevel>(indices) - this->lower_bound[rslevel]) * this->stride[rslevel];
				//If we are still building the destination array then set the stride
				if constexpr (rlevel >= 0)
					other.stride[rlevel] *= this->stride[rslevel];
				if constexpr (rslevel > 0)
				{
					sliceToImpl<drank, slevel + 1, dlevel>(other, startOffset, indices);
				}
			}
			else //Or a Range
			{
				int64_t lb = GET<rslevel>(indices).lower_bound;
				int64_t ub = GET<rslevel>(indices).upper_bound;
				if (lb == INT64_MIN)
					lb = this->lower_bound[rslevel];
				if (ub == INT64_MAX)
					ub = this->upper_bound[rslevel];

				startOffset += (lb - this->lower_bound[rslevel]) * this->stride[rslevel];
				if constexpr (rlevel >= 0)
				{
					other.lower_bound[rlevel] = 0;
					other.upper_bound[rlevel] = ub - lb;
					other.stride[rlevel] = this->stride[rslevel];
					other.size[rlevel] = ub - lb + 1;
					other.elements *= other.size[rlevel];
					other.extent *= other.size[rlevel] * other.stride[rlevel];
					if constexpr (rslevel > 0)
					{
						sliceToImpl<drank, slevel + 1, dlevel + 1>(other, startOffset, indices);
					}
				}
				else
				{
					if constexpr (rslevel > 0)
					{
						sliceToImpl<drank, slevel + 1, dlevel>(other, startOffset, indices);
					}
				}
			}
		}



		template <int dRank, typename... T_indices>
		DEVICEPREFIX INLINE void sliceTo(portableArray<T, dRank, arrayTag> &other, T_indices... indices) const
		{
			size_t startOffset = 0;
			for (int i = 0; i < dRank; ++i)
			{
				// Initialise the stride to 1, so that if needed
				other.stride[i] = 1;
			}
			sliceToImpl<countRanges<T_indices...>(), 0, 0>(other, startOffset, MAKETUPLE(indices...));
			other.data_ = this->data_ + startOffset; // Set the data pointer to the correct offset
			other.ownsData = false;					 // We do not own the data, just point to it
			other.managed = this->managed;			 // Copy the managed state
			other.contiguous = false;				 // Can do better and check if it is a contiguous slice, but for the moment ...
			other.offset = 0;
			other.extent = other.getLastElement();
		}

		/**
		 * Sets the sizes of the wrapper based on the provided ranges.
		 * NOTE that at this point no memory is allocated yet.
		 */
		template <typename... T_ranges>
		DEVICEPREFIX void setSizes(T_ranges... ranges)
		{
			static_assert(sizeof...(ranges) == rank, "Number of ranges must match the rank of the portable array.");
			buildSizes(ranges...);
		}

		/**
		 * Set sizes based on lower and upper bounds arrays
		 */
		template <typename T_arrays>
		DEVICEPREFIX void setSizesArray(const T_arrays *lbounds, const T_arrays *ubounds)
		{
			elements = 1;
			offset = 0;
			for (int i = 0; i < rank; ++i)
			{
				lower_bound[i] = lbounds[i];
				upper_bound[i] = ubounds[i];
				size[i] = ubounds[i] - lbounds[i] + 1;
				elements *= size[i];
			}

			calculateStrides<0>();
		}

		/**
		 * Set sizes based on sizes array
		 */
		DEVICEPREFIX void setSizesArray(const SIZE_TYPE *sizes)
		{
			elements = 1;
			offset = 0;
			for (int i = 0; i < rank; ++i)
			{
				lower_bound[i] = 0;
				upper_bound[i] = sizes[i] - 1;
				size[i] = sizes[i];
				elements *= size[i];
			}
			calculateStrides<0>();
		}

		/**
		 * Binds the data pointer to the wrapper.
		 * This is used to set the data pointer after memory allocation.
		 * It is not a constructor because we want to be able to reallocate
		 * the data without creating a new wrapper instance.
		 */
		DEVICEPREFIX void bind(T *data)
		{
			this->data_ = data;
			ownsData = true; // We own the data now
		}

		/**
		 * Make this array point to another array's data
		 * Does not take ownership of the data
		 */
		template <arrayTags T_tag>
		DEVICEPREFIX void pointTo(const portableArray<T, rank, T_tag> &other)
		{
			if ((char *)this != (char *)&other)
			{
				elements = other.elements;
				for (int i = 0; i < rank; ++i)
				{
					lower_bound[i] = other.lower_bound[i];
					upper_bound[i] = other.upper_bound[i];
					size[i] = other.size[i];
					stride[i] = other.stride[i];
				}
				bind(other.data_);
				ownsData = false; // We do not own the data, just point to it
			}
		}

		/**
		 * Change the shape of this array to new ranges, keeping the same data pointer.
		 * Note, this just changes the bounds, not the starts or sizses. Use a slice
		 * if you need to change those.
		 * The total number of elements in each dimension must match the original array.
		 * @param newRange The new range for the current level
		 * @param otherRanges The new ranges for the other levels
		 */
		template<int level=0, typename T_Range, typename... T_others>
		DEVICEPREFIX void rebaseCore(T_Range newRange, T_others... otherRanges)
		{
			if constexpr(level==0) offset=0;
			static_assert(sizeof...(otherRanges) + 1 == rank, "Number of ranges must match the rank of the portable array.");
			//Only do this check on host, not device
			#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
			if (newRange.upper_bound - newRange.lower_bound != size[level] - 1)
			{
				std::cout << "Rebound range does not match original size in dimension " << level << ": "
						  << "new range [" << newRange.lower_bound << ", " << newRange.upper_bound << "] "
						  << "original size " << size[level] << std::endl;
				throw std::runtime_error("Rebound range does not match original size in dimension " + std::to_string(level));
			}
			#endif
			offset -= newRange.lower_bound * stride[level];
			lower_bound[level] = newRange.lower_bound;
			upper_bound[level] = newRange.upper_bound;			
			if constexpr (sizeof...(otherRanges) > 0)
			{
				rebaseCore<level + 1>(otherRanges...);
			}
		}


	public:
		/**
		 * Is this array using managed memory
		 */
		bool isManaged() const { return managed; }

		/**
		 * Is this array contiguous in memory?
		 */
		bool isContiguous() const { return contiguous; }

		DEVICEPREFIX UNSIGNED_INDEX_TYPE getBytesToEnd() const
		{
			//Because this may be a non-contiguous slice, have to calculate the bytes to the end
			//Using the tuple of upper bounds
			N_ary_tuple_type_t<SIGNED_INDEX_TYPE, rank> ubTuple;
			getUBTuple<0>(ubTuple);
			auto fn =[this](auto... ubounds) {return this->buildIndex(ubounds...);};
			UNSIGNED_INDEX_TYPE endIndex = apply(fn, ubTuple) + 1;
			return endIndex * sizeof(T);
		}

        template <bool zeroBase=false, std::size_t... Is, typename... T_indices>
        DEVICEPREFIX INLINE SIGNED_INDEX_TYPE computeIndexFromPack(std::index_sequence<Is...>, T_indices... indices) const
        {
            SIGNED_INDEX_TYPE index = zeroBase? 0 : offset;
            ((index += static_cast<SIGNED_INDEX_TYPE>(indices) * stride[Is]), ...);
            return index;
        }

		/**
		 * Round bracket operator to access elements in the wrapper.
		 * This operator allows for accessing elements in the wrapper
		 */
		template <typename... T_indices, std::enable_if_t<countRanges<T_indices...>() == 0, int> = 0>
		DEVICEPREFIX INLINE __attribute__((flatten)) selectRefType operator()(T_indices... indices) const
		{
			static_assert(sizeof...(indices) == rank, "Number of indices must match the rank of the portable array.");
			#ifndef ARRAY_BOUNDS_CHECKING
			SIGNED_INDEX_TYPE index = computeIndexFromPack(std::make_index_sequence<rank>{}, indices...);
			#else
			SIGNED_INDEX_TYPE index = buildIndex(indices...);
			#endif
			return data_[index];
		}

		/**
		 * Operator to access elements in the wrapper using zero-based indexing.
		 */
		template <typename... T_indices,
				  typename = std::enable_if_t<countRanges<T_indices...>() == 0>>
		DEVICEPREFIX INLINE selectRefType getZB(T_indices... indices) const
		{
			static_assert(sizeof...(indices) == rank, "Number of indices must match the rank of the portable array.");
			#ifndef ARRAY_BOUNDS_CHECKING
			SIGNED_INDEX_TYPE index = computeIndexFromPack<true>(std::make_index_sequence<rank>{}, indices...);
			#else
			SIGNED_INDEX_TYPE index = buildIndexZB(indices...);
			#endif
			//No offset for zero-based indexing
			return data_[index];
		}

		/**
		 * Get the index offset into the underlying data array using zero-based indexing.
		 * Only used for debugging purposes.
		 */
		template <typename... T_indices>
		DEVICEPREFIX INLINE size_t getIndexOffset(T_indices... indices) const
		{
			return buildIndexZB(indices...);
		}

		/**
		 * Round bracket operator to slice the wrapper. Returns a new portableArray with the specified indices.
		 */
		template <typename... T_indices, std::enable_if_t<countRanges<T_indices...>() != 0, int> = 0>		
		DEVICEPREFIX INLINE portableArray<T, countRanges<T_indices...>(), arrayTag> operator()(T_indices... indices) const
		{
			static_assert(sizeof...(indices) == rank, "Number of indices must match the rank of the portable array.");
			portableArray<T, countRanges<T_indices...>(), arrayTag> slicedArray;
			sliceTo(slicedArray, indices...);
			return slicedArray;
		}

		/**
		 * Get the underlying data pointer.
		 */
		DEVICEPREFIX T *data() const
		{
			return data_;
		}

		/**
		 * Get the number of elements in the wrapper.
		 */
		DEVICEPREFIX size_t getElements() const
		{
			return elements;
		}

		/**
		 * Get the array of sizes for each dimension.
		 */
		DEVICEPREFIX const SIGNED_INDEX_TYPE *getSizes() const
		{
			return size;
		}

		/**
		 * Get the number of elements in a given dimension.
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE getSize(int dimension) const
		{
			return size[dimension];
		}


		/** 
		 * Get the array of strides for each dimension.
		 */
		DEVICEPREFIX const SIGNED_INDEX_TYPE *getStrides() const
		{
			return stride;
		}

		/**
		 * Get the stride of a given dimension.
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE getStride(int dimension) const
		{
			return stride[dimension];
		}


		/**
		 * Get the array of lower bounds for each dimension.
		 */
		DEVICEPREFIX const SIGNED_INDEX_TYPE *getLowerBounds() const
		{
			return lower_bound;
		}

		/**
		 * Get the lower bound of a given dimension.
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE getLowerBound(int dimension) const
		{
			return lower_bound[dimension];
		}

		/**
		 * Get the zero-based lower bound of a given dimension.
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE getLowerBoundZeroBased([[maybe_unused]] int dimension) const
		{
			return 0;
		}

		/**
		 * Get the array of upper bounds for each dimension.
		 */
		DEVICEPREFIX const SIGNED_INDEX_TYPE *getUpperBounds() const
		{
			return upper_bound;
		}

		/**
		 * Get the upper bound of a given dimension.
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE getUpperBound(int dimension) const
		{
			return upper_bound[dimension];
		}

		template <typename... T_indices,
				  typename = std::enable_if_t<countRanges<T_indices...>() == 0>>
		DEVICEPREFIX INLINE UNSIGNED_INDEX_TYPE getIndex(T_indices... indices) const
		{
			static_assert(sizeof...(indices) == rank, "Number of indices must match the rank of the portable array.");
			UNSIGNED_INDEX_TYPE index = buildIndex(indices...);
			return index;
		}

		/**
		 * Bind new data to the array after setting sizes
		 */
		template <typename... T_indices>
		DEVICEPREFIX void bind(T *data, T_indices... indices)
		{
			this->setSizes(std::forward<T_indices>(indices)...);
			this->bind(data);
		}

		/**
		 * Bind new data to the array after setting sizes from lower and upper bounds
		 */
		DEVICEPREFIX void bindArrayBounds(T *data, const SIGNED_INDEX_TYPE *lbounds, const SIGNED_INDEX_TYPE *ubounds)
		{
			this->setSizesArray(lbounds, ubounds);
			this->bind(data);
		}

		/**
		 * Bind new data to the array after setting sizes from sizes array
		 */
		DEVICEPREFIX void bindArrayBounds(T *data, const SIZE_TYPE *sizes)
		{
			this->setSizesArray(sizes);
			this->bind(data);
		}

		/**
		 * Nullify the data pointer
		 */
		DEVICEPREFIX void nullify()
		{
			this->bind(nullptr);
		}

		/**
		 * Return the lower bound of the array in a given dimension
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE lb(int dimension) const
		{
			return lower_bound[dimension];
		}

		/**
		 * Return the upper bound of the array in a given dimension
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE ub(int dimension) const
		{
			return upper_bound[dimension];
		}

		/**
		 * Get the lower and upper bounds as a Range object for a given dimension
		 */
		DEVICEPREFIX Range getRange(int dimension) const
		{
			return Range(lb(dimension), ub(dimension));
		}

		/**
		 * Get the offset into the data array
		 */
		DEVICEPREFIX SIGNED_INDEX_TYPE getOffset() const
		{
			return offset;
		}

		//std::string Name;

		/**
		 * Output an array to a stream 
		 */
		template<typename T_os>
		auto& output(T_os &os) const
		{
			//Loop through all elements and print them
			//Fortran style with elements in native order
			//and a space between each element
			size_t totalElements = getElements();
			for (size_t i = 0; i < totalElements; ++i)
			{
				os << data_[i] << " ";
			}
			return os;
		}

		template<typename... T_ranges>
		void rebase(T_ranges... ranges)
		{
			static_assert(sizeof...(ranges) == rank, "Number of ranges must match the rank of the portable array.");
			rebaseCore<0>(ranges...);
		}
		
	};

	template <typename T, int rank>
	using acceleratedArray = portableArray<T, rank, arrayTags::accelerated>;

	template <typename T, int rank>
	using hostArray = portableArray<T, rank, arrayTags::host>;
};
#endif // ARRAY_H
