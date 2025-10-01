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
#include "callableTraits.h"
#include <limits>

namespace portableWrapper
{
	//Forward declaration of the applyKernel function
	template <typename T_func, typename... T_ranges>
		inline void applyKernel(T_func func, T_ranges... ranges);

	template <typename T_lex=void, typename T_data_in=void, typename T_mapper=void, typename T_reducer=void, typename... T_ranges>
		inline auto applyReduction(T_mapper mapper, T_reducer reducer, T_data_in initialValue, T_ranges... ranges);

	class portableArrayManager;

	namespace detail{

		template <int level=0, typename T, int rank, arrayTags T_tag>
			UNREPEATED void arrayToRanges(const portableArray<T, rank, T_tag> &array, std_N_ary_tuple_type_t<Range,rank> &tuple)
			{
				std::get<level>(tuple) = Range(array.lower_bound[level], array.upper_bound[level]);
				if constexpr (level + 1 < rank)
				{
					arrayToRanges<level + 1>(array, tuple);
				}
			}

		template <int level=0, typename T, int rank, arrayTags T_tag>
			UNREPEATED void arrayToRangesZB(const portableArray<T, rank, T_tag> &array, std_N_ary_tuple_type_t<Range,rank> &tuple)
			{
				std::get<level>(tuple) = Range(0, array.size[level] - 1);
				if constexpr (level + 1 < rank)
				{
					arrayToRangesZB<level + 1>(array, tuple);
				}
			}
		/**
		 * Functor to assign a value to a specific element in the array.
		 * Has to be a functor because HIP/CUDA don't support variadic lambdas
		 */
		template<typename T, int rank, arrayTags arrayTag, typename T2>
			struct assignValue{
				T2 value;
				using pa = portableArray<T, rank, arrayTag>;
				pa array;

				FUNCTORMETHODPREFIX INLINE assignValue(pa arr, T value)
					: array(arr), value(value) {}

				template<typename... T_indices>
					FUNCTORMETHODPREFIX INLINE void operator()(T_indices... indices) const
					{
						array.getZB(indices...) = value;
					}
			};

		/**
		 * Functor to assign one portableArray to another.
		 */
		template<typename T, int rank, arrayTags arrayTag>
			struct assignArray{
				using pa = portableArray<T, rank, arrayTag>;
				pa dest;
				pa src;

				FUNCTORMETHODPREFIX INLINE assignArray(pa &dest, const pa &src)
					: dest(dest), src(src) {}

				template<typename... T_indices>
					FUNCTORMETHODPREFIX INLINE void operator()(T_indices... indices) const
					{
						dest.getZB(indices...) = src.getZB(indices...); // Use the overloaded operator() to assign the value
					}
			};

		/**
		 * Functor for returning a value from an array
		 * Used for reductions like minval, maxval, sum, etc.
		 */
		template<typename T, int rank, arrayTags arrayTag>
			struct returnArrayValue{
				using pa = portableArray<T, rank, arrayTag>;
				pa src;

				FUNCTORMETHODPREFIX INLINE returnArrayValue(const pa &src)
					: src(src) {}

				template<typename... T_indices>
					FUNCTORMETHODPREFIX INLINE T operator()(T_indices... indices) const
					{
						return src.getZB(indices...);
					}
			};

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

			friend class portableArrayManager;
			template <int level2, typename T2, int rank2, arrayTags T_tag2> 
				friend void detail::arrayToRanges(const portableArray<T2, rank2, T_tag2>&, std_N_ary_tuple_type_t<Range, rank2>&);
			template <int level2, typename T2, int rank2, arrayTags T_tag2> 
				friend void detail::arrayToRangesZB(const portableArray<T2, rank2, T_tag2>&, std_N_ary_tuple_type_t<Range, rank2>&);

			template<typename T2, int rank2, arrayTags T_tag2>
				friend class portableArray;

			private:
			using type = T;
			using selectRefType = T&;
			static constexpr int rank = i_rank;
			static constexpr arrayTags arrayTag = tag;
			T *data_ = nullptr;
			portableArrayManager *manager = nullptr;
			bool ownsData = true; // Indicates if this wrapper owns the data pointer
			bool managed=false;
			bool contiguous=true;
			SIZE_TYPE elements = 0;
			SIGNED_INDEX_TYPE lower_bound[rank];
			SIGNED_INDEX_TYPE upper_bound[rank];
			SIGNED_INDEX_TYPE stride[rank];
			SIGNED_INDEX_TYPE size[rank];

			void manage(portableArrayManager *mgr)
			{
				manager = mgr;
			}

			/**
			 * Builds the index into the underlying data array
			 */
			template <int level = 0, typename T_current, typename... T_others>
				DEVICEPREFIX INLINE UNSIGNED_INDEX_TYPE buildIndex(T_current c, T_others... r) const
				{
					constexpr int rlevel= level;
					SIGNED_INDEX_TYPE lb = lower_bound[rlevel];
					UNSIGNED_INDEX_TYPE index = (c - lb) * stride[rlevel];
					if constexpr (sizeof...(r) > 0)
					{
						index += buildIndex<level + 1>(r...);
					}
					return index;
				}

			template <int level = 0, typename T_current, typename... T_others>
				DEVICEPREFIX INLINE UNSIGNED_INDEX_TYPE buildIndexZB(T_current c, T_others... r) const
				{
					constexpr int rlevel= level;
					SIGNED_INDEX_TYPE lb = lower_bound[rlevel];
					UNSIGNED_INDEX_TYPE index = c * stride[rlevel];
					if constexpr (sizeof...(r) > 0)
					{
						index += buildIndexZB<level + 1>(r...);
					}
					return index;
				}

			/**
			 * Is this array a pointer to Managed (host/device) memory?
			 * Set by the allocator
			 */
			void setManaged(bool state){
				managed = state;
			}

			/**
			 * Is this array contiguous in memory?
			 */
			void setContiguous(bool state){
				contiguous = state;
			}

			/**
			 * During setup, calculates the strides for each dimension to
			 * get to the next element in a given dimension.
			 */
			template <int level = 0>
				DEVICEPREFIX void calculateStrides()
				{
					static constexpr int reverse_level = rank - level - 1;
					if (level > 0)
					{
						stride[reverse_level] = stride[reverse_level + 1] * size[reverse_level + 1];
					}
					else
					{
						stride[reverse_level] = 1;
					}
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
					if constexpr (sizeof...(r) > 0)
					{
						buildSizes<level + 1>(r...);
					}
					if constexpr (level == 0)
					{
						calculateStrides<0>();
					}
				}

			template<int drank, int slevel=0, int dlevel=0, typename T_tuple>
				DEVICEPREFIX void sliceToImpl(portableArray<T, drank, arrayTag> &other, size_t& startOffset, T_tuple indices) const
				{
					constexpr int rlevel  = drank - dlevel - 1;
					constexpr int rslevel = rank - slevel - 1;
					if constexpr(dlevel==0) {
						other.elements = 1;
					}
					if constexpr(std::is_integral_v<std::remove_reference_t<portableTuple::tuple_element_t<rslevel, T_tuple>>>){
						startOffset += (GET<rslevel>(indices) - this->lower_bound[rslevel]) * this->stride[rslevel];
						other.stride[rlevel] *= this->stride[rslevel];
						if constexpr (rslevel>0){   
							sliceToImpl<drank, slevel+1, dlevel>(other, startOffset, indices);
						}
					} else {
						int64_t lb = GET<rslevel>(indices).lower_bound;
						int64_t ub = GET<rslevel>(indices).upper_bound;
						if (lb == INT64_MIN) lb = this->lower_bound[rslevel];
						if (ub == INT64_MAX) ub = this->upper_bound[rslevel];

						startOffset += (lb - this->lower_bound[rslevel]) * this->stride[rslevel];
						if constexpr (rlevel>=0){
							other.lower_bound[rlevel] = 0;
							other.upper_bound[rlevel] = ub - lb;
							other.stride[rlevel] = this->stride[rslevel];
							other.size[rlevel] = ub - lb + 1;
							other.elements *= other.size[rlevel];
							if constexpr (rslevel>0){
								sliceToImpl<drank, slevel+1, dlevel + 1>(other, startOffset, indices);
							}
						} else {
							if constexpr (rslevel>0){
								sliceToImpl<drank, slevel+1, dlevel>(other, startOffset, indices);
							}
						}
					}
				}

			template<int dRank, typename ... T_indices>
				DEVICEPREFIX void sliceTo(portableArray<T, dRank, arrayTag> &other, T_indices... indices) const
				{
					size_t startOffset = 0;
					for (int i=0;i<dRank; ++i)
					{
						//Initialise the stride to 1, so that if needed
						other.stride[i] = 1; 
					}            
					sliceToImpl<countRanges<T_indices...>(), 0, 0>(other, startOffset, MAKETUPLE(indices...));
					other.data_ = this->data_ + startOffset; // Set the data pointer to the correct offset
					other.ownsData = false; // We do not own the data, just point to it
					other.managed = this->managed; // Copy the managed state
					other.contiguous = false; //Can do better and check if it is a contiguous slice, but for the moment ...
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

			template<typename T_arrays>
				DEVICEPREFIX void setSizes(const T_arrays* lbounds, const T_arrays* ubounds)
				{
					elements = 1;
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

			template<arrayTags T_tag>
				DEVICEPREFIX void pointTo(const portableArray<T, rank, T_tag> &other)
				{
					if ((char*)this != (char*)&other)
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

			public:

			/**
			 * Is this array using managed memory
			 */
			bool isManaged()const{return managed;}

			/**
			 * Is this array contiguous in memory?
			 */
			bool isContiguous() const { return contiguous; }

			/**
			 * Round bracket operator to access elements in the wrapper.
			 * This operator allows for accessing elements in the wrapper
			 * Only works seamlessly for CUDA because we are using managed memory.
			 * //Only enable if none of the indices are ranges.
			 */
			template <typename... T_indices, 
							 typename = std::enable_if_t<countRanges<T_indices...>()==0>>
								 DEVICEPREFIX INLINE  selectRefType operator()(T_indices... indices) const
								 {
									 static_assert(sizeof...(indices) == rank, "Number of indices must match the rank of the portable array.");
									 UNSIGNED_INDEX_TYPE index = buildIndex(indices...);
									 return data_[index];
								 }

			template <typename... T_indices, 
							 typename = std::enable_if_t<countRanges<T_indices...>()==0>>
								 DEVICEPREFIX INLINE selectRefType getZB(T_indices... indices) const
								 {
									 static_assert(sizeof...(indices) == rank, "Number of indices must match the rank of the portable array.");
									 UNSIGNED_INDEX_TYPE index = buildIndexZB(indices...);
									 return data_[index];
								 } 

			template<typename... T_indices>
				DEVICEPREFIX INLINE size_t getIndexOffset(T_indices... indices) const
				{
					return buildIndexZB(indices...); 
				}

			/**
			 * Round bracket operator to slice the wrapper. Returns a new portableArray with the specified indices.
			 */
			template <typename... T_indices, 
							 typename = std::enable_if_t<countRanges<T_indices...>() != 0 > >
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
			 * Get the array of lower bounds for each dimension.
			 */
			DEVICEPREFIX const SIGNED_INDEX_TYPE *getLowerBounds() const
			{
				return lower_bound;
			}

			/**
			 * Get the lower bound of a given dimension.
			 */
			DEVICEPREFIX  SIGNED_INDEX_TYPE getLowerBound(int dimension) const
			{
				return lower_bound[dimension];
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
							 typename = std::enable_if_t<countRanges<T_indices...>()==0>>
								 DEVICEPREFIX INLINE UNSIGNED_INDEX_TYPE getIndex(T_indices... indices) const
								 {
									 static_assert(sizeof...(indices) == rank, "Number of indices must match the rank of the portable array.");
									 UNSIGNED_INDEX_TYPE index = buildIndex(indices...);
									 return index;
								 }

			template<typename... T_indices>
				DEVICEPREFIX void bind(T* data, T_indices... indices) {
					this->setSizes(std::forward<T_indices>(indices)...);
					this->bind(data);
				}

			DEVICEPREFIX void nullify(){
				this->bind(nullptr);
			}

		};

	template<typename T, int rank>
		using acceleratedArray = portableArray<T, rank, arrayTags::accelerated>;

	template<typename T, int rank>
		using hostArray = portableArray<T, rank, arrayTags::host>;

	/**
	 * Assigns one portableArray to another.
	 * This function checks if the source and destination arrays have the same number of elements
	 * and then applies the assignment in parallel using the provided ranges.
	 */
	template< bool fence=false, typename T=void, int rank=0, arrayTags tag=arrayTags::host>
		UNREPEATED void assign(portableArray<T, rank, tag> dest, const portableArray<T, rank, tag> &src) {
			if ((&dest) != (&src))
			{
				if (dest.getElements() != src.getElements()) {
					throw std::runtime_error("Source and destination arrays must have the same number of elements.");
				}
				std_N_ary_tuple_type_t<Range,rank> ranges;
				detail::arrayToRangesZB<0,T,rank,tag>(src, ranges);
				auto tpl = std::tuple_cat(
						std::make_tuple(detail::assignArray(dest, src)),
						ranges
						);
				std::apply([](auto&&... args) {
						portableWrapper::applyKernel(args...);
						}, tpl);
				//if constexpr(fence) portableWrapper::fence();
			}
		}

	/**
	 * Assigns a value to all elements of the portableArray.
	 */
	template< typename T, int rank, arrayTags tag, typename T2>
		UNREPEATED void assign(portableArray<T, rank, tag> dest, const T2 &src) {
			static_assert(std::is_convertible_v<T,T2>,"Source type must be convertible to destination type in assignment");
			std_N_ary_tuple_type_t<Range,rank> ranges;
			detail::arrayToRangesZB(dest, ranges);

			auto tpl = std::tuple_cat(
					std::make_tuple(detail::assignValue<T,rank,tag,T2>(dest, src)),
					ranges
					);
			std::apply([](auto&&... args) {
					portableWrapper::applyKernel(args...);
					}, tpl);
		}

	/**
	 * Returns the maximum value in the portableArray.
	 */
	template<typename T, int rank, arrayTags tag>
		UNREPEATED T maxval(const portableWrapper::portableArray<T, rank, tag> &array) {
			std_N_ary_tuple_type_t<Range, rank> ranges;
			detail::arrayToRangesZB(array, ranges);

			auto maxValFunc = LAMBDA(T &a, const T &b) {
				a = portableWrapper::max(a, b);
			};

			auto tpl = std::tuple_cat(
					std::make_tuple(detail::returnArrayValue<T, rank, tag>(array)),
					std::make_tuple(maxValFunc),
					std::make_tuple(std::numeric_limits<T>::lowest()),
					ranges
					);

			return std::apply([](auto&&... args) {
					return applyReduction<T>(args...);
					}, tpl);
		}

	/**
	 * Returns the minimum value in the portableArray.
	 */
	template<typename T, int rank, arrayTags tag>
		UNREPEATED T minval(const portableWrapper::portableArray<T, rank, tag> &array) {
			std_N_ary_tuple_type_t<Range, rank> ranges;
			detail::arrayToRangesZB(array, ranges);

			auto minValFunc = LAMBDA(T &a, const T &b) {
				a = portableWrapper::min(a, b);
			};

			auto tpl = std::tuple_cat(
					std::make_tuple(detail::returnArrayValue<T, rank, tag>(array)),
					std::make_tuple(minValFunc),
					std::make_tuple(std::numeric_limits<T>::max()),
					ranges
					);

			return std::apply([](auto&&... args) {
					return applyReduction<T>(args...);
					}, tpl);
		}

	/**
	 * Returns the sum of all elements in the portableArray.
	 */
	template<typename T, int rank, arrayTags tag>
		UNREPEATED T sum(const portableWrapper::portableArray<T, rank, tag> &array) {
			std_N_ary_tuple_type_t<Range, rank> ranges;
			detail::arrayToRangesZB(array, ranges);

			auto sumValFunc = LAMBDA(T &a, const T &b) {
				a +=b;
			};

			auto tpl = std::tuple_cat(
					std::make_tuple(detail::returnArrayValue<T, rank, tag>(array)),
					std::make_tuple(sumValFunc),
					std::make_tuple(0),
					ranges
					);

			return std::apply([](auto&&... args) {
					return applyReduction<T>(args...);
					}, tpl);
		}

};
#endif // ARRAY_H
