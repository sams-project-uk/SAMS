#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "array.h"

namespace portableWrapper
{
	namespace detail{
		/**
		 * Functor to assign a value to a specific element in the array.
		 * Has to be a functor because HIP/CUDA don't support variadic lambdas
		 */
		template<typename T, int rank, arrayTags arrayTag, typename T2>
			struct assignValue{
				using pa = portableArray<T, rank, arrayTag>;
				pa array;
				T2 value;
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

		/**
		 * Functor to apply a kernel with red black ordering. Red elements are those where the sum of the indices is odd
		 * and black elements those where it is even. It is launched over a range along the last axis that has been
		 * halved, keeping its lower bound, so the last index it receives is h and the elements it covers are
		 * 2h - lower and 2h - lower + 1. Which of the two has the wanted colour depends on the sum of the other
		 * indices, so 1 is added when needed. lower and upper are the bounds of the original last range, and an
		 * element above upper is skipped
		 */
		template<bool red, typename T_kernel>
			struct redBlack {
				T_kernel kernel;
				SIGNED_INDEX_TYPE lower, upper;

				FUNCTORMETHODPREFIX INLINE redBlack(T_kernel kernel, SIGNED_INDEX_TYPE lower, SIGNED_INDEX_TYPE upper)
					: kernel(kernel), lower(lower), upper(upper) {}

				/**
				 * The indices are moved from the front to the back one at a time, until the last one, h, is at the front.
				 * Then the real last index is worked out from h and the sum of the others, and the kernel is called
				 */
				template<size_t remaining, typename T_first, typename... T_rest>
					FUNCTORMETHODPREFIX INLINE void apply(T_first first, T_rest... rest) const
					{
						if constexpr (remaining > 0) {
							apply<remaining - 1>(rest..., first);
						} else {
							// Sum of the other indices using a C++17 fold expression
							const SIGNED_INDEX_TYPE sum = (rest + ... + 0);

							// Multiply h by 2, take off the lower bound, and add 1 if that is needed to get the wanted colour
							const SIGNED_INDEX_TYPE last = 2 * static_cast<SIGNED_INDEX_TYPE>(first) - lower + ((sum + lower + (red ? 1 : 0)) & 1);

							if (last <= upper) {
								kernel(rest..., last);
							}
						}
					}

				template<typename... T_indices>
					FUNCTORMETHODPREFIX INLINE void operator()(T_indices... indices) const
					{
						apply<sizeof...(T_indices) - 1>(indices...);
					}
			};

		/**
		 * Launches the two colours of a red black kernel. The ranges are moved from the front to the back one at a time,
		 * until the last one is at the front. It is then halved, keeping its lower bound, and both colours are launched
		 * over the other ranges followed by the halved one
		 */
		template<size_t remaining, typename T_kernel, typename T_first, typename... T_rest>
			UNREPEATED void applyRedBlackRotated(T_kernel kernel, T_first first, T_rest... rest) {
				if constexpr (remaining > 0) {
					applyRedBlackRotated<remaining - 1>(kernel, rest..., first);
				} else {
					const SIGNED_INDEX_TYPE lower = first.lower_bound, upper = first.upper_bound;
					const Range halved(lower, lower + (upper - lower + 2) / 2 - 1);
					portableWrapper::applyKernel(redBlack<true,T_kernel>(kernel, lower, upper), rest..., halved);
					portableWrapper::fence();
					portableWrapper::applyKernel(redBlack<false,T_kernel>(kernel, lower, upper), rest..., halved);
					portableWrapper::fence();
				}
			}


		/**
		 * Functor to apply a kernel with eight colour ordering over three ranges. The colour is the parities of the
		 * three indices, with bit 0 for the first index, bit 1 for the second and bit 2 for the third. Two elements
		 * with the same colour are never neighbours, even diagonally. It is launched over ranges that have been
		 * halved, keeping their lower bounds, so the indices it receives are h and the elements it covers are
		 * 2h - lower and 2h - lower + 1. Which of the two has the wanted parity depends on the lower bound, so 1 is
		 * added when needed. lower and upper are the bounds of the original ranges, and an element above an upper
		 * bound is skipped
		 */
		template<typename T_kernel>
			struct eightColour {
				T_kernel kernel;
				int colour;
				SIGNED_INDEX_TYPE lower[3], upper[3];

				FUNCTORMETHODPREFIX INLINE eightColour(T_kernel kernel, int colour, SIGNED_INDEX_TYPE lower0, SIGNED_INDEX_TYPE lower1, SIGNED_INDEX_TYPE lower2, SIGNED_INDEX_TYPE upper0, SIGNED_INDEX_TYPE upper1, SIGNED_INDEX_TYPE upper2)
					: kernel(kernel), colour(colour), lower{lower0, lower1, lower2}, upper{upper0, upper1, upper2} {}

				template<typename T_0, typename T_1, typename T_2>
					FUNCTORMETHODPREFIX INLINE void operator()(T_0 h0, T_1 h1, T_2 h2) const
					{
						const SIGNED_INDEX_TYPE i = 2 * static_cast<SIGNED_INDEX_TYPE>(h0) - lower[0] + ((lower[0] + colour) & 1);
						const SIGNED_INDEX_TYPE j = 2 * static_cast<SIGNED_INDEX_TYPE>(h1) - lower[1] + ((lower[1] + (colour >> 1)) & 1);
						const SIGNED_INDEX_TYPE k = 2 * static_cast<SIGNED_INDEX_TYPE>(h2) - lower[2] + ((lower[2] + (colour >> 2)) & 1);

						if (i <= upper[0] && j <= upper[1] && k <= upper[2]) {
							kernel(i, j, k);
						}
					}
			};

		template<arrayTags tags, int rank, typename Range>
			UNREPEATED bool squashRange(Range &r){
				if constexpr (getFastestAxis(tags) == rank){
					return true;
				}
				return false;
			}

		template<arrayTags tags, typename... T_ranges, size_t... Is>
			UNREPEATED void squashRangesImpl(std::index_sequence<Is...>, T_ranges& ...ranges){
				(squashRange<tags, Is>(ranges) || ...);
			}

		template<arrayTags tags, typename... T_ranges>
			UNREPEATED void squashRanges(T_ranges& ...ranges){
				squashRangesImpl<tags>(std::make_index_sequence<sizeof...(T_ranges)>{}, ranges...);
			}


	}

	/**
	 * Assigns one portableArray to another.
	 * This function checks if the source and destination arrays have the same number of elements
	 * and then applies the assignment in parallel using the provided ranges.
	 */
	template< bool fence=false, typename T=void, int rank=0, arrayTags tag=arrayTags::accelerated>
		HOSTFLATTEN void assign(portableArray<T, rank, tag> dest, const portableArray<T, rank, tag> &src) {
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
						constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
						portableWrapper::applyKernel<space>(args...);
						}, tpl);
				if constexpr(fence) portableWrapper::fence();
			}
		}

	template< typename T, int rank, arrayTags tag, typename T2>
		HOSTFLATTEN void assign(portableArray<T, rank, tag> dest, const T2 &src) {
			static_assert(std::is_convertible_v<T,T2>,"Source type must be convertible to destination type in assignment");
			std_N_ary_tuple_type_t<Range,rank> ranges;
			detail::arrayToRangesZB(dest, ranges);

			auto tpl = std::tuple_cat(
					std::make_tuple(detail::assignValue<T,rank,tag,T2>(dest, src)),
					ranges
					);
			std::apply([](auto&&... args) {
					constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
					portableWrapper::applyKernel<space>(args...);
					}, tpl);
		}

	/**
	 * Applies a kernel with red black ordering. The last range must be a Range. It is halved, so that no work items
	 * are idle, and the functor multiplies the index back up
	 */
	template<typename T_kernel, typename... T_Ranges>
		UNREPEATED void applyRedBlackKernel(T_kernel kernel, T_Ranges... ranges) {
			static_assert(sizeof...(ranges) > 0, "Must specify at least one range to apply a kernel over");
			detail::applyRedBlackRotated<sizeof...(ranges) - 1>(kernel, ranges...);
		}

	/**
	 * Applies a kernel with eight colour ordering over three ranges, so that no two elements in the same launch are
	 * neighbours, even diagonally. Each range is halved, so that no work items are idle, and the functor multiplies
	 * the indices back up. The eight colours are launched in turn with a fence after each
	 */
	template<typename T_kernel>
		UNREPEATED void applyEightColourKernel(T_kernel kernel, Range x, Range y, Range z) {
			const Range halvedX(x.lower_bound, x.lower_bound + (x.upper_bound - x.lower_bound + 2) / 2 - 1);
			const Range halvedY(y.lower_bound, y.lower_bound + (y.upper_bound - y.lower_bound + 2) / 2 - 1);
			const Range halvedZ(z.lower_bound, z.lower_bound + (z.upper_bound - z.lower_bound + 2) / 2 - 1);
			for (int colour = 0; colour < 8; ++colour) {
				portableWrapper::applyKernel(detail::eightColour<T_kernel>(kernel, colour, x.lower_bound, y.lower_bound, z.lower_bound, x.upper_bound, y.upper_bound, z.upper_bound), halvedX, halvedY, halvedZ);
				//portableWrapper::fence();
			}
			portableWrapper::fence();
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
					constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
					return applyReduction<space>(args...);
					}, tpl);
		}


	/**
	 * Returns the maximum value in the portableArray.
	 */
	template<typename T_kernel, typename... T_Ranges>
		UNREPEATED far::callableTraits<T_kernel>::type maxval(T_kernel kernel, const T_Ranges&... ranges) {
			using T = far::callableTraits<T_kernel>::type;

			auto maxValFunc = LAMBDA(T &a, const T &b) {
				a = portableWrapper::max(a, b);
			};

			auto tpl = std::tuple_cat(
					std::make_tuple(kernel),
					std::make_tuple(maxValFunc),
					std::make_tuple(std::numeric_limits<T>::lowest()),
					std::tuple<T_Ranges...>(ranges...)
					);

			return std::apply([](auto&&... args) {
					return applyReduction(args...);
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
					constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
					return applyReduction<space>(args...);
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
					std::make_tuple(T{}),
					ranges
					);

			return std::apply([](auto&&... args) {
					constexpr executionSpace space = portableWrapper::getDefaultExecutionSpace(tag);
					return applyReduction<space>(args...);
					}, tpl);
		}

}

#endif // ALGORITHM_H
