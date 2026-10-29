#ifndef MULTIGRIDTEST_HELPERS_H
#define MULTIGRIDTEST_HELPERS_H
#include "runner.h"
#include "harness.h"
#include <vector>
#include <type_traits>
#include <string>
#include <array>
#include <algorithm>

// The functions that find an entry of a stencil from its position are consteval, which makes them compile time only, from C++20, and constexpr before it
#ifndef CONSTEVAL
#if __cplusplus >= 202002L
#define CONSTEVAL consteval
#else
#define CONSTEVAL constexpr
#endif
#endif

namespace SAMS
{
	namespace pw = portableWrapper;

	namespace parabolic
	{

	/**
	 * @brief The name of the axis of a direction on a level
	 *
	 * It is the derived axis of that level, or the last derived axis if the direction has fewer levels, or the base axis if it has none.
	 * @param memoryName The memory name of the problem, which is in the names of the derived axes
	 * @param baseName The name of the base axis
	 * @param axisLevels The number of derived axes that are used in each direction
	 * @param direction The direction, 0 to 2
	 * @param level The level, counted from 1 for the first coarse level
	 */
	inline std::string levelAxisName(const std::string &memoryName, const std::string &baseName, const std::array<int, 3> &axisLevels, int direction, int level)
	{
		const int axisLevel = std::min(level, axisLevels[direction]);
		return axisLevel > 0 ? memoryName + "_" + baseName + "_" + std::to_string(axisLevel) : baseName;
	}

	/**
	 * @brief Register an array of the problem on one level, on the axes of that level
	 * @param harness The harness, which holds the variable registry
	 * @param memoryName The memory name of the problem, which is in the names of the array and of the derived axes
	 * @param fieldName The name of the variable that is evolved, which gives the base axes
	 * @param axisLevels The number of derived axes that are used in each direction
	 * @param variable The name of the array, to which the level is added
	 * @param staggerX, staggerY, staggerZ The stagger of the array in each direction
	 * @param level The level, counted from 0 for the finest level
	 */
	inline void registerArray(SAMS::harness &harness, const std::string &memoryName, const std::string &fieldName, const std::array<int, 3> &axisLevels, const std::string &variable, SAMS::staggerType staggerX, SAMS::staggerType staggerY, SAMS::staggerType staggerZ, int level = 0)
	{
		auto &varRegistry = harness.variableRegistry;
		const SAMS::variableDef &fieldDef = varRegistry.getVariable(fieldName);
		const int ghosts = 1;
		std::string axisNames[3];
		for (int d = 0; d < 3; ++d)
			axisNames[d] = levelAxisName(memoryName, fieldDef.getDimension(d).axisName, axisLevels, d, level);
		const std::string arrayName = memoryName + "_" + variable + "_" + std::to_string(level);
		varRegistry.registerVariable<SAMS::T_dataType>(arrayName, pw::arrayTags::accelerated, SAMS::dimension(axisNames[0], ghosts, staggerX), SAMS::dimension(axisNames[1], ghosts, staggerY), SAMS::dimension(axisNames[2], ghosts, staggerZ));
	}

	/**
	 * @brief Register an array of the problem on every level, from the finest to the last
	 *
	 * The arguments are those of registerArray, without the level.
	 */
	inline void registerArrayLevels(SAMS::harness &harness, const std::string &memoryName, const std::string &fieldName, const std::array<int, 3> &axisLevels, const std::string &variable, SAMS::staggerType staggerX, SAMS::staggerType staggerY, SAMS::staggerType staggerZ)
	{
		const int levelCount = std::max(std::max(axisLevels[0], axisLevels[1]), axisLevels[2]);
		for (int level = 0; level <= levelCount; ++level)
			registerArray(harness, memoryName, fieldName, axisLevels, variable, staggerX, staggerY, staggerZ, level);
	}

	/**
	 * @brief Get an array of the problem on one level from the variable registry
	 * @param harness The harness, which holds the variable registry
	 * @param memoryName The memory name of the problem, which is in the name of the array
	 * @param variable The name of the array, as it was registered
	 * @param array The array to fill
	 * @param level The level, counted from 0 for the finest level
	 */
	template <typename T_array>
	void fillArray(SAMS::harness &harness, const std::string &memoryName, const std::string &variable, T_array &array, int level = 0)
	{
		harness.variableRegistry.fillPPArray(memoryName + "_" + variable + "_" + std::to_string(level), array);
	}

	/**
	 * @brief Get an array of the problem on every level from the variable registry
	 * @param levels The levels, finest first
	 * @param getArray A function that returns the array of a level, given the level
	 *
	 * The other arguments are those of fillArray.
	 */
	template <typename T_level, typename T_getArray>
	void fillArrayLevels(SAMS::harness &harness, const std::string &memoryName, const std::string &variable, std::vector<T_level> &levels, T_getArray getArray)
	{
		for (size_t n = 0; n < levels.size(); ++n)
			fillArray(harness, memoryName, variable, getArray(levels[n]), static_cast<int>(n));
	}

	/** @brief The function type of averageCells that means no scaling */
	struct noFn
	{
	};

	/**
	 * @brief Set a coarse cell array to the mean of the fine cells it covers
	 * @param fineArray The array on the fine level
	 * @param coarseArray The array on the coarse level
	 * @param fineLevel The fine level
	 * @param l The coarse level
	 * @param scale A device function of the mean and i, j, k that returns the value to store. noFn stores the mean
	 */
	template <typename T_array, typename T_level, typename T_scale = noFn>
	void averageCells(const T_array &fineArray, const T_array &coarseArray, const T_level &fineLevel, const T_level &l, T_scale scale = T_scale())
	{
		const T_array fine = fineArray, coarse = coarseArray;
		const size_t fx = l.coarsenX ? 2 : 1, fy = l.coarsenY ? 2 : 1, fz = l.coarsenZ ? 2 : 1;
		// Offset of the first coarse cell along the fine cells
		const size_t ox = l.coarsenX ? static_cast<size_t>(2 * (l.firstX - 1) - (fineLevel.firstX - 1)) : 0;
		const size_t oy = l.coarsenY ? static_cast<size_t>(2 * (l.firstY - 1) - (fineLevel.firstY - 1)) : 0;
		const size_t oz = l.coarsenZ ? static_cast<size_t>(2 * (l.firstZ - 1) - (fineLevel.firstZ - 1)) : 0;
		auto kernel = LAMBDA(size_t i, size_t j, size_t k)
		{
			double sum = 0.0;
			for (size_t a = 0; a < fx; ++a)
				for (size_t b = 0; b < fy; ++b)
					for (size_t c = 0; c < fz; ++c)
						sum += fine(fx * (i - 1) + 1 + a + ox, fy * (j - 1) + 1 + b + oy, fz * (k - 1) + 1 + c + oz);
			const double mean = sum / (fx * fy * fz);
			// nvcc must see the captures outside the constexpr-if
			(void)coarse;
			(void)scale;
			if constexpr (std::is_same_v<T_scale, noFn>)
				coarse(i, j, k) = mean;
			else
				coarse(i, j, k) = scale(mean, i, j, k);
		};
		pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
	}

	/**
	 * @brief Set a coarse face array to the mean of the fine faces it covers
	 * @param fineArray The array on the fine level
	 * @param coarseArray The array on the coarse level
	 * @param fineLevel The fine level
	 * @param l The coarse level
	 * @param direction The face normal: 0, 1 or 2 for x, y or z
	 */
	template <typename T_array, typename T_level>
	void averageFaces(const T_array &fineArray, const T_array &coarseArray, const T_level &fineLevel, const T_level &l, int direction)
	{
		const T_array fine = fineArray, coarse = coarseArray;
		const size_t fx = l.coarsenX ? 2 : 1, fy = l.coarsenY ? 2 : 1, fz = l.coarsenZ ? 2 : 1;
		const size_t countX = (direction == 0) ? 1 : fx, countY = (direction == 1) ? 1 : fy, countZ = (direction == 2) ? 1 : fz;
		// Offset of the first coarse cell along the fine cells
		const size_t ox = l.coarsenX ? static_cast<size_t>(2 * (l.firstX - 1) - (fineLevel.firstX - 1)) : 0;
		const size_t oy = l.coarsenY ? static_cast<size_t>(2 * (l.firstY - 1) - (fineLevel.firstY - 1)) : 0;
		const size_t oz = l.coarsenZ ? static_cast<size_t>(2 * (l.firstZ - 1) - (fineLevel.firstZ - 1)) : 0;
		auto kernel = LAMBDA(size_t i, size_t j, size_t k)
		{
			// The first fine index covered. In the face direction the edge is the last fine edge of the cell, which is at fx * i
			const size_t firstI = (direction == 0) ? fx * i + ox : fx * (i - 1) + 1 + ox;
			const size_t firstJ = (direction == 1) ? fy * j + oy : fy * (j - 1) + 1 + oy;
			const size_t firstK = (direction == 2) ? fz * k + oz : fz * (k - 1) + 1 + oz;
			double sum = 0.0;
			for (size_t a = 0; a < countX; ++a)
				for (size_t b = 0; b < countY; ++b)
					for (size_t c = 0; c < countZ; ++c)
						sum += fine(firstI + a, firstJ + b, firstK + c);
			coarse(i, j, k) = sum / (countX * countY * countZ);
		};
		const pw::Range xRange = (direction == 0) ? l.xbLocalDomainRange : l.xcLocalDomainRange;
		const pw::Range yRange = (direction == 1) ? l.ybLocalDomainRange : l.ycLocalDomainRange;
		const pw::Range zRange = (direction == 2) ? l.zbLocalDomainRange : l.zcLocalDomainRange;
		pw::applyKernel(kernel, xRange, yRange, zRange);
	}

	/**
	 * @brief Average a cell array from each level onto the next coarser level
	 * @param levels The levels, finest first
	 * @param getArray A function that returns the array of a level, given the level
	 */
	template <typename T_level, typename T_getArray>
	void restrictCells(std::vector<T_level> &levels, T_getArray getArray)
	{
		for (size_t n = 1; n < levels.size(); ++n)
		{
			averageCells(getArray(levels[n - 1]), getArray(levels[n]), levels[n - 1], levels[n]);
		}
	}

	/**
	 * @brief Average a face array from each level onto the next coarser level
	 * @param levels The levels, finest first
	 * @param getArray A function that returns the array of a level, given the level
	 * @param direction The face normal: 0, 1 or 2 for x, y or z
	 */
	template <typename T_level, typename T_getArray>
	void restrictFaces(std::vector<T_level> &levels, T_getArray getArray, int direction)
	{
		for (size_t n = 1; n < levels.size(); ++n)
		{
			averageFaces(getArray(levels[n - 1]), getArray(levels[n]), levels[n - 1], levels[n], direction);
		}
	}

	/**
	 * @brief A position in the stencil of a problem, which is the offset from the cell, so at<0, 0, 0> is the cell itself
	 *
	 * It is used as an index of stencilWeights, as in weights[at<1, 0, 0>]. The position is a template argument, so a position that is not in the stencil is a compile error.
	 */
	template <int di, int dj, int dk>
	struct stencilPosition
	{
	};

	template <int di, int dj, int dk>
	inline constexpr stencilPosition<di, dj, dk> at{};

	namespace fluxDivergence
	{

	/**
	 * @brief The entry of the stencil of a problem that is at an offset from the cell, or -1 if there is no such entry
	 *
	 * Entry s is stencilCellOffset(s), so stencilCellOffset of the problem must be constexpr.
	 * @tparam T The problem
	 */
	template <typename T>
	DEVICEPREFIX CONSTEVAL int stencilIndex(int di, int dj, int dk)
	{
		for (int s = 0; s < T::stencilSize; ++s)
		{
			int ei = 0, ej = 0, ek = 0;
			T::stencilCellOffset(s, ei, ej, ek);
			if (ei == di && ej == dj && ek == dk)
				return s;
		}
		return -1;
	}

	/**
	 * @brief For each entry of the stencil of a cell, the entry of the cell on the low side of it in each direction
	 *
	 * An entry s of the stencil of the cell on the low side is the entry shift[s] of the stencil of the cell, or -1 if it is not in it.
	 * @tparam T The problem
	 */
	template <typename T>
	struct faceShift
	{
		int x[T::stencilSize];
		int y[T::stencilSize];
		int z[T::stencilSize];
	};

	template <typename T>
	DEVICEPREFIX CONSTEVAL faceShift<T> makeFaceShift()
	{
		faceShift<T> shift{};
		for (int s = 0; s < T::stencilSize; ++s)
		{
			int di = 0, dj = 0, dk = 0;
			T::stencilCellOffset(s, di, dj, dk);
			shift.x[s] = stencilIndex<T>(di - 1, dj, dk);
			shift.y[s] = stencilIndex<T>(di, dj - 1, dk);
			shift.z[s] = stencilIndex<T>(di, dj, dk - 1);
		}
		return shift;
	}

	/**
	 * @brief For each entry of the stencil, where the weights of a face in one direction are copied to
	 *
	 * For an entry on the low side of the face, target[s] is the entry one cell further in the direction of the face, or -1 if it is not in the stencil.
	 * The weight is multiplied by sign[s], which is -1 for the cell itself and for the cells in line with it, and 1 for the cells to the side. high[s] is true for the entries on the high side of the face.
	 * @tparam T The problem
	 */
	template <typename T>
	struct stencilMirror
	{
		int target[T::stencilSize];
		double sign[T::stencilSize];
		bool high[T::stencilSize];
	};

	template <typename T>
	DEVICEPREFIX CONSTEVAL stencilMirror<T> makeStencilMirror(int direction)
	{
		stencilMirror<T> mirror{};
		for (int s = 0; s < T::stencilSize; ++s)
		{
			int d[3] = {0, 0, 0};
			T::stencilCellOffset(s, d[0], d[1], d[2]);
			mirror.target[s] = -1;
			mirror.sign[s] = 1.0;
			mirror.high[s] = (d[direction] == 1);
			if (d[direction] == 0)
			{
				int e[3] = {d[0], d[1], d[2]};
				e[direction] = 1;
				mirror.target[s] = stencilIndex<T>(e[0], e[1], e[2]);
				if (d[(direction + 1) % 3] == 0 && d[(direction + 2) % 3] == 0)
					mirror.sign[s] = -1.0;
			}
		}
		return mirror;
	}

	/**
	 * @brief Weights on the cells of the stencil of a problem, which start at zero
	 *
	 * An entry can be found from its position, as in weights[at<1, 0, 0>], or from its place in the stencil, as in weights[s].
	 * @tparam T The problem
	 */
	template <typename T>
	struct stencilWeights
	{
		double value[T::stencilSize] = {};

		/** @brief Set the entries on the high side of the face to those on the low side, as given by the mirror, replacing what they were */
		DEVICEPREFIX void mirrorLowToHigh(const stencilMirror<T> &mirror)
		{
			for (int s = 0; s < T::stencilSize; ++s)
				if (mirror.high[s])
					value[s] = 0.0;
			for (int s = 0; s < T::stencilSize; ++s)
				if (mirror.target[s] >= 0)
					value[mirror.target[s]] = mirror.sign[s] * value[s];
		}

		template <int di, int dj, int dk>
		DEVICEPREFIX double &operator[](stencilPosition<di, dj, dk>)
		{
			constexpr int index = stencilIndex<T>(di, dj, dk);
			static_assert(index >= 0, "The position is not in the stencil");
			return value[index];
		}

		template <int di, int dj, int dk>
		DEVICEPREFIX const double &operator[](stencilPosition<di, dj, dk>) const
		{
			constexpr int index = stencilIndex<T>(di, dj, dk);
			static_assert(index >= 0, "The position is not in the stencil");
			return value[index];
		}

		DEVICEPREFIX double &operator[](int s) { return value[s]; }
		DEVICEPREFIX const double &operator[](int s) const { return value[s]; }
	};

	/**
	 * @brief The weights of the fluxes through the three faces on the high side of a cell
	 *
	 * Each array holds the weights on the cells of the stencil, found from their positions, as in weights.fluxWeightsX[{1, 0, 0}].
	 * A flux is the flux out of the cell in the positive direction, which is the sum of its weights times the values of the cells.
	 * @tparam T The problem
	 */
	template <typename T>
	struct faceWeights
	{
		stencilWeights<T> fluxWeightsX;
		stencilWeights<T> fluxWeightsY;
		stencilWeights<T> fluxWeightsZ;

		/**
		 * @brief Set the weights on the high side of each face from those on the low side
		 *
		 * Only the weights with a position of 0 in the direction of the face need to be set. The weights with a position of 1 are replaced by them,
		 * with a sign change for the cell itself and for the cells in line with it, and with no sign change for the cells to the side.
		 */
		DEVICEPREFIX void makeSymmetric()
		{
			constexpr stencilMirror<T> mirrorX = makeStencilMirror<T>(0);
			constexpr stencilMirror<T> mirrorY = makeStencilMirror<T>(1);
			constexpr stencilMirror<T> mirrorZ = makeStencilMirror<T>(2);
			fluxWeightsX.mirrorLowToHigh(mirrorX);
			fluxWeightsY.mirrorLowToHigh(mirrorY);
			fluxWeightsZ.mirrorLowToHigh(mirrorZ);
		}
	};

	/**
	 * @brief A view of the coefficients of a cell, with the entries found from their positions
	 *
	 * The entries are set to zero when the view is made, so only the entries that are used need to be set.
	 * @tparam T The problem
	 */
	template <typename T>
	struct stencilView
	{
		double *value;

		DEVICEPREFIX stencilView(double *a) : value(a)
		{
			for (int s = 0; s < T::stencilSize; ++s)
				value[s] = 0.0;
		}

		template <int di, int dj, int dk>
		DEVICEPREFIX double &operator[](stencilPosition<di, dj, dk>) const
		{
			constexpr int index = stencilIndex<T>(di, dj, dk);
			static_assert(index >= 0, "The position is not in the stencil");
			return value[index];
		}

		DEVICEPREFIX double &operator[](int s) const { return value[s]; }
	};

	/**
	 * @brief The coefficients of a cell from the fluxes through its faces
	 *
	 * The net flux into the cell is the flux through the face on its low side, which is the flux through the high side of the
	 * cell before it, minus the flux through the face on its high side. Both cells that share a face use the same weights, so
	 * the coefficients of all of the cells add up to zero for every column. Every cell that a flux uses must be an entry of the
	 * stencil.
	 * @tparam T The problem
	 * @param stencil The stencil, which has fluxWeights(i, j, k) returning the faceWeights of the cell
	 * @param a The coefficients of the cell, in the order of the stencil
	 */
	template <typename T, typename T_stencil>
	DEVICEPREFIX void coefficientsFromFluxes(const T_stencil &stencil, size_t i, size_t j, size_t k, double *a)
	{
		constexpr faceShift<T> shift = makeFaceShift<T>();
		const faceWeights<T> high = stencil.fluxWeights(i, j, k);
		const faceWeights<T> lowX = stencil.fluxWeights(i - 1, j, k);
		const faceWeights<T> lowY = stencil.fluxWeights(i, j - 1, k);
		const faceWeights<T> lowZ = stencil.fluxWeights(i, j, k - 1);
		const stencilView<T> coefficient(a);
		for (int s = 0; s < T::stencilSize; ++s)
		{
			coefficient[s] -= high.fluxWeightsX[s] + high.fluxWeightsY[s] + high.fluxWeightsZ[s];
			if (shift.x[s] >= 0)
				coefficient[shift.x[s]] += lowX.fluxWeightsX[s];
			if (shift.y[s] >= 0)
				coefficient[shift.y[s]] += lowY.fluxWeightsY[s];
			if (shift.z[s] >= 0)
				coefficient[shift.z[s]] += lowZ.fluxWeightsZ[s];
		}
	}

	}

	}
}

#endif // MULTIGRIDTEST_HELPERS_H
