#ifndef MULTIGRIDTEST_H
#define MULTIGRIDTEST_H
#include "runner.h"
#include "harness.h"
#include "builtInBoundaryConditions.h"
#include "deckHelpers/deckBoundaries.h"
#include "deckHelpers/expressionBC.h"
#include "helpers.h"

// Relative tolerance of the solvers of all the MultigridTest packages. Define SOLVER_TOLERANCE when compiling to set it for all of them
#ifndef SOLVER_TOLERANCE
#define SOLVER_TOLERANCE 1.0e-8
#endif

namespace SAMS
{
	namespace parabolic
	{

	/**
	 * @brief The divergence of the operator of a problem in a cell, and the diagonal entry for the cell
	 * @param s The stencil of the problem for the level
	 * @param u The field
	 * @param dx, dy, dz The cell widths of the level
	 */
	template <typename T, typename T_stencil, typename T_array, typename T_line>
	DEVICEPREFIX void evaluateStencil(const T_stencil &s, const T_array &u, const T_line &dx, const T_line &dy, const T_line &dz, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k, SAMS::T_dataType &divergence, SAMS::T_dataType &diagonal)
	{
		SAMS::T_dataType a[T::stencilSize];
		s.coefficients(i, j, k, a);
		SAMS::T_dataType sum = 0.0, centre = 0.0;
		for (int n = 0; n < T::stencilSize; ++n)
		{
			int di, dj, dk;
			T::stencilCellOffset(n, di, dj, dk);
			sum += a[n] * u(static_cast<SIGNED_INDEX_TYPE>(i) + di, static_cast<SIGNED_INDEX_TYPE>(j) + dj, static_cast<SIGNED_INDEX_TYPE>(k) + dk);
			if (di == 0 && dj == 0 && dk == 0)
				centre = a[n];
		}
		const SAMS::T_dataType volume = dx(i) * dy(j) * dz(k);
		divergence = sum / volume;
		diagonal = -centre / volume;
	}

	/**
	 * @brief Geometric multigrid solver with eight colour Gauss-Seidel
	 *
	 * Solves (V beta)(u - u_old) = the net heat into the cell from the operator of the problem, for the new field u,
	 * where V is the cell volume and beta, from the problem, includes the timestep. The solution is in the variable scratch, which is sol of the finest level.
	 * @tparam T The problem. Provides:
	 * - name: a SAMS::constexprName, which is added to the name of the package.
	 * - fieldName(): the name of the variable that is evolved, whose array is rhs of the finest level.
	 * - levelData: a default constructible type holding the arrays of the problem for one level, which is the member data
	 *   of a level.
	 * - allocate(manager, level, finest): allocate the arrays of the problem for a level.
	 * - problemCount(): the number of subproblems that are solved in turn in each step.
	 * - prepareSolve(levels, problemIndex): set the coefficients of the operator of a subproblem on every level, from the old field,
	 *   which is rhs of the finest level, and any other data that the problem holds.
	 * - makeStencil(level, problemIndex): return an object to be copied into kernels, with coefficients(i, j, k, a) and beta(i, j, k).
	 *   The operator must only depend on the cell and its neighbours which differ by at most one in each direction.
	 * The problem must be trivially copyable, because it is copied into the kernels. A level has the sizes nx, ny, nz,
	 * whether it is coarsened from the finer level in each direction, coarsenX, coarsenY, coarsenZ, the ranges of its cells,
	 * the cell widths dxc, dyc, dzc, and the arrays rhs, sol and res.
	 */
	template <typename T>
	struct mg
	{
		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;
		using hostLineArray = portableWrapper::hostArray<SAMS::T_dataType, 1>;
		using packRefs = SAMS::packRefsOf_t<typename T::dataPack>;

		/**
		 * @brief The sizes and arrays of one grid level
		 */
		struct level
		{
			/** @brief Number of cells in each direction */
			SAMS::T_sizeType nx, ny, nz;
			/** @brief The index of the level, counted from 0 for the finest level */
			SAMS::T_sizeType gridLevel = 0;
			/** @brief Whether the level is coarsened from the next finer level in each direction */
			bool coarsenX, coarsenY, coarsenZ;
			/** @brief Global index of the first cell of this rank in each direction, counted from 1 */
			SIGNED_INDEX_TYPE firstX = 1, firstY = 1, firstZ = 1;
			/** @brief Relaxation parameter of the smoother. 1 is Gauss-Seidel. The value here is the largest, and build lowers it on coarse levels */
			SAMS::T_dataType w = 1.6;
			/** @brief Range of the cells in each direction */
			pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
			/** @brief Range of the cell boundaries in each direction */
			pw::Range xbLocalDomainRange, ybLocalDomainRange, zbLocalDomainRange;
			/** @brief Cell widths in each direction */
			lineArray dxc, dyc, dzc;
			/** @brief Host copies of the cell widths */
			hostLineArray dxc_host, dyc_host, dzc_host;
			/** @brief Right hand side, solution and residual. On coarse levels: restricted residual and correction */
			volumeArray rhs, sol, res;
			/** @brief Names of the registered sol and res */
			std::string solName, resName;
			/** @brief The arrays of the problem for this level */
			typename T::levelData data;
		};

		/** @brief The levels, finest first */
		std::vector<level> levels;

		/** @brief The problem being solved */
		T problem;

		/** @brief The field that is evolved, which is rhs of the finest level */
		volumeArray field;

		/** @brief The array that receives the solution, which is the variable scratch and sol of the finest level */
		volumeArray solution;

		/** @brief Number of sweeps between convergence checks */
		int checkInterval = 3;

		/** @brief Number of sweeps done on each level in the last solve */
		std::vector<int> sweepCounts;

		/** @brief Number of times the last solve went down from the finest level, which is the number of cycles */
		int cycleCount = 0;

		/** @brief The index of the subproblem that is being solved */
		SAMS::T_sizeType currentProblem = 0;

		/** @brief Number of derived axes that are used in each direction */
		std::array<int, 3> axisLevels = {0, 0, 0};

		/** @brief The boundary conditions of each subproblem, active only while that subproblem is solved */
		std::vector<std::vector<std::pair<std::string, std::shared_ptr<SAMS::boundaryConditions>>>> problemBoundaries;

		/** @brief The harness */
		SAMS::harness *harnessPtr = nullptr;

		/**
		 * @brief The name of the axis of a direction on a level
		 *
		 * It is the derived axis of that level, or the last derived axis if the direction has fewer levels, or the base axis if it has none.
		 * @param baseName The name of the base axis
		 * @param direction The direction, 0 to 2
		 * @param level The level, counted from 1 for the first coarse level
		 */
		std::string levelAxisName(const std::string &baseName, int direction, int level) const
		{
			return parabolic::levelAxisName(std::string(T::memoryName.c_str()), baseName, axisLevels, direction, level);
		}

		/**
		 * @brief Set the cell widths of a level in one direction from its axis
		 *
		 * The guard widths are those of the first and last cells.
		 * @param harness The harness, which holds the axis registry
		 * @param axisName The name of the axis
		 * @param n Number of cells of the level in the direction
		 * @param first Global index of the first cell of this rank in the direction
		 * @param deviceWidths, hostWidths Widths to set, on the device and on the host
		 */
		void axisWidths(SAMS::harness &harness, const std::string &axisName, SAMS::T_sizeType n, SIGNED_INDEX_TYPE first, lineArray &deviceWidths, hostLineArray &hostWidths)
		{
			lineArray globalDevice;
			hostLineArray globalHost;
			harness.axisRegistry.fillPPDelta(axisName, globalDevice, SAMS::staggerType::CENTRED);
			harness.axisRegistry.fillPPDelta(axisName, globalHost, SAMS::staggerType::CENTRED);
			const lineArray source = globalDevice, widths = deviceWidths;
			auto interior = LAMBDA(SAMS::T_indexType i)
			{
				widths(i) = source(static_cast<SIGNED_INDEX_TYPE>(i) + first - 1);
			};
			pw::applyKernel(interior, pw::Range(1, n));
			pw::fence();
			auto lower = LAMBDA(SAMS::T_indexType i)
			{
				widths(i) = widths(i + 1);
			};
			pw::applyKernel(lower, pw::Range(0, 0));
			auto upper = LAMBDA(SAMS::T_indexType i)
			{
				widths(i) = widths(i - 1);
			};
			pw::applyKernel(upper, pw::Range(n + 1, n + 1));
			pw::fence();

			for (SAMS::T_indexType i = 1; i <= static_cast<SAMS::T_indexType>(n); ++i)
				hostWidths(i) = globalHost(static_cast<SIGNED_INDEX_TYPE>(i) + first - 1);
			hostWidths(0) = hostWidths(1);
			hostWidths(n + 1) = hostWidths(n);
		}

		/**
		 * @brief The optimal SOR parameter for the Laplacian on the cells of a level, from the spectral radius of Jacobi iteration
		 * @param l The level, of which the sizes are used
		 */
		SAMS::T_dataType optimalOmega(const level &l) const
		{
			const SAMS::T_dataType pi = 3.14159265358979323846;
			const SAMS::T_dataType rho = (cos(pi / (l.nx + 1)) + cos(pi / (l.ny + 1)) + cos(pi / (l.nz + 1))) / 3.0;
			return 2.0 / (1.0 + sqrt(1.0 - rho * rho));
		}

		/**
		 * @brief Build the levels from the registered axes and variables of each level
		 *
		 * A direction with fewer levels than the others is left as it is on the coarser levels.
		 * @param harness The harness, which holds the registries
		 */
		void build(SAMS::harness &harness)
		{
			auto &axisReg = harness.axisRegistry;
			const SAMS::variableDef &fieldDef = harness.variableRegistry.getVariable(T::fieldName());
			const std::string xName = fieldDef.getDimension(0).axisName;
			const std::string yName = fieldDef.getDimension(1).axisName;
			const std::string zName = fieldDef.getDimension(2).axisName;

			const auto axisRanks = harness.MPIManager.getCoords();
			auto firstIndex = [&](const std::string &axisName, int direction)
			{
				const int axis = static_cast<int>(fieldDef.getDimension(direction).mpiAxis);
				return static_cast<SIGNED_INDEX_TYPE>(axisReg.getAxis(axisName).decomposition.firstIndexOnRank(axis >= 0 ? axisRanks[axis] : 0));
			};

			level l;
			l.nx = fieldDef.getDimension(0).getLocalDomainCells();
			l.ny = fieldDef.getDimension(1).getLocalDomainCells();
			l.nz = fieldDef.getDimension(2).getLocalDomainCells();
			l.coarsenX = l.coarsenY = l.coarsenZ = false;
			l.firstX = firstIndex(xName, 0);
			l.firstY = firstIndex(yName, 1);
			l.firstZ = firstIndex(zName, 2);
			l.rhs = field;
			l.sol = solution;
			l.xcLocalDomainRange = fieldDef.getDimension(0).getLocalDomainRange(SAMS::staggerType::CENTRED);
			l.ycLocalDomainRange = fieldDef.getDimension(1).getLocalDomainRange(SAMS::staggerType::CENTRED);
			l.zcLocalDomainRange = fieldDef.getDimension(2).getLocalDomainRange(SAMS::staggerType::CENTRED);
			l.xbLocalDomainRange = fieldDef.getDimension(0).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
			l.ybLocalDomainRange = fieldDef.getDimension(1).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
			l.zbLocalDomainRange = fieldDef.getDimension(2).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
			axisReg.fillPPLocalDelta(xName, l.dxc, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalDelta(yName, l.dyc, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalDelta(zName, l.dzc, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalDelta(xName, l.dxc_host, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalDelta(yName, l.dyc_host, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalDelta(zName, l.dzc_host, SAMS::staggerType::CENTRED);
			l.solName = "scratch";
			l.resName = std::string(T::memoryName.c_str()) + "_res_0";
			harness.variableRegistry.fillPPArray(l.resName, l.res);
			l.w = std::min(l.w, optimalOmega(l));
			levels.push_back(l);

			const std::string memory = T::memoryName.c_str();
			const std::string baseNames[3] = {xName, yName, zName};
			const int levelCount = std::max(std::max(axisLevels[0], axisLevels[1]), axisLevels[2]);

			for (int n = 1; n <= levelCount; ++n)
			{
				std::string axisNames[3];
				for (int d = 0; d < 3; ++d)
					axisNames[d] = levelAxisName(baseNames[d], d, n);
				const std::string suffix = "_" + std::to_string(n);
				const SAMS::variableDef &levelDef = harness.variableRegistry.getVariable(memory + "_sol" + suffix);

				level l;
				l.gridLevel = static_cast<SAMS::T_sizeType>(n);
				l.coarsenX = n <= axisLevels[0];
				l.coarsenY = n <= axisLevels[1];
				l.coarsenZ = n <= axisLevels[2];
				l.firstX = firstIndex(axisNames[0], 0);
				l.firstY = firstIndex(axisNames[1], 1);
				l.firstZ = firstIndex(axisNames[2], 2);
				l.nx = levelDef.getDimension(0).getLocalDomainCells();
				l.ny = levelDef.getDimension(1).getLocalDomainCells();
				l.nz = levelDef.getDimension(2).getLocalDomainCells();
				l.xcLocalDomainRange = levelDef.getDimension(0).getLocalDomainRange(SAMS::staggerType::CENTRED);
				l.ycLocalDomainRange = levelDef.getDimension(1).getLocalDomainRange(SAMS::staggerType::CENTRED);
				l.zcLocalDomainRange = levelDef.getDimension(2).getLocalDomainRange(SAMS::staggerType::CENTRED);
				l.xbLocalDomainRange = levelDef.getDimension(0).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
				l.ybLocalDomainRange = levelDef.getDimension(1).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
				l.zbLocalDomainRange = levelDef.getDimension(2).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
				harness.variableRegistry.fillPPArray(memory + "_rhs" + suffix, l.rhs);
				harness.variableRegistry.fillPPArray(memory + "_sol" + suffix, l.sol);
				harness.variableRegistry.fillPPArray(memory + "_res" + suffix, l.res);
				l.solName = memory + "_sol" + suffix;
				l.resName = memory + "_res" + suffix;
				manager.allocate(l.dxc, pw::Range(0, l.nx + 1));
				manager.allocate(l.dyc, pw::Range(0, l.ny + 1));
				manager.allocate(l.dzc, pw::Range(0, l.nz + 1));
				manager.allocate(l.dxc_host, pw::Range(0, l.nx + 1));
				manager.allocate(l.dyc_host, pw::Range(0, l.ny + 1));
				manager.allocate(l.dzc_host, pw::Range(0, l.nz + 1));
				axisWidths(harness, axisNames[0], l.nx, l.firstX, l.dxc, l.dxc_host);
				axisWidths(harness, axisNames[1], l.ny, l.firstY, l.dyc, l.dyc_host);
				axisWidths(harness, axisNames[2], l.nz, l.firstZ, l.dzc, l.dzc_host);
				l.w = std::min(l.w, optimalOmega(l));
				levels.push_back(l);
			}
		}

		/**
		 * @brief Calculate the residual of a level into l.res
		 * @param l The level
		 */
		void computeResidual(level &l)
		{
			const auto s = problem.makeStencil(l, currentProblem);
			const volumeArray u = l.sol, rhs = l.rhs, res = l.res;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;

			auto kernel = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				SAMS::T_dataType divergence, diagonal;
				evaluateStencil<T>(s, u, dx, dy, dz, i, j, k, divergence, diagonal);
				res(i, j, k) = s.beta(i, j, k) * (rhs(i, j, k) - u(i, j, k)) + divergence;
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
		}

		/**
		 * @brief The largest magnitude of the residual of a level
		 * @param l The level
		 * @return The maximum of |res| over all ranks
		 */
		SAMS::T_dataType residualNorm(level &l)
		{
			computeResidual(l);
			const volumeArray res = l.res;
			SAMS::T_dataType norm = pw::maxval(LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) { return std::abs(res(i, j, k)); },
				l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
#ifdef USE_MPI
			MPI_Allreduce(MPI_IN_PLACE, &norm, 1, SAMS::gettypeRegistry().getMPIType<SAMS::T_dataType>(), MPI_MAX, harnessPtr->MPIManager.getComm());
#endif
			return norm;
		}

		/**
		 * @brief One eight colour SOR sweep of l.sol, with the relaxation parameter l.w
		 *
		 * Eight colours are used because the stencil includes the diagonal neighbours.
		 * @param l The level
		 */
		void smooth(level &l)
		{
			const auto s = problem.makeStencil(l, currentProblem);
			const volumeArray u = l.sol, rhs = l.rhs;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
			const SAMS::T_dataType w = l.w;

			auto kernel = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				SAMS::T_dataType divergence, diagonal;
				evaluateStencil<T>(s, u, dx, dy, dz, i, j, k, divergence, diagonal);
				const SAMS::T_dataType factor = s.beta(i, j, k);
				const SAMS::T_dataType r = factor * (rhs(i, j, k) - u(i, j, k)) + divergence;
				u(i, j, k) += w * r / (factor + diagonal);
			};
			pw::applyEightColourKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
		}

		/**
		 * @brief Restrict the residual of a level to the right hand side of the next coarser level
		 *
		 * The mean of the fine cells covered by a coarse cell, scaled by the coarse beta. The coarse correction is set to zero.
		 * @param levelSrc Index of the fine level
		 * @param levelDest Index of the coarse level
		 */
		void restrict(int levelSrc, int levelDest)
		{
			level &src = levels[levelSrc];
			level &dest = levels[levelDest];

			harnessPtr->variableRegistry.applyBoundaryConditions(src.resName);
			const auto s = problem.makeStencil(dest, currentProblem);
			auto scale = LAMBDA(SAMS::T_dataType mean, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				return mean / s.beta(i, j, k);
			};
			averageCells(src.res, dest.rhs, src, dest, scale);

			pw::assign(dest.sol, 0.0);
		}

		/**
		 * @brief Interpolate the correction onto the next finer level
		 * @param levelSrc Index of the coarse level
		 * @param levelDest Index of the fine level
		 */
		void prolongate(int levelSrc, int levelDest)
		{
			level &src = levels[levelSrc];
			level &dest = levels[levelDest];

			const volumeArray coarse = src.sol, fine = dest.sol;
			const bool coarsenX = src.coarsenX, coarsenY = src.coarsenY, coarsenZ = src.coarsenZ;
			const SAMS::T_indexType fineFirstX = dest.firstX, fineFirstY = dest.firstY, fineFirstZ = dest.firstZ;
			const SAMS::T_indexType coarseFirstX = src.firstX, coarseFirstY = src.firstY, coarseFirstZ = src.firstZ;
			auto kernel = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				// In a direction that is not coarsened the coarse cell is the same cell, with weight 1 and none on the neighbour.
				// In a direction that is coarsened the coarse cell and the neighbour come from the global index of the fine cell
				const SAMS::T_indexType gi = fineFirstX + i - 1, gj = fineFirstY + j - 1, gk = fineFirstZ + k - 1;
				const SAMS::T_indexType ic = coarsenX ? (gi + 1) / 2 - (coarseFirstX - 1) : i;
				const SAMS::T_indexType jc = coarsenY ? (gj + 1) / 2 - (coarseFirstY - 1) : j;
				const SAMS::T_indexType kc = coarsenZ ? (gk + 1) / 2 - (coarseFirstZ - 1) : k;
				const SAMS::T_indexType in = coarsenX ? ((gi % 2 == 1) ? ic - 1 : ic + 1) : ic;
				const SAMS::T_indexType jn = coarsenY ? ((gj % 2 == 1) ? jc - 1 : jc + 1) : jc;
				const SAMS::T_indexType kn = coarsenZ ? ((gk % 2 == 1) ? kc - 1 : kc + 1) : kc;
				const SAMS::T_dataType wx = coarsenX ? 0.75 : 1.0, wxn = coarsenX ? 0.25 : 0.0;
				const SAMS::T_dataType wy = coarsenY ? 0.75 : 1.0, wyn = coarsenY ? 0.25 : 0.0;
				const SAMS::T_dataType wz = coarsenZ ? 0.75 : 1.0, wzn = coarsenZ ? 0.25 : 0.0;
				fine(i, j, k) += wx * (wy * (wz * coarse(ic, jc, kc) + wzn * coarse(ic, jc, kn)) +
					wyn * (wz * coarse(ic, jn, kc) + wzn * coarse(ic, jn, kn))) +
					wxn * (wy * (wz * coarse(in, jc, kc) + wzn * coarse(in, jc, kn)) +
					wyn * (wz * coarse(in, jn, kc) + wzn * coarse(in, jn, kn)));
			};
			pw::applyKernel(kernel, dest.xcLocalDomainRange, dest.ycLocalDomainRange, dest.zcLocalDomainRange);
			pw::fence();
			harnessPtr->variableRegistry.applyBoundaryConditions(dest.solName);
		}

		/**
		 * @brief Red-black Gauss-Seidel sweeps on a level until the residual is small enough
		 *
		 * The residual is checked every checkInterval sweeps and after the last sweep, so l.res is current on return.
		 * @param levelIndex Index of the level
		 * @param iterations Limit on the sweeps: iterations - 1 sweeps are done at most
		 * @param tol Absolute residual below which it stops. Zero disables the check
		 * @return True if the residual fell below tol
		 */
		bool GS(int levelIndex, int iterations, SAMS::T_dataType tol = 0.0)
		{
			level &l = levels[levelIndex];
			bool converged = false;

			for (int iter = 1; iter < iterations; ++iter)
			{
				smooth(l);
				harnessPtr->variableRegistry.applyBoundaryConditions(l.solName);
				++sweepCounts[levelIndex];

				if (iter % checkInterval == 0 || iter == iterations - 1)
				{
					const SAMS::T_dataType max_delta = residualNorm(l);

					if (tol > 0.0 && max_delta < tol)
					{
						converged = true;
						break;
					}
				}
			}
			return converged;
		}

#ifdef EXACT_FLUX_CONSERVATION
		/**
		 * @brief Set the field from the old field and the net flux into each cell from the solved field
		 *
		 * The faces give the same flux to both cells that share them, so the total is conserved exactly, however well the
		 * solve converged.
		 */
		void conservativeUpdate()
		{
			level &l = levels[0];
			const auto s = problem.makeStencil(l, currentProblem);
			const volumeArray u = l.sol, field = l.rhs;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;

			auto kernel = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				SAMS::T_dataType divergence, diagonal;
				evaluateStencil<T>(s, u, dx, dy, dz, i, j, k, divergence, diagonal);
				field(i, j, k) += divergence / s.beta(i, j, k);
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
			pw::fence();
		}
#endif

		/**
		 * @brief Solve for the new field
		 * @param ts The timestep data, which the problem needs for its coefficients
		 * @param sweeps Sweep limit for each visit to a level other than the coarsest
		 * @param coarsestSweeps Sweep limit for each visit to the coarsest level
		 * @param tol Tolerance relative to the initial residual. The same absolute residual is used on the coarser levels
		 * @param packs The data packs of the problem, which it needs to set its coefficients
		 */
		void solve(const SAMS::timeState &ts, int sweeps, int coarsestSweeps, SAMS::T_dataType tol, packRefs packs)
		{

			callWithFixedAndInferred(problem, &T::template prepareSolve<level>, packs, levels, currentProblem, ts);

			const SAMS::T_dataType initialResidual = residualNorm(levels[0]);
			if (initialResidual == 0.0)
			{
				pw::assign(levels[0].rhs, levels[0].sol);
				return;
			}
			const SAMS::T_dataType absoluteTolerance = tol * initialResidual;

			sweepCounts.assign(levels.size(), 0);
			cycleCount = 0;

			int levelSrc = 0;
			bool converged = false;
			while (true)
			{
				int cycles = levelSrc == static_cast<SAMS::T_indexType>(levels.size()) - 1 ? coarsestSweeps : sweeps;
				converged = GS(levelSrc, cycles, absoluteTolerance);
				// Converged: done on the finest level, otherwise prolongate up
				if (converged) {
					if (levelSrc == 0) break;
					int levelDest = levelSrc - 1;
					prolongate(levelSrc, levelDest);
					levelSrc--;
					continue;
				}
				// Not converged: restrict down
				int levelDest = levelSrc + 1;
				if (levelDest >= static_cast<SAMS::T_indexType>(levels.size())) continue;
				if (levelSrc == 0) ++cycleCount;
				restrict(levelSrc, levelDest);
				levelSrc++;
			}
#ifdef EXACT_FLUX_CONSERVATION
			conservativeUpdate();
#else
			pw::assign(levels[0].rhs, levels[0].sol);
#endif

			int totalSweeps = 0;
			for (int count : sweepCounts) totalSweeps += count;
			SAMS::debug3 << "Multigrid cycles: " << cycleCount << ", sweeps: " << totalSweeps << " (per level, finest first:";
			for (int count : sweepCounts) SAMS::debug3 << " " << count;
			SAMS::debug3 << ")\n";
		}

		inline static constexpr auto nameType = T::name + SAMS::makeName("_mg");
		constexpr static std::string_view name = nameType;
		constexpr static bool coreSimulation = false;
		constexpr static bool timeSimulation = true;
		using dataPack = SAMS::dataPacks::multiPack<SAMS::deckHostData<3>, typename T::dataPack>;

		/** @brief Activate the deck services, which hold the host data of the deck for expressions */
		template <typename T_runner>
		void runnerInteraction(T_runner &runnerRef)
		{
			runnerRef.activatePackage("deckServices");
		}

		/** @brief Array manager of the harness */
		pw::portableArrayManager &manager;

		/**
		 * @brief Construct the package
		 * @param hr The harness, which provides the array manager
		 */
		mg(SAMS::harness &hr) : manager(hr.memoryRegistry.getArrayManager()) {};

		/**
		 * @brief Register the derived axes of the coarse levels
		 *
		 * The last axis of each direction is the first with 4 or fewer cells on some rank, and is not used.
		 * @param harness The harness, which holds the registries
		 */
		void registerSecondaryAxes(SAMS::harness &harness)
		{
			auto &axisReg = harness.axisRegistry;
			const SAMS::variableDef &fieldDef = harness.variableRegistry.getVariable(T::fieldName());
			const std::string memory = T::memoryName.c_str();

			auto minLocalCells = [&](const std::string &axisName)
			{
				long long localCells = static_cast<long long>(axisReg.getLocalDomainElements(axisName, SAMS::staggerType::CENTRED));
#ifdef USE_MPI
				MPI_Allreduce(MPI_IN_PLACE, &localCells, 1, MPI_LONG_LONG, MPI_MIN, harness.MPIManager.getComm());
#endif
				return localCells;
			};

			for (int d = 0; d < 3; ++d)
			{
				const std::string baseName = fieldDef.getDimension(d).axisName;
				std::string sourceName = baseName;
				SAMS::T_sizeType cells = axisReg.getDomainElements(baseName, SAMS::staggerType::CENTRED);
				long long localCells = minLocalCells(sourceName);
				int created = 0;
				for (int level = 1; localCells > 4; ++level)
				{
					cells /= 2;
					const std::string name = memory + "_" + baseName + "_" + std::to_string(level);
					axisReg.deriveAxis(name, sourceName, cells);
					axisReg.decomposeAxis(name);
					localCells = minLocalCells(name);
					sourceName = name;
					++created;
				}
				axisLevels[d] = std::max(created - 1, 0);
			}
		}

		/**
		 * @brief Register rhs, sol and res of each level on the derived axes
		 *
		 * A direction with fewer levels than the others stays on its last axis.
		 * @param harness The harness, which holds the variable registry
		 * @param packs The data packs of the problem
		 */
		void registerSecondaryVariables(SAMS::harness &harness, packRefs packs)
		{
			auto &varRegistry = harness.variableRegistry;
			const SAMS::variableDef &fieldDef = varRegistry.getVariable(T::fieldName());
			const std::string memory = T::memoryName.c_str();
			const int ghosts = 1;
			const int levelCount = std::max(std::max(axisLevels[0], axisLevels[1]), axisLevels[2]);

			callWithFixedAndInferred(problem, &T::registerSecondaryVariables, packs, harness, axisLevels);

			varRegistry.registerVariable<SAMS::T_dataType>(memory + "_res_0", pw::arrayTags::accelerated, SAMS::dimension(fieldDef.getDimension(0).axisName, ghosts), SAMS::dimension(fieldDef.getDimension(1).axisName, ghosts), SAMS::dimension(fieldDef.getDimension(2).axisName, ghosts));

			for (int level = 1; level <= levelCount; ++level)
			{
				std::string axisNames[3];
				for (int d = 0; d < 3; ++d)
				{
					const std::string baseName = fieldDef.getDimension(d).axisName;
					axisNames[d] = levelAxisName(baseName, d, level);
				}
				for (const char *variable : {"rhs", "sol", "res"})
				{
					const std::string name = memory + "_" + variable + "_" + std::to_string(level);
					varRegistry.registerVariable<SAMS::T_dataType>(name, pw::arrayTags::accelerated, SAMS::dimension(axisNames[0].c_str(), ghosts), SAMS::dimension(axisNames[1].c_str(), ghosts), SAMS::dimension(axisNames[2].c_str(), ghosts));
				}
			}
		}

		/**
		 * @brief Set the default parameters of the problem
		 * @param kvs The key value store
		 */
		void defaultValues(SAMS::KVS &kvs, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::defaultValues, packs, kvs);
		}

		/**
		 * @brief Register scratch, which holds the solution
		 * @param harness The harness, which holds the variable registry
		 */
		void registerVariables(SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::registerVariables, packs, harness);
			auto &varRegistry = harness.variableRegistry;
			const int ghosts = 1;
			varRegistry.registerVariable<SAMS::T_dataType>("scratch", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
		}

		/**
		 * @brief Set the boundary conditions of the problem, and of the temperature, scratch and the coarse levels for each subproblem
		 *
		 * Periodic, zero gradient and the mirrors have their normal meaning on all levels. fixedGhosts uses the initial energy from the core solver as a boundary condition on the temperature. Clamp and expression set the ghost cells of the temperature, which are then fixed in the same way
		 * @param harness The harness, which holds the variable registry
		 * @param kvs The key value store, from which the problem reads its boundary types
		 * @param hostData The host data of the deck, for expressions
		 * @param tData The time state, for expressions
		 */
		void setBoundaryConditions(SAMS::harness &harness, SAMS::KVS &kvs, [[maybe_unused]] SAMS::deckHostData<3> &hostData, [[maybe_unused]] SAMS::timeState &tData, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::setBoundaryConditions, packs, harness);
			auto &varRegistry = harness.variableRegistry;
			using clamp = SAMS::simpleClamp<SAMS::T_dataType, 3>;
			using periodic = SAMS::simplePeriodicBC<SAMS::T_dataType, 3>;
			using zeroGradient = SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>;
			using fixedMirror = SAMS::fixedMirrorBC<SAMS::T_dataType, 3>;
			using floatingMirror = SAMS::floatingMirrorBC<SAMS::T_dataType, 3>;
			const std::string memory = T::memoryName.c_str();
			const int levelCount = std::max(std::max(axisLevels[0], axisLevels[1]), axisLevels[2]);
			SAMS::variableDef &fieldDef = varRegistry.getVariable(T::fieldName());
			problemBoundaries.assign(problem.problemCount(), {});
			for (SAMS::T_sizeType n = 0; n < problem.problemCount(); ++n)
			{
				for (int d = 0; d < 3; ++d)
				{
					for (SAMS::domain::edges edge : {SAMS::domain::edges::lower, SAMS::domain::edges::upper})
					{
						const parabolic::tcBoundaryType boundary = problem.getBoundaryType(kvs, n, d, edge);
						auto attach = [&](const std::string &name, bool coarse)
						{
							SAMS::variableDef &varDef = varRegistry.getVariable(name);
							if (boundary.type == parabolic::boundaryType::periodic)
								problemBoundaries[n].emplace_back(name, varDef.addBoundaryCondition(d, edge, periodic(varDef)));
							else if (boundary.type == parabolic::boundaryType::zeroGradient)
								problemBoundaries[n].emplace_back(name, varDef.addBoundaryCondition(d, edge, zeroGradient(varDef)));
							else if (boundary.type == parabolic::boundaryType::fixedMirror)
								problemBoundaries[n].emplace_back(name, varDef.addBoundaryCondition(d, edge, fixedMirror(varDef, coarse ? 0.0 : boundary.value)));
							else if (boundary.type == parabolic::boundaryType::floatingMirror)
								problemBoundaries[n].emplace_back(name, varDef.addBoundaryCondition(d, edge, floatingMirror(varDef)));
							else if (coarse)
								problemBoundaries[n].emplace_back(name, varDef.addBoundaryCondition(d, edge, clamp(varDef, 0.0)));
						};
						attach("scratch", false);
						for (int level = 1; level <= levelCount; ++level)
						{
							for (const char *variable : {"rhs", "sol", "res"})
								attach(memory + "_" + variable + "_" + std::to_string(level), true);
						}
						if (boundary.type == parabolic::boundaryType::clamp)
							problemBoundaries[n].emplace_back(T::fieldName(), fieldDef.addBoundaryCondition(d, edge, clamp(fieldDef, boundary.value)));
						else if (boundary.type == parabolic::boundaryType::expression)
						{
#ifdef USE_DECK_PARSER
							eis::stack<SAMS::T_dataType> stack;
							eis::error errcode = eis::err_none;
							hostData.parser->tokenize(boundary.text, stack, errcode);
							if (errcode != eis::err_none) harness.abort("Invalid thermal conduction boundary expression: " + boundary.text + "\n" + SAMS::parserErrors(*hostData.parser), false);
							hostData.addPositions(harness, fieldDef);
							problemBoundaries[n].emplace_back(T::fieldName(), fieldDef.addBoundaryCondition(d, edge, SAMS::expressionBC<3>(fieldDef, harness, tData, hostData, stack)));
#else
							harness.abort("Invalid thermal conduction boundary expression: " + boundary.text + "\nExpressions need the deck parser", false);
#endif
						}
					}
				}
			}
		}

		/**
		 * @brief Set the constants of the problem
		 * @param harness The harness
		 */
		void initialConditions(SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::initialConditions, packs, harness);
		}

		/**
		 * @brief Get the variables and build the levels
		 * @param harness The harness, which holds the registries
		 */
		void getVariables(SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::getVariables, packs, harness);
			harness.variableRegistry.fillPPArray(T::fieldName(), field);
			harness.variableRegistry.fillPPArray("scratch", solution);
			harnessPtr = &harness;
			build(harness);
			callWithFixedAndInferred(problem, &T::template getSecondaryVariables<level>, packs, harness, levels);
		}

		/**
		 * @brief Advance the field by one step, solving each subproblem of the problem in turn
		 * @param ts The timestep data
		 * @param harness The harness, which holds the variable registry
		 */
		void endOfTimestep(const SAMS::timeState &ts, SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::beforeAllSolves, packs, harness);
			for (SAMS::T_sizeType n = 0; n < problem.problemCount(); ++n)
			{
				currentProblem = n;
				for (auto &boundaries : problemBoundaries)
				{
					for (auto &[name, bc] : boundaries)
						harness.variableRegistry.deactivateBoundaryCondition(name, bc);
				}
				for (auto &[name, bc] : problemBoundaries[n])
					harness.variableRegistry.activateBoundaryCondition(name, bc);
				callWithFixedAndInferred(problem, &T::beforeSolve, packs, harness, n);
				harness.variableRegistry.applyBoundaryConditions(T::fieldName());
				pw::assign(solution, field);
				harness.variableRegistry.applyBoundaryConditions("scratch");
				solve(ts, 5, 10000, SOLVER_TOLERANCE, packs);
				callWithFixedAndInferred(problem, &T::afterSolve, packs, harness, n);
				SAMS::debug3 << pw::sum(field) << "\n";
			}
			callWithFixedAndInferred(problem, &T::afterAllSolves, packs, harness);
		}
	};

	}
}

#endif
