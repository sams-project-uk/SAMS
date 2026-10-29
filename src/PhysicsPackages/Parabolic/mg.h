#ifndef MULTIGRIDTEST_H
#define MULTIGRIDTEST_H
#include "runner.h"
#include "harness.h"
#include "builtInBoundaryConditions.h"
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
	DEVICEPREFIX void evaluateStencil(const T_stencil &s, const T_array &u, const T_line &dx, const T_line &dy, const T_line &dz, size_t i, size_t j, size_t k, double &divergence, double &diagonal)
	{
		double a[T::stencilSize];
		s.coefficients(i, j, k, a);
		double sum = 0.0, centre = 0.0;
		for (int n = 0; n < T::stencilSize; ++n)
		{
			int di, dj, dk;
			T::stencilCellOffset(n, di, dj, dk);
			sum += a[n] * u(static_cast<SIGNED_INDEX_TYPE>(i) + di, static_cast<SIGNED_INDEX_TYPE>(j) + dj, static_cast<SIGNED_INDEX_TYPE>(k) + dk);
			if (di == 0 && dj == 0 && dk == 0)
				centre = a[n];
		}
		const double volume = dx(i) * dy(j) * dz(k);
		divergence = sum / volume;
		diagonal = -centre / volume;
	}

	/**
	 * @brief Geometric multigrid solver with eight colour Gauss-Seidel
	 *
	 * Solves (V beta/dt)(u - u_old) = the net heat into the cell from the operator of the problem, for the new field u,
	 * where V is the cell volume. The solution is in the variable scratch, which is sol of the finest level.
	 * @tparam T The problem. Provides:
	 * - name: a SAMS::constexprName, which is added to the name of the package.
	 * - fieldName(): the name of the variable that is evolved, whose array is rhs of the finest level.
	 * - levelData: a default constructible type holding the arrays of the problem for one level, which is the member data
	 *   of a level.
	 * - allocate(manager, level, finest): allocate the arrays of the problem for a level.
	 * - prepareSolve(levels): set the coefficients of the operator on every level, from the old field, which is rhs
	 *   of the finest level, and any other data that the problem holds.
	 * - makeStencil(level): return an object to be copied into kernels, with coefficients(i, j, k, a) and beta(i, j, k).
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
			size_t nx, ny, nz;
			/** @brief The index of the level, counted from 0 for the finest level */
			SAMS::T_sizeType gridLevel = 0;
			/** @brief Whether the level is coarsened from the next finer level in each direction */
			bool coarsenX, coarsenY, coarsenZ;
			/** @brief Global index of the first cell of this rank in each direction, counted from 1 */
			SIGNED_INDEX_TYPE firstX = 1, firstY = 1, firstZ = 1;
			/** @brief Relaxation parameter of the smoother. 1 is Gauss-Seidel. The value here is the largest, and build lowers it on coarse levels */
			double w = 1.6;
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

		/** @brief Number of derived axes that are used in each direction */
		std::array<int, 3> axisLevels = {0, 0, 0};

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
		void axisWidths(SAMS::harness &harness, const std::string &axisName, size_t n, SIGNED_INDEX_TYPE first, lineArray &deviceWidths, hostLineArray &hostWidths)
		{
			lineArray globalDevice;
			hostLineArray globalHost;
			harness.axisRegistry.fillPPDelta(axisName, globalDevice, SAMS::staggerType::CENTRED);
			harness.axisRegistry.fillPPDelta(axisName, globalHost, SAMS::staggerType::CENTRED);
			const lineArray source = globalDevice, widths = deviceWidths;
			auto interior = LAMBDA(size_t i)
			{
				widths(i) = source(static_cast<SIGNED_INDEX_TYPE>(i) + first - 1);
			};
			pw::applyKernel(interior, pw::Range(1, n));
			pw::fence();
			auto lower = LAMBDA(size_t i)
			{
				widths(0) = widths(1);
			};
			pw::applyKernel(lower, pw::Range(0, 0));
			auto upper = LAMBDA(size_t i)
			{
				widths(n + 1) = widths(n);
			};
			pw::applyKernel(upper, pw::Range(n + 1, n + 1));
			pw::fence();

			for (size_t i = 1; i <= n; ++i)
				hostWidths(i) = globalHost(static_cast<SIGNED_INDEX_TYPE>(i) + first - 1);
			hostWidths(0) = hostWidths(1);
			hostWidths(n + 1) = hostWidths(n);
		}

		/**
		 * @brief The optimal SOR parameter for the Laplacian on the cells of a level, from the spectral radius of Jacobi iteration
		 * @param l The level, of which the sizes are used
		 */
		double optimalOmega(const level &l) const
		{
			const double pi = 3.14159265358979323846;
			const double rho = (cos(pi / (l.nx + 1)) + cos(pi / (l.ny + 1)) + cos(pi / (l.nz + 1))) / 3.0;
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
		 * @param dt Timestep
		 */
		void computeResidual(level &l, double dt)
		{
			const auto s = problem.makeStencil(l);
			const volumeArray u = l.sol, rhs = l.rhs, res = l.res;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;

			auto kernel = LAMBDA(size_t i, size_t j, size_t k)
			{
				double divergence, diagonal;
				evaluateStencil<T>(s, u, dx, dy, dz, i, j, k, divergence, diagonal);
				res(i, j, k) = s.beta(i, j, k) * (rhs(i, j, k) - u(i, j, k)) / dt + divergence;
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
		}

		/**
		 * @brief The largest magnitude of the residual of a level
		 * @param l The level
		 * @param dt Timestep
		 * @return The maximum of |res| over all ranks
		 */
		double residualNorm(level &l, double dt)
		{
			computeResidual(l, dt);
			const volumeArray res = l.res;
			double norm = pw::maxval(LAMBDA(size_t i, size_t j, size_t k) { return std::abs(res(i, j, k)); },
				l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
#ifdef USE_MPI
			MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_MAX, harnessPtr->MPIManager.getComm());
#endif
			return norm;
		}

		/**
		 * @brief One eight colour SOR sweep of l.sol, with the relaxation parameter l.w
		 *
		 * Eight colours are used because the stencil includes the diagonal neighbours.
		 * @param l The level
		 * @param dt Timestep
		 */
		void smooth(level &l, double dt)
		{
			const auto s = problem.makeStencil(l);
			const volumeArray u = l.sol, rhs = l.rhs;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
			const double w = l.w;

			auto kernel = LAMBDA(size_t i, size_t j, size_t k)
			{
				double divergence, diagonal;
				evaluateStencil<T>(s, u, dx, dy, dz, i, j, k, divergence, diagonal);
				const double factor = s.beta(i, j, k);
				const double r = factor * (rhs(i, j, k) - u(i, j, k)) / dt + divergence;
				u(i, j, k) += w * r / (factor / dt + diagonal);
			};
			pw::applyEightColourKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
		}

		/**
		 * @brief Restrict the residual of a level to the right hand side of the next coarser level
		 *
		 * The mean of the fine cells covered by a coarse cell, scaled by dt and the coarse beta. The coarse correction is set to zero.
		 * @param levelSrc Index of the fine level
		 * @param levelDest Index of the coarse level
		 * @param dt Timestep
		 */
		void restrict(int levelSrc, int levelDest, double dt)
		{
			level &src = levels[levelSrc];
			level &dest = levels[levelDest];

			harnessPtr->variableRegistry.applyBoundaryConditions(src.resName);
			const auto s = problem.makeStencil(dest);
			auto scale = LAMBDA(double mean, size_t i, size_t j, size_t k)
			{
				return dt * mean / s.beta(i, j, k);
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
			const size_t fineFirstX = dest.firstX, fineFirstY = dest.firstY, fineFirstZ = dest.firstZ;
			const size_t coarseFirstX = src.firstX, coarseFirstY = src.firstY, coarseFirstZ = src.firstZ;
			auto kernel = LAMBDA(size_t i, size_t j, size_t k)
			{
				// In a direction that is not coarsened the coarse cell is the same cell, with weight 1 and none on the neighbour.
				// In a direction that is coarsened the coarse cell and the neighbour come from the global index of the fine cell
				const size_t gi = fineFirstX + i - 1, gj = fineFirstY + j - 1, gk = fineFirstZ + k - 1;
				const size_t ic = coarsenX ? (gi + 1) / 2 - (coarseFirstX - 1) : i;
				const size_t jc = coarsenY ? (gj + 1) / 2 - (coarseFirstY - 1) : j;
				const size_t kc = coarsenZ ? (gk + 1) / 2 - (coarseFirstZ - 1) : k;
				const size_t in = coarsenX ? ((gi % 2 == 1) ? ic - 1 : ic + 1) : ic;
				const size_t jn = coarsenY ? ((gj % 2 == 1) ? jc - 1 : jc + 1) : jc;
				const size_t kn = coarsenZ ? ((gk % 2 == 1) ? kc - 1 : kc + 1) : kc;
				const double wx = coarsenX ? 0.75 : 1.0, wxn = coarsenX ? 0.25 : 0.0;
				const double wy = coarsenY ? 0.75 : 1.0, wyn = coarsenY ? 0.25 : 0.0;
				const double wz = coarsenZ ? 0.75 : 1.0, wzn = coarsenZ ? 0.25 : 0.0;
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
		 * @param dt Timestep
		 * @param iterations Limit on the sweeps: iterations - 1 sweeps are done at most
		 * @param tol Absolute residual below which it stops. Zero disables the check
		 * @return True if the residual fell below tol
		 */
		bool GS(int levelIndex, double dt, int iterations, double tol = 0.0)
		{
			level &l = levels[levelIndex];
			bool converged = false;

			for (int iter = 1; iter < iterations; ++iter)
			{
				smooth(l, dt);
				harnessPtr->variableRegistry.applyBoundaryConditions(l.solName);
				++sweepCounts[levelIndex];

				if (iter % checkInterval == 0 || iter == iterations - 1)
				{
					const double max_delta = residualNorm(l, dt);

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
		 * @param dt Timestep
		 */
		void conservativeUpdate(double dt)
		{
			level &l = levels[0];
			const auto s = problem.makeStencil(l);
			const volumeArray u = l.sol, field = l.rhs;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;

			auto kernel = LAMBDA(size_t i, size_t j, size_t k)
			{
				double divergence, diagonal;
				evaluateStencil<T>(s, u, dx, dy, dz, i, j, k, divergence, diagonal);
				field(i, j, k) += dt * divergence / s.beta(i, j, k);
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
			pw::fence();
		}
#endif

		/**
		 * @brief Solve for the new field
		 * @param dt Timestep
		 * @param sweeps Sweep limit for each visit to a level other than the coarsest
		 * @param coarsestSweeps Sweep limit for each visit to the coarsest level
		 * @param tol Tolerance relative to the initial residual. The same absolute residual is used on the coarser levels
		 * @param packs The data packs of the problem, which it needs to set its coefficients
		 */
		void solve(double dt, int sweeps, int coarsestSweeps, double tol, packRefs packs)
		{

			callWithFixedAndInferred(problem, &T::template prepareSolve<level>, packs, levels);

			const double initialResidual = residualNorm(levels[0], dt);
			if (initialResidual == 0.0)
			{
				pw::assign(levels[0].rhs, levels[0].sol);
				return;
			}
			const double absoluteTolerance = tol * initialResidual;

			sweepCounts.assign(levels.size(), 0);
			cycleCount = 0;

			int levelSrc = 0;
			bool converged = false;
			while (true)
			{
				int cycles = levelSrc == levels.size() - 1 ? coarsestSweeps : sweeps;
				converged = GS(levelSrc, dt, cycles, absoluteTolerance);
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
				if (levelDest >= levels.size()) continue;
				if (levelSrc == 0) ++cycleCount;
				restrict(levelSrc, levelDest, dt);
				levelSrc++;
			}
#ifdef EXACT_FLUX_CONSERVATION
			conservativeUpdate(dt);
#else
			pw::assign(levels[0].rhs, levels[0].sol);
#endif

			int totalSweeps = 0;
			for (int count : sweepCounts) totalSweeps += count;
			SAMS::debug3 << "Multigrid cycles: " << cycleCount << ", sweeps: " << totalSweeps << " (per level, finest first:";
			for (int count : sweepCounts) SAMS::debug3 << " " << count;
			SAMS::debug3 << ")\n";
		}

		inline static constexpr auto nameType = T::name + SAMS::constexprName("_mg");
		constexpr static std::string_view name = nameType;
		constexpr static bool coreSimulation = true;
		constexpr static bool timeSimulation = true;
		using dataPack = typename T::dataPack;

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
				size_t cells = axisReg.getDomainElements(baseName, SAMS::staggerType::CENTRED);
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
		 * @brief Set the boundary conditions of the problem, and clamp the variables of the coarse levels to zero
		 * @param harness The harness, which holds the variable registry
		 */
		void setBoundaryConditions(SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::setBoundaryConditions, packs, harness);
			auto &varRegistry = harness.variableRegistry;
			using clamp = SAMS::simpleClamp<SAMS::T_dataType, 3>;
			const std::string memory = T::memoryName.c_str();
			const int levelCount = std::max(std::max(axisLevels[0], axisLevels[1]), axisLevels[2]);
			for (int n = 1; n <= levelCount; ++n)
			{
				for (const char *variable : {"rhs", "sol", "res"})
				{
					const std::string name = memory + "_" + variable + "_" + std::to_string(n);
					SAMS::variableDef &varDef = varRegistry.getVariable(name);
					varRegistry.addBoundaryCondition(name, std::make_shared<clamp>(varDef, 0.0));
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
		 * @brief Advance the field by one step
		 * @param ts The timestep data
		 * @param harness The harness, which holds the variable registry
		 */
		void endOfTimestep(const SAMS::timeState &ts, SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::beforeSolve, packs, harness);
			harness.variableRegistry.applyBoundaryConditions(T::fieldName());
			pw::assign(solution, field);
			harness.variableRegistry.applyBoundaryConditions("scratch");
			solve(ts.dt, 5, 10000, SOLVER_TOLERANCE, packs);
			callWithFixedAndInferred(problem, &T::afterSolve, packs, harness);
			SAMS::debug3 << pw::sum(field) << "\n";
		}
	};

	}
}

#endif
