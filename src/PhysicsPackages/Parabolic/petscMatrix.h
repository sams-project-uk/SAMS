#ifndef MULTIGRIDTEST_PETSCMATRIX_H
#define MULTIGRIDTEST_PETSCMATRIX_H
#include "runner.h"
#include "harness.h"
#include <petscksp.h>
#include <vector>

// Relative tolerance of the solvers of all the MultigridTest packages. Define SOLVER_TOLERANCE when compiling to set it for all of them
#ifndef SOLVER_TOLERANCE
#define SOLVER_TOLERANCE 1.0e-8
#endif

#if defined(USE_PETSC)

//#define PETSC_NO_PRECONDITIONER

namespace SAMS
{
	namespace pw = portableWrapper;

	namespace parabolic
	{

	template <typename T>
	struct petscMatrix
	{
		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

		struct simulationData
		{
			size_t nx, ny, nz;
			pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
		};

		inline static constexpr auto nameType = T::name + SAMS::constexprName("_petscMatrix");
		constexpr static std::string_view name = nameType;
		constexpr static bool coreSimulation = true;
		constexpr static bool timeSimulation = true;
		using dataPack = SAMS::dataPacks::multiPack<simulationData, typename T::dataPack>;
		using packRefs = SAMS::packRefsOf_t<typename T::dataPack>;

		static_assert(std::is_same_v<SAMS::T_dataType, PetscScalar>, "petscMatrix requires SAMS::T_dataType to be PetscScalar");

		pw::portableArrayManager &manager;

		petscMatrix(SAMS::harness &hr) : manager(hr.memoryRegistry.getArrayManager()) {}

		KSP ksp = nullptr;
		Mat A = nullptr;
		Vec x = nullptr, b = nullptr;
#ifdef EXACT_FLUX_CONSERVATION
		Vec residual = nullptr;
#endif

		// The preconditioner is rebuilt every rebuildInterval steps, starting with the first, and reused in between
		static constexpr int rebuildInterval = 20;
		int stepsSinceRebuild = rebuildInterval;

		// Values of the matrix in COO form, stencilSize per row in the order given by the offsets of the problem
		lineArray cooValues;

		MPI_Comm comm = PETSC_COMM_SELF;

		// Whether each side of this rank is on the boundary of the domain
		struct boundaryFlags
		{
			bool lowX, highX, lowY, highY, lowZ, highZ;
		};
		boundaryFlags edges;

		// Whether a cell, with indices on this rank counted from 1, is beyond a boundary of the domain
		DEVICEPREFIX static bool beyondDomain(const boundaryFlags &e, SIGNED_INDEX_TYPE ii, SIGNED_INDEX_TYPE jj, SIGNED_INDEX_TYPE kk,
			SIGNED_INDEX_TYPE nx, SIGNED_INDEX_TYPE ny, SIGNED_INDEX_TYPE nz)
		{
			return (ii < 1 && e.lowX) || (ii > nx && e.highX) || (jj < 1 && e.lowY) || (jj > ny && e.highY) || (kk < 1 && e.lowZ) || (kk > nz && e.highZ);
		}

		// The finest level, with the members that the problem needs
		struct level
		{
			size_t nx, ny, nz;
			// The index of the level, which is 0 for the finest level
			SAMS::T_sizeType gridLevel = 0;
			bool coarsenX = false, coarsenY = false, coarsenZ = false;
			SIGNED_INDEX_TYPE firstX = 1, firstY = 1, firstZ = 1;
			pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
			pw::Range xbLocalDomainRange, ybLocalDomainRange, zbLocalDomainRange;
			lineArray dxc, dyc, dzc;
			// The old field
			volumeArray rhs;
			typename T::levelData data;
		};

		T problem;
		std::vector<level> levels;

		volumeArray field;

		static constexpr int stencilSize = T::stencilSize;

		void initialize(SAMS::harness &harness)
		{
			harness.initialisePETSc();
		}

		void setBoundaryConditions(SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::setBoundaryConditions, packs, harness);
		}

		void initialConditions(SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::initialConditions, packs, harness);
		}

		void registerVariables(SAMS::harness &harness, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::registerVariables, packs, harness);
			callWithFixedAndInferred(problem, &T::registerSecondaryVariables, packs, harness, std::array<int, 3>{0, 0, 0});
		}

		void getVariables(SAMS::harness &harness, simulationData &data, packRefs packs)
		{
			auto &varRegistry = harness.variableRegistry;
			auto &axisReg = harness.axisRegistry;
			const SAMS::variableDef &fieldDef = varRegistry.getVariable(T::fieldName());
			const std::string xName = fieldDef.getDimension(0).axisName;
			const std::string yName = fieldDef.getDimension(1).axisName;
			const std::string zName = fieldDef.getDimension(2).axisName;

			data.nx = fieldDef.getDimension(0).getLocalDomainCells();
			data.ny = fieldDef.getDimension(1).getLocalDomainCells();
			data.nz = fieldDef.getDimension(2).getLocalDomainCells();

			data.xcLocalDomainRange = fieldDef.getDimension(0).getLocalDomainRange(SAMS::staggerType::CENTRED);
			data.ycLocalDomainRange = fieldDef.getDimension(1).getLocalDomainRange(SAMS::staggerType::CENTRED);
			data.zcLocalDomainRange = fieldDef.getDimension(2).getLocalDomainRange(SAMS::staggerType::CENTRED);

#ifdef USE_MPI
			comm = harness.MPIManager.getComm();
#endif

			auto atEdge = [&](int a, SAMS::domain::edges side)
			{
				const int axis = static_cast<int>(fieldDef.getDimension(a).mpiAxis);
				return axis < 0 || harness.MPIManager.isEdge(axis, side);
			};
			edges.lowX = atEdge(0, SAMS::domain::edges::lower);
			edges.highX = atEdge(0, SAMS::domain::edges::upper);
			edges.lowY = atEdge(1, SAMS::domain::edges::lower);
			edges.highY = atEdge(1, SAMS::domain::edges::upper);
			edges.lowZ = atEdge(2, SAMS::domain::edges::lower);
			edges.highZ = atEdge(2, SAMS::domain::edges::upper);

			// The global index of the first cell of this rank in each direction, and the first row that it owns
			const std::array<std::string, 3> axisNames = {xName, yName, zName};
			const auto axisRanks = harness.MPIManager.getCoords();
			std::array<SIGNED_INDEX_TYPE, 3> first;
			for (int a = 0; a < 3; ++a)
			{
				const int axis = static_cast<int>(fieldDef.getDimension(a).mpiAxis);
				first[a] = harness.axisRegistry.getAxis(axisNames[a]).decomposition.firstIndexOnRank(axis >= 0 ? axisRanks[axis] : 0);
			}
			int commRank = 0;
#ifdef USE_MPI
			PetscCallMPIAbort(comm, MPI_Comm_rank(comm, &commRank));
#endif
			const SIGNED_INDEX_TYPE rowStart = harness.axisRegistry.getBlockOffset(axisNames, commRank);

			const PetscInt nx = data.nx, ny = data.ny, nz = data.nz;
			const PetscInt N = nx * ny * nz;
			const PetscCount ncoo = stencilSize * static_cast<PetscCount>(N);

			// The vectors and matrix must be in the same memory space as the kernels that fill them, so use device
			// types for a device backend. These can still be overridden from the PETSc options database
#if defined(USE_KOKKOS)
			const VecType vecType = VECKOKKOS;
			const MatType matType = MATAIJKOKKOS;
#elif defined(USE_CUDA)
			const VecType vecType = VECCUDA;
			const MatType matType = MATAIJCUSPARSE;
#elif defined(USE_HIP)
			const VecType vecType = VECHIP;
			const MatType matType = MATAIJHIPSPARSE;
#else
			const VecType vecType = VECSTANDARD;
			const MatType matType = MATAIJ;
#endif
			PetscCallAbort(comm, VecCreate(comm, &x));
			PetscCallAbort(comm, VecSetSizes(x, N, PETSC_DETERMINE));
			PetscCallAbort(comm, VecSetType(x, vecType));
			PetscCallAbort(comm, VecSetFromOptions(x));
			PetscCallAbort(comm, VecDuplicate(x, &b));
#ifdef EXACT_FLUX_CONSERVATION
			PetscCallAbort(comm, VecDuplicate(x, &residual));
#endif

			// The matrix is set in COO form with global rows and columns. Neighbours beyond the boundary of the domain have
			// negative indices, which PETSc ignores
			std::vector<PetscInt> cooI(ncoo), cooJ(ncoo);
			for (PetscInt k = 0; k < nz; ++k)
			{
				for (PetscInt j = 0; j < ny; ++j)
				{
					for (PetscInt i = 0; i < nx; ++i)
					{
						const PetscInt local = i + nx * (j + ny * k);
						const SIGNED_INDEX_TYPE row = rowStart + local;
						for (int s = 0; s < stencilSize; ++s)
						{
							int di = 0, dj = 0, dk = 0;
							T::stencilCellOffset(s, di, dj, dk);
							const PetscInt ii = i + di, jj = j + dj, kk = k + dk;
							SIGNED_INDEX_TYPE column;
							if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
							{
								column = rowStart + ii + nx * (jj + ny * kk);
							}
							else
							{
								column = harness.axisRegistry.getBlockedIndex(axisNames, std::array<SIGNED_INDEX_TYPE, 3>{first[0] + ii, first[1] + jj, first[2] + kk});
							}
							cooI[stencilSize * local + s] = column >= 0 ? row : -1;
							cooJ[stencilSize * local + s] = column;
						}
					}
				}
			}
			PetscCallAbort(comm, MatCreate(comm, &A));
			PetscCallAbort(comm, MatSetSizes(A, N, N, PETSC_DETERMINE, PETSC_DETERMINE));
			PetscCallAbort(comm, MatSetType(A, matType));
			PetscCallAbort(comm, MatSetFromOptions(A));
			PetscCallAbort(comm, MatSetPreallocationCOO(A, ncoo, cooI.data(), cooJ.data()));
			manager.allocate(cooValues, pw::Range(0, ncoo - 1));

			callWithFixedAndInferred(problem, &T::getVariables, packs, harness);
			levels.resize(1);
			level &fine = levels[0];
			fine.nx = data.nx;
			fine.ny = data.ny;
			fine.nz = data.nz;
			fine.xcLocalDomainRange = data.xcLocalDomainRange;
			fine.ycLocalDomainRange = data.ycLocalDomainRange;
			fine.zcLocalDomainRange = data.zcLocalDomainRange;
			fine.xbLocalDomainRange = fieldDef.getDimension(0).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
			fine.ybLocalDomainRange = fieldDef.getDimension(1).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
			fine.zbLocalDomainRange = fieldDef.getDimension(2).getLocalDomainRange(SAMS::staggerType::HALF_CELL);
			axisReg.fillPPLocalDelta(xName, fine.dxc, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalDelta(yName, fine.dyc, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalDelta(zName, fine.dzc, SAMS::staggerType::CENTRED);
			varRegistry.fillPPArray(T::fieldName(), field);
			fine.rhs = field;
			callWithFixedAndInferred(problem, &T::template getSecondaryVariables<level>, packs, harness, levels);

			// The matrix is not symmetric for a general magnetic field, so GMRES, with hypre's BoomerAMG unless
			// PETSC_NO_PRECONDITIONER is defined. These can be overridden from the PETSc options database
			PC pc;
			PetscCallAbort(comm, KSPCreate(comm, &ksp));
			PetscCallAbort(comm, KSPSetType(ksp, KSPGMRES));
			PetscCallAbort(comm, KSPGetPC(ksp, &pc));
#ifdef PETSC_NO_PRECONDITIONER
			PetscCallAbort(comm, PCSetType(pc, PCNONE));
#else
			PetscCallAbort(comm, PCSetType(pc, PCHYPRE));
			PetscCallAbort(comm, PCHYPRESetType(pc, "boomeramg"));

			// Options for a cheaper BoomerAMG setup in 3D: a larger strong threshold, HMIS coarsening with one level of
			// aggressive coarsening, and truncated extended interpolation
			PetscCallAbort(comm, PetscOptionsSetValue(nullptr, "-pc_hypre_boomeramg_strong_threshold", "0.5"));
			PetscCallAbort(comm, PetscOptionsSetValue(nullptr, "-pc_hypre_boomeramg_coarsen_type", "HMIS"));
			PetscCallAbort(comm, PetscOptionsSetValue(nullptr, "-pc_hypre_boomeramg_agg_nl", "1"));
			PetscCallAbort(comm, PetscOptionsSetValue(nullptr, "-pc_hypre_boomeramg_interp_type", "ext+i"));
			PetscCallAbort(comm, PetscOptionsSetValue(nullptr, "-pc_hypre_boomeramg_P_max", "4"));
#endif

			// The preconditioner is built on the first solve and reused for later ones, even though the matrix changes
			PetscCallAbort(comm, KSPSetReusePreconditioner(ksp, PETSC_TRUE));
			PetscCallAbort(comm, KSPSetTolerances(ksp, SOLVER_TOLERANCE, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
			// The tolerance is relative to the initial residual, as in the multigrid solver, and not to the norm of the right hand side
			PetscCallAbort(comm, KSPConvergedDefaultSetUIRNorm(ksp));
			PetscCallAbort(comm, KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
			PetscCallAbort(comm, KSPSetFromOptions(ksp));
		}

		void endOfTimestep(const SAMS::timeState &ts, SAMS::harness &harness, simulationData &data, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::beforeSolve, packs, harness);
			harness.variableRegistry.applyBoundaryConditions(T::fieldName());
			const volumeArray field = this->field;
			const boundaryFlags e = edges;
			const double dt = ts.dt;
			const size_t nx = levels[0].nx, ny = levels[0].ny, nz = levels[0].nz;
			const lineArray dxc = levels[0].dxc, dyc = levels[0].dyc, dzc = levels[0].dzc;
			lineArray vals = cooValues;

			levels[0].rhs = field;
			callWithFixedAndInferred(problem, &T::template prepareSolve<level>, packs, levels);
			const typename T::stencil s = problem.makeStencil(levels[0]);

			PetscScalar *bp, *xp;
			PetscMemType memType;
			PetscCallAbort(comm, VecGetArrayAndMemType(b, &bp, &memType));
			PetscCallAbort(comm, VecGetArrayAndMemType(x, &xp, &memType));
			// Neighbours beyond the boundary of the domain go on the right hand side
			auto assemble = LAMBDA(size_t i, size_t j, size_t k)
			{
				const double vol = dxc(i) * dyc(j) * dzc(k);

				double a[stencilSize];
				s.coefficients(i, j, k, a);

				const size_t r = (i - 1) + nx * ((j - 1) + ny * (k - 1));
				const double uOld = field(i, j, k);
				const double factor = s.beta(i, j, k);
				double rhs = vol * factor * uOld / dt;

				const double timeTerm = vol * factor / dt;
				for (int s = 0; s < stencilSize; ++s)
				{
					int di, dj, dk;
					T::stencilCellOffset(s, di, dj, dk);
					const SIGNED_INDEX_TYPE ii = static_cast<SIGNED_INDEX_TYPE>(i) + di;
					const SIGNED_INDEX_TYPE jj = static_cast<SIGNED_INDEX_TYPE>(j) + dj;
					const SIGNED_INDEX_TYPE kk = static_cast<SIGNED_INDEX_TYPE>(k) + dk;
					if (!petscMatrix::beyondDomain(e, ii, jj, kk, static_cast<SIGNED_INDEX_TYPE>(nx), static_cast<SIGNED_INDEX_TYPE>(ny), static_cast<SIGNED_INDEX_TYPE>(nz)))
					{
						vals(stencilSize * r + s) = (di == 0 && dj == 0 && dk == 0 ? timeTerm : 0.0) - a[s];
					}
					else
					{
						vals(stencilSize * r + s) = 0.0;
						rhs += a[s] * field(ii, jj, kk);
					}
				}

				bp[r] = rhs;
				xp[r] = uOld;
			};
			pw::applyKernel(assemble, pw::Range(1, nx), pw::Range(1, ny), pw::Range(1, nz));
			pw::fence();
			PetscCallAbort(comm, VecRestoreArrayAndMemType(b, &bp));
			PetscCallAbort(comm, VecRestoreArrayAndMemType(x, &xp));
			// The matrix depends on dt and on the conductivity from the old field, so it is set and handed to the solver
			// every step
			PetscCallAbort(comm, MatSetValuesCOO(A, cooValues.data(), INSERT_VALUES));
			PetscCallAbort(comm, KSPSetOperators(ksp, A, A));
			const bool rebuildPreconditioner = stepsSinceRebuild >= rebuildInterval;
			stepsSinceRebuild = rebuildPreconditioner ? 1 : stepsSinceRebuild + 1;
			PetscCallAbort(comm, KSPSetReusePreconditioner(ksp, rebuildPreconditioner ? PETSC_FALSE : PETSC_TRUE));
			PetscCallAbort(comm, KSPSolve(ksp, b, x));

#ifdef EXACT_FLUX_CONSERVATION
			// The residual b - A x is the old field minus the solved field, scaled by V beta / dt, plus the net flux of the solved
			// field. Adding it to x, scaled, sets x to the old field plus the net flux of the solved field, which is conservative
			PetscCallAbort(comm, MatMult(A, x, residual));
			PetscCallAbort(comm, VecAYPX(residual, -1.0, b));
			PetscScalar *rp;
			PetscCallAbort(comm, VecGetArrayAndMemType(residual, &rp, &memType));
			PetscCallAbort(comm, VecGetArrayAndMemType(x, &xp, &memType));
			auto conserve = LAMBDA(size_t i, size_t j, size_t k)
			{
				const size_t r = (i - 1) + nx * ((j - 1) + ny * (k - 1));
				xp[r] += dt * rp[r] / (dxc(i) * dyc(j) * dzc(k) * s.beta(i, j, k));
			};
			pw::applyKernel(conserve, pw::Range(1, nx), pw::Range(1, ny), pw::Range(1, nz));
			pw::fence();
			PetscCallAbort(comm, VecRestoreArrayAndMemType(x, &xp));
			PetscCallAbort(comm, VecRestoreArrayAndMemType(residual, &rp));
#endif

			PetscCallAbort(comm, VecGetArrayAndMemType(x, &xp, &memType));
			auto copyBack = LAMBDA(size_t i, size_t j, size_t k)
			{
				field(i, j, k) = xp[(i - 1) + nx * ((j - 1) + ny * (k - 1))];
			};
			pw::applyKernel(copyBack, pw::Range(1, nx), pw::Range(1, ny), pw::Range(1, nz));
			pw::fence();
			PetscCallAbort(comm, VecRestoreArrayAndMemType(x, &xp));

			callWithFixedAndInferred(problem, &T::afterSolve, packs, harness);
			const SAMS::T_dataType localSum = pw::sum(field);
			SAMS::T_dataType total = localSum;
#ifdef USE_MPI
			PetscCallMPIAbort(comm, MPIU_Allreduce(&localSum, &total, 1, MPIU_SCALAR, MPIU_SUM, comm));
#endif
			SAMS::debug3 << total << "\n";
		}

		void finalize()
		{
			VecDestroy(&x);
			VecDestroy(&b);
#ifdef EXACT_FLUX_CONSERVATION
			VecDestroy(&residual);
#endif
			MatDestroy(&A);
			KSPDestroy(&ksp);
		}
	};

	}
}
#endif // defined(USE_PETSC)
#endif // MULTIGRIDTEST_PETSCMATRIX_H
