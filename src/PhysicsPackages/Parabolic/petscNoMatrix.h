#ifndef MULTIGRIDTEST_PETSCNOMATRIX_H
#define MULTIGRIDTEST_PETSCNOMATRIX_H
#include "runner.h"
#include "harness.h"
#include <petscksp.h>
#include <vector>

// Relative tolerance of the solvers of all the MultigridTest packages. Define SOLVER_TOLERANCE when compiling to set it for all of them
#ifndef SOLVER_TOLERANCE
#define SOLVER_TOLERANCE 1.0e-8
#endif

#if defined(USE_PETSC)

namespace SAMS
{
	namespace pw = portableWrapper;

	namespace parabolic
	{

	template <typename T>
	struct petscNoMatrix
	{
		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

		struct simulationData
		{
			size_t nx, ny, nz;
			pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
		};

		inline static constexpr auto nameType = T::name + SAMS::constexprName("_petscNoMatrix");
		constexpr static std::string_view name = nameType;
		constexpr static bool coreSimulation = true;
		constexpr static bool timeSimulation = true;
		using dataPack = SAMS::dataPacks::multiPack<simulationData, typename T::dataPack>;
		using packRefs = SAMS::packRefsOf_t<typename T::dataPack>;

		static_assert(std::is_same_v<SAMS::T_dataType, PetscScalar>, "petscNoMatrix requires SAMS::T_dataType to be PetscScalar");

		pw::portableArrayManager &manager;

		petscNoMatrix(SAMS::harness &hr) : manager(hr.memoryRegistry.getArrayManager()), op(hr) {}

		KSP ksp = nullptr;
		Mat A = nullptr;
		Vec x = nullptr, b = nullptr;
#ifdef EXACT_FLUX_CONSERVATION
		Vec residual = nullptr;
#endif

		// The preconditioner is rebuilt every rebuildInterval steps, starting with the first, and reused in between
		static constexpr int rebuildInterval = 20;
		int stepsSinceRebuild = rebuildInterval;

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

		// Whether each side of this rank is on the boundary of the domain
		struct boundaryFlags
		{
			bool lowX, highX, lowY, highY, lowZ, highZ;
		};

		// Whether a cell, with indices on this rank counted from 1, is beyond a boundary of the domain
		DEVICEPREFIX static bool beyondDomain(const boundaryFlags &e, SIGNED_INDEX_TYPE ii, SIGNED_INDEX_TYPE jj, SIGNED_INDEX_TYPE kk,
			SIGNED_INDEX_TYPE nx, SIGNED_INDEX_TYPE ny, SIGNED_INDEX_TYPE nz)
		{
			return (ii < 1 && e.lowX) || (ii > nx && e.highX) || (jj < 1 && e.lowY) || (jj > ny && e.highY) || (kk < 1 && e.lowZ) || (kk > nz && e.highZ);
		}

		// What the shell matrix needs to apply the operator. u and dt are set before each solve
		struct operatorData
		{
			size_t nx, ny, nz;
			double dt;
			lineArray dxc, dyc, dzc;
			typename T::stencil s;
			volumeArray work;
			boundaryFlags edges;
			SAMS::harness &harness;
			operatorData(SAMS::harness &hr) : harness(hr) {}
		};
		operatorData op;

		MPI_Comm comm = PETSC_COMM_SELF;

		// The variable that the input of the matrix vector product is copied into
		static constexpr const char *workName = "petscNoMatrix_work";

		T problem;
		std::vector<level> levels;

		volumeArray field;

		static constexpr int stencilSize = T::stencilSize;

		// The coefficients of cell i, j, k in the order given by the offsets of the problem
		DEVICEPREFIX static void coefficients(const operatorData &d, size_t i, size_t j, size_t k, double *a)
		{
			d.s.coefficients(i, j, k, a);
		}

		// Neighbours beyond the boundary of the domain are not in the vector, as their contribution
		// is on the right hand side
		static PetscErrorCode mult(Mat M, Vec in, Vec out)
		{
			operatorData *data;
			const PetscScalar *inp;
			PetscScalar *outp;
			PetscMemType memType;
			PetscFunctionBeginUser;
			PetscCall(MatShellGetContext(M, &data));
			PetscCall(VecGetArrayReadAndMemType(in, &inp, &memType));
			PetscCall(VecGetArrayAndMemType(out, &outp, &memType));
			const operatorData &d = *data;
			const SIGNED_INDEX_TYPE nx = d.nx, ny = d.ny, nz = d.nz;
			const volumeArray work = d.work;
			auto copyIn = LAMBDA(size_t i, size_t j, size_t k)
			{
				work(i, j, k) = inp[(i - 1) + nx * ((j - 1) + ny * (k - 1))];
			};
			pw::applyKernel(copyIn, pw::Range(1, d.nx), pw::Range(1, d.ny), pw::Range(1, d.nz));
			pw::fence();
			d.harness.variableRegistry.haloExchange(workName);
			auto apply = LAMBDA(size_t i, size_t j, size_t k)
			{
				double a[stencilSize];
				petscNoMatrix::coefficients(d, i, j, k, a);
				const SIGNED_INDEX_TYPE r = (i - 1) + nx * ((j - 1) + ny * (k - 1));
				double sum = d.dxc(i) * d.dyc(j) * d.dzc(k) * d.s.beta(i, j, k) / d.dt * work(i, j, k);
				for (int s = 0; s < stencilSize; ++s)
				{
					int di, dj, dk;
					T::stencilCellOffset(s, di, dj, dk);
					const SIGNED_INDEX_TYPE ii = static_cast<SIGNED_INDEX_TYPE>(i) + di;
					const SIGNED_INDEX_TYPE jj = static_cast<SIGNED_INDEX_TYPE>(j) + dj;
					const SIGNED_INDEX_TYPE kk = static_cast<SIGNED_INDEX_TYPE>(k) + dk;
					if (!petscNoMatrix::beyondDomain(d.edges, ii, jj, kk, nx, ny, nz))
					{
						sum -= a[s] * work(ii, jj, kk);
					}
				}
				outp[r] = sum;
			};
			pw::applyKernel(apply, pw::Range(1, d.nx), pw::Range(1, d.ny), pw::Range(1, d.nz));
			pw::fence();
			PetscCall(VecRestoreArrayReadAndMemType(in, &inp));
			PetscCall(VecRestoreArrayAndMemType(out, &outp));
			PetscFunctionReturn(PETSC_SUCCESS);
		}

		// The diagonal of the shell matrix, for the Jacobi preconditioner
		static PetscErrorCode getDiagonal(Mat M, Vec diag)
		{
			operatorData *data;
			PetscScalar *dp;
			PetscMemType memType;
			PetscFunctionBeginUser;
			PetscCall(MatShellGetContext(M, &data));
			PetscCall(VecGetArrayAndMemType(diag, &dp, &memType));
			const operatorData d = *data;
			auto diagonal = LAMBDA(size_t i, size_t j, size_t k)
			{
				double a[stencilSize];
				petscNoMatrix::coefficients(d, i, j, k, a);
				double centre = 0.0;
					for (int s = 0; s < stencilSize; ++s)
					{
						int di, dj, dk;
						T::stencilCellOffset(s, di, dj, dk);
						if (di == 0 && dj == 0 && dk == 0)
							centre = a[s];
					}
					dp[(i - 1) + d.nx * ((j - 1) + d.ny * (k - 1))] = d.dxc(i) * d.dyc(j) * d.dzc(k) * d.s.beta(i, j, k) / d.dt - centre;
			};
			pw::applyKernel(diagonal, pw::Range(1, d.nx), pw::Range(1, d.ny), pw::Range(1, d.nz));
			pw::fence();
			PetscCall(VecRestoreArrayAndMemType(diag, &dp));
			PetscFunctionReturn(PETSC_SUCCESS);
		}

		void initialize()
		{
			PetscErrorCode ierr = PetscInitializeNoArguments();
			if (ierr)
			{
				SAMS::cerr << "PetscInitializeNoArguments failed with error " << ierr << std::endl;
				std::exit(EXIT_FAILURE);
			}
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
			auto &varRegistry = harness.variableRegistry;
			const int ghosts = 1;
			varRegistry.registerVariable<SAMS::T_dataType>(workName, pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
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
			boundaryFlags edges;
			edges.lowX = atEdge(0, SAMS::domain::edges::lower);
			edges.highX = atEdge(0, SAMS::domain::edges::upper);
			edges.lowY = atEdge(1, SAMS::domain::edges::lower);
			edges.highY = atEdge(1, SAMS::domain::edges::upper);
			edges.lowZ = atEdge(2, SAMS::domain::edges::lower);
			edges.highZ = atEdge(2, SAMS::domain::edges::upper);

			const PetscInt nx = data.nx, ny = data.ny, nz = data.nz;
			const PetscInt N = nx * ny * nz;

			// The vectors must be in the same memory space as the kernels that fill them, so use device
			// types for a device backend. These can still be overridden from the PETSc options database
#if defined(USE_KOKKOS)
			const VecType vecType = VECKOKKOS;
#elif defined(USE_CUDA)
			const VecType vecType = VECCUDA;
#elif defined(USE_HIP)
			const VecType vecType = VECHIP;
#else
			const VecType vecType = VECSTANDARD;
#endif
			PetscCallAbort(comm, VecCreate(comm, &x));
			PetscCallAbort(comm, VecSetSizes(x, N, PETSC_DETERMINE));
			PetscCallAbort(comm, VecSetType(x, vecType));
			PetscCallAbort(comm, VecSetFromOptions(x));
			PetscCallAbort(comm, VecDuplicate(x, &b));
#ifdef EXACT_FLUX_CONSERVATION
			PetscCallAbort(comm, VecDuplicate(x, &residual));
#endif

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

			// The matrix is never formed. It is a shell matrix, whose product with a vector is mult, and whose context is op
			op.nx = fine.nx;
			op.ny = fine.ny;
			op.nz = fine.nz;
			op.dxc = fine.dxc;
			op.dyc = fine.dyc;
			op.dzc = fine.dzc;
			op.edges = edges;
			varRegistry.fillPPArray(workName, op.work);
			PetscCallAbort(comm, MatCreateShell(comm, N, N, PETSC_DETERMINE, PETSC_DETERMINE, &op, &A));
			PetscCallAbort(comm, MatShellSetVecType(A, vecType));
			PetscCallAbort(comm, MatShellSetOperation(A, MATOP_MULT, (void (*)(void))petscNoMatrix::mult));
			PetscCallAbort(comm, MatShellSetOperation(A, MATOP_GET_DIAGONAL, (void (*)(void))petscNoMatrix::getDiagonal));

			// The operator is not symmetric for a general magnetic field, so GMRES. hypre needs an assembled matrix, so the
			// preconditioner is Jacobi, from the diagonal of the shell matrix. These can be overridden from the PETSc options database
			PC pc;
			PetscCallAbort(comm, KSPCreate(comm, &ksp));
			PetscCallAbort(comm, KSPSetType(ksp, KSPGMRES));
			PetscCallAbort(comm, KSPGetPC(ksp, &pc));
			PetscCallAbort(comm, PCSetType(pc, PCJACOBI));
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
			const boundaryFlags e = op.edges;
			const double dt = ts.dt;
			const size_t nx = levels[0].nx, ny = levels[0].ny, nz = levels[0].nz;
			const lineArray dxc = levels[0].dxc, dyc = levels[0].dyc, dzc = levels[0].dzc;

			levels[0].rhs = field;
			callWithFixedAndInferred(problem, &T::template prepareSolve<level>, packs, levels);

			op.s = problem.makeStencil(levels[0]);
			op.dt = dt;
			const operatorData d = op;

			PetscScalar *bp, *xp;
			PetscMemType memType;
			PetscCallAbort(comm, VecGetArrayAndMemType(b, &bp, &memType));
			PetscCallAbort(comm, VecGetArrayAndMemType(x, &xp, &memType));
			// Neighbours beyond the boundary of the domain are not in the operator, so their coefficients times the guard cell values are added here
			auto assemble = LAMBDA(size_t i, size_t j, size_t k)
			{
				const double vol = dxc(i) * dyc(j) * dzc(k);
				const size_t r = (i - 1) + nx * ((j - 1) + ny * (k - 1));
				const double uOld = field(i, j, k);
				double rhs = vol * d.s.beta(i, j, k) * uOld / dt;

				double a[stencilSize];
				petscNoMatrix::coefficients(d, i, j, k, a);
				for (int s = 0; s < stencilSize; ++s)
				{
					int di, dj, dk;
					T::stencilCellOffset(s, di, dj, dk);
					const SIGNED_INDEX_TYPE ii = static_cast<SIGNED_INDEX_TYPE>(i) + di;
					const SIGNED_INDEX_TYPE jj = static_cast<SIGNED_INDEX_TYPE>(j) + dj;
					const SIGNED_INDEX_TYPE kk = static_cast<SIGNED_INDEX_TYPE>(k) + dk;
					if (petscNoMatrix::beyondDomain(e, ii, jj, kk, static_cast<SIGNED_INDEX_TYPE>(nx), static_cast<SIGNED_INDEX_TYPE>(ny), static_cast<SIGNED_INDEX_TYPE>(nz)))
					{
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
			// PETSc can't see that the operator has changed, so its state is increased, which makes the Jacobi preconditioner
			// take the diagonal again when it is rebuilt
			PetscCallAbort(comm, MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
			PetscCallAbort(comm, MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
			PetscCallAbort(comm, KSPSetOperators(ksp, A, A));
			const bool rebuildPreconditioner = stepsSinceRebuild >= rebuildInterval;
			stepsSinceRebuild = rebuildPreconditioner ? 1 : stepsSinceRebuild + 1;
			PetscCallAbort(comm, KSPSetReusePreconditioner(ksp, rebuildPreconditioner ? PETSC_FALSE : PETSC_TRUE));
			PetscCallAbort(comm, KSPSolve(ksp, b, x));

			KSPConvergedReason reason;
			PetscInt iterations;
			PetscCallAbort(comm, KSPGetConvergedReason(ksp, &reason));
			PetscCallAbort(comm, KSPGetIterationNumber(ksp, &iterations));
			SAMS::debug3 << "PETSc iterations: " << iterations << "\n";
			if (reason < 0)
			{
				SAMS::cerr << "Warning: the PETSc solve did not converge at step " << ts.step << ", reason " << KSPConvergedReasons[reason] << " after " << iterations << " iterations" << std::endl;
			}

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
				xp[r] += dt * rp[r] / (dxc(i) * dyc(j) * dzc(k) * d.s.beta(i, j, k));
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
			PetscFinalize();
		}
	};

	}
}
#endif // defined(USE_PETSC)
#endif // MULTIGRIDTEST_PETSCNOMATRIX_H
