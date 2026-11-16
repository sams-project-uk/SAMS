#ifndef MULTIGRIDTEST_PETSCNOMATRIX_H
#define MULTIGRIDTEST_PETSCNOMATRIX_H
#include "runner.h"
#include "harness.h"
#include "builtInBoundaryConditions.h"
#include "deckHelpers/deckBoundaries.h"
#include "deckHelpers/expressionBC.h"
#include "helpers.h"
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
			SAMS::T_sizeType nx, ny, nz;
			pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
		};

		inline static constexpr auto nameType = T::name + SAMS::makeName("_petscNoMatrix");
		constexpr static std::string_view name = nameType;
		constexpr static bool coreSimulation = false;
		constexpr static bool timeSimulation = true;
		using dataPack = SAMS::dataPacks::multiPack<simulationData, SAMS::deckHostData<3>, typename T::dataPack>;
		using packRefs = SAMS::packRefsOf_t<typename T::dataPack>;

		static_assert(std::is_same_v<SAMS::T_dataType, PetscScalar>, "petscNoMatrix requires SAMS::T_dataType to be PetscScalar");

		// Activate the deck services, which hold the host data of the deck for expressions
		template <typename T_runner>
		void runnerInteraction(T_runner &runnerRef)
		{
			runnerRef.activatePackage("deckServices");
		}

		pw::portableArrayManager &manager;

		petscNoMatrix(SAMS::harness &hr) : manager(hr.memoryRegistry.getArrayManager()), op(hr) {}

		// One solver, with its own preconditioner, for each subproblem, because their operators can differ greatly
		std::vector<KSP> ksps;
		Mat A = nullptr;
		Vec x = nullptr, b = nullptr;
#ifdef EXACT_FLUX_CONSERVATION
		Vec residual = nullptr;
#endif

		// The preconditioner of each subproblem is rebuilt every rebuildInterval steps, starting with the first, and reused in between
		static constexpr int rebuildInterval = 20;
		std::vector<int> stepsSinceRebuild;

		// The finest level, with the members that the problem needs
		struct level
		{
			SAMS::T_sizeType nx, ny, nz;
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

		// The boundary conditions on the sides of the domain for each subproblem
		std::vector<parabolic::sideConditions> sides;

		// The boundary conditions of the temperature of each subproblem, active only while that subproblem is solved
		std::vector<std::vector<std::shared_ptr<SAMS::boundaryConditions>>> problemBoundaries;

		// What the shell matrix needs to apply the operator. u and sides are set before each solve
		struct operatorData
		{
			SAMS::T_sizeType nx, ny, nz;
			lineArray dxc, dyc, dzc;
			typename T::stencil s;
			volumeArray work;
			boundaryFlags edges;
			parabolic::sideConditions sides;
			SAMS::harness &harness;
			operatorData(SAMS::harness &hr) : harness(hr) {}
		};
		operatorData op;

		MPI_Comm comm = PETSC_COMM_SELF;

		// The variable that the input of the matrix vector product is copied into. Problems that share memory share it
		inline static constexpr auto workName = T::memoryName + SAMS::makeName("_petscNoMatrix_work");

		T problem;
		std::vector<level> levels;

		volumeArray field;

		static constexpr int stencilSize = T::stencilSize;

		// The coefficients of cell i, j, k in the order given by the offsets of the problem
		DEVICEPREFIX static void coefficients(const operatorData &d, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k, SAMS::T_dataType *a)
		{
			d.s.coefficients(i, j, k, a);
		}

		// A neighbour beyond the boundary of the domain is mapped onto domain cells, which are in the vector, and onto ghost
		// cells and a constant, which are on the right hand side
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
			const SIGNED_INDEX_TYPE cells[3] = {nx, ny, nz};
			const volumeArray work = d.work;
			auto copyIn = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				work(i, j, k) = inp[(i - 1) + nx * ((j - 1) + ny * (k - 1))];
			};
			pw::applyKernel(copyIn, pw::Range(1, d.nx), pw::Range(1, d.ny), pw::Range(1, d.nz));
			pw::fence();
			d.harness.variableRegistry.haloExchange(workName.c_str());
			auto apply = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				SAMS::T_dataType a[stencilSize];
				petscNoMatrix::coefficients(d, i, j, k, a);
				const SIGNED_INDEX_TYPE r = (i - 1) + nx * ((j - 1) + ny * (k - 1));
				SAMS::T_dataType sum = d.dxc(i) * d.dyc(j) * d.dzc(k) * d.s.beta(i, j, k) * work(i, j, k);
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
			// The mapping is in its own kernel because having it in the kernel above makes that much slower, even where it is not used
			auto applyBoundary = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SIGNED_INDEX_TYPE si = i, sj = j, sk = k;
				SAMS::T_dataType a[stencilSize];
				petscNoMatrix::coefficients(d, i, j, k, a);
				SAMS::T_dataType sum = 0.0;
				for (int s = 0; s < stencilSize; ++s)
				{
					int di, dj, dk;
					T::stencilCellOffset(s, di, dj, dk);
					const SIGNED_INDEX_TYPE ii = si + di, jj = sj + dj, kk = sk + dk;
					if (!petscNoMatrix::beyondDomain(d.edges, ii, jj, kk, nx, ny, nz))
						continue;
					const parabolic::ghostMap g = parabolic::mapGhost(d.sides, ii, jj, kk, cells);
					for (int m = 0; m < g.count; ++m)
					{
						if (!petscNoMatrix::beyondDomain(d.edges, g.index[m][0], g.index[m][1], g.index[m][2], nx, ny, nz))
							sum -= a[s] * g.weight[m] * work(g.index[m][0], g.index[m][1], g.index[m][2]);
					}
				}
				outp[(si - 1) + nx * ((sj - 1) + ny * (sk - 1))] += sum;
			};
			bool mapped = false;
			for (int dim = 0; dim < 3; ++dim)
			{
				for (int side = 0; side < 2; ++side)
				{
					const parabolic::boundaryType type = d.sides.type[dim][side];
					if (d.sides.physical[dim][side] && (type == parabolic::boundaryType::zeroGradient || type == parabolic::boundaryType::fixedMirror || type == parabolic::boundaryType::floatingMirror))
						mapped = true;
				}
			}
			if (mapped)
			{
				// The cells next to the sides of this rank, each once
				// The x sides, whole
				pw::applyKernel(applyBoundary, pw::Range(1, 1), pw::Range(1, d.ny), pw::Range(1, d.nz));
				pw::applyKernel(applyBoundary, pw::Range(d.nx, d.nx), pw::Range(1, d.ny), pw::Range(1, d.nz));

				// The y sides, without the cells of the x sides
				pw::applyKernel(applyBoundary, pw::Range(2, d.nx - 1), pw::Range(1, 1), pw::Range(1, d.nz));
				pw::applyKernel(applyBoundary, pw::Range(2, d.nx - 1), pw::Range(d.ny, d.ny), pw::Range(1, d.nz));

				// The z sides, without the cells of the x and y sides
				pw::applyKernel(applyBoundary, pw::Range(2, d.nx - 1), pw::Range(2, d.ny - 1), pw::Range(1, 1));
				pw::applyKernel(applyBoundary, pw::Range(2, d.nx - 1), pw::Range(2, d.ny - 1), pw::Range(d.nz, d.nz));
				pw::fence();
			}
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
			auto diagonal = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				SAMS::T_dataType a[stencilSize];
				petscNoMatrix::coefficients(d, i, j, k, a);
				// A neighbour beyond the boundary can depend on the cell itself, which makes it part of the diagonal
				const SIGNED_INDEX_TYPE cells[3] = {static_cast<SIGNED_INDEX_TYPE>(d.nx), static_cast<SIGNED_INDEX_TYPE>(d.ny), static_cast<SIGNED_INDEX_TYPE>(d.nz)};
				SAMS::T_dataType centre = 0.0;
					for (int s = 0; s < stencilSize; ++s)
					{
						int di, dj, dk;
						T::stencilCellOffset(s, di, dj, dk);
						const parabolic::ghostMap g = parabolic::mapGhost(d.sides, static_cast<SIGNED_INDEX_TYPE>(i) + di, static_cast<SIGNED_INDEX_TYPE>(j) + dj, static_cast<SIGNED_INDEX_TYPE>(k) + dk, cells);
						for (int m = 0; m < g.count; ++m)
						{
							if (g.index[m][0] == static_cast<SIGNED_INDEX_TYPE>(i) && g.index[m][1] == static_cast<SIGNED_INDEX_TYPE>(j) && g.index[m][2] == static_cast<SIGNED_INDEX_TYPE>(k))
								centre += a[s] * g.weight[m];
						}
					}
					dp[(i - 1) + d.nx * ((j - 1) + d.ny * (k - 1))] = d.dxc(i) * d.dyc(j) * d.dzc(k) * d.s.beta(i, j, k) - centre;
			};
			pw::applyKernel(diagonal, pw::Range(1, d.nx), pw::Range(1, d.ny), pw::Range(1, d.nz));
			pw::fence();
			PetscCall(VecRestoreArrayAndMemType(diag, &dp));
			PetscFunctionReturn(PETSC_SUCCESS);
		}

		void initialize(SAMS::harness &harness)
		{
			harness.initialisePETSc();
		}

		void defaultValues(SAMS::KVS &kvs, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::defaultValues, packs, kvs);
		}

		void setBoundaryConditions(SAMS::harness &harness, SAMS::KVS &kvs, [[maybe_unused]] SAMS::deckHostData<3> &hostData, [[maybe_unused]] SAMS::timeState &tData, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::setBoundaryConditions, packs, harness);
			using clamp = SAMS::simpleClamp<SAMS::T_dataType, 3>;
			SAMS::variableDef &fieldDef = harness.variableRegistry.getVariable(T::fieldName());
			sides.assign(problem.problemCount(), parabolic::sideConditions{});
			problemBoundaries.assign(problem.problemCount(), {});
			for (SAMS::T_sizeType n = 0; n < problem.problemCount(); ++n)
			{
				for (int d = 0; d < 3; ++d)
				{
					for (SAMS::domain::edges edge : {SAMS::domain::edges::lower, SAMS::domain::edges::upper})
					{
						const int side = edge == SAMS::domain::edges::lower ? 0 : 1;
						const parabolic::tcBoundaryType boundary = problem.getBoundaryType(kvs, n, d, edge);
						// Clamp and expression set the ghost cells of the temperature, which are then fixed
						const bool setsGhosts = boundary.type == parabolic::boundaryType::clamp || boundary.type == parabolic::boundaryType::expression;
						sides[n].type[d][side] = setsGhosts ? parabolic::boundaryType::fixedGhosts : boundary.type;
						sides[n].value[d][side] = boundary.value;
						if (boundary.type == parabolic::boundaryType::clamp)
							problemBoundaries[n].push_back(fieldDef.addBoundaryCondition(d, edge, clamp(fieldDef, boundary.value)));
						else if (boundary.type == parabolic::boundaryType::expression)
						{
#ifdef USE_DECK_PARSER
							eis::stack<SAMS::T_dataType> stack;
							eis::error errcode = eis::err_none;
							hostData.parser->tokenize(boundary.text, stack, errcode);
							if (errcode != eis::err_none) harness.abort("Invalid thermal conduction boundary expression: " + boundary.text + "\n" + SAMS::parserErrors(*hostData.parser), false);
							hostData.addPositions(harness, fieldDef);
							problemBoundaries[n].push_back(fieldDef.addBoundaryCondition(d, edge, SAMS::expressionBC<3>(fieldDef, harness, tData, hostData, stack)));
#else
							harness.abort("Invalid thermal conduction boundary expression: " + boundary.text + "\nExpressions need the deck parser", false);
#endif
						}
					}
				}
			}
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
			varRegistry.registerVariable<SAMS::T_dataType>(workName.c_str(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
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
			for (parabolic::sideConditions &s : sides)
			{
				s.physical[0][0] = edges.lowX;
				s.physical[0][1] = edges.highX;
				s.physical[1][0] = edges.lowY;
				s.physical[1][1] = edges.highY;
				s.physical[2][0] = edges.lowZ;
				s.physical[2][1] = edges.highZ;
			}

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
			varRegistry.fillPPArray(workName.c_str(), op.work);
			PetscCallAbort(comm, MatCreateShell(comm, N, N, PETSC_DETERMINE, PETSC_DETERMINE, &op, &A));
			PetscCallAbort(comm, MatShellSetVecType(A, vecType));
			PetscCallAbort(comm, MatShellSetOperation(A, MATOP_MULT, (void (*)(void))petscNoMatrix::mult));
			PetscCallAbort(comm, MatShellSetOperation(A, MATOP_GET_DIAGONAL, (void (*)(void))petscNoMatrix::getDiagonal));

			// The operator is not symmetric for a general magnetic field, so GMRES. hypre needs an assembled matrix, so the
			// preconditioner is Jacobi, from the diagonal of the shell matrix. These can be overridden from the PETSc options database
			ksps.assign(problem.problemCount(), nullptr);
			stepsSinceRebuild.assign(problem.problemCount(), rebuildInterval);
			for (KSP &ksp : ksps)
			{
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
		}

		void endOfTimestep(const SAMS::timeState &ts, SAMS::harness &harness, simulationData &data, packRefs packs)
		{
			callWithFixedAndInferred(problem, &T::beforeAllSolves, packs, harness);
			for (SAMS::T_sizeType n = 0; n < problem.problemCount(); ++n)
			{
				solveProblem(n, ts, harness, packs);
			}
			callWithFixedAndInferred(problem, &T::afterAllSolves, packs, harness);
		}

		// Solve subproblem n for one step, with its own solver and preconditioner
		void solveProblem(SAMS::T_sizeType n, const SAMS::timeState &ts, SAMS::harness &harness, packRefs packs)
		{
			for (auto &boundaries : problemBoundaries)
			{
				for (auto &bc : boundaries)
					harness.variableRegistry.deactivateBoundaryCondition(T::fieldName(), bc);
			}
			for (auto &bc : problemBoundaries[n])
				harness.variableRegistry.activateBoundaryCondition(T::fieldName(), bc);
			callWithFixedAndInferred(problem, &T::beforeSolve, packs, harness, n);
			harness.variableRegistry.applyBoundaryConditions(T::fieldName());
			const volumeArray field = this->field;
			const boundaryFlags e = op.edges;
			const SAMS::T_sizeType nx = levels[0].nx, ny = levels[0].ny, nz = levels[0].nz;
			const SIGNED_INDEX_TYPE cells[3] = {static_cast<SIGNED_INDEX_TYPE>(nx), static_cast<SIGNED_INDEX_TYPE>(ny), static_cast<SIGNED_INDEX_TYPE>(nz)};
			const lineArray dxc = levels[0].dxc, dyc = levels[0].dyc, dzc = levels[0].dzc;

			levels[0].rhs = field;
			callWithFixedAndInferred(problem, &T::template prepareSolve<level>, packs, levels, n, ts);

			op.s = problem.makeStencil(levels[0], n);
			op.sides = sides[n];
			const operatorData d = op;

			PetscScalar *bp, *xp;
			PetscMemType memType;
			PetscCallAbort(comm, VecGetArrayAndMemType(b, &bp, &memType));
			PetscCallAbort(comm, VecGetArrayAndMemType(x, &xp, &memType));
			// The ghost cells and constants that neighbours beyond the boundary of the domain are mapped onto are not in the operator,
			// so their coefficients times their values are added here
			auto assemble = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SAMS::T_dataType vol = dxc(i) * dyc(j) * dzc(k);
				const SAMS::T_indexType r = (i - 1) + nx * ((j - 1) + ny * (k - 1));
				const SAMS::T_dataType uOld = field(i, j, k);
				SAMS::T_dataType rhs = vol * d.s.beta(i, j, k) * uOld;

				SAMS::T_dataType a[stencilSize];
				petscNoMatrix::coefficients(d, i, j, k, a);
				for (int s = 0; s < stencilSize; ++s)
				{
					int di, dj, dk;
					T::stencilCellOffset(s, di, dj, dk);
					const SIGNED_INDEX_TYPE ii = static_cast<SIGNED_INDEX_TYPE>(i) + di;
					const SIGNED_INDEX_TYPE jj = static_cast<SIGNED_INDEX_TYPE>(j) + dj;
					const SIGNED_INDEX_TYPE kk = static_cast<SIGNED_INDEX_TYPE>(k) + dk;
					if (!petscNoMatrix::beyondDomain(e, ii, jj, kk, cells[0], cells[1], cells[2]))
						continue;
					const parabolic::ghostMap g = parabolic::mapGhost(d.sides, ii, jj, kk, cells);
					rhs += a[s] * g.constant;
					for (int m = 0; m < g.count; ++m)
					{
						if (petscNoMatrix::beyondDomain(e, g.index[m][0], g.index[m][1], g.index[m][2], cells[0], cells[1], cells[2]))
							rhs += a[s] * g.weight[m] * field(g.index[m][0], g.index[m][1], g.index[m][2]);
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
			PetscCallAbort(comm, KSPSetOperators(ksps[n], A, A));
			const bool rebuildPreconditioner = stepsSinceRebuild[n] >= rebuildInterval;
			stepsSinceRebuild[n] = rebuildPreconditioner ? 1 : stepsSinceRebuild[n] + 1;
			PetscCallAbort(comm, KSPSetReusePreconditioner(ksps[n], rebuildPreconditioner ? PETSC_FALSE : PETSC_TRUE));
			PetscCallAbort(comm, KSPSolve(ksps[n], b, x));

			KSPConvergedReason reason;
			PetscInt iterations;
			PetscCallAbort(comm, KSPGetConvergedReason(ksps[n], &reason));
			PetscCallAbort(comm, KSPGetIterationNumber(ksps[n], &iterations));
			SAMS::debug3 << "PETSc iterations: " << iterations << "\n";
			if (reason < 0)
			{
				SAMS::cerr << "Warning: the PETSc solve did not converge at step " << ts.step << ", reason " << KSPConvergedReasons[reason] << " after " << iterations << " iterations" << std::endl;
			}

#ifdef EXACT_FLUX_CONSERVATION
			// The residual b - A x is the old field minus the solved field, scaled by V beta, plus the net flux of the solved
			// field. Adding it to x, scaled, sets x to the old field plus the net flux of the solved field, which is conservative
			PetscCallAbort(comm, MatMult(A, x, residual));
			PetscCallAbort(comm, VecAYPX(residual, -1.0, b));
			PetscScalar *rp;
			PetscCallAbort(comm, VecGetArrayAndMemType(residual, &rp, &memType));
			PetscCallAbort(comm, VecGetArrayAndMemType(x, &xp, &memType));
			auto conserve = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SAMS::T_indexType r = (i - 1) + nx * ((j - 1) + ny * (k - 1));
				xp[r] += rp[r] / (dxc(i) * dyc(j) * dzc(k) * d.s.beta(i, j, k));
			};
			pw::applyKernel(conserve, pw::Range(1, nx), pw::Range(1, ny), pw::Range(1, nz));
			pw::fence();
			PetscCallAbort(comm, VecRestoreArrayAndMemType(x, &xp));
			PetscCallAbort(comm, VecRestoreArrayAndMemType(residual, &rp));
#endif

			PetscCallAbort(comm, VecGetArrayAndMemType(x, &xp, &memType));
			auto copyBack = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				field(i, j, k) = xp[(i - 1) + nx * ((j - 1) + ny * (k - 1))];
			};
			pw::applyKernel(copyBack, pw::Range(1, nx), pw::Range(1, ny), pw::Range(1, nz));
			pw::fence();
			PetscCallAbort(comm, VecRestoreArrayAndMemType(x, &xp));

			callWithFixedAndInferred(problem, &T::afterSolve, packs, harness, n);
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
			for (KSP &ksp : ksps) KSPDestroy(&ksp);
		}
	};

	}
}
#endif // defined(USE_PETSC)
#endif // MULTIGRIDTEST_PETSCNOMATRIX_H
