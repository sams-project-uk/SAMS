/*
 * PETSc Rosenbrock-W integration of cell-wise stiff source terms.
 * See petscStiffSolver.h. This file must not include any SAMS header (see the note there).
 *
 * Problem         : du/dt = S(u), supplied to PETSc in IMPLICIT form, G(t,u,u') = u' - S(u) = 0,
 *                   through TSSetIFunction / TSSetIJacobian.
 *                   This matters: PETSc's TSROSW is an IMEX scheme. Anything given through
 *                   TSSetRHSFunction is integrated EXPLICITLY, and only the IFunction part is
 *                   treated linearly implicitly. Supplying the stiff sources as an RHS function
 *                   therefore gives an explicit method with error control, whose step size is
 *                   pinned at the explicit stability limit (~1/rate).
 * Source          : supplied by the caller (exact, may couple neighbouring cells).
 * Jacobian        : dS/du, only the stride x stride diagonal block of each cell, built by finite
 *                   differences. All cells are perturbed together (stride evaluations), so
 *                   each block is the Jacobian for a locally uniform state, i.e. couplings
 *                   to neighbouring cells are lagged. TSROSW with a W-method (RA34PW2)
 *                   tolerates an approximate Jacobian, and the linear system is exactly
 *                   block diagonal. The IJacobian handed to PETSc is shift*I - dS/du.
 *                   The finite-difference blocks are cached per state, so a rejected step that
 *                   is retried from the same state with a new step size (new shift) does not
 *                   rebuild them.
 */

#include "petscStiffSolver.h"

#include <petsc.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// Compatibility for PETSc < 3.18, which lacks PetscCall and friends. Unused for PETSc >= 3.18.
#if PETSC_VERSION_LT(3,18,0)
#ifndef PETSC_SUCCESS
#define PETSC_SUCCESS 0
#endif
#define PetscCall(...)            do { PetscErrorCode ss_ierr_ = (__VA_ARGS__); CHKERRQ(ss_ierr_); } while (0)
#define PetscCallAbort(comm, ...) do { PetscErrorCode ss_ierr_ = (__VA_ARGS__); CHKERRABORT(comm, ss_ierr_); } while (0)
#endif

// The raw double arrays handed to the callback are PETSc's own storage
static_assert(std::is_same<PetscScalar, double>::value, "PETSc must be built with real double scalars");

// For functions that return void / bool: abort on any PETSc error
#define SS_ABORT(...) PetscCallAbort(PETSC_COMM_SELF, __VA_ARGS__)

namespace SAMS_PETSC {

namespace {
    bool g_petsc_owner = false; // this file called PetscInitialize and must call PetscFinalize
}

struct StiffSolver::Impl {
    PetscInt ncells = 0;
    PetscInt stride = 0;
    RhsFunction rhs = nullptr;
    void *user = nullptr;
    std::vector<unsigned char> check_cell;
    std::vector<int> positive;

    TS  ts     = nullptr;
    Mat jac    = nullptr;
    Vec u_sol  = nullptr;
    Vec u_work = nullptr;
    Vec f_base = nullptr;
    Vec f_pert = nullptr;
    Vec h_pert = nullptr;
    std::vector<PetscScalar> blocks; // stride*stride dense block per cell
    PetscReal dt_hint = 0.0;         // last suggested internal step, first guess of the next solve

    // Diagnostics, reset at the start of every solve
    bool domain_check = true;        // -tf_domain_check 0 disables the validity check
    long long domain_rejects = 0;
    long long domain_bad_index = -1;
    double domain_bad_value = 0.0;
    long long domain_bad_count = 0;
    double domain_vec_norm = 0.0;
    double domain_c0_min = 0.0;
    double domain_c0_max = 0.0;
    double domain_first_dt = 0.0;
    long long rhs_nonfinite = 0;
    long long rhs_bad_index = -1;

    // Cache key of the finite-difference blocks (the state vector they were built from)
    std::vector<PetscScalar> blocks_u; // copy of the state the blocks were built from
    bool blocks_valid = false;

    static PetscErrorCode Rhs(TS, PetscReal, Vec U, Vec F, void *ctx);
    static PetscErrorCode BuildBlocks(TS ts, PetscReal t, Vec U, Impl *s);
    static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx);
    static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat A, Mat P, void *ctx);
    static PetscErrorCode Domain(TS ts, PetscReal, Vec U, PetscBool *accept);
    static PetscErrorCode CheckStage(TSAdapt adapt, TS ts, PetscReal t, Vec Y, PetscBool *accept);

    // Set by IFunction when the source at a stage state is not finite; read and cleared by
    // CheckStage, which PETSc calls after every stage (see TSStep_RosW / TSAdaptCheckStage).
    bool stage_nonfinite = false;
    long long stage_rejects = 0;
};

// F(U) through the caller's callback
PetscErrorCode StiffSolver::Impl::Rhs(TS, PetscReal, Vec U, Vec F, void *ctx) {
    PetscFunctionBeginUser;
    auto *s = static_cast<Impl *>(ctx);
    const PetscScalar *u;
    PetscScalar *f;
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetArray(F, &f));
    s->rhs(u, f, s->user);
    {
        const size_t n = static_cast<size_t>(s->ncells) * s->stride;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(static_cast<double>(f[i]))) {
                if (s->rhs_nonfinite == 0) s->rhs_bad_index = static_cast<long long>(i);
                ++s->rhs_nonfinite;
                break; // one per evaluation is enough
            }
        }
    }
    PetscCall(VecRestoreArrayRead(U, &u));
    PetscCall(VecRestoreArray(F, &f));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Finite-difference blocks of dS/du at U, (1 + stride) evaluations of S. Cached per state.
PetscErrorCode StiffSolver::Impl::BuildBlocks(TS ts, PetscReal t, Vec U, Impl *s) {
    PetscFunctionBeginUser;
    const PetscInt bs = s->stride;
    const PetscInt nc = s->ncells;
    const size_t n = static_cast<size_t>(nc) * bs;
    {
        const PetscScalar *uc;
        PetscCall(VecGetArrayRead(U, &uc));
        const bool same = s->blocks_valid && std::equal(uc, uc + n, s->blocks_u.begin());
        if (!same) s->blocks_u.assign(uc, uc + n);
        PetscCall(VecRestoreArrayRead(U, &uc));
        if (same) PetscFunctionReturn(PETSC_SUCCESS);
    }
    const long long nonfinite_before = s->rhs_nonfinite;

    PetscCall(Rhs(ts, t, U, s->f_base, s));

    // Perturbation size: sqrt(eps) * max(|u|, 1e-3 * max over cells |u_k|), per entry
    std::vector<PetscReal> vmax(bs);
    PetscCall(VecStrideNormAll(U, NORM_INFINITY, vmax.data()));
    const PetscReal sq = std::sqrt(static_cast<PetscReal>(DBL_EPSILON));

    const PetscScalar *u;
    PetscScalar *h;
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetArray(s->h_pert, &h));
    for (PetscInt cell = 0; cell < nc; ++cell) {
        for (PetscInt k = 0; k < bs; ++k) {
            PetscReal ref = std::max(static_cast<PetscReal>(PetscAbsScalar(u[cell * bs + k])), 1.0e-3 * vmax[k]);
            if (ref == 0.0) ref = 1.0;
            h[cell * bs + k] = sq * ref;
        }
    }
    PetscCall(VecRestoreArrayRead(U, &u));

    for (PetscInt k = 0; k < bs; ++k) {
        // Perturb variable k in every cell at once
        PetscCall(VecCopy(U, s->u_work));
        PetscScalar *w;
        PetscCall(VecGetArray(s->u_work, &w));
        for (PetscInt cell = 0; cell < nc; ++cell) w[cell * bs + k] += h[cell * bs + k];
        PetscCall(VecRestoreArray(s->u_work, &w));

        PetscCall(Rhs(ts, t, s->u_work, s->f_pert, s));

        const PetscScalar *f0, *fp;
        PetscCall(VecGetArrayRead(s->f_base, &f0));
        PetscCall(VecGetArrayRead(s->f_pert, &fp));
        for (PetscInt cell = 0; cell < nc; ++cell) {
            const PetscScalar inv_h = 1.0 / h[cell * bs + k];
            PetscScalar *blk = &s->blocks[static_cast<size_t>(cell) * bs * bs];
            for (PetscInt i = 0; i < bs; ++i) {
                blk[i * bs + k] = (fp[cell * bs + i] - f0[cell * bs + i]) * inv_h; // row-oriented: J(i,k)
            }
        }
        PetscCall(VecRestoreArrayRead(s->f_base, &f0));
        PetscCall(VecRestoreArrayRead(s->f_pert, &fp));
    }
    PetscCall(VecRestoreArray(s->h_pert, &h));


    // Never cache blocks built from a non-finite evaluation: every retry from this state would reuse
    // them and fail again. The stage built with them is rejected by CheckStage (its values are NaN).
    s->blocks_valid = (s->rhs_nonfinite == nonfinite_before);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// G(t, U, U') = U' - S(U)
PetscErrorCode StiffSolver::Impl::IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx) {
    PetscFunctionBeginUser;
    auto *s = static_cast<Impl *>(ctx);
    const long long before = s->rhs_nonfinite;
    PetscCall(Rhs(ts, t, U, F, ctx));   // F = S(U)
    // A non-finite source (e.g. a trial stage with a negative temperature, where the rates take
    // square roots and powers of T) must not reach PETSc's error estimate, which raises a
    // floating-point error on NaN. Flag it so CheckStage rejects the stage and PETSc retries the
    // step with a smaller internal step. F itself is zeroed so no NaN propagates in the meantime.
    if (s->rhs_nonfinite != before) {
        s->stage_nonfinite = true;
        PetscCall(VecZeroEntries(F));
    }
    PetscCall(VecAYPX(F, -1.0, Udot));  // F = Udot - S(U)
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Called by PETSc after each stage (TSAdaptCheckStage). Y is the stage increment, not the state.
// Rejecting here makes PETSc shrink the internal step (by scale_solve_failed, 0.25) and retry,
// without touching the accepted solution.
PetscErrorCode StiffSolver::Impl::CheckStage(TSAdapt, TS ts, PetscReal, Vec Y, PetscBool *accept) {
    PetscFunctionBeginUser;
    void *ctx;
    PetscCall(TSGetApplicationContext(ts, &ctx));
    auto *s = static_cast<Impl *>(ctx);
    bool ok = !s->stage_nonfinite;
    if (ok) {
        const PetscScalar *y;
        PetscInt n;
        PetscCall(VecGetLocalSize(Y, &n));
        PetscCall(VecGetArrayRead(Y, &y));
        for (PetscInt i = 0; i < n; ++i) {
            if (!std::isfinite(static_cast<double>(y[i]))) { ok = false; break; }
        }
        PetscCall(VecRestoreArrayRead(Y, &y));
    }
    s->stage_nonfinite = false;
    if (!ok) {
        ++s->stage_rejects;
        // TSStep_RosW sets the SNES Jacobian lag to -2 (build once, then reuse) at the first stage and
        // restores it only after the LAST stage. When a stage is rejected part-way through a step that
        // restore never happens, so every retry would reuse the Jacobian built for the old (larger)
        // step size, with the wrong shift, and fail. Restoring the default lag here makes the retry
        // rebuild it (the finite-difference blocks themselves are cached, so this is cheap).
        SNES snes;
        PetscCall(TSGetSNES(ts, &snes));
        PetscCall(SNESSetLagJacobian(snes, 1));
    }
    *accept = ok ? PETSC_TRUE : PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
}

// dG/dU + shift * dG/dU' = shift*I - dS/dU, block diagonal
PetscErrorCode StiffSolver::Impl::IJacobian(TS ts, PetscReal t, Vec U, Vec, PetscReal shift, Mat A, Mat P, void *ctx) {
    PetscFunctionBeginUser;
    auto *s = static_cast<Impl *>(ctx);
    const PetscInt bs = s->stride;
    const PetscInt nc = s->ncells;
    PetscCall(BuildBlocks(ts, t, U, s));

    std::vector<PetscScalar> blk(static_cast<size_t>(bs) * bs);
    for (PetscInt cell = 0; cell < nc; ++cell) {
        const PetscScalar *J = &s->blocks[static_cast<size_t>(cell) * bs * bs];
        for (PetscInt i = 0; i < bs; ++i) {
            for (PetscInt k = 0; k < bs; ++k) {
                blk[i * bs + k] = (i == k ? shift : 0.0) - J[i * bs + k]; // row-oriented
            }
        }
        PetscCall(MatSetValuesBlocked(P, 1, &cell, 1, &cell, blk.data(), INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
    if (A != P) { // e.g. -snes_mf_operator: A is matrix-free, P is our block matrix
        PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Reject stages that are non-finite or non-positive in the checked cells; the adaptive time
// stepper then retries with a smaller step.
PetscErrorCode StiffSolver::Impl::Domain(TS ts, PetscReal, Vec U, PetscBool *accept) {
    PetscFunctionBeginUser;
    void *ctx;
    PetscCall(TSGetApplicationContext(ts, &ctx));
    auto *s = static_cast<Impl *>(ctx);

    const PetscScalar *u;
    PetscCall(VecGetArrayRead(U, &u));
    long long bad_count = 0, first_bad_index = -1;
    double first_bad_value = 0.0;
    double c0_min = DBL_MAX, c0_max = -DBL_MAX;
    for (PetscInt cell = 0; cell < s->ncells; ++cell) {
        if (!s->check_cell[cell]) continue;
        const PetscScalar *v = &u[cell * s->stride];
        c0_min = std::min(c0_min, static_cast<double>(PetscRealPart(v[0])));
        c0_max = std::max(c0_max, static_cast<double>(PetscRealPart(v[0])));
        long long bad_index = -1;
        double bad_value = 0.0;
        for (PetscInt k = 0; k < s->stride && bad_index < 0; ++k) {
            if (!std::isfinite(static_cast<double>(PetscRealPart(v[k])))) {
                bad_index = static_cast<long long>(cell) * s->stride + k;
                bad_value = static_cast<double>(PetscRealPart(v[k]));
            }
        }
        for (int comp : s->positive) {
            if (bad_index < 0 && !(PetscRealPart(v[comp]) > 0.0)) {
                bad_index = static_cast<long long>(cell) * s->stride + comp;
                bad_value = static_cast<double>(PetscRealPart(v[comp]));
            }
        }
        if (bad_index >= 0) {
            if (bad_count == 0) { first_bad_index = bad_index; first_bad_value = bad_value; }
            ++bad_count;
        }
    }
    const bool ok = (bad_count == 0);
    if (!ok && s->domain_rejects == 0) {
        // remember the first refused stage of this solve
        double sumsq = 0.0;
        const size_t n = static_cast<size_t>(s->ncells) * s->stride;
        for (size_t i = 0; i < n; ++i) sumsq += static_cast<double>(PetscRealPart(u[i])) * static_cast<double>(PetscRealPart(u[i]));
        PetscReal dt_now;
        PetscCall(TSGetTimeStep(ts, &dt_now));
        s->domain_bad_index = first_bad_index;
        s->domain_bad_value = first_bad_value;
        s->domain_bad_count = bad_count;
        s->domain_vec_norm = std::sqrt(sumsq);
        s->domain_c0_min = c0_min;
        s->domain_c0_max = c0_max;
        s->domain_first_dt = static_cast<double>(dt_now);
    }
    PetscCall(VecRestoreArrayRead(U, &u));
    if (!ok) ++s->domain_rejects;
    *accept = ok ? PETSC_TRUE : PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
}

StiffSolver::StiffSolver(std::int64_t ncells, int stride,
                         const std::vector<unsigned char> &check_cell,
                         const std::vector<int> &positive_comps,
                         RhsFunction rhs, void *user, double rtol, double atol)
    : impl(new Impl) {
    Impl &s = *impl;
    s.ncells = static_cast<PetscInt>(ncells);
    s.stride = static_cast<PetscInt>(stride);
    s.rhs = rhs;
    s.user = user;
    s.check_cell = check_cell;
    s.positive = positive_comps;

    // SAMS may already have initialised MPI; PETSc then reuses it and will not finalise it.
    PetscBool initialized = PETSC_FALSE;
    SS_ABORT(PetscInitialized(&initialized));
    if (!initialized) {
        SS_ABORT(PetscInitializeNoArguments());
        g_petsc_owner = true;
    }

    const PetscInt n = s.ncells * s.stride;

    // Sequential host vectors on PETSC_COMM_SELF: the problem needs no communication
    SS_ABORT(VecCreateSeq(PETSC_COMM_SELF, n, &s.u_sol));
    SS_ABORT(VecSetBlockSize(s.u_sol, s.stride));
    SS_ABORT(VecDuplicate(s.u_sol, &s.u_work));
    SS_ABORT(VecDuplicate(s.u_sol, &s.f_base));
    SS_ABORT(VecDuplicate(s.u_sol, &s.f_pert));
    SS_ABORT(VecDuplicate(s.u_sol, &s.h_pert));
    s.blocks.assign(static_cast<size_t>(s.ncells) * s.stride * s.stride, 0.0);

    // Block diagonal Jacobian: one stride x stride block per cell
    SS_ABORT(MatCreate(PETSC_COMM_SELF, &s.jac));
    SS_ABORT(MatSetSizes(s.jac, n, n, n, n));
    SS_ABORT(MatSetType(s.jac, MATSEQBAIJ));
    SS_ABORT(MatSeqBAIJSetPreallocation(s.jac, s.stride, 1, nullptr));
    SS_ABORT(MatSetUp(s.jac));

    // Rosenbrock-W time stepper
    SS_ABORT(TSCreate(PETSC_COMM_SELF, &s.ts));
    SS_ABORT(TSSetProblemType(s.ts, TS_NONLINEAR));
    SS_ABORT(TSSetType(s.ts, TSROSW));
    SS_ABORT(TSRosWSetType(s.ts, TSROSWRA34PW2));
    SS_ABORT(TSSetApplicationContext(s.ts, impl));
    // Stiff sources in the IMPLICIT slot (see the note at the top of this file): an RHS function
    // would be integrated explicitly by TSROSW.
    SS_ABORT(TSSetIFunction(s.ts, nullptr, Impl::IFunction, impl));
    SS_ABORT(TSSetIJacobian(s.ts, s.jac, s.jac, Impl::IJacobian, impl));
    {
        // PETSc's validity hook is OFF by default: for TSROSW the vector it receives is not
        // guaranteed to be the stage state (it can be the stage increment, which is legitimately
        // zero or negative), so a positivity test on it wrongly rejects every step. SAMS validates
        // the starting state and the final solution itself instead.
        // PETSC_OPTIONS="-tf_domain_check 1" turns it on (for experiments / diagnostics).
        PetscBool dc = PETSC_FALSE;
        SS_ABORT(PetscOptionsGetBool(nullptr, nullptr, "-tf_domain_check", &dc, nullptr));
        s.domain_check = (dc == PETSC_TRUE);
        if (s.domain_check) SS_ABORT(TSSetFunctionDomainError(s.ts, Impl::Domain));
    }
    SS_ABORT(TSSetExactFinalTime(s.ts, TS_EXACTFINALTIME_MATCHSTEP));
    SS_ABORT(TSSetMaxSteps(s.ts, 100000000));
    SS_ABORT(TSSetErrorIfStepFails(s.ts, PETSC_FALSE)); // report failures through SolveInfo instead of a PETSc error
    SS_ABORT(TSSetTolerances(s.ts, atol, nullptr, rtol, nullptr));

    // Linear solve: the Jacobian is exactly block diagonal, so a direct solve has no fill.
    // (-pc_type pbjacobi is a cheaper alternative to try.)
    // (For a nonlinear TS the KSP is reached through the internal SNES, not TSGetKSP.)
    SNES snes;
    KSP  ksp;
    PC   pc;
    SS_ABORT(TSGetSNES(s.ts, &snes));
    SS_ABORT(SNESGetKSP(snes, &ksp));
    SS_ABORT(KSPSetType(ksp, KSPPREONLY));
    SS_ABORT(KSPGetPC(ksp, &pc));
    SS_ABORT(PCSetType(pc, PCLU));

    // Options from PETSC_OPTIONS / a petscrc file override everything set above
    SS_ABORT(TSSetFromOptions(s.ts));
    {
        // Reject stages with non-finite sources or stage values (see CheckStage). After TSSetFromOptions, so
        // a -ts_adapt_type option cannot replace the adaptor it is attached to.
        TSAdapt adapt;
        SS_ABORT(TSGetAdapt(s.ts, &adapt));
        SS_ABORT(TSAdaptSetCheckStage(adapt, Impl::CheckStage));
    }
}

StiffSolver::~StiffSolver() {
    PetscBool finalized = PETSC_FALSE;
    PetscFinalized(&finalized);
    if (!finalized) { // PETSc objects must not be destroyed after PetscFinalize
        Impl &s = *impl;
        if (s.u_sol)  VecDestroy(&s.u_sol);
        if (s.u_work) VecDestroy(&s.u_work);
        if (s.f_base) VecDestroy(&s.f_base);
        if (s.f_pert) VecDestroy(&s.f_pert);
        if (s.h_pert) VecDestroy(&s.h_pert);
        if (s.jac)    MatDestroy(&s.jac);
        if (s.ts)     TSDestroy(&s.ts);
    }
    delete impl;
}

void StiffSolver::setUser(void *user) {
    impl->user = user;
}

bool StiffSolver::solve(double *u, double dt, SolveInfo &info) {
    Impl &s = *impl;
    const size_t n = static_cast<size_t>(s.ncells) * s.stride;

    PetscScalar *a;
    SS_ABORT(VecGetArray(s.u_sol, &a));
    std::copy(u, u + n, a);
    SS_ABORT(VecRestoreArray(s.u_sol, &a));

    s.domain_rejects = 0;
    s.domain_bad_index = -1;
    s.domain_bad_value = 0.0;
    s.domain_bad_count = 0;
    s.domain_vec_norm = 0.0;
    s.domain_c0_min = s.domain_c0_max = 0.0;
    s.domain_first_dt = 0.0;
    s.rhs_nonfinite = 0;
    s.rhs_bad_index = -1;
    s.blocks_valid = false;
    s.stage_nonfinite = false;
    s.stage_rejects = 0;

    // The source terms do not depend explicitly on time, so solve on t = [0, dt]
    SS_ABORT(TSSetStepNumber(s.ts, 0));
    SS_ABORT(TSSetTime(s.ts, 0.0));
    SS_ABORT(TSSetMaxTime(s.ts, static_cast<PetscReal>(dt)));
    const PetscReal dt0 = (s.dt_hint > 0.0 && s.dt_hint < dt) ? s.dt_hint : static_cast<PetscReal>(dt);
    SS_ABORT(TSSetTimeStep(s.ts, dt0));

    // A PETSc error inside TSSolve (for example a NaN that still reaches the error estimate) is
    // returned as a failed solve rather than aborting the process, so the caller can roll back and
    // retry with smaller steps. The error handler is swapped so PETSc does not print the traceback.
    PetscCallAbort(PETSC_COMM_SELF, PetscPushErrorHandler(PetscReturnErrorHandler, nullptr));
    const PetscErrorCode solve_err = TSSolve(s.ts, s.u_sol);
    PetscCallAbort(PETSC_COMM_SELF, PetscPopErrorHandler());
    info.stage_rejects = s.stage_rejects;
    info.rhs_nonfinite = s.rhs_nonfinite;
    info.rhs_bad_index = s.rhs_bad_index;
    if (solve_err != PETSC_SUCCESS) {
        info.petsc_error = static_cast<int>(solve_err);
        info.reason = -1000;
        return false;
    }

    TSConvergedReason reason;
    PetscInt steps, rejects;
    SS_ABORT(TSGetConvergedReason(s.ts, &reason));
    SS_ABORT(TSGetStepNumber(s.ts, &steps));
    SS_ABORT(TSGetStepRejections(s.ts, &rejects));
    info.reason = static_cast<int>(reason);
    info.steps = static_cast<int>(steps);
    info.rejected = static_cast<int>(rejects);
    PetscReal t_reached, dt_last;
    SS_ABORT(TSGetTime(s.ts, &t_reached));
    SS_ABORT(TSGetTimeStep(s.ts, &dt_last));
    info.time_reached = static_cast<double>(t_reached);
    info.last_dt = static_cast<double>(dt_last);
    info.domain_rejects = s.domain_rejects;
    info.domain_bad_index = s.domain_bad_index;
    info.domain_bad_value = s.domain_bad_value;
    info.domain_bad_count = s.domain_bad_count;
    info.domain_vec_norm = s.domain_vec_norm;
    info.domain_c0_min = s.domain_c0_min;
    info.domain_c0_max = s.domain_c0_max;
    info.domain_first_dt = s.domain_first_dt;
    info.rhs_nonfinite = s.rhs_nonfinite;
    info.rhs_bad_index = s.rhs_bad_index;
    if (reason < 0) return false;

    SS_ABORT(TSGetTimeStep(s.ts, &s.dt_hint));
    SS_ABORT(VecGetArray(s.u_sol, &a));
    std::copy(a, a + n, u);
    SS_ABORT(VecRestoreArray(s.u_sol, &a));
    return true;
}

std::string StiffSolver::describe() const {
    const Impl &s = *impl;
    TSType ts_type = nullptr;
    SS_ABORT(TSGetType(s.ts, &ts_type));

    // Rosenbrock-W scheme name: only meaningful when the stepper is still TSROSW
    // (PETSC_OPTIONS="-ts_type ..." can replace it)
    const char *rosw_type = "n/a";
    PetscBool is_rosw = PETSC_FALSE;
    SS_ABORT(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(s.ts), TSROSW, &is_rosw));
    TSRosWType rt = nullptr;
    if (is_rosw) {
        SS_ABORT(TSRosWGetType(s.ts, &rt));
        if (rt) rosw_type = rt;
    }

    PetscReal atol = 0.0, rtol = 0.0;
    SS_ABORT(TSGetTolerances(s.ts, &atol, nullptr, &rtol, nullptr));

    TSAdapt adapt;
    TSAdaptType adapt_type = nullptr;
    SS_ABORT(TSGetAdapt(s.ts, &adapt));
    SS_ABORT(TSAdaptGetType(adapt, &adapt_type));

    SNES snes;
    KSP ksp;
    PC pc;
    KSPType ksp_type = nullptr;
    PCType pc_type = nullptr;
    SS_ABORT(TSGetSNES(s.ts, &snes));
    SS_ABORT(SNESGetKSP(snes, &ksp));
    SS_ABORT(KSPGetType(ksp, &ksp_type));
    SS_ABORT(KSPGetPC(ksp, &pc));
    SS_ABORT(PCGetType(pc, &pc_type));

    PetscInt max_steps = 0;
    SS_ABORT(TSGetMaxSteps(s.ts, &max_steps));

    char buf[1024];
    std::snprintf(buf, sizeof buf,
        "  PETSc time stepper     : %s (scheme %s)\n"
        "  PETSc tolerances       : rtol = %.3e, atol = %.3e\n"
        "  PETSc step adaptor     : %s, max internal steps %lld\n"
        "  PETSc linear solver    : KSP %s, PC %s\n"
        "  Jacobian               : block diagonal %lldx%lld, finite differences, %lld cells\n"
        "  Stage domain check     : %s\n",
        ts_type ? ts_type : "?", rosw_type,
        static_cast<double>(rtol), static_cast<double>(atol),
        adapt_type ? adapt_type : "?", static_cast<long long>(max_steps),
        ksp_type ? ksp_type : "?", pc_type ? pc_type : "?",
        static_cast<long long>(s.stride), static_cast<long long>(s.stride), static_cast<long long>(s.ncells),
        s.domain_check ? "on (-tf_domain_check 1)" : "off");
    return std::string(buf);
}

std::string petscVersion() {
    char buf[256] = {0};
    if (PetscGetVersion(buf, sizeof buf) != PETSC_SUCCESS) return "unknown";
    return std::string(buf);
}

void finalizePETSc() {
    if (g_petsc_owner) {
        PetscFinalize();
        g_petsc_owner = false;
    }
}

} // namespace SAMS_PETSC
