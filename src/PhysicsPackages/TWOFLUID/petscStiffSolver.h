#ifndef SAMS_PETSC_STIFF_SOLVER_H
#define SAMS_PETSC_STIFF_SOLVER_H

/*
 * Thin wrapper around PETSc's Rosenbrock-W time stepper for stiff, cell-wise source terms.
 *
 * This header deliberately includes neither PETSc nor any SAMS header. PETSc is only
 * compiled in petscStiffSolver.cpp, which avoids clashes between PETSc's mpi.h (or its
 * serial MPIUNI stub) and SAMS's own MPI types. The interface works on plain host arrays.
 *
 * The state is ncells*stride doubles, cell by cell (cell-major, `stride` variables per cell).
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace SAMS_PETSC {

    // Evaluate f = F(u). u and f are host arrays of ncells*stride doubles.
    using RhsFunction = void (*)(const double *u, double *f, void *user);

    struct SolveInfo {
        int steps = 0;      // accepted internal steps
        int rejected = 0;   // rejected internal steps
        int reason = 0;     // PETSc TSConvergedReason (negative = failure)

        // Diagnostics for failed solves. "index" is cell*stride + component.
        double time_reached = 0.0;           // internal time reached (target is dt)
        double last_dt = 0.0;                // last internal step size
        long long domain_rejects = 0;        // stages refused by the validity check
        // The following describe the FIRST stage refused by the validity check in this solve:
        long long domain_bad_index = -1;     // first offending entry
        double domain_bad_value = 0.0;
        long long domain_bad_count = 0;      // number of checked cells that failed
        double domain_vec_norm = 0.0;        // 2-norm of the whole vector that was checked
        double domain_c0_min = 0.0;          // min / max of component 0 over the checked cells
        double domain_c0_max = 0.0;
        double domain_first_dt = 0.0;        // internal step size when it was first refused
        long long rhs_nonfinite = 0;         // right hand side evaluations containing NaN/Inf
        long long rhs_bad_index = -1;        // first non-finite entry
        long long stage_rejects = 0;         // stages rejected (and retried with a smaller internal step)
                                             // because the source or the stage values were NaN/Inf
        int petsc_error = 0;                 // non-zero: TSSolve raised this PETSc error (solve failed)
    };

    class StiffSolver {
    public:
        /*
         * ncells, stride  : size of the state
         * check_cell      : ncells flags; only flagged cells are checked for physical validity
         *                   (a stage that fails the check is rejected and retried with a smaller step)
         * positive_comps  : components that must stay > 0 in checked cells; every component of a
         *                   checked cell must also be finite
         * rhs, user       : right hand side callback and its context
         * rtol, atol      : tolerances of the adaptive time stepper
         * PETSC_OPTIONS in the environment can override the defaults chosen here.
         */
        StiffSolver(std::int64_t ncells, int stride,
                    const std::vector<unsigned char> &check_cell,
                    const std::vector<int> &positive_comps,
                    RhsFunction rhs, void *user, double rtol, double atol);
        ~StiffSolver();
        StiffSolver(const StiffSolver &) = delete;
        StiffSolver &operator=(const StiffSolver &) = delete;

        // The user pointer is refreshed before each solve because the owner may have moved.
        void setUser(void *user);

        // Advance u in place over [0, dt]. Returns false if the solve failed (u is then untouched).
        bool solve(double *u, double dt, SolveInfo &info);

        // Multi-line summary of the settings PETSc is actually using (after PETSC_OPTIONS /
        // petscrc overrides): time stepper, scheme, tolerances, step adaptor, linear solver.
        std::string describe() const;

    private:
        struct Impl;
        Impl *impl;
    };

    // PETSc version string, e.g. "PETSc Release Version 3.20.5, ...". Safe to call before PETSc
    // is initialised (it only reads compile-time information).
    std::string petscVersion();

    // Call once at the end of the run, after all StiffSolver objects are destroyed.
    void finalizePETSc();

} // namespace SAMS_PETSC

#endif // SAMS_PETSC_STIFF_SOLVER_H
