// Stand-in for petscStiffSolver.cpp in builds without PETSc (CMake USE_PETSC=OFF).
// It provides the same interface so the two-fluid package links; the explicit source update
// (implicit=false) needs none of it. Asking for the implicit solver stops the run with a message
// (normally already at start-up, in PIP::printConfiguration).

#include "petscStiffSolver.h"

#ifndef SAMS_HAVE_PETSC

#include <stdexcept>

namespace SAMS_PETSC {

    namespace {
        [[noreturn]] void noPETSc()
        {
            throw std::runtime_error("this build has no PETSc (CMake USE_PETSC=OFF), so the implicit two-fluid "
                                     "source solver is not available: set implicit=false or rebuild with -DUSE_PETSC=ON");
        }
    }

    struct StiffSolver::Impl {};

    StiffSolver::StiffSolver(std::int64_t, int, const std::vector<unsigned char> &, const std::vector<int> &,
                             RhsFunction, void *, double, double) : impl(nullptr) { noPETSc(); }
    StiffSolver::~StiffSolver() = default;
    void StiffSolver::setUser(void *) { noPETSc(); }
    bool StiffSolver::solve(double *, double, SolveInfo &) { noPETSc(); }
    std::string StiffSolver::describe() const { noPETSc(); }

    std::string petscVersion() { return "not built (USE_PETSC=OFF)"; }
    void finalizePETSc() {}

} // namespace SAMS_PETSC

#endif // !SAMS_HAVE_PETSC
