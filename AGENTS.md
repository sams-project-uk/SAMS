# Guidance for coding agents working on SAMS

SAMS, the Solar Atmospheric Modelling Suite, is a performance portable C++20 HPC code. Read `README.md` and the documents in `docs/` before making changes. `docs/Developing SAMS Packages.md` describes how packages and the runner work.

## Layout

* `include/harness/` - The harness: axis, variable and memory registries, MPI, IO
* `include/runner/` - The runner, which calls the events of the packages, and shared deck key names (`commonNames.h`)
* `include/boundaryConditions/` - The boundary condition base class and the built in boundary conditions
* `include/KVS/` - The key value store and the deck readers and writers
* `src/PhysicsPackages/` - Solvers. LARE3D is the main solver. HLLE3D and KT3D only exist to show that more than one solver can be built in, and should not be used as examples
* `src/InitialConditions/` - Initial condition packages, including those read from the deck
* `src/Operations/` - Output control, restarts, termination and derived quantities
* `src/main.cpp` - The list of packages built into the executable

## Building

SAMS uses CMake. All `.cpp` files under `src/` are found automatically. Options include `MPI`, `KOKKOS_CUDA`, `KOKKOS_HIP`, `KOKKOS_SYCL`, `KOKKOS_OPENMP`, `NATIVE_CUDA`, `NATIVE_HIP`, `HDF5`, `ADIOS2`, `PETSC` and `DECK_PARSER`. Code must work with all of the backends, with comparable functionality on each.

## No defensive code

SAMS does not check for misuse by its own code. Do not add argument, range, null, rank, type or state checks, throws or asserts for validation, or catch blocks, fallbacks and default branches added "to be safe". This applies everywhere, not only in kernels: setup code, IO code and helpers included. It *does* check for errors in data provided by the user. It does *not* check for errors in the code itself. An existing check is not a reason to add a similar one. Short `static_assert`s are acceptable, because they cost nothing at run time.

The reasons are:

* Performance at scale is expensive. SAMS is written to run on systems up to exascale, where a run can occupy thousands of nodes for days. A machine of that size draws around 20 MW, so a code that is 10% slower than it needs to be wastes the equivalent of 2 MW of power for the whole run, and does 10% less science with an allocation that took months to win. Small costs are not small when multiplied by billions of cells, millions of steps and thousands of ranks, and checks written in setup code tend to be copied into hot paths
* The cost of a check is not the comparison. A branch inside a kernel can stop the compiler vectorising the loop, and on GPUs it causes threads to diverge. Code compiled as a device kernel cannot throw, so a check there either does nothing useful or needs special handling
* Fallbacks are worse than failures. A "safe" default branch or fallback turns a programming error into a run that completes and produces wrong physics without telling anyone. A crash at least shows that something is wrong
* Under MPI a local check can hang the run. A check that aborts or branches on one rank and not the others can leave the other ranks waiting in a collective operation
* Checking is chosen at build time. SAMS provides it where it is wanted through the CMake options `ARRAY_BOUNDS_CHECKING`, `ARRAY_SET_CHECKER` and `DEBUG_LEVEL`, which developers turn on to find errors and production builds do not pay for. A hand written check bypasses that choice and costs every build
* SAMS works by contract. Packages rely on each other to do their part, for example a solver relies on an initial conditions package to set the domain. A broken contract is a programming error, to be found and fixed during development, not tested for in every run
* Checks hide the algorithm. Code that states what it does, without layers of guards, is easier to read, review and optimise

Defensive code is different from reporting real failures, which is required. Errors from files, MPI, HDF5, ADIOS2, PETSc and other external libraries must be reported with a clear message, and so must errors in the input deck, because the user controls these and they can fail in a correct program. If you think a check is needed, say so and explain why rather than adding it.

## Other rules for changes

* Do only what was asked. Do not refactor, rename or add features that were not requested. If something looks wrong, report it rather than changing it
* Use C++20
* Do not use leading or trailing underscores in names
* Avoid inheritance except where an existing design uses it, such as boundary conditions
* Comments must match the style of the surrounding code and be brief
* Kernels use `pw::applyKernel` with `LAMBDA`. nvcc does not allow device lambdas in private member functions, or private nested types in kernels
* Use the `SAMSMPI` types (`SAMSMPI::comm`, `SAMSMPI::datatype`) rather than raw MPI types, so that the code builds without MPI
* Gate optional libraries on `USE_HDF5`, `USE_ADIOS2`, `USE_PETSC` and `USE_MPI`, not on `__has_include`
* The harness and the writers must not depend on the deck
* Never leave the code in a state that does not compile

## Comments

* A member, method or type has a one line `/** @brief ... */` saying what it is or what it does. A longer doc block, with `@param` and `@return`, is only for a function whose arguments need explaining
* An inline comment is one line of `//`, saying what a block of code does, or why, where the code does not make that clear
* Where a good programmer would have trouble working out the purpose of code from the code itself, for example because it is complex or heavily optimised, explain it, and give a clearer version of the code in the comment. If the clearer version would be more than a few lines, explain in words only
* Where code implements a published technique or algorithm, cite it if possible
* Otherwise comments describe the code as it is. They do not repeat what the code says, justify design choices or record history
* Comments are for a maintainer who can read the code, not for a reviewer to be convinced that it is correct. Do not comment each step, as in tutorial code
* Match the form of the comments next to the change, including whether there is a space after `//`
