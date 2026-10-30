#define PRINT_PARALLELIZATION_INFO

#include "pp/range.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "harness.h"
#include "shared_data.h"
#ifdef USE_KOKKOS
#include "LAREKokkos/shared_data.h"
#endif
#include "timer.h"
#include "axisRegistry.h"
#include "variableRegistry.h"
#include "mpiManager.h"
#include "welcome.h"

#include "tags.h"
#include "braginskii.h"
#include "mg.h"
#ifdef USE_PETSC
#include "petscMatrix.h"
#include "petscNoMatrix.h"
#endif
#include "TCTest.h"

//Example initial conditions
#include "SodShockTube.h"
#include "BrioAndWu.h"
#include "MHDRotor.h"
#include "OrszagTang.h"
#include "OrszagTang3D.h"
#include "EmeryWindTunnel.h"
#include "KarmanVortex.h"

#include "builtInBoundaryConditions.h"

#include "runner.h"
#include "lareic.h"

int main(int argc, char *argv[]){

    //Initialize MPI
    SAMS::MPI::initialize(argc, argv);
    //Initialize portable wrapper
    portableWrapper::initialize(argc, argv);

    //Print welcome message
    SAMS::printWelcomeMessage();

    //Create and initialize the runner
    SAMS::runner<
    LARE::LARE3D<>, //The main LARE3D solver
    LARE::LARE3DInitialConditions<>,// Classic LARE style initial conditions
    examples::SodShockTube<>, //Sod shock tube initial conditions
    examples::BrioAndWu<>, //Brio and Wu initial conditions
    examples::MHDRotor<>, //MHD rotor initial conditions
    examples::OrszagTang<>, //Orszag-Tang initial conditions
    examples::OrszagTang3D<>, //Orszag-Tang 3D initial conditions
    examples::EmeryWindTunnel<>, //Emery wind tunnel initial conditions
    examples::KarmanVortex<>, //Karman vortex initial conditions
    SAMS::TCTest<SAMS::braginskii<SAMS::electron>>, //Thermal conduction test initial conditions for electrons
    SAMS::TCTest<SAMS::braginskii<SAMS::ion>>, //Thermal conduction test initial conditions for ions
    SAMS::parabolic::mg<SAMS::braginskii<SAMS::electron>>, //Multigrid solver for Braginskii electron equations
    SAMS::parabolic::mg<SAMS::braginskii<SAMS::ion>> //Multigrid solver for Braginskii ion equations
#ifdef USE_PETSC
    ,SAMS::parabolic::petscMatrix<SAMS::braginskii<SAMS::electron>>, //PETSc solver with a matrix for Braginskii electron equations
    SAMS::parabolic::petscMatrix<SAMS::braginskii<SAMS::ion>>, //PETSc solver with a matrix for Braginskii ion equations
    SAMS::parabolic::petscNoMatrix<SAMS::braginskii<SAMS::electron>>, //PETSc solver without a matrix for Braginskii electron equations
    SAMS::parabolic::petscNoMatrix<SAMS::braginskii<SAMS::ion>> //PETSc solver without a matrix for Braginskii ion equations
#endif
#ifdef USE_KOKKOS
    ,LARE::LARE3DKokkos //LARE3D written directly in Kokkos, to compare with the portable wrapper
#endif
    > runner;
    runner.initialize(argc, argv);
    //Finish welcome message
    SAMS::finishWelcomeMessage();
    
    //Use the parameters passed to set up and run the simulations
    for (int i=1;i<argc;i++){
        std::string argStr = argv[i];
        runner.activatePackage(argStr);
    }
    //Initialize the packages
    runner.initializePackages();
    //Run the packages until a package requests to stop
    runner.runPackages();
    //Finish the packages
    runner.finalizePackages();
    //Finalize the runner
    runner.finalize();
    portableWrapper::finalize();
		SAMS::MPI::finalize();
}
