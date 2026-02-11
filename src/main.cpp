#define PRINT_PARALLELIZATION_INFO

#include "pp/range.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "harness.h"
#include "shared_data.h"
#include "timer.h"
#include "axisRegistry.h"
#include "variableRegistry.h"
#include "mpiManager.h"
#include "welcome.h"

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
    SAMS::runner<LARE::LARE3D, LARE::LARE3DInitialConditions, examples::SodShockTube, examples::BrioAndWu, examples::MHDRotor, examples::OrszagTang, examples::OrszagTang3D, examples::EmeryWindTunnel, 
        examples::KarmanVortex> runner;
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
}
