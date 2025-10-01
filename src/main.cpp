#define PRINT_PARALLELIZATION_INFO

#include <iostream>
#include <iomanip>
#include <fstream>
#include "shared_data.h"
#include "include/timer.h"
#include "axisRegistry.h"
#include "variableRegistry.h"

int main(int argc, char *argv[]){
    portableWrapper::initialize(argc, argv);

    simulation S;
    simulationData data;

    S.controlvariables(data);
    auto& axRegistry = SAMS::getaxisRegistry();
    axRegistry.registerAxis("X");
    axRegistry.registerAxis("Y");
    axRegistry.registerAxis("Z");
    axRegistry.setElements("X", data.nx);
    axRegistry.setElements("Y", data.ny);
    axRegistry.setElements("Z", data.nz);
    
    S.registerVars();
    auto& varRegistry = SAMS::getvariableRegistry();
    varRegistry.allocateAll();
		S.allocate(data);
    S.grid(data);
    data.visc2_norm=data.visc2;
		portableWrapper::fence();
    S.initial_conditions(data);
		portableWrapper::fence();
    timer t;
    t.begin("Main Loop");
    data.step=0;

    while (true)
    {
      std::cout << data.step << " " << data.time << std::endl;
      if ((data.step >= data.nsteps && data.nsteps >= 0) || (data.time >= data.t_end))
        break;
      S.lagrangian_step(data);    // lagran.cpp
      S.eulerian_remap(data); // remap.cpp
      data.step++;
      if (data.rke) S.energy_correction(data); // diagnostics.cpp
      S.eta_calc(data);            // lagran.cpp
    }
    t.end();

		S.output(data);

		S.manager.clear();
    varRegistry.deallocateAll();
    portableWrapper::finalize();

}
