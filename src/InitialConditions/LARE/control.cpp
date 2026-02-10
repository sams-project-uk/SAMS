#include "lareic.h"
#include "lareBoundaryClass.h"

namespace LARE{

    void LARE3DInitialConditions::control_variables([[maybe_unused]] SAMS::harness &harnessRef, [[maybe_unused]] simulationData &data){
        std::cout << "Setting control variables for LARE3D simulation" << std::endl;
        data.nx = 128; // Number of cells in the x-direction
        data.ny = 128; // Number of cells in the y-direction
        data.nz = 128; // Number of cells in the z-direction

        data.dt_multiplier = 0.8; // Default multiplier for time step
        data.dt = 0.0;

        // Maximum number of iterations; if nsteps < 0, run until t_end
        data.nsteps = 10;
        data.t_end = 60.0 * 60.0 * 24.0 * 24.0; // One day in seconds

        // Geometry options: cartesian, cylindrical, spherical
        data.geometry = geometryType::Cartesian;

        // Domain limits
        data.x_min = -1.0e6;
        data.x_max = 1.0e6;
        data.y_min = -1.0e6;
        data.y_max = 1.0e6;
        data.z_min = -1.0e6;
        data.z_max = 1.0e6;

        // Boundary conditions
        data.xbc_min = BCType::BC_OTHER;
        data.xbc_max = BCType::BC_OTHER;
        data.ybc_min = BCType::BC_OTHER;
        data.ybc_max = BCType::BC_OTHER;
        data.zbc_min = BCType::BC_OTHER;
        data.zbc_max = BCType::BC_OTHER;

        // Grid stretching
        data.x_stretch = false;
        data.y_stretch = false;
        data.z_stretch = false;

        // Shock viscosity coefficients
        data.visc1 = 0.1;
        data.visc2 = 1.0;

        // Ratio of specific heat capacities
        data.gas_gamma = 1.4;

        // Average mass of an ion in proton masses
        data.mf = 1.2;

        // Resistive MHD options
        data.resistiveMHD = false;
        data.eta_background = 1.e-10;
        data.j_max = 1.0;
        data.eta0 = 2.e-10;

        // Remap kinetic energy correction
        data.rke = true;

        // Output frequency and directory
        data.dt_snapshots = 10.0;
    }

 void LARE3DInitialConditions::initial_conditions([[maybe_unused]] SAMS::harness &harness, [[maybe_unused]] simulationData &data)
  {

    SAMS::cout << "Setting up initial conditions" << std::endl;
    // Set initial conditions for the LARE3D
    pw::assign(data.vx, 0.0);
    pw::assign(data.vy, 0.0);
    pw::assign(data.vz, 0.0);

    pw::assign(data.rho, 1.0);
    pw::assign(data.energy_electron, 1.0);
    pw::assign(data.energy_ion, 1.0);

    pw::assign(data.bx, 0.0);
    pw::assign(data.by, 0.0);
    pw::assign(data.bz, 0.0);
  }

}//namespace LARE