/*
 *    Copyright 2025 SAMS Team
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "shared_data.h"

void simulation::controlvariables(simulationData &data) {

  data.nx=128; // Number of cells in the x-direction
  data.ny=128; // Number of cells in the y-direction
  data.nz=128; // Number of cells in the z-direction

  data.dt_multiplier = 0.8; // Default multiplier for time step
  data.dt=0.0;

  // Maximum number of iterations; if nsteps < 0, run until t_end
  data.nsteps = 40;
  data.t_end = 60.0 * 60.0 * 24.0; // One day in seconds

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

void simulation::initial_conditions(simulationData &data) {

  using Range = portableWrapper::Range;

  //std::cout << "Setting up initial conditions" << std::endl;
  // Set initial conditions for the simulation
  portableWrapper::assign(data.vx,0.0);
  portableWrapper::assign(data.vy,0.0);
  portableWrapper::assign(data.vz,0.0);

  printf("Setting up initial conditions\n");
  
  T_dataType v0 = 0.e3;
  T_dataType a0 = 1.0e5;
  T_dataType a2 = a0 * a0;

  // Set the initial thermal energy of electrons and ions
  T_dataType T0 = 1.e6;
  T_dataType energy = 0.5 * kb_si * T0 / mh_si / (data.gas_gamma - 1.0);
  portableWrapper::assign(data.energy_electron, energy);
  portableWrapper::assign(data.energy_ion, energy);

  portableWrapper::applyKernel(
    LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
      T_dataType r2 = data.xb(ix) * data.xb(ix) + data.yb(iy) * data.yb(iy) + data.zb(iz) * data.zb(iz);
      T_dataType v = v0 * std::exp(-r2 / a2);
      data.vx(ix, iy, iz) = data.xb(ix)/1.0e6 * v;
      data.vy(ix, iy, iz) = data.yb(iy)/1.0e6 * v;
      data.vz(ix, iy, iz) = data.zb(iz)/1.0e6 * v;

      data.energy_electron(ix,iy,iz) *= (1.0+std::exp(-r2 / a2));
      data.energy_ion(ix,iy,iz) *= (1.0+std::exp(-r2 / a2));
    },
    portableWrapper::Range(0, data.nx),
    portableWrapper::Range(0, data.ny),
    portableWrapper::Range(0, data.nz)
  );

  std::cout << "Range of vx: " << portableWrapper::minval(data.vx) << " to " << portableWrapper::maxval(data.vx) << "\n";
  std::cout << "Range of vy: " << portableWrapper::minval(data.vy) << " to " << portableWrapper::maxval(data.vy) << "\n";
  std::cout << "Range of vz: " << portableWrapper::minval(data.vz) << " to " << portableWrapper::maxval(data.vz) << "\n";

  T_dataType bmult = 000.0;
  portableWrapper::assign(data.bx,0.01*bmult);
  portableWrapper::assign(data.by,0.00*bmult);
  portableWrapper::assign(data.bz,0.00*bmult);
  // Set the initial density field in kg/m^3
  portableWrapper::assign(data.rho, 1.0e-6);

}
