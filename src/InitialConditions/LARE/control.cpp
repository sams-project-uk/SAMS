#include "lareic.h"
#include "lareBoundaryClass.h"
#include <cmath>

namespace LARE{

    template<typename T_EOS>
    std::string LARE3DInitialConditions<T_EOS>::units(){
        // Either "SI" or "normalised"
        return "normalised";
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::control_variables([[maybe_unused]] SAMS::harness &harnessRef, icData &data, [[maybe_unused]] SAMS::KVS &kvs){
        std::cout << "Setting control variables for LARE3D simulation" << std::endl;
        // Number of cells in each direction
        data.nx_global = 128;
        data.ny_global = 128;
        data.nz_global = 128;

        // Multiplier for the time step
        data.dt_multiplier = 0.8;

        // Maximum number of iterations; if nsteps < 0, run until t_end
        data.nsteps = -1;
        data.t_end = 0.15;

        // Domain limits. The rotor is at the centre of the unit cube
        data.x_min = 0.0;
        data.x_max = 1.0;
        data.y_min = 0.0;
        data.y_max = 1.0;
        data.z_min = 0.0;
        data.z_max = 1.0;

        // Boundary conditions: BC_OTHER, BC_PERIODIC, BC_REFLECTIVE, BC_OUTFLOW, BC_INFLOW or BC_EXTERNAL
        //BC_PERIODIC - Periodic boundaries
        //BC_REFLECTIVE - Reflecting walls
        //BC_OUTFLOW - Outflow boundaries, with zero gradient in the normal direction
        //BC_INFLOW - Inflow boundaries, with the state set to the inflow state
        //BC_OTHER - Use the boundary conditions in boundary_other.cpp, intended to be user defined
        //BC_EXTERNAL - Boundaries defined by other SAMS packages, LAREIC will apply no boundaries to any variables on edges marked as BC_EXTERNAL

        // Reflecting walls on every side
        data.xbc_min = BCType::BC_REFLECTIVE;
        data.xbc_max = BCType::BC_REFLECTIVE;
        data.ybc_min = BCType::BC_REFLECTIVE;
        data.ybc_max = BCType::BC_REFLECTIVE;
        data.zbc_min = BCType::BC_REFLECTIVE;
        data.zbc_max = BCType::BC_REFLECTIVE;

        // Grid stretching
        data.x_stretch = false;
        data.y_stretch = false;
        data.z_stretch = false;

        // Shock viscosity coefficients
        data.visc1 = 0.0;
        data.visc2 = 0.0;

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
        data.rke = false;

        // Time between outputs
        data.dt_snapshots = 0.015;
    }

template<typename T_EOS>
 void LARE3DInitialConditions<T_EOS>::initial_conditions([[maybe_unused]] SAMS::harness &harness, icData &data)
  {
    // MHD rotor in 3D. A dense sphere spins about the z axis through the centre of the domain, in a uniform field along x
    // Cells are numbered 1 to n in each direction. Nodes and faces are numbered 0 to n
    const SAMS::T_dataType mag = 5.0 / std::sqrt(4.0 * M_PI);
    const SAMS::T_dataType r0 = 0.1;
    const SAMS::T_dataType r1 = 0.115;

    // State held by inflow boundaries. The first index is the direction (0 = x, 1 = y, 2 = z) and the second is the
    // edge (0 = lower, 1 = upper). Every edge holds the background state
    for (int direction = 0; direction < 3; ++direction){
      for (int edge = 0; edge < 2; ++edge){
        data.inflow[direction][edge].rho = 1.0;
        data.inflow[direction][edge].energy_electron = 1.0 / ((data.gas_gamma - 1.0) * 1.0) / 2.0;
        data.inflow[direction][edge].energy_ion = 1.0 / ((data.gas_gamma - 1.0) * 1.0) / 2.0;
        data.inflow[direction][edge].vx = 0.0;
        data.inflow[direction][edge].vy = 0.0;
        data.inflow[direction][edge].vz = 0.0;
        data.inflow[direction][edge].bx = mag;
        data.inflow[direction][edge].by = 0.0;
        data.inflow[direction][edge].bz = 0.0;
      }
    }

    // Density and specific internal energy in each cell. Uniform pressure
    for (SAMS::T_indexType i = 1; i <= data.nx; ++i){
      for (SAMS::T_indexType j = 1; j <= data.ny; ++j){
        for (SAMS::T_indexType k = 1; k <= data.nz; ++k){
          // Position of the cell centre relative to the centre of the domain
          SAMS::T_dataType x = data.xc(i) - 0.5;
          SAMS::T_dataType y = data.yc(j) - 0.5;
          SAMS::T_dataType z = data.zc(k) - 0.5;
          SAMS::T_dataType r = std::sqrt(x * x + y * y + z * z);
          SAMS::T_dataType pressure = 1.0;
          if (r <= r0){
            data.rho(i, j, k) = 10.0;
          } else if (r <= r1){
            data.rho(i, j, k) = 1.0 + 9.0 * (r1 - r) / (r1 - r0);
          } else {
            data.rho(i, j, k) = 1.0;
          }
          data.energy_electron(i, j, k) = pressure / ((data.gas_gamma - 1.0) * data.rho(i, j, k)) / 2.0;
          data.energy_ion(i, j, k) = data.energy_electron(i, j, k);
        }
      }
    }

    // Velocity at each node. Rigid rotation about the z axis, tapered to zero between r0 and r1
    for (SAMS::T_indexType i = 0; i <= data.nx; ++i){
      for (SAMS::T_indexType j = 0; j <= data.ny; ++j){
        for (SAMS::T_indexType k = 0; k <= data.nz; ++k){
          // Position of the node relative to the centre of the domain
          SAMS::T_dataType x = data.xb(i) - 0.5;
          SAMS::T_dataType y = data.yb(j) - 0.5;
          SAMS::T_dataType z = data.zb(k) - 0.5;
          SAMS::T_dataType r = std::sqrt(x * x + y * y + z * z);
          SAMS::T_dataType omega = 0.0;
          if (r <= r0){
            omega = 20.0;
          } else if (r <= r1){
            omega = 20.0 * (r1 - r) / (r1 - r0);
          }
          data.vx(i, j, k) = -omega * y;
          data.vy(i, j, k) = omega * x;
          data.vz(i, j, k) = 0.0;
        }
      }
    }

    // Magnetic field on each face. Uniform along x
    for (SAMS::T_indexType i = 0; i <= data.nx; ++i){
      for (SAMS::T_indexType j = 1; j <= data.ny; ++j){
        for (SAMS::T_indexType k = 1; k <= data.nz; ++k){
          data.bx(i, j, k) = mag;
        }
      }
    }
    for (SAMS::T_indexType i = 1; i <= data.nx; ++i){
      for (SAMS::T_indexType j = 0; j <= data.ny; ++j){
        for (SAMS::T_indexType k = 1; k <= data.nz; ++k){
          data.by(i, j, k) = 0.0;
        }
      }
    }
    for (SAMS::T_indexType i = 1; i <= data.nx; ++i){
      for (SAMS::T_indexType j = 1; j <= data.ny; ++j){
        for (SAMS::T_indexType k = 0; k <= data.nz; ++k){
          data.bz(i, j, k) = 0.0;
        }
      }
    }
  }
    #include "./instantiate.h"
}//namespace LARE