#include "lareic.h"
#include "lareic_neutral.h"
#include "lareBoundaryClass.h"
#include "lareBoundaryClass_neutral.h"

namespace LARE_neutral{

    void LARE3D_neutral::defaultValues([[maybe_unused]] simulationData &data){

        data.dt_multiplier = 0.8; // Default multiplier for time step

        // Geometry options: cartesian, cylindrical, spherical
        data.geometry = geometryType::Cartesian;

        // Shock viscosity coefficients
        data.visc1 = 0.1;
        data.visc2 = 1.0;

        // Ratio of specific heat capacities
        data.gas_gamma = 1.4;

        // Average mass of an ion in proton masses
        data.mf = 1.2;

        data.rke = false; // Remap kinetic energy correction off by default

        // Physical constants
        data.mu0 = LARE::mu0_si;

        // Initialize time=0
        data.time = 0.0;
    }

 void LARE3D_neutral::defaultVariables([[maybe_unused]] simulationData &data)
  {

    // Set default variable values
    pw::assign(data.vx, 0.0);
    pw::assign(data.vy, 0.0);
    pw::assign(data.vz, 0.0);

    pw::assign(data.rho, 0.0);
    pw::assign(data.energy_neutral, 0.0);

    pw::assign(data.bx, 0.0);
    pw::assign(data.by, 0.0);
    pw::assign(data.bz, 0.0);

    if (data.rke)
    {
        pw::assign(data.delta_ke, 0.0);
    }

  }

}//namespace LARE_neutral
