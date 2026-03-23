//#include "lareic.h"
//#include "lareBoundaryClass.h"
#include "LARE3DNeutralFluid/shared_data.h"

namespace LARE{

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::defaultValues(simulationData &data){

        // Shock viscosity coefficients
        data.visc1 = 0.1;
        data.visc2 = 1.0;

        // Ratio of specific heat capacities
        data.gas_gamma = 1.4;

        data.rke = false; // Remap kinetic energy correction off by default
    }

template<typename T_EOS>
 void LARE3DNF<T_EOS>::defaultVariables([[maybe_unused]] simulationData &data)
  {

    // Set default variable values
    pw::assign(data.vx, 0.0);
    pw::assign(data.vy, 0.0);
    pw::assign(data.vz, 0.0);

    pw::assign(data.rho, 0.0);
    pw::assign(data.energy, 0.0);

    if (data.rke)
    {
        pw::assign(data.delta_ke, 0.0);
    }

  }

}//namespace LARE