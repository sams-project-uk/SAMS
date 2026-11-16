//#include "lareic.h"
//#include "lareBoundaryClass.h"
#include "shared_data.h"

namespace LARE{

    template<typename T_EOS>
    void LARE3D<T_EOS>::defaultValues(simulationData &data, SAMS::KVS &kvs){

       // Ratio of specific heat capacities
        data.gas_gamma = kvs.getReal("/LARE/gas_gamma", 1.4);

        // Average mass of an ion in proton masses
        data.mf = kvs.getReal("/LARE/mf", 1.2);

        // Remap kinetic energy correction off by default
        data.rke = kvs.getBool("/LARE/rke", false);

        data.dt_multiplier = kvs.getReal("/LARE/dt_multiplier", 0.8); // Default multiplier for time step

        // Geometry options: cartesian, cylindrical, spherical
        std::string geometry = kvs.getString("/LARE/geometry", "cartesian");
        if (geometry == "cartesian") {
            data.geometry = geometryType::Cartesian;
        } else if (geometry == "cylindrical") {
            data.geometry = geometryType::Cylindrical;
        } else if (geometry == "spherical") {
            data.geometry = geometryType::Spherical;
        } else {
            harness.abort("Invalid /LARE/geometry: " + geometry, false);
        }

        // Shock viscosity coefficients
        data.visc1 = kvs.getReal("/LARE/visc1", 0.1);
        data.visc2 = kvs.getReal("/LARE/visc2", 1.0);

        // Physical constants: SI or normalised units, each of which can be overridden
        std::string units = kvs.getString("/LARE/units", "SI");
        if (units == "SI") {
            data.mu0 = mu0_si;
            data.mh = mh_si;
            data.kb = kb_si;
        } else if (units == "normalised") {
            data.mu0 = 1.0;
            data.mh = 1.0;
            data.kb = 1.0;
        } else {
            harness.abort("Invalid /LARE/units: " + units, false);
        }
        data.mu0 = kvs.getReal("/LARE/mu0", data.mu0);
        data.mh = kvs.getReal("/LARE/mh", data.mh);
        data.kb = kvs.getReal("/LARE/kb", data.kb);

        // Maximum number of steps; if nsteps < 0, run until t_end
        data.nsteps = kvs.getInt("/time/nsteps", -1);

        // Grid stretching
        data.x_stretch = kvs.getBool("/LARE/x_stretch", false);
        data.y_stretch = kvs.getBool("/LARE/y_stretch", false);
        data.z_stretch = kvs.getBool("/LARE/z_stretch", false);

        // Resistive MHD options
        data.resistiveMHD = kvs.getBool("/LARE/resistiveMHD", false);
        data.eta_background = kvs.getReal("/LARE/eta_background", 0.0);
        data.j_max = kvs.getReal("/LARE/j_max", data.largest_number);
        data.eta0 = kvs.getReal("/LARE/eta0", 0.0);

        // Initialize time=0
        data.time = kvs.getReal("/LARE/time", 0.0);
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::repopulateDeck(simulationData &data, SAMS::KVS &kvs){

        kvs.setContentPath("/LARE/dt_multiplier", data.dt_multiplier);

        // Geometry
        if (data.geometry == geometryType::Cartesian) {
            kvs.setContentPath("/LARE/geometry", "cartesian");
        } else if (data.geometry == geometryType::Cylindrical) {
            kvs.setContentPath("/LARE/geometry", "cylindrical");
        } else if (data.geometry == geometryType::Spherical) {
            kvs.setContentPath("/LARE/geometry", "spherical");
        }

        // Shock viscosity coefficients
        kvs.setContentPath("/LARE/visc1", data.visc1);
        kvs.setContentPath("/LARE/visc2", data.visc2);

        kvs.setContentPath("/LARE/gas_gamma", data.gas_gamma);
        kvs.setContentPath("/LARE/mf", data.mf);
        kvs.setContentPath("/LARE/rke", data.rke);

        // Physical constants
        kvs.setContentPath("/LARE/mu0", data.mu0);
        kvs.setContentPath("/LARE/mh", data.mh);
        kvs.setContentPath("/LARE/kb", data.kb);

        kvs.setContentPath("/time/nsteps", data.nsteps);

        // Grid stretching
        kvs.setContentPath("/LARE/x_stretch", data.x_stretch);
        kvs.setContentPath("/LARE/y_stretch", data.y_stretch);
        kvs.setContentPath("/LARE/z_stretch", data.z_stretch);

        // Resistive MHD options
        kvs.setContentPath("/LARE/resistiveMHD", data.resistiveMHD);
        kvs.setContentPath("/LARE/eta_background", data.eta_background);
        kvs.setContentPath("/LARE/j_max", data.j_max);
        kvs.setContentPath("/LARE/eta0", data.eta0);
    }

template<typename T_EOS>
 void LARE3D<T_EOS>::defaultVariables([[maybe_unused]] simulationData &data)
  {

    // Set default variable values
    pw::assign(data.vx, 0.0);
    pw::assign(data.vy, 0.0);
    pw::assign(data.vz, 0.0);

    pw::assign(data.rho, 0.0);
    pw::assign(data.energy_electron, 0.0);
    pw::assign(data.energy_ion, 0.0);

    pw::assign(data.bx, 0.0);
    pw::assign(data.by, 0.0);
    pw::assign(data.bz, 0.0);

    if (data.rke)
    {
        pw::assign(data.delta_ke, 0.0);
    }

  }
    #include "./instantiate.h"
}//namespace LARE