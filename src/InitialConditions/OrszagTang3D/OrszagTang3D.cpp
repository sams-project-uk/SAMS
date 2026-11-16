
#include "OrszagTang3D.h"
#include "../../Operations/OutputControl/defaultSchedule.h"
#include "uniformDomain.h"
#include "io/writers.h"
#include "io/readers.h"
#include <iomanip>
#include <limits>
#include <sstream>

namespace examples
{

    /**
     * Internal function to attach boundary conditions to a variable
     * @param varName Name of the variable to attach boundary conditions to
     * @param harness SAMS harness
     */
    template<typename T_EOS>
    void OrszagTang3D<T_EOS>::attachBoundaryConditions(const std::string &varName, SAMS::harness &harness)
    {
        SAMS::variableDef &varDef = harness.variableRegistry.getVariable(varName);
        // Periodic boundaries in all directions
        std::shared_ptr<SAMS::boundaryConditions> periodicBC = std::make_shared<SAMS::simplePeriodicBC<SAMS::T_dataType, 3>>(varDef);
        varDef.addBoundaryCondition(0, periodicBC);
        varDef.addBoundaryCondition(1, periodicBC);
        varDef.addBoundaryCondition(2, periodicBC);
    }

    /**
     * Set the deck keys that the deck does not set
     * @param kvs The key value store
     */
    template<typename T_EOS>
    void OrszagTang3D<T_EOS>::defaultDeck(SAMS::KVS &kvs)
    {
        // Run with normalised units
        kvs.setDefaultPath("/LARE/units", "normalised");
        // Periodic in all directions
        kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{true, true, true});
        // End time of the simulation
        kvs.setDefaultPath(SAMS::deckKeys::endTime, 3.0);
        // Domain
        kvs.setDefaultPath(SAMS::deckKeys::nx, 128);
        kvs.setDefaultPath(SAMS::deckKeys::ny, 128);
        kvs.setDefaultPath(SAMS::deckKeys::nz, 128);
        kvs.setDefaultPath(SAMS::deckKeys::xMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::xMax, 2.0 * M_PI);
        kvs.setDefaultPath(SAMS::deckKeys::yMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::yMax, 2.0 * M_PI);
        kvs.setDefaultPath(SAMS::deckKeys::zMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::zMax, 2.0 * M_PI);
        // Write the default collection 10 times over the run if the deck has no output schedules
        SAMS::addDefaultSchedule(kvs, kvs.getReal(SAMS::deckKeys::endTime) / 10.0);
    }

    /**
     * Set up the LARE style simulation control variables
     * @note This should later be moved to a more general approach
     * when we have multiple core solvers.
     * @param data LARE3D simulation data
     */
    template<typename T_EOS>
    void OrszagTang3D<T_EOS>::controlVariables(typename LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs)
    {

        data.t_end = kvs.getReal(SAMS::deckKeys::endTime, 3.0); // End time of the simulation

        data.dt_multiplier = kvs.getReal("/LARE/dt_multiplier", 0.8); // Default multiplier for time step
        // Shock viscosity coefficients
        data.visc1 = kvs.getReal("/LARE/visc1", 0.1);
        data.visc2 = kvs.getReal("/LARE/visc2", 1.0);

        // Ratio of specific heat capacities
        data.gas_gamma = kvs.getReal("/LARE/gas_gamma", 5.0 / 3.0);

        // Average mass of an ion in proton masses
        data.mf = kvs.getReal("/LARE/mf", 1.2);

        // Resistive MHD options
        data.resistiveMHD = kvs.getBool("/LARE/resistiveMHD", false);
        data.eta_background = kvs.getReal("/LARE/eta_background", 1.e-10);
        data.j_max = kvs.getReal("/LARE/j_max", 1.0);
        data.eta0 = kvs.getReal("/LARE/eta0", 2.e-10);

        // Remap kinetic energy correction
        data.rke = kvs.getBool("/LARE/rke", false);

        // Amplitude of the velocity perturbation that varies in z
        epsilon = kvs.getReal("/OrszagTang3D/epsilon", 0.2);
    }

    /**
     * Set up the simulation domain
     * @param harness SAMS harness
     * @param kvs The key value store
     */
    template<typename T_EOS>
    void OrszagTang3D<T_EOS>::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
    {
        SAMS::setUniformDomain(harness, kvs);
    }

    /**
     * Set boundary conditions for the simulation
     * @param harness SAMS harness
     */
    template<typename T_EOS>
    void OrszagTang3D<T_EOS>::setBoundaryConditions(SAMS::harness &harness)
    {
        // Grab variables and set boundary condition functions
        attachBoundaryConditions("bx", harness);
        attachBoundaryConditions("by", harness);
        attachBoundaryConditions("bz", harness);
        attachBoundaryConditions("energy_ion", harness);
        attachBoundaryConditions("energy_electron", harness);
        attachBoundaryConditions("rho", harness);
        attachBoundaryConditions("vx", harness);
        attachBoundaryConditions("vy", harness);
        attachBoundaryConditions("vz", harness);
        attachBoundaryConditions("LARE/vx1", harness);
        attachBoundaryConditions("LARE/vy1", harness);
        attachBoundaryConditions("LARE/vz1", harness);
        attachBoundaryConditions("LARE/dm", harness);
    }

    /**
     * Initialize the Sod Shock Tube initial conditions
     * @param harnessRef SAMS harness
     * @param data LARE3D simulation data
     */
    template<typename T_EOS>
    void OrszagTang3D<T_EOS>::initialConditions(SAMS::harness &harnessRef, typename LARE::LARE3D<T_EOS>::simulationData &data)
    {
        pw::portableArray<SAMS::T_dataType, 3> rho;
        pw::portableArray<SAMS::T_dataType, 3> energy_electron, energy_ion;
        pw::portableArray<SAMS::T_dataType, 3> vx, vy, vz;
        pw::portableArray<SAMS::T_dataType, 3> bx, by, bz;
        pw::portableArray<SAMS::T_dataType, 1> xc, yc, zc;
        pw::portableArray<SAMS::T_dataType, 1> xb, yb, zb;

        auto &axisRegistry = harnessRef.axisRegistry;
        axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
        axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);
        axisRegistry.fillPPLocalAxis("Z", zc, SAMS::staggerType::CENTRED);
        axisRegistry.fillPPLocalAxis("X", xb, SAMS::staggerType::HALF_CELL);
        axisRegistry.fillPPLocalAxis("Y", yb, SAMS::staggerType::HALF_CELL);
        axisRegistry.fillPPLocalAxis("Z", zb, SAMS::staggerType::HALF_CELL);

        auto &varRegistry = harnessRef.variableRegistry;
        varRegistry.fillPPArray("rho", rho);
        varRegistry.fillPPArray("energy_electron", energy_electron);
        varRegistry.fillPPArray("energy_ion", energy_ion);
        varRegistry.fillPPArray("vx", vx);
        varRegistry.fillPPArray("vy", vy);
        varRegistry.fillPPArray("vz", vz);
        varRegistry.fillPPArray("bx", bx);
        varRegistry.fillPPArray("by", by);
        varRegistry.fillPPArray("bz", bz);

        // Copied so that the kernel does not capture this
        const SAMS::T_dataType amplitude = epsilon;
        pw::applyKernel(
            LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz) {
                using T_dataType = SAMS::T_dataType;
                T_dataType pressure = data.gas_gamma;
                // The 2D Orszag-Tang vortex with a velocity perturbation that varies in z, so that the problem is genuinely three dimensional
                const T_dataType perturbation = amplitude * std::sin(zb(iz));

                rho(ix, iy, iz) = data.gas_gamma * data.gas_gamma;
                vx(ix, iy, iz) = -(1.0 + perturbation) * std::sin(yb(iy));
                vy(ix, iy, iz) = (1.0 + perturbation) * std::sin(xb(ix));
                vz(ix, iy, iz) = perturbation;
                bx(ix, iy, iz) = -std::sin(yc(iy));
                by(ix, iy, iz) = std::sin(2.0 * xc(ix));
                bz(ix, iy, iz) = 0.0;

                // Specific internal energy
                energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz)) / 2.0;
                energy_ion(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz)) / 2.0;
            },
            rho.getRange(0), rho.getRange(1), rho.getRange(2));
    }

    #define EOS_DEF(value) template class OrszagTang3D<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
