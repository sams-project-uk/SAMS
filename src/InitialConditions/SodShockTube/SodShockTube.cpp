
#include "SodShockTube.h"
#include "../../Operations/OutputControl/defaultSchedule.h"
#include "uniformDomain.h"

namespace examples
{

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        template<typename T_EOS>
        void SodShockTube<T_EOS>::attachBoundaryConditions(const std::string& varName, SAMS::harness &harness)
        {
            SAMS::variableDef &varDef = harness.variableRegistry.getVariable(varName);
            //Periodic boundaries in Y and Z
            std::shared_ptr<SAMS::boundaryConditions> periodicBC = std::make_shared<SAMS::simplePeriodicBC<SAMS::T_dataType, 3>>(varDef);
            varDef.addBoundaryCondition(1, periodicBC);
            varDef.addBoundaryCondition(2, periodicBC);

            //Reflecting walls in X. The velocity normal to the walls is antisymmetric, and everything else is symmetric
            if (varName == "vx" || varName == "LARE/vx1") {
                varDef.addBoundaryCondition(0, std::make_shared<SAMS::fixedMirrorBC<SAMS::T_dataType, 3>>(varDef, 0.0));
            } else {
                varDef.addBoundaryCondition(0, std::make_shared<SAMS::simpleSymmetricBC<SAMS::T_dataType, 3>>(varDef));
            }
        }

        /**
         * Set the deck keys that the deck does not set
         * @param kvs The key value store
         */
        template<typename T_EOS>
        void SodShockTube<T_EOS>::defaultDeck(SAMS::KVS &kvs)
        {
            //Run with normalised units
            kvs.setDefaultPath("/LARE/units", "normalised");
            //Periodic in Y and Z
            kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{false, true, true});
            //End time of the simulation
            kvs.setDefaultPath(SAMS::deckKeys::endTime, 0.2);
            //Domain. Y and Z have cells of the same size as X
            kvs.setDefaultPath(SAMS::deckKeys::nx, 1024);
            kvs.setDefaultPath(SAMS::deckKeys::ny, 2);
            kvs.setDefaultPath(SAMS::deckKeys::nz, 2);
            kvs.setDefaultPath(SAMS::deckKeys::xMin, 0.0);
            kvs.setDefaultPath(SAMS::deckKeys::xMax, 1.0);
            kvs.setDefaultPath(SAMS::deckKeys::yMin, 0.0);
            kvs.setDefaultPath(SAMS::deckKeys::zMin, 0.0);
            const SAMS::T_dataType width = kvs.getReal(SAMS::deckKeys::xMax) - kvs.getReal(SAMS::deckKeys::xMin);
            kvs.setDefaultPath(SAMS::deckKeys::yMax, width * kvs.getInt(SAMS::deckKeys::ny) / kvs.getInt(SAMS::deckKeys::nx));
            kvs.setDefaultPath(SAMS::deckKeys::zMax, width * kvs.getInt(SAMS::deckKeys::nz) / kvs.getInt(SAMS::deckKeys::nx));
            //Write the default collection 10 times over the run if the deck has no output schedules
            SAMS::addDefaultSchedule(kvs, kvs.getReal(SAMS::deckKeys::endTime) / 10.0);
        }

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        template<typename T_EOS>
        void SodShockTube<T_EOS>::controlVariables(LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs)
        {

            data.t_end = kvs.getReal(SAMS::deckKeys::endTime, 0.2);

            data.dt_multiplier = kvs.getReal("/LARE/dt_multiplier", 0.8); // Default multiplier for time step
            // Shock viscosity coefficients
            data.visc1 = kvs.getReal("/LARE/visc1", 0.1);
            data.visc2 = kvs.getReal("/LARE/visc2", 1.0);

            // Ratio of specific heat capacities
            data.gas_gamma = kvs.getReal("/LARE/gas_gamma", 1.4);

            // Average mass of an ion in proton masses
            data.mf = kvs.getReal("/LARE/mf", 1.2);

            // Resistive MHD options
            data.resistiveMHD = kvs.getBool("/LARE/resistiveMHD", false);
            data.eta_background = kvs.getReal("/LARE/eta_background", 1.e-10);
            data.j_max = kvs.getReal("/LARE/j_max", 1.0);
            data.eta0 = kvs.getReal("/LARE/eta0", 2.e-10);

            // Remap kinetic energy correction
            data.rke = kvs.getBool("/LARE/rke", false);
        }

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param kvs The key value store
         */
        template<typename T_EOS>
        void SodShockTube<T_EOS>::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
        {
            SAMS::setUniformDomain(harness, kvs);
        }

        /**
         * Set boundary conditions for the simulation
         * @param harness SAMS harness
         */
        template<typename T_EOS>
        void SodShockTube<T_EOS>::setBoundaryConditions(SAMS::harness &harness)
        {
            //Grab variables and set boundary condition functions
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
        void SodShockTube<T_EOS>::initialConditions(SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs)
        {
            pw::portableArray<SAMS::T_dataType, 3> rho;
            pw::portableArray<SAMS::T_dataType, 3> energy_electron;
            pw::portableArray<SAMS::T_dataType, 1> xc, yc, zc;

            auto &axisRegistry = harnessRef.axisRegistry;
            axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
            axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);
            axisRegistry.fillPPLocalAxis("Z", zc, SAMS::staggerType::CENTRED);

            auto &varRegistry = harnessRef.variableRegistry;
            varRegistry.fillPPArray("rho", rho);
            varRegistry.fillPPArray("energy_electron", energy_electron);

            SAMS::T_dataType interface = kvs.getReal("/SodShockTube/interface", 0.5);
            SAMS::T_dataType rho_left = kvs.getReal("/SodShockTube/rho_left", 1.0);
            SAMS::T_dataType pressure_left = kvs.getReal("/SodShockTube/pressure_left", 1.0);
            SAMS::T_dataType rho_right = kvs.getReal("/SodShockTube/rho_right", 0.125);
            SAMS::T_dataType pressure_right = kvs.getReal("/SodShockTube/pressure_right", 0.1);

            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    SAMS::T_dataType pressure;
                    if (xc(ix) < interface)
                    {
                        rho(ix, iy, iz) = rho_left;
                        pressure = pressure_left;
                    }
                    else
                    {
                        rho(ix, iy, iz) = rho_right;
                        pressure = pressure_right;
                    }
                    //Specific internal energy
                    energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
                },
                rho.getRange(0), rho.getRange(1), rho.getRange(2));
        }

    #define EOS_DEF(value) template class SodShockTube<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF
}
