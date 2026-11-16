#include "KarmanVortex.h"
#include "../../Operations/OutputControl/defaultSchedule.h"
#include "uniformDomain.h"

namespace examples
{

        /**
         * Attach peridoic boundary conditions in Z
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setZbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(2, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Periodic in y
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setYbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(1, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set inflow X boundary conditions, zero gradient outflow X boundary conditions
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setInflowXbcs( SAMS::variableDef &varDef, SAMS::T_dataType clampValue)
        {
            //Lower X boundary is just clamped to the inflow value
            varDef.addBoundaryCondition(0, SAMS::domain::edges::lower, SAMS::simpleClamp<SAMS::T_dataType, 3>(varDef, clampValue));

            //Upper X boundary is zero gradient
            varDef.addBoundaryCondition(0, SAMS::domain::edges::upper, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set zero gradient X boundary conditions
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setZeroGradientXbcs( SAMS::variableDef &varDef)
        {
            //Both X boundaries are zero gradient
            varDef.addBoundaryCondition(0, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setBCS(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZbcs(varDef);
            setYbcs(varDef);
            setInflowXbcs(varDef, clampValue);
        }

        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setBCS(std::string varName, SAMS::harness &harness)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZbcs(varDef);
            setYbcs(varDef);
            setZeroGradientXbcs(varDef);
        }

        /**
         * Set the deck keys that the deck does not set
         * @param kvs The key value store
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::defaultDeck(SAMS::KVS &kvs)
        {
            //Run with normalised units
            kvs.setDefaultPath("/LARE/units", "normalised");
            //Periodic in Y and Z
            kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{false, true, true});
            //End time of the simulation
            kvs.setDefaultPath(SAMS::deckKeys::endTime, 20.0);
            //Domain. Z has cells of the same size as X
            kvs.setDefaultPath(SAMS::deckKeys::nx, 512);
            kvs.setDefaultPath(SAMS::deckKeys::ny, 256);
            kvs.setDefaultPath(SAMS::deckKeys::nz, 2);
            kvs.setDefaultPath(SAMS::deckKeys::xMin, -1.0);
            kvs.setDefaultPath(SAMS::deckKeys::xMax, 3.0);
            kvs.setDefaultPath(SAMS::deckKeys::yMin, -1.0);
            kvs.setDefaultPath(SAMS::deckKeys::yMax, 1.0);
            kvs.setDefaultPath(SAMS::deckKeys::zMin, 0.0);
            const SAMS::T_dataType width = kvs.getReal(SAMS::deckKeys::xMax) - kvs.getReal(SAMS::deckKeys::xMin);
            kvs.setDefaultPath(SAMS::deckKeys::zMax, width * kvs.getInt(SAMS::deckKeys::nz) / kvs.getInt(SAMS::deckKeys::nx));
            //Write the default collection 100 times over the run if the deck has no output schedules
            SAMS::addDefaultSchedule(kvs, kvs.getReal(SAMS::deckKeys::endTime) / 100.0);
        }

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::controlVariables(LARE::LARE3D<T_EOS>::simulationData &data, KarmanVortexParams &problemParams, SAMS::KVS &kvs)
        {

            data.t_end = kvs.getReal(SAMS::deckKeys::endTime, 20); // End time of the simulation

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

            //Set up the specific parameters for the Karman Vortex problem
            //Note that this is just a Karman vortex street type problem, but we are just calling it KarmanVortex for simplicity. We can add more complex vortex shedding problems later if needed.
            //Mach =0.5 flow, cylinder radius = 0.1, ambient pressure = 1.0, density = 1.0
            //Note that there is no explicit viscosity in LARE, so we cannot set a Reynolds number, but with the shock viscosity coefficients above we get a reasonable vortex shedding pattern for these parameters.
            problemParams.ambPressure = kvs.getReal("/KarmanVortex/amb_pressure", 1.0);
            problemParams.density = kvs.getReal("/KarmanVortex/density", 1.0);
            problemParams.flowVx = kvs.getReal("/KarmanVortex/flow_vx", 0.5);
            problemParams.cylinderRadius = kvs.getReal("/KarmanVortex/cylinder_radius", 0.1);
        }

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param kvs The key value store
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
        {
            SAMS::setUniformDomain(harness, kvs);
        }

        /**
         * Set boundary conditions for the simulation
         * @param harness SAMS harness
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::setBoundaryConditions(SAMS::harness &harness, LARE::LARE3D<T_EOS>::simulationData &data, KarmanVortexParams &problemParams)
        {
            SAMS::T_dataType pressure = problemParams.ambPressure;
            SAMS::T_dataType density = problemParams.density;
            SAMS::T_dataType flowVx= problemParams.flowVx;
            //Set driven boundary conditions at lower X boundary, and zero gradient outflow boundary conditions at upper X boundary. Periodic boundaries in Y and Z
            setBCS("energy_ion", harness, 0.5*pressure/(density*(data.gas_gamma - 1.0)));
            setBCS("energy_electron", harness, 0.5*pressure/(density*(data.gas_gamma - 1.0)));
            setBCS("rho", harness, density);
            setBCS("vx", harness, flowVx);
            setBCS("LARE/vx1", harness, flowVx);

            
            //For other variables, just set zero gradient boundaries everywhere
            setBCS("bx", harness);
            setBCS("by", harness);
            setBCS("bz", harness);
            setBCS("vy", harness);
            setBCS("vz", harness);
            setBCS("LARE/vy1", harness);
            setBCS("LARE/vz1", harness);
            setBCS("LARE/dm", harness);
        }

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        template<typename T_EOS>
        void KarmanVortex<T_EOS>::initialConditions(SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, KarmanVortexParams &problemParams)
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

            SAMS::T_dataType ambPressure = problemParams.ambPressure;
            SAMS::T_dataType density = problemParams.density;
            SAMS::T_dataType flowVx=problemParams.flowVx;

            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    using T_dataType = SAMS::T_dataType;
                    rho(ix,iy,iz) = density;
                    vx(ix,iy,iz) = flowVx;
                    vy(ix,iy,iz) = 0.0;
                    vz(ix,iy,iz) = 0.0;
                    bx(ix,iy,iz) = 0.0;
                    by(ix,iy,iz) = 0.0;
                    bz(ix,iy,iz) = 0.0;

                    SAMS::T_dataType r=std::sqrt(data.xc(ix)*data.xc(ix) + data.yc(iy)*data.yc(iy));
                    if (r < problemParams.cylinderRadius)
                    {
                        data.vx(ix,iy,iz) = 0.0;
                        data.vy(ix,iy,iz) = 0.0;
                    }

                    T_dataType pressure = ambPressure;
 
                    //Specific internal energy
                    energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;
                    energy_ion(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;

                },
                rho.getRange(0), rho.getRange(1), rho.getRange(2));
        }

        template<typename T_EOS>
        void KarmanVortex<T_EOS>::startOfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, KarmanVortexParams &problemParams){
            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    SAMS::T_dataType r=std::sqrt(data.xc(ix)*data.xc(ix) + data.yc(iy)*data.yc(iy));
                    if (r < problemParams.cylinderRadius)
                    {
                        data.vx(ix,iy,iz) = 0.0;
                        data.vy(ix,iy,iz) = 0.0;
                    }

                },
                pw::Range(0,data.nx), pw::Range(0,data.ny), pw::Range(0,data.nz));
        }

        template<typename T_EOS>
        void KarmanVortex<T_EOS>::halfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, KarmanVortexParams &problemParams){
             pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    SAMS::T_dataType r=std::sqrt(data.xc(ix)*data.xc(ix) + data.yc(iy)*data.yc(iy));
                    if (r < problemParams.cylinderRadius)
                    {
                        data.vx(ix,iy,iz) = 0.0;
                        data.vy(ix,iy,iz) = 0.0;
                    }
                },
                pw::Range(0,data.nx), pw::Range(0,data.ny), pw::Range(0,data.nz));
        }

        template<typename T_EOS>
        void KarmanVortex<T_EOS>::endOfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, KarmanVortexParams &problemParams){
            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    SAMS::T_dataType r=std::sqrt(data.xc(ix)*data.xc(ix) + data.yc(iy)*data.yc(iy));
                    if (r < problemParams.cylinderRadius)
                    {
                        data.vx(ix,iy,iz) = 0.0;
                        data.vy(ix,iy,iz) = 0.0;
                    }
                },
                pw::Range(0,data.nx), pw::Range(0,data.ny), pw::Range(0,data.nz));
        }

    #define EOS_DEF(value) template class KarmanVortex<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF
    }