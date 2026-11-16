#include "EmeryWindTunnel.h"
#include "uniformDomain.h"
#include "../../Operations/OutputControl/defaultSchedule.h"

namespace examples
{

        /**
         * Attach peridoic boundary conditions in Z
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setZbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(2, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set inflow X boundary conditions, zero gradient outflow X boundary conditions
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setInflowXbcs( SAMS::variableDef &varDef, SAMS::T_dataType clampValue)
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
        void EmeryWindTunnel<T_EOS>::setZeroGradientXbcs( SAMS::variableDef &varDef)
        {
            //Both X boundaries are zero gradient
            varDef.addBoundaryCondition(0, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set Y reflecting wall boundary conditions for the velocity normal to the walls, which is zero on them
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setMirrorYbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(1, SAMS::fixedMirrorBC<SAMS::T_dataType, 3>(varDef, 0.0));
        }

        /**
         * Set scalar boundary conditions
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setScalarBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue)
        {
            //Clamped scalars are reflecting in Y, periodic in Z, clamped to the inflow value at xMin and zero gradient at xMax
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setInflowXbcs(varDef, clampValue);
            //The velocity normal to the Y walls is antisymmetric, and everything else is symmetric
            if (varName == "vy" || varName == "LARE/vy1") setMirrorYbcs(varDef);
            else varDef.addBoundaryCondition(1, SAMS::simpleSymmetricBC<SAMS::T_dataType, 3>(varDef));
            setZbcs(varDef);
        }

        /**
         * Set scalar boundary conditions
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setScalarBcs(std::string varName, SAMS::harness &harness)
        {
            //Scalars are reflecting in Y, periodic in Z and zero gradient in X
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZeroGradientXbcs(varDef);
            //The velocity normal to the Y walls is antisymmetric, and everything else is symmetric
            if (varName == "vy" || varName == "LARE/vy1") setMirrorYbcs(varDef);
            else varDef.addBoundaryCondition(1, SAMS::simpleSymmetricBC<SAMS::T_dataType, 3>(varDef));
            setZbcs(varDef);
        }


        /**
         * Set boundary conditions for components parallel to the Y walls (symmetric in y, periodic in z, inflow clamped at xMin, zero gradient at xMax)
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setYParallelVectorBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setInflowXbcs(varDef, clampValue);
            //Symmetric at the Y walls
            varDef.addBoundaryCondition(1, SAMS::simpleSymmetricBC<SAMS::T_dataType, 3>(varDef));
            setZbcs(varDef);
        }

        /**
         * Set boundary conditions for components parallel to the Y walls (symmetric in y, periodic in z, zero gradient in x)
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setYParallelVectorBcs(std::string varName, SAMS::harness &harness)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZeroGradientXbcs(varDef);
            //Symmetric at the Y walls
            varDef.addBoundaryCondition(1, SAMS::simpleSymmetricBC<SAMS::T_dataType, 3>(varDef));
            setZbcs(varDef);
        }
            

        /**
         * Set the deck keys that the deck does not set
         * @param kvs The key value store
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::defaultDeck(SAMS::KVS &kvs)
        {
            //Run with normalised units
            kvs.setDefaultPath("/LARE/units", "normalised");
            //Periodic in Z
            kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{false, false, true});
            //End time of the simulation
            kvs.setDefaultPath(SAMS::deckKeys::endTime, 3.6);
            //Domain. Z has cells of the same size as X
            kvs.setDefaultPath(SAMS::deckKeys::nx, 480);
            kvs.setDefaultPath(SAMS::deckKeys::ny, 160);
            kvs.setDefaultPath(SAMS::deckKeys::nz, 2);
            kvs.setDefaultPath(SAMS::deckKeys::xMin, 0.0);
            kvs.setDefaultPath(SAMS::deckKeys::xMax, 3.0);
            kvs.setDefaultPath(SAMS::deckKeys::yMin, 0.0);
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
        void EmeryWindTunnel<T_EOS>::controlVariables(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams, SAMS::KVS &kvs)
        {

            data.t_end = kvs.getReal(SAMS::deckKeys::endTime, 3.6); // End time of the simulation

            data.dt_multiplier = kvs.getReal("/LARE/dt_multiplier", 0.8); // Default multiplier for time step
            // Shock viscosity coefficients
            data.visc1 = kvs.getReal("/LARE/visc1", 1.0);
            data.visc2 = kvs.getReal("/LARE/visc2", 0.1);

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
            data.rke = kvs.getBool("/LARE/rke", true);

            //Set up the specific parameters for the emery wind tunnel
            problemParams.ambPressure = kvs.getReal("/EmeryWindTunnel/amb_pressure", 1.0);
            problemParams.density = kvs.getReal("/EmeryWindTunnel/density", 1.4);
            problemParams.flowVx = kvs.getReal("/EmeryWindTunnel/flow_vx", 1.0);
        }

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param kvs The key value store
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
        {
            SAMS::setUniformDomain(harness, kvs);
        }

        /**
         * Set boundary conditions for the simulation
         * @param harness SAMS harness
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::setBoundaryConditions(SAMS::harness &harness, LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams)
        {
            SAMS::T_dataType pressure = problemParams.ambPressure;
            SAMS::T_dataType density = problemParams.density;
            SAMS::T_dataType flowVx= problemParams.flowVx;
            //Set driven boundary conditions at lower X boundary, and zero gradient outflow boundary conditions at upper X boundary. Periodic boundaries in Y and Z
            setScalarBcs("energy_ion", harness, 0.5*pressure/(density*(data.gas_gamma - 1.0)));
            setScalarBcs("energy_electron", harness, 0.5*pressure/(density*(data.gas_gamma - 1.0)));
            setScalarBcs("rho", harness, density);
            setYParallelVectorBcs("vx", harness, flowVx);
            setYParallelVectorBcs("LARE/vx1", harness, flowVx);

            
            //For other variables, just set zero gradient boundaries everywhere
            setYParallelVectorBcs("bx", harness);
            setScalarBcs("by", harness);
            setYParallelVectorBcs("bz", harness);
            setScalarBcs("vy", harness);
            setYParallelVectorBcs("vz", harness);
            setScalarBcs("LARE/vy1", harness);
            setYParallelVectorBcs("LARE/vz1", harness);
            setScalarBcs("LARE/dm", harness);
        }

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::initialConditions(SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams)
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
                    rho(ix,iy,iz) = density;
                    vx(ix,iy,iz) = flowVx;
                    vy(ix,iy,iz) = 0.0;
                    vz(ix,iy,iz) = 0.0;
                    bx(ix,iy,iz) = 0.0;
                    by(ix,iy,iz) = 0.0;
                    bz(ix,iy,iz) = 0.0;

                    if (data.xb(ix) >= problemParams.step_x && data.yb(iy) <= problemParams.step_height)
                    {
                        //The velocity along the top face of the step is free, so the flow slips along it
                        const bool topFace = data.yb(iy+1) > problemParams.step_height;
                        const bool frontFace = data.xb(ix-1) < problemParams.step_x;
                        if (frontFace || !topFace) vx(ix,iy,iz) = 0.0;
                    }

                    //Specific internal energy
                    energy_electron(ix, iy, iz) = ambPressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;
                    energy_ion(ix, iy, iz) = ambPressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;

                },
                rho.getRange(0), rho.getRange(1), rho.getRange(2));
        }

        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::startOfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams){
            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    if (data.xb(ix) >= problemParams.step_x && data.yb(iy) <= problemParams.step_height)
                    {
                        //Inside the step every component is zero. On a face only the component normal to it is, so the flow slips along the face
                        const bool topFace = data.yb(iy+1) > problemParams.step_height;
                        const bool frontFace = data.xb(ix-1) < problemParams.step_x;
                        if (frontFace || !topFace) data.vx1(ix,iy,iz) = 0.0;
                        if (topFace || !frontFace) data.vy1(ix,iy,iz) = 0.0;
                        if (!topFace && !frontFace) data.vz1(ix,iy,iz) = 0.0;
                    }
                    if (data.xc(ix) >= problemParams.step_x && data.yc(iy) <= problemParams.step_height)
                    {
                        data.rho(ix,iy,iz) = problemParams.density;
                        data.energy_electron(ix,iy,iz) = problemParams.ambPressure / ((data.gas_gamma - 1.0) * data.rho(ix,iy,iz))/2.0;
                        data.energy_ion(ix,iy,iz) = problemParams.ambPressure / ((data.gas_gamma - 1.0) * data.rho(ix,iy,iz))/2.0;
                    }
                },
                pw::Range(0,data.nx), pw::Range(0,data.ny), pw::Range(0,data.nz));
        }

        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::halfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams){
             pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    if (data.xb(ix) >= problemParams.step_x && data.yb(iy) <= problemParams.step_height)
                    {
                        //Inside the step every component is zero. On a face only the component normal to it is, so the flow slips along the face
                        const bool topFace = data.yb(iy+1) > problemParams.step_height;
                        const bool frontFace = data.xb(ix-1) < problemParams.step_x;
                        if (frontFace || !topFace) data.vx(ix,iy,iz) = 0.0;
                        if (topFace || !frontFace) data.vy(ix,iy,iz) = 0.0;
                        if (!topFace && !frontFace) data.vz(ix,iy,iz) = 0.0;
                    }
                    if (data.xc(ix) >= problemParams.step_x && data.yc(iy) <= problemParams.step_height)
                    {
                        data.rho(ix,iy,iz) = problemParams.density;
                        data.energy_electron(ix,iy,iz) = problemParams.ambPressure / ((data.gas_gamma - 1.0) * data.rho(ix,iy,iz))/2.0;
                        data.energy_ion(ix,iy,iz) = problemParams.ambPressure / ((data.gas_gamma - 1.0) * data.rho(ix,iy,iz))/2.0;
                    }
                },
                pw::Range(0,data.nx), pw::Range(0,data.ny), pw::Range(0,data.nz));
        }

        template<typename T_EOS>
        void EmeryWindTunnel<T_EOS>::endOfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams){
            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    if (data.xb(ix) >= problemParams.step_x && data.yb(iy) <= problemParams.step_height)
                    {
                        //Inside the step every component is zero. On a face only the component normal to it is, so the flow slips along the face
                        const bool topFace = data.yb(iy+1) > problemParams.step_height;
                        const bool frontFace = data.xb(ix-1) < problemParams.step_x;
                        if (frontFace || !topFace) data.vx(ix,iy,iz) = 0.0;
                        if (topFace || !frontFace) data.vy(ix,iy,iz) = 0.0;
                        if (!topFace && !frontFace) data.vz(ix,iy,iz) = 0.0;
                    }
                    if (data.xc(ix) >= problemParams.step_x && data.yc(iy) <= problemParams.step_height)
                    {
                        data.rho(ix,iy,iz) = problemParams.density;
                        data.energy_electron(ix,iy,iz) = problemParams.ambPressure / ((data.gas_gamma - 1.0) * data.rho(ix,iy,iz))/2.0;
                        data.energy_ion(ix,iy,iz) = problemParams.ambPressure / ((data.gas_gamma - 1.0) * data.rho(ix,iy,iz))/2.0;
                    }
                },
                pw::Range(0,data.nx), pw::Range(0,data.ny), pw::Range(0,data.nz));
        }

    #define EOS_DEF(value) template class EmeryWindTunnel<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF
    }
