#include "EmeryWindTunnel.h"

namespace examples
{

        /**
         * Attach peridoic boundary conditions in Z
         */
        void EmeryWindTunnel::setZbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(2, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set inflow X boundary conditions, zero gradient outflow X boundary conditions
         */
        void EmeryWindTunnel::setInflowXbcs( SAMS::variableDef &varDef, SAMS::T_dataType clampValue)
        {
            //Lower X boundary is just clamped to the inflow value
            varDef.addBoundaryCondition(0, SAMS::domain::edges::lower, SAMS::simpleClamp<SAMS::T_dataType, 3>(varDef, clampValue));

            //Upper X boundary is zero gradient
            varDef.addBoundaryCondition(0, SAMS::domain::edges::upper, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set zero gradient X boundary conditions
         */
        void EmeryWindTunnel::setZeroGradientXbcs( SAMS::variableDef &varDef)
        {
            //Both X boundaries are zero gradient
            varDef.addBoundaryCondition(0, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set Y mirror boundary conditions
         */
        void EmeryWindTunnel::setMirrorYbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(1, SAMS::simpleMirrorBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set zero gradient Y boundary conditions
         */
        void EmeryWindTunnel::setZeroGradientYbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(1, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set scalar boundary conditions
         */
        void EmeryWindTunnel::setScalarBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue)
        {
            //Simple clamped scalars are y mirror, Z periodic, xMin clamped to inflow value, xMax zero gradient
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setInflowXbcs(varDef, clampValue);
            setMirrorYbcs(varDef);
            setZbcs(varDef);
        }

        /**
         * Set scalar boundary conditions
         */
        void EmeryWindTunnel::setScalarBcs(std::string varName, SAMS::harness &harness)
        {
            //Simple clamped scalars are y mirror, Z periodic, xMin clamped to inflow value, xMax zero gradient
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZeroGradientXbcs(varDef);
            setMirrorYbcs(varDef);
            setZbcs(varDef);
        }


        /**
         * Set y parallel vector boundary conditions (zero gradient in y, periodic in z, inflow clamped at xMin, zero gradient at xMax)
         */
        void EmeryWindTunnel::setYParallelVectorBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setInflowXbcs(varDef, clampValue);
            setZeroGradientYbcs(varDef);
            setZbcs(varDef);
        }

        /**
         * Set y parallel vector boundary conditions (zero gradient in y, periodic in z, inflow clamped at xMin, zero gradient at xMax)
         */
        void EmeryWindTunnel::setYParallelVectorBcs(std::string varName, SAMS::harness &harness)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZeroGradientXbcs(varDef);
            setZeroGradientYbcs(varDef);
            setZbcs(varDef);
        }
            

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        void EmeryWindTunnel::controlVariables(LARE::simulationData &data, emeryParams &problemParams)
        {

            data.t_end = 3.6; // End time of the simulation
            data.dt_snapshots = data.t_end/100;
            data.nx = 480*2;
            data.ny = 160*2;
            data.nz = 2;

            data.x_min = 0.0;
            data.x_max = 3.0;
            data.y_min = 0.0;
            data.y_max = 1.0;
            data.z_min = 0.0;
            data.z_max = (data.x_max - data.x_min) * data.nz / data.nx;

            data.dt_multiplier = 0.8; // Default multiplier for time step
            // Geometry options: cartesian, cylindrical, spherical
            data.geometry = LARE::geometryType::Cartesian;
            // Shock viscosity coefficients
            data.visc1 = 0.0;
            data.visc2 = 0.0;

            // Ratio of specific heat capacities
            data.gas_gamma = 1.4;

            // Average mass of an ion in proton masses
            data.mf = 1.2;

            //Run with normalised mu0
            data.mu0 = 1.0;

            // Resistive MHD options
            data.resistiveMHD = false;
            data.eta_background = 1.e-10;
            data.j_max = 1.0;
            data.eta0 = 2.e-10;

            // Remap kinetic energy correction
            data.rke = true;

            //Set up the specific parameters for the emery wind tunnel
            problemParams.ambPressure = 1.0;
            problemParams.density = 1.4;
            problemParams.flowVx = 1.0;
        }

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param data LARE3D simulation data
         */
        void EmeryWindTunnel::setDomain(SAMS::harness &harness, LARE::simulationData &data) 
        {
            auto &axisReg = harness.axisRegistry;
            //Just hard code the domain for the Sod Shock Tube
            axisReg.setDomain("X", data.nx, data.x_min, data.x_max);
            axisReg.setDomain("Y", data.ny, data.y_min, data.y_max);
            axisReg.setDomain("Z", data.nz, data.z_min, data.z_max);
        }

        /**
         * Set boundary conditions for the simulation
         * @param harness SAMS harness
         */
        void EmeryWindTunnel::setBoundaryConditions(SAMS::harness &harness, LARE::simulationData &data, emeryParams &problemParams)
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
        void EmeryWindTunnel::initialConditions(SAMS::harness &harnessRef, LARE::simulationData &data, emeryParams &problemParams)
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
                        vx(ix,iy,iz) = 0.0;
                    }

                    //Specific internal energy
                    energy_electron(ix, iy, iz) = ambPressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;
                    energy_ion(ix, iy, iz) = ambPressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;

                },
                rho.getRange(0), rho.getRange(1), rho.getRange(2));
        }

        void EmeryWindTunnel::startOfTimestep(LARE::simulationData &data, emeryParams &problemParams){
            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    if (data.xb(ix) >= problemParams.step_x && data.yb(iy) <= problemParams.step_height)
                    {
                        data.vx1(ix,iy,iz) = 0.0;
                        data.vy1(ix,iy,iz) = 0.0;
                        data.vz1(ix,iy,iz) = 0.0;
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

        void EmeryWindTunnel::halfTimestep(LARE::simulationData &data, emeryParams &problemParams){
             pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    if (data.xb(ix) >= problemParams.step_x && data.yb(iy) <= problemParams.step_height)
                    {
                        data.vx(ix,iy,iz) = 0.0;
                        data.vy(ix,iy,iz) = 0.0;
                        data.vz(ix,iy,iz) = 0.0;
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

        void EmeryWindTunnel::endOfTimestep(LARE::simulationData &data, emeryParams &problemParams){
            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    if (data.xb(ix) >= problemParams.step_x && data.yb(iy) <= problemParams.step_height)
                    {
                        data.vx(ix,iy,iz) = 0.0;
                        data.vy(ix,iy,iz) = 0.0;
                        data.vz(ix,iy,iz) = 0.0;
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

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        void EmeryWindTunnel::queryTerminate(bool &terminate, LARE::simulationData &data, SAMS::timeState &tData){
            //End at correct time for Sod Shock Tube (ends at t=0.2)
            if (tData.time >= data.t_end){
                terminate |= true;
            }
        }

        /**
         * Check whether to output data to disk
         * @param shouldOutput Boolean flag to set to true to output data
         * @param data LARE3D simulation data
         * This function checks whether data should be output to disk based on LARE3D data.
         * It returns true if data should be output.
         */
        void EmeryWindTunnel::queryOutput(bool &shouldOutput, LARE::simulationData &data, SAMS::timeState &tData){
            static double nextOutputTime = data.dt_snapshots;
            if (tData.time >= (nextOutputTime) || (tData.time == 0.0)){
                shouldOutput |= true;
                nextOutputTime += data.dt_snapshots;
            }
        }
    }
