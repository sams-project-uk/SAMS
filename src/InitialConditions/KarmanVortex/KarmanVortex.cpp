#include "KarmanVortex.h"

namespace examples
{

        /**
         * Attach peridoic boundary conditions in Z
         */
        void KarmanVortex::setZbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(2, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Periodic in y
         */
        void KarmanVortex::setYbcs( SAMS::variableDef &varDef)
        {
            varDef.addBoundaryCondition(1, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set inflow X boundary conditions, zero gradient outflow X boundary conditions
         */
        void KarmanVortex::setInflowXbcs( SAMS::variableDef &varDef, SAMS::T_dataType clampValue)
        {
            //Lower X boundary is just clamped to the inflow value
            varDef.addBoundaryCondition(0, SAMS::domain::edges::lower, SAMS::simpleClamp<SAMS::T_dataType, 3>(varDef, clampValue));

            //Upper X boundary is zero gradient
            varDef.addBoundaryCondition(0, SAMS::domain::edges::upper, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        /**
         * Set zero gradient X boundary conditions
         */
        void KarmanVortex::setZeroGradientXbcs( SAMS::variableDef &varDef)
        {
            //Both X boundaries are zero gradient
            varDef.addBoundaryCondition(0, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }

        void KarmanVortex::setBCS(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZbcs(varDef);
            setYbcs(varDef);
            setInflowXbcs(varDef, clampValue);
        }

        void KarmanVortex::setBCS(std::string varName, SAMS::harness &harness)
        {
            auto &varDef = harness.variableRegistry.getVariable(varName);
            setZbcs(varDef);
            setYbcs(varDef);
            setZeroGradientXbcs(varDef);
        }

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        void KarmanVortex::controlVariables(LARE::simulationData &data, KarmanVortexParams &problemParams)
        {

            data.t_end = 20; // End time of the simulation
            data.dt_snapshots = data.t_end/100;
            data.nx = 512;
            data.ny = 256;
            data.nz = 2;

            data.x_min = -1.0;
            data.x_max = 3.0;
            data.y_min = -1.0;
            data.y_max = 1.0;
            data.z_min = 0.0;
            data.z_max = (data.x_max - data.x_min) * data.nz / data.nx;

            data.dt_multiplier = 0.8; // Default multiplier for time step
            // Geometry options: cartesian, cylindrical, spherical
            data.geometry = LARE::geometryType::Cartesian;
            // Shock viscosity coefficients
            data.visc1 = 0.1;
            data.visc2 = 1.0;

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
            data.rke = false;

            //Set up the specific parameters for the Karman Vortex problem
            //Note that this is just a Karman vortex street type problem, but we are just calling it KarmanVortex for simplicity. We can add more complex vortex shedding problems later if needed.
            //Mach =0.5 flow, cylinder radius = 0.1, ambient pressure = 1.0, density = 1.0
            //Note that there is no explicit viscosity in LARE, so we cannot set a Reynolds number, but with the shock viscosity coefficients above we get a reasonable vortex shedding pattern for these parameters.
            problemParams.ambPressure = 1.0;
            problemParams.density = 1.0;
            problemParams.flowVx = 0.5;
            problemParams.cylinderRadius = 0.1;
        }

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param data LARE3D simulation data
         */
        void KarmanVortex::setDomain(SAMS::harness &harness, LARE::simulationData &data) 
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
        void KarmanVortex::setBoundaryConditions(SAMS::harness &harness, LARE::simulationData &data, KarmanVortexParams &problemParams)
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
        void KarmanVortex::initialConditions(SAMS::harness &harnessRef, LARE::simulationData &data, KarmanVortexParams &problemParams)
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

        void KarmanVortex::startOfTimestep(LARE::simulationData &data, KarmanVortexParams &problemParams){
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

        void KarmanVortex::halfTimestep(LARE::simulationData &data, KarmanVortexParams &problemParams){
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

        void KarmanVortex::endOfTimestep(LARE::simulationData &data, KarmanVortexParams &problemParams){
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

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        void KarmanVortex::queryTerminate(bool &terminate, LARE::simulationData &data, SAMS::timeState &tData){
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
        void KarmanVortex::queryOutput(bool &shouldOutput, LARE::simulationData &data, SAMS::timeState &tData){
            static double nextOutputTime = data.dt_snapshots;
            if (tData.time >= (nextOutputTime) || (tData.time == 0.0)){
                shouldOutput |= true;
                nextOutputTime += data.dt_snapshots;
            }
        }
    }