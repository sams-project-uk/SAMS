
#include "SodShockTube.h"

namespace examples
{

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        template<typename T_EOS>
        void SodShockTubeNeutral<T_EOS>::attachBoundaryConditions(const std::string& varName, SAMS::harness &harness)
        {
            SAMS::variableDef &varDef = harness.variableRegistry.getVariable(varName);
            //Periodic boundaries in Y and Z
            std::shared_ptr<SAMS::boundaryConditions> periodicBC = std::make_shared<SAMS::simplePeriodicBC<SAMS::T_dataType, 3>>(varDef);
            varDef.addBoundaryCondition(1, periodicBC);
            varDef.addBoundaryCondition(2, periodicBC);

            //Mirror boundaries in X
            std::shared_ptr<SAMS::boundaryConditions> mirrorBC = std::make_shared<SAMS::simpleMirrorBC<SAMS::T_dataType, 3>>(varDef);
            varDef.addBoundaryCondition(0, mirrorBC);
        }

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        template<typename T_EOS>
        void SodShockTubeNeutral<T_EOS>::controlVariables(LARE::LARE3DNF<T_EOS>::simulationData &data, LARE::LARE3DST<T_EOS>::simulationData & coreData)
        {

            coreData.t_end = 0.2;
            coreData.dt_snapshots = data.t_end / 10;

            coreData.nx = 1024;
            coreData.ny = 2;
            coreData.nz = 2;

            coreData.x_min = 0.0;
            coreData.x_max = 1.0;
            coreData.y_min = 0.0;
            coreData.y_max = (coreData.x_max - coreData.x_min) * coreData.ny / coreData.nx;
            coreData.z_min = 0.0;
            coreData.z_max = (coreData.x_max - coreData.x_min) * coreData.nz / coreData.nx;

            coreData.dt_multiplier = 0.8; // Default multiplier for time step
            // Geometry options: cartesian, cylindrical, spherical
            coreData.geometry = LARE::geometryType::Cartesian;
            // Shock viscosity coefficients
            data.visc1 = 0.1;
            data.visc2 = 1.0;

            // Ratio of specific heat capacities
            data.gas_gamma = 1.4;

            coreData.mf = 1.2;

            // Remap kinetic energy correction
            data.rke = false;
        }

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param data LARE3D simulation data
         */
        template<typename T_EOS>
        void SodShockTubeNeutral<T_EOS>::setDomain(SAMS::harness &harness, LARE::LARE3DST<T_EOS>::simulationData &data) 
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
        template<typename T_EOS>
        void SodShockTubeNeutral<T_EOS>::setBoundaryConditions(SAMS::harness &harness)
        {
            //Grab variables and set boundary condition functions
            attachBoundaryConditions("LARENF/energy", harness);
            attachBoundaryConditions("LARENF/rho", harness);
            attachBoundaryConditions("LARENF/vx", harness);
            attachBoundaryConditions("LARENF/vy", harness);
            attachBoundaryConditions("LARENF/vz", harness);
            attachBoundaryConditions("LARENF/vx1", harness);
            attachBoundaryConditions("LARENF/vy1", harness);
            attachBoundaryConditions("LARENF/vz1", harness);
            attachBoundaryConditions("LARENF/dm", harness);
        }

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        template<typename T_EOS>
        void SodShockTubeNeutral<T_EOS>::initialConditions(SAMS::harness &harnessRef, LARE::LARE3DNF<T_EOS>::simulationData &data)
        {
            pw::portableArray<SAMS::T_dataType, 3> rho;
            pw::portableArray<SAMS::T_dataType, 3> energy;
            pw::portableArray<SAMS::T_dataType, 1> xc, yc, zc;

            auto &axisRegistry = harnessRef.axisRegistry;
            axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
            axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);
            axisRegistry.fillPPLocalAxis("Z", zc, SAMS::staggerType::CENTRED);

            auto &varRegistry = harnessRef.variableRegistry;
            varRegistry.fillPPArray("LARENF/rho", rho);
            varRegistry.fillPPArray("LARENF/energy", energy);

            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    SAMS::T_dataType pressure;
                    if (xc(ix) < 0.5)
                    {
                        rho(ix, iy, iz) = 1.0;
                        pressure = 1.0;
                    }
                    else
                    {
                        rho(ix, iy, iz) = 0.125;
                        pressure = 0.1;
                    }
                    //Specific internal energy
                    energy(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
                },
                rho.getRange(0), rho.getRange(1), rho.getRange(2));
        }

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * @param tData SAMS time state data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        template<typename T_EOS>
        void SodShockTubeNeutral<T_EOS>::queryTerminate(bool &terminate, LARE::LARE3DNF<T_EOS>::simulationData &data, SAMS::timeState &tData){
            //End at correct time for Sod Shock Tube (ends at t=0.2)
            if (tData.time >= data.t_end){
                terminate |= true;
            }
        }

        /**
         * Check whether to output data to disk
         * @param shouldOutput Boolean flag to set to true to output data
         * @param data LARE3D simulation data
        * @param tData SAMS time state data
         * This function checks whether data should be output to disk based on LARE3D data.
         * It returns true if data should be output.
         */
        template<typename T_EOS>
        void SodShockTubeNeutral<T_EOS>::queryOutput(bool &shouldOutput, LARE::LARE3DST<T_EOS>::simulationData &data, SAMS::timeState &tData){
            static double nextOutputTime = data.dt_snapshots;
            if (tData.time >= (nextOutputTime) || (tData.time == 0.0)){
                shouldOutput |= true;
                nextOutputTime += data.dt_snapshots;
            }
        }
}