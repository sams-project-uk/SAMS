
#include "TwoFluidTest.h"

namespace examples
{

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        void TwoFluidTest::attachBoundaryConditions(const std::string& varName, SAMS::harness &harness)
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
        void TwoFluidTest::controlVariables(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral)
        {

            data.t_end = 0.2;
            //data.t_end = 0.0000001;
            data.dt_snapshots = data.t_end / 10;
            //dataNeutral.t_end = data.t_end;
            //dataNeutral.dt_snapshots = data.dt_snapshots;

            data.nx = 512;
            data.ny = 2;
            data.nz = 2;

            data.x_min = -0.5;
            data.x_max = 1.5;
            data.y_min = 0.0;
            data.y_max = (data.x_max - data.x_min) * data.ny / data.nx;
            data.z_min = 0.0;
            data.z_max = (data.x_max - data.x_min) * data.nz / data.nx;

            data.dt_multiplier = 0.8; // Default multiplier for time step
            // Geometry options: cartesian, cylindrical, spherical
            data.geometry = LARE::geometryType::Cartesian;
            // Shock viscosity coefficients
            data.visc1 = 0.1;
            data.visc2 = 1.0;
            dataNeutral.dt_multiplier = data.dt_multiplier; // Default multiplier for time step
            // Geometry options: cartesian, cylindrical, spherical
    //        dataNeutral.geometry = LARE::LARE3DNF<T_EOS>::geometryType::Cartesian;
            // Shock viscosity coefficients
            dataNeutral.visc1 = data.visc1;
            dataNeutral.visc2 = data.visc2;

            // Ratio of specific heat capacities
            data.gas_gamma = 1.4;
            // Ratio of specific heat capacities
            dataNeutral.gas_gamma = data.gas_gamma;

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
            // Remap kinetic energy correction
            dataNeutral.rke = data.rke;
 
        }

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param data LARE3D simulation data
         */
        void TwoFluidTest::setDomain(SAMS::harness &harness, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral) 
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
        void TwoFluidTest::setBoundaryConditions(SAMS::harness &harness)
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
        
            attachBoundaryConditions("bx_n", harness);
            attachBoundaryConditions("by_n", harness);
            attachBoundaryConditions("bz_n", harness);
            attachBoundaryConditions("energy_neutral", harness);
            attachBoundaryConditions("rho_n", harness);
            attachBoundaryConditions("vx_n", harness);
            attachBoundaryConditions("vy_n", harness);
            attachBoundaryConditions("vz_n", harness);
            attachBoundaryConditions("LARE/vx1_n", harness);
            attachBoundaryConditions("LARE/vy1_n", harness);
            attachBoundaryConditions("LARE/vz1_n", harness);
            attachBoundaryConditions("LARE/dm_n", harness);
        }

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void TwoFluidTest::initialConditions(SAMS::harness &harnessRef, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral)
        {
        
            //char shock_tube_problem[8]="sod";
            //char shock_tube_problem[8]="briowu";
            
            if (problem =="sod"){
                printf("sod Shock Tube \n");
                pw::portableArray<SAMS::T_dataType, 3> rho;
                pw::portableArray<SAMS::T_dataType, 3> energy_electron;
                pw::portableArray<SAMS::T_dataType, 3> rho_n;
                pw::portableArray<SAMS::T_dataType, 3> energy_neutral;
                pw::portableArray<SAMS::T_dataType, 1> xc, yc, zc;
                pw::portableArray<SAMS::T_dataType, 3> bx,by,bz;

                auto &axisRegistry = harnessRef.axisRegistry;
                axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Z", zc, SAMS::staggerType::CENTRED);

                auto &varRegistry = harnessRef.variableRegistry;
                varRegistry.fillPPArray("rho", rho);
                varRegistry.fillPPArray("energy_ion", energy_electron);
                varRegistry.fillPPArray("energy_electron", energy_electron);
                varRegistry.fillPPArray("rho_n", rho_n);
                varRegistry.fillPPArray("energy_neutral", energy_neutral);
                varRegistry.fillPPArray("bx", bx);
                varRegistry.fillPPArray("by", by);
                varRegistry.fillPPArray("bz", bz);

                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        SAMS::T_dataType pressure;
                        SAMS::T_dataType pressure_n;
                        if (xc(ix) < 0.5)
                        {
                            rho(ix, iy, iz) = 1.0;
                            pressure = 1.0;
                            rho_n(ix, iy, iz) = 0.125;
                            pressure_n = 0.1;
                        }
                        else
                        {
                            rho(ix, iy, iz) = 0.125;
                            pressure = 0.1;
                            rho_n(ix, iy, iz) = 1.0;
                            pressure_n = 1.0;
                        }
                        //Specific internal energy
                        energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
                        energy_neutral(ix, iy, iz) = pressure_n / ((dataNeutral.gas_gamma - 1.0) * rho_n(ix, iy, iz));
                        bx(ix,iy,iz)=0.0;
                        by(ix,iy,iz)=0.0;
                        bz(ix,iy,iz)=0.0;
                    },
                    rho.getRange(0), rho.getRange(1), rho.getRange(2));
            }
            if (problem == "briowu"){
                // Brio & Wu Shock tube
                printf("Brio Wu Shock Tube \n");
                
                pw::portableArray<SAMS::T_dataType, 3> rho;
                pw::portableArray<SAMS::T_dataType, 3> bx,by,bz;
                pw::portableArray<SAMS::T_dataType, 3> energy_electron;
                pw::portableArray<SAMS::T_dataType, 3> energy_ion;
                pw::portableArray<SAMS::T_dataType, 3> rho_n;
                pw::portableArray<SAMS::T_dataType, 3> energy_neutral;
                pw::portableArray<SAMS::T_dataType, 1> xc, yc, zc;

                auto &axisRegistry = harnessRef.axisRegistry;
                axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Z", zc, SAMS::staggerType::CENTRED);

                auto &varRegistry = harnessRef.variableRegistry;
                varRegistry.fillPPArray("rho", rho);
                varRegistry.fillPPArray("bx", bx);
                varRegistry.fillPPArray("by", by);
                varRegistry.fillPPArray("bz", bz);
                varRegistry.fillPPArray("energy_electron", energy_electron);
                varRegistry.fillPPArray("energy_ion", energy_ion);
                varRegistry.fillPPArray("rho_n", rho_n);
                varRegistry.fillPPArray("energy_neutral", energy_neutral);
                
                LARE::T_dataType xi_n=0.9;
                LARE::T_dataType xi_p=1.0-xi_n;
                LARE::T_dataType f_p_p=2.0*xi_p/(xi_n+2.0*xi_p);
                LARE::T_dataType f_p_n=xi_n/(xi_n+2.0*xi_p);
                
                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        SAMS::T_dataType pressure;
                        SAMS::T_dataType pressure_n;
                        if (xc(ix) < 0.5)
                        {
                            rho(ix, iy, iz) = 1.0*(1.0-xi_n);
                            pressure = 1.0*f_p_p;
                            bx(ix, iy, iz)=0.75;
                            by(ix, iy, iz)=1.0;
                            rho_n(ix, iy, iz) = 1.0*xi_n;
                            pressure_n = 1.0*f_p_n;
                        }
                        else
                        {
                            rho(ix, iy, iz) = 0.125*(1.0-xi_n);
                            pressure = 0.1*f_p_p;
                            bx(ix, iy, iz)=0.75;
                            by(ix, iy, iz)=-1.0;
                            rho_n(ix, iy, iz) = 0.125*xi_n;
                            pressure_n = 0.1*f_p_n;
                        }
                        //Specific internal energy
                        //energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;
                        energy_ion(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
                        energy_neutral(ix, iy, iz) = pressure_n / ((dataNeutral.gas_gamma - 1.0) * rho_n(ix, iy, iz));
                    },
                    rho.getRange(0), rho.getRange(1), rho.getRange(2));
            }
            ///////////////////////////////////////////////////////////////////////////
            if (problem =="hillier"){
                // Brio & Wu Shock tube
                printf("Hillier Shock Tube \n");
                
                pw::portableArray<SAMS::T_dataType, 3> rho;
                pw::portableArray<SAMS::T_dataType, 3> bx,by,bz;
                pw::portableArray<SAMS::T_dataType, 3> energy_electron;
                pw::portableArray<SAMS::T_dataType, 3> energy_ion;
                pw::portableArray<SAMS::T_dataType, 3> rho_n;
                pw::portableArray<SAMS::T_dataType, 3> energy_neutral;
                pw::portableArray<SAMS::T_dataType, 1> xc, yc, zc;

                auto &axisRegistry = harnessRef.axisRegistry;
                axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Z", zc, SAMS::staggerType::CENTRED);

                auto &varRegistry = harnessRef.variableRegistry;
                varRegistry.fillPPArray("rho", rho);
                varRegistry.fillPPArray("bx", bx);
                varRegistry.fillPPArray("by", by);
                varRegistry.fillPPArray("bz", bz);
                varRegistry.fillPPArray("energy_electron", energy_electron);
                varRegistry.fillPPArray("energy_ion", energy_ion);
                varRegistry.fillPPArray("rho_n", rho_n);
                varRegistry.fillPPArray("energy_neutral", energy_neutral);
                
                //if empirical should go here
                
                //TWOFLUID::PIP::get_equilibrium_ion_fraction(data.T_reference,LARE::T_dataType xi_n);
                
                LARE::T_dataType xi_n=0.9;
                LARE::T_dataType xi_p=1.0-xi_n;
                LARE::T_dataType f_p_p=2.0*xi_p/(xi_n+2.0*xi_p);
                LARE::T_dataType f_p_n=xi_n/(xi_n+2.0*xi_p);
                
                printf("neutral fraction=%f \n",xi_n);
                
                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        SAMS::T_dataType pressure;
                        SAMS::T_dataType pressure_n;
                        if (xc(ix) < 0.5)
                        {
                            rho(ix, iy, iz) = 1.0*(1.0-xi_n);
                            pressure = 1.0*f_p_p;
                            bx(ix, iy, iz)=0.75;
                            by(ix, iy, iz)=1.0;
                            rho_n(ix, iy, iz) = 1.0*xi_n;
                            pressure_n = 1.0*f_p_n;
                        }
                        else
                        {
                            rho(ix, iy, iz) = 1.0*(1.0-xi_n);
                            pressure = 1.0*f_p_p;
                            bx(ix, iy, iz)=0.75;
                            by(ix, iy, iz)=-1.0;
                            rho_n(ix, iy, iz) = 1.0*xi_n;
                            pressure_n = 1.0*f_p_n;
                        }
                        //Specific internal energy
                        //energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;
                        energy_ion(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
                        energy_neutral(ix, iy, iz) = pressure_n / ((dataNeutral.gas_gamma - 1.0) * rho_n(ix, iy, iz));
                    },
                    rho.getRange(0), rho.getRange(1), rho.getRange(2));
            }
            //////////////////////////////////////////////////////////////////////////////
        }

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * @param tData SAMS time state data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        void TwoFluidTest::queryTerminate(bool &terminate, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, SAMS::timeState &tData){
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
        void TwoFluidTest::queryOutput(bool &shouldOutput, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, SAMS::timeState &tData){
            static double nextOutputTime = data.dt_snapshots;
            if (tData.time >= (nextOutputTime) || (tData.time == 0.0)){
                shouldOutput |= true;
                nextOutputTime += data.dt_snapshots;
            }
        }
}
