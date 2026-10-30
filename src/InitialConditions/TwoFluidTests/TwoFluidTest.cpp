
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
            //std::shared_ptr<SAMS::boundaryConditions> mirrorBC = std::make_shared<SAMS::simpleMirrorBC<SAMS::T_dataType, 3>>(varDef);
            //varDef.addBoundaryCondition(0, mirrorBC);
            std::shared_ptr<SAMS::boundaryConditions> freeBC = std::make_shared<SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>>(varDef);
            varDef.addBoundaryCondition(0, freeBC);
        }

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        void TwoFluidTest::controlVariables(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                                            TWOFLUID::data_two_fluid_source &plasma_source)
        {

            data.t_end = 100.0;
            data.dt_snapshots = data.t_end / 1000;

            data.nx = 4096;
            data.ny = 2;
            data.nz = 2;

            data.x_min = -3000.0;
            data.x_max = 3000.0;
            data.y_min = 0.0;
            data.y_max = (data.x_max - data.x_min) * data.ny / data.nx;
            data.z_min = 0.0;
            data.z_max = (data.x_max - data.x_min) * data.nz / data.nx;

            data.dt_multiplier = 0.8; // Default multiplier for time step
            // Geometry options: cartesian, cylindrical, spherical
            data.geometry = LARE::geometryType::Cartesian;
            // Shock viscosity coefficients
            data.visc1 = 0.8;
            data.visc2 = 1.0;
            // Geometry options: cartesian, cylindrical, spherical
            // Shock viscosity coefficients
            dataNeutral.visc1 = data.visc1;
            dataNeutral.visc2 = data.visc2;

            // Ratio of specific heat capacities
            data.gas_gamma = 5.0/3.0;
            // Ratio of specific heat capacities
            dataNeutral.gas_gamma = data.gas_gamma;

            // Average mass of an ion in proton masses
            data.mf = 1.0;
            
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

            if (problem == "snow_hillier_2021" || problem == "symmetric_pulse") {
                // Run length and grid
                data.t_end = sh21.t_end;
                data.dt_snapshots = data.t_end / 10;
                data.nx = sh21.nx;
                data.x_min = -sh21.x_half;
                data.x_max = sh21.x_half;
                data.y_max = (data.x_max - data.x_min) * data.ny / data.nx;
                data.z_max = (data.x_max - data.x_min) * data.nz / data.nx;

                // The two-fluid parameters that define this problem. They are set here, rather than in
                // twofluid.h, so that the initial state and the source terms cannot disagree: the
                // equilibrium in initialConditions is built from the same T0, and setNormalisation()
                // derives T_norm and phi_hat from the same reference state. Solver choices (implicit,
                // tolerances, diagnostics) are left to twofluid.h.
                plasma_source.T0 = sh21.T0;
                plasma_source.t_ir = sh21.tau_IR;
                plasma_source.alpha0 = sh21.alpha0;
                plasma_source.rho_ref = sh21.rho_t;
                plasma_source.P_ref = 0.5 * sh21.beta * sh21.B0 * sh21.B0;
                plasma_source.collisions = true;
                plasma_source.ion_rec_empirical = true;
                plasma_source.ion_rec_nlevel = false;
                plasma_source.ionisation_potential = sh21.ionisation_potential;
                // Mirror parity of B_y for the symmetry check: odd across the sheet, even for a uniform field
                plasma_source.symmetry_by_odd = (problem == "snow_hillier_2021");
                if (!sh21.shock_viscosity) {
                    data.visc1 = 0.0; data.visc2 = 0.0;
                    dataNeutral.visc1 = 0.0; dataNeutral.visc2 = 0.0;
                    printf("snow_hillier_2021 / symmetric_pulse: SHOCK VISCOSITY OFF (sh21.shock_viscosity = false)\n");
                }

                printf("Snow & Hillier (2021) slow-mode shock: setting T0 = %g K, t_ir = %g, alpha0 = %g,\n"
                       "  rho_ref = %g, P_ref = %g, collisions and empirical rates on, n-level rates off, %s model\n",
                       static_cast<double>(plasma_source.T0), static_cast<double>(plasma_source.t_ir),
                       static_cast<double>(plasma_source.alpha0), static_cast<double>(plasma_source.rho_ref),
                       static_cast<double>(plasma_source.P_ref),
                       plasma_source.ionisation_potential ? "IRIP" : "IR");
            }
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
            
            printf("Species array temporarily disabled for MPI compatability \n");
            //axisReg.setDomainValues("species", 7, [](COUNT_TYPE i){ return i; });
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
            attachBoundaryConditions("rho", harness);
            attachBoundaryConditions("vx", harness);
            attachBoundaryConditions("vy", harness);
            attachBoundaryConditions("vz", harness);
            attachBoundaryConditions("LARE/vx1", harness);
            attachBoundaryConditions("LARE/vy1", harness);
            attachBoundaryConditions("LARE/vz1", harness);
            attachBoundaryConditions("LARE/dm", harness);
        
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
        void TwoFluidTest::initialConditions(SAMS::harness &harnessRef, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral)
        {
        
            //char shock_tube_problem[8]="sod";
            //char shock_tube_problem[8]="briowu";
            
            if (problem =="sod"){
                printf("sod Shock Tube \n");
                pw::portableArray<SAMS::T_dataType, 3> rho;
                pw::portableArray<SAMS::T_dataType, 3> energy_ion;
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
                varRegistry.fillPPArray("energy_ion", energy_ion);
                varRegistry.fillPPArray("LARENF/rho", rho_n);
                varRegistry.fillPPArray("LARENF/energy", energy_neutral);
                varRegistry.fillPPArray("bx", bx);
                varRegistry.fillPPArray("by", by);
                varRegistry.fillPPArray("bz", bz);

                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        SAMS::T_dataType pressure;
                        SAMS::T_dataType pressure_n;
                        if (xc(ix) < 0.0)
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
                        energy_ion(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
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
                varRegistry.fillPPArray("energy_ion", energy_ion);
                varRegistry.fillPPArray("LARENF/rho", rho_n);
                varRegistry.fillPPArray("LARENF/energy_neutral", energy_neutral);
                
                LARE::T_dataType xi_n=0.9;
                LARE::T_dataType xi_p=1.0-xi_n;
                LARE::T_dataType f_p_p=2.0*xi_p/(xi_n+2.0*xi_p);
                LARE::T_dataType f_p_n=xi_n/(xi_n+2.0*xi_p);
                
                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        SAMS::T_dataType pressure;
                        SAMS::T_dataType pressure_n;
                        if (xc(ix) < 0.0)
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
                pw::portableArray<SAMS::T_dataType, 3> energy;
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
                varRegistry.fillPPArray("energy_ion", energy);
                varRegistry.fillPPArray("LARENF/rho", rho_n);
                varRegistry.fillPPArray("LARENF/energy", energy_neutral);
                
                //if empirical should go here
                
                //TWOFLUID::PIP::get_equilibrium_ion_fraction(data.T_reference,LARE::T_dataType xi_n);
                
                //Much of this should go elsewhere
                LARE::T_dataType T0=10000.0;//data.T_reference; //Reference temperature
                LARE::T_dataType n0=1.0e16;//data.ne_reference; //Reference electron number density

                LARE::T_dataType Te_0=T0/1.1604e4; //Calculate electron temperature in eV
                LARE::T_dataType rec_fac=2.6e-19*(n0)/std::sqrt(Te_0);  //reference recombination rate (n0 converted to m^-3)

                //initial equilibrium fractions
                LARE::T_dataType ioneq=(2.6e-19/std::sqrt(Te_0))/(2.91e-14/(0.232+13.6/Te_0)*std::pow(13.6/Te_0,0.39)*std::exp(-13.6/Te_0));
                
                //LARE::T_dataType xi_n=0.95;
                LARE::T_dataType xi_n=ioneq/(ioneq+1.0);
                LARE::T_dataType xi_p=1.0-xi_n;
                LARE::T_dataType f_p_p=2.0*xi_p/(xi_n+2.0*xi_p);
                LARE::T_dataType f_p_n=xi_n/(xi_n+2.0*xi_p);

		LARE::T_dataType beta=0.1;
		LARE::T_dataType B0=1.0*std::sqrt(2.0*1.0/beta); //B^2=2P/beta
                
                printf("neutral fraction=%f \n",xi_n);
                
                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        SAMS::T_dataType pressure;
                        SAMS::T_dataType pressure_n;
                        if (xc(ix) < 0.0)
                        {
                            rho(ix, iy, iz) = 1.0*(1.0-xi_n);
                            pressure = 1.0*f_p_p;
                            bx(ix, iy, iz)=B0*0.3;
                            by(ix, iy, iz)=B0;
                            rho_n(ix, iy, iz) = 1.0*xi_n;
                            pressure_n = 1.0*f_p_n;
                        }
                        else
                        {
                            rho(ix, iy, iz) = 1.0*(1.0-xi_n);
                            pressure = 1.0*f_p_p;
                            bx(ix, iy, iz)=B0*0.3;
                            by(ix, iy, iz)=-B0;
                            rho_n(ix, iy, iz) = 1.0*xi_n;
                            pressure_n = 1.0*f_p_n;
                        }
                        //Specific internal energy
                        //energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;
                        energy(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
                        energy_neutral(ix, iy, iz) = pressure_n / ((dataNeutral.gas_gamma - 1.0) * rho_n(ix, iy, iz));
                    },
                    rho.getRange(0), rho.getRange(1), rho.getRange(2));
            }
            //////////////////////////////////////////////////////////////////////////////
            if (problem == "snow_hillier_2021" || problem == "symmetric_pulse"){
                // Snow & Hillier (2021), A&A 645, A81, eqs. 79-84: a current sheet at x = 0 in a uniform
                // partially ionised plasma in thermal and ionisation equilibrium at T0
                const bool pulse_test = (problem == "symmetric_pulse");
                if (pulse_test) {
                    printf("Mirror-symmetry test: Gaussian hot spot (amplitude %g, width %g) in a uniform field B_y = %g\n",
                           static_cast<double>(pulse.amplitude), static_cast<double>(pulse.width), static_cast<double>(pulse.By));
                } else {
                    printf("Snow & Hillier (2021) reconnection-driven slow-mode shock\n");
                }

                pw::portableArray<SAMS::T_dataType, 3> rho;
                pw::portableArray<SAMS::T_dataType, 3> bx,by,bz;
                pw::portableArray<SAMS::T_dataType, 3> energy;
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
                varRegistry.fillPPArray("energy_ion", energy);
                varRegistry.fillPPArray("LARENF/rho", rho_n);
                varRegistry.fillPPArray("LARENF/energy", energy_neutral);

                // Equilibrium composition at T0, from the same formula setNormalisation() uses
                SAMS::T_dataType xi_n;
                TWOFLUID::PIP<T_EOS>::get_equilibrium_ion_fraction(sh21.T0, xi_n);
                const SAMS::T_dataType xi_i = 1.0 - xi_n;
                const SAMS::T_dataType P_t = 0.5 * sh21.beta * sh21.B0 * sh21.B0;   // eqs. 83-84
                const SAMS::T_dataType P_p = 2.0*xi_i/(xi_n + 2.0*xi_i) * P_t;
                const SAMS::T_dataType P_n = xi_n/(xi_n + 2.0*xi_i) * P_t;
                const SAMS::T_dataType rho_p0 = xi_i * sh21.rho_t;                  // eq. 82
                const SAMS::T_dataType rho_n0 = xi_n * sh21.rho_t;                  // eq. 81
                const SAMS::T_dataType e_p = P_p / ((data.gas_gamma - 1.0) * rho_p0);        // specific energies
                const SAMS::T_dataType e_n = P_n / ((dataNeutral.gas_gamma - 1.0) * rho_n0);
                const SAMS::T_dataType Bx = sh21.Bx, B0 = sh21.B0;
                const SAMS::T_dataType dx_cell = (data.x_max - data.x_min) / data.nx;
                const SAMS::T_dataType w_sheet = sh21.sheet_width_cells * dx_cell;   // 0: sharp step
                const SAMS::T_dataType pi = 3.14159265358979323846;
                const SAMS::T_dataType amp = pulse_test ? pulse.amplitude : 0.0;
                const SAMS::T_dataType pw_inv = 1.0 / pulse.width;
                const SAMS::T_dataType By_uniform = pulse.By;
                if (pulse_test) {
                } else if (w_sheet > 0.0) {
                    printf("  current sheet: B_y = -B0 tanh(pi x / %g) (%g cells, as PIP with sharp_interface = .false.)\n",
                           static_cast<double>(w_sheet), static_cast<double>(sh21.sheet_width_cells));
                } else {
                    printf("  current sheet: sharp step at x = 0 (as PIP with sharp_interface = .true.)\n");
                }

                printf("  xi_n = %.6f, P_t = %g, T = %.6f, plasma beta = %g, B_x = %g, B_y = -/+%g for x >/< 0\n",
                       static_cast<double>(xi_n), static_cast<double>(P_t),
                       static_cast<double>(dataNeutral.gas_gamma*P_n/rho_n0),
                       static_cast<double>(sh21.beta), static_cast<double>(Bx), static_cast<double>(B0));

                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        rho(ix, iy, iz) = rho_p0;
                        rho_n(ix, iy, iz) = rho_n0;
                        // symmetric_pulse: both pressures (hence temperatures) raised by the same Gaussian factor
                        const SAMS::T_dataType heat = 1.0 + amp * std::exp(-(xc(ix) * pw_inv) * (xc(ix) * pw_inv));
                        energy(ix, iy, iz) = e_p * heat;
                        energy_neutral(ix, iy, iz) = e_n * heat;
                        bx(ix, iy, iz) = Bx;
                        by(ix, iy, iz) = pulse_test ? By_uniform : (w_sheet > 0.0) ? -B0 * std::tanh(pi * xc(ix) / w_sheet)   // smooth sheet
                                                         : ((xc(ix) > 0.0) ? -B0 : B0);           // eq. 80
                    },
                    rho.getRange(0), rho.getRange(1), rho.getRange(2));
            }
            //////////////////////////////////////////////////////////////////////////////
            if (problem =="sedov"){
                // Brio & Wu Shock tube
                printf("Sedov-Taylor Expansion \n");
                
                pw::portableArray<SAMS::T_dataType, 3> rho;
                pw::portableArray<SAMS::T_dataType, 3> bx,by,bz;
                pw::portableArray<SAMS::T_dataType, 3> energy;
                pw::portableArray<SAMS::T_dataType, 3> rho_n;
                pw::portableArray<SAMS::T_dataType, 3> energy_neutral;
                pw::portableArray<SAMS::T_dataType, 1> xc, yc, zc;

                auto &axisRegistry = harnessRef.axisRegistry;
                axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);
                axisRegistry.fillPPLocalAxis("Z", zc, SAMS::staggerType::CENTRED);

                auto &varRegistry = harnessRef.variableRegistry;
                varRegistry.fillPPArray("rho", rho);
                varRegistry.fillPPArray("energy_ion", energy);
                varRegistry.fillPPArray("LARENF/rho", rho_n);
                varRegistry.fillPPArray("LARENF/energy", energy_neutral);
                
                //if empirical should go here
                
                //TWOFLUID::PIP::get_equilibrium_ion_fraction(data.T_reference,LARE::T_dataType xi_n);
                
                //Much of this should go elsewhere
                LARE::T_dataType T0=10000.0;//data.T_reference; //Reference temperature
                LARE::T_dataType n0=1.0e16;//data.ne_reference; //Reference electron number density

                LARE::T_dataType Te_0=T0/1.1604e4; //Calculate electron temperature in eV
                LARE::T_dataType rec_fac=2.6e-19*(n0)/std::sqrt(Te_0);  //reference recombination rate (n0 converted to m^-3)

                //initial equilibrium fractions
                LARE::T_dataType ioneq=(2.6e-19/std::sqrt(Te_0))/(2.91e-14/(0.232+13.6/Te_0)*std::pow(13.6/Te_0,0.39)*std::exp(-13.6/Te_0));
                
                //LARE::T_dataType xi_n=0.9;
                LARE::T_dataType xi_n=ioneq/(ioneq+1.0);
                LARE::T_dataType xi_p=1.0-xi_n;
                LARE::T_dataType f_p_p=2.0*xi_p/(xi_n+2.0*xi_p);
                LARE::T_dataType f_p_n=xi_n/(xi_n+2.0*xi_p);
                
                printf("neutral fraction=%f \n",xi_n);
                
                pw::applyKernel(
                    LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                    {
                        SAMS::T_dataType pressure;
                        SAMS::T_dataType pressure_n;
                        if ((xc(ix) > -0.1) and (xc(ix) < 0.1))
                        {
                            rho(ix, iy, iz) = 1.0*(1.0-xi_n);
                            pressure = 2.0*f_p_p;
                            rho_n(ix, iy, iz) = 1.0*xi_n;
                            pressure_n = 1.0*f_p_n;
                        }
                        else
                        {
                            rho(ix, iy, iz) = 1.0*(1.0-xi_n);
                            pressure = 1.0*f_p_p;
                            rho_n(ix, iy, iz) = 1.0*xi_n;
                            pressure_n = 1.0*f_p_n;
                        }
                        //Specific internal energy
                        //energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz))/2.0;
                        energy(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz));
                        energy_neutral(ix, iy, iz) = pressure_n / ((dataNeutral.gas_gamma - 1.0) * rho_n(ix, iy, iz));
                    },
                    rho.getRange(0), rho.getRange(1), rho.getRange(2));
            }
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
