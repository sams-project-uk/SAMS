#ifndef TWOFLUIDTEST_H
#define TWOFLUIDTEST_H

#include "harness.h"
#include "runner.h"
#include "builtInBoundaryConditions.h"
#include "TWOFLUID/twofluid.h"
//#include "../PhysicsPackages/TWOFLUID/twofluid.h"


namespace examples
{

    namespace pw = portableWrapper;

    /**
     * Initial conditions, boundary conditions and domain setup for the Sod Shock Tube problem
     */
    class TwoFluidTest
    {
    private:

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        void attachBoundaryConditions(const std::string& varName, SAMS::harness &harness);

    public:

        using T_EOS = LARE::idealGas;
        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = "TwoFluidTest";

        // Which problem: "sod", "briowu", "hillier", "sedov" or "snow_hillier_2021"
        // TODO - this should be a data item somewhere...
        const std::string problem = "snow_hillier_2021";

        /**
         * Parameters of the reconnection-driven slow-mode shock of Snow & Hillier (2021), A&A 645, A81,
         * Sect. 4 and Sect. 5 (reference case). Used only when problem == "snow_hillier_2021".
         * Units are the paper's: Alfven speed B0/sqrt(rho_t) = 1, rho_t = 1, and alpha0 = 1 so that the
         * unit of time is the collision time.
         */
        struct snowHillier2021Parameters {
            SAMS::T_dataType T0 = 1.0e4;     // reference temperature [K]: xi_n = 0.99734
            SAMS::T_dataType beta = 0.1;     // plasma beta, P_t = beta * B0^2 / 2
            SAMS::T_dataType B0 = 1.0;       // |B_y| either side of the current sheet
            SAMS::T_dataType Bx = 0.1;       // guide (normal) field component, eq. 79
            SAMS::T_dataType rho_t = 1.0;    // total density
            SAMS::T_dataType tau_IR = 1.0e-3;// recombination rate relative to the collision rate
            SAMS::T_dataType alpha0 = 1.0;   // collisional coefficient at t = 0
            bool ionisation_potential = true;// true: IRIP model (the paper's main case, and PIP's model); false: IR
            // Current sheet profile. 0: sharp step at x = 0 (eq. 80). > 0: B_y = -B0 tanh(pi x / (w dx)) with
            // w = sheet_width_cells, a half-width of about w/pi cells. w = 10 is exactly PIP's smooth sheet
            // (shock_tube_ion.f90 with sharp_interface = .false.: mask = pi x, wtr = 10 dx), so set the same
            // choice in both codes. As in PIP, only B_y is smoothed (the sheet is not in pressure balance and
            // collapses, which drives the reconnection). A step across one cell face makes the first response
            // of the sheet as violent as the grid allows and amplifies round-off by many orders of magnitude.
            SAMS::T_dataType sheet_width_cells = 10.0;
            // Run length and grid for a first validation run (the bottom row of Fig. 7, t = 100), matched
            // cell for cell by the PIP reference set-up (shock_tube_ion.f90 with ix = 4096 on [-150, 150],
            // open boundaries, 11 outputs). The paper's equilibrium figures are at t = 40000 with 256000
            // cells; scale x_half with t_end (fast-mode signals travel at ~1.05 in these units) and nx to
            // keep dx, and change both codes together.
            SAMS::T_dataType t_end = 100.0;
            SAMS::T_dataType x_half = 150.0; // domain is [-x_half, x_half]
            SAMS::T_indexType nx = 4096;     // dx = 0.073
        } sh21;
        /**
         * Initial conditions should not be timed
         */
        constexpr static bool timeSimulation = false;

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        void controlVariables(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                              TWOFLUID::data_two_fluid_source &plasma_source);

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param data LARE3D simulation data
         */
        void setDomain(SAMS::harness &harness, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral);

        /**
         * Set boundary conditions for the simulation
         * @param harness SAMS harness
         */
        void setBoundaryConditions(SAMS::harness &harness);

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void initialConditions(SAMS::harness &harnessRef, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral);

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * @param tData SAMS time state data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        void queryTerminate(bool &terminate, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, SAMS::timeState &tData);

        /**
         * Check whether to output data to disk
         * @param shouldOutput Boolean flag to set to true to output data
         * @param data LARE3D simulation data
        * @param tData SAMS time state data
         * This function checks whether data should be output to disk based on LARE3D data.
         * It returns true if data should be output.
         */
        void queryOutput(bool &shouldOutput, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, SAMS::timeState &tData);

    };
}
#endif // SODSHOCKTUBE_H
