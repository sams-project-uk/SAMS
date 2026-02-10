
#include "OrszagTang3D.h"

namespace examples
{

    /**
     * Internal function to attach boundary conditions to a variable
     * @param varName Name of the variable to attach boundary conditions to
     * @param harness SAMS harness
     */
    void OrszagTang3D::attachBoundaryConditions(const std::string &varName, SAMS::harness &harness)
    {
        SAMS::variableDef &varDef = harness.variableRegistry.getVariable(varName);
        // Periodic boundaries in all directions
        std::shared_ptr<SAMS::boundaryConditions> periodicBC = std::make_shared<SAMS::simplePeriodicBC<SAMS::T_dataType, 3>>(varDef);
        varDef.addBoundaryCondition(0, periodicBC);
        varDef.addBoundaryCondition(1, periodicBC);
        varDef.addBoundaryCondition(2, periodicBC);
    }

    /**
     * Set up the LARE style simulation control variables
     * @note This should later be moved to a more general approach
     * when we have multiple core solvers.
     * @param data LARE3D simulation data
     */
    void OrszagTang3D::controlVariables(LARE::simulationData &data)
    {

        data.t_end = 3.0; // End time of the simulation
        data.dt_snapshots = data.t_end / 10;

        data.nx = 128;
        data.ny = 128;
        data.nz = 128;

        data.x_min = 0.0;
        data.x_max = 2.0 * M_PI;
        data.y_min = 0.0;
        data.y_max = 2.0 * M_PI;
        data.z_min = 0.0;
        data.z_max = 2.0 * M_PI;

        data.dt_multiplier = 0.8; // Default multiplier for time step
        // Geometry options: cartesian, cylindrical, spherical
        data.geometry = LARE::geometryType::Cartesian;
        // Shock viscosity coefficients
        data.visc1 = 0.1;
        data.visc2 = 1.0;

        // Ratio of specific heat capacities
        data.gas_gamma = 5.0 / 3.0;

        // Average mass of an ion in proton masses
        data.mf = 1.2;

        // Run with normalised mu0
        data.mu0 = 1.0;

        // Resistive MHD options
        data.resistiveMHD = false;
        data.eta_background = 1.e-10;
        data.j_max = 1.0;
        data.eta0 = 2.e-10;

        // Remap kinetic energy correction
        data.rke = false;
    }

    /**
     * Set up the simulation domain
     * @param harness SAMS harness
     * @param data LARE3D simulation data
     */
    void OrszagTang3D::setDomain(SAMS::harness &harness, LARE::simulationData &data)
    {
        auto &axisReg = harness.axisRegistry;
        // Just hard code the domain for the Sod Shock Tube
        axisReg.setDomain("X", data.nx, data.x_min, data.x_max);
        axisReg.setDomain("Y", data.ny, data.y_min, data.y_max);
        axisReg.setDomain("Z", data.nz, data.z_min, data.z_max);
    }

    /**
     * Set boundary conditions for the simulation
     * @param harness SAMS harness
     */
    void OrszagTang3D::setBoundaryConditions(SAMS::harness &harness)
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
    void OrszagTang3D::initialConditions(SAMS::harness &harnessRef, LARE::simulationData &data)
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

        pw::applyKernel(
            LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz) {
                using T_dataType = SAMS::T_dataType;
                T_dataType pressure = data.gas_gamma;

                rho(ix, iy, iz) = data.gas_gamma * data.gas_gamma;
                vx(ix, iy, iz) = -std::sin(yb(iy));
                vy(ix, iy, iz) = std::sin(xb(ix));
                vz(ix, iy, iz) = 0.0;
                bx(ix, iy, iz) = -std::sin(yc(iy));
                by(ix, iy, iz) = std::sin(2.0 * xc(ix));
                bz(ix, iy, iz) = 0.0;

                // Specific internal energy
                energy_electron(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz)) / 2.0;
                energy_ion(ix, iy, iz) = pressure / ((data.gas_gamma - 1.0) * rho(ix, iy, iz)) / 2.0;
            },
            rho.getRange(0), rho.getRange(1), rho.getRange(2));
    }

    /**
     * Check whether to terminate the simulation
     * @param terminate Boolean flag to set to true to terminate the simulation
     * @param data LARE3D simulation data
     * This function checks whether the simulation should terminate based on LARE3D data.
     * It sets the terminate flag to true if the simulation should end.
     */
    void OrszagTang3D::queryTerminate(bool &terminate, LARE::simulationData &data, SAMS::timeState &tData)
    {
        // End at correct time for Sod Shock Tube (ends at t=0.2)
        if (tData.time >= data.t_end)
        {
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
    void OrszagTang3D::queryOutput(bool &shouldOutput, LARE::simulationData &data, SAMS::timeState &tData)
    {
            static double nextOutputTime = data.dt_snapshots;
            if (tData.time >= (nextOutputTime) || (tData.time == 0.0)){
                shouldOutput |= true;
                nextOutputTime += data.dt_snapshots;
            }
    }

}
