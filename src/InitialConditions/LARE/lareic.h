#ifndef LAREIC_SHARED_DATA_H
#define LAREIC_SHARED_DATA_H

#include "shared_data.h"
#include "harness.h"
#include "lareBoundaryClass.h"
#include "runner.h"
#include "../../Operations/OutputControl/defaultSchedule.h"
#include <array>
#include <string>
#include <utility>

namespace LARE{

    /**
     * The state that an inflow boundary holds in its ghost cells
     */
    struct inflowState
    {
        SAMS::T_dataType rho, energy_electron, energy_ion;
        SAMS::T_dataType vx, vy, vz;
        SAMS::T_dataType bx, by, bz;
    };

    /**
     * Data for the LARE3D style initial conditions
     */
    struct icData
    {
        // Boundary conditions
        BCType xbc_min, xbc_max;
        BCType ybc_min, ybc_max;
        BCType zbc_min, zbc_max;
        // State of inflow boundaries by dimension, then lower and upper edge
        inflowState inflow[3][2];
        // Number of cells in each direction
        SAMS::T_indexType nx_global, ny_global, nz_global;
        // Multiplier for the time step
        SAMS::T_dataType dt_multiplier;
        // Maximum number of iterations; if nsteps < 0, run until t_end
        SAMS::T_indexType nsteps;
        SAMS::T_dataType t_end;
        // Domain limits
        SAMS::T_dataType x_min, x_max, y_min, y_max, z_min, z_max;
        // Grid stretching
        bool x_stretch, y_stretch, z_stretch;
        // Shock viscosity coefficients
        SAMS::T_dataType visc1, visc2;
        // Ratio of specific heat capacities
        SAMS::T_dataType gas_gamma;
        // Average mass of an ion in proton masses
        SAMS::T_dataType mf;
        // Resistive MHD options
        bool resistiveMHD;
        SAMS::T_dataType eta_background, j_max, eta0;
        // Remap kinetic energy correction
        bool rke;
        // Host versions of the variables and axes, and the number of cells on this rank. Only set while the initial conditions are set
        portableWrapper::portableArray<SAMS::T_dataType, 3, portableWrapper::arrayTags::host> rho, energy_electron, energy_ion;
        portableWrapper::portableArray<SAMS::T_dataType, 3, portableWrapper::arrayTags::host> vx, vy, vz;
        portableWrapper::portableArray<SAMS::T_dataType, 3, portableWrapper::arrayTags::host> bx, by, bz;
        portableWrapper::portableArray<SAMS::T_dataType, 1, portableWrapper::arrayTags::host> xc, yc, zc, xb, yb, zb;
        SAMS::T_indexType nx, ny, nz;
        // Time between snapshots
        SAMS::T_dataType dt_snapshots;
    };

    template<typename T_EOS = LARE::idealGas>
    class LARE3DInitialConditions{

        using bcFuncPtr = void(*)(typename LARE::LARE3D<T_EOS>::simulationData &, icData &, SAMS::timeState &time, int, SAMS::domain::edges);
        using bcFunction = void(typename LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);

        //Definitions for boundary condition functions. Each class of boundary condition is in its own file
        static bcFunction bx_bcs_other, by_bcs_other, bz_bcs_other, energy_ion_bcs_other, energy_electron_bcs_other, density_bcs_other;
        static bcFunction vx_bcs_other, vy_bcs_other, vz_bcs_other, remap_vx_bcs_other, remap_vy_bcs_other, remap_vz_bcs_other;
        static bcFunction bx_bcs_reflective, by_bcs_reflective, bz_bcs_reflective, energy_ion_bcs_reflective, energy_electron_bcs_reflective, density_bcs_reflective;
        static bcFunction vx_bcs_reflective, vy_bcs_reflective, vz_bcs_reflective, remap_vx_bcs_reflective, remap_vy_bcs_reflective, remap_vz_bcs_reflective;
        static bcFunction bx_bcs_outflow, by_bcs_outflow, bz_bcs_outflow, energy_ion_bcs_outflow, energy_electron_bcs_outflow, density_bcs_outflow;
        static bcFunction vx_bcs_outflow, vy_bcs_outflow, vz_bcs_outflow, remap_vx_bcs_outflow, remap_vy_bcs_outflow, remap_vz_bcs_outflow;
        static bcFunction bx_bcs_inflow, by_bcs_inflow, bz_bcs_inflow, energy_ion_bcs_inflow, energy_electron_bcs_inflow, density_bcs_inflow;
        static bcFunction vx_bcs_inflow, vy_bcs_inflow, vz_bcs_inflow, remap_vx_bcs_inflow, remap_vy_bcs_inflow, remap_vz_bcs_inflow;
        //dm has the same boundary conditions for every class of boundary condition
        static bcFunction dm_x_bcs, dm_y_bcs, dm_z_bcs;

        inline static constexpr auto nameType = SAMS::makeName("LARE3DInitialConditions") + T_EOS::name;

        /**
         * Read a boundary condition type from the deck. The type is given as the name of the BCType without BC_, in lower case
         * @param harnessRef SAMS harness, used to abort if the type is not recognised
         * @param kvs The key value store
         * @param path The path of the key
         * @param defaultValue The type to use if the key is not in the deck
         */
        static BCType getBCType(SAMS::harness &harnessRef, SAMS::KVS &kvs, const std::string &path, BCType defaultValue);

        /**
         * Convert a boundary condition type to its name in the deck, which is the name of the BCType without BC_, in lower case
         * @param type The boundary condition type
         */
        static std::string toLAREBCString(BCType type);

        public:

        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = nameType;

        /**
         * Initial conditions should not be timed
         */
        constexpr static bool timeSimulation = false;

        /**
         * The initial conditions' dataPack is icData
         */
        using dataPack = icData;

        /**
         * Set up control variables for LARE3D
         * @param harnessRef SAMS harness
         * @param data Initial conditions data
         */
        void control_variables(SAMS::harness &harnessRef, icData &data, SAMS::KVS &kvs);

        /**
         * The units that LARE3D runs in, either SI or normalised
         */
        static std::string units();

        /**
         * Set the initial conditions for LARE3D, and the state of inflow boundaries
         * @param harnessRef SAMS harness
         * @param data Initial conditions data, including host versions of the variables and axes
         */
        void initial_conditions(SAMS::harness &harnessRef, icData &data);

        /**
         * Shim to match the SAMS runner control function style
         * Copies the control variables, which control_variables set in defaultDeck, to LARE3D
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         * @param icd Initial conditions data
         */
        void controlVariables(SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, SAMS::KVS &kvs){
            // Periodic boundaries must be periodic at both edges
            if ((icd.xbc_min == BCType::BC_PERIODIC) != (icd.xbc_max == BCType::BC_PERIODIC)) harnessRef.abort("Inconsistent periodic boundaries in x", false);
            if ((icd.ybc_min == BCType::BC_PERIODIC) != (icd.ybc_max == BCType::BC_PERIODIC)) harnessRef.abort("Inconsistent periodic boundaries in y", false);
            if ((icd.zbc_min == BCType::BC_PERIODIC) != (icd.zbc_max == BCType::BC_PERIODIC)) harnessRef.abort("Inconsistent periodic boundaries in z", false);
            //The control variables go to LARE3D
            data.nx = icd.nx_global;
            data.ny = icd.ny_global;
            data.nz = icd.nz_global;
            data.dt_multiplier = icd.dt_multiplier;
            data.dt = 0.0;
            data.nsteps = icd.nsteps;
            data.t_end = icd.t_end;
            data.x_min = icd.x_min;
            data.x_max = icd.x_max;
            data.y_min = icd.y_min;
            data.y_max = icd.y_max;
            data.z_min = icd.z_min;
            data.z_max = icd.z_max;
            data.x_stretch = icd.x_stretch;
            data.y_stretch = icd.y_stretch;
            data.z_stretch = icd.z_stretch;
            data.visc1 = icd.visc1;
            data.visc2 = icd.visc2;
            data.gas_gamma = icd.gas_gamma;
            data.mf = icd.mf;
            data.resistiveMHD = icd.resistiveMHD;
            data.eta_background = icd.eta_background;
            data.j_max = icd.j_max;
            data.eta0 = icd.eta0;
            data.rke = icd.rke;
            //The end time and the domain that control sets go in the deck
            kvs.setContentPath(SAMS::deckKeys::endTime, data.t_end);
            kvs.setContentPath(SAMS::deckKeys::nx, data.nx);
            kvs.setContentPath(SAMS::deckKeys::ny, data.ny);
            kvs.setContentPath(SAMS::deckKeys::nz, data.nz);
            kvs.setContentPath(SAMS::deckKeys::xMin, data.x_min);
            kvs.setContentPath(SAMS::deckKeys::xMax, data.x_max);
            kvs.setContentPath(SAMS::deckKeys::yMin, data.y_min);
            kvs.setContentPath(SAMS::deckKeys::yMax, data.y_max);
            kvs.setContentPath(SAMS::deckKeys::zMin, data.z_min);
            kvs.setContentPath(SAMS::deckKeys::zMax, data.z_max);
        }

        /**
         * Interact with the runner before initialization
         * @param runnerRef The runner
         */
        template<typename T_runner>
        void runnerInteraction([[maybe_unused]] T_runner &runnerRef){
        }

        /**
         * Set the control variables, and the deck keys that the deck does not set. The control variables are set here
         * because the MPI decomposition, which needs the periodic boundaries, happens before controlVariables
         * @param harnessRef SAMS harness
         * @param icd Initial conditions data
         * @param kvs The key value store
         */
        void defaultDeck(SAMS::harness &harnessRef, icData &icd, SAMS::KVS &kvs){
            control_variables(harnessRef, icd, kvs);
            //The end of the run and the output, which the standard packages handle
            kvs.setContentPath(SAMS::deckKeys::endTime, icd.t_end);
            kvs.setContentPath(SAMS::deckKeys::stepLimit, icd.nsteps);
            SAMS::addDefaultSchedule(kvs, icd.dt_snapshots);
            kvs.setDefaultPath("/LARE/units", units());
            //Periodic in each direction whose lower boundary is periodic
            kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{icd.xbc_min == BCType::BC_PERIODIC,
                icd.ybc_min == BCType::BC_PERIODIC,
                icd.zbc_min == BCType::BC_PERIODIC});
        }

        /**
         * Put the boundary condition types back into the deck
         * @param icd Initial conditions data
         * @param kvs The key value store
         */
        void repopulateDeck(icData &icd, SAMS::KVS &kvs){
            kvs.setContentPath("/LARE/xbc_min", toLAREBCString(icd.xbc_min));
            kvs.setContentPath("/LARE/xbc_max", toLAREBCString(icd.xbc_max));
            kvs.setContentPath("/LARE/ybc_min", toLAREBCString(icd.ybc_min));
            kvs.setContentPath("/LARE/ybc_max", toLAREBCString(icd.ybc_max));
            kvs.setContentPath("/LARE/zbc_min", toLAREBCString(icd.zbc_min));
            kvs.setContentPath("/LARE/zbc_max", toLAREBCString(icd.zbc_max));
        }

        /**
         * Set the initial conditions for LARE3D
         * @param data LARE3D simulation data
         * @param icd Initial conditions data
         */
        void initialConditions(SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, icData &icd){
            portableWrapper::portableArrayManager mgr;

            //Host versions of the variables. Variables that are already on the host are used directly
            icd.rho = mgr.makeHostAvailable(data.rho);
            icd.energy_electron = mgr.makeHostAvailable(data.energy_electron);
            icd.energy_ion = mgr.makeHostAvailable(data.energy_ion);
            icd.vx = mgr.makeHostAvailable(data.vx);
            icd.vy = mgr.makeHostAvailable(data.vy);
            icd.vz = mgr.makeHostAvailable(data.vz);
            icd.bx = mgr.makeHostAvailable(data.bx);
            icd.by = mgr.makeHostAvailable(data.by);
            icd.bz = mgr.makeHostAvailable(data.bz);

            //Axes and numbers of cells on this rank
            auto &axisReg = harnessRef.axisRegistry;
            icd.xc = axisReg.getPPLocalAxis<SAMS::T_dataType, portableWrapper::arrayTags::host>("X", SAMS::staggerType::CENTRED);
            icd.yc = axisReg.getPPLocalAxis<SAMS::T_dataType, portableWrapper::arrayTags::host>("Y", SAMS::staggerType::CENTRED);
            icd.zc = axisReg.getPPLocalAxis<SAMS::T_dataType, portableWrapper::arrayTags::host>("Z", SAMS::staggerType::CENTRED);
            icd.xb = axisReg.getPPLocalAxis<SAMS::T_dataType, portableWrapper::arrayTags::host>("X", SAMS::staggerType::HALF_CELL);
            icd.yb = axisReg.getPPLocalAxis<SAMS::T_dataType, portableWrapper::arrayTags::host>("Y", SAMS::staggerType::HALF_CELL);
            icd.zb = axisReg.getPPLocalAxis<SAMS::T_dataType, portableWrapper::arrayTags::host>("Z", SAMS::staggerType::HALF_CELL);
            icd.nx = static_cast<SAMS::T_indexType>(axisReg.getLocalDomainElements("X", SAMS::staggerType::CENTRED));
            icd.ny = static_cast<SAMS::T_indexType>(axisReg.getLocalDomainElements("Y", SAMS::staggerType::CENTRED));
            icd.nz = static_cast<SAMS::T_indexType>(axisReg.getLocalDomainElements("Z", SAMS::staggerType::CENTRED));

            SAMS::cout << "Setting up initial conditions" << std::endl;
            initial_conditions(harnessRef, icd);

            //Copy the host versions back to the variables that are not on the host
            copyBack(mgr, data.rho, icd.rho);
            copyBack(mgr, data.energy_electron, icd.energy_electron);
            copyBack(mgr, data.energy_ion, icd.energy_ion);
            copyBack(mgr, data.vx, icd.vx);
            copyBack(mgr, data.vy, icd.vy);
            copyBack(mgr, data.vz, icd.vz);
            copyBack(mgr, data.bx, icd.bx);
            copyBack(mgr, data.by, icd.by);
            copyBack(mgr, data.bz, icd.bz);

            //The host versions are freed with the manager, so they are not kept
            icd.rho = {};
            icd.energy_electron = {};
            icd.energy_ion = {};
            icd.vx = {};
            icd.vy = {};
            icd.vz = {};
            icd.bx = {};
            icd.by = {};
            icd.bz = {};
        }

        /**
         * Copy the host version of a variable back to the variable, unless they are the same memory
         * @param mgr The manager of the host version
         * @param variable The variable
         * @param host The host version
         */
        template<typename T_variable, typename T_host>
        static void copyBack(portableWrapper::portableArrayManager &mgr, T_variable &variable, T_host &host){
            if (static_cast<void*>(variable.data()) != static_cast<void*>(host.data())) mgr.copyData(variable, host);
        }

        /**
         * Attach the boundary condition functions to a variable based on LARE3D BC types
         * @param varName Name of the variable
         * @param bcFuncs The boundary condition functions for other, reflective, outflow and inflow boundaries
         * @param harnessRef Reference to the SAMS harness
         * @param data LARE3D simulation data
         * @param icd Initial conditions data
         */
        void setBoundary(const std::string& varName, std::array<bcFuncPtr, 4> bcFuncs, SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, SAMS::timeState &timeStateRef);

        /**
         * Set the domain and boundary conditions for LARE3D
         * @details
         * Sets up the number of grid points and physical dimensions in each direction.
         * @param harness Reference to the SAMS harness
         * @param data LARE3D simulation data
         */
        void setDomain(SAMS::harness &harness, LARE::LARE3D<T_EOS>::simulationData &data)
        {
            auto &axisReg = harness.axisRegistry;
            axisReg.setDomain("X", data.nx, data.x_min, data.x_max);
            axisReg.setDomain("Y", data.ny, data.y_min, data.y_max);
            axisReg.setDomain("Z", data.nz, data.z_min, data.z_max);
        }

        /**
         * Set the boundary conditions for LARE3D
         * @details
         * Attaches boundary condition functions to each variable based on LARE3D BC types.
         * @param harness Reference to the SAMS harness
         * @param data LARE3D simulation data
         * @param icd Initial conditions data
         * @param timeStateRef Reference to the SAMS time state (used for time-dependent BCs)
         */
        void setBoundaryConditions(SAMS::harness &harness, LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, SAMS::timeState &timeStateRef);

        /**
         * Write the state of the inflow boundaries to restart dumps
         * @param writer The writer of the dump
         * @param level The outputs that the dump writes
         * @param icd Initial conditions data
         */
        template<typename T>
        void writeDumpVariables(SAMS::IO::writer<T> &writer, SAMS::IO::registryWriter<T> &, const SAMS::outputLevel &level, icData &icd){
            if (!level.isActive("restart")) return;
            for (int dimension = 0; dimension < 3; ++dimension){
                for (int edge = 0; edge < 2; ++edge){
                    for (const auto &[key, value] : inflowValues(icd.inflow[dimension][edge])){
                        writer.template setKeyValue<SAMS::T_dataType>(inflowKey(dimension, edge, key).c_str(), *value);
                    }
                }
            }
        }

        /**
         * Read the state of the inflow boundaries from a restart dump
         * @param reader The reader of the dump
         * @param icd Initial conditions data
         */
        template<typename T>
        void restart(SAMS::IO::reader<T> &reader, SAMS::IO::registryReader<T> &, icData &icd){
            for (int dimension = 0; dimension < 3; ++dimension){
                for (int edge = 0; edge < 2; ++edge){
                    for (const auto &[key, value] : inflowValues(icd.inflow[dimension][edge])){
                        *value = reader.template readKeyValue<SAMS::T_dataType>(inflowKey(dimension, edge, key).c_str());
                    }
                }
            }
        }

        private:

        /**
         * The name and the address of each value of an inflow state
         */
        static std::array<std::pair<const char*, SAMS::T_dataType*>, 9> inflowValues(inflowState &state){
            return {{{"rho", &state.rho}, {"energy_electron", &state.energy_electron}, {"energy_ion", &state.energy_ion},
                {"vx", &state.vx}, {"vy", &state.vy}, {"vz", &state.vz},
                {"bx", &state.bx}, {"by", &state.by}, {"bz", &state.bz}}};
        }

        /**
         * The key in restart dumps of a value of the inflow state of an edge
         */
        static std::string inflowKey(int dimension, int edge, const char *name){
            return std::string("LAREIC/inflow_") + "xyz"[dimension] + (edge == 0 ? "_min/" : "_max/") + name;
        }

    };

    #define EOS_DEF(value) extern template class LARE3DInitialConditions<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}//namespace LARE

#endif
