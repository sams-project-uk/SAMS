/*
 *    Copyright 2025 SAMS Team
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#include <cstdint>
#include <cassert>
#include <string>
#include "constants.h"
#include "include/parallelWrapper.h"
#include "remapData.h"
#include "typedefs.h"

//Possible geometry types
enum class geometryType {
    Cartesian = 0,
    Cylindrical = 1,
    Spherical = 2,
};

//Boundary condition types
enum class BCType {
    BC_OTHER = 0, // Other boundary condition
    BC_PERIODIC = 1, // Periodic boundary condition
    BC_REFLECTIVE = 2, // Reflective boundary condition
    BC_OUTFLOW = 3, // Outflow boundary condition
    BC_INFLOW = 4, // Inflow boundary condition
};

/**
 * This is a struct that hold all simulation data
 * Putting this is a struct allows separate simulations in the same code
 */
struct simulationData{

    bool configured = false; // Indicates if the simulation data has been configured

    T_dataType none_zero = std::numeric_limits<T_dataType>::epsilon(); // Smallest non-zero value for T_dataTyp
    T_dataType largest_number = std::numeric_limits<T_dataType>::max(); // Largest number for T_dataType

    // Simulation parameters
    T_dataType dt, dtr, dt_multiplier;
    T_dataType time; //Current simulation time
    size_t step; //Current simulation step
    int64_t nsteps; // Maximum number of steps, if < 0 run until t_end
    T_dataType t_end; // End time of the simulation

    // Domain parameters
    T_indexType nx, ny, nz; //Could be unsigned but when comparing signed and unsigned, unsigned wins
    T_dataType x_min, x_max, length_x, dx;
    T_dataType y_min, y_max, length_y, dy;
    T_dataType z_min, z_max, length_z, dz;
    bool x_stretch, y_stretch, z_stretch;
    geometryType geometry;

    // Boundary conditions
    BCType xbc_min, xbc_max;
    BCType ybc_min, ybc_max;
    BCType zbc_min, zbc_max;

    //Physics selectors
    bool resistiveMHD; // Resistive MHD
    bool rke; // Remap phase kinetic energy correction

    //Shock viscosity coefficients
    T_dataType visc1; // Linear shock viscosity coefficient
    T_dataType visc2; // Quadratic shock viscosity coefficient
    T_dataType visc2_norm; // Normalized quadratic shock viscosity coefficient

    //Physical constants
    T_dataType gas_gamma; // Ratio of specific heats
    T_dataType j_max; // Maximum current
    T_dataType eta0; // Limited resisivity (applied J>j_max)
    T_dataType eta_background; // Background resisivity
    T_dataType mf; // Average mass of an ion in proton masses
    T_dataType mu0_si; // Vacuum permeability in SI units

    //IO control
    T_dataType dt_snapshots; // Time between snapshots

    //Physical arrays
    volumeArray energy_electron; // Electron specific internal energy
    volumeArray energy_ion; // Ion specific
    volumeArray p_visc; // Viscous pressure
    volumeArray rho; // Density
    volumeArray vx; // X-velocity
    volumeArray vy; // Y-velocity
    volumeArray vz; // Z-velocity
    volumeArray vx1; // Half timestep X-velocity
    volumeArray vy1; // Half timestep Y-velocity
    volumeArray vz1; // Half timestep Z-velocity
    volumeArray bx; // X-magnetic field
    volumeArray by; // Y-magnetic field
    volumeArray bz; // Z-magnetic field
    volumeArray eta; // Resisivity
    volumeArray dxab;
    volumeArray dyab;
    volumeArray dzab;
    volumeArray dxac;
    volumeArray dyac;
    volumeArray dzac;
    volumeArray cv; // Control volume
    volumeArray cv1; // Half timestep control volume
    volumeArray cvc;
    lineArray xc; //Cell center X-coordinates
    lineArray yc; //Cell center Y-coordinates
    lineArray zc; //Cell center Z-coordinates
    lineArray xb; //Cell boundary X-coordinates
    lineArray yb; //Cell boundary Y-coordinates
    lineArray zb; //Cell boundary Z-coordinates
    lineArray xb_global; //Global cell boundary X-coordinates
    lineArray yb_global; //Global cell boundary Y-coordinates
    lineArray zb_global; //Global cell boundary Z-coordinates
    lineArray dxc; //Cell center X-dx
    lineArray dyc; //Cell center Y-dy
    lineArray dzc; //Cell center Z-dz
    lineArray dxb; //Cell boundary X-dx
    lineArray dyb; //Cell boundary Y-dy
    lineArray dzb; //Cell boundary Z-dz
    lineArray hy;
    planeArray hz;
    lineArray hyc;
    planeArray hzc;
    planeArray hz1;
    planeArray hz2;
    lineArray grav_r;
    lineArray grav_z;
    volumeArray delta_ke; //Remap kinetic energy correction
    volumeArray x,y,z;
    volumeArray xp, yp, zp;
};

class simulation{
public:

    /**
     * Portable array manager for handling memory allocation and deallocation
     */
    portableWrapper::portableArrayManager manager;

    /**
     * Boundary conditions for magnetic field
     */
    void bfield_bcs(simulationData &data);
    /**
     * Boundary conditions for specific internal energy
     */
    void energy_bcs(simulationData &data);
    /**
     * Boundary conditions for density
     */
    void density_bcs(simulationData &data);
    /**
     * Boundary conditions for velocity
     */
    void velocity_bcs(simulationData &data);
    /**
     * Boundary conditions for remap velocity
     * Normally the same as velocity_bcs, but can be different for some simulations
     */
    void remap_v_bcs(simulationData &data);

    /**
     * Boundary conditions for remap phase X mass flux
     */
    void dm_x_bcs(simulationData &data, remapData &remapData);

    /**
     * Boundary conditions for remap phase Y mass flux
     */
    void dm_y_bcs(simulationData &data, remapData &remapData);

    /**
     * Boundary conditions for remap phase Z mass flux
     */
    void dm_z_bcs(simulationData &data, remapData &remapData);

    /** 
     * Function to perform the X sweep remap
     */
    void remap_x(simulationData &data, remapData &remapData);

    /**
     * Function to perform the Y sweep remap
     */
    void remap_y(simulationData &data, remapData &remapData);

    /**
     * Function to perform the Z sweep remap
     */
    void remap_z(simulationData &data, remapData &remapData);

public:

    /**
     * Register variables with the portable array manager.
     */
    void registerVars();

    /**
     * Allocate the simulation data arrays
     * @param data Simulation data struct
     * @param nx Number of cells in the x-direction
     * @param ny Number of cells in the y-direction
     * @param nz Number of cells in the z-direction
     * This function allocates the arrays in the simulationData struct.
     * It uses the portableArrayManager to handle the memory allocation and deallocation.
     */
    void allocate(simulationData &data);

    /**
     * Setup the simulation data
     * @param data Simulation data struct
     * This function sets up the simulation data, including the geometry, boundary conditions,
     * and other parameters. It also calculates the cell sizes and initializes the arrays.
     */
    void controlvariables(simulationData &data);
    /**
     * Setup the grid for the simulation
     * @param data Simulation data struct
     * This function sets up the grid for the simulation, including the cell sizes and coordinates. Automatically creates for the specified geometry.
     */
    void grid(simulationData &data);
    /**
     * Setup the initial conditions for the simulation
     * @param data Simulation data struct
     * This function sets up the initial conditions for the simulation, including the initial values of the physical variables.
     */
    void initial_conditions(simulationData &data);

    /**
     * Call all the boundary condition functions
     * @param data Simulation data struct
     * This function calls all the boundary condition functions for the simulation.
     * It is called at the end of each time step to apply the boundary conditions.
     */
    void boundary_conditions(simulationData &data);

    /**
     * Lagrangian step for the simulation
     * @param data Simulation data struct
     * This function performs a Lagrangian step for the simulation
     */
    void lagrangian_step(simulationData &data);

    /**
     * Calculate the resistivity eta based on current density
     * @param data Simulation data struct
     */
    void eta_calc(simulationData &data);

    /**
     * Core remap control function
     */
    void eulerian_remap(simulationData &data);

    /**
     * Function to add back the kinetic energy correction
     */
    void energy_correction(simulationData &data);

    /**
     * Function to output data to disk
     */
    void output(simulationData &data);
};

#endif // SHARED_DATA_H
