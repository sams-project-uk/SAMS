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
#include "shared_data.h"
#include "variableRegistry.h"

void simulation::boundary_conditions(simulationData &data)
{
  bfield_bcs(data);
  energy_bcs(data);
  density_bcs(data);
  velocity_bcs(data);
  portableWrapper::fence(); // Ensure all operations are complete before returning
}

void simulation::bfield_bcs(simulationData &data)
{
    SAMS::getvariableRegistry().haloExchange("bx");
    SAMS::getvariableRegistry().haloExchange("by");
    SAMS::getvariableRegistry().haloExchange("bz");
    if (data.xbc_min == BCType::BC_OTHER && data.isxLB){
        portableWrapper::assign(
            data.bx(-2, portableWrapper::Range(), portableWrapper::Range()), 
            data.bx(2, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(-1, -1), portableWrapper::Range(), portableWrapper::Range()), 
            data.bx(portableWrapper::Range(1, 1), portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(-1, -1), portableWrapper::Range(), portableWrapper::Range()), 
            data.by(portableWrapper::Range(2, 2), portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(0, 0), portableWrapper::Range(), portableWrapper::Range()), 
            data.by(portableWrapper::Range(1, 1), portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(-1, -1), portableWrapper::Range(), portableWrapper::Range()), 
            data.bz(portableWrapper::Range(2, 2), portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(0, 0), portableWrapper::Range(), portableWrapper::Range()), 
            data.bz(portableWrapper::Range(1, 1), portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.xbc_max == BCType::BC_OTHER && data.isxUB){
        portableWrapper::assign(
            data.bx(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.bx(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bx(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.bx(data.nx - 2, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.by(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.by(data.nx, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.by(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.by(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.bz(data.nx, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.bz(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.ybc_min == BCType::BC_OTHER && data.isyLB){
        portableWrapper::assign(
            data.by(portableWrapper::Range(), -2, portableWrapper::Range()), 
            data.by(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.by(portableWrapper::Range(), 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.bx(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), 0, portableWrapper::Range()), 
            data.bx(portableWrapper::Range(), 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.bz(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), 0, portableWrapper::Range()), 
            data.bz(portableWrapper::Range(), 1, portableWrapper::Range())
        );
    }

    if (data.ybc_max == BCType::BC_OTHER && data.isyUB){
        portableWrapper::assign(
            data.by(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.by(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.by(portableWrapper::Range(), data.ny - 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.bx(portableWrapper::Range(), data.ny, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.bx(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.bz(portableWrapper::Range(), data.ny, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.bz(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
    }

    if (data.zbc_min == BCType::BC_OTHER && data.iszLB){
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), portableWrapper::Range(), -2), 
            data.bz(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.bz(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.by(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(), portableWrapper::Range(), 0), 
            data.by(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.bx(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), portableWrapper::Range(), 0), 
            data.bx(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
    }

    if (data.zbc_max == BCType::BC_OTHER && data.iszUB){
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.bz(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
        portableWrapper::assign(
            data.bz(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.bz(portableWrapper::Range(), portableWrapper::Range(), data.nz - 2)
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.by(portableWrapper::Range(), portableWrapper::Range(), data.nz)
        );
        portableWrapper::assign(
            data.by(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.by(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.bx(portableWrapper::Range(), portableWrapper::Range(), data.nz)
        );
        portableWrapper::assign(
            data.bx(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.bx(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
    }
}

void simulation::energy_bcs(simulationData &data)
{
    SAMS::getvariableRegistry().haloExchange("energy_electron");
    SAMS::getvariableRegistry().haloExchange("energy_ion");
    if (data.xbc_min == BCType::BC_OTHER && data.isxLB){
        portableWrapper::assign(
            data.energy_electron(-1, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_electron(2, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_electron(0, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_electron(1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(-1, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_ion(2, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(0, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_ion(1, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.xbc_max == BCType::BC_OTHER && data.isxUB){
        portableWrapper::assign(
            data.energy_electron(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_electron(data.nx, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_electron(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_electron(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_ion(data.nx, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.energy_ion(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.ybc_min == BCType::BC_OTHER && data.isyLB){
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.energy_electron(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), 0, portableWrapper::Range()), 
            data.energy_electron(portableWrapper::Range(), 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.energy_ion(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), 0, portableWrapper::Range()), 
            data.energy_ion(portableWrapper::Range(), 1, portableWrapper::Range())
        );
    }

    if (data.ybc_max == BCType::BC_OTHER && data.isyUB){
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.energy_electron(portableWrapper::Range(), data.ny, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.energy_electron(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.energy_ion(portableWrapper::Range(), data.ny, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.energy_ion(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
    }
    if (data.zbc_min == BCType::BC_OTHER && data.iszLB){
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), 0), 
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), 0), 
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
    }
    if (data.zbc_max == BCType::BC_OTHER && data.iszUB){
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), data.nz)
        );
        portableWrapper::assign(
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.energy_electron(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), data.nz)
        );
        portableWrapper::assign(
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.energy_ion(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
    }
}

void simulation::density_bcs(simulationData &data)
{
    SAMS::getvariableRegistry().haloExchange("rho");
    if (data.xbc_min == BCType::BC_OTHER && data.isxLB){
        portableWrapper::assign(
            data.rho(-1, portableWrapper::Range(), portableWrapper::Range()), 
            data.rho(2, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.rho(0, portableWrapper::Range(), portableWrapper::Range()), 
            data.rho(1, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.xbc_max == BCType::BC_OTHER && data.isxUB){
        portableWrapper::assign(
            data.rho(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.rho(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.rho(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.rho(data.nx - 2, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.ybc_min == BCType::BC_OTHER && data.isyLB){
        portableWrapper::assign(
            data.rho(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.rho(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.rho(portableWrapper::Range(), 0, portableWrapper::Range()), 
            data.rho(portableWrapper::Range(), 1, portableWrapper::Range())
        );
    }

    if (data.ybc_max == BCType::BC_OTHER && data.isyUB){
        portableWrapper::assign(
            data.rho(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.rho(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.rho(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.rho(portableWrapper::Range(), data.ny - 2, portableWrapper::Range())
        );
    }

    if (data.zbc_min == BCType::BC_OTHER && data.iszLB){
        portableWrapper::assign(
            data.rho(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.rho(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.rho(portableWrapper::Range(), portableWrapper::Range(), 0), 
            data.rho(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
    }
}

void simulation::velocity_bcs(simulationData &data)
{
    SAMS::getvariableRegistry().haloExchange("vx");
    SAMS::getvariableRegistry().haloExchange("vy");
    SAMS::getvariableRegistry().haloExchange("vz");
    //Other boundaries clamp v=0
    if (data.xbc_min == BCType::BC_OTHER && data.isxLB){
        portableWrapper::assign(
            data.vx(portableWrapper::Range(-2,0), portableWrapper::Range(), portableWrapper::Range()), 
            0.0
        );
        portableWrapper::assign(
            data.vy(-1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy(1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vy(-2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy(2, portableWrapper::Range(), portableWrapper::Range())
        );

        portableWrapper::assign(
            data.vz(-1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz(1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz(-2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz(2, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.xbc_max == BCType::BC_OTHER && data.isxUB){
        portableWrapper::assign(
            data.vx(portableWrapper::Range(data.nx, data.nx + 2), portableWrapper::Range(), portableWrapper::Range()), 
            0.0
        );

        portableWrapper::assign(
            data.vy(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy(data.nx-1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vy(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy(data.nx-2, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz(data.nx, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz(data.nx-1, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.ybc_min == BCType::BC_OTHER && data.isyLB){
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.vx(portableWrapper::Range(), 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), -2, portableWrapper::Range()), 
            data.vx(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vy(portableWrapper::Range(), portableWrapper::Range(-2,0), portableWrapper::Range()), 
            0.0
        );
        portableWrapper::assign(
            data.vz(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.vz(portableWrapper::Range(), 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz(portableWrapper::Range(), -2, portableWrapper::Range()), 
            data.vz(portableWrapper::Range(), 2, portableWrapper::Range())
        );
    }

    if (data.ybc_max == BCType::BC_OTHER && data.isyUB){
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.vx(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.vx(portableWrapper::Range(), data.ny - 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vy(portableWrapper::Range(), portableWrapper::Range(data.ny, data.ny + 2), portableWrapper::Range()), 
            0.0
        );
        portableWrapper::assign(
            data.vz(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.vz(portableWrapper::Range(), data.ny, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.vz(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
    }

    if (data.zbc_min == BCType::BC_OTHER && data.iszLB){
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), portableWrapper::Range(), -2), 
            data.vx(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.vx(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
        portableWrapper::assign(
            data.vy(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.vy(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
        portableWrapper::assign(
            data.vy(portableWrapper::Range(), portableWrapper::Range(), -2), 
            data.vy(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.vz(portableWrapper::Range(), portableWrapper::Range(), portableWrapper::Range(-2,0)), 
            0.0
        );
    }

    if (data.zbc_max == BCType::BC_OTHER && data.iszUB){
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.vx(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
        portableWrapper::assign(
            data.vy(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.vy(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
        portableWrapper::assign(
            data.vx(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.vx(portableWrapper::Range(), portableWrapper::Range(), data.nz - 2)
        );
        portableWrapper::assign(
            data.vy(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.vy(portableWrapper::Range(), portableWrapper::Range(), data.nz - 2)
        );
        portableWrapper::assign(
            data.vz(portableWrapper::Range(), portableWrapper::Range(), portableWrapper::Range(data.nz, data.nz + 2)), 
            0.0
        );
    }
}

void simulation::remap_v_bcs(simulationData &data)
{
    SAMS::getvariableRegistry().haloExchange("vx1");
    SAMS::getvariableRegistry().haloExchange("vy1");
    SAMS::getvariableRegistry().haloExchange("vz1");
    //Other boundaries clamp v=0
    if (data.xbc_min == BCType::BC_OTHER && data.isxLB){
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(-2,0), portableWrapper::Range(), portableWrapper::Range()), 
            0.0
        );

        portableWrapper::assign(
            data.vy1(-2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy1(2, portableWrapper::Range(), portableWrapper::Range())
        );

        portableWrapper::assign(
            data.vy1(-1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy1(1, portableWrapper::Range(), portableWrapper::Range())
        );

        portableWrapper::assign(
            data.vz1(-2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz1(2, portableWrapper::Range(), portableWrapper::Range())
        );

        portableWrapper::assign(
            data.vz1(-1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz1(1, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.xbc_max == BCType::BC_OTHER && data.isxUB){
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(data.nx, data.nx + 2), portableWrapper::Range(), portableWrapper::Range()), 
            0.0
        );
        portableWrapper::assign(
            data.vy1(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy1(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vy1(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vy1(data.nx - 2, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz1(data.nx + 1, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz1(data.nx - 1, portableWrapper::Range(), portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz1(data.nx + 2, portableWrapper::Range(), portableWrapper::Range()), 
            data.vz1(data.nx - 2, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    if (data.ybc_min == BCType::BC_OTHER && data.isyLB){
        portableWrapper::assign(
            data.vy1(portableWrapper::Range(), portableWrapper::Range(-2,0), portableWrapper::Range()), 
            0.0
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), -2, portableWrapper::Range()), 
            data.vx1(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.vx1(portableWrapper::Range(), 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz1(portableWrapper::Range(), -2, portableWrapper::Range()), 
            data.vz1(portableWrapper::Range(), 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz1(portableWrapper::Range(), -1, portableWrapper::Range()), 
            data.vz1(portableWrapper::Range(), 1, portableWrapper::Range())
        );
    }

    if (data.ybc_max == BCType::BC_OTHER && data.isyUB){
        portableWrapper::assign(
            data.vy1(portableWrapper::Range(), portableWrapper::Range(data.ny, data.ny + 2), portableWrapper::Range()), 
            0.0
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.vx1(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.vx1(portableWrapper::Range(), data.ny - 2, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz1(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            data.vz1(portableWrapper::Range(), data.ny - 1, portableWrapper::Range())
        );
        portableWrapper::assign(
            data.vz1(portableWrapper::Range(), data.ny + 2, portableWrapper::Range()), 
            data.vz1(portableWrapper::Range(), data.ny - 2, portableWrapper::Range())
        );
    }

    if (data.zbc_min == BCType::BC_OTHER && data.iszLB){
        portableWrapper::assign(
            data.vz1(portableWrapper::Range(), portableWrapper::Range(), portableWrapper::Range(-2,0)), 
            0.0
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), -2), 
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
        portableWrapper::assign(
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), -2), 
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), 2)
        );
        portableWrapper::assign(
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), -1), 
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), 1)
        );
    }

    if (data.zbc_max == BCType::BC_OTHER && data.iszUB){
        portableWrapper::assign(
            data.vz1(portableWrapper::Range(), portableWrapper::Range(), portableWrapper::Range(data.nz, data.nz + 2)), 
            0.0
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
        portableWrapper::assign(
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.vx1(portableWrapper::Range(), portableWrapper::Range(), data.nz - 2)
        );
        portableWrapper::assign(
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), data.nz - 1)
        );
        portableWrapper::assign(
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), data.nz + 2), 
            data.vy1(portableWrapper::Range(), portableWrapper::Range(), data.nz - 2)
        );
    }
}

void simulation::dm_x_bcs(simulationData &data, remapData &remap_data)
{
    SAMS::getvariableRegistry().haloExchange("dm");
    //Continous flux at x-max boundary
    if (data.isxUB){
        portableWrapper::assign(
            remap_data.dm(data.nx+1, portableWrapper::Range(), portableWrapper::Range()), 
            remap_data.dm(data.nx, portableWrapper::Range(), portableWrapper::Range())
        );
    }

    //Continous flux at x-min boundary
    if (data.isxLB){
        portableWrapper::assign(
            remap_data.dm(-1, portableWrapper::Range(), portableWrapper::Range()), 
            remap_data.dm(0, portableWrapper::Range(), portableWrapper::Range())
        );
    }
    portableWrapper::fence();
}

void simulation::dm_y_bcs(simulationData &data, remapData &remap_data)
{
    SAMS::getvariableRegistry().haloExchange("dm");
    //Continous flux at y-max boundary
    if (data.isyUB){
        portableWrapper::assign(
            remap_data.dm(portableWrapper::Range(), data.ny + 1, portableWrapper::Range()), 
            remap_data.dm(portableWrapper::Range(), data.ny, portableWrapper::Range())
        );
    }
    //Continous flux at y-min boundary
    if (data.isyLB){
        portableWrapper::assign(
            remap_data.dm(portableWrapper::Range(), -1, portableWrapper::Range()), 
            remap_data.dm(portableWrapper::Range(), 0, portableWrapper::Range())
        );
    }

    portableWrapper::fence();
}

void simulation::dm_z_bcs(simulationData &data, remapData &remap_data)
{
    
    //Continous flux at z-max boundary
    if (data.iszUB){
        portableWrapper::assign(
            remap_data.dm(portableWrapper::Range(), portableWrapper::Range(), data.nz + 1), 
            remap_data.dm(portableWrapper::Range(), portableWrapper::Range(), data.nz)
        );
    }
 
    //Continous flux at z-min boundary
    if (data.iszLB){
        portableWrapper::assign(
            remap_data.dm(portableWrapper::Range(), portableWrapper::Range(), -1), 
            remap_data.dm(portableWrapper::Range(), portableWrapper::Range(), 0)
        );
    }
    portableWrapper::fence();
}