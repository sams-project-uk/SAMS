#include "lareic.h"

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
#include "lareic.h"

namespace LARE
{

    namespace pw = portableWrapper;
    using namespace LARE;

    /**
     * Apply boundary conditions to bx
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::bx_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.bx(-2, pw::Range(), pw::Range()),
                    data.bx(2, pw::Range(), pw::Range()));
                pw::assign(
                    data.bx(-1, pw::Range(), pw::Range()),
                    data.bx(1, pw::Range(), pw::Range()));
            }
        }

        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER && data.isxUB)
            {
                pw::assign(
                    data.bx(data.nx + 1, pw::Range(), pw::Range()),
                    data.bx(data.nx - 1, pw::Range(), pw::Range()));
                pw::assign(
                    data.bx(data.nx + 2, pw::Range(), pw::Range()),
                    data.bx(data.nx - 2, pw::Range(), pw::Range()));
            }
        }

        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER && data.isyLB)
            {
                pw::assign(
                    data.bx(pw::Range(), -1, pw::Range()),
                    data.bx(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.bx(pw::Range(), 0, pw::Range()),
                    data.bx(pw::Range(), 1, pw::Range()));
            }
        }

        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER && data.isyUB)
            {
                pw::assign(
                    data.bx(pw::Range(), data.ny + 1, pw::Range()),
                    data.bx(pw::Range(), data.ny, pw::Range()));
                pw::assign(
                    data.bx(pw::Range(), data.ny + 2, pw::Range()),
                    data.bx(pw::Range(), data.ny - 1, pw::Range()));
            }
        }

        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER && data.iszLB)
            {
                pw::assign(
                    data.bx(pw::Range(), pw::Range(), -1),
                    data.bx(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.bx(pw::Range(), pw::Range(), 0),
                    data.bx(pw::Range(), pw::Range(), 1));
            }
        }

        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER && data.iszUB)
            {
                pw::assign(
                    data.bx(pw::Range(), pw::Range(), data.nz + 1),
                    data.bx(pw::Range(), pw::Range(), data.nz));
                pw::assign(
                    data.bx(pw::Range(), pw::Range(), data.nz + 2),
                    data.bx(pw::Range(), pw::Range(), data.nz - 1));
            }
        }
    }

    /**
     * Apply boundary conditions to by
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::by_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.by(-1, pw::Range(), pw::Range()),
                    data.by(2, pw::Range(), pw::Range()));
                pw::assign(
                    data.by(0, pw::Range(), pw::Range()),
                    data.by(1, pw::Range(), pw::Range()));
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER && data.isxUB)
            {
                pw::assign(
                    data.by(data.nx + 1, pw::Range(), pw::Range()),
                    data.by(data.nx, pw::Range(), pw::Range()));
                pw::assign(
                    data.by(data.nx + 2, pw::Range(), pw::Range()),
                    data.by(data.nx - 1, pw::Range(), pw::Range()));
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER && data.isyLB)
            {
                pw::assign(
                    data.by(pw::Range(), -2, pw::Range()),
                    data.by(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.by(pw::Range(), -1, pw::Range()),
                    data.by(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER && data.isyUB)
            {
                pw::assign(
                    data.by(pw::Range(), data.ny + 1, pw::Range()),
                    data.by(pw::Range(), data.ny, pw::Range()));
                pw::assign(
                    data.by(pw::Range(), data.ny + 2, pw::Range()),
                    data.by(pw::Range(), data.ny - 1, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER && data.iszLB)
            {
                pw::assign(
                    data.by(pw::Range(), pw::Range(), -1),
                    data.by(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.by(pw::Range(), pw::Range(), 0),
                    data.by(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER && data.iszUB)
            {
                pw::assign(
                    data.by(pw::Range(), pw::Range(), data.nz + 1),
                    data.by(pw::Range(), pw::Range(), data.nz));
                pw::assign(
                    data.by(pw::Range(), pw::Range(), data.nz + 2),
                    data.by(pw::Range(), pw::Range(), data.nz - 1));
            }
        }
    }

    /**
     * Apply boundary conditions to bz
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::bz_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.bz(-1, pw::Range(), pw::Range()),
                    data.bz(2, pw::Range(), pw::Range()));
                pw::assign(
                    data.bz(0, pw::Range(), pw::Range()),
                    data.bz(1, pw::Range(), pw::Range()));
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER && data.isxUB)
            {
                pw::assign(
                    data.bz(data.nx + 1, pw::Range(), pw::Range()),
                    data.bz(data.nx, pw::Range(), pw::Range()));
                pw::assign(
                    data.bz(data.nx + 2, pw::Range(), pw::Range()),
                    data.bz(data.nx - 1, pw::Range(), pw::Range()));
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER && data.isyLB)
            {
                pw::assign(
                    data.bz(pw::Range(), -1, pw::Range()),
                    data.bz(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.bz(pw::Range(), 0, pw::Range()),
                    data.bz(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER && data.isyUB)
            {
                pw::assign(
                    data.bz(pw::Range(), data.ny + 1, pw::Range()),
                    data.bz(pw::Range(), data.ny, pw::Range()));
                pw::assign(
                    data.bz(pw::Range(), data.ny + 2, pw::Range()),
                    data.bz(pw::Range(), data.ny - 1, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER && data.iszLB)
            {
                pw::assign(
                    data.bz(pw::Range(), pw::Range(), -2),
                    data.bz(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.bz(pw::Range(), pw::Range(), -1),
                    data.bz(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER && data.iszUB)
            {
                pw::assign(
                    data.bz(pw::Range(), pw::Range(), data.nz + 1),
                    data.bz(pw::Range(), pw::Range(), data.nz));
                pw::assign(
                    data.bz(pw::Range(), pw::Range(), data.nz + 2),
                    data.bz(pw::Range(), pw::Range(), data.nz - 1));
            }
        }
    }

    /**
     * Apply boundary conditions to energy_ion
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::energy_ion_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_ion(-1, pw::Range(), pw::Range()),
                    data.energy_ion(2, pw::Range(), pw::Range()));
                pw::assign(
                    data.energy_ion(0, pw::Range(), pw::Range()),
                    data.energy_ion(1, pw::Range(), pw::Range()));
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if(data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_ion(data.nx + 1, pw::Range(), pw::Range()),
                    data.energy_ion(data.nx - 1, pw::Range(), pw::Range()));
                pw::assign(
                    data.energy_ion(data.nx + 2, pw::Range(), pw::Range()),
                    data.energy_ion(data.nx - 2, pw::Range(), pw::Range()));
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_ion(pw::Range(), -1, pw::Range()),
                    data.energy_ion(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.energy_ion(pw::Range(), 0, pw::Range()),
                    data.energy_ion(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_ion(pw::Range(), data.ny + 1, pw::Range()),
                    data.energy_ion(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.energy_ion(pw::Range(), data.ny + 2, pw::Range()),
                    data.energy_ion(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_ion(pw::Range(), pw::Range(), -1),
                    data.energy_ion(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.energy_ion(pw::Range(), pw::Range(), 0),
                    data.energy_ion(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_ion(pw::Range(), pw::Range(), data.nz + 1),
                    data.energy_ion(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.energy_ion(pw::Range(), pw::Range(), data.nz + 2),
                    data.energy_ion(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }


    /**
     * Apply boundary conditions to energy_electron
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::energy_electron_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_electron(-1, pw::Range(), pw::Range()),
                    data.energy_electron(2, pw::Range(), pw::Range()));
                pw::assign(
                    data.energy_electron(0, pw::Range(), pw::Range()),
                    data.energy_electron(1, pw::Range(), pw::Range()));
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if(data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_electron(data.nx + 1, pw::Range(), pw::Range()),
                    data.energy_electron(data.nx - 1, pw::Range(), pw::Range()));
                pw::assign(
                    data.energy_electron(data.nx + 2, pw::Range(), pw::Range()),
                    data.energy_electron(data.nx - 2, pw::Range(), pw::Range()));
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_electron(pw::Range(), -1, pw::Range()),
                    data.energy_electron(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.energy_electron(pw::Range(), 0, pw::Range()),
                    data.energy_electron(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_electron(pw::Range(), data.ny + 1, pw::Range()),
                    data.energy_electron(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.energy_electron(pw::Range(), data.ny + 2, pw::Range()),
                    data.energy_electron(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_electron(pw::Range(), pw::Range(), -1),
                    data.energy_electron(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.energy_electron(pw::Range(), pw::Range(), 0),
                    data.energy_electron(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.energy_electron(pw::Range(), pw::Range(), data.nz + 1),
                    data.energy_electron(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.energy_electron(pw::Range(), pw::Range(), data.nz + 2),
                    data.energy_electron(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }

    /**
     * Apply boundary conditions to density
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::density_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        // Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.rho(-1, pw::Range(), pw::Range()),
                    data.rho(2, pw::Range(), pw::Range()));
                pw::assign(
                    data.rho(0, pw::Range(), pw::Range()),
                    data.rho(1, pw::Range(), pw::Range()));
            }
        }

        // Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.rho(data.nx + 1, pw::Range(), pw::Range()),
                    data.rho(data.nx - 1, pw::Range(), pw::Range()));
                pw::assign(
                    data.rho(data.nx + 2, pw::Range(), pw::Range()),
                    data.rho(data.nx - 2, pw::Range(), pw::Range()));
            }
        }

        // Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.rho(pw::Range(), -1, pw::Range()),
                    data.rho(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.rho(pw::Range(), 0, pw::Range()),
                    data.rho(pw::Range(), 1, pw::Range()));
            }
        }

        // Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.rho(pw::Range(), data.ny + 1, pw::Range()),
                    data.rho(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.rho(pw::Range(), data.ny + 2, pw::Range()),
                    data.rho(pw::Range(), data.ny - 2, pw::Range()));
            }
        }

        // Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.rho(pw::Range(), pw::Range(), -1),
                    data.rho(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.rho(pw::Range(), pw::Range(), 0),
                    data.rho(pw::Range(), pw::Range(), 1));
            }
        }

        // Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.rho(pw::Range(), pw::Range(), data.nz + 1),
                    data.rho(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.rho(pw::Range(), pw::Range(), data.nz + 2),
                    data.rho(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }

    /**
     * Apply boundary conditions to vx
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::vx_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Continuous into each boundary except for X where assign zero velocity
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx(pw::Range(-2,0), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx(pw::Range(data.nx, data.nx + 2), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx(pw::Range(), -2, pw::Range()),
                    data.vx(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.vx(pw::Range(), -1, pw::Range()),
                    data.vx(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx(pw::Range(), data.ny + 1, pw::Range()),
                    data.vx(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.vx(pw::Range(), data.ny + 2, pw::Range()),
                    data.vx(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx(pw::Range(), pw::Range(), -2),
                    data.vx(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.vx(pw::Range(), pw::Range(), -1),
                    data.vx(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx(pw::Range(), pw::Range(), data.nz + 1),
                    data.vx(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.vx(pw::Range(), pw::Range(), data.nz + 2),
                    data.vx(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }

    /**
     * Apply boundary conditions to vx at the half step
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::remap_vx_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Continuous into each boundary except for X where assign zero velocity
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx1(pw::Range(-2,0), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx1(pw::Range(data.nx, data.nx + 2), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx1(pw::Range(), -2, pw::Range()),
                    data.vx1(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.vx1(pw::Range(), -1, pw::Range()),
                    data.vx1(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx1(pw::Range(), data.ny + 1, pw::Range()),
                    data.vx1(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.vx1(pw::Range(), data.ny + 2, pw::Range()),
                    data.vx1(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx1(pw::Range(), pw::Range(), -2),
                    data.vx1(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.vx1(pw::Range(), pw::Range(), -1),
                    data.vx1(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vx1(pw::Range(), pw::Range(), data.nz + 1),
                    data.vx1(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.vx1(pw::Range(), pw::Range(), data.nz + 2),
                    data.vx1(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }


   /**
     * Apply boundary conditions to vy
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::vy_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Continuous into each boundary except for X where assign zero velocity
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy(pw::Range(-2,0), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy(pw::Range(data.nx, data.nx + 2), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy(pw::Range(), -2, pw::Range()),
                    data.vy(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.vy(pw::Range(), -1, pw::Range()),
                    data.vy(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy(pw::Range(), data.ny + 1, pw::Range()),
                    data.vy(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.vy(pw::Range(), data.ny + 2, pw::Range()),
                    data.vy(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy(pw::Range(), pw::Range(), -2),
                    data.vy(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.vy(pw::Range(), pw::Range(), -1),
                    data.vy(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy(pw::Range(), pw::Range(), data.nz + 1),
                    data.vy(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.vy(pw::Range(), pw::Range(), data.nz + 2),
                    data.vy(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }

    /**
     * Apply boundary conditions to vy at the half time step
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::remap_vy_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Continuous into each boundary except for X where assign zero velocity
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy1(pw::Range(-2,0), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy1(pw::Range(data.nx, data.nx + 2), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy1(pw::Range(), -2, pw::Range()),
                    data.vy1(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.vy1(pw::Range(), -1, pw::Range()),
                    data.vy1(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy1(pw::Range(), data.ny + 1, pw::Range()),
                    data.vy1(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.vy1(pw::Range(), data.ny + 2, pw::Range()),
                    data.vy1(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy1(pw::Range(), pw::Range(), -2),
                    data.vy1(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.vy1(pw::Range(), pw::Range(), -1),
                    data.vy1(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vy1(pw::Range(), pw::Range(), data.nz + 1),
                    data.vy1(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.vy1(pw::Range(), pw::Range(), data.nz + 2),
                    data.vy1(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }

   /**
     * Apply boundary conditions to vz
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::vz_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Continuous into each boundary except for X where assign zero velocity
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz(pw::Range(-2,0), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz(pw::Range(data.nx, data.nx + 2), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz(pw::Range(), -2, pw::Range()),
                    data.vz(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.vz(pw::Range(), -1, pw::Range()),
                    data.vz(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz(pw::Range(), data.ny + 1, pw::Range()),
                    data.vz(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.vz(pw::Range(), data.ny + 2, pw::Range()),
                    data.vz(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz(pw::Range(), pw::Range(), -2),
                    data.vz(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.vz(pw::Range(), pw::Range(), -1),
                    data.vz(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz(pw::Range(), pw::Range(), data.nz + 1),
                    data.vz(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.vz(pw::Range(), pw::Range(), data.nz + 2),
                    data.vz(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }

    /**
     * Apply boundary conditions to vz at the half time step
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
     void LARE3DInitialConditions::remap_vz_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        //Continuous into each boundary except for X where assign zero velocity
        //Lower X BC
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            if (data.xbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz1(pw::Range(-2,0), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Upper X BC
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            if (data.xbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz1(pw::Range(data.nx, data.nx + 2), pw::Range(), pw::Range()),
                    0.0);
            }
        }
        //Lower Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            if (data.ybc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz1(pw::Range(), -2, pw::Range()),
                    data.vz1(pw::Range(), 2, pw::Range()));
                pw::assign(
                    data.vz1(pw::Range(), -1, pw::Range()),
                    data.vz1(pw::Range(), 1, pw::Range()));
            }
        }
        //Upper Y BC
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            if (data.ybc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz1(pw::Range(), data.ny + 1, pw::Range()),
                    data.vz1(pw::Range(), data.ny - 1, pw::Range()));
                pw::assign(
                    data.vz1(pw::Range(), data.ny + 2, pw::Range()),
                    data.vz1(pw::Range(), data.ny - 2, pw::Range()));
            }
        }
        //Lower Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            if (data.zbc_min == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz1(pw::Range(), pw::Range(), -2),
                    data.vz1(pw::Range(), pw::Range(), 2));
                pw::assign(
                    data.vz1(pw::Range(), pw::Range(), -1),
                    data.vz1(pw::Range(), pw::Range(), 1));
            }
        }
        //Upper Z BC
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            if (data.zbc_max == BCType::BC_OTHER)
            {
                pw::assign(
                    data.vz1(pw::Range(), pw::Range(), data.nz + 1),
                    data.vz1(pw::Range(), pw::Range(), data.nz - 1));
                pw::assign(
                    data.vz1(pw::Range(), pw::Range(), data.nz + 2),
                    data.vz1(pw::Range(), pw::Range(), data.nz - 2));
            }
        }
    }

     void LARE3DInitialConditions::dm_x_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        // Continous flux at x-max boundary
        if (dimension == 0 && edge == SAMS::domain::edges::upper)
        {
            pw::assign(
                data.dm(data.nx + 1, pw::Range(), pw::Range()),
                data.dm(data.nx, pw::Range(), pw::Range()));
        }

        // Continous flux at x-min boundary
        if (dimension == 0 && edge == SAMS::domain::edges::lower)
        {
            pw::assign(
                data.dm(-1, pw::Range(), pw::Range()),
                data.dm(0, pw::Range(), pw::Range()));
        }
       //Only need X boundaries for dm_x 
    }

    void LARE3DInitialConditions::dm_y_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        // Continous flux at y-max boundary
        if (dimension == 1 && edge == SAMS::domain::edges::upper)
        {
            pw::assign(
                data.dm(pw::Range(), data.ny + 1, pw::Range()),
                data.dm(pw::Range(), data.ny, pw::Range()));
        }
        // Continous flux at y-min boundary
        if (dimension == 1 && edge == SAMS::domain::edges::lower)
        {
            pw::assign(
                data.dm(pw::Range(), -1, pw::Range()),
                data.dm(pw::Range(), 0, pw::Range()));
        }

    }

    void LARE3DInitialConditions::dm_z_bcs([[maybe_unused]] simulationData &data, [[maybe_unused]] SAMS::timeState &time, [[maybe_unused]] int dimension, [[maybe_unused]] SAMS::domain::edges edge)
    {
        // Continous flux at z-max boundary
        if (dimension == 2 && edge == SAMS::domain::edges::upper)
        {
            pw::assign(
                data.dm(pw::Range(), pw::Range(), data.nz + 1),
                data.dm(pw::Range(), pw::Range(), data.nz));
        }

        // Continous flux at z-min boundary
        if (dimension == 2 && edge == SAMS::domain::edges::lower)
        {
            pw::assign(
                data.dm(pw::Range(), pw::Range(), -1),
                data.dm(pw::Range(), pw::Range(), 0));
        }
    }
}
