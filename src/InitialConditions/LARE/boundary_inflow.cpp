#include "lareic.h"
#include "boundary_helpers.h"

/**
 * Inflow boundary conditions. The ghost cells, and the boundary nodes of node centred dimensions, hold the inflow
 * state of the edge, which is set in initial_conditions
 */
namespace LARE
{

    using namespace boundaryHelpers;

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::bx_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 0) fillNode(data.bx, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).bx);
        else fillCell(data.bx, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).bx);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::by_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 1) fillNode(data.by, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).by);
        else fillCell(data.by, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).by);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::bz_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 2) fillNode(data.bz, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).bz);
        else fillCell(data.bz, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).bz);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::energy_ion_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillCell(data.energy_ion, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).energy_ion);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::energy_electron_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillCell(data.energy_electron, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).energy_electron);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::density_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillCell(data.rho, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).rho);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vx_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillNode(data.vx, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).vx);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vy_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillNode(data.vy, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).vy);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vz_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillNode(data.vz, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).vz);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vx_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillNode(data.vx1, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).vx);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vy_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillNode(data.vy1, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).vy);
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vz_bcs_inflow(LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        fillNode(data.vz1, dimension, edge, cells(data, dimension), inflow(icd, dimension, edge).vz);
    }

    #include "./instantiate.h"
}
