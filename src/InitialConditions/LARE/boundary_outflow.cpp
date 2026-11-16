#include "lareic.h"
#include "boundary_helpers.h"

/**
 * Outflow boundary conditions. Everything has zero gradient across the boundary
 */
namespace LARE
{

    using namespace boundaryHelpers;

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::bx_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 0) copyNode(data.bx, dimension, edge, cells(data, dimension));
        else copyCell(data.bx, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::by_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 1) copyNode(data.by, dimension, edge, cells(data, dimension));
        else copyCell(data.by, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::bz_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 2) copyNode(data.bz, dimension, edge, cells(data, dimension));
        else copyCell(data.bz, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::energy_ion_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyCell(data.energy_ion, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::energy_electron_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyCell(data.energy_electron, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::density_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyCell(data.rho, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vx_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyNode(data.vx, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vy_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyNode(data.vy, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vz_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyNode(data.vz, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vx_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyNode(data.vx1, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vy_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyNode(data.vy1, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vz_bcs_outflow(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        copyNode(data.vz1, dimension, edge, cells(data, dimension));
    }

    #include "./instantiate.h"
}
