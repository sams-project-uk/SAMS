#include "lareic.h"
#include "boundary_helpers.h"

/**
 * Reflective boundary conditions. Everything is mirrored about the boundary, except the component of velocity normal
 * to it, which changes sign and is zero on the boundary
 */
namespace LARE
{

    using namespace boundaryHelpers;

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::bx_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 0) mirrorNode(data.bx, dimension, edge, cells(data, dimension));
        else mirrorCell(data.bx, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::by_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 1) mirrorNode(data.by, dimension, edge, cells(data, dimension));
        else mirrorCell(data.by, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::bz_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 2) mirrorNode(data.bz, dimension, edge, cells(data, dimension));
        else mirrorCell(data.bz, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::energy_ion_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        mirrorCell(data.energy_ion, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::energy_electron_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        mirrorCell(data.energy_electron, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::density_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        mirrorCell(data.rho, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vx_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 0) mirrorNodeNegated(data.vx, dimension, edge, cells(data, dimension));
        else mirrorNode(data.vx, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vy_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 1) mirrorNodeNegated(data.vy, dimension, edge, cells(data, dimension));
        else mirrorNode(data.vy, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::vz_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 2) mirrorNodeNegated(data.vz, dimension, edge, cells(data, dimension));
        else mirrorNode(data.vz, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vx_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 0) mirrorNodeNegated(data.vx1, dimension, edge, cells(data, dimension));
        else mirrorNode(data.vx1, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vy_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 1) mirrorNodeNegated(data.vy1, dimension, edge, cells(data, dimension));
        else mirrorNode(data.vy1, dimension, edge, cells(data, dimension));
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::remap_vz_bcs_reflective(LARE::LARE3D<T_EOS>::simulationData &data, [[maybe_unused]] icData &icd, [[maybe_unused]] SAMS::timeState &time, int dimension, SAMS::domain::edges edge)
    {
        if (dimension == 2) mirrorNodeNegated(data.vz1, dimension, edge, cells(data, dimension));
        else mirrorNode(data.vz1, dimension, edge, cells(data, dimension));
    }

    #include "./instantiate.h"
}
