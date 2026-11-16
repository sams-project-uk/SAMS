#include "lareic.h"
#include "builtInBoundaryConditions.h"

namespace LARE{

    template<typename T_EOS>
    BCType LARE3DInitialConditions<T_EOS>::getBCType(SAMS::harness &harnessRef, SAMS::KVS &kvs, const std::string &path, BCType defaultValue){
        std::string type = kvs.getString(path, "");
        if (type == "") return defaultValue;
        if (type == "other") return BCType::BC_OTHER;
        if (type == "periodic") return BCType::BC_PERIODIC;
        if (type == "reflective") return BCType::BC_REFLECTIVE;
        if (type == "outflow") return BCType::BC_OUTFLOW;
        if (type == "inflow") return BCType::BC_INFLOW;
        if (type == "external") return BCType::BC_EXTERNAL;
        harnessRef.abort("Invalid " + path + ": " + type, false);
        return defaultValue;
    }

    template<typename T_EOS>
    std::string LARE3DInitialConditions<T_EOS>::toLAREBCString(BCType type){
        if (type == BCType::BC_PERIODIC) return "periodic";
        if (type == BCType::BC_REFLECTIVE) return "reflective";
        if (type == BCType::BC_OUTFLOW) return "outflow";
        if (type == BCType::BC_INFLOW) return "inflow";
        if (type == BCType::BC_EXTERNAL) return "external";
        return "other";
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::setBoundary(const std::string& varName, std::array<bcFuncPtr, 4> bcFuncs, SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, SAMS::timeState &timeStateRef){
        auto &varRegistry = harnessRef.variableRegistry;
        //The boundary condition for an edge, from the class of boundary condition on that edge
        auto bcPtr = [&](BCType type) -> std::shared_ptr<SAMS::boundaryConditions> {
            bcFuncPtr bcFunc = bcFuncs[0];
            if (type == BCType::BC_REFLECTIVE) bcFunc = bcFuncs[1];
            if (type == BCType::BC_OUTFLOW) bcFunc = bcFuncs[2];
            if (type == BCType::BC_INFLOW) bcFunc = bcFuncs[3];
            return std::make_shared<LAREBoundaryConditions<T_EOS>>(bcFunc, data, icd, timeStateRef);
        };
        SAMS::variableDef &varDef = varRegistry.getVariable(varName);
        std::shared_ptr<SAMS::boundaryConditions> periodicBC = std::make_shared<SAMS::simplePeriodicBC<SAMS::T_dataType, 3>>(varDef);

        //Periodic boundaries are applied to both edges
        if (icd.xbc_min == BCType::BC_PERIODIC){
            varDef.addBoundaryCondition(0, periodicBC);
        }
        if (icd.ybc_min == BCType::BC_PERIODIC){
            varDef.addBoundaryCondition(1, periodicBC);
        }
        if (icd.zbc_min == BCType::BC_PERIODIC){
            varDef.addBoundaryCondition(2, periodicBC);
        }

        if (icd.xbc_min != BCType::BC_EXTERNAL && icd.xbc_min != BCType::BC_PERIODIC){
            varRegistry.addBoundaryCondition(varName, 0, SAMS::domain::edges::lower, bcPtr(icd.xbc_min));
        }
        if (icd.xbc_max != BCType::BC_EXTERNAL && icd.xbc_max != BCType::BC_PERIODIC){
            varRegistry.addBoundaryCondition(varName, 0, SAMS::domain::edges::upper, bcPtr(icd.xbc_max));
        }
        if (icd.ybc_min != BCType::BC_EXTERNAL && icd.ybc_min != BCType::BC_PERIODIC){
            varRegistry.addBoundaryCondition(varName, 1, SAMS::domain::edges::lower, bcPtr(icd.ybc_min));
        }
        if (icd.ybc_max != BCType::BC_EXTERNAL && icd.ybc_max != BCType::BC_PERIODIC){
            varRegistry.addBoundaryCondition(varName, 1, SAMS::domain::edges::upper, bcPtr(icd.ybc_max));
        }
        if (icd.zbc_min != BCType::BC_EXTERNAL && icd.zbc_min != BCType::BC_PERIODIC){
            varRegistry.addBoundaryCondition(varName, 2, SAMS::domain::edges::lower, bcPtr(icd.zbc_min));
        }
        if (icd.zbc_max != BCType::BC_EXTERNAL && icd.zbc_max != BCType::BC_PERIODIC){
            varRegistry.addBoundaryCondition(varName, 2, SAMS::domain::edges::upper, bcPtr(icd.zbc_max));
        }
    }

    template<typename T_EOS>
    void LARE3DInitialConditions<T_EOS>::setBoundaryConditions(SAMS::harness &harness, LARE::LARE3D<T_EOS>::simulationData &data, icData &icd, SAMS::timeState &timeStateRef)
    {
        //Grab variables and set boundary condition functions
        setBoundary("bx", {bx_bcs_other, bx_bcs_reflective, bx_bcs_outflow, bx_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("by", {by_bcs_other, by_bcs_reflective, by_bcs_outflow, by_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("bz", {bz_bcs_other, bz_bcs_reflective, bz_bcs_outflow, bz_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("energy_ion", {energy_ion_bcs_other, energy_ion_bcs_reflective, energy_ion_bcs_outflow, energy_ion_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("energy_electron", {energy_electron_bcs_other, energy_electron_bcs_reflective, energy_electron_bcs_outflow, energy_electron_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("rho", {density_bcs_other, density_bcs_reflective, density_bcs_outflow, density_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("vx", {vx_bcs_other, vx_bcs_reflective, vx_bcs_outflow, vx_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("vy", {vy_bcs_other, vy_bcs_reflective, vy_bcs_outflow, vy_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("vz", {vz_bcs_other, vz_bcs_reflective, vz_bcs_outflow, vz_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("LARE/vx1", {remap_vx_bcs_other, remap_vx_bcs_reflective, remap_vx_bcs_outflow, remap_vx_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("LARE/vy1", {remap_vy_bcs_other, remap_vy_bcs_reflective, remap_vy_bcs_outflow, remap_vy_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("LARE/vz1", {remap_vz_bcs_other, remap_vz_bcs_reflective, remap_vz_bcs_outflow, remap_vz_bcs_inflow}, harness, data, icd, timeStateRef);
        setBoundary("LARE/dm", {dm_x_bcs, dm_x_bcs, dm_x_bcs, dm_x_bcs}, harness, data, icd, timeStateRef);
    }

    #include "./instantiate.h"
}//namespace LARE
