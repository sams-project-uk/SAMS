
#include "HLLEOrszagTang.h"
#include "../../Operations/OutputControl/defaultSchedule.h"
#include "uniformDomain.h"
#include "io/writers.h"
#include "io/readers.h"
#include <iomanip>
#include <limits>
#include <sstream>

namespace examples
{

    /**
     * Internal function to attach boundary conditions to a variable
     * @param varName Name of the variable to attach boundary conditions to
     * @param harness SAMS harness
     */
    template<typename T_EOS>
    void HLLEOrszagTang<T_EOS>::attachBoundaryConditions(const std::string &varName, SAMS::harness &harness)
    {
        SAMS::variableDef &varDef = harness.variableRegistry.getVariable(varName);
        // Periodic boundaries in all directions
        std::shared_ptr<SAMS::boundaryConditions> periodicBC = std::make_shared<SAMS::simplePeriodicBC<SAMS::T_dataType, 3>>(varDef);
        varDef.addBoundaryCondition(0, periodicBC);
        varDef.addBoundaryCondition(1, periodicBC);
        varDef.addBoundaryCondition(2, periodicBC);
    }

    /**
     * Set the deck keys that the deck does not set
     * @param kvs The key value store
     */
    template<typename T_EOS>
    void HLLEOrszagTang<T_EOS>::defaultDeck(SAMS::KVS &kvs)
    {
        // Periodic in all directions
        kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{true, true, true});
        // End time of the simulation
        kvs.setDefaultPath(SAMS::deckKeys::endTime, 3.0);
        // Domain
        kvs.setDefaultPath(SAMS::deckKeys::nx, 128);
        kvs.setDefaultPath(SAMS::deckKeys::ny, 128);
        kvs.setDefaultPath(SAMS::deckKeys::nz, 128);
        kvs.setDefaultPath(SAMS::deckKeys::xMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::xMax, 2.0 * M_PI);
        kvs.setDefaultPath(SAMS::deckKeys::yMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::yMax, 2.0 * M_PI);
        kvs.setDefaultPath(SAMS::deckKeys::zMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::zMax, 2.0 * M_PI);
        // Write the default collection 10 times over the run if the deck has no output schedules
        SAMS::addDefaultSchedule(kvs, kvs.getReal(SAMS::deckKeys::endTime) / 10.0);
    }

    /**
     * Set up the simulation domain
     * @param harness SAMS harness
     * @param kvs The key value store
     */
    template<typename T_EOS>
    void HLLEOrszagTang<T_EOS>::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
    {
        SAMS::setUniformDomain(harness, kvs);
    }

    /**
     * Set boundary conditions for the simulation
     * @param harness SAMS harness
     */
    template<typename T_EOS>
    void HLLEOrszagTang<T_EOS>::setBoundaryConditions(SAMS::harness &harness)
    {
        // Grab variables and set boundary condition functions
        attachBoundaryConditions("rho", harness);
        attachBoundaryConditions("mx", harness);
        attachBoundaryConditions("my", harness);
        attachBoundaryConditions("mz", harness);
        attachBoundaryConditions("energy", harness);
        attachBoundaryConditions("bx", harness);
        attachBoundaryConditions("by", harness);
        attachBoundaryConditions("bz", harness);
    }

    /**
     * Initialize the Orszag-Tang initial conditions. All of the variables are cell centred
     * @param harnessRef SAMS harness
     * @param data HLLE3D simulation data
     */
    template<typename T_EOS>
    void HLLEOrszagTang<T_EOS>::initialConditions(SAMS::harness &harnessRef, typename HLLE::HLLE3D<T_EOS>::simulationData &data)
    {
        pw::portableArray<SAMS::T_dataType, 1> xc, yc;
        auto &axisRegistry = harnessRef.axisRegistry;
        axisRegistry.fillPPLocalAxis("X", xc, SAMS::staggerType::CENTRED);
        axisRegistry.fillPPLocalAxis("Y", yc, SAMS::staggerType::CENTRED);

        const HLLE::conservedVariables U = data.U;
        const SAMS::T_dataType gas_gamma = data.gas_gamma;

        pw::applyKernel(
            LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz) {
                using T_dataType = SAMS::T_dataType;
                T_dataType pressure = gas_gamma;
                T_dataType rho = gas_gamma * gas_gamma;
                T_dataType vx = -std::sin(yc(iy));
                T_dataType vy = std::sin(xc(ix));
                T_dataType bx = -std::sin(yc(iy));
                T_dataType by = std::sin(2.0 * xc(ix));

                U.rho(ix, iy, iz) = rho;
                U.mx(ix, iy, iz) = rho * vx;
                U.my(ix, iy, iz) = rho * vy;
                U.mz(ix, iy, iz) = 0.0;
                U.bx(ix, iy, iz) = bx;
                U.by(ix, iy, iz) = by;
                U.bz(ix, iy, iz) = 0.0;
                // Total energy density
                U.energy(ix, iy, iz) = pressure / (gas_gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy) + 0.5 * (bx * bx + by * by);
            },
            data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
    }

    #define EOS_DEF(value) template class HLLEOrszagTang<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
