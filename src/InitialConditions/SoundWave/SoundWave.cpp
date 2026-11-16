#include "SoundWave.h"
#include "../../Operations/OutputControl/defaultSchedule.h"
#include "uniformDomain.h"
#include <cmath>

namespace examples
{

    /**
     * Sets the ghost cells of a velocity to a pulse that is Gaussian in y about 0.5 with width 0.05 and in time about t_peak with width t_width
     */
    template<typename T>
    class soundWaveDriveBC : public SAMS::singleVariableBC<T, 3>
    {
        pw::portableArray<T, 1> y;
        SAMS::timeState *tData;
        T amplitude, t_peak, t_width;
    public:
        soundWaveDriveBC(const SAMS::variableDef &varDef, SAMS::harness &harness, SAMS::timeState &tData, T amplitude, T t_peak, T t_width)
            : SAMS::singleVariableBC<T, 3>(varDef), tData(&tData), amplitude(amplitude), t_peak(t_peak), t_width(t_width)
        {
            const SAMS::dimension &dim = varDef.getDimension(1);
            harness.axisRegistry.fillPPLocalAxis(dim.axisName, y, dim.stagger);
        }

        void apply(int dimension, SAMS::domain::edges edge) override
        {
            const int edgeIndex = (edge == SAMS::domain::edges::lower) ? 0 : 1;
            const auto &ranges = this->boundaryRanges[dimension * 2 + edgeIndex];
            const T timeFactor = (tData->time - t_peak) / t_width;
            const T peak = amplitude * std::exp(-timeFactor * timeFactor);
            pw::portableArray<T, 3> variable = this->variable;
            pw::portableArray<T, 1> yc = y;
            pw::applyKernel(
                LAMBDA(SAMS::T_indexType ix, SAMS::T_indexType iy, SAMS::T_indexType iz)
                {
                    const T yFactor = (yc(iy) - 0.5) / 0.05;
                    variable(ix, iy, iz) = peak * std::exp(-yFactor * yFactor);
                },
                std::get<0>(ranges), std::get<1>(ranges), std::get<2>(ranges));
        }
    };

    template<typename T_EOS>
    void SoundWave<T_EOS>::attachBoundaryConditions(const std::string &varName, SAMS::harness &harness, soundWaveParams &problemParams, SAMS::timeState &tData)
    {
        SAMS::variableDef &varDef = harness.variableRegistry.getVariable(varName);
        //Periodic in y and z
        varDef.addBoundaryCondition(1, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        varDef.addBoundaryCondition(2, SAMS::simplePeriodicBC<SAMS::T_dataType, 3>(varDef));
        //Zero gradient in x, except that the x velocity is driven at the lower boundary
        if (varName == "vx" || varName == "LARE/vx1") {
            varDef.addBoundaryCondition(0, SAMS::domain::edges::lower, soundWaveDriveBC<SAMS::T_dataType>(varDef, harness, tData,
                problemParams.mach * problemParams.c_s, problemParams.t_peak, problemParams.t_width));
            varDef.addBoundaryCondition(0, SAMS::domain::edges::upper, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        } else {
            varDef.addBoundaryCondition(0, SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>(varDef));
        }
    }

    template<typename T_EOS>
    void SoundWave<T_EOS>::defaultDeck(SAMS::KVS &kvs)
    {
        kvs.setDefaultPath("/LARE/units", "normalised");
        kvs.setDefaultPath("/LARE/gas_gamma", 5.0 / 3.0);
        //Periodic in y and z
        kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{false, true, true});
        //Domain. Z has cells of the same size as X
        kvs.setDefaultPath(SAMS::deckKeys::nx, 256);
        kvs.setDefaultPath(SAMS::deckKeys::ny, 256);
        kvs.setDefaultPath(SAMS::deckKeys::nz, 2);
        kvs.setDefaultPath(SAMS::deckKeys::xMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::xMax, 1.0);
        kvs.setDefaultPath(SAMS::deckKeys::yMin, 0.0);
        kvs.setDefaultPath(SAMS::deckKeys::yMax, 1.0);
        kvs.setDefaultPath(SAMS::deckKeys::zMin, 0.0);
        const SAMS::T_dataType width = kvs.getReal(SAMS::deckKeys::xMax) - kvs.getReal(SAMS::deckKeys::xMin);
        kvs.setDefaultPath(SAMS::deckKeys::zMax, width * kvs.getInt(SAMS::deckKeys::nz) / kvs.getInt(SAMS::deckKeys::nx));
        //Write the default collection 20 times over the run if the deck has no output schedules
        const SAMS::T_dataType gamma = kvs.getReal("/LARE/gas_gamma");
        //Ends when the peak of the pulse reaches the upper x boundary
        kvs.setDefaultPath(SAMS::deckKeys::endTime, kvs.getReal("/SoundWave/t_peak", 0.1) + 1.0 / std::sqrt(gamma * (gamma - 1.0)));
        SAMS::addDefaultSchedule(kvs, kvs.getReal(SAMS::deckKeys::endTime) / 20.0);
    }

    template<typename T_EOS>
    void SoundWave<T_EOS>::controlVariables(LARE::LARE3D<T_EOS>::simulationData &data, soundWaveParams &problemParams, SAMS::KVS &kvs)
    {
        problemParams.mach = kvs.getReal("/SoundWave/mach", 0.5);
        problemParams.t_peak = kvs.getReal("/SoundWave/t_peak", 0.1);
        problemParams.t_width = kvs.getReal("/SoundWave/t_width", 0.03);
        //Total specific internal energy is 1
        const SAMS::T_dataType gamma = kvs.getReal("/LARE/gas_gamma");
        problemParams.c_s = std::sqrt(gamma * (gamma - 1.0));

        //Ends when the peak of the pulse reaches the upper x boundary
        data.t_end = kvs.getReal(SAMS::deckKeys::endTime, problemParams.t_peak + 1.0 / problemParams.c_s);
    }

    template<typename T_EOS>
    void SoundWave<T_EOS>::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
    {
        SAMS::setUniformDomain(harness, kvs);
    }

    template<typename T_EOS>
    void SoundWave<T_EOS>::setBoundaryConditions(SAMS::harness &harness, soundWaveParams &problemParams, SAMS::timeState &tData)
    {
        for (const char *varName : {"rho", "energy_electron", "energy_ion", "vx", "vy", "vz", "bx", "by", "bz", "LARE/vx1", "LARE/vy1", "LARE/vz1", "LARE/dm"}){
            attachBoundaryConditions(varName, harness, problemParams, tData);
        }
    }

    template<typename T_EOS>
    void SoundWave<T_EOS>::initialConditions(LARE::LARE3D<T_EOS>::simulationData &data)
    {
        pw::assign(data.rho, 1.0);
        pw::assign(data.energy_electron, 0.5);
        pw::assign(data.energy_ion, 0.5);
        pw::assign(data.vx, 0.0);
        pw::assign(data.vy, 0.0);
        pw::assign(data.vz, 0.0);
        pw::assign(data.bx, 0.0);
        pw::assign(data.by, 0.0);
        pw::assign(data.bz, 0.0);
    }

    #define EOS_DEF(value) template class SoundWave<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
