#ifndef SOUNDWAVE_H
#define SOUNDWAVE_H

#include "harness.h"
#include "runner.h"
#include "shared_data.h"
#include "builtInBoundaryConditions.h"

namespace examples
{

    namespace pw = portableWrapper;

    /**
     * A sound wave pulse launched from a Gaussian spot in the middle of the lower x boundary. The same problem as the SoundWave deck, for speed comparisons
     */
    template<typename T_EOS = LARE::idealGas>
    class SoundWave
    {
    public:
        struct soundWaveParams{
            SAMS::T_dataType mach;
            SAMS::T_dataType t_peak;
            SAMS::T_dataType t_width;
            SAMS::T_dataType c_s;
        };

    private:
        void attachBoundaryConditions(const std::string &varName, SAMS::harness &harness, soundWaveParams &problemParams, SAMS::timeState &tData);

        inline static constexpr auto nameType = SAMS::makeName("SoundWave") + T_EOS::name;

    public:
        constexpr static std::string_view name = nameType;

        constexpr static bool timeSimulation = false;

        using dataPack = soundWaveParams;

        void defaultDeck(SAMS::KVS &kvs);

        void controlVariables(LARE::LARE3D<T_EOS>::simulationData &data, soundWaveParams &problemParams, SAMS::KVS &kvs);

        void setDomain(SAMS::harness &harness, SAMS::KVS &kvs);

        void setBoundaryConditions(SAMS::harness &harness, soundWaveParams &problemParams, SAMS::timeState &tData);

        void initialConditions(LARE::LARE3D<T_EOS>::simulationData &data);
    };

    #define EOS_DEF(value) extern template class SoundWave<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
#endif // SOUNDWAVE_H
