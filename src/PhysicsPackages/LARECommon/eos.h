#ifndef EOS_H
#define EOS_H 

#include "types.h"

namespace LARE
{
    template<typename T, int VAR_TYPE>
    struct eosVariable{
        T value;
        explicit eosVariable(T value_in) : value(value_in) {}
        operator T&() { return value; }
    };

    inline constexpr int densityVar = 0;
    inline constexpr int energyVar = 1;
    inline constexpr int pressureVar = 2;
    inline constexpr int temperatureVar = 3;
    inline constexpr int ionizationVar = 4;
    inline constexpr int indexVar = 5; // For EOS that require index inputs

    using eosDensity = eosVariable<T_dataType, densityVar>;
    using eosEnergy = eosVariable<T_dataType, energyVar>;
    using eosPressure = eosVariable<T_dataType, pressureVar>;
    using eosTemperature = eosVariable<T_dataType, temperatureVar>;
    using eosIonization = eosVariable<T_dataType, ionizationVar>;
    using eosIndex = eosVariable<T_indexType, indexVar>;

    struct idealGas
    {
        static constexpr SAMS::constexprName name = SAMS::constexprName("");
        T_dataType gamma;
        FUNCTORMETHODPREFIX INLINE T_dataType getPressure(eosDensity density, eosEnergy energy) const
        {
            return (gamma - 1.0) * density * energy;
        }
        FUNCTORMETHODPREFIX INLINE T_dataType getSoundSpeedSquared(eosDensity density, eosEnergy energy) const
        {
            return gamma * (gamma - 1.0) * energy;
        }
        FUNCTORMETHODPREFIX INLINE T_dataType getEnergy(eosTemperature temperature) const
        {
            return temperature / (gamma - 1.0);
        }
        void setGamma(T_dataType gamma_in)
        {
            gamma = gamma_in;
        }

    };

    //X Macro for energy density based EOS types. This can be used to generate code for all EOS types that use density and energy as inputs, such as ideal gas, ionization, etc.
    #define EOS_DENSITY_ENERGY \
        EOS_DEF(LARE::idealGas)
}

#endif // EOS_H