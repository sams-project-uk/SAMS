#if !defined(GENERAL_REMAP_H)
#define GENERAL_REMAP_H

#include "typedefs_neutral.h"
#include "shared_data_neutral.h"
#include "remapData_neutral.h"

namespace LARE_neutral {
    enum class AxisName {
        X = 0,
        Y = 1,
        Z = 2
    };

    static constexpr bool USE_GENERAL_REMAP = true;

    template <AxisName axis>
    void remap(LARE3D_neutral& lare, simulationData& data, remapData& remap_data);

    template <AxisName v_comp, AxisName b_comp>
    void v_b_flux(simulationData& data, remapData& remap_data);

    template <AxisName axis>
    void mass_flux(simulationData& data, remapData& remap_data);

    template <AxisName axis, auto mPtr>
    void mom_flux(simulationData& data, remapData& remap_data);

    template <AxisName axis, auto mPtr>
    void energy_flux(simulationData& data, remapData& remap_data);
}
#endif
