#ifndef LARE_TYPES_H
#define LARE_TYPES_H

#include "harness.h"

namespace LARE
{

    using T_dataType = SAMS::T_dataType;
    using T_sizeType = SAMS::T_sizeType;
    using T_indexType = SAMS::T_indexType;

    using volumeArray = portableWrapper::acceleratedArray<T_dataType, 3>;
    using hostVolumeArray = portableWrapper::hostArray<T_dataType, 3>;
    using planeArray = portableWrapper::acceleratedArray<T_dataType, 2>;
    using hostPlaneArray = portableWrapper::hostArray<T_dataType, 2>;
    using lineArray = portableWrapper::acceleratedArray<T_dataType, 1>;
    using hostLineArray = portableWrapper::hostArray<T_dataType, 1>;

    inline constexpr double mu0_si = static_cast<double>(4.0e-7) * M_PI; // N/A^2 (exact)

    // Boltzmann's Constant
    inline constexpr double kb_si = static_cast<double>(1.3806488e-23); // J/K (+/- 1.3e-29)

    // Mass of hydrogen ion
    inline constexpr double mh_si = static_cast<double>(1.672621898e-27); // kg (+/- 7.4e-35)

    // Mass of electron
    inline constexpr double me_si = static_cast<double>(9.10938291e-31); // kg (+/- 4e-38)

    // Planck's constant
    inline constexpr double hp_si = static_cast<double>(6.62606957e-34); // J s (+/- 2.9e-41)

    inline constexpr double dt_multiplier = static_cast<double>(0.8);

    inline constexpr double none_zero = std::numeric_limits<double>::min();
    inline constexpr double largest_number = std::numeric_limits<double>::max();
    inline constexpr double third = static_cast<double>(1.0) / static_cast<double>(3.0);
    inline constexpr double sixth = static_cast<double>(1.0) / static_cast<double>(6.0);

    // Possible geometry types
    enum class geometryType
    {
        Cartesian = 0,
        Cylindrical = 1,
        Spherical = 2,
    };

    // Boundary condition types
    enum class BCType
    {
        BC_OTHER = 0,      // Other boundary condition
        BC_PERIODIC = 1,   // Periodic boundary condition
        BC_REFLECTIVE = 2, // Reflective boundary condition
        BC_OUTFLOW = 3,    // Outflow boundary condition
        BC_INFLOW = 4,     // Inflow boundary condition
        BC_EXTERNAL = 5    // External boundary condition (do not apply Lare Style BCs)
    };
}
#endif