
/*
 *    Copyright 2025 SAMS Team
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <limits>

namespace LARE
{

  // These are the real SI physical constants

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

  inline constexpr int BC_PERIODIC = 1;
  inline constexpr int BC_OPEN = 2;
  inline constexpr int BC_SLIP = 3;
  inline constexpr int BC_DRIVEN = 4;
  inline constexpr int BC_COPY = 5;
  inline constexpr int BC_OTHER = 6;

  // Geometry constants
  inline constexpr int c_geometry_null = 0;
  inline constexpr int c_geometry_cartesian = 1;
  inline constexpr int c_geometry_cylindrical = 2;
  inline constexpr int c_geometry_spherical = 3;

}

#endif // CONSTANTS_H
