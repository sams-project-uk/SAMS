#ifndef MULTIGRIDTEST_TAGS_H
#define MULTIGRIDTEST_TAGS_H
#include "runner.h"
#include "harness.h"

namespace SAMS
{
	/** @brief Tag for the electron species of a problem */
	struct electron
	{
		/** @brief The name of the species */
		inline static constexpr auto name = SAMS::makeName("electron");

		/** @brief Spitzer's parallel conductivity in SI units, times the Coulomb logarithm */
		static constexpr SAMS::T_dataType spitzerCoefficient = 1.84e-10;

		/** @brief The name of the variable that is evolved, which is the energy of the species */
		inline static constexpr auto field = SAMS::makeName("energy_") + name;

		static constexpr const char *fieldName()
		{
			return field.c_str();
		}
	};

	/** @brief Tag for the ion species of a problem */
	struct ion
	{
		/** @brief The name of the species */
		inline static constexpr auto name = SAMS::makeName("ion");

		/** @brief Braginskii's parallel ion conductivity in SI units, times the Coulomb logarithm. It is about 0.041 of the electron value */
		static constexpr SAMS::T_dataType spitzerCoefficient = 7.5e-12;

		/** @brief The name of the variable that is evolved, which is the energy of the species */
		inline static constexpr auto field = SAMS::makeName("energy_") + name;

		static constexpr const char *fieldName()
		{
			return field.c_str();
		}
	};
}

#endif // MULTIGRIDTEST_TAGS_H
