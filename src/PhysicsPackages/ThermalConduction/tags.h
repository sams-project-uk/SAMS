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
		inline static constexpr auto name = SAMS::constexprName("electron");

		/** @brief The name of the variable that is evolved, which is the energy of the species */
		inline static constexpr auto field = SAMS::constexprName("energy_") + name;

		static constexpr const char *fieldName()
		{
			return field.c_str();
		}
	};

	/** @brief Tag for the ion species of a problem */
	struct ion
	{
		/** @brief The name of the species */
		inline static constexpr auto name = SAMS::constexprName("ion");

		/** @brief The name of the variable that is evolved, which is the energy of the species */
		inline static constexpr auto field = SAMS::constexprName("energy_") + name;

		static constexpr const char *fieldName()
		{
			return field.c_str();
		}
	};
}

#endif // MULTIGRIDTEST_TAGS_H
