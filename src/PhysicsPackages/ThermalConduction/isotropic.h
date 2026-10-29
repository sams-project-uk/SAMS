#ifndef MULTIGRIDTEST_ISOTROPIC_H
#define MULTIGRIDTEST_ISOTROPIC_H
#include "runner.h"
#include "harness.h"
#include <vector>
#include "LARE/shared_data.h"

namespace SAMS
{
	namespace pw = portableWrapper;

	/**
	 * @brief Isotropic heat conduction with a constant conductivity
	 *
	 * It has the same interface as braginskii, but its stencil has only the cell and its six face neighbours.
	 */
	template <typename T_species, typename T_eos = LARE::idealGas>
	struct isotropic
	{
		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;
		using hostLineArray = portableWrapper::hostArray<SAMS::T_dataType, 1>;

		using eosType = T_eos;
		using species = T_species;
		using lareData = typename LARE::LARE3D<T_eos>::simulationData;
		/** @brief The constants of the problem, which are in a data pack */
		struct parameters
		{
			/** @brief The conductivity */
			double kappa = 0.0;
		};

		using dataPack = SAMS::dataPacks::multiPack<parameters, lareData>;
		using packRefs = SAMS::packRefsOf_t<dataPack>;

		/** @brief The name of the temperature variable */
		inline static constexpr auto temperatureName = SAMS::constexprName("temperature_") + T_species::name + T_eos::name;

		/** @brief The conductance of the x edge between cells i and i + 1 */
		DEVICEPREFIX double faceX(const parameters &pr, const lineArray &dx, const lineArray &dy, const lineArray &dz, size_t i, size_t j, size_t k) const
		{
			return pr.kappa * dy(j) * dz(k) / (0.5 * (dx(i) + dx(i + 1)));
		}

		/** @brief The conductance of the y edge between cells j and j + 1 */
		DEVICEPREFIX double faceY(const parameters &pr, const lineArray &dx, const lineArray &dy, const lineArray &dz, size_t i, size_t j, size_t k) const
		{
			return pr.kappa * dx(i) * dz(k) / (0.5 * (dy(j) + dy(j + 1)));
		}

		/** @brief The conductance of the z edge between cells k and k + 1 */
		DEVICEPREFIX double faceZ(const parameters &pr, const lineArray &dx, const lineArray &dy, const lineArray &dz, size_t i, size_t j, size_t k) const
		{
			return pr.kappa * dx(i) * dy(j) / (0.5 * (dz(k) + dz(k + 1)));
		}

		/** @brief Set a coarse cell array to the mean of the fine cells it covers */
		template <typename T_level>
		void averageCells(const volumeArray &fineArray, const volumeArray &coarseArray, const T_level &fineLevel, const T_level &l) const
		{
			const volumeArray fine = fineArray, coarse = coarseArray;
			const size_t fx = l.coarsenX ? 2 : 1, fy = l.coarsenY ? 2 : 1, fz = l.coarsenZ ? 2 : 1;
			// Offset of the first coarse cell along the fine cells
			const size_t ox = l.coarsenX ? static_cast<size_t>(2 * (l.firstX - 1) - (fineLevel.firstX - 1)) : 0;
			const size_t oy = l.coarsenY ? static_cast<size_t>(2 * (l.firstY - 1) - (fineLevel.firstY - 1)) : 0;
			const size_t oz = l.coarsenZ ? static_cast<size_t>(2 * (l.firstZ - 1) - (fineLevel.firstZ - 1)) : 0;
			auto kernel = LAMBDA(size_t i, size_t j, size_t k)
			{
				double sum = 0.0;
				for (size_t a = 0; a < fx; ++a)
					for (size_t b = 0; b < fy; ++b)
						for (size_t c = 0; c < fz; ++c)
							sum += fine(fx * (i - 1) + 1 + a + ox, fy * (j - 1) + 1 + b + oy, fz * (k - 1) + 1 + c + oz);
				coarse(i, j, k) = sum / (fx * fy * fz);
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
		}

		/** @brief Set beta from the density and heat capacity, and average it onto the coarse levels */
		template <typename T_level>
		void setBeta(std::vector<T_level> &levels, const lareData &lare) const
		{
			T_level &l = levels[0];
			const auto &eos = lare.eos;
			const volumeArray density = rho, factor = l.data.beta;
			auto kernel = LAMBDA(size_t i, size_t j, size_t k)
			{
				factor(i, j, k) = density(i, j, k) * eos.getCV();
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
			for (size_t n = 1; n < levels.size(); ++n)
			{
				averageCells(levels[n - 1].data.beta, levels[n].data.beta, levels[n - 1], levels[n]);
			}
		}

		/** @brief Set the conductances on every level */
		template <typename T_level>
		void setConductances(std::vector<T_level> &levels, const parameters pr) const
		{
			const isotropic p = *this;
			for (size_t n = 0; n < levels.size(); ++n)
			{
				T_level &l = levels[n];
				const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
				const volumeArray cx = l.data.conductance.cx, cy = l.data.conductance.cy, cz = l.data.conductance.cz;

				auto conductanceX = LAMBDA(size_t i, size_t j, size_t k)
				{
					cx(i, j, k) = p.faceX(pr, dx, dy, dz, i, j, k);
				};
				pw::applyKernel(conductanceX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceY = LAMBDA(size_t i, size_t j, size_t k)
				{
					cy(i, j, k) = p.faceY(pr, dx, dy, dz, i, j, k);
				};
				pw::applyKernel(conductanceY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceZ = LAMBDA(size_t i, size_t j, size_t k)
				{
					cz(i, j, k) = p.faceZ(pr, dx, dy, dz, i, j, k);
				};
				pw::applyKernel(conductanceZ, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zbLocalDomainRange);
			}
		}

		/** @brief Density */
		volumeArray rho;

		/** @brief The name of the problem, which includes the species */
		inline static constexpr auto name = SAMS::constexprName("isotropic_") + T_species::name + T_eos::name;

		/** @brief The name that the arrays and axes of the problem are registered under, which the species share */
		inline static constexpr auto memoryName = SAMS::constexprName("isotropic") + T_eos::name;

		/** @brief The name of the variable that is evolved */
		static constexpr const char *fieldName()
		{
			return temperatureName.c_str();
		}

		/** @brief The conductances of the edges. cx(i) is between cells i and i + 1 */
		struct conductanceData
		{
			volumeArray cx, cy, cz;
		};

		/** @brief The arrays for one level */
		struct levelData
		{
			conductanceData conductance;
			/** @brief The prefactor of the time derivative, per unit volume */
			volumeArray beta;
		};

		/** @brief The number of entries of the stencil: the cell and its six face neighbours */
		static constexpr int stencilSize = 7;

		/** @brief The offset of each entry in the stencil, which includes the {0, 0, 0} centre element, in any order */
		DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
		{
			const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
			di = table[s][0];
			dj = table[s][1];
			dk = table[s][2];
		}

		/** @brief The operator of a level, for use in kernels */
		struct stencil
		{
			volumeArray cx, cy, cz;
			volumeArray betaArray;

			/** @brief The prefactor of the time derivative */
			DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
			{
				return betaArray(i, j, k);
			}

			/**
			 * @brief The weights of the fluxes out of a cell through its faces on the high side
			 *
			 * The entries are found from their positions relative to the cell, so at<1, 0, 0> is the cell at i + 1.
			 */
			DEVICEPREFIX parabolic::fluxDivergence::faceWeights<isotropic> fluxWeights(size_t i, size_t j, size_t k) const
			{
				using parabolic::at;
				parabolic::fluxDivergence::faceWeights<isotropic> w{};
				w.fluxWeightsX[at<0, 0, 0>] = cx(i, j, k);
				w.fluxWeightsY[at<0, 0, 0>] = cy(i, j, k);
				w.fluxWeightsZ[at<0, 0, 0>] = cz(i, j, k);
				w.makeSymmetric();
				return w;
			}

			/** @brief The coefficients of a cell */
			DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
			{
				parabolic::fluxDivergence::coefficientsFromFluxes<isotropic>(*this, i, j, k, a);
			}
		};

		/** @brief Set the boundary conditions. There are none */
		void setBoundaryConditions(SAMS::harness &harness)
		{
		}

		/** @brief Get the variables */
		void getVariables(SAMS::harness &harness)
		{
			harness.variableRegistry.fillPPArray("rho", rho);
		}

		/** @brief Set the constants of the problem. There are none, because the initial conditions package sets the parameters */
		void initialConditions(SAMS::harness &harness)
		{
		}

		/** @brief Register the temperature */
		void registerVariables(SAMS::harness &harness)
		{
			const int ghosts = 1;
			harness.variableRegistry.registerVariable<SAMS::T_dataType>(fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
		}

		/**
		 * @brief Register the arrays of every level
		 *
		 * An edge array is staggered in the direction of its edges.
		 * @param axisLevels The number of derived axes that are used in each direction
		 */
		void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
		{
			const std::string memory = memoryName.c_str();
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cx", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cy", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cz", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "beta", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
		}

		/** @brief Get the arrays of every level, from the first level to the last */
		template <typename T_level>
		void getSecondaryVariables(SAMS::harness &harness, std::vector<T_level> &levels)
		{
			const std::string memory = memoryName.c_str();
			parabolic::fillArrayLevels(harness, memory, "cx", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cx; });
			parabolic::fillArrayLevels(harness, memory, "cy", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cy; });
			parabolic::fillArrayLevels(harness, memory, "cz", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cz; });
			parabolic::fillArrayLevels(harness, memory, "beta", levels, [](T_level &l) -> volumeArray & { return l.data.beta; });
		}

		/** @brief Set the temperature from the energy, after applying the boundary conditions of the energy */
		void beforeSolve(SAMS::harness &harness, const lareData &lare)
		{
			auto &varRegistry = harness.variableRegistry;
			varRegistry.applyBoundaryConditions(T_species::fieldName());
			volumeArray energy, temperature;
			varRegistry.fillPPArray(T_species::fieldName(), energy);
			varRegistry.fillPPArray(fieldName(), temperature);
			const SAMS::variableDef &def = varRegistry.getVariable(fieldName());
			const auto &eos = lare.eos;
			auto convert = LAMBDA(size_t i, size_t j, size_t k)
			{
				temperature(i, j, k) = eos.getTemperature(LARE::eosEnergy(energy(i, j, k)));
			};
			pw::applyKernel(convert, def.getDimension(0).getLocalRange(SAMS::staggerType::CENTRED), def.getDimension(1).getLocalRange(SAMS::staggerType::CENTRED), def.getDimension(2).getLocalRange(SAMS::staggerType::CENTRED));
			pw::fence();
		}

		/** @brief Set the energy from the temperature */
		void afterSolve(SAMS::harness &harness, const lareData &lare)
		{
			auto &varRegistry = harness.variableRegistry;
			volumeArray energy, temperature;
			varRegistry.fillPPArray(T_species::fieldName(), energy);
			varRegistry.fillPPArray(fieldName(), temperature);
			const SAMS::variableDef &def = varRegistry.getVariable(fieldName());
			const auto &eos = lare.eos;
			auto convert = LAMBDA(size_t i, size_t j, size_t k)
			{
				energy(i, j, k) = eos.getEnergy(LARE::eosTemperature(temperature(i, j, k)));
			};
			pw::applyKernel(convert, def.getDimension(0).getLocalDomainRange(SAMS::staggerType::CENTRED), def.getDimension(1).getLocalDomainRange(SAMS::staggerType::CENTRED), def.getDimension(2).getLocalDomainRange(SAMS::staggerType::CENTRED));
			pw::fence();
		}

		/** @brief The operator of a level */
		template <typename T_level>
		stencil makeStencil(const T_level &l) const
		{
			stencil s;
			s.cx = l.data.conductance.cx;
			s.cy = l.data.conductance.cy;
			s.cz = l.data.conductance.cz;
			s.betaArray = l.data.beta;
			return s;
		}

		/** @brief Populate the arrays of levelData that are needed, on every level */
		template <typename T_level>
		void prepareSolve(std::vector<T_level> &levels, const parameters &params, const lareData &lare) const
		{
			setConductances(levels, params);
			setBeta(levels, lare);
		}
	};
}

#endif // MULTIGRIDTEST_ISOTROPIC_H
