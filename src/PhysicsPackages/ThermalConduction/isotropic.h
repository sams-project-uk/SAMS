#ifndef MULTIGRIDTEST_ISOTROPIC_H
#define MULTIGRIDTEST_ISOTROPIC_H
#include "runner.h"
#include "harness.h"
#include <vector>
#include "LARE/shared_data.h"
#include "../Parabolic/helpers.h"
#include "tags.h"

namespace SAMS
{
	namespace pw = portableWrapper;

	/**
	 * @brief Isotropic heat conduction with a constant conductivity for each species
	 *
	 * It has the same interface as braginskii, but its stencil has only the cell and its six face neighbours.
	 */
	template <typename T_eos = LARE::idealGas>
	struct isotropic
	{
		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;
		using hostLineArray = portableWrapper::hostArray<SAMS::T_dataType, 1>;

		using eosType = T_eos;
		using lareData = typename LARE::LARE3D<T_eos>::simulationData;
		/** @brief The constants of one species */
		struct speciesParameters
		{
			/** @brief The name of the energy variable of the species */
			const char *energyName = nullptr;
			/** @brief The conductivity */
			SAMS::T_dataType kappa = 0.0;
		};

		/** @brief The constants of the problem, which are in a data pack. There is one entry for each species that is solved, in the order of the subproblems */
		struct parameters
		{
			std::vector<speciesParameters> species;
		};

		using dataPack = SAMS::dataPacks::multiPack<parameters, lareData>;
		using packRefs = SAMS::packRefsOf_t<dataPack>;

		/** @brief The number of species that are solved, which defaultValues sets */
		SAMS::T_sizeType speciesCount = 0;

		/** @brief The name of the energy variable of each subproblem, which defaultValues sets */
		std::array<const char *, 2> energyNames{};

		/** @brief The name of the temperature variable, which the species share */
		inline static constexpr auto temperatureName = SAMS::makeName("temperature_") + T_eos::name;

		/** @brief The conductance of the x edge between cells i and i + 1 */
		DEVICEPREFIX SAMS::T_dataType faceX(const speciesParameters &pr, const lineArray &dx, const lineArray &dy, const lineArray &dz, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return pr.kappa * dy(j) * dz(k) / (0.5 * (dx(i) + dx(i + 1)));
		}

		/** @brief The conductance of the y edge between cells j and j + 1 */
		DEVICEPREFIX SAMS::T_dataType faceY(const speciesParameters &pr, const lineArray &dx, const lineArray &dy, const lineArray &dz, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return pr.kappa * dx(i) * dz(k) / (0.5 * (dy(j) + dy(j + 1)));
		}

		/** @brief The conductance of the z edge between cells k and k + 1 */
		DEVICEPREFIX SAMS::T_dataType faceZ(const speciesParameters &pr, const lineArray &dx, const lineArray &dy, const lineArray &dz, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return pr.kappa * dx(i) * dy(j) / (0.5 * (dz(k) + dz(k + 1)));
		}

		/** @brief Set beta from the density, heat capacity and timestep, and average it onto the coarse levels */
		template <typename T_level>
		void setBeta(std::vector<T_level> &levels, const lareData &lare, const SAMS::timeState &ts) const
		{
			T_level &l = levels[0];
			const auto &eos = lare.eos;
			const SAMS::T_dataType mass = lare.mf * lare.mh / lare.kb;
			const SAMS::T_dataType dt = ts.dt;
			const volumeArray density = rho, factor = l.data.beta, temperature = l.rhs;
			auto kernel = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SAMS::T_dataType energy = eos.getEnergy(LARE::eosDensity(density(i, j, k)), LARE::eosTemperature(temperature(i, j, k)), LARE::eosMass(mass));
				factor(i, j, k) = density(i, j, k) * eos.getCV(LARE::eosDensity(density(i, j, k)), LARE::eosEnergy(energy), LARE::eosMass(mass)) / dt;
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
			parabolic::restrictCells(levels, [](T_level &l) -> volumeArray & { return l.data.beta; });
		}

		/** @brief Set the conductances of one species on every level */
		template <typename T_level>
		void setConductances(std::vector<T_level> &levels, const speciesParameters pr) const
		{
			const isotropic p = *this;
			for (SAMS::T_sizeType n = 0; n < levels.size(); ++n)
			{
				T_level &l = levels[n];
				const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
				const volumeArray cx = l.data.conductance.cx, cy = l.data.conductance.cy, cz = l.data.conductance.cz;

				auto conductanceX = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
				{
					cx(i, j, k) = p.faceX(pr, dx, dy, dz, i, j, k);
				};
				pw::applyKernel(conductanceX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceY = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
				{
					cy(i, j, k) = p.faceY(pr, dx, dy, dz, i, j, k);
				};
				pw::applyKernel(conductanceY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceZ = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
				{
					cz(i, j, k) = p.faceZ(pr, dx, dy, dz, i, j, k);
				};
				pw::applyKernel(conductanceZ, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zbLocalDomainRange);
			}
		}

		/** @brief Density */
		volumeArray rho;

		/** @brief The name of the problem */
		inline static constexpr auto name = SAMS::makeName("isotropic") + T_eos::name;

		/** @brief The name that the arrays and axes of the problem are registered under, which the species share */
		inline static constexpr auto memoryName = SAMS::makeName("isotropic") + T_eos::name;

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
			/** @brief The prefactor of the time derivative per unit volume, divided by the timestep */
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

			/** @brief The prefactor of the time derivative, divided by the timestep */
			DEVICEPREFIX SAMS::T_dataType beta(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
			{
				return betaArray(i, j, k);
			}

			/**
			 * @brief The weights of the fluxes out of a cell through its faces on the high side
			 *
			 * The entries are found from their positions relative to the cell, so at<1, 0, 0> is the cell at i + 1.
			 */
			DEVICEPREFIX parabolic::fluxDivergence::faceWeights<isotropic> fluxWeights(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
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
			DEVICEPREFIX void coefficients(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k, SAMS::T_dataType *a) const
			{
				parabolic::fluxDivergence::coefficientsFromFluxes<isotropic>(*this, i, j, k, a);
			}
		};

		/** @brief Set the boundary conditions. There are none */
		void setBoundaryConditions([[maybe_unused]] SAMS::harness &harness)
		{
		}

		/** @brief Get the variables */
		void getVariables(SAMS::harness &harness)
		{
			harness.variableRegistry.fillPPArray("rho", rho);
		}

		/** @brief Set the constants of the problem. There are none, because the initial conditions package sets the parameters */
		void initialConditions([[maybe_unused]] SAMS::harness &harness)
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

		/** @brief The number of subproblems that are solved in each step, which is the number of species */
		SAMS::T_sizeType problemCount() const
		{
			return speciesCount;
		}

		/** @brief Called once before the solves of each step. There is nothing to do */
		void beforeAllSolves([[maybe_unused]] SAMS::harness &harness)
		{
		}

		/** @brief Called once after the solves of each step. There is nothing to do */
		void afterAllSolves([[maybe_unused]] SAMS::harness &harness)
		{
		}

		/** @brief Set the temperature from the energy of the species of the subproblem, after applying the boundary conditions of the energy */
		void beforeSolve(SAMS::harness &harness, SAMS::T_sizeType problemIndex, const parameters &params, const lareData &lare)
		{
			auto &varRegistry = harness.variableRegistry;
			const char *energyName = params.species[problemIndex].energyName;
			varRegistry.applyBoundaryConditions(energyName);
			volumeArray energy, temperature;
			varRegistry.fillPPArray(energyName, energy);
			varRegistry.fillPPArray(fieldName(), temperature);
			const SAMS::variableDef &def = varRegistry.getVariable(fieldName());
			const auto &eos = lare.eos;
			const SAMS::T_dataType mass = lare.mf * lare.mh / lare.kb;
			const volumeArray density = rho;
			auto convert = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				temperature(i, j, k) = eos.getTemperature(LARE::eosDensity(density(i, j, k)), LARE::eosEnergy(energy(i, j, k)), LARE::eosMass(mass));
			};
			pw::applyKernel(convert, def.getDimension(0).getLocalRange(SAMS::staggerType::CENTRED), def.getDimension(1).getLocalRange(SAMS::staggerType::CENTRED), def.getDimension(2).getLocalRange(SAMS::staggerType::CENTRED));
			pw::fence();
		}

		/** @brief Set the energy of the species of the subproblem from the temperature */
		void afterSolve(SAMS::harness &harness, SAMS::T_sizeType problemIndex, const parameters &params, const lareData &lare)
		{
			auto &varRegistry = harness.variableRegistry;
			volumeArray energy, temperature;
			varRegistry.fillPPArray(params.species[problemIndex].energyName, energy);
			varRegistry.fillPPArray(fieldName(), temperature);
			const SAMS::variableDef &def = varRegistry.getVariable(fieldName());
			const auto &eos = lare.eos;
			const SAMS::T_dataType mass = lare.mf * lare.mh / lare.kb;
			const volumeArray density = rho;
			auto convert = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				energy(i, j, k) = eos.getEnergy(LARE::eosDensity(density(i, j, k)), LARE::eosTemperature(temperature(i, j, k)), LARE::eosMass(mass));
			};
			pw::applyKernel(convert, def.getDimension(0).getLocalDomainRange(SAMS::staggerType::CENTRED), def.getDimension(1).getLocalDomainRange(SAMS::staggerType::CENTRED), def.getDimension(2).getLocalDomainRange(SAMS::staggerType::CENTRED));
			pw::fence();
		}

		/** @brief The operator of a level, from the arrays that prepareSolve set for the subproblem */
		template <typename T_level>
		stencil makeStencil(const T_level &l, [[maybe_unused]] SAMS::T_sizeType problemIndex) const
		{
			stencil s;
			s.cx = l.data.conductance.cx;
			s.cy = l.data.conductance.cy;
			s.cz = l.data.conductance.cz;
			s.betaArray = l.data.beta;
			return s;
		}

		/** @brief Add a species to the species that are solved, with its conductivity from the deck */
		template <typename T_species>
		static void addSpecies(SAMS::KVS &kvs, parameters &params)
		{
			speciesParameters sp;
			sp.energyName = T_species::fieldName();
			sp.kappa = kvs.getReal(std::string("/ThermalConduction/") + T_species::name.c_str() + "/kappa", 0.0);
			params.species.push_back(sp);
		}

		/** @brief Set the default parameters, which can be changed in the deck. /ThermalConduction/species is electron, ion, (electron, ion) or both, and both are solved if it is absent */
		void defaultValues(SAMS::KVS &kvs, parameters &params)
		{
			SAMS::KVS::Element speciesElement = kvs.getPath("/ThermalConduction/species", SAMS::KVS::Element("both"));
			std::vector<std::string> names;
			if (speciesElement.isType(SAMS::KVS::Type::Array))
			{
				for (const SAMS::KVS::Element &element : speciesElement.get<SAMS::KVS::Array &>()) names.push_back(element.asString());
			}
			else
			{
				names.push_back(speciesElement.asString());
			}
			params.species.clear();
			for (const std::string &speciesName : names)
			{
				if (speciesName == "both")
				{
					addSpecies<SAMS::electron>(kvs, params);
					addSpecies<SAMS::ion>(kvs, params);
				}
				else if (speciesName == SAMS::electron::name.c_str())
				{
					addSpecies<SAMS::electron>(kvs, params);
				}
				else if (speciesName == SAMS::ion::name.c_str())
				{
					addSpecies<SAMS::ion>(kvs, params);
				}
				else
				{
					const std::string message = "/ThermalConduction/species: unknown species " + speciesName;
					kvs.reportFatalError(message);
					throw std::runtime_error(message);
				}
			}
			speciesCount = params.species.size();
			for (SAMS::T_sizeType n = 0; n < speciesCount; ++n) energyNames[n] = params.species[n].energyName;
		}

		/** @brief The boundary condition of the temperature of a subproblem on an edge, from /ThermalConduction/boundaries, or else from the condition of its energy in /boundaries */
		parabolic::tcBoundaryType getBoundaryType(SAMS::KVS &kvs, SAMS::T_sizeType problemIndex, int dimension, SAMS::domain::edges edge) const
		{
			return parabolic::readBoundaryType(kvs, energyNames[problemIndex], dimension, edge);
		}

		/** @brief Populate the arrays of levelData that are needed, on every level, for the species of the subproblem. rhs of the first level is the old field */
		template <typename T_level>
		void prepareSolve(std::vector<T_level> &levels, SAMS::T_sizeType problemIndex, const SAMS::timeState &ts, const parameters &params, const lareData &lare) const
		{
			setConductances(levels, params.species[problemIndex]);
			setBeta(levels, lare, ts);
		}
	};
}

#endif // MULTIGRIDTEST_ISOTROPIC_H
