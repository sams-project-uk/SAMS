#ifndef MULTIGRIDTEST_TCTEST_H
#define MULTIGRIDTEST_TCTEST_H
#include "runner.h"
#include "harness.h"
#include "builtInBoundaryConditions.h"
#include "LARE/shared_data.h"
#include "braginskii.h"
#include "isotropic.h"
#include <cmath>

namespace SAMS
{
	/**
	 * @brief Domain, variables and initial conditions of the thermal conduction test
	 */
	template <typename T_problem>
	struct TCTest
	{
		using T_eos = typename T_problem::eosType;

		inline static constexpr auto nameType = SAMS::makeName("TCTest_") + T_problem::name;
		constexpr static std::string_view name = nameType;

		/**
		 * Initial conditions should not be timed
		 */
		constexpr static bool timeSimulation = false;

		/**
		 * The thermal conduction test is a core simulation
		 */
		constexpr static bool coreSimulation = true;

		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using hostVolumeArray = portableWrapper::hostArray<SAMS::T_dataType, 3>;
		using hostLineArray = portableWrapper::hostArray<SAMS::T_dataType, 1>;

		/**
		 * @brief The data pack of the package
		 */
		struct simulationData
		{
			size_t nx, ny, nz;
			hostLineArray xc_host, yc_host, zc_host;
			pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
		};
		using dataPack = simulationData;

		/**
		 * @brief Deactivate the packages that are active by default. The test ends and writes its output itself
		 * @param runnerRef The runner
		 */
		template <typename T_runner>
		void runnerInteraction(T_runner &runnerRef)
		{
			runnerRef.deactivatePackage();
		}

		/**
		 * @brief Set the deck keys that the deck does not set
		 * @param kvs The key value store
		 */
		void defaultDeck(SAMS::KVS &kvs)
		{
			// Not periodic in any direction
			kvs.setDefaultPath("/domain/MPI_periodic", SAMS::KVS::Array{false, false, false});
		}

		/**
		 * @brief Set the domain
		 * @param harness The harness, which holds the axis registry
		 */
		void setDomain(SAMS::harness &harness)
		{
			auto &axisReg = harness.axisRegistry;
			axisReg.setDomain("X", 128, -1.0, 1.0);
			axisReg.setDomain("Y", 128, -1.0, 1.0);
			axisReg.setDomain("Z", 128, -1.0, 1.0);
		}

		/**
		 * @brief Register the axes
		 * @param harness The harness, which holds the axis registry
		 */
		void registerAxes(SAMS::harness &harness)
		{
			auto &axisReg = harness.axisRegistry;
			axisReg.registerAxis("X", SAMS::MPIAxis(0));
			axisReg.registerAxis("Y", SAMS::MPIAxis(1));
			axisReg.registerAxis("Z", SAMS::MPIAxis(2));
		}

		/**
		 * @brief Register the variables
		 * @param harness The harness, which holds the variable registry
		 */
		void registerVariables(SAMS::harness &harness)
		{
			auto &varRegistry = harness.variableRegistry;
			const int ghosts = 1;
			for (const char *name : {SAMS::electron::fieldName(), SAMS::ion::fieldName()})
			{
				varRegistry.registerVariable<SAMS::T_dataType>(name, pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
			}
			varRegistry.registerVariable<SAMS::T_dataType>("rho", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
			// Each component of B is staggered by half a cell in the direction that it points
			varRegistry.registerVariable<SAMS::T_dataType>("bx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
			varRegistry.registerVariable<SAMS::T_dataType>("by", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts));
			varRegistry.registerVariable<SAMS::T_dataType>("bz", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
		}

		/**
		 * @brief Set the boundary conditions
		 * @param harness The harness, which holds the variable registry
		 */
		void setBoundaryConditions(SAMS::harness &harness)
		{
			auto &varRegistry = harness.variableRegistry;
			using clamp = SAMS::simpleClamp<SAMS::T_dataType, 3>;
			// The energy is shared equally between the species
			for (const char *name : {SAMS::electron::fieldName(), SAMS::ion::fieldName()})
			{
				SAMS::variableDef &energy = varRegistry.getVariable(name);
				for (int dim = 0; dim < 3; ++dim)
				{
					varRegistry.addBoundaryCondition(name, dim, SAMS::domain::edges::lower, std::make_shared<clamp>(energy, dim == 0 ? 2.5 : 0.5));
					varRegistry.addBoundaryCondition(name, dim, SAMS::domain::edges::upper, std::make_shared<clamp>(energy, 0.5));
				}
			}

			for (const char *name : {"bx", "by", "bz"})
			{
				SAMS::variableDef &varDef = varRegistry.getVariable(name);
				varRegistry.addBoundaryCondition(name, std::make_shared<SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>>(varDef));
			}
		}

		/**
		 * @brief Set the initial conditions
		 * @param harness The harness, which holds the variable registry
		 * @param lare LARE3D Parameter pack
		 * @param braginskiiParams The parameters of Braginskii conduction, null if it is not active
		 * @param isotropicParams The parameters of isotropic conduction, null if it is not active
		 * @param kvs The key value store
		 */
		void initialConditions(SAMS::harness &harness, typename T_problem::lareData &lare, typename SAMS::braginskii<T_eos>::parameters *braginskiiParams,
			typename SAMS::isotropic<T_eos>::parameters *isotropicParams, SAMS::KVS &kvs)
		{
			auto &varRegistry = harness.variableRegistry;
			volumeArray electronEnergy, ionEnergy, bx, by, bz, rho;
			varRegistry.fillPPArray(SAMS::electron::fieldName(), electronEnergy);
			varRegistry.fillPPArray(SAMS::ion::fieldName(), ionEnergy);
			varRegistry.fillPPArray("bx", bx);
			varRegistry.fillPPArray("by", by);
			varRegistry.fillPPArray("bz", bz);
			varRegistry.fillPPArray("rho", rho);

			// The energy is shared equally between the species
			pw::assign(electronEnergy, 0.5);
			pw::assign(ionEnergy, 0.5);
			// Direction of the field, in radians. phi is the angle from x in the x-y plane and theta the angle from z. The default is 45 degrees between x and y
			const SAMS::T_dataType theta = kvs.getReal("/TCTest/theta", M_PI / 2.0);
			const SAMS::T_dataType phi = kvs.getReal("/TCTest/phi", M_PI / 4.0);
			// Strength of the field. The default gives the field of 1 in x and y
			const SAMS::T_dataType B0 = kvs.getReal("/TCTest/B0", std::sqrt(2.0));
			pw::assign(bx, B0 * std::sin(theta) * std::cos(phi));
			pw::assign(by, B0 * std::sin(theta) * std::sin(phi));
			pw::assign(bz, B0 * std::cos(theta));
			pw::assign(rho, 1.0);
			lare.gas_gamma = 2.0;
			lare.eos.setGamma(2.0);
			// LARE3D does not run in the test, so the constants that the conduction uses are set here, in normalised units
			lare.mf = 1.0;
			lare.mh = 1.0;
			lare.kb = 1.0;
			if (braginskiiParams)
			{
				for (auto &sp : braginskiiParams->species)
				{
					sp.kappa0 = 0.1;
					sp.exponent = 2.5;
					sp.fieldSoftening = 1.0e-6;
					sp.limiterFlux = 1.0;
					sp.limiterFluxSet = true;
					sp.limiterExponent = 2.0;
					sp.limiterSoftening = 1.0e-10;
				}
			}
			if (isotropicParams)
			{
				for (auto &sp : isotropicParams->species)
					sp.kappa = 0.1;
			}
		}

		/**
		 * @brief Set the timestep
		 * @param timeData The timestep data
		 */
		void calculateTimestep(SAMS::timeState &timeData)
		{
			timeData.dt = 0.1;
		}

		/**
		 * @brief Ask for the run to end
		 * @param terminate Set to true when the run should end
		 * @param tData The timestep data
		 */
		void queryTerminate(bool &terminate, SAMS::timeState &tData)
		{
			if (tData.step >= 10)
			{
				terminate |= true;
			}
		}

		/**
		 * @brief Ask for output
		 * @param request The collections of outputs. The default collection is activated when output is wanted
		 * @param tData The timestep data
		 */
		void queryOutput(SAMS::outputRequest &request, [[maybe_unused]] SAMS::timeState &tData)
		{
			// Every step
			request.activate("default", "diagnostics{_NNNNN}");
		}

		/**
		 * @brief Fill the data pack
		 * @param harness The harness, which holds the axis registry
		 * @param data The data pack
		 */
		void getVariables(SAMS::harness &harness, simulationData &data)
		{
			auto &axisReg = harness.axisRegistry;

			data.nx = axisReg.getLocalDomainElements("X", SAMS::staggerType::CENTRED);
			data.ny = axisReg.getLocalDomainElements("Y", SAMS::staggerType::CENTRED);
			data.nz = axisReg.getLocalDomainElements("Z", SAMS::staggerType::CENTRED);

			data.xcLocalDomainRange = axisReg.getLocalDomainRange("X", SAMS::staggerType::CENTRED);
			data.ycLocalDomainRange = axisReg.getLocalDomainRange("Y", SAMS::staggerType::CENTRED);
			data.zcLocalDomainRange = axisReg.getLocalDomainRange("Z", SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalAxis("X", data.xc_host, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalAxis("Y", data.yc_host, SAMS::staggerType::CENTRED);
			axisReg.fillPPLocalAxis("Z", data.zc_host, SAMS::staggerType::CENTRED);
		}

		/**
		 * @brief Handles of the outputs of the energies of the electrons and the ions
		 */
		SAMS::outputHandle electronOutput, ionOutput;

		/**
		 * @brief Register the outputs that the test can write, which are written by default and needed for a restart
		 * @param outputs The registry of outputs
		 */
		void registerOutputVariables(SAMS::outputRegistry &outputs)
		{
			electronOutput = outputs.registerOutput(SAMS::electron::fieldName(), std::string("Specific internal energy of the ") + SAMS::electron::name.c_str());
			ionOutput = outputs.registerOutput(SAMS::ion::fieldName(), std::string("Specific internal energy of the ") + SAMS::ion::name.c_str());
		}

		/**
		 * @brief Register the output variables that the dump writes
		 * @tparam T_writer The writer type
		 * @param regWriter The writer for variables in the registries
		 * @param level The outputs that the dump writes
		 */
		template <typename T_writer>
		void registerDumpVariables(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &regWriter, const SAMS::outputLevel &level)
		{
			if (level.includes(electronOutput)) regWriter.template registerData<SAMS::T_dataType, 3>(SAMS::electron::fieldName(), "MeshCC", SAMS::electron::fieldName());
			if (level.includes(ionOutput)) regWriter.template registerData<SAMS::T_dataType, 3>(SAMS::ion::fieldName(), "MeshCC", SAMS::ion::fieldName());
		}

		/**
		 * @brief Register the output meshes
		 * @tparam T_writer The writer type
		 * @param regWriter The writer for variables in the registries
		 */
		template <typename T_writer>
		void registerDumpMeshes(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &regWriter, const SAMS::outputLevel &)
		{
			regWriter.registerRectilinearMesh("MeshCC", {{"X", SAMS::staggerType::CENTRED}, {"Y", SAMS::staggerType::CENTRED}, {"Z", SAMS::staggerType::CENTRED}});
		}

		/**
		 * @brief Copy a device array to a contiguous host array
		 * @param data The data pack
		 * @param manager Array manager
		 * @param device The array to copy
		 * @param host The host array, which is allocated here
		 */
		inline void getHostVersion(simulationData &data, pw::portableArrayManager &manager, volumeArray &device, hostVolumeArray &host)
		{
			auto fullHost = manager.makeHostAvailable(device);

			manager.allocate(host, data.xcLocalDomainRange, data.ycLocalDomainRange, data.zcLocalDomainRange);
			// Have to copy data since the HDF writer expects contiguous data
			manager.copyDataHost(host, fullHost(data.xcLocalDomainRange, data.ycLocalDomainRange, data.zcLocalDomainRange));
			manager.deallocate(fullHost);
		}

		/**
		 * @brief Write the output meshes
		 * @tparam T_writer The writer type
		 */
		template <typename T_writer>
		void writeDumpMeshes(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &)
		{
		}

		/**
		 * @brief Write the output variables
		 * @tparam T_writer The writer type
		 */
		template <typename T_writer>
		void writeDumpVariables(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &)
		{
		}	};
}

#endif // MULTIGRIDTEST_TCTEST_H
