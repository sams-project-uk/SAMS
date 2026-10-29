#ifndef MULTIGRIDTEST_TCTEST_H
#define MULTIGRIDTEST_TCTEST_H
#include "runner.h"
#include "harness.h"
#include "builtInBoundaryConditions.h"
#include "LARE/shared_data.h"
#include "braginskii.h"
#include "isotropic.h"

namespace SAMS
{
	/**
	 * @brief Domain, variables and initial conditions of the thermal conduction test
	 */
	template <typename T_problem>
	struct TCTest
	{
		using T_eos = typename T_problem::eosType;

		inline static constexpr auto nameType = SAMS::constexprName("TCTest_") + T_problem::name;
		constexpr static std::string_view name = nameType;

		/**
		 * Initial conditions should not be timed
		 */
		constexpr static bool timeSimulation = false;

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
		 * @brief Set the domain
		 * @param harness The harness, which holds the axis registry
		 */
		void setDomain(SAMS::harness &harness)
		{
			auto &axisReg = harness.axisRegistry;
			axisReg.setDomain("X", 100, -1.0, 1.0);
			axisReg.setDomain("Y", 100, -1.0, 1.0);
			axisReg.setDomain("Z", 100, -1.0, 1.0);
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
			varRegistry.registerVariable<SAMS::T_dataType>(T_problem::species::fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
			varRegistry.registerVariable<SAMS::T_dataType>("rho", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
			// Each component of B is staggered by half a cell in the direction that it points
			varRegistry.registerVariable<SAMS::T_dataType>("Bx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
			varRegistry.registerVariable<SAMS::T_dataType>("By", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts));
			varRegistry.registerVariable<SAMS::T_dataType>("Bz", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
		}

		/**
		 * @brief Set the boundary conditions
		 * @param harness The harness, which holds the variable registry
		 */
		void setBoundaryConditions(SAMS::harness &harness)
		{
			auto &varRegistry = harness.variableRegistry;
			using clamp = SAMS::simpleClamp<SAMS::T_dataType, 3>;
			SAMS::variableDef &energy = varRegistry.getVariable(T_problem::species::fieldName());
			for (int dim = 0; dim < 3; ++dim)
			{
				varRegistry.addBoundaryCondition(T_problem::species::fieldName(), dim, SAMS::domain::edges::lower, std::make_shared<clamp>(energy, dim == 0 ? 5.0 : 1.0));
				varRegistry.addBoundaryCondition(T_problem::species::fieldName(), dim, SAMS::domain::edges::upper, std::make_shared<clamp>(energy, 1.0));
			}

			for (const char *name : {"Bx", "By", "Bz"})
			{
				SAMS::variableDef &varDef = varRegistry.getVariable(name);
				varRegistry.addBoundaryCondition(name, std::make_shared<SAMS::simpleZeroGradientBC<SAMS::T_dataType, 3>>(varDef));
			}
		}

		/**
		 * @brief Set the initial conditions
		 * @param harness The harness, which holds the variable registry
		 * @param lare LARE3D Parameter pack
		 * @param params The parameters of the problem
		 */
		void initialConditions(SAMS::harness &harness, typename T_problem::lareData &lare, typename T_problem::parameters &params)
		{
			auto &varRegistry = harness.variableRegistry;
			volumeArray field, bx, by, bz, rho;
			varRegistry.fillPPArray(T_problem::species::fieldName(), field);
			varRegistry.fillPPArray("Bx", bx);
			varRegistry.fillPPArray("By", by);
			varRegistry.fillPPArray("Bz", bz);
			varRegistry.fillPPArray("rho", rho);

			pw::assign(field, 1.0);
			pw::assign(bx, 1.0);
			pw::assign(by, 1.0);
			pw::assign(bz, 0.0);
			pw::assign(rho, 1.0);
			lare.gas_gamma = 2.0;
			lare.eos.setGamma(2.0);
			if constexpr (std::is_same_v<T_problem, SAMS::braginskii<SAMS::electron, T_eos>> ||
			std::is_same_v<T_problem, SAMS::braginskii<SAMS::ion, T_eos>>)
			{
				params.kappa0 = 0.1;
				params.exponent = 2.5;
				params.fieldSoftening = 1.0e-6;
				params.limiterFlux = 1.0;
				params.limiterExponent = 2.0;
				params.limiterSoftening = 1.0e-10;
			}
			else if constexpr (std::is_same_v<T_problem, SAMS::isotropic<SAMS::electron, T_eos>> ||
			std::is_same_v<T_problem, SAMS::isotropic<SAMS::ion, T_eos>>)
			{
				params.kappa = 0.1;
			}
			else
			{
				static_assert(!std::is_same_v<T_problem, T_problem>, "TCTest does not know the parameters of this problem");
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
		 * @param shouldOutput Set to true when output is wanted
		 * @param tData The timestep data
		 */
		void queryOutput(bool &shouldOutput, SAMS::timeState &tData)
		{
			if ((tData.step & 1) == 0)
				shouldOutput = true;
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
		 * @brief Register the output variables
		 * @tparam T_writer The writer type
		 * @param writer The writer
		 */
		template <typename T_writer>
		void registerOutputVariables(writer<T_writer> &writer)
		{
			writer.template registerData<double>(T_problem::species::fieldName(), "MeshCC");
		}

		/**
		 * @brief Register the output meshes
		 * @tparam T_writer The writer type
		 * @param writer The writer
		 * @param data The data pack
		 */
		template <typename T_writer>
		void registerOutputMeshes(writer<T_writer> &writer, simulationData &data)
		{
			writer.template registerRectilinearMesh<double>("MeshCC", data.nx, data.ny, data.nz);
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
			using Range = pw::Range;
			auto fullHost = manager.makeHostAvailable(device);

			manager.allocate(host, data.xcLocalDomainRange, data.ycLocalDomainRange, data.zcLocalDomainRange);
			// Have to copy data since the HDF writer expects contiguous data
			manager.copyDataHost(host, fullHost(data.xcLocalDomainRange, data.ycLocalDomainRange, data.zcLocalDomainRange));
			manager.deallocate(fullHost);
		}

		/**
		 * @brief Write the output meshes
		 * @tparam T_writer The writer type
		 * @param writer The writer
		 * @param data The data pack
		 */
		template <typename T_writer>
		void writeOutputMeshes(writer<T_writer> &writer, simulationData &data)
		{
			writer.writeRectilinearMesh("MeshCC", &data.xc_host(1), &data.yc_host(1), &data.zc_host(1));
		}

		/**
		 * @brief Write the output variables
		 * @tparam T_writer The writer type
		 * @param writer The writer
		 * @param harness The harness, which holds the variable registry
		 * @param data The data pack
		 */
		template <typename T_writer>
		void writeOutputVariables(writer<T_writer> &writer, SAMS::harness &harness, simulationData &data)
		{
			pw::portableArrayManager manager;
			hostVolumeArray host;
			volumeArray field;
			harness.variableRegistry.fillPPArray(T_problem::species::fieldName(), field);

			getHostVersion(data, manager, field, host);
			writer.writeData(T_problem::species::fieldName(), host.data());
		}
	};
}

#endif // MULTIGRIDTEST_TCTEST_H
