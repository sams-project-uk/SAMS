#ifndef MULTIGRIDTEST_H
#define MULTIGRIDTEST_H
#include "runner.h"
#include "harness.h"

namespace pw = portableWrapper;

struct mg
{

	using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
	using hostVolumeArray = portableWrapper::hostArray<SAMS::T_dataType, 3>;
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;
	using hostLineArray = portableWrapper::hostArray<SAMS::T_dataType, 1>;

	struct simulationData
	{
		size_t nx, ny, nz;
		lineArray xc, yc, zc;
		lineArray dxc, dyc, dzc;
		hostLineArray xc_host, yc_host, zc_host;
		hostLineArray dxc_host, dyc_host, dzc_host;
		pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
	};

	struct level
	{
		size_t nx, ny, nz;
		pw::Range xcLocalDomainRange, ycLocalDomainRange, zcLocalDomainRange;
		lineArray dxc, dyc, dzc;
		hostLineArray dxc_host, dyc_host, dzc_host;
		volumeArray currentTime, nextTime;
		volumeArray residual;
	};

	std::vector<level> levels;

	constexpr static std::string_view name = "mg";
	constexpr static bool coreSimulation = true;
	constexpr static bool timeSimulation = true;
	using dataPack = simulationData;

	pw::portableArrayManager &manager;

	mg(SAMS::harness &hr) : manager(hr.memoryRegistry.getArrayManager()) {};

	void setDomain(SAMS::harness &harness)
	{
		auto &axisReg = harness.axisRegistry;
		axisReg.setDomain("X", 100, -1.0, 1.0);
		axisReg.setDomain("Y", 100, -1.0, 1.0);
		axisReg.setDomain("Z", 100, -1.0, 1.0);
	}

	void calculateTimestep(SAMS::timeState &timeData)
	{
		timeData.dt = 0.1;
	}

	void registerAxes(SAMS::harness &harness)
	{
		auto &axisReg = harness.axisRegistry;
		axisReg.registerAxis("X", SAMS::MPIAxis(0));
		axisReg.registerAxis("Y", SAMS::MPIAxis(1));
		axisReg.registerAxis("Z", SAMS::MPIAxis(2));
	}

	void registerVariables(SAMS::harness &harness)
	{
		auto &varRegistry = harness.variableRegistry;
		const int ghosts = 1; // 1 Ghost cells at top and bottom of each dimension
		varRegistry.registerVariable<SAMS::T_dataType>("energy_electron", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
		varRegistry.registerVariable<SAMS::T_dataType>("scratch", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
	}

	void queryTerminate(bool &terminate, SAMS::timeState &tData)
	{
		if (tData.step >= 120)
		{
			terminate |= true;
		}
	}

	void getVariables(SAMS::harness &harness, simulationData &data)
	{
		auto &axisReg = harness.axisRegistry;

		data.nx = axisReg.getLocalDomainElements("X", SAMS::staggerType::CENTRED);
		data.ny = axisReg.getLocalDomainElements("Y", SAMS::staggerType::CENTRED);
		data.nz = axisReg.getLocalDomainElements("Z", SAMS::staggerType::CENTRED);

		auto &varRegistry = harness.variableRegistry;

		data.xcLocalDomainRange = axisReg.getLocalDomainRange("X", SAMS::staggerType::CENTRED);
		data.ycLocalDomainRange = axisReg.getLocalDomainRange("Y", SAMS::staggerType::CENTRED);
		data.zcLocalDomainRange = axisReg.getLocalDomainRange("Z", SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalAxis("X", data.xc, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalAxis("Y", data.yc, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalAxis("Z", data.zc, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalDelta("X", data.dxc, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalDelta("Y", data.dyc, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalDelta("Z", data.dzc, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalAxis("X", data.xc_host, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalAxis("Y", data.yc_host, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalAxis("Z", data.zc_host, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalDelta("X", data.dxc_host, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalDelta("Y", data.dyc_host, SAMS::staggerType::CENTRED);
		axisReg.fillPPLocalDelta("Z", data.dzc_host, SAMS::staggerType::CENTRED);

		portableWrapper::acceleratedArray<SAMS::T_dataType, 3> energy_electron, scratch;
		varRegistry.fillPPArray("energy_electron", energy_electron);
		varRegistry.fillPPArray("scratch", scratch);

		level l;
		l.nx = data.nx;
		l.ny = data.ny;
		l.nz = data.nz;
		l.currentTime = energy_electron;
		l.nextTime = scratch;
		l.xcLocalDomainRange = pw::Range(1, l.nx);
		l.ycLocalDomainRange = pw::Range(1, l.ny);
		l.zcLocalDomainRange = pw::Range(1, l.nz);
		l.dxc = data.dxc;
		l.dyc = data.dyc;
		l.dzc = data.dzc;
		l.dxc_host = data.dxc_host;
		l.dyc_host = data.dyc_host;
		l.dzc_host = data.dzc_host;
		manager.allocate(l.residual, pw::Range(0, l.nx + 1), pw::Range(0, l.ny + 1), pw::Range(0, l.nz + 1));
		levels.push_back(l);

		int levelSrc = 0;

		while (true)
		{
			level l;
			l.nx = levels[levelSrc].nx / 2.0;
			l.ny = levels[levelSrc].ny / 2.0;
			l.nz = levels[levelSrc].nz / 2.0;
			l.xcLocalDomainRange = pw::Range(1, l.nx);
			l.ycLocalDomainRange = pw::Range(1, l.ny);
			l.zcLocalDomainRange = pw::Range(1, l.nz);
			manager.allocate(l.currentTime, pw::Range(0, l.nx + 1), pw::Range(0, l.ny + 1), pw::Range(0, l.nz + 1));
			manager.allocate(l.nextTime, pw::Range(0, l.nx + 1), pw::Range(0, l.ny + 1), pw::Range(0, l.nz + 1));
			manager.allocate(l.residual, pw::Range(0, l.nx + 1), pw::Range(0, l.ny + 1), pw::Range(0, l.nz + 1));
			manager.allocate(l.dxc, pw::Range(0, l.nx + 1));
			manager.allocate(l.dyc, pw::Range(0, l.ny + 1));
			manager.allocate(l.dzc, pw::Range(0, l.nz + 1));
			manager.allocate(l.dxc_host, pw::Range(0, l.nx + 1));
			manager.allocate(l.dyc_host, pw::Range(0, l.ny + 1));
			manager.allocate(l.dzc_host, pw::Range(0, l.nz + 1));
			pw::assign(l.dxc, levels[levelSrc].dxc_host(1) * 2.0);
			pw::assign(l.dyc, levels[levelSrc].dyc_host(1) * 2.0);
			pw::assign(l.dzc, levels[levelSrc].dzc_host(1) * 2.0);
			pw::assign(l.dxc_host, levels[levelSrc].dxc_host(1) * 2.0);
			pw::assign(l.dyc_host, levels[levelSrc].dyc_host(1) * 2.0);
			pw::assign(l.dzc_host, levels[levelSrc].dzc_host(1) * 2.0);
			levels.push_back(l);
			levelSrc++;
			if (l.nx <= 4 || l.ny <= 4 || l.nz <= 4)
				break;
		}
	}

	void initialConditions(SAMS::harness &harness, simulationData &data)
	{

		auto &varRegistry = harness.variableRegistry;

		portableWrapper::acceleratedArray<SAMS::T_dataType, 3> energy_electron, scratch;
		varRegistry.fillPPArray("energy_electron", energy_electron);
		varRegistry.fillPPArray("scratch", scratch);

		bc(0);
	}

	void bc(int level){
		auto &l = levels[level];
		pw::assign(l.currentTime(0, pw::Range(), pw::Range()), 5.0);
		pw::assign(l.currentTime(l.nx + 1, pw::Range(), pw::Range()), 1.0);
		pw::assign(l.currentTime(pw::Range(), 0, pw::Range()), 1.0);
		pw::assign(l.currentTime(pw::Range(), l.ny + 1, pw::Range()), 1.0);
		pw::assign(l.currentTime(pw::Range(), pw::Range(), 0), 1.0);
		pw::assign(l.currentTime(pw::Range(), pw::Range(), l.nz + 1), 1.0);
		pw::assign(l.nextTime(0, pw::Range(), pw::Range()), 5.0);
		pw::assign(l.nextTime(l.nx + 1, pw::Range(), pw::Range()), 1.0);
		pw::assign(l.nextTime(pw::Range(), 0, pw::Range()), 1.0);
		pw::assign(l.nextTime(pw::Range(), l.ny + 1, pw::Range()), 1.0);
		pw::assign(l.nextTime(pw::Range(), pw::Range(), 0), 1.0);
		pw::assign(l.nextTime(pw::Range(), pw::Range(), l.nz + 1), 1.0);

	}

	// Coarse (correction) levels solve for the *error*, which must satisfy
	// homogeneous (zero) Dirichlet boundary conditions. The restricted
	// residual stored in currentTime is only ever read on the interior, so
	// it must not be overwritten with physical boundary values.
	void bcCorrection(int level){
		auto &l = levels[level];
		pw::assign(l.nextTime(0, pw::Range(), pw::Range()), 0.0);
		pw::assign(l.nextTime(l.nx + 1, pw::Range(), pw::Range()), 0.0);
		pw::assign(l.nextTime(pw::Range(), 0, pw::Range()), 0.0);
		pw::assign(l.nextTime(pw::Range(), l.ny + 1, pw::Range()), 0.0);
		pw::assign(l.nextTime(pw::Range(), pw::Range(), 0), 0.0);
		pw::assign(l.nextTime(pw::Range(), pw::Range(), l.nz + 1), 0.0);
	}

	void restrict(int levelSrc, int levelDest, double dt)
	{
		auto &src = levels[levelSrc];
		auto &dest = levels[levelDest];

		const double alpha = 0.1;

		// Compute the residual r = RHS - L(u) on the fine (source) level
		auto res = LAMBDA(size_t i, size_t j, size_t k)
		{
			const double dx2 = src.dxc(i) * src.dxc(i);
			const double dy2 = src.dyc(j) * src.dyc(j);
			const double dz2 = src.dzc(k) * src.dzc(k);
			const double cx = alpha / dx2;
			const double cy = alpha / dy2;
			const double cz = alpha / dz2;
			const double diag = 1.0 / dt + 2.0 * (cx + cy + cz);
			double neighbors =
				cx * (src.nextTime(i - 1, j, k) + src.nextTime(i + 1, j, k)) +
				cy * (src.nextTime(i, j - 1, k) + src.nextTime(i, j + 1, k)) +
				cz * (src.nextTime(i, j, k - 1) + src.nextTime(i, j, k + 1));
			double lhs = diag * src.nextTime(i, j, k) - neighbors;
			double rhs = src.currentTime(i, j, k) / dt;
			src.residual(i, j, k) = rhs - lhs;
		};
		pw::applyKernel(res, src.xcLocalDomainRange, src.ycLocalDomainRange, src.zcLocalDomainRange);

		// Restrict the residual to become the RHS on the coarse (destination) level.
		// cycle()/GS() always divides RHSSrc by dt internally (since on the finest
		// level RHSSrc is the physical field u_old and the equation solved is
		// L(u) = u_old/dt). The residual computed above is already in "divided by
		// dt" units, so we must multiply by dt here to cancel out that later
		// division and avoid an extra spurious 1/dt scaling on every V-cycle.
		auto l = LAMBDA(size_t i, size_t j, size_t k)
		{
			dest.currentTime(i, j, k) = dt * 0.125 * (src.residual(2 * i - 1, 2 * j - 1, 2 * k - 1) + src.residual(2 * i - 1, 2 * j - 1, 2 * k) +
												 src.residual(2 * i - 1, 2 * j, 2 * k - 1) + src.residual(2 * i - 1, 2 * j, 2 * k) +
												 src.residual(2 * i, 2 * j - 1, 2 * k - 1) + src.residual(2 * i, 2 * j - 1, 2 * k) +
												 src.residual(2 * i, 2 * j, 2 * k - 1) + src.residual(2 * i, 2 * j, 2 * k));
		};
		pw::applyKernel(l, dest.xcLocalDomainRange, dest.ycLocalDomainRange, dest.zcLocalDomainRange);

		// Start the coarse-level error/solution guess at zero
		pw::assign(dest.nextTime, 0.0);

		bcCorrection(levelDest);
	}

	void prolongate(int levelSrc, int levelDest)
	{
		auto &src = levels[levelSrc];
		auto &dest = levels[levelDest];

		// Interpolate the coarse-grid error and add it as a correction to the
		// existing fine-grid solution (do NOT zero/overwrite dest.nextTime here).
		auto l = LAMBDA(size_t i, size_t j, size_t k)
		{
			dest.nextTime(2 * i - 1, 2 * j - 1, 2 * k - 1) += src.nextTime(i, j, k);
			dest.nextTime(2 * i - 1, 2 * j - 1, 2 * k) += src.nextTime(i, j, k);
			dest.nextTime(2 * i - 1, 2 * j, 2 * k - 1) += src.nextTime(i, j, k);
			dest.nextTime(2 * i - 1, 2 * j, 2 * k) += src.nextTime(i, j, k);
			dest.nextTime(2 * i, 2 * j - 1, 2 * k - 1) += src.nextTime(i, j, k);
			dest.nextTime(2 * i, 2 * j - 1, 2 * k) += src.nextTime(i, j, k);
			dest.nextTime(2 * i, 2 * j, 2 * k - 1) += src.nextTime(i, j, k);
			dest.nextTime(2 * i, 2 * j, 2 * k) += src.nextTime(i, j, k);
		};
		pw::applyKernel(l, src.xcLocalDomainRange, src.ycLocalDomainRange, src.zcLocalDomainRange);

		// Only the finest level carries the real physical boundary values;
		// any other (coarser) level is still solving for an error correction
		// and must keep homogeneous boundary conditions.
		if (levelDest == 0)
			bc(levelDest);
		else
			bcCorrection(levelDest);
	}

	double cycle(
		portableWrapper::acceleratedArray<SAMS::T_dataType, 3> &dest,
		const portableWrapper::acceleratedArray<SAMS::T_dataType, 3> &LHSSrc,
		const portableWrapper::acceleratedArray<SAMS::T_dataType, 3> &RHSSrc,
		double dt, lineArray dx, lineArray dy, lineArray dz, double alpha)
	{

		size_t nx = RHSSrc.getUpperBound(0) - 1;
		size_t ny = RHSSrc.getUpperBound(1) - 1;
		size_t nz = RHSSrc.getUpperBound(2) - 1;

		auto l = LAMBDA(size_t i, size_t j, size_t k)
		{
			const double dx2 = dx(i) * dx(i);
			const double dy2 = dy(j) * dy(j);
			const double dz2 = dz(k) * dz(k);
			const double cx = alpha / dx2;
			const double cy = alpha / dy2;
			const double cz = alpha / dz2;
			const double diag = 1.0 / dt + 2.0 * (cx + cy + cz);
			double neighbors =
				cx * (LHSSrc(i - 1, j, k) + LHSSrc(i + 1, j, k)) +
				cy * (LHSSrc(i, j - 1, k) + LHSSrc(i, j + 1, k)) +
				cz * (LHSSrc(i, j, k - 1) + LHSSrc(i, j, k + 1));

			double rhs = RHSSrc(i, j, k) / dt + neighbors;

			double new_val = rhs / diag;

			dest(i, j, k) = new_val;
		};
		pw::applyRedBlackKernel(l, pw::Range(1, nx), pw::Range(1, ny), pw::Range(1, nz));
		return pw::maxval(LAMBDA(size_t i, size_t j, size_t k) {
				const double dx2 = dx(i) * dx(i);
				const double dy2 = dy(j) * dy(j);
				const double dz2 = dz(k) * dz(k);
				const double cx = alpha / dx2;
				const double cy = alpha / dy2;
				const double cz = alpha / dz2;
				const double diag = 1.0 / dt + 2.0 * (cx + cy + cz);
				double neighbors =
				cx * (dest(i - 1, j, k) + dest(i + 1, j, k)) +
				cy * (dest(i, j - 1, k) + dest(i, j + 1, k)) +
				cz * (dest(i, j, k - 1) + dest(i, j, k + 1));

				double lhs = diag * dest(i, j, k) - neighbors;
				double rhs = RHSSrc(i, j, k) / dt;

				return std::abs(rhs - lhs); }, pw::Range(1, nx), pw::Range(1, ny), pw::Range(1, nz));
	}

	bool GS(
		portableWrapper::acceleratedArray<SAMS::T_dataType, 3> &dest,
		const portableWrapper::acceleratedArray<SAMS::T_dataType, 3> &RHSSrc,
		lineArray dx, lineArray dy, lineArray dz,
		double alpha, double dt,
		int iterations, double tol = 0.0)
	{

		double max_delta = 1.0e9;//cycle(dest, RHSSrc, RHSSrc, dt, dx, dy, dz, alpha);
		bool converged = false;

		if (!(tol > 0.0 && max_delta < tol))
		{
			for (int iter = 1; iter < iterations; ++iter)
			{
				max_delta = cycle(dest, dest, RHSSrc, dt, dx, dy, dz, alpha);

				if (tol > 0.0 && max_delta < tol)
				{
					converged = true;
					break;
				}
			}
		}
		return converged;
	}

	void endOfTimestep(const SAMS::timeState &ts, SAMS::harness &harness, simulationData &data)
	{
		auto &axisReg = harness.axisRegistry;
		auto &varRegistry = harness.variableRegistry;
		portableWrapper::acceleratedArray<SAMS::T_dataType, 3> energy_electron, scratch;
		varRegistry.fillPPArray("energy_electron", energy_electron);
		varRegistry.fillPPArray("scratch", scratch);

		bool unconverged = true;

		int levelSrc = 0;
		bool converged = false;
		pw::assign(scratch, energy_electron);
		while (true)
		{
			int cycles = levelSrc == levels.size() - 1 ? 1000: 10;
			// Run a few iterations of GS on the source level
			converged = GS(levels[levelSrc].nextTime, levels[levelSrc].currentTime, levels[levelSrc].dxc, levels[levelSrc].dyc, levels[levelSrc].dzc, 0.1, ts.dt, cycles, 0.001);
			// If converged we're done
			if (converged) {
				//On the finest level the solution is done if we're converged
				if (levelSrc == 0) break;
				//Otherwise prolongate the error correction to the next level and continue
				int levelDest = levelSrc - 1;
				prolongate(levelSrc, levelDest);
				levelSrc--;
				continue;
			}
			// If we get here then we need to restrict the residual to the next level and continue
			int levelDest = levelSrc + 1;
			if (levelDest >= levels.size()) continue;
			restrict(levelSrc, levelDest, ts.dt);
			levelSrc++;
		}
		pw::assign(levels[levelSrc].currentTime, levels[levelSrc].nextTime);
		std::cout << pw::sum(energy_electron) << "\n";
	}

	void queryOutput(bool &shouldOutput, SAMS::timeState &tData)
	{
		if ((tData.step & 1) == 0)
			shouldOutput = true;
	}

	template <typename T_writer>
	void registerOutputVariables(writer<T_writer> &writer)
	{
		writer.template registerData<double>("energy_electron", "MeshCC");
	}

	template <typename T_writer>
	void registerOutputMeshes(writer<T_writer> &writer, simulationData &data)
	{
		writer.template registerRectilinearMesh<double>("MeshCC", data.nx, data.ny, data.nz);
	}

	inline void getHostVersion(simulationData &data, pw::portableArrayManager &manager, volumeArray &device, hostVolumeArray &host)
	{
		using Range = pw::Range;
		// Copy the data if needed
		auto fullHost = manager.makeHostAvailable(device);

		// Now allocate memory for just the part we want to write
		manager.allocate(host, data.xcLocalDomainRange, data.ycLocalDomainRange, data.zcLocalDomainRange);
		// Have to copy data since the HDF writer expects contiguous data
		manager.copyDataHost(host, fullHost(data.xcLocalDomainRange, data.ycLocalDomainRange, data.zcLocalDomainRange));
		manager.deallocate(fullHost);
	}

	template <typename T_writer>
	void writeOutputMeshes(writer<T_writer> &writer, simulationData &data)
	{
		writer.writeRectilinearMesh("MeshCC", &data.xc_host(1), &data.yc_host(1), &data.zc_host(1));
	}

	template <typename T_writer>
	void writeOutputVariables(writer<T_writer> &writer, SAMS::harness &harness, simulationData &data)
	{
		pw::portableArrayManager manager;
		hostVolumeArray host;

		auto &axisReg = harness.axisRegistry;
		auto &varRegistry = harness.variableRegistry;
		portableWrapper::acceleratedArray<SAMS::T_dataType, 3> energy_electron;
		varRegistry.fillPPArray("energy_electron", energy_electron);

		getHostVersion(data, manager, energy_electron, host);
		writer.writeData("energy_electron", host.data());
	}
};

#endif
