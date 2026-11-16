#ifndef MULTIGRIDTEST_DIFFUSION_H
#define MULTIGRIDTEST_DIFFUSION_H
#include "runner.h"
#include "harness.h"
#include <vector>
#include "LARE/shared_data.h"
#include "../Parabolic/helpers.h"
#include "tags.h"

// Define NO_BRAGINSKII_LARSON to turn off Larsen's flux limiter, and NO_BRAGINSKII_VAN_LEER to turn off the van Leer limiter of the cross conductances

namespace SAMS
{
	namespace pw = portableWrapper;

	/**
	 * @brief Braginskii thermal conduction along the magnetic field
	 *
	 * A level is anything with nx, ny, nz, coarsenX, coarsenY, coarsenZ, the cell ranges, dxc, dyc, dzc, rhs and data.
	 */
	template <typename T_eos = LARE::idealGas>
	struct braginskii
	{
		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;
		using hostLineArray = portableWrapper::hostArray<SAMS::T_dataType, 1>;

		/** @brief The equation of state, and the data of LARE that holds it */
		using eosType = T_eos;
		using lareData = typename LARE::LARE3D<T_eos>::simulationData;
		/** @brief The constants of one species */
		struct speciesParameters
		{
			/** @brief The name of the energy variable of the species */
			const char *energyName = nullptr;
			/** @brief Whether the species is the electrons, which sets the mass in the free streaming flux */
			bool electron = false;
			/** @brief Conductivity at T = 1 */
			SAMS::T_dataType kappa0 = 0.0;
			/** @brief Power of T in the conductivity */
			SAMS::T_dataType exponent = 2.5;
			/** @brief Softening of the magnetic field, below which the conduction is isotropic */
			SAMS::T_dataType fieldSoftening = 1.0e-6;
			/** @brief Free streaming heat flux at T = 1, which goes as T^(3/2) */
			SAMS::T_dataType limiterFlux = 0.0;
			/** @brief Exponent of the limiter. 2 gives Larsen's limiter */
			SAMS::T_dataType limiterExponent = 2.0;
			/** @brief Softening of the free streaming flux, which keeps the limiter finite where T is zero */
			SAMS::T_dataType limiterSoftening = 1.0e-10;
			/** @brief The fraction of the free streaming heat flux n k T (k T / m)^(1/2) that the limiter uses */
			SAMS::T_dataType limiterFraction = 0.32;
			/** @brief Whether limiterFlux was set explicitly. If it was not, it is calculated from limiterFraction before each solve */
			bool limiterFluxSet = false;
		};

		/** @brief The constants of the problem, which are in a data pack. There is one entry for each species that is solved, in the order of the subproblems */
		struct parameters
		{
			std::vector<speciesParameters> species;
		};

		using dataPack = SAMS::dataPacks::multiPack<parameters, lareData>;
		using packRefs = SAMS::packRefsOf_t<dataPack>;

		/** @brief Magnetic field, each component staggered in its own direction. Component i is on the high face of cell i */
		volumeArray bx, by, bz;

		/** @brief Parallel conductivity in each cell of the finest level */
		volumeArray kappa;

		/** @brief Density */
		volumeArray rho;

		/** @brief The number of species that are solved, which defaultValues sets */
		SAMS::T_sizeType speciesCount = 0;

		/** @brief The name of the energy variable of each subproblem, which defaultValues sets */
		std::array<const char *, 2> energyNames{};

		/** @brief The name of the temperature variable, which the species share */
		inline static constexpr auto temperatureName = SAMS::makeName("temperature_") + T_eos::name;

		/** @brief The conductivity tensor on the edges of each cell. Index i is the edge between cells i and i + 1, so index 0 is the lower edge of the domain. txy is the x flux from the y gradient */
		struct tensorData
		{
			volumeArray txx, txy, txz, tyx, tyy, tyz, tzx, tzy, tzz;
		};

		/** @brief The conductances of the edges, named like the tensor. cx(i) is between cells i and i + 1 */
		struct conductanceData
		{
			volumeArray cx, cy, cz;
			volumeArray cxy, cxz, cyx, cyz, czx, czy;
		};

		/** @brief The parallel conductivity */
		DEVICEPREFIX SAMS::T_dataType conductivity(const speciesParameters &pr, SAMS::T_dataType energy) const
		{
			return pr.kappa0 * pow(energy, pr.exponent);
		}

		/** @brief The harmonic mean, which is zero if either value is */
		DEVICEPREFIX SAMS::T_dataType harmonic(SAMS::T_dataType a, SAMS::T_dataType b) const
		{
			const SAMS::T_dataType sum = a + b;
			return sum > 0.0 ? 2.0 * a * b / sum : 0.0;
		}

		/** @brief The normal conductance of the x edge between cells i and i + 1 */
		DEVICEPREFIX SAMS::T_dataType faceX(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return component(i, j, k) * dy(j) * dz(k) / (0.5 * (dx(i) + dx(i + 1)));
		}

		/** @brief The normal conductance of the y edge between cells j and j + 1 */
		DEVICEPREFIX SAMS::T_dataType faceY(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return component(i, j, k) * dx(i) * dz(k) / (0.5 * (dy(j) + dy(j + 1)));
		}

		/** @brief The normal conductance of the z edge between cells k and k + 1 */
		DEVICEPREFIX SAMS::T_dataType faceZ(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return component(i, j, k) * dx(i) * dy(j) / (0.5 * (dz(k) + dz(k + 1)));
		}

		/** @brief The cross conductance of the x edge between cells i and i + 1 */
		DEVICEPREFIX SAMS::T_dataType crossX([[maybe_unused]] const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return component(i, j, k) * dy(j) * dz(k);
		}

		/** @brief The cross conductance of the y edge between cells j and j + 1 */
		DEVICEPREFIX SAMS::T_dataType crossY(const lineArray &dx, [[maybe_unused]] const lineArray &dy, const lineArray &dz, const volumeArray &component, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return component(i, j, k) * dx(i) * dz(k);
		}

		/** @brief The cross conductance of the z edge between cells k and k + 1 */
		DEVICEPREFIX SAMS::T_dataType crossZ(const lineArray &dx, const lineArray &dy, [[maybe_unused]] const lineArray &dz, const volumeArray &component, SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
		{
			return component(i, j, k) * dx(i) * dy(j);
		}

		/** @brief Larsen's flux limiter for the conductance of a face, given the unlimited flux through it and the temperature and density on it */
		DEVICEPREFIX SAMS::T_dataType limitConductance([[maybe_unused]] const speciesParameters &pr, SAMS::T_dataType conductance, [[maybe_unused]] SAMS::T_dataType flux, [[maybe_unused]] SAMS::T_dataType temperature, [[maybe_unused]] SAMS::T_dataType density, [[maybe_unused]] SAMS::T_indexType i, [[maybe_unused]] SAMS::T_indexType j, [[maybe_unused]] SAMS::T_indexType k, [[maybe_unused]] int direction) const
		{
#ifndef NO_BRAGINSKII_LARSON
			const SAMS::T_dataType freeStreaming = pr.limiterFlux * density * pow(temperature, 1.5) + pr.limiterSoftening;
			return conductance / pow(1.0 + pow(flux / freeStreaming, pr.limiterExponent), 1.0 / pr.limiterExponent);
#else
			return conductance;
#endif
		}

		/** @brief The van Leer limiter factor for the gradient across a cell, from the field either side of it. It is 1 for a smooth field and 0 at an extremum */
		DEVICEPREFIX static SAMS::T_dataType vanLeerFactor(SAMS::T_dataType lower, SAMS::T_dataType centre, SAMS::T_dataType upper)
		{
			const SAMS::T_dataType dm = centre - lower, dp = upper - centre;
			return dm * dp > 0.0 ? 4.0 * dm * dp / ((dm + dp) * (dm + dp)) : 0.0;
		}

		/** @brief The van Leer limiter factor for a face, which is the smaller of the factors of the two cells that share it */
		DEVICEPREFIX static SAMS::T_dataType vanLeerFace(SAMS::T_dataType lowerA, SAMS::T_dataType centreA, SAMS::T_dataType upperA, SAMS::T_dataType lowerB, SAMS::T_dataType centreB, SAMS::T_dataType upperB)
		{
			const SAMS::T_dataType a = vanLeerFactor(lowerA, centreA, upperA), b = vanLeerFactor(lowerB, centreB, upperB);
			return a < b ? a : b;
		}

		/** @brief Half the reciprocal of the central difference width at cell i */
		DEVICEPREFIX static SAMS::T_dataType halfReciprocalWidth(const lineArray &d, SAMS::T_indexType i)
		{
			return 0.5 / (0.5 * (d(i - 1) + d(i + 1)) + d(i));
		}

		/**
		 * @brief Set kappa and the tensor kappa (B B + eps^2 I) / (|B|^2 + eps^2) on the faces of the finest level
		 *
		 * B on a face has its normal component there and the mean of the four nearest values for the others. The normal tensor
		 * components use the harmonic mean of kappa across the face and the cross components the arithmetic mean.
		 */
		template <typename T_level>
		void setTensor(T_level &l, const speciesParameters pr) const
		{
			const braginskii p = *this;
			const volumeArray u0 = l.rhs, Bx = bx, By = by, Bz = bz, kap = kappa;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
			auto cellKernel = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
#ifdef CONSTANT_CONDUCTIVITY
				kap(i, j, k) = pr.kappa0;
#else
				kap(i, j, k) = p.conductivity(pr, u0(i, j, k));
#endif
			};
			pw::applyKernel(cellKernel, pw::Range(0, l.nx + 1), pw::Range(0, l.ny + 1), pw::Range(0, l.nz + 1));

			const volumeArray txx0 = l.data.tensor.txx, txy0 = l.data.tensor.txy, txz0 = l.data.tensor.txz;
			const volumeArray tyx0 = l.data.tensor.tyx, tyy0 = l.data.tensor.tyy, tyz0 = l.data.tensor.tyz;
			const volumeArray tzx0 = l.data.tensor.tzx, tzy0 = l.data.tensor.tzy, tzz0 = l.data.tensor.tzz;

			auto tensorX = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SAMS::T_dataType bxf = Bx(i, j, k);
				const SAMS::T_dataType byf = 0.25 * (By(i, j - 1, k) + By(i + 1, j - 1, k) + By(i, j, k) + By(i + 1, j, k));
				const SAMS::T_dataType bzf = 0.25 * (Bz(i, j, k - 1) + Bz(i + 1, j, k - 1) + Bz(i, j, k) + Bz(i + 1, j, k));
				const SAMS::T_dataType eps2 = pr.fieldSoftening * pr.fieldSoftening;
				const SAMS::T_dataType denominator = bxf * bxf + byf * byf + bzf * bzf + eps2;
				const SAMS::T_dataType normal = p.harmonic(kap(i, j, k), kap(i + 1, j, k)) / denominator;
				const SAMS::T_dataType cross = 0.5 * (kap(i, j, k) + kap(i + 1, j, k)) / denominator;
				const SAMS::T_dataType txx = normal * (bxf * bxf + eps2);
				const SAMS::T_dataType flux = txx * std::abs(u0(i, j, k) - u0(i + 1, j, k)) / (0.5 * (dx(i) + dx(i + 1)));
				const SAMS::T_dataType temperature = 0.5 * (u0(i, j, k) + u0(i + 1, j, k));
				const SAMS::T_dataType density = 0.5 * (p.rho(i, j, k) + p.rho(i + 1, j, k));
				txx0(i, j, k) = p.limitConductance(pr, txx, flux, temperature, density, i, j, k, 0);
#ifndef NO_BRAGINSKII_VAN_LEER
				const SAMS::T_dataType limitY = p.vanLeerFace(u0(i, j - 1, k), u0(i, j, k), u0(i, j + 1, k), u0(i + 1, j - 1, k), u0(i + 1, j, k), u0(i + 1, j + 1, k));
				const SAMS::T_dataType limitZ = p.vanLeerFace(u0(i, j, k - 1), u0(i, j, k), u0(i, j, k + 1), u0(i + 1, j, k - 1), u0(i + 1, j, k), u0(i + 1, j, k + 1));
#else
				const SAMS::T_dataType limitY = 1.0, limitZ = 1.0;
#endif
				txy0(i, j, k) = p.limitConductance(pr, limitY * cross * bxf * byf, flux, temperature, density, i, j, k, 0);
				txz0(i, j, k) = p.limitConductance(pr, limitZ * cross * bxf * bzf, flux, temperature, density, i, j, k, 0);
			};
			pw::applyKernel(tensorX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

			auto tensorY = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SAMS::T_dataType bxf = 0.25 * (Bx(i - 1, j, k) + Bx(i, j, k) + Bx(i - 1, j + 1, k) + Bx(i, j + 1, k));
				const SAMS::T_dataType byf = By(i, j, k);
				const SAMS::T_dataType bzf = 0.25 * (Bz(i, j, k - 1) + Bz(i, j + 1, k - 1) + Bz(i, j, k) + Bz(i, j + 1, k));
				const SAMS::T_dataType eps2 = pr.fieldSoftening * pr.fieldSoftening;
				const SAMS::T_dataType denominator = bxf * bxf + byf * byf + bzf * bzf + eps2;
				const SAMS::T_dataType normal = p.harmonic(kap(i, j, k), kap(i, j + 1, k)) / denominator;
				const SAMS::T_dataType cross = 0.5 * (kap(i, j, k) + kap(i, j + 1, k)) / denominator;
				const SAMS::T_dataType tyy = normal * (byf * byf + eps2);
				const SAMS::T_dataType flux = tyy * std::abs(u0(i, j, k) - u0(i, j + 1, k)) / (0.5 * (dy(j) + dy(j + 1)));
				const SAMS::T_dataType temperature = 0.5 * (u0(i, j, k) + u0(i, j + 1, k));
				const SAMS::T_dataType density = 0.5 * (p.rho(i, j, k) + p.rho(i, j + 1, k));
#ifndef NO_BRAGINSKII_VAN_LEER
				const SAMS::T_dataType limitX = p.vanLeerFace(u0(i - 1, j, k), u0(i, j, k), u0(i + 1, j, k), u0(i - 1, j + 1, k), u0(i, j + 1, k), u0(i + 1, j + 1, k));
				const SAMS::T_dataType limitZ = p.vanLeerFace(u0(i, j, k - 1), u0(i, j, k), u0(i, j, k + 1), u0(i, j + 1, k - 1), u0(i, j + 1, k), u0(i, j + 1, k + 1));
#else
				const SAMS::T_dataType limitX = 1.0, limitZ = 1.0;
#endif
				tyx0(i, j, k) = p.limitConductance(pr, limitX * cross * byf * bxf, flux, temperature, density, i, j, k, 1);
				tyy0(i, j, k) = p.limitConductance(pr, tyy, flux, temperature, density, i, j, k, 1);
				tyz0(i, j, k) = p.limitConductance(pr, limitZ * cross * byf * bzf, flux, temperature, density, i, j, k, 1);
			};
			pw::applyKernel(tensorY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

			auto tensorZ = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SAMS::T_dataType bxf = 0.25 * (Bx(i - 1, j, k) + Bx(i, j, k) + Bx(i - 1, j, k + 1) + Bx(i, j, k + 1));
				const SAMS::T_dataType byf = 0.25 * (By(i, j - 1, k) + By(i, j, k) + By(i, j - 1, k + 1) + By(i, j, k + 1));
				const SAMS::T_dataType bzf = Bz(i, j, k);
				const SAMS::T_dataType eps2 = pr.fieldSoftening * pr.fieldSoftening;
				const SAMS::T_dataType denominator = bxf * bxf + byf * byf + bzf * bzf + eps2;
				const SAMS::T_dataType normal = p.harmonic(kap(i, j, k), kap(i, j, k + 1)) / denominator;
				const SAMS::T_dataType cross = 0.5 * (kap(i, j, k) + kap(i, j, k + 1)) / denominator;
				const SAMS::T_dataType tzz = normal * (bzf * bzf + eps2);
				const SAMS::T_dataType flux = tzz * std::abs(u0(i, j, k) - u0(i, j, k + 1)) / (0.5 * (dz(k) + dz(k + 1)));
				const SAMS::T_dataType temperature = 0.5 * (u0(i, j, k) + u0(i, j, k + 1));
				const SAMS::T_dataType density = 0.5 * (p.rho(i, j, k) + p.rho(i, j, k + 1));
#ifndef NO_BRAGINSKII_VAN_LEER
				const SAMS::T_dataType limitX = p.vanLeerFace(u0(i - 1, j, k), u0(i, j, k), u0(i + 1, j, k), u0(i - 1, j, k + 1), u0(i, j, k + 1), u0(i + 1, j, k + 1));
				const SAMS::T_dataType limitY = p.vanLeerFace(u0(i, j - 1, k), u0(i, j, k), u0(i, j + 1, k), u0(i, j - 1, k + 1), u0(i, j, k + 1), u0(i, j + 1, k + 1));
#else
				const SAMS::T_dataType limitX = 1.0, limitY = 1.0;
#endif
				tzx0(i, j, k) = p.limitConductance(pr, limitX * cross * bzf * bxf, flux, temperature, density, i, j, k, 2);
				tzy0(i, j, k) = p.limitConductance(pr, limitY * cross * bzf * byf, flux, temperature, density, i, j, k, 2);
				tzz0(i, j, k) = p.limitConductance(pr, tzz, flux, temperature, density, i, j, k, 2);
			};
			pw::applyKernel(tensorZ, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zbLocalDomainRange);
		}

		/** @brief Restrict the tensor to the coarse levels */
		template <typename T_level>
		void restrictTensor(std::vector<T_level> &levels) const
		{
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.txx; }, 0);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.txy; }, 0);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.txz; }, 0);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tyx; }, 1);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tyy; }, 1);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tyz; }, 1);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tzx; }, 2);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tzy; }, 2);
			parabolic::restrictFaces(levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tzz; }, 2);
		}

		/** @brief Set the stored conductances on every level from the tensor, which already has the limiter in it */
		template <typename T_level>
		void setConductances(std::vector<T_level> &levels) const
		{
			const braginskii p = *this;
			for (SAMS::T_sizeType n = 0; n < levels.size(); ++n)
			{
				T_level &l = levels[n];
				const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
				const volumeArray txx = l.data.tensor.txx, txy = l.data.tensor.txy, txz = l.data.tensor.txz;
				const volumeArray tyx = l.data.tensor.tyx, tyy = l.data.tensor.tyy, tyz = l.data.tensor.tyz;
				const volumeArray tzx = l.data.tensor.tzx, tzy = l.data.tensor.tzy, tzz = l.data.tensor.tzz;
				const volumeArray cx = l.data.conductance.cx, cxy = l.data.conductance.cxy, cxz = l.data.conductance.cxz;
				const volumeArray cy = l.data.conductance.cy, cyx = l.data.conductance.cyx, cyz = l.data.conductance.cyz;
				const volumeArray cz = l.data.conductance.cz, czx = l.data.conductance.czx, czy = l.data.conductance.czy;

				auto conductanceX = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
				{
					cx(i, j, k) = p.faceX(dx, dy, dz, txx, i, j, k);
					cxy(i, j, k) = p.crossX(dx, dy, dz, txy, i, j, k);
					cxz(i, j, k) = p.crossX(dx, dy, dz, txz, i, j, k);
				};
				pw::applyKernel(conductanceX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceY = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
				{
					cy(i, j, k) = p.faceY(dx, dy, dz, tyy, i, j, k);
					cyx(i, j, k) = p.crossY(dx, dy, dz, tyx, i, j, k);
					cyz(i, j, k) = p.crossY(dx, dy, dz, tyz, i, j, k);
				};
				pw::applyKernel(conductanceY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceZ = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
				{
					cz(i, j, k) = p.faceZ(dx, dy, dz, tzz, i, j, k);
					czx(i, j, k) = p.crossZ(dx, dy, dz, tzx, i, j, k);
					czy(i, j, k) = p.crossZ(dx, dy, dz, tzy, i, j, k);
				};
				pw::applyKernel(conductanceZ, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zbLocalDomainRange);
			}
		}

		/** @brief Set beta from the density, heat capacity and timestep, and average it onto the coarse levels */
		template <typename T_level>
		void setBeta(std::vector<T_level> &levels, const lareData &lare, const SAMS::timeState &ts) const
		{
			T_level &l = levels[0];
			const auto &eos = lare.eos;
			const SAMS::T_dataType dt = ts.dt;
			const SAMS::T_dataType mass = lare.mf * lare.mh / lare.kb;
			const volumeArray density = rho, factor = l.data.beta, temperature = l.rhs;
			auto kernel = LAMBDA(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k)
			{
				const SAMS::T_dataType energy = eos.getEnergy(LARE::eosDensity(density(i, j, k)), LARE::eosTemperature(temperature(i, j, k)), LARE::eosMass(mass));
				factor(i, j, k) = density(i, j, k) * eos.getCV(LARE::eosDensity(density(i, j, k)), LARE::eosEnergy(energy), LARE::eosMass(mass)) / dt;
			};
			pw::applyKernel(kernel, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
			parabolic::restrictCells(levels, [](T_level &l) -> volumeArray & { return l.data.beta; });
		}

		/** @brief The name of the problem */
		inline static constexpr auto name = SAMS::makeName("Braginskii") + T_eos::name;

		/** @brief The name that the arrays and axes of the problem are registered under, which the species share */
		inline static constexpr auto memoryName = SAMS::makeName("Braginskii") + T_eos::name;

		/** @brief The name of the variable that is evolved */
		static constexpr const char *fieldName()
		{
			return temperatureName.c_str();
		}

		/** @brief The arrays for one level. The conductances are only allocated if the solver stores them */
		struct levelData
		{
			tensorData tensor;
			conductanceData conductance;
			/** @brief The prefactor of the time derivative per unit volume, divided by the timestep */
			volumeArray beta;
		};

		/** @brief The number of entries of the stencil: the cell, its six face neighbours and its twelve edge neighbours */
		static constexpr int stencilSize = 19;

		/** @brief The offset of each entry in the stencil, which includes the {0, 0, 0} centre element, in any order */
		DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
		{
			const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
				{-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}, {1, 1, 0}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1}, {1, 0, 1},
				{0, -1, -1}, {0, 1, -1}, {0, -1, 1}, {0, 1, 1}};
			di = table[s][0];
			dj = table[s][1];
			dk = table[s][2];
		}

		/** @brief The operator of a level, for use in kernels */
		struct stencil
		{
			volumeArray cx, cxy, cxz, cy, cyx, cyz, cz, czx, czy;
			volumeArray betaArray;
			lineArray dx, dy, dz;

			/** @brief The prefactor of the time derivative, divided by the timestep */
			DEVICEPREFIX SAMS::T_dataType beta(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
			{
				return betaArray(i, j, k);
			}

			/**
			 * @brief The weights of the fluxes out of a cell through its faces on the high side
			 *
			 * The entries are found from their positions relative to the cell, so at<1, 0, 0> is the cell at i + 1.
			 * Each flux is the conductance across the face times the difference of the two cells, plus the cross terms from the
			 * gradients along the face, which use the two cells on each side of the face.
			 */
			DEVICEPREFIX parabolic::fluxDivergence::faceWeights<braginskii> fluxWeights(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k) const
			{
				using parabolic::at;
				parabolic::fluxDivergence::faceWeights<braginskii> w{};
				const SAMS::T_dataType hx = halfReciprocalWidth(dx, i);
				const SAMS::T_dataType hy = halfReciprocalWidth(dy, j);
				const SAMS::T_dataType hz = halfReciprocalWidth(dz, k);

				const SAMS::T_dataType xx = cx(i, j, k), xy = hy * cxy(i, j, k), xz = hz * cxz(i, j, k);
				w.fluxWeightsX[at<0, 0, 0>] = xx;
				w.fluxWeightsX[at<0, -1, 0>] = xy;
				w.fluxWeightsX[at<0, 1, 0>] = -xy;
				w.fluxWeightsX[at<0, 0, -1>] = xz;
				w.fluxWeightsX[at<0, 0, 1>] = -xz;

				const SAMS::T_dataType yy = cy(i, j, k), yx = hx * cyx(i, j, k), yz = hz * cyz(i, j, k);
				w.fluxWeightsY[at<0, 0, 0>] = yy;
				w.fluxWeightsY[at<-1, 0, 0>] = yx;
				w.fluxWeightsY[at<1, 0, 0>] = -yx;
				w.fluxWeightsY[at<0, 0, -1>] = yz;
				w.fluxWeightsY[at<0, 0, 1>] = -yz;

				const SAMS::T_dataType zz = cz(i, j, k), zx = hx * czx(i, j, k), zy = hy * czy(i, j, k);
				w.fluxWeightsZ[at<0, 0, 0>] = zz;
				w.fluxWeightsZ[at<-1, 0, 0>] = zx;
				w.fluxWeightsZ[at<1, 0, 0>] = -zx;
				w.fluxWeightsZ[at<0, -1, 0>] = zy;
				w.fluxWeightsZ[at<0, 1, 0>] = -zy;
				w.makeSymmetric();
				return w;
			}

			/** @brief The coefficients of a cell */
			DEVICEPREFIX void coefficients(SAMS::T_indexType i, SAMS::T_indexType j, SAMS::T_indexType k, SAMS::T_dataType *a) const
			{
				parabolic::fluxDivergence::coefficientsFromFluxes<braginskii>(*this, i, j, k, a);
			}
		};

		/** @brief Set the boundary conditions. There are none */
		void setBoundaryConditions([[maybe_unused]] SAMS::harness &harness)
		{
		}

		/** @brief Get the variables */
		void getVariables(SAMS::harness &harness)
		{
			auto &varRegistry = harness.variableRegistry;
			varRegistry.fillPPArray("bx", bx);
			varRegistry.fillPPArray("by", by);
			varRegistry.fillPPArray("bz", bz);
			varRegistry.fillPPArray("rho", rho);
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
		 * An edge array is staggered in the direction of its edges. kappa is only on the finest level.
		 * @param axisLevels The number of derived axes that are used in each direction
		 */
		void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
		{
			const std::string memory = memoryName.c_str();
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "txx", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "txy", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "txz", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "tyx", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "tyy", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "tyz", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "tzx", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "tzy", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "tzz", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cx", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cxy", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cxz", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cy", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cyx", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cyz", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cz", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "czx", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "czy", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
			parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "beta", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
			parabolic::registerArray(harness, memory, fieldName(), axisLevels, "kappa", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
		}

		/** @brief Get the arrays of every level, from the first level to the last */
		template <typename T_level>
		void getSecondaryVariables(SAMS::harness &harness, std::vector<T_level> &levels)
		{
			const std::string memory = memoryName.c_str();
			parabolic::fillArrayLevels(harness, memory, "txx", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.txx; });
			parabolic::fillArrayLevels(harness, memory, "txy", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.txy; });
			parabolic::fillArrayLevels(harness, memory, "txz", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.txz; });
			parabolic::fillArrayLevels(harness, memory, "tyx", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tyx; });
			parabolic::fillArrayLevels(harness, memory, "tyy", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tyy; });
			parabolic::fillArrayLevels(harness, memory, "tyz", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tyz; });
			parabolic::fillArrayLevels(harness, memory, "tzx", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tzx; });
			parabolic::fillArrayLevels(harness, memory, "tzy", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tzy; });
			parabolic::fillArrayLevels(harness, memory, "tzz", levels, [](T_level &l) -> volumeArray & { return l.data.tensor.tzz; });
			parabolic::fillArrayLevels(harness, memory, "cx", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cx; });
			parabolic::fillArrayLevels(harness, memory, "cxy", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cxy; });
			parabolic::fillArrayLevels(harness, memory, "cxz", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cxz; });
			parabolic::fillArrayLevels(harness, memory, "cy", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cy; });
			parabolic::fillArrayLevels(harness, memory, "cyx", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cyx; });
			parabolic::fillArrayLevels(harness, memory, "cyz", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cyz; });
			parabolic::fillArrayLevels(harness, memory, "cz", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.cz; });
			parabolic::fillArrayLevels(harness, memory, "czx", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.czx; });
			parabolic::fillArrayLevels(harness, memory, "czy", levels, [](T_level &l) -> volumeArray & { return l.data.conductance.czy; });
			parabolic::fillArrayLevels(harness, memory, "beta", levels, [](T_level &l) -> volumeArray & { return l.data.beta; });
			parabolic::fillArray(harness, memory, "kappa", kappa);
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

		/** @brief The operator of a level. There is only one subproblem */
		template <typename T_level>
		stencil makeStencil(const T_level &l, [[maybe_unused]] SAMS::T_sizeType problemIndex) const
		{
			stencil s;
			s.cx = l.data.conductance.cx;
			s.cxy = l.data.conductance.cxy;
			s.cxz = l.data.conductance.cxz;
			s.cy = l.data.conductance.cy;
			s.cyx = l.data.conductance.cyx;
			s.cyz = l.data.conductance.cyz;
			s.cz = l.data.conductance.cz;
			s.czx = l.data.conductance.czx;
			s.czy = l.data.conductance.czy;
			s.betaArray = l.data.beta;
			s.dx = l.dxc;
			s.dy = l.dyc;
			s.dz = l.dzc;
			return s;
		}

		/** @brief Add a species to the species that are solved, with its parameters from the deck. The default conductivity is Spitzer's, in SI units */
		template <typename T_species>
		static void addSpecies(SAMS::KVS &kvs, parameters &params)
		{
			const std::string block = std::string("/ThermalConduction/") + T_species::name.c_str() + "/";
			const SAMS::T_dataType coulombLog = kvs.getReal("/ThermalConduction/coulomb_log", 20.0);
			speciesParameters sp;
			sp.energyName = T_species::fieldName();
			sp.electron = std::is_same_v<T_species, SAMS::electron>;
			sp.kappa0 = kvs.getReal(block + "kappa0", T_species::spitzerCoefficient / coulombLog);
			sp.exponent = kvs.getReal(block + "exponent", 2.5);
			sp.fieldSoftening = kvs.getReal(block + "field_softening", 1.0e-6);
			sp.limiterExponent = kvs.getReal(block + "limiter_exponent", 2.0);
			sp.limiterSoftening = kvs.getReal(block + "limiter_softening", 1.0e-10);
			sp.limiterFraction = kvs.getReal(block + "limiter_fraction", 0.32);
			//limiterFlux is calculated before each solve unless the deck sets it
			if (!kvs.getPath(block + "limiter_flux", SAMS::KVS::Element()).isType(SAMS::KVS::Type::None))
			{
				sp.limiterFlux = kvs.getReal(block + "limiter_flux");
				sp.limiterFluxSet = true;
			}
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
			speciesParameters pr = params.species[problemIndex];
			if (!pr.limiterFluxSet)
			{
				//The free streaming flux is limiterFraction n k T (k T / m)^(1/2), where n is the density divided by the ion mass
				const SAMS::T_dataType ionMass = lare.mf * LARE::mh_si;
				const SAMS::T_dataType speciesMass = pr.electron ? LARE::me_si : ionMass;
				pr.limiterFlux = pr.limiterFraction * pow(LARE::kb_si, 1.5) / (sqrt(speciesMass) * ionMass);
			}
			setTensor(levels[0], pr);
			restrictTensor(levels);
			setConductances(levels);
			setBeta(levels, lare, ts);
		}
	};
}

#endif // MULTIGRIDTEST_DIFFUSION_H
