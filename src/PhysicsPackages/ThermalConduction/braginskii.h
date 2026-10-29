#ifndef MULTIGRIDTEST_DIFFUSION_H
#define MULTIGRIDTEST_DIFFUSION_H
#include "runner.h"
#include "harness.h"
#include <vector>
#include "LARE/shared_data.h"
#include "../Parabolic/helpers.h"

// Define NO_BRAGINSKII_LARSON to turn off Larsen's flux limiter, and NO_BRAGINSKII_VAN_LEER to turn off the van Leer limiter of the cross conductances

namespace SAMS
{
	namespace pw = portableWrapper;

	/**
	 * @brief Braginskii thermal conduction along the magnetic field
	 *
	 * A level is anything with nx, ny, nz, coarsenX, coarsenY, coarsenZ, the cell ranges, dxc, dyc, dzc, rhs and data.
	 */
	template <typename T_species, typename T_eos = LARE::idealGas>
	struct braginskii
	{
		using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
		using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;
		using hostLineArray = portableWrapper::hostArray<SAMS::T_dataType, 1>;

		/** @brief The equation of state, and the data of LARE that holds it */
		using eosType = T_eos;
		using species = T_species;
		using lareData = typename LARE::LARE3D<T_eos>::simulationData;
		/** @brief The constants of the problem, which are in a data pack */
		struct parameters
		{
			/** @brief Conductivity at T = 1 */
			double kappa0;
			/** @brief Power of T in the conductivity */
			double exponent;
			/** @brief Softening of the magnetic field, below which the conduction is isotropic */
			double fieldSoftening;
			/** @brief Free streaming heat flux at T = 1, which goes as T^(3/2) */
			double limiterFlux;
			/** @brief Exponent of the limiter. 2 gives Larsen's limiter */
			double limiterExponent;
			/** @brief Softening of the free streaming flux, which keeps the limiter finite where T is zero */
			double limiterSoftening;
		};

		using dataPack = SAMS::dataPacks::multiPack<parameters, lareData>;
		using packRefs = SAMS::packRefsOf_t<dataPack>;

		/** @brief Magnetic field, each component staggered in its own direction. Component i is on the high face of cell i */
		volumeArray bx, by, bz;

		/** @brief Parallel conductivity in each cell of the finest level */
		volumeArray kappa;

		/** @brief Density */
		volumeArray rho;

		/** @brief The name of the temperature variable */
		inline static constexpr auto temperatureName = SAMS::constexprName("temperature_") + T_species::name + T_eos::name;

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
		DEVICEPREFIX double conductivity(const parameters &pr, double energy) const
		{
			return pr.kappa0 * pow(energy, pr.exponent);
		}

		/** @brief The harmonic mean, which is zero if either value is */
		DEVICEPREFIX double harmonic(double a, double b) const
		{
			const double sum = a + b;
			return sum > 0.0 ? 2.0 * a * b / sum : 0.0;
		}

		/** @brief The normal conductance of the x edge between cells i and i + 1 */
		DEVICEPREFIX double faceX(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, size_t i, size_t j, size_t k) const
		{
			return component(i, j, k) * dy(j) * dz(k) / (0.5 * (dx(i) + dx(i + 1)));
		}

		/** @brief The normal conductance of the y edge between cells j and j + 1 */
		DEVICEPREFIX double faceY(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, size_t i, size_t j, size_t k) const
		{
			return component(i, j, k) * dx(i) * dz(k) / (0.5 * (dy(j) + dy(j + 1)));
		}

		/** @brief The normal conductance of the z edge between cells k and k + 1 */
		DEVICEPREFIX double faceZ(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, size_t i, size_t j, size_t k) const
		{
			return component(i, j, k) * dx(i) * dy(j) / (0.5 * (dz(k) + dz(k + 1)));
		}

		/** @brief The cross conductance of the x edge between cells i and i + 1 */
		DEVICEPREFIX double crossX(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, size_t i, size_t j, size_t k) const
		{
			return component(i, j, k) * dy(j) * dz(k);
		}

		/** @brief The cross conductance of the y edge between cells j and j + 1 */
		DEVICEPREFIX double crossY(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, size_t i, size_t j, size_t k) const
		{
			return component(i, j, k) * dx(i) * dz(k);
		}

		/** @brief The cross conductance of the z edge between cells k and k + 1 */
		DEVICEPREFIX double crossZ(const lineArray &dx, const lineArray &dy, const lineArray &dz, const volumeArray &component, size_t i, size_t j, size_t k) const
		{
			return component(i, j, k) * dx(i) * dy(j);
		}

		/** @brief Larsen's flux limiter for the conductance of a face, given the unlimited flux through it and the temperature on it */
		DEVICEPREFIX double limitConductance(const parameters &pr, double conductance, double flux, double temperature, size_t i, size_t j, size_t k, int direction) const
		{
#ifndef NO_BRAGINSKII_LARSON
			const double freeStreaming = pr.limiterFlux * pow(temperature, 1.5) + pr.limiterSoftening;
			return conductance / pow(1.0 + pow(flux / freeStreaming, pr.limiterExponent), 1.0 / pr.limiterExponent);
#else
			return conductance;
#endif
		}

		/** @brief The van Leer limiter factor for the gradient across a cell, from the field either side of it. It is 1 for a smooth field and 0 at an extremum */
		DEVICEPREFIX static double vanLeerFactor(double lower, double centre, double upper)
		{
			const double dm = centre - lower, dp = upper - centre;
			return dm * dp > 0.0 ? 4.0 * dm * dp / ((dm + dp) * (dm + dp)) : 0.0;
		}

		/** @brief The van Leer limiter factor for a face, which is the smaller of the factors of the two cells that share it */
		DEVICEPREFIX static double vanLeerFace(double lowerA, double centreA, double upperA, double lowerB, double centreB, double upperB)
		{
			const double a = vanLeerFactor(lowerA, centreA, upperA), b = vanLeerFactor(lowerB, centreB, upperB);
			return a < b ? a : b;
		}

		/** @brief Half the reciprocal of the central difference width at cell i */
		DEVICEPREFIX static double halfReciprocalWidth(const lineArray &d, size_t i)
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
		void setTensor(T_level &l, const parameters pr) const
		{
			const braginskii p = *this;
			const volumeArray u0 = l.rhs, Bx = bx, By = by, Bz = bz, kap = kappa;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
			auto cellKernel = LAMBDA(size_t i, size_t j, size_t k)
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

			auto tensorX = LAMBDA(size_t i, size_t j, size_t k)
			{
				const double bxf = Bx(i, j, k);
				const double byf = 0.25 * (By(i, j - 1, k) + By(i + 1, j - 1, k) + By(i, j, k) + By(i + 1, j, k));
				const double bzf = 0.25 * (Bz(i, j, k - 1) + Bz(i + 1, j, k - 1) + Bz(i, j, k) + Bz(i + 1, j, k));
				const double eps2 = pr.fieldSoftening * pr.fieldSoftening;
				const double denominator = bxf * bxf + byf * byf + bzf * bzf + eps2;
				const double normal = p.harmonic(kap(i, j, k), kap(i + 1, j, k)) / denominator;
				const double cross = 0.5 * (kap(i, j, k) + kap(i + 1, j, k)) / denominator;
				const double txx = normal * (bxf * bxf + eps2);
				const double flux = txx * std::abs(u0(i, j, k) - u0(i + 1, j, k)) / (0.5 * (dx(i) + dx(i + 1)));
				const double temperature = 0.5 * (u0(i, j, k) + u0(i + 1, j, k));
				txx0(i, j, k) = p.limitConductance(pr, txx, flux, temperature, i, j, k, 0);
#ifndef NO_BRAGINSKII_VAN_LEER
				const double limitY = p.vanLeerFace(u0(i, j - 1, k), u0(i, j, k), u0(i, j + 1, k), u0(i + 1, j - 1, k), u0(i + 1, j, k), u0(i + 1, j + 1, k));
				const double limitZ = p.vanLeerFace(u0(i, j, k - 1), u0(i, j, k), u0(i, j, k + 1), u0(i + 1, j, k - 1), u0(i + 1, j, k), u0(i + 1, j, k + 1));
#else
				const double limitY = 1.0, limitZ = 1.0;
#endif
				txy0(i, j, k) = p.limitConductance(pr, limitY * cross * bxf * byf, flux, temperature, i, j, k, 0);
				txz0(i, j, k) = p.limitConductance(pr, limitZ * cross * bxf * bzf, flux, temperature, i, j, k, 0);
			};
			pw::applyKernel(tensorX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

			auto tensorY = LAMBDA(size_t i, size_t j, size_t k)
			{
				const double bxf = 0.25 * (Bx(i - 1, j, k) + Bx(i, j, k) + Bx(i - 1, j + 1, k) + Bx(i, j + 1, k));
				const double byf = By(i, j, k);
				const double bzf = 0.25 * (Bz(i, j, k - 1) + Bz(i, j + 1, k - 1) + Bz(i, j, k) + Bz(i, j + 1, k));
				const double eps2 = pr.fieldSoftening * pr.fieldSoftening;
				const double denominator = bxf * bxf + byf * byf + bzf * bzf + eps2;
				const double normal = p.harmonic(kap(i, j, k), kap(i, j + 1, k)) / denominator;
				const double cross = 0.5 * (kap(i, j, k) + kap(i, j + 1, k)) / denominator;
				const double tyy = normal * (byf * byf + eps2);
				const double flux = tyy * std::abs(u0(i, j, k) - u0(i, j + 1, k)) / (0.5 * (dy(j) + dy(j + 1)));
				const double temperature = 0.5 * (u0(i, j, k) + u0(i, j + 1, k));
#ifndef NO_BRAGINSKII_VAN_LEER
				const double limitX = p.vanLeerFace(u0(i - 1, j, k), u0(i, j, k), u0(i + 1, j, k), u0(i - 1, j + 1, k), u0(i, j + 1, k), u0(i + 1, j + 1, k));
				const double limitZ = p.vanLeerFace(u0(i, j, k - 1), u0(i, j, k), u0(i, j, k + 1), u0(i, j + 1, k - 1), u0(i, j + 1, k), u0(i, j + 1, k + 1));
#else
				const double limitX = 1.0, limitZ = 1.0;
#endif
				tyx0(i, j, k) = p.limitConductance(pr, limitX * cross * byf * bxf, flux, temperature, i, j, k, 1);
				tyy0(i, j, k) = p.limitConductance(pr, tyy, flux, temperature, i, j, k, 1);
				tyz0(i, j, k) = p.limitConductance(pr, limitZ * cross * byf * bzf, flux, temperature, i, j, k, 1);
			};
			pw::applyKernel(tensorY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

			auto tensorZ = LAMBDA(size_t i, size_t j, size_t k)
			{
				const double bxf = 0.25 * (Bx(i - 1, j, k) + Bx(i, j, k) + Bx(i - 1, j, k + 1) + Bx(i, j, k + 1));
				const double byf = 0.25 * (By(i, j - 1, k) + By(i, j, k) + By(i, j - 1, k + 1) + By(i, j, k + 1));
				const double bzf = Bz(i, j, k);
				const double eps2 = pr.fieldSoftening * pr.fieldSoftening;
				const double denominator = bxf * bxf + byf * byf + bzf * bzf + eps2;
				const double normal = p.harmonic(kap(i, j, k), kap(i, j, k + 1)) / denominator;
				const double cross = 0.5 * (kap(i, j, k) + kap(i, j, k + 1)) / denominator;
				const double tzz = normal * (bzf * bzf + eps2);
				const double flux = tzz * std::abs(u0(i, j, k) - u0(i, j, k + 1)) / (0.5 * (dz(k) + dz(k + 1)));
				const double temperature = 0.5 * (u0(i, j, k) + u0(i, j, k + 1));
#ifndef NO_BRAGINSKII_VAN_LEER
				const double limitX = p.vanLeerFace(u0(i - 1, j, k), u0(i, j, k), u0(i + 1, j, k), u0(i - 1, j, k + 1), u0(i, j, k + 1), u0(i + 1, j, k + 1));
				const double limitY = p.vanLeerFace(u0(i, j - 1, k), u0(i, j, k), u0(i, j + 1, k), u0(i, j - 1, k + 1), u0(i, j, k + 1), u0(i, j + 1, k + 1));
#else
				const double limitX = 1.0, limitY = 1.0;
#endif
				tzx0(i, j, k) = p.limitConductance(pr, limitX * cross * bzf * bxf, flux, temperature, i, j, k, 2);
				tzy0(i, j, k) = p.limitConductance(pr, limitY * cross * bzf * byf, flux, temperature, i, j, k, 2);
				tzz0(i, j, k) = p.limitConductance(pr, tzz, flux, temperature, i, j, k, 2);
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
			for (size_t n = 0; n < levels.size(); ++n)
			{
				T_level &l = levels[n];
				const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;
				const volumeArray txx = l.data.tensor.txx, txy = l.data.tensor.txy, txz = l.data.tensor.txz;
				const volumeArray tyx = l.data.tensor.tyx, tyy = l.data.tensor.tyy, tyz = l.data.tensor.tyz;
				const volumeArray tzx = l.data.tensor.tzx, tzy = l.data.tensor.tzy, tzz = l.data.tensor.tzz;
				const volumeArray cx = l.data.conductance.cx, cxy = l.data.conductance.cxy, cxz = l.data.conductance.cxz;
				const volumeArray cy = l.data.conductance.cy, cyx = l.data.conductance.cyx, cyz = l.data.conductance.cyz;
				const volumeArray cz = l.data.conductance.cz, czx = l.data.conductance.czx, czy = l.data.conductance.czy;

				auto conductanceX = LAMBDA(size_t i, size_t j, size_t k)
				{
					cx(i, j, k) = p.faceX(dx, dy, dz, txx, i, j, k);
					cxy(i, j, k) = p.crossX(dx, dy, dz, txy, i, j, k);
					cxz(i, j, k) = p.crossX(dx, dy, dz, txz, i, j, k);
				};
				pw::applyKernel(conductanceX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceY = LAMBDA(size_t i, size_t j, size_t k)
				{
					cy(i, j, k) = p.faceY(dx, dy, dz, tyy, i, j, k);
					cyx(i, j, k) = p.crossY(dx, dy, dz, tyx, i, j, k);
					cyz(i, j, k) = p.crossY(dx, dy, dz, tyz, i, j, k);
				};
				pw::applyKernel(conductanceY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

				auto conductanceZ = LAMBDA(size_t i, size_t j, size_t k)
				{
					cz(i, j, k) = p.faceZ(dx, dy, dz, tzz, i, j, k);
					czx(i, j, k) = p.crossZ(dx, dy, dz, tzx, i, j, k);
					czy(i, j, k) = p.crossZ(dx, dy, dz, tzy, i, j, k);
				};
				pw::applyKernel(conductanceZ, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zbLocalDomainRange);
			}
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
			parabolic::restrictCells(levels, [](T_level &l) -> volumeArray & { return l.data.beta; });
		}

		/** @brief The name of the problem, which includes the species */
		inline static constexpr auto name = SAMS::constexprName("braginskii_") + T_species::name + T_eos::name;

		/** @brief The name that the arrays and axes of the problem are registered under, which the species share */
		inline static constexpr auto memoryName = SAMS::constexprName("braginskii") + T_eos::name;

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
			/** @brief The prefactor of the time derivative, per unit volume */
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

			/** @brief The prefactor of the time derivative */
			DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
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
			DEVICEPREFIX parabolic::fluxDivergence::faceWeights<braginskii> fluxWeights(size_t i, size_t j, size_t k) const
			{
				using parabolic::at;
				parabolic::fluxDivergence::faceWeights<braginskii> w{};
				const double hx = halfReciprocalWidth(dx, i);
				const double hy = halfReciprocalWidth(dy, j);
				const double hz = halfReciprocalWidth(dz, k);

				const double xx = cx(i, j, k), xy = hy * cxy(i, j, k), xz = hz * cxz(i, j, k);
				w.fluxWeightsX[at<0, 0, 0>] = xx;
				w.fluxWeightsX[at<0, -1, 0>] = xy;
				w.fluxWeightsX[at<0, 1, 0>] = -xy;
				w.fluxWeightsX[at<0, 0, -1>] = xz;
				w.fluxWeightsX[at<0, 0, 1>] = -xz;

				const double yy = cy(i, j, k), yx = hx * cyx(i, j, k), yz = hz * cyz(i, j, k);
				w.fluxWeightsY[at<0, 0, 0>] = yy;
				w.fluxWeightsY[at<-1, 0, 0>] = yx;
				w.fluxWeightsY[at<1, 0, 0>] = -yx;
				w.fluxWeightsY[at<0, 0, -1>] = yz;
				w.fluxWeightsY[at<0, 0, 1>] = -yz;

				const double zz = cz(i, j, k), zx = hx * czx(i, j, k), zy = hy * czy(i, j, k);
				w.fluxWeightsZ[at<0, 0, 0>] = zz;
				w.fluxWeightsZ[at<-1, 0, 0>] = zx;
				w.fluxWeightsZ[at<1, 0, 0>] = -zx;
				w.fluxWeightsZ[at<0, -1, 0>] = zy;
				w.fluxWeightsZ[at<0, 1, 0>] = -zy;
				w.makeSymmetric();
				return w;
			}

			/** @brief The coefficients of a cell */
			DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
			{
				parabolic::fluxDivergence::coefficientsFromFluxes<braginskii>(*this, i, j, k, a);
			}
		};

		/** @brief Set the boundary conditions. There are none */
		void setBoundaryConditions(SAMS::harness &harness)
		{
		}

		/** @brief Get the variables */
		void getVariables(SAMS::harness &harness)
		{
			auto &varRegistry = harness.variableRegistry;
			varRegistry.fillPPArray("Bx", bx);
			varRegistry.fillPPArray("By", by);
			varRegistry.fillPPArray("Bz", bz);
			varRegistry.fillPPArray("rho", rho);
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

		/** @brief Populate the arrays of levelData that are needed, on every level. rhs of the first level is the old field */
		template <typename T_level>
		void prepareSolve(std::vector<T_level> &levels, const parameters &params, const lareData &lare) const
		{
			setTensor(levels[0], params);
			restrictTensor(levels);
			setConductances(levels);
			setBeta(levels, lare);
		}
	};
}

#endif // MULTIGRIDTEST_DIFFUSION_H
