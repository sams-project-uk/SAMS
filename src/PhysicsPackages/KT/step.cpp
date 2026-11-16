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
#include "kt.h"

namespace KT
{

    namespace{

        /**
         * Number of conserved variables
         */
        constexpr int nVar = 8;

        /**
         * Limited slope from the differences either side of a cell (van Leer)
         */
        DEVICEPREFIX INLINE T_dataType limiter(T_dataType left, T_dataType right)
        {
            if (left * right <= 0.0) return 0.0;
            return 2.0 * left * right / (left + right);
        }

        /**
         * Primitive variables (rho, vx, vy, vz, p, bx, by, bz) of a cell
         */
        DEVICEPREFIX INLINE void primitive(const conservedVariables &U, T_dataType gas_gamma, T_indexType ix, T_indexType iy, T_indexType iz, T_dataType q[nVar])
        {
            const T_dataType rho = U.rho(ix, iy, iz);
            const T_dataType vx = U.mx(ix, iy, iz) / rho;
            const T_dataType vy = U.my(ix, iy, iz) / rho;
            const T_dataType vz = U.mz(ix, iy, iz) / rho;
            const T_dataType bx = U.bx(ix, iy, iz);
            const T_dataType by = U.by(ix, iy, iz);
            const T_dataType bz = U.bz(ix, iy, iz);
            q[0] = rho;
            q[1] = vx;
            q[2] = vy;
            q[3] = vz;
            q[4] = (gas_gamma - 1.0) * (U.energy(ix, iy, iz) - 0.5 * rho * (vx * vx + vy * vy + vz * vz) - 0.5 * (bx * bx + by * by + bz * bz));
            q[5] = bx;
            q[6] = by;
            q[7] = bz;
        }

        /**
         * Conserved variables, flux, normal velocity and fast magnetosonic speed of a state in a direction
         */
        DEVICEPREFIX INLINE void state(const T_dataType q[nVar], int direction, T_dataType gas_gamma, T_dataType u[nVar], T_dataType f[nVar], T_dataType &vn, T_dataType &cf)
        {
            const T_dataType rho = q[0];
            const T_dataType p = q[4];
            const T_dataType v[3] = {q[1], q[2], q[3]};
            const T_dataType b[3] = {q[5], q[6], q[7]};
            const T_dataType v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            const T_dataType b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
            const T_dataType vb = v[0] * b[0] + v[1] * b[1] + v[2] * b[2];
            const T_dataType ptotal = p + 0.5 * b2;
            const T_dataType energy = p / (gas_gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2;
            vn = v[direction];
            const T_dataType bn = b[direction];

            u[0] = rho;
            u[4] = energy;
            f[0] = rho * vn;
            f[4] = (energy + ptotal) * vn - bn * vb;
            for (int c = 0; c < 3; ++c){
                u[1 + c] = rho * v[c];
                u[5 + c] = b[c];
                f[1 + c] = rho * v[c] * vn - b[c] * bn + (c == direction ? ptotal : 0.0);
                f[5 + c] = vn * b[c] - v[c] * bn;
            }

            const T_dataType a2 = gas_gamma * p / rho;
            const T_dataType ca2 = b2 / rho;
            const T_dataType cn2 = bn * bn / rho;
            cf = std::sqrt(0.5 * (a2 + ca2 + std::sqrt(pw::max((a2 + ca2) * (a2 + ca2) - 4.0 * a2 * cn2, T_dataType(0.0)))));
        }

        /**
         * Kurganov-Tadmor flux between a left and a right state in a direction, with the largest local wave speed of the two states.
         * The flux of the normal magnetic field is zero
         */
        DEVICEPREFIX INLINE void kt(const T_dataType qL[nVar], const T_dataType qR[nVar], int direction, T_dataType gas_gamma, T_dataType f[nVar])
        {
            T_dataType uL[nVar], fL[nVar], uR[nVar], fR[nVar];
            T_dataType vnL, cfL, vnR, cfR;
            state(qL, direction, gas_gamma, uL, fL, vnL, cfL);
            state(qR, direction, gas_gamma, uR, fR, vnR, cfR);
            const T_dataType a = pw::max(std::abs(vnL) + cfL, std::abs(vnR) + cfR);
            for (int v = 0; v < nVar; ++v){
                f[v] = 0.5 * (fL[v] + fR[v]) - 0.5 * a * (uR[v] - uL[v]);
            }
            f[5 + direction] = 0.0;
        }
    }

    /**
     * Compute the Kurganov-Tadmor fluxes on the faces normal to one direction. Face i + 1/2 is at index i
     * The fluxes are computed on the domain faces in the direction and one ghost cell beyond the domain in the other directions, which flux-CT needs
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::compute_fluxes(simulationData &data, int direction, conservedVariables &flux)
    {
        using Range = pw::Range;
        const conservedVariables U = data.U;
        const conservedVariables F = flux;
        const T_dataType gas_gamma = data.gas_gamma;
        const T_indexType dix = direction == 0 ? 1 : 0;
        const T_indexType diy = direction == 1 ? 1 : 0;
        const T_indexType diz = direction == 2 ? 1 : 0;

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_dataType qm[nVar], q0[nVar], q1[nVar], q2[nVar];
            primitive(U, gas_gamma, ix - dix, iy - diy, iz - diz, qm);
            primitive(U, gas_gamma, ix, iy, iz, q0);
            primitive(U, gas_gamma, ix + dix, iy + diy, iz + diz, q1);
            primitive(U, gas_gamma, ix + 2 * dix, iy + 2 * diy, iz + 2 * diz, q2);

            T_dataType qL[nVar], qR[nVar];
            for (int v = 0; v < nVar; ++v){
                qL[v] = q0[v] + 0.5 * limiter(q0[v] - qm[v], q1[v] - q0[v]);
                qR[v] = q1[v] - 0.5 * limiter(q1[v] - q0[v], q2[v] - q1[v]);
            }
            // The normal magnetic field is continuous across the face
            const T_dataType bn = 0.5 * (qL[5 + direction] + qR[5 + direction]);
            qL[5 + direction] = bn;
            qR[5 + direction] = bn;

            T_dataType f[nVar];
            kt(qL, qR, direction, gas_gamma, f);
            F.rho(ix, iy, iz) = f[0];
            F.mx(ix, iy, iz) = f[1];
            F.my(ix, iy, iz) = f[2];
            F.mz(ix, iy, iz) = f[3];
            F.energy(ix, iy, iz) = f[4];
            F.bx(ix, iy, iz) = f[5];
            F.by(ix, iy, iz) = f[6];
            F.bz(ix, iy, iz) = f[7];
        },
        direction == 0 ? Range(0, data.nx) : Range(0, data.nx + 1),
        direction == 1 ? Range(0, data.ny) : Range(0, data.ny + 1),
        direction == 2 ? Range(0, data.nz) : Range(0, data.nz + 1));
    }

    /**
     * Replace the fluxes of the magnetic field with those of Toth's flux-CT. The electric field at each cell edge is the average
     * of the four face fluxes around the edge, and the flux on each face is the average of the electric field at the two edges of the face
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::constrained_transport(simulationData &data)
    {
        using Range = pw::Range;
        const conservedVariables fx = data.fx;
        const conservedVariables fy = data.fy;
        const conservedVariables fz = data.fz;
        const volumeArray ex = data.ex;
        const volumeArray ey = data.ey;
        const volumeArray ez = data.ez;

        // Ex = -(vy bz - vz by) = -fy(bz) = fz(by)
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            ex(ix, iy, iz) = 0.25 * (-fy.bz(ix, iy, iz) - fy.bz(ix, iy, iz + 1) + fz.by(ix, iy, iz) + fz.by(ix, iy + 1, iz));
        }, Range(1, data.nx), Range(0, data.ny), Range(0, data.nz));

        // Ey = -(vz bx - vx bz) = -fz(bx) = fx(bz)
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            ey(ix, iy, iz) = 0.25 * (-fz.bx(ix, iy, iz) - fz.bx(ix + 1, iy, iz) + fx.bz(ix, iy, iz) + fx.bz(ix, iy, iz + 1));
        }, Range(0, data.nx), Range(1, data.ny), Range(0, data.nz));

        // Ez = -(vx by - vy bx) = -fx(by) = fy(bx)
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            ez(ix, iy, iz) = 0.25 * (-fx.by(ix, iy, iz) - fx.by(ix, iy + 1, iz) + fy.bx(ix, iy, iz) + fy.bx(ix + 1, iy, iz));
        }, Range(0, data.nx), Range(0, data.ny), Range(1, data.nz));

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            fx.by(ix, iy, iz) = -0.5 * (ez(ix, iy, iz) + ez(ix, iy - 1, iz));
            fx.bz(ix, iy, iz) = 0.5 * (ey(ix, iy, iz) + ey(ix, iy, iz - 1));
        }, Range(0, data.nx), Range(1, data.ny), Range(1, data.nz));

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            fy.bx(ix, iy, iz) = 0.5 * (ez(ix, iy, iz) + ez(ix - 1, iy, iz));
            fy.bz(ix, iy, iz) = -0.5 * (ex(ix, iy, iz) + ex(ix, iy, iz - 1));
        }, Range(1, data.nx), Range(0, data.ny), Range(1, data.nz));

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            fz.bx(ix, iy, iz) = -0.5 * (ey(ix, iy, iz) + ey(ix - 1, iy, iz));
            fz.by(ix, iy, iz) = 0.5 * (ex(ix, iy, iz) + ex(ix, iy - 1, iz));
        }, Range(1, data.nx), Range(1, data.ny), Range(0, data.nz));
    }

    /**
     * Update the conserved variables as weightOld * U0 + weightNew * (U - dt * div F) in the domain
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::update(simulationData &data, T_dataType weightOld, T_dataType weightNew)
    {
        using Range = pw::Range;
        const conservedVariables U = data.U;
        const conservedVariables U0 = data.U0;
        const conservedVariables fx = data.fx;
        const conservedVariables fy = data.fy;
        const conservedVariables fz = data.fz;
        const lineArray dxb = data.dxb;
        const lineArray dyb = data.dyb;
        const lineArray dzb = data.dzb;
        const T_dataType dt = data.dt;

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            const T_dataType rx = dt / dxb(ix);
            const T_dataType ry = dt / dyb(iy);
            const T_dataType rz = dt / dzb(iz);
            auto advance = [&](const volumeArray &u, const volumeArray &u0, const volumeArray &gx, const volumeArray &gy, const volumeArray &gz){
                const T_dataType updated = u(ix, iy, iz)
                    - rx * (gx(ix, iy, iz) - gx(ix - 1, iy, iz))
                    - ry * (gy(ix, iy, iz) - gy(ix, iy - 1, iz))
                    - rz * (gz(ix, iy, iz) - gz(ix, iy, iz - 1));
                u(ix, iy, iz) = weightOld * u0(ix, iy, iz) + weightNew * updated;
            };
            advance(U.rho, U0.rho, fx.rho, fy.rho, fz.rho);
            advance(U.mx, U0.mx, fx.mx, fy.mx, fz.mx);
            advance(U.my, U0.my, fx.my, fy.my, fz.my);
            advance(U.mz, U0.mz, fx.mz, fy.mz, fz.mz);
            advance(U.energy, U0.energy, fx.energy, fy.energy, fz.energy);
            advance(U.bx, U0.bx, fx.bx, fy.bx, fz.bx);
            advance(U.by, U0.by, fx.by, fy.by, fz.by);
            advance(U.bz, U0.bz, fx.bz, fy.bz, fz.bz);
        }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
    }

    /**
     * Compute the fluxes and update the conserved variables for one stage of the Runge-Kutta step
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::stage(simulationData &data, T_dataType weightOld, T_dataType weightNew)
    {
        boundary_conditions();
        compute_fluxes(data, 0, data.fx);
        compute_fluxes(data, 1, data.fy);
        compute_fluxes(data, 2, data.fz);
        constrained_transport(data);
        update(data, weightOld, weightNew);
    }

    /**
     * First stage of the Runge-Kutta step: U = U0 + dt L(U0)
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::startOfTimestep(simulationData &data)
    {
        pw::assign(data.U0.rho, data.U.rho);
        pw::assign(data.U0.mx, data.U.mx);
        pw::assign(data.U0.my, data.U.my);
        pw::assign(data.U0.mz, data.U.mz);
        pw::assign(data.U0.energy, data.U.energy);
        pw::assign(data.U0.bx, data.U.bx);
        pw::assign(data.U0.by, data.U.by);
        pw::assign(data.U0.bz, data.U.bz);
        stage(data, 0.0, 1.0);
    }

    /**
     * Second stage of the Runge-Kutta step: U = (U0 + U + dt L(U)) / 2
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::halfTimestep(simulationData &data)
    {
        stage(data, 0.5, 0.5);
    }

    /**
     * Set the timestep from the fast magnetosonic speed in each direction
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::calculateTimestep(SAMS::timeState &timeData, simulationData &data)
    {
        using Range = pw::Range;
        const conservedVariables U = data.U;
        const lineArray dxb = data.dxb;
        const lineArray dyb = data.dyb;
        const lineArray dzb = data.dzb;
        const T_dataType gas_gamma = data.gas_gamma;

        const T_dataType dt = data.cfl * pw::applyReduction(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_dataType q[nVar], u[nVar], f[nVar];
            primitive(U, gas_gamma, ix, iy, iz, q);
            const T_dataType width[3] = {dxb(ix), dyb(iy), dzb(iz)};
            // The unsplit update is stable when the sum of the Courant numbers in the three directions is below one
            T_dataType rate = 0.0;
            for (int direction = 0; direction < 3; ++direction){
                T_dataType vn, cf;
                state(q, direction, gas_gamma, u, f, vn, cf);
                rate += (std::abs(vn) + cf) / width[direction];
            }
            return 1.0 / rate;
        }, LAMBDA(T_dataType & a, const T_dataType &b) { a = pw::min(a, b); }, std::numeric_limits<T_dataType>::max(), Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));

        timeData.dt = dt < timeData.dt ? dt : timeData.dt;
    }

    #include "./instantiate.h"
}
