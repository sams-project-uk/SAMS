#include "general_remap_neutral.hpp"

namespace LARE_neutral {
    namespace pw = portableWrapper;
    using idx_t = T_indexType;
    using fp_t = T_dataType;

    template <AxisName axis>
    DEVICEPREFIX volumeArray v1_component(simulationData& data) {
        if constexpr (axis == AxisName::X) {
            return data.vx1;
        } 
        if constexpr (axis == AxisName::Y) {
            return data.vy1;
        }
        return data.vz1;
    }

    template <AxisName axis>
    DEVICEPREFIX volumeArray b_component(simulationData& data) {
        if constexpr (axis == AxisName::X) {
            return data.bx;
        } 
        if constexpr (axis == AxisName::Y) {
            return data.by;
        }
        return data.bz;
    }

    template <AxisName axis>
    DEVICEPREFIX volumeArray dab_component(simulationData& data) {
        if constexpr (axis == AxisName::X) {
            return data.dxab;
        } 
        if constexpr (axis == AxisName::Y) {
            return data.dyab;
        }
        return data.dzab;
    }

    template <AxisName axis>
    DEVICEPREFIX volumeArray dac_component(simulationData& data) {
        if constexpr (axis == AxisName::X) {
            return data.dxac;
        } 
        if constexpr (axis == AxisName::Y) {
            return data.dyac;
        }
        return data.dzac;
    }

    template <AxisName axis>
    fp_t& pass(remapData& data) {
        if constexpr (axis == AxisName::X) {
            return data.xpass;
        } 
        if constexpr (axis == AxisName::Y) {
            return data.ypass;
        }
        return data.zpass;
    }

    template <AxisName axis>
    DEVICEPREFIX lineArray d_b(simulationData& data) {
        if constexpr (axis == AxisName::X) {
            return data.dxb;
        } 
        if constexpr (axis == AxisName::Y) {
            return data.dyb;
        }
        return data.dzb;
    }

    template <AxisName axis>
    void dm_bcs(LARE3D_neutral& lare) {
        if constexpr (axis == AxisName::X) {
            return lare.dm_x_bcs();
        } else if constexpr (axis == AxisName::Y) {
            return lare.dm_y_bcs();
        }
        return lare.dm_z_bcs();
    }

    template <AxisName a_comp, AxisName b_comp>
    struct RemapIndexer {
        const volumeArray& arr;

        DEVICEPREFIX fp_t& operator()(idx_t ia, idx_t ib, idx_t iperp) const {
            if constexpr (a_comp == AxisName::X && b_comp == AxisName::Y) {
                return arr(ia, ib, iperp);
            }
            if constexpr (a_comp == AxisName::X && b_comp == AxisName::Z) {
                return arr(ia, iperp, ib);
            }
            if constexpr (a_comp == AxisName::Y && b_comp == AxisName::X) {
                return arr(ib, ia, iperp);
            }
            if constexpr (a_comp == AxisName::Y && b_comp == AxisName::Z) {
                return arr(iperp, ia, ib);
            }
            if constexpr (a_comp == AxisName::Z && b_comp == AxisName::X) {
                return arr(ib, iperp, ia);
            }
            if constexpr (a_comp == AxisName::Z && b_comp == AxisName::Y) {
                return arr(iperp, ib, ia);
            }
        }
    };

    template <AxisName v_comp, AxisName b_comp>
    void v_b_flux(simulationData& data, remapData& remap_data) {
        using Range = pw::Range;
        using RIndexer = RemapIndexer<v_comp, b_comp>;
        constexpr AxisName perp_comp = AxisName(3 - int(v_comp) - int(b_comp)); 
        auto v1_arr = v1_component<v_comp>(data);
        auto b_arr = b_component<b_comp>(data);

        pw::applyKernel(
            LAMBDA(idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(v_comp)];
                const idx_t ib = idxs[int(b_comp)];
                const idx_t ic = idxs[int(perp_comp)];
                const idx_t iap = ia + 1;
                const idx_t ibp = ib + 1;
                const idx_t icm = ic - 1;
                const idx_t iam = ia - 1;
                const idx_t iap2 = ia + 2;

                RIndexer v1 {
                    .arr = v1_arr
                };
                const fp_t v_advect = 0.5 * (v1(ia, ib, ic) + v1(ia, ib, icm));
                const fp_t o_v = v_advect * data.dt;

                RIndexer db1 {
                    .arr = remap_data.db1
                };
                const fp_t dba = 0.5 * (db1(ia, ib, ic) + db1(ia, ibp, ic));
                const fp_t dbap = 0.5 * (db1(iap, ib, ic) + db1(iap, ibp, ic));
                const fp_t dbap2 = 0.5 * (db1(iap2, ib, ic) + db1(iap2, ibp, ic));
                const fp_t dbam = 0.5 * (db1(iam, ib, ic) + db1(iam, ibp, ic));

                RIndexer b {
                    .arr = b_arr
                };
                const fp_t fm = b(iam, ib, ic) / dbam;
                const fp_t fi = b(ia, ib, ic) / dba;
                const fp_t fp = b(iap, ib, ic) / dbap;
                const fp_t fp2 = b(iap2, ib, ic) / dbap2;

                const fp_t dfm = fi - fm;
                const fp_t dfi = fp - fi;
                const fp_t dfp = fp2 - fp;

                const fp_t sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
                const fp_t vad_p = (sign_v + 1.0) * 0.5;
                const fp_t vad_m = 1.0 - vad_p;

                RIndexer cvc1 {
                    .arr = remap_data.cvc1
                };
                RIndexer cv1 {
                    .arr = data.cv1
                };
                const fp_t fu = fi * vad_p + fp * vad_m;
                const fp_t dfu = dfm * vad_p + dfp * vad_m;
                const fp_t daci = cvc1(ia, ib, ic);
                const fp_t dacu = cvc1(iam, ib, ic) * vad_p + cvc1(iap, ib, ic) * vad_m;
                const fp_t dabu = cv1(ia, ib, ic) * vad_p + cv1(iap, ib, ic) * vad_m;

                const fp_t dau = dba * vad_p + dbap * vad_m;
                const fp_t phi = std::abs(o_v) / dau;

                const fp_t Da = sixth * (
                    (2.0 - phi) * std::abs(dfi) / daci + (1.0 + phi) * std::abs(dfu) / dacu 
                );
                const fp_t ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));
                const fp_t Di = sign_v * ss * pw::min({std::abs(Da) * dabu, std::abs(dfi), std::abs(dfu)});

                RIndexer flux {
                    .arr = remap_data.flux
                };
                flux(ia, ib, ic) = (fu + Di * (1.0 - phi)) * o_v;
            },
            Range(0, data.nx),
            Range(0, data.ny),
            Range(0, data.nz)
        );
        pw::fence();
    }


    template <AxisName axis>
    void mass_flux(simulationData& data, remapData& remap_data) {
        using Range = pw::Range;
        constexpr AxisName axis_b = AxisName((int(axis) + 1) % 3);
        constexpr AxisName axis_c = AxisName((int(axis) + 2) % 3);
        using RIndexer = RemapIndexer<axis, axis_b>;
        auto daab_arr = dab_component<axis>(data);
        auto v1_arr = v1_component<axis>(data);
        auto nx = data.nx;
        auto ny = data.ny;
        auto nz = data.nz;
        if constexpr (axis == AxisName::X) {
            ny += 1;
            nz += 1;
        } else if constexpr (axis == AxisName::Y) {
            nx += 1;
            nz += 1;
        } else {
            nx += 1;
            ny += 1;
        }

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;
                const idx_t iap = ia + 1;
                const idx_t iap2 = ia + 2;
                const idx_t ibm = ib - 1;
                const idx_t icm = ic - 1;

                RIndexer daab {
                    .arr = daab_arr
                };
                const fp_t area = daab(ia, ib, ic);

                RIndexer v1 {
                    .arr = v1_arr
                };
                const fp_t v_advect = 0.25 * (
                    v1(ia, ib, ic) + 
                    v1(ia, ibm, ic) + 
                    v1(ia, ib, icm) + 
                    v1(ia, ibm, icm)
                );
                const fp_t o_v = v_advect * data.dt * area;

                RIndexer rho {
                    .arr = data.rho
                };
                const fp_t fm = rho(iam, ib, ic);
                const fp_t fi = rho(ia, ib, ic);
                const fp_t fp = rho(iap, ib, ic);
                const fp_t fp2 = rho(iap2, ib, ic);

                const fp_t dfm = fi - fm;
                const fp_t dfi = fp - fi;
                const fp_t dfp = fp2 - fp;

                const fp_t sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
                const fp_t vad_p = (sign_v + 1.0) * 0.5;
                const fp_t vad_m = 1.0 - vad_p;

                const fp_t fu = fi * vad_p + fp * vad_m;
                const fp_t dfu = dfm * vad_p + dfp * vad_m;
                RIndexer cvc1 {
                    .arr = remap_data.cvc1
                };
                RIndexer cv1 {
                    .arr = data.cv1
                };
                const fp_t daci = cvc1(ia, ib, ic);
                const fp_t dacu = cvc1(iam, ib, ic) * vad_p + cvc1(iap, ib, ic) * vad_m;
                const fp_t dabu = cv1(ia, ib, ic) * vad_p + cv1(iap, ib, ic) * vad_m;

                const fp_t phi = std::abs(o_v) / dabu;
                const fp_t Da = sixth * (
                    (2.0 - phi) * std::abs(dfi) / daci + (1.0 + phi) * std::abs(dfu) / dacu
                );

                const fp_t ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));
                const fp_t Di = sign_v * ss * pw::min({std::abs(Da) * dabu, std::abs(dfi), std::abs(dfu)});
                RIndexer dm {
                    .arr = data.dm
                };
                dm(ia, ib, ic) = (fu + Di * (1.0 - phi)) * o_v;
            },
            Range(0, nx),
            Range(0, ny),
            Range(0, nz)
        );
        pw::fence();
    }

    template <AxisName axis, auto mPtr>
    void mom_flux(simulationData& data, remapData& remap_data) {
        using Range = pw::Range;
        constexpr AxisName axis_b = AxisName((int(axis) + 1) % 3);
        constexpr AxisName axis_c = AxisName((int(axis) + 2) % 3);
        using RIndexer = RemapIndexer<axis, axis_b>;
        auto daac_arr = dac_component<axis>(data);
        auto v1_arr = v1_component<axis>(data);
        int xs = 0;
        int ys = 0;
        int zs = 0;
        if constexpr (axis == AxisName::X) {
            xs = -1;
        } else if constexpr (axis == AxisName::Y) {
            ys = -1;
        } else {
            zs = -1;
        }

        // NOTE(cmo): This is only present in the y flux, but if it's needed
        // there, it's needed everywhere.
        pw::assign(remap_data.flux, 0.0);
        pw::fence();

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;
                const idx_t iap = ia + 1;
                const idx_t iap2 = ia + 2;

                RIndexer daac {
                    .arr = daac_arr
                };
                const fp_t area = daac(ia, ib, ic);
                RIndexer v1 {
                    .arr = v1_arr
                };
                const fp_t v_advect = v1(ia, ib, ic);
                const fp_t o_v = v_advect * data.dt * area;

                RIndexer f {
                    .arr = (data.*mPtr)
                };
                const fp_t fm = f(iam, ib, ic);
                const fp_t fi = f(ia, ib, ic);
                const fp_t fp = f(iap, ib, ic);
                const fp_t fp2 = f(iap2, ib, ic);

                const fp_t dfm = fi - fm;
                const fp_t dfi = fp - fi;
                const fp_t dfp = fp2 - fp;

                const fp_t sign_v = (o_v >= 0.0) ? 1.0 : -1.0;
                const fp_t vad_p = (sign_v + 1.0) * 0.5;
                const fp_t vad_m = 1.0 - vad_p;

                const fp_t fu = fi * vad_p + fp * vad_m;
                const fp_t dfu = dfm * vad_p + dfp * vad_m;
                RIndexer cv1 {
                    .arr = data.cv1
                };
                RIndexer cvc1 {
                    .arr = remap_data.cvc1
                };
                const fp_t daci = cv1(iap, ib, ic);
                const fp_t dacu = cv1(ia, ib, ic) * vad_p + cv1(iap2, ib, ic) * vad_m;
                const fp_t dabu = cvc1(ia, ib, ic) * vad_p + cvc1(iap, ib, ic) * vad_m;

                const fp_t phi = std::abs(o_v) / dabu;
                const fp_t Da = sixth * (
                    (2.0 - phi) * std::abs(dfi) / daci + (1.0 + phi) * std::abs(dfu) / dacu
                );
                const fp_t ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));
                const fp_t Di = sign_v * ss * pw::min({std::abs(Da) * dabu, std::abs(dfi), std::abs(dfu)});

                RIndexer rho_v {
                    .arr = remap_data.rho_v
                };
                RIndexer dm {
                    .arr = data.dm
                };
                const fp_t rhou = rho_v(ia, ib, ic) * vad_p + rho_v(iap, ib, ic) * vad_m;
                const fp_t dmu = std::abs(dm(ia, ib, ic)) / dabu / rhou;

                RIndexer flux {
                    .arr = remap_data.flux
                };
                flux(ia, ib, ic) = fu + Di * (1.0 - dmu);
            },
            Range(xs, data.nx),
            Range(ys, data.ny),
            Range(zs, data.nz)
        );
        pw::fence();
        if (data.rke) {
            auto nx = data.nx;
            auto ny = data.ny;
            auto nz = data.nz;
            if constexpr (axis == AxisName::X) {
                nx -= 1;
            } else if constexpr (axis == AxisName::Y) {
                ny -= 1;
            } else {
                nz -= 1;
            }
            pw::applyKernel(
                LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                    const idx_t idxs[3] = {ix, iy, iz};
                    const idx_t ia = idxs[int(axis)];
                    const idx_t ib = idxs[int(axis_b)];
                    const idx_t ic = idxs[int(axis_c)];
                    const idx_t iam = ia - 1;
                    const idx_t iap = ia + 1;
                    const idx_t ibp = ib + 1;
                    const idx_t icp = ic + 1;

                    RIndexer rho_v1 {
                        .arr = remap_data.rho_v1
                    };
                    RIndexer cv2 {
                        .arr = remap_data.cv2
                    };
                    const fp_t m = rho_v1(ia, ib, ic) * cv2(ia, ib, ic);
                    const fp_t mp = rho_v1(iap, ib, ic) * cv2(iap, ib, ic);

                    RIndexer f {
                        .arr = (data.*mPtr)
                    };
                    RIndexer flux {
                        .arr = remap_data.flux
                    };
                    RIndexer dm {
                        .arr = data.dm
                    };

                    const fp_t ai = (
                        (f(ia, ib, ic) - flux(iam, ib, ic)) * dm(iam, ib, ic) / m 
                        - (f(ia, ib, ic) - flux(ia, ib, ic)) * dm(ia, ib, ic) / m
                    );
                    const fp_t aip = (
                        (f(iap, ib, ic) - flux(ia, ib, ic)) * dm(ia, ib, ic) / mp
                        - (f(iap, ib, ic) - flux(iap, ib, ic)) * dm(iap, ib, ic) / mp
                    );
                    const fp_t dk = 0.5 * dm(ia, ib, ic) * (
                        (f(iap, ib, ic) - f(ia, ib, ic)) 
                        * (flux(ia, ib, ic) - 0.5 * (f(iap, ib, ic) + f(ia, ib, ic)))
                        - 0.5 * ai * (f(ia, ib, ic) - flux(ia, ib, ic))
                        + 0.5 * aip * (f(iap, ib, ic) - flux(ia, ib, ic))
                    );

                    RIndexer delta_ke {
                        .arr = data.delta_ke
                    };
                    pw::atomic::accelerated::Add(delta_ke(iap, ib, ic), dk);
                    pw::atomic::accelerated::Add(delta_ke(iap, ibp, ic), dk);
                    pw::atomic::accelerated::Add(delta_ke(iap, ibp, icp), dk);
                    pw::atomic::accelerated::Add(delta_ke(iap, ib, icp), dk);
                },
                Range(0, nx),
                Range(0, ny),
                Range(0, nz)
            );
            pw::fence();
        }
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                remap_data.flux(ix, iy, iz) *= data.dm(ix, iy, iz);
            },
            Range(xs, data.nx),
            Range(ys, data.ny),
            Range(zs, data.nz)
        );
        pw::fence();
    }

    template <AxisName axis, auto mPtr>
    void energy_flux(simulationData& data, remapData& remap_data) {
        using Range = pw::Range;
        constexpr AxisName axis_b = AxisName((int(axis) + 1) % 3);
        constexpr AxisName axis_c = AxisName((int(axis) + 2) % 3);
        using RIndexer = RemapIndexer<axis, axis_b>;
        auto daab_arr = dab_component<axis>(data);
        auto v1_arr = v1_component<axis>(data);

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;
                const idx_t iap = ia + 1;
                const idx_t iap2 = ia + 2;
                const idx_t ibm = ib - 1;
                const idx_t icm = ic - 1;

                RIndexer daab {
                    .arr = daab_arr
                };
                const fp_t area = daab(ia, ib, ic);
                RIndexer v1 {
                    .arr = v1_arr
                };
                const fp_t v_advect = 0.25 * (
                    v1(ia, ib, ic) + 
                    v1(ia, ibm, ic) + 
                    v1(ia, ib, icm) + 
                    v1(ia, ibm, icm) 
                );
                const fp_t o_v = v_advect * data.dt * area;

                RIndexer f {
                    .arr = (data.*mPtr)
                };
                const fp_t fm = f(iam, ib, ic);
                const fp_t fi = f(ia, ib, ic);
                const fp_t fp = f(iap, ib, ic);
                const fp_t fp2 = f(iap2, ib, ic);

                const fp_t dfm = fi - fm;
                const fp_t dfi = fp - fi;
                const fp_t dfp = fp2 - fp;

                const fp_t sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
                const fp_t vad_p = (sign_v + 1.0) * 0.5;
                const fp_t vad_m = 1.0 - vad_p;

                const fp_t fu = fi * vad_p + fp * vad_m;
                const fp_t dfu = dfm * vad_p + dfp * vad_m;
                RIndexer cvc1 {
                    .arr = remap_data.cvc1
                };
                RIndexer cv1 {
                    .arr = data.cv1
                };
                const fp_t daci = cvc1(ia, ib, ic);
                const fp_t dacu = cvc1(iam, ib, ic) * vad_p + cvc1(iap, ib, ic) * vad_m;
                const fp_t dabu = cv1(ia, ib, ic) * vad_p + cv1(iap, ib, ic) * vad_m;

                const fp_t phi = std::abs(o_v) / dabu;
                const fp_t Da = sixth * (
                    (2.0 - phi) * std::abs(dfi) / daci + (1.0 + phi) * std::abs(dfu) / dacu
                );

                const fp_t ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));
                const fp_t Di = sign_v * ss * pw::min({std::abs(Da) * dabu, std::abs(dfi), std::abs(dfu)});

                RIndexer rho1 {
                    .arr = remap_data.rho1
                };
                RIndexer dm {
                    .arr = data.dm
                };
                const fp_t rhou = rho1(ia, ib, ic) * vad_p + rho1(iap, ib, ic) * vad_m;
                const fp_t dmu = std::abs(dm(ia, ib, ic)) / dabu / rhou;

                RIndexer flux {
                    .arr = remap_data.flux
                };
                flux(ia, ib, ic) = (fu + Di * (1.0 - dmu)) * dm(ia, ib, ic);
            },
            Range(0, data.nx),
            Range(0, data.ny),
            Range(0, data.nz)
        );
        pw::fence();
    }

    template <AxisName axis>
    void remap(LARE3D_neutral& lare, simulationData& data, remapData& remap_data) {
        using Range = pw::Range;
        constexpr AxisName axis_b = AxisName((int(axis) + 1) % 3);
        constexpr AxisName axis_c = AxisName((int(axis) + 2) % 3);
        using RIndexer = RemapIndexer<axis, axis_b>;
        auto daab_arr = dab_component<axis>(data);
        auto dbab_arr = dab_component<axis_b>(data);
        auto dcab_arr = dab_component<axis_c>(data);
        auto va1_arr = v1_component<axis>(data);
        auto vb1_arr = v1_component<axis_b>(data);
        auto vc1_arr = v1_component<axis_c>(data);
        auto dab = d_b<axis>(data);

        pw::assign(data.dm, 0.0);
        pw::assign(remap_data.rho1, data.rho);
        pw::fence();
        const fp_t bpass = pass<axis_b>(remap_data);
        const fp_t cpass = pass<axis_c>(remap_data);

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];

                const idx_t iam = ia - 1;
                const idx_t ibm = ib - 1;
                const idx_t icm = ic - 1;

                RIndexer va1 {
                    .arr = va1_arr
                };
                RIndexer vb1 {
                    .arr = vb1_arr
                };
                RIndexer vc1 {
                    .arr = vc1_arr
                };

                // va at Ba(i, j, k)
                const fp_t vab = 0.25 * (
                    va1(ia, ib, ic)
                    + va1(ia, ibm, ic)
                    + va1(ia, ib, icm)
                    + va1(ia, ibm, icm)
                );
                // va at Ba(i-1, j, k)
                const fp_t vabm = 0.25 * (
                    va1(iam, ib, ic)
                    + va1(iam, ibm, ic)
                    + va1(iam, ib, icm)
                    + va1(iam, ibm, icm)
                );

                // vb at Bb(i, j, k)
                const fp_t vbb = 0.25 * (
                    vb1(ia, ib, ic)
                    + vb1(iam, ib, ic)
                    + vb1(ia, ib, icm)
                    + vb1(iam, ib, icm)
                );
                // vb at Bb(i, j-1, k)
                const fp_t vbbm = 0.25 * (
                    vb1(ia, ibm, ic)
                    + vb1(iam, ibm, ic)
                    + vb1(ia, ibm, icm)
                    + vb1(iam, ibm, icm)
                );
                // vc at Bc(i, j, k)
                const fp_t vcb = 0.25 * (
                    vc1(ia, ib, ic)
                    + vc1(iam, ib, ic)
                    + vc1(ia, ibm, ic)
                    + vc1(iam, ibm, ic)
                );
                // vc at Bc(i, j, k-1)
                const fp_t vcbm = 0.25 * (
                    vc1(ia, ib, icm)
                    + vc1(iam, ib, icm)
                    + vc1(ia, ibm, icm)
                    + vc1(iam, ibm, icm)
                );

                RIndexer cv {
                    .arr = data.cv
                };
                RIndexer daab {
                    .arr = daab_arr
                };
                RIndexer dbab {
                    .arr = dbab_arr
                };
                RIndexer dcab {
                    .arr = dcab_arr
                };
                const fp_t vol = cv(ia, ib, ic);
                const fp_t dvada = (vab * daab(ia, ib, ic) - vabm * daab(iam, ib, ic)) / vol;
                const fp_t dvbdb = bpass * (vbb * dbab(ia, ib, ic) - vbbm * dbab(ia, ibm, ic)) / vol;
                const fp_t dvcdc = cpass * (vcb * dcab(ia, ib, ic) - vcbm * dcab(ia, ib, icm)) / vol;

                fp_t dv = (dvbdb + dvcdc) * data.dt;

                // Control volume after remap
                RIndexer cv2 {
                    .arr = remap_data.cv2
                };
                cv2(ia, ib, ic) = vol * (1.0 + dv);

                dv += dvada * data.dt;

                // Control volume before remap
                RIndexer cv1 {
                    .arr = data.cv1
                };
                cv1(ia, ib, ic) = vol * (1.0 + dv);

                // dab before remap
                RIndexer db1 {
                    .arr = remap_data.db1
                };
                fp_t dab_elem = dab(ia);
                if constexpr (axis == AxisName::Y) {
                    dab_elem *= data.hyc(ic);
                } else if constexpr (axis == AxisName::Z) {
                    dab_elem *= data.hzc(ib, ic);
                }
                db1(ia, ib, ic) = dab_elem + (vab - vabm) * data.dt;
            },
            Range(-1, data.nx+2),
            Range(-1, data.ny+2),
            Range(-1, data.nz+2)
        );
        pw::fence();

        // cvc1 = vertex CV before remap
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t ixp = ix + 1;
                const idx_t iyp = iy + 1;
                const idx_t izp = iz + 1;

                remap_data.cvc1(ix, iy, iz) = 0.125 * (
                    data.cv1(ix, iy, iz)
                    + data.cv1(ixp, iy, iz)
                    + data.cv1(ix, iyp, iz)
                    + data.cv1(ixp, iyp, iz)
                    + data.cv1(ix, iy, izp)
                    + data.cv1(ixp, iy, izp)
                    + data.cv1(ix, iyp, izp)
                    + data.cv1(ixp, iyp, izp)
                );
            },
            Range(-1, data.nx + 1),
            Range(-1, data.ny + 1),
            Range(-1, data.nz + 1)
        );
        pw::fence();

        v_b_flux<axis, axis_b>(data, remap_data);
        auto ba_arr = b_component<axis>(data);
        auto bb_arr = b_component<axis_b>(data);
        auto bc_arr = b_component<axis_c>(data);
        int xs = 1;
        int ys = 1;
        int zs = 1;
        if constexpr (axis == AxisName::X) {
            ys = 0;
        } else if constexpr (axis == AxisName::Y) {
            zs = 0;
        } else {
            xs = 0;
        }
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer bb {
                    .arr = bb_arr
                };
                RIndexer flux {
                    .arr = remap_data.flux
                };
                bb(ia, ib, ic) += -flux(ia, ib, ic) + flux(iam, ib, ic);
            },
            Range(xs, data.nx),
            Range(ys, data.ny),
            Range(zs, data.nz)
        );

        xs = 1;
        ys = 1;
        zs = 1;
        if constexpr (axis == AxisName::X) {
            xs = 0;
        } else if constexpr (axis == AxisName::Y) {
            ys = 0;
        } else {
            zs = 0;
        }

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t ibm = ib - 1;

                RIndexer ba {
                    .arr = ba_arr
                };
                RIndexer flux {
                    .arr = remap_data.flux
                };
                ba(ia, ib, ic) += flux(ia, ib, ic) - flux(ia, ibm, ic);
            },
            Range(xs, data.nx),
            Range(ys, data.ny),
            Range(zs, data.nz)
        );
        pw::fence();

        v_b_flux<axis, axis_c>(data, remap_data);
        xs = 1;
        ys = 1;
        zs = 1;
        if constexpr (axis == AxisName::X) {
            zs = 0;
        } else if constexpr (axis == AxisName::Y) {
            xs = 0;
        } else {
            ys = 0;
        }
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer bc {
                    .arr = bc_arr
                };
                RIndexer flux {
                    .arr = remap_data.flux
                };
                bc(ia, ib, ic) += -flux(ia, ib, ic) + flux(iam, ib, ic);
            },
            Range(xs, data.nx), Range(ys, data.ny), Range(zs, data.nz)
        );
        xs = 1;
        ys = 1;
        zs = 1;
        if constexpr (axis == AxisName::X) {
            xs = 0;
        } else if constexpr (axis == AxisName::Y) {
            ys = 0;
        } else {
            zs = 0;
        }
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t icm = ic - 1;

                RIndexer ba {
                    .arr = ba_arr
                };
                RIndexer flux {
                    .arr = remap_data.flux
                };
                ba(ia, ib, ic) += flux(ia, ib, ic) - flux(ia, ib, icm);
            },
            Range(xs, data.nx), Range(ys, data.ny), Range(zs, data.nz)
        );
        pw::fence();

        mass_flux<axis>(data, remap_data);
        dm_bcs<axis>(lare);

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer rho { data.rho };
                RIndexer rho1 { remap_data.rho1 };
                RIndexer cv1 { data.cv1 };
                RIndexer dm { data.dm };
                RIndexer cv2 { remap_data.cv2 };
                rho(ia, ib, ic) = (rho1(ia, ib, ic) * cv1(ia, ib, ic) + dm(iam, ib, ic) - dm(ia, ib, ic)) / cv2(ia, ib, ic);
            },
            Range(1, data.nx), Range(1, data.ny), Range(1, data.nz)
        );

        /*energy_flux<axis, &simulationData::energy_electron>(data, remap_data);

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer energy_electron { data.energy_electron };
                RIndexer rho { data.rho };
                RIndexer rho1 { remap_data.rho1 };
                RIndexer cv1 { data.cv1 };
                RIndexer flux { remap_data.flux };
                RIndexer cv2 { remap_data.cv2 };
                energy_electron(ia, ib, ic) = (
                    energy_electron(ia, ib, ic) * cv1(ia, ib, ic) * rho1(ia, ib, ic)
                    + flux(iam, ib, ic) - flux(ia, ib, ic)
                ) / (
                    cv2(ia, ib, ic) * rho(ia, ib, ic)
                );
            },
            Range(1, data.nx), Range(1, data.ny), Range(1, data.nz)
        );
        pw::fence();
        */

        energy_flux<axis, &simulationData::energy_neutral>(data, remap_data);

        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer energy_neutral { data.energy_neutral };
                RIndexer rho { data.rho };
                RIndexer rho1 { remap_data.rho1 };
                RIndexer cv1 { data.cv1 };
                RIndexer flux { remap_data.flux };
                RIndexer cv2 { remap_data.cv2 };
                energy_neutral(ia, ib, ic) = (
                    energy_neutral(ia, ib, ic) * cv1(ia, ib, ic) * rho1(ia, ib, ic)
                    + flux(iam, ib, ic) - flux(ia, ib, ic)
                ) / (
                    cv2(ia, ib, ic) * rho(ia, ib, ic)
                );
            },
            Range(1, data.nx), Range(1, data.ny), Range(1, data.nz)
        );
        pw::fence();

        // Redefine dab1, cv1, cv2, dm and vx1 for velocity (vertex) cells.
        // In some of these calculations the flux variable is used as a temporary array

        // Calculate vertex density
        xs = 0;
        ys = 0;
        zs = 0;
        int xe = data.nx;
        int ye = data.ny;
        int ze = data.nz;
        if constexpr (axis == AxisName::X) {
            xs = -1;
            xe += 1;
        } else if constexpr (axis == AxisName::Y) {
            ys = -1;
            ye += 1;
        } else {
            zs = -1;
            ze += 1;
        }
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iap = ia + 1;
                const idx_t ibp = ib + 1;
                const idx_t icp = ic + 1;

                RIndexer rho_v { remap_data.rho_v };
                RIndexer rho1 { remap_data.rho1 };
                RIndexer cv1 { data.cv1 };
                RIndexer cvc1 { remap_data.cvc1 };

                rho_v(ia, ib, ic) = (0.125 / cvc1(ia, ib, ic)) * (
                    rho1(ia, ib, ic) * cv1(ia, ib, ic) + 
                    rho1(iap, ib, ic) * cv1(iap, ib, ic) + 
                    rho1(ia, ibp, ic) * cv1(ia, ibp, ic) + 
                    rho1(iap, ibp, ic) * cv1(iap, ibp, ic) + 
                    rho1(ia, ib, icp) * cv1(ia, ib, icp) + 
                    rho1(iap, ib, icp) * cv1(iap, ib, icp) + 
                    rho1(ia, ibp, icp) * cv1(ia, ibp, icp) + 
                    rho1(iap, ibp, icp) * cv1(iap, ibp, icp)
                );
            },
            Range(xs, xe), Range(ys, ye), Range(zs, ze)
        );

        // Use flux as a temporary array to store the new cv2
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;

                remap_data.flux(ix, iy, iz) = 0.125 * (
                    remap_data.cv2(ix, iy, iz) + 
                    remap_data.cv2(ixp, iy, iz) + 
                    remap_data.cv2(ix, iyp, iz) + 
                    remap_data.cv2(ixp, iyp, iz) + 
                    remap_data.cv2(ix, iy, izp) + 
                    remap_data.cv2(ixp, iy, izp) + 
                    remap_data.cv2(ix, iyp, izp) + 
                    remap_data.cv2(ixp, iyp, izp)
                );
            },
            Range(0, data.nx), Range(0, data.ny), Range(0, data.nz)
        );
        pw::fence();

        // Now copy it back to cv2
        pw::assign(
            remap_data.cv2(Range(0, data.nx), Range(0, data.ny), Range(0, data.nz)),
            remap_data.flux(Range(0, data.nx), Range(0, data.ny), Range(0, data.nz))
        );
        pw::fence();

        // Now shift va
        xs = 0;
        ys = 0;
        zs = 0;
        xe = data.nx;
        ye = data.ny;
        ze = data.nz;
        if constexpr (axis == AxisName::X) {
            xs = -2;
            xe += 1;
        } else if constexpr (axis == AxisName::Y) {
            ys = -2;
            ye += 1;
        } else {
            zs = -2;
            ze += 1;
        }
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iap = ia + 1;

                RIndexer flux { remap_data.flux };
                RIndexer va1 { va1_arr };

                flux(ia, ib, ic) = (va1(ia, ib, ic) + va1(iap, ib, ic)) * 0.5;
            },
            Range(xs, xe), Range(ys, ye), Range(zs, ze)
        );
        pw::fence();

        pw::assign(
            va1_arr(Range(xs, xe), Range(ys, ye), Range(zs, ze)),
            remap_data.flux(Range(xs, xe), Range(ys, ye), Range(zs, ze))
        );
        pw::fence();

        // Now shift mass flux to temporary
        xs = 0;
        ys = 0;
        zs = 0;
        if constexpr (axis == AxisName::X) {
            xs = -1;
        } else if constexpr (axis == AxisName::Y) {
            ys = -1;
        } else {
            zs = -1;
        }
        pw::applyKernel(
            LAMBDA (idx_t ix, idx_t iy, idx_t iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iap = ia + 1;
                const idx_t ibp = ib + 1;
                const idx_t icp = ic + 1;

                RIndexer flux { remap_data.flux };
                RIndexer dm { data.dm };

                flux(ia, ib, ic) = 0.125 * (
                    dm(ia, ib, ic) + 
                    dm(iap, ib, ic) + 
                    dm(ia, ibp, ic) +
                    dm(iap, ibp, ic) + 
                    dm(ia, ib, icp) + 
                    dm(iap, ib, icp) + 
                    dm(ia, ibp, icp) +
                    dm(iap, ibp, icp)
                );
            },
            Range(xs, data.nx), Range(ys, data.ny), Range(zs, data.nz)
        );
        pw::fence();

        // And copy back to dm
        pw::assign(
            data.dm(Range(xs, data.nx), Range(ys, data.ny), Range(zs, data.nz)),
            remap_data.flux(Range(xs, data.nx), Range(ys, data.ny), Range(zs, data.nz))
        );
        pw::fence();

        // Calculate vertex density after remap
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer dm { data.dm };

                // Vertex density after remap
                remap_data.rho_v1(ix, iy, iz) = (
                    remap_data.rho_v(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + 
                    dm(iam, ib, ic) - dm(ia, ib, ic)
                ) / remap_data.cv2(ix, iy, iz);
            },
            Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();

        mom_flux<axis, &simulationData::vx>(data, remap_data);
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer flux { remap_data.flux };

                data.vx(ix, iy, iz) = (
                    remap_data.rho_v(ix, iy, iz) * data.vx(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + 
                    flux(iam, ib, ic) - flux(ia, ib, ic)
                ) / (
                    remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz)
                );
            },
            Range(0, data.nx), Range(0, data.ny), Range(0, data.nz)
        );
        pw::fence();

        mom_flux<axis, &simulationData::vy>(data, remap_data);
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer flux { remap_data.flux };

                data.vy(ix, iy, iz) = (
                    remap_data.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + 
                    flux(iam, ib, ic) - flux(ia, ib, ic)
                ) / (
                    remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz)
                );
            },
            Range(0, data.nx), Range(0, data.ny), Range(0, data.nz)
        );
        pw::fence();

        mom_flux<axis, &simulationData::vz>(data, remap_data);
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                const idx_t idxs[3] = {ix, iy, iz};
                const idx_t ia = idxs[int(axis)];
                const idx_t ib = idxs[int(axis_b)];
                const idx_t ic = idxs[int(axis_c)];
                const idx_t iam = ia - 1;

                RIndexer flux { remap_data.flux };

                data.vz(ix, iy, iz) = (
                    remap_data.rho_v(ix, iy, iz) * data.vz(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + 
                    flux(iam, ib, ic) - flux(ia, ib, ic)
                ) / (
                    remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz)
                );
            },
            Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();

        pass<axis>(remap_data) = 0;
        lare.boundary_conditions();
    }

    // NOTE(cmo): Instantiate templates in this TU
    template void v_b_flux<AxisName::X, AxisName::Y>(simulationData&, remapData&);
    template void v_b_flux<AxisName::X, AxisName::Z>(simulationData&, remapData&);
    template void v_b_flux<AxisName::Y, AxisName::X>(simulationData&, remapData&);
    template void v_b_flux<AxisName::Y, AxisName::Z>(simulationData&, remapData&);
    template void v_b_flux<AxisName::Z, AxisName::X>(simulationData&, remapData&);
    template void v_b_flux<AxisName::Z, AxisName::Y>(simulationData&, remapData&);

    template void mass_flux<AxisName::X>(simulationData&, remapData&);
    template void mass_flux<AxisName::Y>(simulationData&, remapData&);
    template void mass_flux<AxisName::Z>(simulationData&, remapData&);

    // NOTE(cmo): mom_flux is the one the introduces a small amount of error
    template void mom_flux<AxisName::X, &simulationData::vx>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::X, &simulationData::vy>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::X, &simulationData::vz>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::Y, &simulationData::vx>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::Y, &simulationData::vy>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::Y, &simulationData::vz>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::Z, &simulationData::vx>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::Z, &simulationData::vy>(simulationData& data, remapData& remap_data);
    template void mom_flux<AxisName::Z, &simulationData::vz>(simulationData& data, remapData& remap_data);

    //template void energy_flux<AxisName::X, &simulationData::energy_electron>(simulationData& data, remapData& remap_data);
    template void energy_flux<AxisName::X, &simulationData::energy_neutral>(simulationData& data, remapData& remap_data);
    //template void energy_flux<AxisName::Y, &simulationData::energy_electron>(simulationData& data, remapData& remap_data);
    template void energy_flux<AxisName::Y, &simulationData::energy_neutral>(simulationData& data, remapData& remap_data);
    //template void energy_flux<AxisName::Z, &simulationData::energy_electron>(simulationData& data, remapData& remap_data);
    template void energy_flux<AxisName::Z, &simulationData::energy_neutral>(simulationData& data, remapData& remap_data);

    template void remap<AxisName::X>(LARE3D_neutral& lare, simulationData& data, remapData& remap_data);
    template void remap<AxisName::Y>(LARE3D_neutral& lare, simulationData& data, remapData& remap_data);
    template void remap<AxisName::Z>(LARE3D_neutral& lare, simulationData& data, remapData& remap_data);
}
