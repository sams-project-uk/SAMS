// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_ALGORITHM_H
#define EIS_ALGORITHM_H

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "eis_constants.h"

namespace eis {

  //! Find the cell of a monotonically increasing axis that contains a point
  //! "cell" is the zero based index of the lower end of the cell and "frac"
  //! is the fractional position of the point within the cell. Points outside
  //! the axis are clamped to the first or last cell
  //! The vector parameters are wrapped in type_identity_t so that Num is
  //! deduced from "point"
  template <typename Num>
  void bisect_axis(Num point,
      const std::type_identity_t<std::vector<Num>>& axis, i8& cell,
      Num& frac) {
    const count_type n = axis.size();

    if (point <= axis[0]) {
      cell = 0;
      frac = Num(0);
      return;
    } else if (point >= axis[n - 1]) {
      cell = static_cast<i8>(n) - 2;
      frac = Num(1);
      return;
    }

    count_type lower = 0;
    count_type upper = n - 1;
    count_type mid = (lower + upper) / 2;
    while (true) {
      if (point >= axis[mid] && point <= axis[mid + 1]) {
        frac = (point - axis[mid]) / (axis[mid + 1] - axis[mid]);
        cell = static_cast<i8>(mid);
        break;
      } else if (point < axis[mid]) {
        upper = mid;
      } else {
        lower = mid;
      }
      mid = (lower + upper) / 2;
    }
  }

  //! Linear interpolation of y(x) at a point
  template <typename Num>
  Num interpolate1d(Num point, const std::type_identity_t<std::vector<Num>>& x,
      const std::type_identity_t<std::vector<Num>>& y, error& errcode) {
    i8 cell;
    Num frac;

    errcode = err_none;
    bisect_axis<Num>(point, x, cell, frac);

    const auto c = static_cast<count_type>(cell);
    return y[c] + frac * (y[c + 1] - y[c]);
  }

  //! Gradient of the linear reconstruction of y(x) in the cell containing a
  //! point
  template <typename Num>
  Num deriv_interpolate1d(Num point,
      const std::type_identity_t<std::vector<Num>>& x,
      const std::type_identity_t<std::vector<Num>>& y, error& errcode) {
    i8 cell;
    Num frac;

    errcode = err_none;
    bisect_axis<Num>(point, x, cell, frac);

    const auto c = static_cast<count_type>(cell);
    return (y[c + 1] - y[c]) / (x[c + 1] - x[c]);
  }

  //! Bilinear interpolation of z(x, y) at a point
  //! z is a flat array of x.size() * y.size() values with the x index varying
  //! fastest, so z(ix, iy) is z[ix + iy * x.size()]
  template <typename Num>
  Num interpolate2d(const std::array<Num, 2>& point,
      const std::type_identity_t<std::vector<Num>>& x,
      const std::type_identity_t<std::vector<Num>>& y,
      const std::type_identity_t<std::vector<Num>>& z, error& errcode) {
    i8 cell_x, cell_y;
    Num frac_x, frac_y;

    errcode = err_none;

    bisect_axis<Num>(point[0], x, cell_x, frac_x);
    bisect_axis<Num>(point[1], y, cell_y, frac_y);

    const auto nx = x.size();
    const auto cx = static_cast<count_type>(cell_x);
    const auto cy = static_cast<count_type>(cell_y);
    auto at = [&](count_type ix, count_type iy) { return z[ix + iy * nx]; };

    const Num val1 = at(cx, cy) + frac_x * (at(cx + 1, cy) - at(cx, cy));
    const Num val2 = at(cx, cy + 1)
        + frac_x * (at(cx + 1, cy + 1) - at(cx, cy + 1));

    return val1 + frac_y * (val2 - val1);
  }

}

#endif
