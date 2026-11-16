// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_CORE_FUNCTIONS_H
#define EIS_CORE_FUNCTIONS_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <math.h>
#endif

#include "../common/eis_algorithm.h"
#include "../common/eis_constants.h"
#include "eis_parser_constants.h"

namespace eis {

  //! A dummy function that signals an error if ever called
  template <typename Num>
  Num dummy(count_type, const Num*, void*, parser<Num>*,
      status&, error& errcode) {
    errcode = err_bad_value;
    return Num(0);
  }

  //! A function that signals a maths domain error whenever it is evaluated.
  //! Used by the simplifier in place of an expression that fails
  template <typename Num>
  Num maths_domain_error(count_type, const Num*, void*, parser<Num>*,
      status&, error& errcode) {
    errcode = errcode | err_maths_domain;
    return Num(1);
  }

  //! Unary plus
  template <typename Num>
  Num uplus(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return params[0];
  }

  //! Unary minus
  template <typename Num>
  Num uminus(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return -params[0];
  }

  //! Binary plus
  template <typename Num>
  Num bplus(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return params[0] + params[1];
  }

  //! Binary minus
  template <typename Num>
  Num bminus(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return params[0] - params[1];
  }

  //! Multiply
  template <typename Num>
  Num times(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return params[0] * params[1];
  }

  //! Divide
  template <typename Num>
  Num divide(count_type, const Num* params, void*, parser<Num>*,
      status&,
      [[maybe_unused]] error& errcode) {
#ifdef SAFEMATHS
    //Flag a zero denominator without a branch
    errcode = errcode
        | (-static_cast<error>(params[1] == Num(0)) & err_maths_domain);
#endif
    return params[0] / (std::copysign(Num(1), params[1])
        * std::max(std::abs(params[1]), tiny<Num>));
  }

  //! Raise to power
  template <typename Num>
  Num pow(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::pow(params[0], params[1]);
  }

  //! Exponentiation
  template <typename Num>
  Num expo(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return params[0] * std::pow(Num(10), params[1]);
  }

  //! Less than
  template <typename Num>
  Num lt(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (params[0] < params[1]) res = Num(1);
    return res;
  }

  //! Less than or equal to
  template <typename Num>
  Num le(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (params[0] <= params[1]) res = Num(1);
    return res;
  }

  //! Greater than
  template <typename Num>
  Num gt(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (params[0] > params[1]) res = Num(1);
    return res;
  }

  //! Greater than or equal to
  template <typename Num>
  Num ge(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (params[0] >= params[1]) res = Num(1);
    return res;
  }

  //! Equal to
  template <typename Num>
  Num eq(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (std::abs(params[0] - params[1]) < tiny<Num>) res = Num(1);
    return res;
  }

  //! Not equal to
  template <typename Num>
  Num neq(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (std::abs(params[0] - params[1]) > tiny<Num>) res = Num(1);
    return res;
  }

  //! And
  template <typename Num>
  Num parser_and(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (std::abs(params[0]) > tiny<Num> && std::abs(params[1]) > tiny<Num>) {
      res = Num(1);
    }
    return res;
  }

  //! Or
  template <typename Num>
  Num parser_or(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res = Num(0);
    if (std::abs(params[0]) > tiny<Num> || std::abs(params[1]) > tiny<Num>) {
      res = Num(1);
    }
    return res;
  }

  //! Absolute value
  template <typename Num>
  Num abs(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::abs(params[0]);
  }

  //! Integer by flooring
  template <typename Num>
  Num floor(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::floor(params[0]);
  }

  //! Integer by ceiling
  template <typename Num>
  Num ceil(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::ceil(params[0]);
  }

  //! Integer by rounding to nearest integer
  template <typename Num>
  Num nint(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::round(params[0]);
  }

  //! Integer by truncation
  template <typename Num>
  Num aint(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::trunc(params[0]);
  }

  //! Square root
  template <typename Num>
  Num sqrt(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (params[0] > Num(0)) {
      res = std::sqrt(params[0]);
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Sine
  template <typename Num>
  Num sin(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::sin(params[0]);
  }

  //! Cosine
  template <typename Num>
  Num cos(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::cos(params[0]);
  }

  //! Tangent
  template <typename Num>
  Num tan(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::tan(params[0]);
  }

  //! Arcsine
  template <typename Num>
  Num asin(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (std::abs(params[0]) <= Num(1)) {
      res = std::asin(params[0]);
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Arccosine
  template <typename Num>
  Num acos(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (std::abs(params[0]) <= Num(1)) {
      res = std::acos(params[0]);
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Arctangent
  template <typename Num>
  Num atan(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::atan(params[0]);
  }

  //! Two parameter arctangent
  template <typename Num>
  Num atan2(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::atan2(params[0], params[1]);
  }

  //! Hyperbolic sine
  template <typename Num>
  Num sinh(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::sinh(params[0]);
  }

  //! Hyperbolic cosine
  template <typename Num>
  Num cosh(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::cosh(params[0]);
  }

  //! Hyperbolic tangent
  template <typename Num>
  Num tanh(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::tanh(params[0]);
  }

  //! E to the power of
  template <typename Num>
  Num exp(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::exp(params[0]);
  }

  //! Natural logarithm
  template <typename Num>
  Num loge(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (params[0] > Num(0)) {
      res = std::log(params[0]);
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Logarithm base 10
  template <typename Num>
  Num log10(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (params[0] > Num(0)) {
      res = std::log10(params[0]);
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Logarithm to arbitrary base
  template <typename Num>
  Num log_base(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (params[0] > Num(0) && params[1] > Num(0)) {
      res = std::log(params[0]) / std::log(params[1]);
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Gaussian function
  template <typename Num>
  Num gauss(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (std::abs(params[2]) > Num(0)) {
      const Num arg = (params[0] - params[1]) / params[2];
      res = std::exp(-(arg * arg));
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Semi-Gaussian function. Gaussian that rises to a fixed plateau
  template <typename Num>
  Num semigauss(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (params[2] > Num(0) && std::abs(params[1]) > Num(0)
        && params[1] > Num(0)) {
      const Num t0 = params[3] * std::sqrt(-std::log(params[2] / params[1]));
      if (params[0] <= t0) {
        const Num arg = (params[0] - t0) / params[3];
        res = params[1] * std::exp(-(arg * arg));
      } else {
        res = params[1];
      }
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Derivative of semiGaussian function
  //! Params 1 to 4 are parameters, params 5 to 8 are derivatives of 1 to 4
  template <typename Num>
  Num dsemigauss(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (params[2] > Num(0) && std::abs(params[1]) > Num(0)
        && params[1] > Num(0)) {
      const Num t0 = std::sqrt(-std::log(params[2] / params[1]));
      const Num arg = (params[0] - params[3] * t0) / params[3];
      res = (Num(-2) * t0 * params[0] * params[1] * params[2] * params[3]
          * params[4]
          + t0 * params[1] * std::pow(params[3], Num(3)) * params[6]
          + params[0] * params[3] * params[3]
          * (params[2] * params[5] - params[1] * params[6]) + Num(2) * t0
          * params[0] * params[0] * params[1] * params[2] * params[7]
          + Num(2) * std::log(params[2] / params[1]) * params[1] * params[2]
          * params[3] * (params[0] * params[7] - params[3] * params[4]))
          / (params[2] * std::pow(params[3], Num(3)) * t0
          * std::exp(arg * arg));
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Super-Gaussian function. Gaussian with a power greater than 2
  template <typename Num>
  Num supergauss(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    Num res;
    if (std::abs(params[2]) > Num(0)) {
      res = std::exp(-std::pow((params[0] - params[1]) / params[2],
          static_cast<int>(std::lround(params[3]))));
    } else {
      res = Num(1);
      errcode = errcode | err_maths_domain;
    }
    return res;
  }

  //! Inverse hyperbolic sine
  template <typename Num>
  Num asinh(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::asinh(params[0]);
  }

  //! Inverse hyperbolic cosine
  template <typename Num>
  Num acosh(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::acosh(params[0]);
  }

  //! Inverse hyperbolic tangent
  template <typename Num>
  Num atanh(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::atanh(params[0]);
  }

  //! If condition is true return first value, if not return second
  template <typename Num>
  Num parser_if(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    Num res;
    if (std::abs(params[0]) > tiny<Num>) {
      res = params[1];
    } else {
      res = params[2];
    }
    return res;
  }

  //! Interpolate through a set of control points
  template <typename Num>
  Num interpol(count_type nparams, const Num* params, void*, parser<Num>*,
      status&,
      error& errcode) {
    count_type n_items;

    if (nparams % 2 == 0) {
      //EPOCH style interpolate
      n_items = (nparams - 2) / 2;
      if (static_cast<offset_type>(n_items) != std::lround(params[nparams - 1])) {
        errcode = err_wrong_parameters;
        return Num(0);
      }
    } else {
      //EIS style interpolate
      n_items = (nparams - 1) / 2;
    }
    std::vector<Num> x(n_items);
    std::vector<Num> y(n_items);
    for (count_type i = 0; i < n_items; ++i) {
      x[i] = params[2 * i + 1];
      y[i] = params[2 * i + 2];
    }
    return interpolate1d(params[0], x, y, errcode);
  }

  //! Piecewise constant gradient of the reconstruction from interpol
  template <typename Num>
  Num deriv_interpol(count_type nparams, const Num* params, void*, parser<Num>*,
      status&,
      error& errcode) {
    count_type n_items;

    if (nparams % 2 == 0) {
      //EPOCH style interpolate
      n_items = (nparams - 2) / 2;
      if (static_cast<offset_type>(n_items) != std::lround(params[nparams - 1])) {
        errcode = err_wrong_parameters;
        return Num(0);
      }
    } else {
      //EIS style interpolate
      n_items = (nparams - 1) / 2;
    }
    std::vector<Num> x(n_items);
    std::vector<Num> y(n_items);
    for (count_type i = 0; i < n_items; ++i) {
      x[i] = params[2 * i + 1];
      y[i] = params[2 * i + 2];
    }
    return deriv_interpolate1d(params[0], x, y, errcode);
  }

  //! Bessel function of the first kind
  template <typename Num>
  Num bessel_j(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    if (params[0] < Num(0)) {
      errcode = errcode | err_maths_domain;
      return Num(0);
    }

#if defined(__unix__) || defined(__APPLE__)
    return static_cast<Num>(::jn(static_cast<int>(std::lround(params[0])),
        static_cast<double>(params[1])));
#else
    const long order = std::lround(params[0]);
    if (params[1] < Num(0)) {
      //For integer order J(-x) = (-1)**order * J(x)
      const Num res = std::cyl_bessel_j(static_cast<Num>(order), -params[1]);
      return (order % 2 != 0) ? -res : res;
    }
    return std::cyl_bessel_j(static_cast<Num>(order), params[1]);
#endif
  }

  //! Bessel function of the second kind
  template <typename Num>
  Num bessel_y(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    if (params[0] < Num(0)) {
      errcode = errcode | err_maths_domain;
      return Num(0);
    }

#if defined(__unix__) || defined(__APPLE__)
    return static_cast<Num>(::yn(static_cast<int>(std::lround(params[0])),
        static_cast<double>(params[1])));
#else
    if (params[1] < Num(0)) return std::numeric_limits<Num>::quiet_NaN();
    return std::cyl_neumann(static_cast<Num>(std::lround(params[0])),
        params[1]);
#endif
  }

  //! Error function
  template <typename Num>
  Num erf(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::erf(params[0]);
  }

  //! Complementary error function
  template <typename Num>
  Num erfc(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::erfc(params[0]);
  }

  //! Exponentially scaled complementary error function
  template <typename Num>
  Num erfc_scaled(count_type, const Num* params, void*, parser<Num>*,
      status&, error&) {
    return std::exp(params[0] * params[0]) * std::erfc(params[0]);
  }

  //! Bernoulli gamma function
  template <typename Num>
  Num gamma_fn(count_type, const Num* params, void*, parser<Num>*,
      status&, error& errcode) {
    if (params[0] <= Num(0)) {
      errcode = errcode | err_maths_domain;
      return Num(0);
    }

    return std::tgamma(params[0]);
  }

  //! Log of Bernoulli gamma function
  template <typename Num>
  Num log_gamma_fn(count_type, const Num* params, void*, parser<Num>*,
      status&,
      error& errcode) {
    if (params[0] <= Num(0)) {
      errcode = errcode | err_maths_domain;
      return Num(0);
    }

    return std::lgamma(params[0]);
  }

}

#endif
