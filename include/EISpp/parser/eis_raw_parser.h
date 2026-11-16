// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_RAW_PARSER_H
#define EIS_RAW_PARSER_H

#include <cstddef>
#include <cstdio>
#include <string>

#include "../common/eis_constants.h"
#include "../common/eis_utils.h"

namespace eis {

  //! Compare two strings for equality
  //! @param[in] str1 - First string to test for equality
  //! @param[in] str2 - Second string to test for equality
  //! @return strcmp - Logical result for string equality
  inline bool strcmp(const std::string& str1, const std::string& str2) {
    if (detail::trimmed(str1).size() != detail::trimmed(str2).size()) {
      return false;
    }
    return detail::trimmed(str1) == detail::trimmed(str2);
  }

  //! Turn a string into a real
  //! @param[in] str_in - String to test for being a real
  //! @param[inout] err - Error code for conversion test
  //! result will have the err_bad_value bit set if
  //! str_in cannot be parsed as a real
  //! @return parse_string_as_real - Number that str_in can be parsed to
  template <typename Num>
  Num parse_string_as_real(const std::string& str_in, error& err) {
    double value = 0.0;
    int consumed = 0;

    const bool parsed = std::sscanf(str_in.c_str(), "%lf%n", &value,
        &consumed) == 1
        && str_in.find_first_not_of(' ', static_cast<count_type>(consumed))
        == std::string::npos;
    if (!parsed) err = err | err_bad_value;
    return static_cast<Num>(value);
  }

  //! Turn a string into an integer
  //! @param[in] str_in - String to test for being an integer
  //! @param[inout] err - Error code for conversion test
  //! result will have the err_bad_value bit set if
  //! str_in cannot be parsed as an integer
  //! @return parse_string_as_integer - Number that str_in can be parsed to
  inline i8 parse_string_as_integer(const std::string& str_in, error& err) {
    long long value = 0;
    int consumed = 0;

    const bool parsed = std::sscanf(str_in.c_str(), "%lld%n", &value,
        &consumed) == 1
        && str_in.find_first_not_of(' ', static_cast<count_type>(consumed))
        == std::string::npos;
    if (!parsed) err = err | err_bad_value;
    return static_cast<i8>(value);
  }

}

#endif
