// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_CONSTANTS_H
#define EIS_CONSTANTS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace eis {
  //! Version numbers of EIS
  struct version_numbers {
    int v1; //!< First digit
    int v2; //!< Second digit
    int v3; //!< Third digit
  };

  //! Returns the version as a set of integers
  //! Uses accessor function rather than variables
  //! against future version wanting to have a more complex version system
  //! (backward compatability modes etc.)
  constexpr version_numbers get_version_numbers() noexcept {
    return {1, 0, 0};
  }

  //! 4 byte integers used in EIS
  using i4 = std::int32_t;
  //! 8 byte integers used in EIS
  using i8 = std::int64_t;
  //! Unsigned 4 byte integers used in EIS
  using ui4 = std::make_unsigned_t<i4>;
  //! Unsigned 8 byte integers used in EIS
  using ui8 = std::make_unsigned_t<i8>;
  //! Integers used as error codes in EIS
  using error = ui8;
  //! Integers used as status codes in EIS
  using status = ui8;
  //! Integers used for user bitmasks in EIS
  using bitmask = ui8;
  //! Integers used as type codes in EIS
  using typecode = i4;
  //! Integers used for UIDs from the UID generator
  using uid_kind = i4;
  //! Integers used for counts and indices in EIS
  using count_type = std::size_t;
  //! Signed integers used for counts and positions that can be negative in EIS
  using offset_type = std::ptrdiff_t;

  //! Returns the version as a string. level is the number of digits of the
  //! version to include, from 1 to 3. Values outside that range are clamped
  inline std::string get_version_string(offset_type level = 3) {
    level = std::clamp<offset_type>(level, 1, 3);
    const version_numbers numbers = get_version_numbers();

    std::string str = std::to_string(numbers.v1);
    if (level >= 2) str += "." + std::to_string(numbers.v2);
    if (level >= 3) str += "." + std::to_string(numbers.v3);
    return str;
  }

  //! Returns a string detailing the version of EIS being used. If needed
  //! other copyright information will be returned here
  inline std::string get_version_report(offset_type level = 3) {
    return "Using EIS-2 C++ parser version " + get_version_string(level);
  }

  //! TINY value for variables of type Num
  template <typename Num>
  inline constexpr Num tiny = std::numeric_limits<Num>::min();
  //! HUGE value for variables of type Num
  template <typename Num>
  inline constexpr Num huge = std::numeric_limits<Num>::max();

  //! No error condition
  inline constexpr error err_none = 0;
  //! Error in parser
  inline constexpr error err_parser = error(1) << 0;
  //! Error in simplify
  inline constexpr error err_simplifier = error(1) << 1;
  //! Error in emplace
  inline constexpr error err_emplacer = error(1) << 2;
  //! Error in evaluate
  inline constexpr error err_evaluator = error(1) << 3;
  //! Error in file
  inline constexpr error err_file = error(1) << 4;
  //! Error in deck file
  inline constexpr error err_deck_file = error(1) << 5;
  //! Error in deck definition
  inline constexpr error err_deck_definition = error(1) << 6;
  //! Error in deck parser
  inline constexpr error err_deck_parser = error(1) << 7;
  //! Name not found
  inline constexpr error err_not_found = error(1) << 8;
  //! Malformed mathematical expression
  inline constexpr error err_malformed = error(1) << 9;
  //! Incorrect number of parameters specified
  inline constexpr error err_wrong_parameters = error(1) << 10;
  //! Specified expression is mathematically invalid (log of 0 etc.)
  inline constexpr error err_maths_domain = error(1) << 11;
  //! Value is nonsensical in non-mathematical way
  inline constexpr error err_bad_value = error(1) << 12;
  //! Stack has deferred elements that cannot be resolved
  inline constexpr error err_has_deferred = error(1) << 13;
  //! Stack has emplaced elements that have not been resolved
  inline constexpr error err_has_emplaced = error(1) << 14;
  //! Attempted to subscript a constant as though it was a function
  inline constexpr error err_bracketed_constant = error(1) << 15;
  //! Extra bracket in expression
  inline constexpr error err_extra_bracket = error(1) << 16;
  //! Stack specified through interoperable interface is invalid
  inline constexpr error err_bad_stack = error(1) << 17;
  //! Stack returned more results through interoperable interface than expected
  inline constexpr error err_extra_results = error(1) << 18;
  //! String handler couldn't find a LUN when loading a file
  inline constexpr error err_no_luns = error(1) << 19;
  //! String handler couldn't find a specified file
  inline constexpr error err_no_file = error(1) << 20;
  //! String handler was given an invalid serialisation string
  inline constexpr error err_malformed_file = error(1) << 21;
  //! Deck handler found an unmatched start and end block description
  inline constexpr error err_mismatched_begin_end = error(1) << 22;
  //! Deck handler found a deck block that was deeper than permitted
  inline constexpr error err_deck_too_deep = error(1) << 23;
  //! Deck handler found an unpermitted empty block
  inline constexpr error err_deck_empty_block = error(1) << 24;
  //! Deck handler found an unpermitted key in the root block
  inline constexpr error err_root_keys = error(1) << 25;
  //! Deck handler tried to use an empty definition to parse a deck
  inline constexpr error err_bad_deck_definition = error(1) << 26;
  //! Deck handler found an unknown block in a deck
  inline constexpr error err_unknown_block = error(1) << 27;
  //! Deck handler found an unknown key in a deck
  inline constexpr error err_unknown_key = error(1) << 28;
  //! Deck handler found a key that it was unable to handle
  inline constexpr error err_bad_key = error(1) << 29;
  //! Fatal error was reported by the host code
  inline constexpr error err_host = error(1) << 30;
  //! Function had a text parameter that should not have had a text parameter
  inline constexpr error err_text = error(1) << 31;
  //! Trying to use a function that wants an interoperable parser but
  //! the actual parser is not interoperable
  inline constexpr error err_interop = error(1) << 32;
  //! Parameter was valid mathematically but was out of range in some other sense
  inline constexpr error err_out_of_range = error(1) << 33;
  //! Stack variable that returns more than one parameter was used in a maths
  //! expression
  inline constexpr error err_stack_params = error(1) << 34;
  //! Trying to perform an operation that is invalid after minification
  inline constexpr error err_invalid_minify = error(1) << 35;
  //! Requested a bad parser through the interoperability interface
  inline constexpr error err_bad_parser = error(1) << 36;
  //! No valid derivative has been found
  inline constexpr error err_no_deriv = error(1) << 37;

  //! No status specified
  inline constexpr status status_none = 0;
  //! Terminate operations status
  inline constexpr status status_terminate = status(1) << 0;
  //! Interoperable stack is a temporary and should be deleted
  inline constexpr status status_delete_stack = status(1) << 1;
  //! This element in a stack should not be simplified through
  inline constexpr status status_no_simplify = status(1) << 2;
  //! This element in a stack is an emplaced value that should not be emplaced
  //! yet
  inline constexpr status status_no_emplace = status(1) << 3;
  //! This deck key is a directive (of the form key:value rather than
  //! key = value)
  inline constexpr status status_directive = status(1) << 4;
  //! This deck key or block handling function has decided that it cannot
  //! handle this key or block
  inline constexpr status status_not_handled = status(1) << 5;

  //! Function called whenever an error is added to an error handler
  using on_error_callback = void (*)(error);

}

#endif
