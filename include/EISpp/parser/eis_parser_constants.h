// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_PARSER_CONSTANTS_H
#define EIS_PARSER_CONSTANTS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "../common/eis_constants.h"

namespace eis {

  //! Function definition for the parser evaluator function
  //! The parser class. Declared here so that function types can take a parser
  template <typename Num>
  class parser;

  template <typename Num>
  using parser_eval_fn = Num (*)(count_type nparams, const Num* params,
      void* host_params, parser<Num>* calling_parser, status& status_code,
      error& errcode);

  //! Functor. Carries persistent information with it and is called in the
  //! same way as a parser evaluator function
  template <typename Num>
  using functor = std::function<Num(count_type nparams, const Num* params,
      void* host_params, parser<Num>* calling_parser, status& status_code,
      error& errcode)>;

  template <typename Num>
  using parser_result_function = void (*)(count_type& nresults, Num* results,
      void* host_params, parser<Num>* calling_parser, status& status_code,
      error& errcode);

  //! Evaluator function supplied by C code
  template <typename Num>
  using c_parser_eval_fn = Num (*)(int nparams, Num* params, void* host_params,
      std::int64_t* status_code, std::int64_t* errcode);

  //! Result function supplied by C code
  template <typename Num>
  using c_parser_result_function = void (*)(int* nresults, Num* results,
      void* host_params, std::int64_t* status_code, std::int64_t* errcode);

  //! What a stack element or a symbol in the registry evaluates to
  template <typename Num>
  using element_data = std::variant<std::monostate, Num, std::int32_t*,
      std::int64_t*, float*, double*, parser_eval_fn<Num>, functor<Num>*,
      functor<Num>, c_parser_eval_fn<Num>>;

  //! No physical units specified
  inline constexpr typecode physics_none = 0;
  //! SI system
  inline constexpr typecode physics_si = 1;
  //! CGS Gaussian units
  inline constexpr typecode physics_cgs_gauss = 2;

  // block type constants
  //! Type variable
  inline constexpr typecode pt_variable = 1;
  //! Type constant
  inline constexpr typecode pt_constant = 2;
  //! Type operator
  inline constexpr typecode pt_operator = 3;
  //! Type function
  inline constexpr typecode pt_function = 4;
  //! Type parenthesis
  inline constexpr typecode pt_parenthesis = 5;
  //! Type separator
  inline constexpr typecode pt_separator = 6;
  //! Type character
  inline constexpr typecode pt_character = 7;
  //! Type stack variable (stored variable)
  inline constexpr typecode pt_stored_variable = 8;
  //! Type deferred variable
  inline constexpr typecode pt_deferred_variable = 9;
  //! Type deferred function
  inline constexpr typecode pt_deferred_function = 10;
  //! Type emplaced variable
  inline constexpr typecode pt_emplaced_variable = 11;
  //! Type emplaced function
  inline constexpr typecode pt_emplaced_function = 12;
  //! Type pointer variable
  inline constexpr typecode pt_pointer_variable = 13;
  //! Type functor
  inline constexpr typecode pt_functor = 14;
  //! Parameter to a deck function
  inline constexpr typecode pt_param = 15;
  //! Derivative parameter to a function
  inline constexpr typecode pt_dparam = 16;
  //! Stack specified function
  inline constexpr typecode pt_stack_function = 17;
  //! Actual zero
  inline constexpr typecode pt_zero = 18;
  //! Actual unity
  inline constexpr typecode pt_unity = 19;
  //! Addition operator
  inline constexpr typecode pt_op_plus = 20;
  //! Subtraction operator
  inline constexpr typecode pt_op_minus = 21;
  //! Multiplication operator
  inline constexpr typecode pt_op_multiply = 22;
  //! Division operator
  inline constexpr typecode pt_op_divide = 23;
  //! Raise to the power of operator
  inline constexpr typecode pt_op_power = 24;
  //! Unitary plus
  inline constexpr typecode pt_op_unary_plus = 25;
  //! Unitary minus
  inline constexpr typecode pt_op_unary_minus = 26;
  //! Function used in deriv
  inline constexpr typecode pt_placed_function = 27;
  //! True integer
  inline constexpr typecode pt_integer = 28;
  //! Derivative function
  inline constexpr typecode pt_deriv_fn = 29;
  //! Conditional jump. Pops a value and skips the number of elements held in
  //! the value of the element if the popped value is zero
  inline constexpr typecode pt_jump_conditional = 30;
  //! Unconditional jump. Skips the number of elements held in the value of the
  //! element
  inline constexpr typecode pt_jump_unconditional = 31;
  //! Conditional function. Only the selected parameter is evaluated
  inline constexpr typecode pt_control_if = 32;
  //! End of a conditional function that has been converted to jumps
  inline constexpr typecode pt_control_if_join = 33;
  //! Block is of bad type (invalid)
  inline constexpr typecode pt_bad = 1024;
  //! Block is of null (empty) type
  inline constexpr typecode pt_null = 1025;

  // Associativity constants
  //! No associativity specified
  inline constexpr typecode assoc_null = 0;
  //! Fully associative operator
  inline constexpr typecode assoc_a = 1;
  //! Left associative operator
  inline constexpr typecode assoc_la = 2;
  //! Right associative operator
  inline constexpr typecode assoc_ra = 3;

  //! Parenthesis is left bracket
  inline constexpr typecode paren_left_bracket = 1;
  //! Parenthesis is right bracket
  inline constexpr typecode paren_right_bracket = 2;

  inline constexpr typecode registry_none = 0;
  inline constexpr typecode registry_global = 1;
  inline constexpr typecode registry_local = 2;

  //! Symbol display information
  inline constexpr typecode fsn_auto = 0;
  inline constexpr typecode fsn_always = 1;
  inline constexpr typecode fsn_never = 2;

  //! Information about a stack element that is needed during parsing but is
  //! not essential for evaluation
  struct stack_co_element {
    typecode associativity;
    i4 precedence;
    offset_type expected_params = -1;
    bool can_have_string_params = false;
    bool can_accept_maths_domain_errors = false;
    //! Does calling this element a second time with the same parameters in the
    //! same evaluation have no further effect and return the same result.
    //! Elements that have this can be merged when they are repeated
    bool is_idempotent = false;
    //! Parameter positions that must be text. Bit 0 is the first parameter. 0
    //! means that positions are not checked
    bitmask text_param_mask = 0;
    offset_type charindex = -1;
    offset_type full_line_pos = -1;
    bool defer = false;
    bitmask cap_bits = 0;
    std::string text, full_line, filename;
    offset_type line_number = -1;
    typecode registry_type = registry_none;
    offset_type deriv_values = -1;

    stack_co_element() = default;
    stack_co_element(const stack_co_element& src) {
      *this = src;
    }

    stack_co_element& operator=(const stack_co_element& src) {
      associativity = src.associativity;
      precedence = src.precedence;
      expected_params = src.expected_params;
      can_have_string_params = src.can_have_string_params;
      charindex = src.charindex;
      full_line_pos = src.full_line_pos;
      defer = src.defer;
      cap_bits = src.cap_bits;
      text = src.text;
      full_line = src.full_line;
      filename = src.filename;
      line_number = src.line_number;
      registry_type = src.registry_type;
      deriv_values = src.deriv_values;
      can_accept_maths_domain_errors = src.can_accept_maths_domain_errors;
      is_idempotent = src.is_idempotent;
      text_param_mask = src.text_param_mask;
      return *this;
    }
  };

  //! Information about a stack element that is always needed for the entire
  //! stack lifetime
  //! data holds what the element evaluates to: monostate if there is nothing,
  //! a number (the numerical_data of the Fortran), a pointer to an integer or
  //! real variable, a function pointer, a pointer to a functor that is shared
  //! with other stacks, or a functor that belongs to this element (a per stack
  //! functor)
  template <typename Num>
  struct stack_element {
    typecode ptype;
    typecode rtype;
    offset_type value;
    offset_type actual_params = -1;
    bool has_string_params = false;
    bool can_simplify = true;
    bool per_stack_functor = false;
    element_data<Num> data;

    stack_element() = default;
    stack_element(const stack_element& src) {
      *this = src;
    }

    //! Copy of an element. Every field is copied, as for the Fortran intrinsic
    //! assignment. A functor that belongs to the source is shared with the
    //! source rather than copied
    stack_element& operator=(const stack_element& src) {
      if (this == &src) return *this;
      ptype = src.ptype;
      rtype = src.rtype;
      value = src.value;
      actual_params = src.actual_params;
      has_string_params = src.has_string_params;
      can_simplify = src.can_simplify;
      per_stack_functor = src.per_stack_functor;
      if (const auto* owned = std::get_if<functor<Num>>(&src.data)) {
        data = const_cast<functor<Num>*>(owned);
      } else {
        data = src.data;
      }
      return *this;
    }

    stack_element(stack_element&& src) noexcept {
      *this = std::move(src);
    }

    //! Move of an element. A functor that belongs to the source belongs to the
    //! destination afterwards
    stack_element& operator=(stack_element&& src) noexcept {
      if (this == &src) return *this;
      ptype = src.ptype;
      rtype = src.rtype;
      value = src.value;
      actual_params = src.actual_params;
      has_string_params = src.has_string_params;
      can_simplify = src.can_simplify;
      per_stack_functor = src.per_stack_functor;
      data = std::move(src.data);
      return *this;
    }
  };

  //! Assignment of a single element. has_string_params and per_stack_functor
  //! are not copied and a functor that belongs to the source is shared with
  //! the source rather than copied
  template <typename Num>
  void element_assign(stack_element<Num>& dest,
      const stack_element<Num>& src) {
    dest.ptype = src.ptype;
    dest.rtype = src.rtype;
    dest.value = src.value;
    dest.actual_params = src.actual_params;
    dest.has_string_params = false;
    dest.can_simplify = src.can_simplify;
    dest.per_stack_functor = false;
    if (const auto* owned = std::get_if<functor<Num>>(&src.data)) {
      dest.data = const_cast<functor<Num>*>(owned);
    } else {
      dest.data = src.data;
    }
  }

  //! Assignment of an array of elements. has_string_params is not copied. If
  //! per_stack_functor is set in the source then the functor is copied,
  //! otherwise it is shared with the source
  template <typename Num>
  void element_array_assign(std::span<stack_element<Num>> dest,
      std::span<const stack_element<Num>> src) {
    for (std::size_t i = 0; i < dest.size(); ++i) {
      dest[i].ptype = src[i].ptype;
      dest[i].rtype = src[i].rtype;
      dest[i].value = src[i].value;
      dest[i].actual_params = src[i].actual_params;
      dest[i].has_string_params = false;
      dest[i].can_simplify = src[i].can_simplify;
      const auto* shared = std::get_if<functor<Num>*>(&src[i].data);
      const auto* owned = std::get_if<functor<Num>>(&src[i].data);
      if (src[i].per_stack_functor) {
        if (shared != nullptr) {
          dest[i].data = **shared;
        } else {
          dest[i].data = src[i].data;
        }
      } else {
        if (owned != nullptr) {
          dest[i].data = const_cast<functor<Num>*>(owned);
        } else {
          dest[i].data = src[i].data;
        }
      }
      dest[i].per_stack_functor = src[i].per_stack_functor;
    }
  }

  //! Assignment of an array of co-elements
  inline void co_element_assign_array(std::span<stack_co_element> dest,
      std::span<const stack_co_element> src) {
    for (std::size_t i = 0; i < dest.size(); ++i) {
      dest[i].associativity = src[i].associativity;
      dest[i].precedence = src[i].precedence;
      dest[i].expected_params = src[i].expected_params;
      dest[i].can_have_string_params = src[i].can_have_string_params;
      dest[i].charindex = src[i].charindex;
      dest[i].full_line_pos = src[i].full_line_pos;
      dest[i].defer = src[i].defer;
      dest[i].cap_bits = src[i].cap_bits;
      dest[i].text = src[i].text;
      dest[i].full_line = src[i].full_line;
      dest[i].filename = src[i].filename;
      dest[i].line_number = src[i].line_number;
      dest[i].registry_type = src[i].registry_type;
      dest[i].deriv_values = src[i].deriv_values;
      dest[i].can_accept_maths_domain_errors =
          src[i].can_accept_maths_domain_errors;
      dest[i].is_idempotent = src[i].is_idempotent;
      dest[i].text_param_mask = src[i].text_param_mask;
    }
  }

  //! Entire tokenized stack
  template <typename Num>
  struct stack {
    std::vector<stack_element<Num>> entries;
    std::vector<stack_co_element> co_entries;
    parser_result_function<Num> eval_fn = nullptr;
    c_parser_result_function<Num> c_eval_fn = nullptr;
    std::string eval_string, full_line;
    std::string filename;
    offset_type line_number = -1;
    offset_type char_offset = 0;
    count_type params = 0;
    bitmask cap_bits = 0;
    count_type stack_point = 0, stack_size = 0;
    bool init = false;
    bool sanity_checked = false;
    bool has_emplaced = false;
    bool has_deferred = false;
    bool can_accept_maths_domain_errors = false;
    //! The parser that this stack belongs to. Used only to compare identity.
    //! Null means that the stack is not tied to a parser and bypasses checks
    const void* owner = nullptr;
  };

  //! Description of the function that is used to emplace a function
  template <typename Num>
  using parser_late_bind_fn = void (*)(std::string_view orig_string,
      count_type nparams, const Num* params, void* host_params,
      parser<Num>* calling_parser, stack<Num>& stack_out,
      status& status_code, error& errcode);

  template <typename Num>
  using parser_late_bind_stack_fn = void (*)(std::string_view orig_string,
      count_type nparams, const stack<Num>* params, void* host_params,
      parser<Num>* calling_parser, stack<Num>& stack_out,
      status& status_code, error& errcode);

  //! Emplacement function that can hold state
  template <typename Num>
  using parser_late_bind_functor = std::function<void(
      std::string_view orig_string, count_type nparams, const Num* params,
      void* host_params, parser<Num>* calling_parser, stack<Num>& stack_out,
      status& status_code, error& errcode)>;

}

#endif
