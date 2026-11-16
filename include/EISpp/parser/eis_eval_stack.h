// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_EVAL_STACK_H
#define EIS_EVAL_STACK_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "../common/eis_constants.h"
#include "../common/eis_error.h"
#include "eis_parser_constants.h"

namespace eis {

  //! The parser class. Declared here so that the evaluator can refer to it
  template <typename Num>
  class parser;

  template <typename Num>
  struct eval_stack {
    std::vector<Num> entries;
    std::vector<Num> fn_call_vals;
    //! The parser that this evaluator is used by
    parser<Num>* host_parser = nullptr;
    count_type stack_point = 0;

    //! Resize the backing store. Will not become smaller than
    //! the part of the stack that is in use
    //! @param[in] n_elements - Number of elements to set the size to
    void resize(count_type n_elements, error&) {
      const count_type newcount =
          std::max({n_elements, stack_point, count_type(1)});

      std::vector<Num> temp(newcount);
      if (!entries.empty()) {
        for (count_type i = 0; i < stack_point; ++i) {
          temp[i] = entries[i];
        }
      }
      entries = std::move(temp);
    }

    //! Push a value onto a stack
    //! @param[in] value - Value to push on the stack
    //! @param[inout] errcode - Error code information
    void push(Num value, error& errcode) {
      if (entries.empty()) resize(100, errcode);
      if (stack_point == entries.size()) {
        resize(stack_point * 2, errcode);
      }

      stack_point = stack_point + 1;
      entries[stack_point - 1] = value;
    }

    //! Get the value from the top of the stack
    //! @param[out] value - Value to pop off the stack
    //! @param[inout] errcode - Error code information
    void pop_scalar(Num& value, error&) {
      if (stack_point > 0) {
        value = entries[stack_point - 1];
        stack_point = stack_point - 1;
      }
    }

    //! Get the values from the top of the stack
    //! @param[in] value_count - Number of values to pop off the stack
    //! @param[out] values - Array to hold the popped values
    //! @param[inout] errcode - Error code information
    void pop_vector(count_type value_count, Num* values, error&) {
      if (value_count == 0) return;

      if (stack_point >= value_count) {
        for (count_type i = 0; i < value_count; ++i) {
          values[i] = entries[stack_point - value_count + i];
        }
        stack_point = stack_point - value_count;
      }
    }

    //! Evaluate a single stack element
    //! @param[in] element - Element to be evaluated
    //! @param[in] host_params - Host code supplied parameters
    //! @param[inout] status_code - Status code information
    //! @param[inout] errcode - Error code information
    //! @param[in] force_trip_maths_domain - Should maths domain errors always
    //! trigger
    void eval_element(const stack_element<Num>& element, void* host_params,
        status& status_code, error& errcode,
        bool force_trip_maths_domain = false) {
      [[maybe_unused]] bool maths_domain_trip = force_trip_maths_domain;

      //If a ptype constant has made it this far it should be evaluated as a
      //constant
      if (element.ptype == pt_constant) {
        push(std::get<Num>(element.data), errcode);
      } else if (element.rtype == pt_pointer_variable) {
        const auto* i32 = std::get_if<std::int32_t*>(&element.data);
        if (i32 != nullptr && *i32 != nullptr) {
          push(static_cast<Num>(**i32), errcode);
        }
        const auto* i64 = std::get_if<std::int64_t*>(&element.data);
        if (i64 != nullptr && *i64 != nullptr) {
          push(static_cast<Num>(**i64), errcode);
        }
        const auto* r32 = std::get_if<float*>(&element.data);
        if (r32 != nullptr && *r32 != nullptr) {
          push(static_cast<Num>(**r32), errcode);
        }
        const auto* r64 = std::get_if<double*>(&element.data);
        if (r64 != nullptr && *r64 != nullptr) {
          push(static_cast<Num>(**r64), errcode);
        }
      } else {
        const auto* fn = std::get_if<parser_eval_fn<Num>>(&element.data);
        const auto* shared = std::get_if<functor<Num>*>(&element.data);
        const auto* owned = std::get_if<functor<Num>>(&element.data);
        const auto* c_fn = std::get_if<c_parser_eval_fn<Num>>(&element.data);
        const count_type params =
            static_cast<count_type>(element.actual_params);
        if (fn != nullptr && *fn != nullptr) {
          if (fn_call_vals.empty()) {
            fn_call_vals.resize(params);
          } else if (params > fn_call_vals.size()) {
            fn_call_vals.clear();
            fn_call_vals.resize(params);
          }
          pop_vector(params, fn_call_vals.data(), errcode);
          push((*fn)(params, fn_call_vals.data(), host_params, host_parser,
              status_code, errcode), errcode);
        } else if ((shared != nullptr && *shared != nullptr)
            || owned != nullptr) {
          if (fn_call_vals.empty()) {
            fn_call_vals.resize(params);
          } else if (params > fn_call_vals.size()) {
            fn_call_vals.clear();
            fn_call_vals.resize(params);
          }
          pop_vector(params, fn_call_vals.data(), errcode);
          const functor<Num>& call = (shared != nullptr) ? **shared : *owned;
          push(call(params, fn_call_vals.data(), host_params, host_parser,
              status_code, errcode), errcode);
        } else if (c_fn != nullptr && *c_fn != nullptr) {
          if (fn_call_vals.empty()) {
            fn_call_vals.resize(params);
          } else if (params > fn_call_vals.size()) {
            fn_call_vals.clear();
            fn_call_vals.resize(params);
          }
          pop_vector(params, fn_call_vals.data(), errcode);
          push((*c_fn)(static_cast<int>(params), fn_call_vals.data(),
              host_params, reinterpret_cast<std::int64_t*>(&status_code),
              reinterpret_cast<std::int64_t*>(&errcode)), errcode);
        }
      }
    }

    //! Pull the bottom element of the stack and push the stack down
    //! @param[out] value - Value removed from the bottom of the stack
    //! @param[inout] errcode - Error code information
    void trim_first(Num& value, error&) {
      if (stack_point > 0) {
        value = entries[0];
        for (count_type i = 0; i < stack_point - 1; ++i) {
          entries[i] = entries[i + 1];
        }
        stack_point = stack_point - 1;
      }
    }

    //! Get the number of stack elements to skip after a jump element. A
    //! conditional jump pops the condition from the stack and does not skip
    //! any elements if it is non zero
    //! @param[in] element - Jump element
    //! @param[inout] errcode - Error code information
    count_type jump_offset(const stack_element<Num>& element, error& errcode) {
      if (element.ptype == pt_jump_conditional) {
        Num condition = Num(0);

        pop_scalar(condition, errcode);
        if (std::abs(condition) > tiny<Num>) return 0;
      }
      return static_cast<count_type>(element.value);
    }

    //! Evaluate a stack to a set of results. If the result array is too small
    //! to hold the results then it is reallocated to be large enough to hold
    //! them
    //! @param[in] stack - Stack to evaluate
    //! @param[inout] result_vals - Array holding the results of the evaluation
    //! @param[in] host_params - Host code user supplied parameters
    //! @param[inout] errcode - Error code
    //! @param[inout] err_handler - Error handler instance
    //! @return result_count
    count_type evaluate(const eis::stack<Num>& stack,
        std::vector<Num>& result_vals, void* host_params, error& errcode,
        error_handler& err_handler) {
      error err;
      status stat_in;
      status status_value = status_none;
      count_type result_count;

      errcode = err_none;

      if (stack.eval_fn != nullptr) {
        if (result_vals.empty()) result_vals.resize(1);
        if (result_vals.size() < 1) {
          result_vals.clear();
          result_vals.resize(1);
        }
        result_count = result_vals.size();
        stack.eval_fn(result_count, result_vals.data(), host_params,
            host_parser, stat_in, errcode);
        if (result_count > result_vals.size() && errcode == err_none) {
          result_vals.clear();
          result_vals.resize(result_count);
          stack.eval_fn(result_count, result_vals.data(), host_params,
              host_parser, stat_in, errcode);
        }
        if (errcode != err_none) {
          if (!stack.eval_string.empty()) {
            add_error_options err_options;
            err_options.errstring = stack.eval_string;
            err_options.charindex = 0;
            err_handler.add_error(err_evaluator, errcode, err_options);
          } else {
            err_handler.add_error(err_evaluator, errcode);
          }
        }
        return result_count;
      }

      if (stack.c_eval_fn != nullptr) {
        int c_count;

        stat_in = status_none;
        if (result_vals.empty()) result_vals.resize(1);
        c_count = static_cast<int>(result_vals.size());
        stack.c_eval_fn(&c_count, result_vals.data(), host_params,
            reinterpret_cast<std::int64_t*>(&stat_in),
            reinterpret_cast<std::int64_t*>(&errcode));
        result_count = static_cast<count_type>(c_count);
        if (result_count > result_vals.size() && errcode == err_none) {
          result_vals.clear();
          result_vals.resize(result_count);
          stack.c_eval_fn(&c_count, result_vals.data(), host_params,
              reinterpret_cast<std::int64_t*>(&stat_in),
              reinterpret_cast<std::int64_t*>(&errcode));
          result_count = static_cast<count_type>(c_count);
        }
        if (errcode != err_none) {
          if (!stack.eval_string.empty()) {
            add_error_options err_options;
            err_options.errstring = stack.eval_string;
            err_options.charindex = 0;
            err_handler.add_error(err_evaluator, errcode, err_options);
          } else {
            err_handler.add_error(err_evaluator, errcode);
          }
        }
        return result_count;
      }

      for (count_type istack = 0; istack < stack.stack_point; ++istack) {
        if (stack.entries[istack].ptype == pt_jump_conditional
            || stack.entries[istack].ptype == pt_jump_unconditional) {
          istack = istack + jump_offset(stack.entries[istack], err);
          continue;
        }
        err = err_none;
        stat_in = status_value;
        eval_element(stack.entries[istack], host_params, stat_in, err);
        status_value = status_value | stat_in;
        if (err != err_none && !(err == err_maths_domain
            && stack.can_accept_maths_domain_errors)) {
          if (!stack.co_entries.empty()) {
            const stack_co_element& coelement = stack.co_entries[istack];
            if (!coelement.full_line.empty()) {
              if (stack.filename.empty()) {
                err_handler.add_error(err_evaluator, err,
                    {.errstring = coelement.text,
                    .charindex = coelement.charindex,
                    .filename = coelement.filename,
                    .line_number = coelement.line_number,
                    .full_line = coelement.full_line,
                    .full_line_pos = coelement.full_line_pos});
              } else {
                err_handler.add_error(err_evaluator, err,
                    {.errstring = coelement.text,
                    .charindex = coelement.charindex,
                    .filename = coelement.filename,
                    .line_number = coelement.line_number,
                    .full_line = coelement.full_line,
                    .full_line_pos = coelement.full_line_pos,
                    .context_line_number = stack.line_number,
                    .context_filename = stack.filename});
              }
            } else {
              err_handler.add_error(err_evaluator, err,
                  {.errstring = coelement.text,
                  .charindex = coelement.charindex});
            }
          } else {
            err_handler.add_error(err_evaluator, err);
          }
        }
        errcode = errcode | err;
      }

      if (stack.can_accept_maths_domain_errors) {
        errcode = errcode & ~err_maths_domain;
      }

      if (result_vals.empty()) {
        result_vals.resize(stack_point);
      } else if (result_vals.size() < stack_point) {
        result_vals.clear();
        result_vals.resize(stack_point);
      }

      result_count = stack_point;
      pop_vector(stack_point, result_vals.data(), errcode);

      return result_count;
    }

    //! Evaluate a stack to a set of results. If the result array is too small
    //! to hold the results then it is reallocated to be large enough to hold
    //! them. This version does not report errors to an error handler
    //! @param[in] stack - Stack to evaluate
    //! @param[inout] result_vals - Array holding the results of the evaluation
    //! @param[in] host_params - Host code user supplied parameters
    //! @param[inout] errcode - Error code
    //! @return result_count
    count_type evaluate(const eis::stack<Num>& stack,
        std::vector<Num>& result_vals, void* host_params, error& errcode) {
      error err;
      status stat_in;
      status status_value = status_none;
      count_type result_count;

      errcode = err_none;

      if (stack.eval_fn != nullptr) {
        if (result_vals.empty()) result_vals.resize(1);
        if (result_vals.size() < 1) {
          result_vals.clear();
          result_vals.resize(1);
        }
        result_count = result_vals.size();
        stack.eval_fn(result_count, result_vals.data(), host_params,
            host_parser, stat_in, errcode);
        if (result_count > result_vals.size() && errcode == err_none) {
          result_vals.clear();
          result_vals.resize(result_count);
          stack.eval_fn(result_count, result_vals.data(), host_params,
              host_parser, stat_in, errcode);
        }
        return result_count;
      }

      if (stack.c_eval_fn != nullptr) {
        int c_count;

        stat_in = status_none;
        if (result_vals.empty()) result_vals.resize(1);
        c_count = static_cast<int>(result_vals.size());
        stack.c_eval_fn(&c_count, result_vals.data(), host_params,
            reinterpret_cast<std::int64_t*>(&stat_in),
            reinterpret_cast<std::int64_t*>(&errcode));
        result_count = static_cast<count_type>(c_count);
        if (result_count > result_vals.size() && errcode == err_none) {
          result_vals.clear();
          result_vals.resize(result_count);
          stack.c_eval_fn(&c_count, result_vals.data(), host_params,
              reinterpret_cast<std::int64_t*>(&stat_in),
              reinterpret_cast<std::int64_t*>(&errcode));
          result_count = static_cast<count_type>(c_count);
        }
        return result_count;
      }

      for (count_type istack = 0; istack < stack.stack_point; ++istack) {
        if (stack.entries[istack].ptype == pt_jump_conditional
            || stack.entries[istack].ptype == pt_jump_unconditional) {
          istack = istack + jump_offset(stack.entries[istack], err);
          continue;
        }
        err = err_none;
        stat_in = status_value;
        eval_element(stack.entries[istack], host_params, stat_in, err);
        status_value = status_value | stat_in;
        errcode = errcode | err;
      }

      if (stack.can_accept_maths_domain_errors) {
        errcode = errcode & ~err_maths_domain;
      }

      if (result_vals.empty()) {
        result_vals.resize(stack_point);
      } else if (result_vals.size() < stack_point) {
        result_vals.clear();
        result_vals.resize(stack_point);
      }

      result_count = stack_point;
      pop_vector(stack_point, result_vals.data(), errcode);

      return result_count;
    }
  };

}

#endif
