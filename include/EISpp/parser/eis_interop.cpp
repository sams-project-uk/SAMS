// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "interop_version.inc"
#include "eis_interop_handles.h"
#include "../interop/eis_parser.h"

#if (EIS_C_INTEROP_VERSION == 2)

namespace {

  //! Set the error code that is returned to C
  void report(EIS_ERROR* errcode, eis::error value) {
    if (errcode != nullptr) *errcode = static_cast<EIS_ERROR>(value);
  }

  //! Add to the error code that is returned to C
  void report_or(EIS_ERROR* errcode, eis::error value) {
    if (errcode != nullptr) *errcode = *errcode | static_cast<EIS_ERROR>(value);
  }

  //! Option state from a C integer. Negative means automatic
  eis::option_state to_state(int value) {
    if (value < 0) return eis::option_state::automatic;
    return value != 0 ? eis::option_state::yes : eis::option_state::no;
  }

  eis::option_state yes_no(bool value) {
    return value ? eis::option_state::yes : eis::option_state::no;
  }

  //! Report that a stack and/or its parser were not valid
  void report_bad_stack(EIS_ERROR* errcode, const eis::stack<double>* stack,
      const eis::parser<double>* parser) {
    if (stack == nullptr) report_or(errcode, eis::err_bad_stack);
    if (parser == nullptr) report_or(errcode, eis::err_bad_parser);
  }

  //! Emplacement function that calls a function supplied by C code
  eis::parser_late_bind_functor<double> emplace_bound(parser_emplace_fn fn) {
    return [fn](std::string_view orig_string, eis::count_type nparams,
        const double* params, void* host_params,
        eis::parser<double>* calling_parser, eis::stack<double>& stack_out,
        eis::status& status_code, eis::error& error_code) {
      if (eis_c_interop::find_parser_id(calling_parser) < 1) {
        error_code = eis::err_interop;
        return;
      }

      std::string name(orig_string);
      std::vector<double> values(params, params + nparams);
      int stack_id = -1;

      fn(name.data(), static_cast<int>(nparams), values.data(), host_params,
          &stack_id, reinterpret_cast<EIS_STATUS*>(&status_code),
          reinterpret_cast<EIS_ERROR*>(&error_code));

      const eis::stack<double>* source = eis_c_interop::get_stack(stack_id);
      if (source == nullptr) return;
      eis::copy_stack(*source, stack_out);
      if ((status_code & eis::status_delete_stack) != 0) {
        eis_c_interop::release_stack(stack_id);
      }
    };
  }

  //! Functor that calls a function supplied by C code with bound data
  eis::functor<double> functor_bound(parser_functor_fn fn, void* bound_data) {
    return [fn, bound_data](eis::count_type nparams, const double* values,
        void* host_params, eis::parser<double>*, eis::status& status_code,
        eis::error& error_code) -> double {
      if (fn == nullptr) return 0.0;
      return fn(static_cast<int>(nparams), const_cast<double*>(values),
          host_params, bound_data, reinterpret_cast<EIS_STATUS*>(&status_code),
          reinterpret_cast<EIS_ERROR*>(&error_code));
    };
  }

}

extern "C" {

  void eis_create_parser_params_init(eis_create_parser_params* params) {
    params->should_simplify = 1;
    params->should_minify = 0;
    params->allow_text = 0;
    params->no_import = 0;
    params->maths_fail_quickly = 0;
    params->errhandler = 0;
  }

  void eis_create_stack_params_init(eis_create_stack_params* params) {
    params->simplify = -1;
    params->minify = -1;
    params->allow_text = -1;
    params->filename = nullptr;
    params->line_number = -1;
    params->char_offset = 0;
  }

  void eis_append_stack_params_init(eis_append_stack_params* params) {
    params->simplify = -1;
    params->minify = -1;
    params->allow_text = -1;
    params->filename = nullptr;
    params->line_number = -1;
    params->char_offset = 0;
  }

  void eis_populate_stack_params_init(eis_populate_stack_params* params) {
    params->simplify = -1;
    params->minify = -1;
    params->allow_text = -1;
    params->filename = nullptr;
    params->line_number = -1;
    params->char_offset = 0;
  }

  void eis_empty_stack_params_init(eis_empty_stack_params* params) {
    params->dummy = 0;
  }

  void eis_bind_result_to_stack_params_init(
      eis_bind_result_to_stack_params* params) {
    params->filename = nullptr;
    params->line_number = -1;
    params->char_offset = 0;
    params->cap_bits = 0;
    params->append = 0;
  }

  void eis_evaluate_stack_params_init(eis_evaluate_stack_params* params) {
    params->dummy = 0;
  }

  void eis_add_function_params_init(eis_add_function_params* params) {
    params->cap_bits = 0;
    params->can_simplify = 1;
    params->defer = 0;
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
    params->text_params = 0;
    params->is_idempotent = 0;
    params->text_param_mask = 0;
  }

  void eis_add_emplaced_function_params_init(
      eis_add_emplaced_function_params* params) {
    params->description = nullptr;
    params->hidden = 0;
  }

  void eis_emplace_stack_params_init(eis_emplace_stack_params* params) {
    params->stack_id_out = -1;
  }

  void eis_add_functor_params_init(eis_add_functor_params* params) {
    params->cap_bits = 0;
    params->can_simplify = 1;
    params->defer = 0;
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
    params->text_params = 0;
    params->is_idempotent = 0;
    params->text_param_mask = 0;
  }

  void eis_add_variable_params_init(eis_add_variable_params* params) {
    params->cap_bits = 0;
    params->can_simplify = 0;
    params->defer = 0;
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
  }

  void eis_add_stack_variable_params_init(
      eis_add_stack_variable_params* params) {
    params->defer = 0;
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
  }

  void eis_add_stack_expression_variable_params_init(
      eis_add_stack_expression_variable_params* params) {
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
  }

  void eis_add_pointer_variable_params_init(
      eis_add_pointer_variable_params* params) {
    params->cap_bits = 0;
    params->can_simplify = 0;
    params->defer = 0;
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
  }

  void eis_add_constant_params_init(eis_add_constant_params* params) {
    params->cap_bits = 0;
    params->can_simplify = 1;
    params->defer = 0;
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
  }

  void eis_add_deferred_constant_params_init(
      eis_add_deferred_constant_params* params) {
    params->cap_bits = 0;
    params->can_simplify = 1;
    params->global = 0;
    params->description = nullptr;
    params->hidden = 0;
  }

  void eis_parser_get_error_count_params_init(
      eis_parser_get_error_count_params* params) {
    params->dummy = 0;
  }

  void eis_parser_get_error_report_params_init(
      eis_parser_get_error_report_params* params) {
    params->dummy = 0;
  }

  void eis_parser_flush_errors_params_init(
      eis_parser_flush_errors_params* params) {
    params->dummy = 0;
  }

  void eis_stack_inc_ref_params_init(eis_stack_inc_ref_params* params) {
    params->dummy = 0;
  }

  void eis_parser_inc_ref_params_init(eis_parser_inc_ref_params* params) {
    params->dummy = 0;
  }

  void eis_stack_dec_ref_params_init(eis_stack_dec_ref_params* params) {
    params->dummy = 0;
  }

  void eis_parser_dec_ref_params_init(eis_parser_dec_ref_params* params) {
    params->dummy = 0;
  }

  void eis_copy_stack_params_init(eis_copy_stack_params* params) {
    params->dummy = 0;
  }

  void eis_visualise_stack_params_init(eis_visualise_stack_params* params) {
    params->formatstr = nullptr;
  }

  void eis_get_infix_stack_params_init(eis_get_infix_stack_params* params) {
    params->formatstr = nullptr;
  }

  void eis_get_deriv_stack_params_init(eis_get_deriv_stack_params* params) {
    params->dummy = 0;
  }

  void eis_set_symbol_derivative_params_init(
      eis_set_symbol_derivative_params* params) {
    params->global = 0;
  }

  void eis_lock_symbol_derivative_params_init(
      eis_lock_symbol_derivative_params* params) {
    params->global = 0;
  }

  int eis_create_parser(int physics, EIS_ERROR* errcode,
      eis_create_parser_params* params) {
    eis_create_parser_params defaults;
    eis_create_parser_params_init(&defaults);
    const eis_create_parser_params& p = params != nullptr ? *params : defaults;
    eis::error err = eis::err_none;

    eis::parser<double>* parser = new eis::parser<double>;
    const int parser_id = eis_c_interop::add_parser(parser, true);
    parser->init(err, {.should_simplify = yes_no(p.should_simplify != 0),
        .should_minify = yes_no(p.should_minify != 0),
        .allow_text = yes_no(p.allow_text != 0),
        .maths_fail_quickly = yes_no(p.maths_fail_quickly != 0),
        .no_import = p.no_import != 0, .physics = physics});
    report(errcode, err);

    return parser_id;
  }

  int eis_create_stack(int parser_id, char* expression, EIS_BITMASK* cap_bits,
      EIS_ERROR* errcode, eis_create_stack_params* params) {
    eis_create_stack_params defaults;
    eis_create_stack_params_init(&defaults);
    const eis_create_stack_params& p = params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return -1;

    eis::stack<double>* stack = new eis::stack<double>;
    const int stack_id = eis_c_interop::add_stack(stack, parser, true);
    if (expression != nullptr) {
      eis::error err = eis::err_none;
      parser->tokenize(expression, *stack, err,
          {.simplify = to_state(p.simplify), .minify = to_state(p.minify),
          .filename = eis_c_interop::to_string(p.filename),
          .line_number = p.line_number, .char_offset = p.char_offset,
          .allow_text = to_state(p.allow_text)});
      if (cap_bits != nullptr) {
        *cap_bits = static_cast<EIS_BITMASK>(stack->cap_bits);
      }
      report_or(errcode, err);
    }

    return stack_id;
  }

  void eis_append_stack(int stack_id, char* expression, EIS_BITMASK* cap_bits,
      EIS_ERROR* errcode, eis_append_stack_params* params) {
    eis_append_stack_params defaults;
    eis_append_stack_params_init(&defaults);
    const eis_append_stack_params& p = params != nullptr ? *params : defaults;
    eis::parser<double>* parser = nullptr;

    report(errcode, eis::err_none);

    eis::stack<double>* stack = eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) {
      report_bad_stack(errcode, stack, parser);
      return;
    }

    eis::error err = eis::err_none;
    parser->tokenize(eis_c_interop::to_string(expression), *stack, err,
        {.simplify = to_state(p.simplify), .minify = to_state(p.minify),
        .filename = eis_c_interop::to_string(p.filename),
        .line_number = p.line_number, .char_offset = p.char_offset,
        .append = true, .allow_text = to_state(p.allow_text)});
    if (cap_bits != nullptr) {
      *cap_bits = static_cast<EIS_BITMASK>(stack->cap_bits);
    }
    report_or(errcode, err);
  }

  void eis_populate_stack(int stack_id, char* expression,
      EIS_BITMASK* cap_bits, EIS_ERROR* errcode,
      eis_populate_stack_params* params) {
    eis_populate_stack_params defaults;
    eis_populate_stack_params_init(&defaults);
    const eis_populate_stack_params& p =
        params != nullptr ? *params : defaults;
    eis::parser<double>* parser = nullptr;

    report(errcode, eis::err_none);

    eis::stack<double>* stack = eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) {
      report_bad_stack(errcode, stack, parser);
      return;
    }

    eis::error err = eis::err_none;
    parser->tokenize(eis_c_interop::to_string(expression), *stack, err,
        {.simplify = to_state(p.simplify), .minify = to_state(p.minify),
        .filename = eis_c_interop::to_string(p.filename),
        .line_number = p.line_number, .char_offset = p.char_offset,
        .append = false, .allow_text = to_state(p.allow_text)});
    if (cap_bits != nullptr) {
      *cap_bits = static_cast<EIS_BITMASK>(stack->cap_bits);
    }
    report_or(errcode, err);
  }

  void eis_empty_stack(int stack_id, EIS_ERROR* errcode,
      eis_empty_stack_params*) {
    eis::parser<double>* parser = nullptr;

    report(errcode, eis::err_none);

    eis::stack<double>* stack = eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) {
      report_bad_stack(errcode, stack, parser);
      return;
    }

    parser->blank_stack(*stack);
  }

  void eis_bind_result_to_stack(int stack_id, parser_result_fn fn,
      EIS_ERROR* errcode, eis_bind_result_to_stack_params* params) {
    eis_bind_result_to_stack_params defaults;
    eis_bind_result_to_stack_params_init(&defaults);
    const eis_bind_result_to_stack_params& p =
        params != nullptr ? *params : defaults;
    eis::parser<double>* parser = nullptr;

    report(errcode, eis::err_none);

    eis::stack<double>* stack = eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) {
      report_bad_stack(errcode, stack, parser);
      return;
    }

    eis::error err = eis::err_none;
    parser->set_result_function(fn, *stack, err,
        {.filename = eis_c_interop::to_string(p.filename),
        .line_number = p.line_number, .char_offset = p.char_offset,
        .cap_bits = static_cast<eis::bitmask>(p.cap_bits),
        .append = p.append != 0});
    report_or(errcode, err);
  }

  int eis_evaluate_stack(int stack_id, int res_len, EIS_NUM* result,
      void* host_params, EIS_ERROR* errcode, eis_evaluate_stack_params*) {
    eis::parser<double>* parser = nullptr;
    std::vector<double> values;
    eis::error err = eis::err_none;

    report(errcode, eis::err_none);

    eis::stack<double>* stack = eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) {
      report_bad_stack(errcode, stack, parser);
      return -1;
    }

    const eis::count_type count = parser->evaluate(*stack, values, err,
        {.host_params = host_params});
    report_or(errcode, err);
    if (err != eis::err_none && count == 0) return -1;

    const eis::count_type capacity =
        static_cast<eis::count_type>(std::max(res_len, 0));
    if (capacity < count) report_or(errcode, eis::err_extra_results);
    if (result != nullptr) {
      std::copy_n(values.begin(), std::min(count, capacity), result);
    }

    return static_cast<int>(count);
  }

  void eis_add_function(int parser_id, char* name, parser_eval_fn fn,
      int expected_params, EIS_ERROR* errcode, eis_add_function_params* params) {
    eis_add_function_params defaults;
    eis_add_function_params_init(&defaults);
    const eis_add_function_params& p = params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;

    eis::error err = eis::err_none;
    const eis::parser_function_options options{
        .cap_bits = static_cast<eis::bitmask>(p.cap_bits),
        .expected_params = expected_params,
        .can_simplify = p.can_simplify != 0,
        .defer = p.defer != 0 || fn == nullptr, .global = p.global != 0,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0, .text_params = p.text_params != 0,
        .text_param_mask = static_cast<eis::bitmask>(p.text_param_mask),
        .is_idempotent = p.is_idempotent != 0};
    if (fn != nullptr) {
      parser->add_function(name, fn, err, options);
    } else {
      parser->add_function(name, err, options);
    }
    report_or(errcode, err);
  }

  void eis_add_emplaced_function(int parser_id, char* name,
      parser_emplace_fn fn, int expected_params, EIS_ERROR* errcode,
      eis_add_emplaced_function_params* params) {
    eis_add_emplaced_function_params defaults;
    eis_add_emplaced_function_params_init(&defaults);
    const eis_add_emplaced_function_params& p =
        params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;
    if (fn == nullptr) {
      report_or(errcode, eis::err_bad_value);
      return;
    }

    eis::error err = eis::err_none;
    parser->add_emplaced_functor(name, emplace_bound(fn), err,
        {.expected_params = expected_params,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0});
    report_or(errcode, err);
  }

  void eis_emplace_stack(int stack_id, void* host_params, EIS_ERROR* errcode,
      eis_emplace_stack_params* params) {
    eis_emplace_stack_params defaults;
    eis_emplace_stack_params_init(&defaults);
    const eis_emplace_stack_params& p = params != nullptr ? *params : defaults;
    eis::parser<double>* parser = nullptr;

    report(errcode, eis::err_none);

    eis::stack<double>* stack = eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) {
      report_bad_stack(errcode, stack, parser);
      return;
    }

    eis::error err = eis::err_none;
    parser->emplace(*stack, err, {.host_params = host_params,
        .destination = eis_c_interop::get_stack(p.stack_id_out)});
    report_or(errcode, err);
  }

  void eis_add_functor(int parser_id, char* name, parser_functor_fn fn,
      void* bound_data, int expected_params, EIS_ERROR* errcode,
      eis_add_functor_params* params) {
    eis_add_functor_params defaults;
    eis_add_functor_params_init(&defaults);
    const eis_add_functor_params& p = params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;

    eis::error err = eis::err_none;
    parser->add_functor(name, functor_bound(fn, bound_data), err,
        {.cap_bits = static_cast<eis::bitmask>(p.cap_bits),
        .expected_params = expected_params,
        .can_simplify = p.can_simplify != 0,
        .defer = p.defer != 0 || fn == nullptr, .global = p.global != 0,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0, .text_params = p.text_params != 0,
        .text_param_mask = static_cast<eis::bitmask>(p.text_param_mask),
        .is_idempotent = p.is_idempotent != 0});
    report_or(errcode, err);
  }

  void eis_add_variable(int parser_id, char* name, parser_eval_fn fn,
      EIS_ERROR* errcode, eis_add_variable_params* params) {
    eis_add_variable_params defaults;
    eis_add_variable_params_init(&defaults);
    const eis_add_variable_params& p = params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;

    eis::error err = eis::err_none;
    const eis::parser_variable_options options{
        .cap_bits = static_cast<eis::bitmask>(p.cap_bits),
        .can_simplify = p.can_simplify != 0,
        .defer = p.defer != 0 || fn == nullptr, .global = p.global != 0,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0};
    if (fn != nullptr) {
      parser->add_variable(name, fn, err, options);
    } else {
      parser->add_variable(name, err, options);
    }
    report_or(errcode, err);
  }

  void eis_add_stack_variable(int parser_id, char* name, int stack_id,
      EIS_ERROR* errcode, eis_add_stack_variable_params* params) {
    eis_add_stack_variable_params defaults;
    eis_add_stack_variable_params_init(&defaults);
    const eis_add_stack_variable_params& p =
        params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;
    const eis::stack<double>* stack = eis_c_interop::get_stack(stack_id);

    eis::error err = eis::err_none;
    if (stack != nullptr) {
      parser->add_stack_variable(name, *stack, err,
          {.defer = p.defer != 0, .global = p.global != 0,
          .description = eis_c_interop::to_string(p.description),
          .hidden = p.hidden != 0});
    } else {
      parser->add_stack_variable(name, err,
          {.global = p.global != 0,
          .description = eis_c_interop::to_string(p.description),
          .hidden = p.hidden != 0});
    }
    report_or(errcode, err);
  }

  void eis_add_stack_expression_variable(int parser_id, char* name,
      char* stack_text, EIS_ERROR* errcode,
      eis_add_stack_expression_variable_params* params) {
    eis_add_stack_expression_variable_params defaults;
    eis_add_stack_expression_variable_params_init(&defaults);
    const eis_add_stack_expression_variable_params& p =
        params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;

    eis::error err = eis::err_none;
    const eis::parser_stack_variable_options options{.global = p.global != 0,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0};
    if (stack_text != nullptr) {
      parser->add_stack_variable(name, std::string(stack_text), err, options);
    } else {
      parser->add_stack_variable(name, err, options);
    }
    report_or(errcode, err);
  }

  void eis_add_pointer_variable(int parser_id, char* name, void* ptr,
      int ptr_is_integer, int ptr_is_64_bit, EIS_ERROR* errcode,
      eis_add_pointer_variable_params* params) {
    eis_add_pointer_variable_params defaults;
    eis_add_pointer_variable_params_init(&defaults);
    const eis_add_pointer_variable_params& p =
        params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;

    eis::error err = eis::err_none;
    const eis::parser_variable_options options{
        .cap_bits = static_cast<eis::bitmask>(p.cap_bits),
        .can_simplify = p.can_simplify != 0,
        .defer = p.defer != 0 || ptr == nullptr, .global = p.global != 0,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0};
    if (ptr_is_integer != 0) {
      if (ptr_is_64_bit != 0) {
        parser->add_pointer_variable(name, static_cast<std::int64_t*>(ptr),
            err, options);
      } else {
        parser->add_pointer_variable(name, static_cast<std::int32_t*>(ptr),
            err, options);
      }
    } else {
      if (ptr_is_64_bit != 0) {
        parser->add_pointer_variable(name, static_cast<double*>(ptr), err,
            options);
      } else {
        parser->add_pointer_variable(name, static_cast<float*>(ptr), err,
            options);
      }
    }
    report_or(errcode, err);
  }

  void eis_add_constant(int parser_id, char* name, EIS_NUM value,
      EIS_ERROR* errcode, eis_add_constant_params* params) {
    eis_add_constant_params defaults;
    eis_add_constant_params_init(&defaults);
    const eis_add_constant_params& p = params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;

    eis::error err = eis::err_none;
    parser->add_constant(name, value, err,
        {.cap_bits = static_cast<eis::bitmask>(p.cap_bits),
        .can_simplify = p.can_simplify != 0, .defer = p.defer != 0,
        .global = p.global != 0,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0});
    report_or(errcode, err);
  }

  void eis_add_deferred_constant(int parser_id, char* name,
      EIS_ERROR* errcode, eis_add_deferred_constant_params* params) {
    eis_add_deferred_constant_params defaults;
    eis_add_deferred_constant_params_init(&defaults);
    const eis_add_deferred_constant_params& p =
        params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) return;

    eis::error err = eis::err_none;
    parser->add_constant(name, err,
        {.cap_bits = static_cast<eis::bitmask>(p.cap_bits),
        .can_simplify = p.can_simplify != 0, .global = p.global != 0,
        .description = eis_c_interop::to_string(p.description),
        .hidden = p.hidden != 0});
    report_or(errcode, err);
  }

  int eis_parser_get_error_count(int parser_id,
      eis_parser_get_error_count_params*) {
    const eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);

    if (parser == nullptr) return 0;
    return static_cast<int>(parser->get_error_count());
  }

  int eis_parser_get_error_report(int parser_id, int error_id, int buflen,
      char* string_out, eis_parser_get_error_report_params*) {
    const eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);

    if (parser == nullptr) return 0;
    if (error_id < 1
        || static_cast<eis::count_type>(error_id) > parser->get_error_count()) {
      return 0;
    }

    const std::string report_text = parser->get_error_report(
        static_cast<eis::count_type>(error_id) - 1);
    eis_c_interop::copy_to_buffer(report_text, buflen, string_out);
    return static_cast<int>(report_text.size()) + 1;
  }

  void eis_parser_flush_errors(int parser_id, eis_parser_flush_errors_params*) {
    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);

    if (parser == nullptr) return;
    parser->flush_errors();
  }

  void eis_stack_inc_ref(int stack_id, eis_stack_inc_ref_params*) {
    eis_c_interop::inc_stack_ref(stack_id);
  }

  void eis_stack_dec_ref(int stack_id, eis_stack_dec_ref_params*) {
    eis_c_interop::dec_stack_ref(stack_id);
  }

  void eis_parser_inc_ref(int parser_id, eis_parser_inc_ref_params*) {
    eis_c_interop::inc_parser_ref(parser_id);
  }

  void eis_parser_dec_ref(int parser_id, eis_parser_dec_ref_params*) {
    eis_c_interop::dec_parser_ref(parser_id);
  }

  int eis_copy_stack(int stack_id, eis_copy_stack_params*) {
    return eis_c_interop::copy_stack(stack_id);
  }

  int eis_visualise_stack(int stack_id, int text_length, char* text,
      eis_visualise_stack_params* params) {
    eis_visualise_stack_params defaults;
    eis_visualise_stack_params_init(&defaults);
    const eis_visualise_stack_params& p =
        params != nullptr ? *params : defaults;
    eis::parser<double>* parser = nullptr;

    const eis::stack<double>* stack =
        eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) return -1;

    const std::string dot = parser->visualise_stack(*stack,
        {.nformat = eis_c_interop::to_string(p.formatstr)});
    eis_c_interop::copy_to_buffer(dot, text_length, text);
    return static_cast<int>(dot.size());
  }

  int eis_get_infix_stack(int stack_id, int text_length, char* text,
      eis_get_infix_stack_params* params) {
    eis_get_infix_stack_params defaults;
    eis_get_infix_stack_params_init(&defaults);
    const eis_get_infix_stack_params& p =
        params != nullptr ? *params : defaults;
    eis::parser<double>* parser = nullptr;

    const eis::stack<double>* stack =
        eis_c_interop::get_stack(stack_id, &parser);
    if (stack == nullptr || parser == nullptr) return -1;

    const std::string infix = parser->get_infix(*stack,
        {.nformat = eis_c_interop::to_string(p.formatstr)});
    eis_c_interop::copy_to_buffer(infix, text_length, text);
    return static_cast<int>(infix.size());
  }

  void eis_get_deriv_stack(int stack_id, int output_id, char* wrt,
      EIS_ERROR* errcode, eis_get_deriv_stack_params*) {
    eis::parser<double>* parser = nullptr;

    report(errcode, eis::err_none);

    eis::stack<double>* stack = eis_c_interop::get_stack(stack_id, &parser);
    eis::stack<double>* output = eis_c_interop::get_stack(output_id);
    if (stack != nullptr && output != nullptr && parser != nullptr
        && wrt != nullptr) {
      eis::error err = eis::err_none;
      parser->get_deriv_stack(*stack, *output, wrt, err);
      report_or(errcode, err);
    } else {
      if (stack == nullptr || output == nullptr) {
        report_or(errcode, eis::err_bad_stack);
      }
      if (parser == nullptr) report_or(errcode, eis::err_bad_parser);
      if (wrt == nullptr) report_or(errcode, eis::err_bad_value);
    }
  }

  void eis_set_symbol_derivative(int parser_id, char* symbol, char* deriv,
      char* wrt, EIS_ERROR* errcode, eis_set_symbol_derivative_params* params) {
    eis_set_symbol_derivative_params defaults;
    eis_set_symbol_derivative_params_init(&defaults);
    const eis_set_symbol_derivative_params& p =
        params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) {
      report_or(errcode, eis::err_bad_parser);
      return;
    }
    if (symbol == nullptr || deriv == nullptr) {
      report_or(errcode, eis::err_bad_value);
      return;
    }

    eis::error err = eis::err_none;
    parser->set_symbol_derivative(symbol, deriv, err,
        {.wrt = eis_c_interop::to_string(wrt), .global = p.global != 0});
    report_or(errcode, err);
  }

  void eis_lock_symbol_derivative(int parser_id, char* symbol, char* wrt,
      EIS_ERROR* errcode, eis_lock_symbol_derivative_params* params) {
    eis_lock_symbol_derivative_params defaults;
    eis_lock_symbol_derivative_params_init(&defaults);
    const eis_lock_symbol_derivative_params& p =
        params != nullptr ? *params : defaults;

    report(errcode, eis::err_none);

    eis::parser<double>* parser = eis_c_interop::get_parser(parser_id);
    if (parser == nullptr) {
      report_or(errcode, eis::err_bad_parser);
      return;
    }
    if (symbol == nullptr) {
      report_or(errcode, eis::err_bad_value);
      return;
    }

    eis::error err = eis::err_none;
    parser->lock_symbol_derivative(symbol, err,
        {.wrt = eis_c_interop::to_string(wrt), .global = p.global != 0});
    report_or(errcode, err);
  }

}

#else
#error Unsupported EIS_C_INTEROP_VERSION
#endif
