// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_PARSER_H
#define EIS_PARSER_H

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/eis_constants.h"
#include "../common/eis_error.h"
#include "../common/eis_optional.h"
#include "../common/eis_string_store.h"
#include "../common/eis_utils.h"
#include "eis_core_functions.h"
#include "eis_eval_stack.h"
#include "eis_parser_constants.h"
#include "eis_raw_parser.h"
#include "eis_registry.h"
#include "eis_stack.h"
#include "eis_tree.h"

namespace eis {

  //! Character is part of a numerical value
  inline constexpr typecode char_numeric = 1;
  //! Character is part of an alphanumeric value
  inline constexpr typecode char_alpha = 2;
  //! character is a delimiter
  inline constexpr typecode char_delimiter = 3;
  //! character is a space
  inline constexpr typecode char_space = 4;
  //! character is an opcode/special character
  inline constexpr typecode char_opcode = 5;
  //! character is of unknown type
  inline constexpr typecode char_unknown = 1024;

  //! State of an optional logical value. Automatic means use the value that
  //! is currently set for the parser
  enum class option_state { automatic, yes, no };

  namespace detail {

    //! Function to determine the type of a character
    //! Determines whether a character is a part of a number, a string, an
    //! operator etc.
    inline typecode char_type(char chr) {
      typecode char_type_value = char_unknown;

      //Non printable characters are spaces
      if (static_cast<unsigned char>(chr) <= 32) {
        char_type_value = char_space;
      } else if ((chr >= '0' && chr <= '9') || chr == '.') {
        char_type_value = char_numeric;
      } else if ((chr >= 'A' && chr <= 'Z') || (chr >= 'a' && chr <= 'z')
          || chr == '_' || chr == '"') {
        char_type_value = char_alpha;
      } else if (chr == '(' || chr == ')' || chr == ',') {
        char_type_value = char_delimiter;
      // 92 is the ASCII code for backslash
      } else if (chr == '+' || chr == '-' || chr == 92 || chr == '/'
          || chr == '*' || chr == '^' || chr == '>' || chr == '<'
          || chr == '=' || chr == '!') {
        char_type_value = char_opcode;
      }

      return char_type_value;
    }

    //! Function to identify the type of a parenthesis. Currently only
    //! identifies left and right round brackets. May expand in future
    //! Returns 0 if the string is not a bracket
    inline typecode as_parenthesis(const std::string& name) {
      typecode as_parenthesis_value = 0;

      if (eis::strcmp(name, "(")) {
        as_parenthesis_value = paren_left_bracket;
      } else if (eis::strcmp(name, ")")) {
        as_parenthesis_value = paren_right_bracket;
      }

      return as_parenthesis_value;
    }

    //! Value of an option_state, using "fallback" if it is automatic
    inline bool resolve(option_state state, bool fallback) {
      if (state == option_state::automatic) return fallback;
      return state == option_state::yes;
    }

  }

  //! Optional parameters to initialising a parser
  struct parser_init_options {
    //! Should the parser autosimplify. Automatic means unchanged
    option_state should_simplify = option_state::automatic;
    //! Should the parser autominify. Automatic means unchanged
    option_state should_minify = option_state::automatic;
    //! Should the parser allow text blocks. Automatic means unchanged
    option_state allow_text = option_state::automatic;
    //! Should simplification report maths domain errors instead of deferring
    //! them to evaluation. Automatic means unchanged
    option_state maths_fail_quickly = option_state::automatic;
    //! Should the parser not import default namespaces
    bool no_import = false;
    //! What (if any) physics module should be loaded. -1 means unchanged
    typecode physics = -1;
    //! Externally supplied error handler. If specified and null then an error
    //! handler is created unless there is already one in place
    optional<error_handler*> err_handler;
  };

  //! Optional parameters to operations that can use host code parameters
  struct host_options {
    //! Host code specified parameters
    void* host_params = nullptr;
  };

  //! Optional parameters to simplifying a stack
  struct simplify_options {
    //! Host code specified parameters. Can be used by stack elements to
    //! determine if they can be simplified or not
    void* host_params = nullptr;
    //! Report maths domain errors during simplification. Automatic means use
    //! the setting for the parser
    option_state maths_fail_quickly = option_state::automatic;
  };

  //! Optional parameters to tokenizing an expression
  struct tokenize_options {
    //! Should the stack be simplified after generation
    option_state simplify = option_state::automatic;
    //! Should the stack be minified after generation
    option_state minify = option_state::automatic;
    //! The filename of the file containing the expression. Empty means that
    //! no filename is given
    std::string filename;
    //! The line number of the expression within the file. -1 means that no
    //! line number is given
    offset_type line_number = -1;
    //! The character offset of the expression within the line
    offset_type char_offset = 0;
    //! Whether the newly parsed value should be appended to the values in the
    //! stack
    bool append = false;
    //! Should the parser allow text blocks
    option_state allow_text = option_state::automatic;
    //! Should the parser allow function parameters
    bool allow_params = false;
    //! Should the parser allow function derivative parameters
    bool allow_dparams = false;
  };

  //! Optional parameters to setting a result function
  struct set_result_function_options {
    std::string filename;
    offset_type line_number = -1;
    offset_type char_offset = 0;
    bitmask cap_bits = 0;
    //! Should the eval function be appended to the stack
    bool append = false;
  };

  //! Optional parameters to tokenizing a number
  struct tokenize_number_options {
    bool append = false;
  };

  //! Optional parameters to evaluating a string
  struct evaluate_string_options {
    void* host_params = nullptr;
    option_state simplify = option_state::automatic;
    option_state minify = option_state::automatic;
    std::string filename;
    offset_type line_number = -1;
    offset_type char_offset = 0;
    option_state allow_text = option_state::automatic;
  };

  //! Results from evaluating a string
  struct parser_results {
    //! Number of results returned by the evaluation
    count_type count = 0;
    //! Capability bits of the parsed stack
    bitmask cap_bits = 0;
  };

  //! Optional parameters to emplacing a stack
  template <typename Num>
  struct emplace_options {
    void* host_params = nullptr;
    //! Destination for final emplaced stack. Null means build in the stack
    eis::stack<Num>* destination = nullptr;
  };

  //! Optional parameters to expanding functions inline
  template <typename Num>
  struct inline_expand_options {
    const std::vector<tree_item<Num>>* dparams = nullptr;
    const tree_item<Num>* func = nullptr;
  };

  //! Optional parameters to loading a block
  struct load_block_options {
    //! Can this be a unary operator. Automatic means determine from the
    //! previous block
    option_state unary = option_state::automatic;
    bool allow_params = false;
    bool allow_dparams = false;
    //! Does this block flag itself as allowing maths domain errors
    bool* can_accept_maths_domain_errors = nullptr;
  };

  //! Optional parameters to getting information about a symbol
  struct get_symbol_info_options {
    bool unary = false;
  };

  //! Information about a symbol
  struct symbol_info {
    //! Does the symbol exist in the symbol list
    bool exists = false;
    typecode symbol_type = pt_bad;
    bitmask cap_bits = 0;
    bool deferred = false;
    std::string description;
    bool is_hidden = false;
    offset_type expected_params = -1;
  };

  //! Optional parameters to operations that can use the global registry
  struct namespace_options {
    bool in_global = false;
  };

  //! Optional parameters to adding a function or functor
  struct parser_function_options {
    bitmask cap_bits = 0;
    //! Number of expected parameters. -1 is variadic
    offset_type expected_params = -1;
    bool can_simplify = true;
    bool defer = false;
    bool global = false;
    std::string description;
    bool hidden = false;
    bool text_params = false;
    //! Parameter positions that must be text. Bit 0 is the first parameter. 0
    //! means that positions are not checked
    bitmask text_param_mask = 0;
    //! Does calling the function a second time with the same parameters in the
    //! same evaluation have no further effect and return the same result.
    //! If it does then repeated calls can be merged
    bool is_idempotent = false;
  };

  //! Optional parameters to adding a functor by pointer
  struct parser_functor_pointer_options {
    bitmask cap_bits = 0;
    offset_type expected_params = -1;
    bool can_simplify = true;
    bool defer = false;
    bool global = false;
    std::string description;
    bool hidden = false;
    //! Should the parser take ownership of the pointer
    bool owns = false;
    bool text_params = false;
    //! Parameter positions that must be text. Bit 0 is the first parameter. 0
    //! means that positions are not checked
    bitmask text_param_mask = 0;
    //! Does calling the functor a second time with the same parameters in the
    //! same evaluation have no further effect and return the same result.
    //! If it does then repeated calls can be merged
    bool is_idempotent = false;
  };

  //! Optional parameters to adding a variable
  struct parser_variable_options {
    bitmask cap_bits = 0;
    bool can_simplify = false;
    bool defer = false;
    bool global = false;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding a constant
  struct parser_constant_options {
    bitmask cap_bits = 0;
    bool can_simplify = true;
    bool defer = false;
    bool global = false;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding a stack variable
  struct parser_stack_variable_options {
    bool defer = false;
    bool global = false;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding a stack function
  struct parser_stack_function_options {
    bool global = false;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding an emplaced function or variable
  struct parser_emplaced_options {
    //! Number of expected parameters. -1 is variadic
    offset_type expected_params = -1;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to setting or locking a derivative
  struct symbol_derivative_options {
    //! What to take the derivative with respect to. Empty means that the
    //! derivative is with respect to any variable
    std::string wrt;
    //! Is the symbol in the global registry
    bool global = false;
    bool unary = false;
  };

  //! Optional parameters to combining stacks
  struct combine_stacks_options {
    bool append = false;
    //! String for function to use to combine the stacks
    std::string function_str;
  };

  //! Optional parameters to producing a textual representation of a stack
  struct structure_options {
    //! Title of the document. Empty means no title
    std::string title;
    typecode function_show_name = fsn_auto;
  };

  //! Information about a single error
  struct error_info {
    std::string error_cause;
    offset_type error_cause_location = -1;
    std::string error_phase;
    std::string error_type;
    std::string error_filename;
    offset_type error_line_number = -1;
  };

  //! Type representing a maths parser
  template <typename Num>
  class parser {

    eis::registry<Num> registry;
    eval_stack<Num> evaluator;
    string_store string_param_store;
    error_handler* err_handler = nullptr;
    bool owns_err_handler = false;

    eis::stack<Num>* zero_stack = nullptr;

    //State information used during tokenization
    typecode last_block_type = pt_null;
    offset_type last_block_value = 0;
    offset_type last_charindex = 1;
    std::string last_block_text;
    count_type last_block_item_count = 1;

    bool is_init = false;
    bool should_simplify = true;
    bool should_minify = false;
    bool allow_text = false;
    bool maths_fail_quickly = false;
    eis::stack<Num> stack, brackets;
    typecode physics_units = physics_none;

  public:

    parser() {
      evaluator.host_parser = this;
    }
    parser(const parser&) = delete;
    parser& operator=(const parser&) = delete;

    ~parser() {
      if (err_handler != nullptr && owns_err_handler) delete err_handler;
      delete zero_stack;
    }

    //! Registry holding the symbols that are available to all parsers
    inline static eis::registry<Num> global_registry;

    //! Has the global registry been set up
    inline static bool global_setup = false;

    //! Calculate a stack that contains the derivative of another stack
    //! @param[inout] stack - Input stack containing function that you want the
    //! derivative of
    //! @param[inout] output - Output stack filled with the derivative of stack
    //! @param[in] wrt - String saying what the derivative is with respect to
    //! @param[inout] errcode - Error code
    void get_deriv_stack(eis::stack<Num>& stack, eis::stack<Num>& output,
        const std::string& wrt, error& errcode) {
      eis::stack<Num> intermediate;
      count_type sp;

      initialise_stack(output);
      output.owner = this;
      sp = stack.stack_point;
      while (sp > 0) {
        tree_item<Num> root, result;
        build_node(stack, sp, root);
        errcode = err_none;
        gen_deriv_tree(root, result, wrt, output, errcode);
        if (errcode != err_none) return;
        initialise_stack(intermediate);
        tree_to_stack(result, intermediate);
        prepend_stack(output, intermediate);
        deallocate_stack(intermediate);
      }

      if (this->should_simplify) simplify(output, errcode);
    }

    //! Core function to produce a tree_item corresponding to the
    //! derivative of another tree_item. Operates recursively
    //! @param[inout] tree - Tree item of source data
    //! @param[inout] deriv_tree - Tree item to hold derivative
    //! @param[in] wrt - String holding variable derivative is with respect to
    //! @param[inout] output - Stack that will hold the derivative when
    //! deriv_tree is converted back to a stack. Used to set stack parameters
    //! @param[inout] errcode - Error code for the operation
    void gen_deriv_tree(tree_item<Num>& tree, tree_item<Num>& deriv_tree,
        const std::string& wrt, eis::stack<Num>& output, error& errcode) {
      count_type sp;
      eis::stack<Num>* stack_res;
      std::vector<tree_item<Num>> deriv_nodes;
      bool locked = false;

      if (tree.co_value.registry_type == registry_local) {
        stack_res = this->registry.get_symbol_derivative_value(
            tree.co_value.deriv_values, errcode, wrt, locked);
      } else {
        stack_res = global_registry.get_symbol_derivative_value(
            tree.co_value.deriv_values, errcode, wrt, locked);
      }

      if (stack_res == nullptr) {
        if (locked) {
          errcode = err_no_deriv;
          add_error_options err_options;
          err_options.errstring = tree.co_value.text;
          err_options.charindex = tree.co_value.charindex;
          err_handler->add_error(err_parser, errcode, err_options);
          return;
        }
        stack_res = zero_stack;
      }

      output.cap_bits = output.cap_bits | stack_res->cap_bits;
      output.has_deferred = output.has_deferred || stack_res->has_deferred;
      output.has_emplaced = output.has_emplaced || stack_res->has_emplaced;
      output.can_accept_maths_domain_errors =
          output.can_accept_maths_domain_errors
          || stack_res->can_accept_maths_domain_errors;

      if (tree.nodes_associated) {
        deriv_nodes.resize(tree.nodes.size());
        for (count_type inode = 0; inode < tree.nodes.size(); ++inode) {
          gen_deriv_tree(tree.nodes[inode], deriv_nodes[inode], wrt, output,
              errcode);
        }

        sp = stack_res->stack_point;
        build_node(*stack_res, sp, deriv_tree);
        inline_function_expand(deriv_tree, tree.nodes, errcode,
            {.dparams = &deriv_nodes, .func = &tree});
      } else {
        sp = stack_res->stack_point;
        build_node(*stack_res, sp, deriv_tree);
      }
    }

    //! Core function to replace all stack functions in a stack
    //! with the stacks that represent them
    //! @param[inout] stack - Stack to operate on
    //! @param[inout] errcode - Error code from operation
    void expand_fn(eis::stack<Num>& stack, error& errcode) {
      eis::stack<Num> intermediate, output;
      count_type sp;

      errcode = err_none;

      sp = stack.stack_point;
      initialise_stack(output);
      while (sp > 0) {
        tree_item<Num> root;
        build_node(stack, sp, root);
        errcode = err_none;
        replace_fn(root, errcode, &output);
        if (errcode != err_none) return;
        initialise_stack(intermediate);
        tree_to_stack(root, intermediate);
        prepend_stack(output, intermediate);
        deallocate_stack(intermediate);
      }

      output.cap_bits = output.cap_bits | stack.cap_bits;
      output.has_deferred = output.has_deferred || stack.has_deferred;
      output.has_emplaced = output.has_emplaced || stack.has_emplaced;
      output.can_accept_maths_domain_errors =
          output.can_accept_maths_domain_errors
          || stack.can_accept_maths_domain_errors;
      output.full_line = stack.full_line;
      output.params = stack.params;
      output.filename = stack.filename;

      output.line_number = stack.line_number;
      output.owner = stack.owner;

      stack = output;
    }

    //! Core function to replace an individual tree_item for a
    //! stack function with the true stack, replacing operands as needed
    //! @param[inout] node - Node of a tree to operate on
    //! @param[inout] errcode - Error code from results
    //! @param[inout] output - Stack that will contain the replaced code. May be
    //! null
    void replace_fn(tree_item<Num>& node, error& errcode,
        eis::stack<Num>* output = nullptr) {
      count_type sp;
      error errcode_l;
      eis::stack<Num> stack;
      tree_item<Num> node2;

      errcode = err_none;

      if (node.nodes_associated) {
        for (count_type inode = 0; inode < node.nodes.size(); ++inode) {
          replace_fn(node.nodes[inode], errcode_l);
          errcode = errcode | errcode_l;
        }
      }

      if (node.value.rtype == pt_stack_function) {
        if (node.co_value.registry_type == registry_local) {
          this->registry.get_stack_function(
              static_cast<count_type>(node.value.value), stack, errcode);
        } else {
          global_registry.get_stack_function(
              static_cast<count_type>(node.value.value), stack, errcode);
        }
        if (output != nullptr) {
          output->cap_bits = output->cap_bits | stack.cap_bits;
          output->has_deferred = output->has_deferred || stack.has_deferred;
          output->has_emplaced = output->has_emplaced || stack.has_emplaced;
          output->can_accept_maths_domain_errors =
              output->can_accept_maths_domain_errors
              || stack.can_accept_maths_domain_errors;
        }
        sp = stack.stack_point;
        build_node(stack, sp, node2);
        if (node.nodes_associated) {
          inline_function_expand(node2, node.nodes, errcode);
          copy_tree(node2, node);
        } else {
          errcode = err_wrong_parameters;
        }
      }
    }

    //! Initialise a parser
    //! @param[inout] errcode - Error code from initialising parser
    //! @param[in] options - Optional parameters to initialisation
    void init(error& errcode, const parser_init_options& options = {}) {
      error err;
      const Num pi = static_cast<Num>(3.141592653589793238462643383279503);
      Num c;
      bool no_import_l;

      errcode = err_none;

      if (owns_err_handler && err_handler != nullptr) {
        delete err_handler;
        err_handler = nullptr;
      }

      //If you specify a error handler then that error handler is used
      //If you specify a NULL error handler then an existing error handler that
      //the parser does not own is kept, otherwise an error handler is created
      //If you specify no error handler then a new error handler is created
      if (options.err_handler.is_set()) {
        error_handler* supplied = options.err_handler.get_or_value(nullptr);
        if (supplied != nullptr) {
          err_handler = supplied;
          owns_err_handler = false;
        } else if (err_handler != nullptr) {
          //Do nothing
        } else {
          err_handler = new error_handler;
          owns_err_handler = true;
        }
      } else {
        err_handler = new error_handler;
        owns_err_handler = true;
      }

      no_import_l = options.no_import;
      if (options.should_simplify != option_state::automatic) {
        this->should_simplify = options.should_simplify == option_state::yes;
      }
      if (options.should_minify != option_state::automatic) {
        this->should_minify = options.should_minify == option_state::yes;
      }
      if (options.physics != -1) physics_units = options.physics;
      if (options.allow_text != option_state::automatic) {
        this->allow_text = options.allow_text == option_state::yes;
      }
      if (options.maths_fail_quickly != option_state::automatic) {
        this->maths_fail_quickly =
            options.maths_fail_quickly == option_state::yes;
      }

      err_handler->init(errcode);
      if (is_init) return;
      is_init = true;

      if (!global_setup) {
        err = err_none;
        global_setup = true;
        global_registry.add_operator("+", uplus<Num>, assoc_ra, 4, err,
            {.unary = true, .description = "Unary plus",
            .operator_id = pt_op_unary_plus});
        global_registry.add_operator("-", uminus<Num>, assoc_ra, 4, err,
            {.unary = true, .description = "Unary minus",
            .operator_id = pt_op_unary_minus});
        global_registry.add_operator("+", bplus<Num>, assoc_a, 2, err,
            {.description = "Addition operator",
            .operator_id = pt_op_plus});
        global_registry.add_operator("-", bminus<Num>, assoc_la, 2, err,
            {.description = "Subtraction operator",
            .operator_id = pt_op_minus});
        global_registry.add_operator("*", times<Num>, assoc_a, 3, err,
            {.description = "Multiplication operator",
            .operator_id = pt_op_multiply});
        global_registry.add_operator("/", divide<Num>, assoc_la, 3, err,
            {.description = "Division operator",
            .operator_id = pt_op_divide});
        global_registry.add_operator("^", pow<Num>, assoc_ra, 4, err,
            {.description = "Raise to power operator",
            .operator_id = pt_op_power});
        global_registry.add_operator("e", expo<Num>, assoc_la, 4, err,
            {.description = "Exponentiation operator"});
        global_registry.add_operator("lt", lt<Num>, assoc_la, 1, err,
            {.description = "less than operator"});
        global_registry.add_operator("<", lt<Num>, assoc_la, 1, err,
            {.description = "less than operator"});
        global_registry.add_operator("le", le<Num>, assoc_la, 1, err,
            {.description = "less than or equal to operator"});
        global_registry.add_operator("<=", le<Num>, assoc_la, 1, err,
            {.description = "less than or equal to operator"});
        global_registry.add_operator("gt", gt<Num>, assoc_la, 1, err,
            {.description = "greater than operator"});
        global_registry.add_operator(">", gt<Num>, assoc_la, 1, err,
            {.description = "greater than operator"});
        global_registry.add_operator("ge", ge<Num>, assoc_la, 1, err,
            {.description = "greater than or equal to operator"});
        global_registry.add_operator(">=", ge<Num>, assoc_la, 1, err,
            {.description = "greater than or equal to operator"});
        global_registry.add_operator("eq", eq<Num>, assoc_la, 1, err,
            {.description = "equal to operator"});
        global_registry.add_operator("==", eq<Num>, assoc_la, 1, err,
            {.description = "equal to operator"});
        global_registry.add_operator("ne", neq<Num>, assoc_la, 1, err,
            {.description = "not equal to operator"});
        global_registry.add_operator("/=", neq<Num>, assoc_la, 1, err,
            {.description = "not equal to operator"});
        global_registry.add_operator("!=", neq<Num>, assoc_la, 1, err,
            {.description = "not equal to operator"});
        global_registry.add_operator("and", parser_and<Num>, assoc_la, 0, err,
            {.description = "logical and"});
        global_registry.add_operator("or", parser_or<Num>, assoc_la, 0, err,
            {.description = "logical or"});

        global_registry.add_constant("math.pi", pi, err,
            {.description = "Pi, the ratio of a circle's circumference to "
            "it's diameter"});

        global_registry.add_constant("scale.yotta", static_cast<Num>(1.0e24), err,
            {.description = "10^24"});
        global_registry.add_constant("scale.zetta", static_cast<Num>(1.0e21), err,
            {.description = "10^21"});
        global_registry.add_constant("scale.exa", static_cast<Num>(1.0e18), err,
            {.description = "10^18"});
        global_registry.add_constant("scale.peta", static_cast<Num>(1.0e15), err,
            {.description = "10^15"});
        global_registry.add_constant("scale.tera", static_cast<Num>(1.0e12), err,
            {.description = "10^12"});
        global_registry.add_constant("scale.giga", static_cast<Num>(1.0e9), err,
            {.description = "10^9"});
        global_registry.add_constant("scale.mega", static_cast<Num>(1.0e6), err,
            {.description = "10^6"});
        global_registry.add_constant("scale.kilo", static_cast<Num>(1.0e3), err,
            {.description = "10^3"});
        global_registry.add_constant("scale.hecto", static_cast<Num>(1.0e2), err,
            {.description = "10^2"});
        global_registry.add_constant("scale.deca", static_cast<Num>(1.0e1), err,
            {.description = "10^1"});
        global_registry.add_constant("scale.deci", static_cast<Num>(1.0e-1), err,
            {.description = "10^-1"});
        global_registry.add_constant("scale.centi", static_cast<Num>(1.0e-2), err,
            {.description = "10^-2"});
        global_registry.add_constant("scale.milli", static_cast<Num>(1.0e-3), err,
            {.description = "10^-3"});
        global_registry.add_constant("scale.micro", static_cast<Num>(1.0e-6), err,
            {.description = "10^-6"});
        global_registry.add_constant("scale.nano", static_cast<Num>(1.0e-9), err,
            {.description = "10^-9"});
        global_registry.add_constant("scale.pico", static_cast<Num>(1.0e-12), err,
            {.description = "10^-12"});
        global_registry.add_constant("scale.femto", static_cast<Num>(1.0e-15), err,
            {.description = "10^-15"});
        global_registry.add_constant("scale.atto", static_cast<Num>(1.0e-18), err,
            {.description = "10^-18"});
        global_registry.add_constant("scale.zepto", static_cast<Num>(1.0e-21), err,
            {.description = "10^-21"});
        global_registry.add_constant("scale.yocto", static_cast<Num>(1.0e-24), err,
            {.description = "10^-24"});

        global_registry.add_constant("math.huge", huge<Num>, err,
            {.description = "Largest number"});
        global_registry.add_constant("math.tiny", tiny<Num>, err,
            {.description = "Smallest number"});
        global_registry.add_function("math.abs", abs<Num>, 1, err,
            {.description = "`abs(a)` - Returns absolute value of a"});
        global_registry.add_function("math.floor", floor<Num>, 1, err,
            {.description = "`floor(a)` - Returns the nearest integer to a "
            "rounding towards zero"});
        global_registry.add_function("math.ceil", ceil<Num>, 1, err,
            {.description = "`ceil(a)` - Returns the nearest integer to a "
            "rounding away from zero"});
        global_registry.add_function("math.ceiling", ceil<Num>, 1, err,
            {.description = "`ceiling(a)` - Returns the nearest integer to a "
            "rounding away from zero"});
        global_registry.add_function("math.nint", nint<Num>, 1, err,
            {.description = "`nint(a)` - Returns the nearest integer to a "
            "rounding to nearest integer"});
        global_registry.add_function("math.trunc", aint<Num>, 1, err,
            {.description = "`trunc(a)` - Returns the nearest integer to a by "
            "ignoring the non-integer part"});
        global_registry.add_function("math.truncate", aint<Num>, 1, err,
            {.description = "`truncate(a)` - Returns the nearest integer to a "
            "by ignoring the non-integer part"});
        global_registry.add_function("math.aint", aint<Num>, 1, err,
            {.description = "`aint(a)` - Returns the nearest integer to a by "
            "ignoring the non-integer part"});
        global_registry.add_function("math.sqrt", sqrt<Num>, 1, err,
            {.description = "`sqrt(a)` - Returns the square root of a"});
        global_registry.add_function("math.sin", sin<Num>, 1, err,
            {.description = "`sin(a)` - Returns the sine of a, specifying a "
            "in radians"});
        global_registry.add_function("math.cos", cos<Num>, 1, err,
            {.description = "`cos(a)` - Returns the cosine of a, specifying a "
            "in radians"});
        global_registry.add_function("math.tan", tan<Num>, 1, err,
            {.description = "`tan(a)` - Returns the tangent of a, specifying "
            "a in radians"});
        global_registry.add_function("math.asin", asin<Num>, 1, err,
            {.description = "`asin(a)` - Returns the inverse sin of a, "
            "specifying the resulting angle in radians"});
        global_registry.add_function("math.acos", acos<Num>, 1, err,
            {.description = "`acos(a)` - Returns the inverse cosin of a, "
            "specifying the resulting angle in radians"});
        global_registry.add_function("math.atan", atan<Num>, 1, err,
            {.description = "`atan(a)` - Returns the inverse tangent of a, "
            "specifying the resulting angle in radians"});
        global_registry.add_function("math.atan2", atan2<Num>, 2, err,
            {.description = "`atan2(y, x)` - 2 argument arctangent. Returns "
            "the angle in radians between the positive x axis and the line "
            "joining the origin to the point (x,y)"});
        global_registry.add_function("math.sinh", sinh<Num>, 1, err,
            {.description = "`sinh(a)` - Returns the hyperbolic sine of a"});
        global_registry.add_function("math.cosh", cosh<Num>, 1, err,
            {.description = "`cosh(a)` - Returns the hyperbolic cosine of "
            "a"});
        global_registry.add_function("math.tanh", tanh<Num>, 1, err,
            {.description = "`tanh(a)` - Returns the hyperbolic tangent of "
            "a"});
        global_registry.add_function("math.asinh", asinh<Num>, 1, err,
            {.description = "`asinh(a)` - Returns the inverse hyperbolic "
            "sine of a"});
        global_registry.add_function("math.acosh", acosh<Num>, 1, err,
            {.description = "`acosh(a)` - Returns the inverse hyperbolic "
            "cosine of a"});
        global_registry.add_function("math.atanh", atanh<Num>, 1, err,
            {.description = "`atanh(a)` - Returns the inverse hyperbolic "
            "tangent of a"});
        global_registry.add_function("math.exp", exp<Num>, 1, err,
            {.description = "`exp(a)` - Returns the base e exponential of "
            "a"});
        global_registry.add_function("math.loge", loge<Num>, 1, err,
            {.description = "`loge(a)` - Returns the base e logarithm of a"});
        global_registry.add_function("math.log10", log10<Num>, 1, err,
            {.description = "`log10(a)` - Returns the base 10 logarithm of "
            "a"});
        global_registry.add_function("math.log_base", log_base<Num>, 2, err,
            {.description = "`log_base(value, base)` Returns the base "
            "\"base\" logarithm of \"value\""});
        global_registry.add_function("math.bessel_j", bessel_j<Num>, 2, err,
            {.description = "`bessel_j(n, x)` - Returns the Bessel function "
            "of the first kind of order n of x. Does not have derivative "
            "with respect to order"});
        global_registry.add_function("math.bessel_y", bessel_y<Num>, 2, err,
            {.description = "`bessel_y(n, x)` - Returns the Bessel function "
            "of the second kind of order n of x. Does not have derivative "
            "with respect to order"});
        global_registry.add_function("math.erf", erf<Num>, 1, err,
            {.description = "`erf(x)` - Return the error function of x"});
        global_registry.add_function("math.erfc", erfc<Num>, 1, err,
            {.description = "`erfc(x)` - Return the complementary error "
            "function of x"});
        global_registry.add_function("math.erfc_scaled", erfc_scaled<Num>, 1, err,
            {.description = "`erfc_scaled(x)` - Returns the exponentially "
            "scaled complementary error function of x"});
        global_registry.add_function("math.gamma", gamma_fn<Num>, 1, err,
            {.description = "`gamma(x)` - Returns the Bernoulli gamma "
            "function of x"});
        global_registry.add_function("math.log_gamma", log_gamma_fn<Num>, 1, err,
            {.description = "`log_gamma(x)` - Returns the logarithm of the "
            "gamma function of x"});

        global_registry.add_function("utility.gauss", gauss<Num>, 3, err,
            {.description = "`gauss(x,x0,w)` - Gaussian of the form "
            "exp(-((x-x0)/w)^2). The FWHM is 2wsqrt(ln(2))"});
        global_registry.add_function("utility.semigauss", semigauss<Num>, 4, err,
            {.description = "`semigauss(x, A, A0, w)`. Gives a semi-Gaussian "
            "profile in the variable x. The other parameters are the maximum "
            "value of the profile (A), the value of the function at x=0 (A0) "
            "and the characteristic rise of the function"});
        global_registry.add_function("utility.dsemigauss", dsemigauss<Num>, -1, err,
            {.description = "`dsemigauss(x, A, A0, w, dx, dA, DA0, dw)`. "
            "Derivative of semigauss"});
        global_registry.add_function("utility.supergauss", supergauss<Num>, 4, err,
            {.description = "`supergauss(x,x0,w,n)` - Returns a superGaussian "
            "function of the form exp(-((x-x0)/w)^n)"});
        global_registry.add_function("utility.interpolate", interpol<Num>, -1, err,
            {.description = "interpolate(p, (x0, y0), (x1, y1), ...) - get a "
            "value from a specified interpolation function. Specify a point "
            "p and a series of control points as (x,y) pairs. The nearest x "
            "points to p will be selected and the corresponding y value will "
            "be selected by linear interpolation"});
        global_registry.add_function("utility.deriv_interpolate", deriv_interpol<Num>,
            -1, err,
            {.description = "deriv_interpolate(p, (x0, y0), (x1, y1), ...) - "
            "get a gradient from a specified interpolation function. Specify "
            "a point p and a series of control points as (x,y) pairs. The "
            "nearest x points to p will be selected and the gradient of the "
            "corresponding linear reconstruction from "
            "\"utility.interpolate\" will be selected and returned"});

        global_registry.add_function("logic.if", parser_if<Num>, 3, err,
            {.description = "`if(a,b,c)` -  If a is not equal to zero returns "
            "b, otherwise returns c",
            .rtype = pt_control_if});

        //Unit indepdendent physical constants
        global_registry.add_constant("physics.na", static_cast<Num>(6.02214076e23), err,
            {.description = "Avogadro's constant"});

        //SI Physical constants
        global_registry.add_constant("physics.si.c", static_cast<Num>(2.99792458e8), err,
            {.description = "Speed of light (m s^-1)"});
        global_registry.add_constant("physics.si.G", static_cast<Num>(6.67408e-11), err,
            {.description = "Gravitational constant (m^3 kg^-1 s^-2)"});
        global_registry.add_constant("physics.si.h", static_cast<Num>(6.62607015e-34),
            err, {.description = "Planck's constant (m^2 kg s^-2)"});
        global_registry.add_constant("physics.si.q0", static_cast<Num>(1.602176565e-19),
            err, {.description = "Elementary charge (C)"});
        global_registry.add_constant("physics.si.epsilon0",
            static_cast<Num>(8.854187817620389850536563031710750e-12), err,
            {.description = "Vacuum permittivity (F m^-1)"});
        global_registry.add_constant("physics.si.mu0", static_cast<Num>(4.e-7) * pi, err,
            {.description = "Vacuum permeability (H m^-1)"});
        global_registry.add_constant("physics.si.kb", static_cast<Num>(1.3806488e-23),
            err, {.description = "Boltzmann's constant (m^2 kg s^-2 K^-1)"});
        global_registry.add_constant("physics.si.me", static_cast<Num>(9.10938291e-31),
            err, {.description = "Mass of electron (kg)"});
        global_registry.add_constant("physics.si.mp", static_cast<Num>(1.6726219e-27),
            err, {.description = "Mass of proton (kg)"});
        global_registry.add_constant("physics.si.mn", static_cast<Num>(1.674929e-27),
            err, {.description = "Mass of neutron (kg)"});

        //CGS physical constants
        c = static_cast<Num>(2.99792458e10);
        global_registry.add_constant("physics.cgs.gauss.c", c, err,
            {.description = "Speed of light (cm s^-1)"});
        global_registry.add_constant("physics.cgs.gauss.G",
            static_cast<Num>(6.67428e-8), err,
            {.description = "Gravitational constant (cm^3 g^-1 s^-2)"});
        global_registry.add_constant("physics.cgs.gauss.h",
            static_cast<Num>(6.62606885e-27), err,
            {.description = "Planck's constant (erg s)"});
        global_registry.add_constant("physics.cgs.gauss.q0",
            static_cast<Num>(4.80320427e-10), err,
            {.description = "Elementary charge (Fr)"});
        global_registry.add_constant("physics.cgs.gauss.epsilon0",
            static_cast<Num>(1.0), err,
            {.description = "Vacuum permittivity (unnecessary)"});
        global_registry.add_constant("physics.cgs.gauss.mu0",
            static_cast<Num>(1.0) / (c * c), err,
            {.description = "Vacuum permiability (unnecessary)"});
        global_registry.add_constant("physics.cgs.gauss.kb",
            static_cast<Num>(1.3806504e-16), err,
            {.description = "Boltzmann's constant (erg K^-1)"});
        global_registry.add_constant("physics.cgs.gauss.mu",
            static_cast<Num>(1.67377e-27), err);
        global_registry.add_constant("physics.cgs.gauss.me",
            static_cast<Num>(9.10938215e-28), err,
            {.description = "Mass of electrion (g)"});
        global_registry.add_constant("physics.cgs.gauss.mp",
            static_cast<Num>(1.6726219e-24), err,
            {.description = "Mass of proton (g)"});
        global_registry.add_constant("physics.cgs.gauss.mn",
            static_cast<Num>(1.674929e-24), err,
            {.description = "Mass of neutron (g)"});

        if (!no_import_l) {
          global_registry.include_namespace("math");
          global_registry.include_namespace("scale");
          global_registry.include_namespace("utility");
          global_registry.include_namespace("logic");
          this->registry.include_namespace("math");
          this->registry.include_namespace("scale");
          this->registry.include_namespace("utility");
          this->registry.include_namespace("logic");
          if (physics_units != physics_none) {
            global_registry.include_namespace("physics");
            this->registry.include_namespace("physics");
            if (physics_units == physics_si) {
              global_registry.include_namespace("physics.si");
              this->registry.include_namespace("physics.si");
            } else if (physics_units == physics_cgs_gauss) {
              global_registry.include_namespace("physics.cgs");
              global_registry.include_namespace("physics.cgs.gauss");
              this->registry.include_namespace("physics.cgs");
              this->registry.include_namespace("physics.cgs.gauss");
            }
          }
        }

        set_symbol_derivative_inner("+", "dparam(1)", errcode,
            {.global = true, .unary = true});
        set_symbol_derivative_inner("-", "-dparam(1)", errcode,
            {.global = true, .unary = true});
        set_symbol_derivative("+", "dparam(1) + dparam(2)", errcode,
            {.global = true});
        set_symbol_derivative("-", "dparam(1) - dparam(2)", errcode,
            {.global = true});
        set_symbol_derivative("*",
            "dparam(1)*param(2) + dparam(2)*param(1)", errcode,
            {.global = true});
        set_symbol_derivative("/",
            "(param(2)*dparam(1)-param(1)*dparam(2))/(param(2)^2)", errcode,
            {.global = true});
        set_symbol_derivative("^",
            "param(1)^(param(2)-1)*(param(2)*dparam(1) + "
            "param(1)*loge(param(1))*dparam(2))", errcode, {.global = true});
        set_symbol_derivative("e", "dparam(1) * 10^param(2) + "
            "param(1) * loge(10) * dparam(2)", errcode, {.global = true});
        set_symbol_derivative("lt", "param(1) lt param(2)", errcode,
            {.global = true});
        set_symbol_derivative("<", "param(1) < param(2)", errcode,
            {.global = true});
        set_symbol_derivative("le", "param(1) le param(2)", errcode,
            {.global = true});
        set_symbol_derivative("<=", "param(1) <= param(2)", errcode,
            {.global = true});
        set_symbol_derivative("gt", "param(1) gt param(2)", errcode,
            {.global = true});
        set_symbol_derivative(">", "param(1) > param(2)", errcode,
            {.global = true});
        set_symbol_derivative("ge", "param(1) ge param(2)", errcode,
            {.global = true});
        set_symbol_derivative(">=", "param(1) >= param(2)", errcode,
            {.global = true});
        set_symbol_derivative("eq", "param(1) eq param(2)", errcode,
            {.global = true});
        set_symbol_derivative("==", "param(1) == param(2)", errcode,
            {.global = true});
        set_symbol_derivative("ne", "param(1) ne param(2)", errcode,
            {.global = true});
        set_symbol_derivative("/=", "param(1) /= param(2)", errcode,
            {.global = true});
        set_symbol_derivative("!=", "param(1) != param(2)", errcode,
            {.global = true});
        set_symbol_derivative("and", "param(1) and param(2)", errcode,
            {.global = true});
        set_symbol_derivative("or", "param(1) or param(2)", errcode,
            {.global = true});

        set_symbol_derivative("math.abs", "if(param(1) > 0, dparam(1), "
            "if(param(1) < 0, -dparam(1), math.huge))", errcode,
            {.global = true});
        set_symbol_derivative("math.floor", "0", errcode, {.global = true});
        set_symbol_derivative("math.ceil", "0", errcode, {.global = true});
        set_symbol_derivative("math.ceiling", "0", errcode, {.global = true});
        set_symbol_derivative("math.nint", "0", errcode, {.global = true});
        set_symbol_derivative("math.trunc", "0", errcode, {.global = true});
        set_symbol_derivative("math.truncate", "0", errcode, {.global = true});
        set_symbol_derivative("math.aint", "0", errcode, {.global = true});
        set_symbol_derivative("math.sqrt", "dparam(1)/(2*sqrt(param(1)))",
            errcode, {.global = true});
        set_symbol_derivative("math.sin", "math.cos(param(1)) "
            "* dparam(1)", errcode, {.global = true});
        set_symbol_derivative("math.cos", "-math.sin(param(1)) "
            "* dparam(1)", errcode, {.global = true});
        set_symbol_derivative("math.tan", "dparam(1)/math.cos(param(1))^2",
            errcode, {.global = true});
        set_symbol_derivative("math.asin", "dparam(1)/math.sqrt(1-param(1)^2)",
            errcode, {.global = true});
        set_symbol_derivative("math.acos",
            "-dparam(1)/math.sqrt(1-param(1)^2)", errcode, {.global = true});
        set_symbol_derivative("math.atan", "dparam(1)/(1+param(1)^2)",
            errcode, {.global = true});
        set_symbol_derivative("math.atan2", "(param(2)*dparam(1) - "
            "param(1)*dparam(2))/(param(1)^2+param(2)^2)", errcode,
            {.global = true});
        set_symbol_derivative("math.sinh", "dparam(1) * math.cosh(param(1))",
            errcode, {.global = true});
        set_symbol_derivative("math.cosh", "dparam(1) * math.sinh(param(1))",
            errcode, {.global = true});
        set_symbol_derivative("math.tanh", "dparam(1) / math.cosh(param(1))^2",
            errcode, {.global = true});
        set_symbol_derivative("math.asinh",
            "dparam(1) / (math.sqrt(1+param(1)^2))", errcode,
            {.global = true});
        set_symbol_derivative("math.acosh", "dparam(1) / "
            "(math.sqrt(param(1)-1) * math.sqrt(1+param(1)))", errcode,
            {.global = true});
        set_symbol_derivative("math.atanh", "dparam(1) / (1-param(1)^2)",
            errcode, {.global = true});
        set_symbol_derivative("math.exp", "dparam(1) * math.exp(param(1))",
            errcode, {.global = true});
        set_symbol_derivative("math.loge", "dparam(1) / param(1)", errcode,
            {.global = true});
        set_symbol_derivative("math.log10",
            "dparam(1) / (param(1) * math.loge(10))", errcode,
            {.global = true});
        set_symbol_derivative("math.log_base", "dparam(1) / "
            "(param(1)*math.loge(param(2))) - math.loge(param(1)) "
            "* dparam(2)/(param(2)*math.loge(param(2))^2)", errcode,
            {.global = true});

        set_symbol_derivative("utility.gauss", "gauss(param(1), param(2), "
            "param(3)) * (-(2*(param(1)-param(2)) "
            "* (dparam(1) - dparam(2)))/param(3)^2 + (2*(param(1)-param(2))^2 "
            "* dparam(3)) / param(3)^3)", errcode, {.global = true});

        set_symbol_derivative("utility.semigauss",
            "if (param(1) <= sqrt(-loge(param(3)/param(2))) * param(4), "
            "(-2 * sqrt(-loge(param(3)/param(2))) * param(1) * param(2) "
            "* param(3) * param(4) * dparam(1) + "
            "sqrt(-loge(param(3)/param(2))) "
            "* param(2) * param(4)^3 * dparam(3) + param(1) * param(4)^2 "
            "* (param(3) * dparam(2) - param(2) * dparam(3)) + 2.0 "
            "* sqrt(-loge(param(3)/param(2))) * param(1)^2 * param(2) "
            "* param(3) * dparam(4) + 2 * loge(param(3)/param(2)) * param(2) "
            "* param(3) * param(4) * ( param(1) * dparam(4) - param(4) "
            "* dparam(1))) / (exp((param(1) - "
            "sqrt(-loge(param(3)/param(2))) "
            "* param(4)^2) / param(4)^2) * sqrt(-loge(param(3)/param(2))) "
            "* param(3) * param(4)^3), 0.0)", errcode, {.global = true});

        set_symbol_derivative("utility.supergauss",
            "-(((param(1)-param(2))/param(3))^param(4) * (param(4) "
            "* ((dparam(1)-dparam(2))/(param(1)-param(2)) - "
            "dparam(3)/param(3)))) "
            "* supergauss(param(1), param(2), param(3), param(4))", errcode,
            {.global = true});

        set_symbol_derivative("utility.interpolate",
            "utility.deriv_interpolate(all_params) * dparam(1)", errcode,
            {.global = true});
        lock_symbol_derivative("utility.dsemigauss", errcode,
            {.global = true});
        lock_symbol_derivative("utility.deriv_interpolate", errcode,
            {.global = true});

        set_symbol_derivative("logic.if", "if(param(1), dparam(2), dparam(3))",
            errcode, {.global = true});

        set_symbol_derivative("math.bessel_j",
            "0.5 * dparam(2) * (math.bessel_j(param(1)-1, param(2)) "
            "- math.bessel_j(param(1)+1, param(2)))", errcode,
            {.global = true});
        set_symbol_derivative("math.bessel_y",
            "0.5 * dparam(2) * (math.bessel_y(param(1)-1, param(2)) "
            "- math.bessel_y(param(1)+1, param(2)))", errcode,
            {.global = true});
        set_symbol_derivative("math.erf",
            "2 * dparam(1) * math.exp(-param(1)^2) / math.sqrt(math.pi)",
            errcode, {.global = true});
        set_symbol_derivative("math.erfc",
            "-2 * dparam(1) * math.exp(-param(1)^2) / math.sqrt(math.pi)",
            errcode, {.global = true});
        set_symbol_derivative("math.erfc_scaled",
            "-2 * dparam(1)/math.sqrt(math.pi) + 2 "
            " * erfc_scaled(param(1)) * param(1) * dparam(1)", errcode,
            {.global = true});
        lock_symbol_derivative("math.gamma", errcode, {.global = true});
        lock_symbol_derivative("math.log_gamma", errcode, {.global = true});

      }

      if (zero_stack == nullptr) {
        zero_stack = new eis::stack<Num>;
        tokenize("0", *zero_stack, errcode);
      }
    }

    //! Function to load a given block with information from the registry
    //! @param[in] symbol - Name of the symbol to get information about
    //! @param[in] options - Optional parameters
    //! @return Information about the symbol
    symbol_info get_symbol_info(const std::string& symbol,
        const get_symbol_info_options& options = {}) {
      symbol_info info;
      stack_element<Num> iblock{};
      stack_co_element icoblock{};
      bool is_unary, is_hidden = false;
      std::string str;

      is_unary = options.unary;

      if (!is_unary) {
        this->registry.fill_block(symbol, iblock, icoblock, is_unary, str,
            is_hidden);
        if (iblock.ptype == pt_bad) {
          global_registry.fill_block(symbol, iblock, icoblock, is_unary, str,
              is_hidden);
        }
        if (iblock.ptype == pt_bad) is_unary = true;
      }

      if (is_unary) {
        this->registry.fill_block(symbol, iblock, icoblock, is_unary, str,
            is_hidden);
        if (iblock.ptype == pt_bad) {
          global_registry.fill_block(symbol, iblock, icoblock, is_unary, str,
              is_hidden);
        }
      }

      info.exists = iblock.ptype != pt_bad;
      info.description = str;
      info.symbol_type = iblock.ptype;
      info.cap_bits = icoblock.cap_bits;
      info.deferred = (iblock.ptype == pt_deferred_variable
          || iblock.ptype == pt_deferred_function);
      info.is_hidden = is_hidden;
      info.expected_params = icoblock.expected_params;

      return info;
    }

    //! Function to test if a given namespace has been included
    //! @param[in] namespace_name - Name of the namespace to get information
    //! about
    //! @param[in] options - Should the test be in the global registry
    bool namespace_is_included(const std::string& namespace_name,
        const namespace_options& options = {}) {
      if (options.in_global) {
        return global_registry.is_included(namespace_name);
      } else {
        return this->registry.is_included(namespace_name);
      }
    }

    //! Include a namespace in the list of namespaces to search when a symbol
    //! is looked up by name
    //! @param[in] namespace_name - Name of the namespace to include
    //! @param[in] options - Should the namespace be included in the global
    //! registry
    void include_namespace(const std::string& namespace_name,
        const namespace_options& options = {}) {
      if (options.in_global) {
        global_registry.include_namespace(namespace_name);
      } else {
        this->registry.include_namespace(namespace_name);
      }
    }

    //! Function to test if a given symbol specified with full namespace
    //! qualification needs the namespace to resolve.
    //! @param[in] symbol - Name of the symbol
    //! @param[in] options - Should the test be in the global registry
    bool symbol_needs_namespace(const std::string& symbol,
        const namespace_options& options = {}) {
      bool symbol_needs_namespace_value;

      const std::size_t ldotloc = symbol.rfind('.');
      if (ldotloc == std::string::npos || ldotloc == symbol.size() - 1) {
        symbol_needs_namespace_value = false;
      } else {
        if (options.in_global) {
          symbol_needs_namespace_value =
              !global_registry.is_included(symbol.substr(0, ldotloc));
        } else {
          symbol_needs_namespace_value =
              !this->registry.is_included(symbol.substr(0, ldotloc));
        }
      }

      return symbol_needs_namespace_value;
    }

    //! Get the number of symbols in a specified registry
    count_type get_registry_symbol_count(eis::registry<Num>& registry) {
      return registry.get_name_count();
    }

    //! Get a symbol from the registry
    std::string get_registry_symbol(eis::registry<Num>& registry,
        count_type index) {
      return registry.get_name(index);
    }

    //! Function to load a given block with information from the registry
    //! @param[in] name - Name to look up in registry
    //! @param[out] iblock - Block to fill with registry information
    //! @param[out] icoblock - Coblock to fill with registry information
    //! @param[in] options - Optional parameters
    void load_block(const std::string& name, stack_element<Num>& iblock,
        stack_co_element& icoblock, const load_block_options& options = {}) {
      error work;
      Num value;
      bool can_be_unary, should_allow_params, should_allow_dparams;
      std::string::size_type slen;

      iblock = stack_element<Num>();
      icoblock = stack_co_element();

      should_allow_params = options.allow_params;

      should_allow_dparams = options.allow_dparams;

      iblock.ptype = pt_bad;
      iblock.rtype = pt_bad;
      iblock.value = 0;
      iblock.data = Num(0);
      icoblock.text = detail::trim_copy(name);
      work = 0;

      if (detail::trim_copy(name).empty()) {
        iblock.ptype = pt_null;
        iblock.rtype = pt_null;
        iblock.value = 0;
        iblock.data = Num(0);
        return;
      }

      const typecode paren_type = detail::as_parenthesis(name);
      if (paren_type != 0) {
        // block is a parenthesis
        iblock.ptype = pt_parenthesis;
        iblock.value = paren_type;
        return;
      }

      if (eis::strcmp(name, ",")) {
        iblock.ptype = pt_separator;
        iblock.rtype = pt_separator;
        iblock.value = 0;
        return;
      }

      slen = detail::trim_copy(name).size();
      if (name[0] == '"' && name[slen - 1] == '"') {
        iblock.ptype = pt_character;
        iblock.rtype = pt_character;
        return;
      }

      if (options.unary == option_state::automatic) {
        can_be_unary = !(this->last_block_type == pt_variable
            || this->last_block_type == pt_pointer_variable
            || this->last_block_type == pt_constant
            || this->last_block_type == pt_stored_variable
            || this->last_block_type == pt_function
            || this->last_block_type == pt_emplaced_function
            || (this->last_block_type == pt_parenthesis
            && this->last_block_value == paren_right_bracket));
      } else {
        can_be_unary = options.unary == option_state::yes;
      }

      if (eis::strcmp(name, "param") && should_allow_params) {
        iblock.ptype = pt_function;
        iblock.rtype = pt_param;
        icoblock.expected_params = 1;
        iblock.can_simplify = false;
        return;
      }

      if (eis::strcmp(name, "all_params") && should_allow_params) {
        iblock.ptype = pt_constant;
        iblock.rtype = pt_param;
        icoblock.expected_params = 0;
        iblock.can_simplify = false;
        return;
      }

      if (eis::strcmp(name, "dparam") && should_allow_dparams) {
        iblock.ptype = pt_function;
        iblock.rtype = pt_dparam;
        icoblock.expected_params = 1;
        iblock.can_simplify = false;
        return;
      }

      if (eis::strcmp(name, "all_dparams") && should_allow_dparams) {
        iblock.ptype = pt_constant;
        iblock.rtype = pt_dparam;
        icoblock.expected_params = 0;
        iblock.can_simplify = false;
        return;
      }

      if (eis::strcmp(name, "func") && should_allow_dparams) {
        iblock.ptype = pt_function;
        iblock.rtype = pt_placed_function;
        icoblock.expected_params = 1;
        iblock.can_simplify = false;
        return;
      }

      if (eis::strcmp(name, "zero") || eis::strcmp(name, "0")) {
        iblock.ptype = pt_constant;
        iblock.rtype = pt_zero;
        iblock.value = 0;
        iblock.data = Num(0);
        iblock.can_simplify = true;
        return;
      }

      if (eis::strcmp(name, "unity") || eis::strcmp(name, "1")) {
        iblock.ptype = pt_constant;
        iblock.rtype = pt_unity;
        iblock.value = 0;
        iblock.data = Num(1);
        iblock.can_simplify = true;
        return;
      }

      this->registry.fill_block(name, iblock, icoblock, can_be_unary);
      if (iblock.ptype != pt_bad) {
        icoblock.registry_type = registry_local;
        if (options.can_accept_maths_domain_errors != nullptr) {
          *options.can_accept_maths_domain_errors =
              *options.can_accept_maths_domain_errors
              || icoblock.can_accept_maths_domain_errors;
        }
        return;
      }

      global_registry.fill_block(name, iblock, icoblock, can_be_unary);
      if (iblock.ptype != pt_bad) {
        icoblock.registry_type = registry_global;
        if (options.can_accept_maths_domain_errors != nullptr) {
          *options.can_accept_maths_domain_errors =
              *options.can_accept_maths_domain_errors
              || icoblock.can_accept_maths_domain_errors;
        }
        return;
      }

      value = parse_string_as_real<Num>(name, work);
      if ((work & err_bad_value) == 0) {
        // block is a simple variable
        iblock.ptype = pt_constant;
        iblock.rtype = pt_constant;
        iblock.value = 0;
        iblock.data = value;
        return;
      }
    }

    //! Function to set a stack's eval_fn to a specified function
    //! @param[in] function_in - Function to be called when the stack is
    //! evaluated
    //! @param[inout] output - Stack to contain the output. If stack is not
    //! empty then new values are pushed to the end of the stack. This will
    //! usually cause multi valued results
    //! @param[inout] err - Error code for errors during tokenize
    //! @param[in] options - Optional parameters
    void set_result_function(parser_result_function<Num> function_in,
        eis::stack<Num>& output, error& err,
        const set_result_function_options& options = {}) {
      bool should_append;

      err = err_none;

      if (!this->is_init) init(err);

      should_append = options.append;

      if (!options.append) initialise_stack(output);
      output.eval_fn = function_in;
      output.filename = options.filename;
      output.line_number = options.line_number;
      output.char_offset = options.char_offset;
      output.cap_bits = output.cap_bits | options.cap_bits;
      output.owner = this;

      if (should_append) {
        this->registry.result_function_to_functor_stack(output, err);
        if (err != err_none) {
          err_handler->add_error(err_parser, err,
              {.filename = options.filename,
              .line_number = options.line_number});
          return;
        }
      }
    }

    //! Function to set a stack's result function to a function supplied by C
    //! code
    //! @param[in] function_in - Function to be called when the stack is
    //! evaluated
    //! @param[inout] output - Stack to contain the output
    //! @param[inout] err - Error code
    //! @param[in] options - Optional parameters
    void set_result_function(c_parser_result_function<Num> function_in,
        eis::stack<Num>& output, error& err,
        const set_result_function_options& options = {}) {
      err = err_none;

      if (!this->is_init) init(err);

      if (!options.append) initialise_stack(output);
      output.c_eval_fn = function_in;
      output.filename = options.filename;
      output.line_number = options.line_number;
      output.char_offset = options.char_offset;
      output.cap_bits = output.cap_bits | options.cap_bits;
      output.owner = this;

      if (options.append) {
        this->registry.result_function_to_functor_stack(output, err);
        if (err != err_none) {
          err_handler->add_error(err_parser, err,
              {.filename = options.filename,
              .line_number = options.line_number});
        }
      }
    }

    //! Function to tokenize an expression to a stack
    //! @param[in] expression_in - Expression to convert to stack
    //! @param[inout] output - Stack to contain the output. If stack is not
    //! empty then new values are pushed to the end of the stack. This will
    //! usually cause multi valued results
    //! @param[inout] err - Error code for errors during tokenize
    //! @param[in] options - Optional parameters
    void tokenize(const std::string& expression_in, eis::stack<Num>& output,
        error& err, const tokenize_options& options = {}) {
      tokenize_options inner = options;

      inner.allow_params = false;
      inner.allow_dparams = false;
      tokenize_inner(expression_in, output, err, inner);
      output.owner = this;
    }

    //! Function to tokenize an expression to a stack
    //! @param[in] expression_in - Expression to convert to stack
    //! @param[inout] output - Stack to contain the output. If stack is not
    //! empty then new values are pushed to the end of the stack. This will
    //! usually cause multi valued results
    //! @param[inout] err - Error code for errors during tokenize
    //! @param[in] options - Optional parameters
    void tokenize_inner(const std::string& expression_in,
        eis::stack<Num>& output, error& err,
        const tokenize_options& options = {}) {
      bool maybe_e, should_simplify, should_minify, should_dealloc;
      std::string current, expression;
      typecode current_type, ptype;
      offset_type current_pointer;
      bitmask cap_bits;
      offset_type charindex;
      bool atext, camde;

      stack_element<Num> iblock{};
      stack_co_element icoblock{};

      should_minify = detail::resolve(options.minify, this->should_minify);
      should_simplify = detail::resolve(options.simplify,
          this->should_simplify);
      atext = detail::resolve(options.allow_text, this->allow_text);

      camde = false;

      if (!this->is_init) init(err);
      should_dealloc = !options.append;
      output.filename = options.filename;
      output.line_number = options.line_number;
      output.char_offset = options.char_offset;

      if (should_dealloc) {
        deallocate_stack(output);
      } else {
        if (output.eval_fn != nullptr || output.c_eval_fn != nullptr) {
          err = err_none;
          this->registry.result_function_to_functor_stack(output, err);
          if (err != err_none) {
            err_handler->add_error(err_parser, err,
                {.filename = options.filename,
                .line_number = options.line_number});
            return;
          }
        }
      }
      if (!output.init) initialise_stack(output);

      expression = detail::trim_copy(expression_in);

      initialise_stack(this->stack);
      initialise_stack(this->brackets);
      push_to_stack(this->brackets, iblock, icoblock);
      this->brackets.entries[0].actual_params = 1;
      this->last_block_item_count = 1;

      current.assign(expression.size(), ' ');
      output.full_line = expression;

      current[0] = expression[0];
      current_pointer = 1;
      current_type = detail::char_type(expression[0]);
      maybe_e = false;

      err = err_none;
      charindex = 1;

      this->last_block_type = pt_null;
      this->last_block_value = 0;
      this->last_charindex = 1;
      this->last_block_text = " ";

      for (count_type i = 1; i < expression.size(); ++i) {
        ptype = detail::char_type(expression[i]);
        if ((ptype == current_type && ptype != char_delimiter)
            || (ptype == char_numeric && current_type == char_alpha
            && !eis::strcmp(current, "e"))) {
          current[static_cast<count_type>(current_pointer)] = expression[i];
          current_pointer = current_pointer + 1;
        } else if (eis::strcmp(current, "e") && !maybe_e) {
          current[static_cast<count_type>(current_pointer)] = expression[i];
          current_pointer = current_pointer + 1;
        } else {
          tokenize_subexpression_infix(current, iblock, icoblock, cap_bits,
              charindex + output.char_offset, charindex, output, err,
              options.filename, options.line_number, expression, atext,
              options.allow_params, options.allow_dparams, camde);
          charindex = static_cast<offset_type>(i) + 1;
          if (err != err_none) {
            err = err | err_parser;
            deallocate_stack(this->stack);
            deallocate_stack(this->brackets);
            return;
          }
          output.cap_bits = output.cap_bits | cap_bits;
          current.assign(current.size(), ' ');
          current_pointer = 1;
          current[0] = expression[i];
          current_type = ptype;
          maybe_e = (iblock.ptype == pt_variable)
              || (iblock.ptype == pt_constant)
              || (iblock.ptype == pt_pointer_variable);
        }
      }

      if (current_type != char_space) {
        tokenize_subexpression_infix(current, iblock, icoblock, cap_bits,
            charindex + output.char_offset, charindex, output, err,
            options.filename, options.line_number, expression, atext,
            options.allow_params, options.allow_dparams, camde);
        output.cap_bits = output.cap_bits | cap_bits;
      }

      if (err == err_none) {
        for (offset_type i = static_cast<offset_type>(this->stack.stack_point)
            - 1; i >= 0; --i) {
          const count_type iu = static_cast<count_type>(i);
          if (this->stack.entries[iu].ptype == pt_function) {
            if (this->stack.co_entries[iu].expected_params > 0) {
              err = err | err_wrong_parameters;
              add_error_options err_options;
              err_options.errstring = this->stack.co_entries[iu].text;
              err_options.charindex = this->stack.co_entries[iu].charindex;
              err_options.filename = this->stack.co_entries[iu].filename;
              err_options.line_number = this->stack.co_entries[iu].line_number;
              err_options.context_line_number = options.line_number;
              err_options.context_filename = options.filename;
              err_handler->add_error(err_parser, err, err_options);
            } else {
              this->stack.entries[iu].actual_params = 0;
            }
          }
          pop_to_stack(this->stack, output);
        }
      } else {
        err = err | err_parser;
      }
      deallocate_stack(this->stack);
      output.params =
          static_cast<count_type>(this->brackets.entries[0].actual_params);
      deallocate_stack(this->brackets);

      if (err != err_none) return;

      for (count_type iu = 0; iu < output.stack_point; ++iu) {
        if (output.co_entries[iu].full_line.empty()) {
          output.co_entries[iu].full_line = expression;
        }
        if (output.co_entries[iu].filename.empty()) {
          output.co_entries[iu].filename = options.filename;
        }
        if (options.line_number != -1
            && output.co_entries[iu].line_number == -1) {
          output.co_entries[iu].line_number = options.line_number;
        }
      }

      output.can_accept_maths_domain_errors = camde;

      expand_fn(output, err);
      if (should_simplify) simplify(output, err, {.host_params = nullptr});
      if (should_minify) minify(output, err);
    }

    //! Function to tokenize a numer to a stack
    //! @param[in] value_in - Numerical value to convert to stack
    //! @param[inout] output - Stack to contain the output. If stack is not
    //! empty then new values are pushed to the end of the stack. This will
    //! usually cause multi valued results
    //! @param[inout] err - Error code for errors during tokenize
    //! @param[in] options - Whether the newly parsed value should be appended
    //! to the values in the stack or no
    void tokenize_number(Num value_in, eis::stack<Num>& output, error&,
        const tokenize_number_options& options = {}) {
      bool should_dealloc;
      stack_element<Num> iblock{};
      stack_co_element icoblock{};

      should_dealloc = !options.append;
      if (should_dealloc) deallocate_stack(output);
      if (!output.init) initialise_stack(output);

      iblock.ptype = pt_constant;
      iblock.value = 0;
      iblock.data = value_in;
      icoblock.text = detail::format_number("%.17E", value_in);

      push_to_stack(output, iblock, icoblock);
      output.owner = this;
    }

    //! Function to evaluate a stack
    //! @param[inout] stack - Stack to evaluate
    //! @param[inout] result - Array holding all the results from the
    //! evaluation. Will only be reallocated if it is too small to hold all the
    //! results
    //! @param[inout] errcode - Error code returned by the evaluation
    //! @param[in] options - Host code parameters
    //! @return Number of results returned by the evaluation
    count_type evaluate(eis::stack<Num>& stack, std::vector<Num>& result,
        error& errcode, const host_options& options = {}) {
      void* params = options.host_params;
      count_type evaluate_value;

      errcode = err_none;

      //A stack can only be evaluated by the parser that it belongs to
      if (stack.owner != nullptr && stack.owner != this) {
        errcode = err_bad_stack;
        err_handler->add_error(err_evaluator, errcode);
        return 0;
      }

      if (stack.has_emplaced) {
        errcode = errcode | err_has_emplaced;
        err_handler->add_error(err_evaluator, errcode);
        return 0;
      }

      if (stack.has_deferred) {
        undefer(stack, errcode, {.host_params = params});
        if (errcode != err_none) return 0;
        stack.has_deferred = false;
      }

      evaluate_value = this->evaluator.evaluate(stack, result, params, errcode,
          *err_handler);

      stack.sanity_checked = (errcode == err_none);

      return evaluate_value;
    }

    //! Function to evaluate a string straight to a result
    //! @param[in] str - String to evaluate as maths
    //! @param[inout] result - Array holding all the results from the
    //! evaluation. Will only be reallocated if it is too small to hold all the
    //! results
    //! @param[inout] errcode - Error code from the combined tokenization,
    //! optionally simplification and evaluation
    //! @param[in] options - Optional parameters
    //! @return Number of results and the capability bits of the parsed stack
    parser_results evaluate(const std::string& str,
        std::vector<Num>& result, error& errcode,
        const evaluate_string_options& options = {}) {
      parser_results evaluate_value;
      eis::stack<Num> stack;

      errcode = err_none;

      tokenize(str, stack, errcode, {.simplify = options.simplify,
          .minify = options.minify, .filename = options.filename,
          .line_number = options.line_number,
          .char_offset = options.char_offset,
          .allow_text = options.allow_text});
      if (errcode != err_none) return evaluate_value;
      if (stack.has_emplaced) {
        emplace(stack, errcode, {.host_params = options.host_params});
      }
      if (errcode == err_none) {
        evaluate_value.count = evaluate(stack, result, errcode,
            {.host_params = options.host_params});
        evaluate_value.cap_bits = stack.cap_bits;
        deallocate_stack(stack);
      }

      return evaluate_value;
    }

    //! Function to evaluate a stack once for each of a number of sets of host
    //! parameters. Evaluations that fail are not passed to the sink and
    //! evaluation continues with the next set of host parameters
    //! @param[inout] stack - Stack to evaluate
    //! @param[in] count - Number of evaluations
    //! @param[in] input_fn - Callable that takes the index of an evaluation and
    //! returns the host parameters for it
    //! @param[in] sink_fn - Callable that takes the index of an evaluation, a
    //! pointer to the results of the evaluation and the number of results. It
    //! returns whether or not to continue evaluating
    //! @param[inout] errcode - Bitwise OR of the error codes of all of the
    //! evaluations
    //! @return Index of the first evaluation that failed. -1 if none failed
    template <typename InputFn, typename SinkFn>
    offset_type evaluate_each(eis::stack<Num>& stack, count_type count,
        InputFn input_fn, SinkFn sink_fn, error& errcode) {
      std::vector<Num> result;
      error err;
      offset_type first_failed = -1;

      errcode = err_none;

      for (count_type i = 0; i < count; ++i) {
        const count_type result_count = evaluate(stack, result, err,
            {.host_params = input_fn(i)});
        errcode = errcode | err;
        if (err != err_none) {
          if (first_failed < 0) first_failed = static_cast<offset_type>(i);
          continue;
        }
        if (!sink_fn(i, result.data(), result_count)) break;
      }

      return first_failed;
    }

    //! Function to evaluate a stack once for each of a number of sets of host
    //! parameters where each evaluation must return exactly one result. The
    //! number of evaluations is the size of "results"
    //! @param[inout] stack - Stack to evaluate
    //! @param[in] input_fn - Callable that takes the index of an evaluation and
    //! returns the host parameters for it
    //! @param[inout] results - Results of the evaluations. Any sized range
    //! that can be assigned a value. Must be sized by the caller
    //! @param[inout] errcode - Bitwise OR of the error codes of all of the
    //! evaluations. An evaluation that does not return exactly one result
    //! stops the evaluation and sets err_stack_params
    //! @return Index of the first evaluation that failed. -1 if none failed
    template <typename InputFn, std::ranges::sized_range R>
        requires std::invocable<InputFn&, count_type>
        && std::convertible_to<std::invoke_result_t<InputFn&, count_type>,
        void*>
        && std::assignable_from<std::ranges::range_reference_t<R>, Num>
    offset_type evaluate_range(eis::stack<Num>& stack, InputFn input_fn,
        R& results, error& errcode) {
      const count_type count = std::ranges::size(results);
      offset_type bad_count_index = -1;
      offset_type first_failed;
      auto target = std::ranges::begin(results);
      count_type position = 0;

      first_failed = evaluate_each(stack, count, input_fn,
          [&](count_type i, const Num* values, count_type n) -> bool {
            if (n != 1) {
              bad_count_index = static_cast<offset_type>(i);
              return false;
            }
            std::ranges::advance(target, i - position);
            position = i;
            *target = values[0];
            return true;
          }, errcode);

      if (bad_count_index >= 0) {
        errcode = errcode | err_stack_params;
        if (first_failed < 0 || bad_count_index < first_failed) {
          first_failed = bad_count_index;
        }
      }

      return first_failed;
    }

    //! Function to evaluate a stack once for each of a number of sets of host
    //! parameters where each evaluation must return exactly one result. The
    //! number of evaluations is the size of "results"
    //! @param[inout] stack - Stack to evaluate
    //! @param[in] inputs - Array of host parameters, one for each evaluation.
    //! Null means that there are no host parameters
    //! @param[in] n_inputs - Number of host parameters in "inputs". Must be 1,
    //! in which case it is used for every evaluation, or the size of "results"
    //! @param[inout] results - Results of the evaluations. Any sized range
    //! that can be assigned a value. Must be sized by the caller
    //! @param[inout] errcode - Bitwise OR of the error codes of all of the
    //! evaluations. An evaluation that does not return exactly one result
    //! stops the evaluation and sets err_stack_params
    //! @return Index of the first evaluation that failed. -1 if none failed
    template <std::ranges::sized_range R>
        requires std::assignable_from<std::ranges::range_reference_t<R>, Num>
    offset_type evaluate_range(eis::stack<Num>& stack, void* const* inputs,
        count_type n_inputs, R& results, error& errcode) {
      const count_type count = std::ranges::size(results);

      if (inputs != nullptr && n_inputs != 1 && n_inputs != count) {
        errcode = err_wrong_parameters;
        return -1;
      }

      return evaluate_range(stack,
          [&](count_type i) -> void* {
            if (inputs == nullptr) return nullptr;
            return inputs[n_inputs == 1 ? 0 : i];
          }, results, errcode);
    }

    //! Function to evaluate a stack a number of times without host parameters
    //! where each evaluation must return exactly one result. The number of
    //! evaluations is the size of "results"
    //! @param[inout] stack - Stack to evaluate
    //! @param[inout] results - Results of the evaluations. Any sized range
    //! that can be assigned a value. Must be sized by the caller
    //! @param[inout] errcode - Bitwise OR of the error codes of all of the
    //! evaluations. An evaluation that does not return exactly one result
    //! stops the evaluation and sets err_stack_params
    //! @return Index of the first evaluation that failed. -1 if none failed
    template <std::ranges::sized_range R>
        requires std::assignable_from<std::ranges::range_reference_t<R>, Num>
    offset_type evaluate_range(eis::stack<Num>& stack, R& results,
        error& errcode) {
      return evaluate_range(stack, [](count_type) -> void* { return nullptr; },
          results, errcode);
    }

    //! Function to evaluate a stack once for each of a number of sets of host
    //! parameters where each evaluation can return any number of results. The
    //! number of evaluations is the size of "results"
    //! @param[inout] stack - Stack to evaluate
    //! @param[in] input_fn - Callable that takes the index of an evaluation and
    //! returns the host parameters for it
    //! @param[inout] results - Results of the evaluations. Any sized range
    //! whose elements can be built from a pair of pointers to the results of
    //! an evaluation. Must be sized by the caller
    //! @param[inout] errcode - Bitwise OR of the error codes of all of the
    //! evaluations
    //! @return Index of the first evaluation that failed. -1 if none failed
    template <typename InputFn, std::ranges::sized_range R>
        requires std::invocable<InputFn&, count_type>
        && std::convertible_to<std::invoke_result_t<InputFn&, count_type>,
        void*>
        && std::constructible_from<std::ranges::range_value_t<R>, const Num*,
        const Num*>
        && std::assignable_from<std::ranges::range_reference_t<R>,
        std::ranges::range_value_t<R>>
    offset_type evaluate_range_multi(eis::stack<Num>& stack, InputFn input_fn,
        R& results, error& errcode) {
      const count_type count = std::ranges::size(results);
      auto target = std::ranges::begin(results);
      count_type position = 0;

      return evaluate_each(stack, count, input_fn,
          [&](count_type i, const Num* values, count_type n) -> bool {
            std::ranges::advance(target, i - position);
            position = i;
            *target = std::ranges::range_value_t<R>(values, values + n);
            return true;
          }, errcode);
    }

    //! Function to evaluate a stack once for each of a number of sets of host
    //! parameters where each evaluation can return any number of results. The
    //! number of evaluations is the size of "results"
    //! @param[inout] stack - Stack to evaluate
    //! @param[in] inputs - Array of host parameters, one for each evaluation.
    //! Null means that there are no host parameters
    //! @param[in] n_inputs - Number of host parameters in "inputs". Must be 1,
    //! in which case it is used for every evaluation, or the size of "results"
    //! @param[inout] results - Results of the evaluations. Any sized range
    //! whose elements can be built from a pair of pointers to the results of
    //! an evaluation. Must be sized by the caller
    //! @param[inout] errcode - Bitwise OR of the error codes of all of the
    //! evaluations
    //! @return Index of the first evaluation that failed. -1 if none failed
    template <std::ranges::sized_range R>
        requires std::constructible_from<std::ranges::range_value_t<R>,
        const Num*, const Num*>
        && std::assignable_from<std::ranges::range_reference_t<R>,
        std::ranges::range_value_t<R>>
    offset_type evaluate_range_multi(eis::stack<Num>& stack,
        void* const* inputs, count_type n_inputs, R& results,
        error& errcode) {
      const count_type count = std::ranges::size(results);

      if (inputs != nullptr && n_inputs != 1 && n_inputs != count) {
        errcode = err_wrong_parameters;
        return -1;
      }

      return evaluate_range_multi(stack,
          [&](count_type i) -> void* {
            if (inputs == nullptr) return nullptr;
            return inputs[n_inputs == 1 ? 0 : i];
          }, results, errcode);
    }

    //! Function to evaluate a stack a number of times without host parameters
    //! where each evaluation can return any number of results. The number of
    //! evaluations is the size of "results"
    //! @param[inout] stack - Stack to evaluate
    //! @param[inout] results - Results of the evaluations. Any sized range
    //! whose elements can be built from a pair of pointers to the results of
    //! an evaluation. Must be sized by the caller
    //! @param[inout] errcode - Bitwise OR of the error codes of all of the
    //! evaluations
    //! @return Index of the first evaluation that failed. -1 if none failed
    template <std::ranges::sized_range R>
        requires std::constructible_from<std::ranges::range_value_t<R>,
        const Num*, const Num*>
        && std::assignable_from<std::ranges::range_reference_t<R>,
        std::ranges::range_value_t<R>>
    offset_type evaluate_range_multi(eis::stack<Num>& stack, R& results,
        error& errcode) {
      return evaluate_range_multi(stack,
          [](count_type) -> void* { return nullptr; }, results, errcode);
    }

    //! Function to resolve deferred elements in a stack that have now been
    //! given final values.
    //! This will happen automatically if a stack is marked as having
    //! deferred elements when the stack is evaluated. Undeferring is a
    //! permanent action that changes the stack so that it is now evaluatable.
    //! The resultant stack will be simplified and minified after undeferring
    //! using the default values for this stack
    //! @param[inout] stack - Stack to undefer
    //! @param[inout] errcode - Error code from undefering stage
    //! @param[in] options - Optional host code parameters that might be needed
    void undefer(eis::stack<Num>& stack, error& errcode,
        const host_options& options = {}) {
      offset_type stored_params, stored_charpos, stored_full_line_pos;
      offset_type stored_line_number;
      [[maybe_unused]] bitmask cap_bits;
      std::string str, stored_full_line, stored_filename;
      void* params;

      if (stack.co_entries.empty()) {
        errcode = err_invalid_minify;
        return;
      }

      params = options.host_params;

      cap_bits = 0;

      const count_type count = stack.stack_point;
      for (count_type iu = 0; iu < count; ++iu) {
        if (stack.co_entries[iu].defer) {
          //This copy is a workaround, I do not believe it is required per
          //standard
          str = stack.co_entries[iu].text;
          stored_params = stack.entries[iu].actual_params;
          stored_charpos = stack.co_entries[iu].charindex;
          stored_full_line_pos = stack.co_entries[iu].full_line_pos;
          stored_line_number = stack.co_entries[iu].line_number;
          stored_full_line = stack.co_entries[iu].full_line;
          stored_filename = stack.co_entries[iu].filename;
          load_block(str, stack.entries[iu], stack.co_entries[iu]);
          stack.entries[iu].actual_params = stored_params;
          stack.co_entries[iu].charindex = stored_charpos;
          stack.co_entries[iu].full_line_pos = stored_full_line_pos;
          stack.co_entries[iu].line_number = stored_line_number;
          stack.co_entries[iu].full_line = stored_full_line;
          stack.co_entries[iu].filename = stored_filename;
          if (stack.co_entries[iu].defer) {
            errcode = errcode | err_has_deferred;
            add_error_options err_options;
            err_options.errstring = str;
            err_options.charindex = stack.co_entries[iu].charindex;
            err_options.filename = stack.co_entries[iu].filename;
            err_options.line_number = stack.co_entries[iu].line_number;
            err_handler->add_error(err_parser, errcode, err_options);
            return;
          }
          if (stack.entries[iu].ptype == pt_stored_variable) {
            this->registry.copy_in_stored(
                static_cast<count_type>(stack.entries[iu].value), stack,
                errcode, {.insert_point = static_cast<offset_type>(iu) + 1,
                .allow_multiple = false});
            if (errcode != err_none) {
              add_error_options err_options;
              err_options.errstring = str;
              err_options.charindex = stack.co_entries[iu].charindex;
              err_options.filename = stack.co_entries[iu].filename;
              err_options.line_number = stack.co_entries[iu].line_number;
              err_handler->add_error(err_parser, errcode, err_options);
              return;
            }
          }
          stack.cap_bits = stack.cap_bits | stack.co_entries[iu].cap_bits;
        }
      }

      if (this->should_minify) minify(stack, errcode);
      if (this->should_simplify) {
        simplify(stack, errcode, {.host_params = params});
      }
    }

    //! Function to simplify a stack. This ignores the default parser settings
    //! and always simplifies the stack
    //! @param[inout] stack - Stack to simplify
    //! @param[inout] errcode - Error code from the simplification step
    //! @param[in] options - Optional host code parameters. Can be used by
    //! stack elements to determine if they can be simplified or not
    void simplify(eis::stack<Num>& stack, error& errcode,
        const simplify_options& options = {}) {
      void* params = options.host_params;

      simplify_stack(stack, params, errcode,
          simplify_stack_options<Num>{.err_handler = this->err_handler,
          .maths_fail_quickly = detail::resolve(options.maths_fail_quickly,
          this->maths_fail_quickly), .host_parser = this});
    }

    //! Function to minify a stack. This ignores the default parser settings
    //! and always minify the stack
    //! @param[inout] stack - Stack to minify
    //! @param[inout] errcode - Error code from the simplification step
    void minify(eis::stack<Num>& stack, error&) {
      minify_stack(stack);
    }

    //! Function to emplace a single node of an AST.
    //! This function goes down to the bottom of an AST and then goes upwards
    //! When it encounters a node that needs emplacement on the way up all of
    //! the parameters to that function (if any) are evaluated and their values
    //! are passed to the emplacement function. The entire node and it's
    //! children are replaced with the stack returned from the emplacement
    //! function. The parameter "remaining_functions" is set to true if any
    //! functions need emplacement but cannot be emplaced
    //! @param[inout] tree_node - Node to operate on
    //! @param[in] host_params - Host code specified parameters
    //! @param[out] remaining_functions - Logical determining if there are
    //! remaining unemplaced functions below this node
    //! @param[inout] capbits - Capability bits for this node.
    //! @param[inout] can_accept_maths_domain_errors - Should this stack trap
    //! maths domain error
    //! @param[inout] errcode - Error code for this node
    void emplace_node(tree_item<Num>& tree_node, void* host_params,
        bool& remaining_functions, bitmask& capbits,
        bool& can_accept_maths_domain_errors, error& errcode) {
      tree_item<Num> new_node;
      error errcode_l;
      count_type nparams, sp;
      eis::stack<Num> temp_stack, functor_stack;
      eis::stack<Num>* emplace_stack = nullptr;
      std::vector<Num> results, params;
      parser_late_bind_fn<Num> late_bind_fn;
      parser_late_bind_stack_fn<Num> stack_late_bind_fn;
      parser_late_bind_functor<Num> late_bind_functor;
      status status_code;
      std::vector<eis::stack<Num>> stacks;

      status_code = status_none;
      errcode_l = err_none;

      if (tree_node.nodes_associated) {
        for (count_type inode = 0; inode < tree_node.nodes.size(); ++inode) {
          emplace_node(tree_node.nodes[inode], host_params,
              remaining_functions, capbits, can_accept_maths_domain_errors,
              errcode);
        }
      }

      //This is not an stored function, no emplacement necessary
      if (tree_node.value.ptype != pt_emplaced_function
          && tree_node.value.ptype != pt_emplaced_variable) return;

      const late_bind_fn_holder<Num> holder =
          this->registry.get_stored_emplacement(
          static_cast<count_type>(tree_node.value.value));
      late_bind_fn = holder.lb_contents;
      stack_late_bind_fn = holder.lbs_contents;
      late_bind_functor = holder.lb_functor;

      //Get stacks for the parameters
      nparams = 0;
      if (tree_node.nodes_associated) {
        nparams = tree_node.nodes.size();
        stacks.resize(nparams);
        errcode_l = err_none;
        for (count_type iu = 0; iu < nparams; ++iu) {
          initialise_stack(stacks[iu]);
          tree_to_stack(tree_node.nodes[nparams - iu - 1], stacks[iu]);
          if (stacks[iu].eval_fn != nullptr
              || stacks[iu].c_eval_fn != nullptr) {
            this->registry.result_function_to_functor_stack(stacks[iu],
                errcode);
          }
        }
      } else {
        stacks.resize(nparams);
      }

      //Fortran stack bind function
      if (stack_late_bind_fn != nullptr) {
        errcode_l = err_none;
        stack_late_bind_fn(tree_node.co_value.text, nparams, stacks.data(),
            host_params, this, temp_stack, status_code, errcode_l);
        emplace_stack = &temp_stack;
        if (errcode_l != err_none) {
          errcode = errcode | errcode_l;
          if (!tree_node.co_value.text.empty()) {
            add_error_options err_options;
            err_options.errstring = tree_node.co_value.text;
            err_options.charindex = tree_node.co_value.charindex;
            err_handler->add_error(err_emplacer, errcode_l, err_options);
          } else {
            err_handler->add_error(err_emplacer, errcode_l);
          }
        }
      }

      //Now deal with emplaced functions that want numerical rather than stack
      //parameters

      //First generate the parameters
      if (tree_node.nodes_associated) {
        params.resize(nparams);
        errcode_l = err_none;
        for (count_type iu = 0; iu < nparams; ++iu) {
          evaluate(stacks[iu], results, errcode_l,
              {.host_params = host_params});
          params[iu] = results[0];
          errcode = errcode | errcode_l;
        }
        if (errcode_l != err_none) {
          errcode = errcode | errcode_l;
          if (!tree_node.co_value.text.empty()) {
            add_error_options err_options;
            err_options.errstring = tree_node.co_value.text;
            err_options.charindex = tree_node.co_value.charindex;
            err_handler->add_error(err_emplacer, errcode_l, err_options);
          } else {
            err_handler->add_error(err_emplacer, errcode_l);
          }
        }
      } else {
        params.resize(nparams);
      }

      //Fortran bind function
      if (late_bind_fn != nullptr || late_bind_functor) {
        initialise_stack(temp_stack);
        emplace_stack = &temp_stack;
        if (late_bind_fn != nullptr) {
          late_bind_fn(tree_node.co_value.text, nparams, params.data(),
              host_params, this, temp_stack, status_code, errcode_l);
        } else {
          late_bind_functor(tree_node.co_value.text, nparams, params.data(),
              host_params, this, temp_stack, status_code, errcode_l);
        }
        errcode = errcode | errcode_l;
        if (!tree_node.co_value.text.empty()) {
          add_error_options err_options;
          err_options.errstring = tree_node.co_value.text;
          err_options.charindex = tree_node.co_value.charindex;
          err_handler->add_error(err_emplacer, errcode_l, err_options);
        } else {
          err_handler->add_error(err_emplacer, errcode_l);
        }
      }
      params.clear();
      for (auto& element : stacks) deallocate_stack(element);
      stacks.clear();

      remaining_functions = true;
      if (emplace_stack == nullptr) return;
      if (!emplace_stack->init) return;

      if (emplace_stack->has_emplaced) {
        emplace(*emplace_stack, errcode_l, {.host_params = host_params});
        errcode = errcode | errcode_l;
      }

      if (emplace_stack->eval_fn != nullptr
          || emplace_stack->c_eval_fn != nullptr) {
        this->registry.result_function_to_functor_stack(*emplace_stack,
            errcode, functor_stack);
        emplace_stack = &functor_stack;
      }
      if (emplace_stack->stack_point == 0) return;

      capbits = capbits | emplace_stack->cap_bits;
      can_accept_maths_domain_errors = can_accept_maths_domain_errors
          && emplace_stack->can_accept_maths_domain_errors;
      //If emplacement is not forbidden by status then build new node
      if ((status_code & status_no_emplace) == 0) {
        sp = emplace_stack->stack_point;
        build_node(*emplace_stack, sp, new_node);
        tree_node.nodes.clear();
        tree_node = std::move(new_node);
        remaining_functions = false;
      } else {
        remaining_functions = true;
      }

      deallocate_stack(*emplace_stack);
    }

    //! Function to emplace all functions in a stack
    //! Function emplaces all emplaceable functions in a stack. If the
    //! "destination" parameter is specified then the original stack is left
    //! unaltered and the emplaced stack is built in the destination parameter
    //! This allows multiple eplacements with different "host_params" to
    //! produce different stacks
    //! @param[inout] stack - Stack to emplace
    //! @param[inout] errcode - Error code from emplacement
    //! @param[in] options - Host code specified parameters and destination for
    //! the final emplaced stack
    void emplace(eis::stack<Num>& stack, error& errcode,
        const emplace_options<Num>& options = {}) {
      tree_item<Num> root;
      count_type sp;
      bitmask capbits;
      bool remaining_functions;
      eis::stack<Num>* sptr;
      void* params = options.host_params;

      if (!stack.has_emplaced) {
        if (options.destination != nullptr) {
          copy_stack(stack, *options.destination);
        }
        return;
      }

      if (options.destination != nullptr) {
        copy_stack(stack, *options.destination);
        sptr = options.destination;
      } else {
        sptr = &stack;
      }

      remaining_functions = false;
      capbits = sptr->cap_bits;
      sp = sptr->stack_point;
      build_node(stack, sp, root);
      emplace_node(root, params, remaining_functions, capbits,
          stack.can_accept_maths_domain_errors, errcode);
      deallocate_stack(*sptr);
      initialise_stack(*sptr);
      tree_to_stack(root, *sptr);
      sptr->cap_bits = capbits;
      sptr->has_emplaced = remaining_functions;

      expand_fn(*sptr, errcode);
      if (this->should_minify) minify(*sptr, errcode);
      if (this->should_simplify) {
        simplify(*sptr, errcode, {.host_params = params});
      }
    }

    void inline_function_expand(tree_item<Num>& fn_tree,
        const std::vector<tree_item<Num>>& params, error& errcode,
        const inline_expand_options<Num>& options = {}) {
      eis::stack<Num> stack;
      std::vector<Num> results;
      offset_type el;
      count_type param_count, pdest;
      error errcode_l = 0;
      bool need_node_replace;

      errcode = err_none;

      if (fn_tree.nodes_associated) {
        need_node_replace = false;
        param_count = fn_tree.nodes.size();
        for (offset_type inode = static_cast<offset_type>(fn_tree.nodes.size())
            - 1; inode >= 0; --inode) {
          const count_type iu = static_cast<count_type>(inode);
          inline_function_expand(fn_tree.nodes[iu], params, errcode, options);
          if (fn_tree.nodes[iu].value.rtype == pt_param
              && fn_tree.nodes[iu].value.ptype == pt_constant) {
            need_node_replace = true;
            param_count = param_count - 1 + params.size();
          }
          if (fn_tree.nodes[iu].value.rtype == pt_dparam
              && fn_tree.nodes[iu].value.ptype == pt_constant) {
            need_node_replace = true;
            param_count = param_count - 1 + options.dparams->size();
          }
        }
        if (need_node_replace) {
          //Can only pass all parameters to variadic function
          if (fn_tree.co_value.expected_params > -1) {
            errcode_l = err_bad_value;
            add_error_options err_options;
            err_options.errstring = fn_tree.co_value.text;
            err_options.charindex = fn_tree.co_value.charindex;
            err_handler->add_error(err_parser, errcode_l, err_options);
            errcode = errcode_l;
            return;
          }
          std::vector<tree_item<Num>> temp = std::move(fn_tree.nodes);
          fn_tree.nodes = std::vector<tree_item<Num>>(param_count);
          fn_tree.value.actual_params = static_cast<offset_type>(param_count);
          pdest = 0;
          for (count_type inode = 0; inode < temp.size(); ++inode) {
            if (temp[inode].value.rtype == pt_param
                || (temp[inode].value.rtype == pt_dparam
                && temp[inode].value.ptype == pt_constant)) {
              if (temp[inode].value.rtype == pt_param) {
                for (count_type iparam = 0; iparam < params.size();
                    ++iparam) {
                  copy_tree(params[iparam], fn_tree.nodes[pdest]);
                  pdest = pdest + 1;
                }
              } else {
                for (count_type iparam = 0;
                    iparam < options.dparams->size(); ++iparam) {
                  copy_tree((*options.dparams)[iparam], fn_tree.nodes[pdest]);
                  pdest = pdest + 1;
                }
              }
            } else {
              copy_tree(temp[inode], fn_tree.nodes[pdest]);
              pdest = pdest + 1;
            }
          }
        }
      }

      initialise_stack(stack);

      //Only replace a node if it is a param, or a dparam and is a function
      //If not a function then it's an "all_params" or "all_dparams" command
      //and should be handled at the level of its parent
      if ((fn_tree.value.rtype == pt_param || fn_tree.value.rtype == pt_dparam)
          && fn_tree.value.ptype == pt_function) {
        tree_to_stack(fn_tree.nodes[0], stack);
        evaluate(stack, results, errcode_l);
        if (errcode_l != err_none) {
          add_error_options err_options;
          err_options.errstring = fn_tree.co_value.text;
          err_options.charindex = fn_tree.co_value.charindex;
          err_handler->add_error(err_parser, errcode_l, err_options);
          errcode = errcode_l;
          return;
        }
        el = static_cast<offset_type>(std::lround(results[0]));
        if (fn_tree.value.rtype == pt_param) {
          if (el < 1 || el > static_cast<offset_type>(params.size())) {
            errcode = err_bad_value;
            add_error_options err_options;
            err_options.errstring = fn_tree.nodes[0].co_value.text;
            err_options.charindex = fn_tree.nodes[0].co_value.charindex;
            err_handler->add_error(err_parser, errcode, err_options);
            return;
          }
          copy_tree(params[params.size()
              - static_cast<count_type>(static_cast<offset_type>(results[0]))],
              fn_tree);
        } else if (options.dparams != nullptr) {
          if (el < 1
              || el > static_cast<offset_type>(options.dparams->size())) {
            errcode = err_bad_value;
            add_error_options err_options;
            err_options.errstring = fn_tree.nodes[0].co_value.text;
            err_options.charindex = fn_tree.nodes[0].co_value.charindex;
            err_handler->add_error(err_parser, errcode, err_options);
            return;
          }
          copy_tree((*options.dparams)[params.size()
              - static_cast<count_type>(static_cast<offset_type>(results[0]))],
              fn_tree);
        } else {
          errcode = err_bad_value;
          add_error_options err_options;
          err_options.errstring = fn_tree.co_value.text;
          err_options.charindex = fn_tree.co_value.charindex;
          err_handler->add_error(err_parser, errcode, err_options);
        }
      }

      if (fn_tree.value.rtype == pt_placed_function) {
        if (options.func != nullptr) {
          std::vector<tree_item<Num>> temp = std::move(fn_tree.nodes);
          const bool temp_associated = fn_tree.nodes_associated;
          fn_tree.nodes_associated = false;
          copy_tree(*options.func, fn_tree);
          fn_tree.nodes = std::move(temp);
          fn_tree.nodes_associated = temp_associated;
        } else {
          errcode = err_bad_value;
          add_error_options err_options;
          err_options.errstring = fn_tree.co_value.text;
          err_options.charindex = fn_tree.co_value.charindex;
          err_handler->add_error(err_parser, errcode, err_options);
        }
      }
    }

    //! Add a function to the parser
    //! @param[in] name - Name to register function with. Will be used in
    //! expressions to call the function
    //! @param[in] fn - Function to call when the expression is evaluated
    //! @param[inout] errcode - Error code from storing the function
    //! @param[in] options - Optional parameters
    void add_function(const std::string& name, parser_eval_fn<Num> fn,
        error& errcode, const parser_function_options& options = {}) {
      offset_type params;

      params = options.expected_params;

      if (options.global) {
        global_registry.add_function(name, fn, params, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      } else {
        this->registry.add_function(name, fn, params, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      }
    }

    //! Add a deferred function to the parser. A non-deferred version
    //! must be added before any stacks using the function are evaluated
    //! @param[in] name - Name to register function with. Will be used in
    //! expressions to call the function
    //! @param[inout] errcode - Error code from storing the function
    //! @param[in] options - Optional parameters
    void add_function(const std::string& name, error& errcode,
        const parser_function_options& options = {}) {
      parser_function_options deferred = options;

      deferred.defer = true;
      add_function(name, dummy<Num>, errcode, deferred);
    }

    //! Add a functor to the parser
    //! Functors are like functions but carry persistent information with them.
    //! When you call this function the functor that you supply is copied. If
    //! your functor is large enough that you don't want to make copies then
    //! use add_functor_pointer
    //! @param[in] name - Name to register function with. Will be used in
    //! expressions to call the function
    //! @param[in] functor - Functor to use when the block is accessed
    //! @param[inout] errcode - Error code from storing the function
    //! @param[in] options - Optional parameters
    void add_functor(const std::string& name,
        const eis::functor<Num>& functor, error& errcode,
        const parser_function_options& options = {}) {
      offset_type params;

      params = options.expected_params;

      if (options.global) {
        global_registry.add_functor(name, functor, params, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      } else {
        this->registry.add_functor(name, functor, params, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      }

      lock_symbol_derivative(name, errcode, {.global = options.global});
    }

    //! Add a functor to the parser using a pointer to the functor.
    //! Unlike add_functor no copy is made of the functor passed to this
    //! function the pointer is used as is. If "owns" is true then EIS assumes
    //! ownership of the pointer and it is automatically deleted when the
    //! parser wants to discard it. If the host code is maintaining the pointer
    //! itself, leave the "owns" parameter false then the parser will discard
    //! the pointer without deleting it. If you want the same functor under
    //! different names without making copies then at most only one of then
    //! should own the functor. Care must be taken because reusing the name
    //! that is associated with the ownership of the functor will invalidate
    //! all of the other names. Null pointers cause this routine to return
    //! immediately
    //! @param[in] name - Name to register function with. Will be used in
    //! expressions to call the function
    //! @param[in] functor - Functor to use when the block is accessed
    //! @param[inout] errcode - Error code from storing the function
    //! @param[in] options - Optional parameters
    void add_functor_pointer(const std::string& name,
        eis::functor<Num>* functor, error& errcode,
        const parser_functor_pointer_options& options = {}) {
      offset_type params;

      if (functor == nullptr) return;

      params = options.expected_params;

      if (options.global) {
        global_registry.add_functor_pointer(name, functor, params, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .owns = options.owns, .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      } else {
        this->registry.add_functor_pointer(name, functor, params, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .owns = options.owns, .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      }

      lock_symbol_derivative(name, errcode, {.global = options.global});
    }

    //! Set the derivative for a symbol
    //! This function specifies a derivative for a given parser symbol
    //! in terms of a valid parser expression. If the derivative has a
    //! different value depending on what the derivative is with respect to
    //! then you can specify this with the optional "wrt" parameter
    //! @param[in] symbol - Symbol to set the derivative of
    //! @param[in] deriv - Derivative parser expression for the symbol
    //! @param[inout] errcode - Error code for the operation
    //! @param[in] options - What to take the derivative with respect to and
    //! whether the symbol is in the global registry
    void set_symbol_derivative(const std::string& symbol,
        const std::string& deriv, error& errcode,
        const symbol_derivative_options& options = {}) {
      set_symbol_derivative_inner(symbol, deriv, errcode,
          {.wrt = options.wrt, .global = options.global});
    }

    //! Lock the derivative for a symbol
    //! This function locks a given derivative of a symbol as being invalid
    //! This function should be used for functions that do not have a
    //! meaningful derivative. If this is not used then all parser symbols
    //! except functors are assumed to have a derivative of zero
    //! @param[in] symbol - Symbol to lock
    //! @param[inout] errcode - Error code
    //! @param[in] options - What to take the derivative with respect to and
    //! whether the symbol is in the global registry
    void lock_symbol_derivative(const std::string& symbol, error& errcode,
        const symbol_derivative_options& options = {}) {
      std::string str;

      errcode = err_none;
      if (!options.wrt.empty()) {
        str = options.wrt;
      } else {
        str = "[***]";
      }

      if (options.global) {
        global_registry.lock_symbol_derivative(symbol, str, false, errcode);
      } else {
        this->registry.lock_symbol_derivative(symbol, str, false, errcode);
      }

      if (errcode != err_none) {
        err_handler->add_error(err_parser, errcode);
      }
    }

    //! Core inner function for setting symbolic derivatives
    void set_symbol_derivative_inner(const std::string& symbol,
        const std::string& deriv, error& errcode,
        const symbol_derivative_options& options = {}) {
      std::string str;
      eis::stack<Num> stack;

      errcode = err_none;
      if (!options.wrt.empty()) {
        str = options.wrt;
      } else {
        str = "[***]";
      }
      tokenize_inner(deriv, stack, errcode,
          {.allow_params = true, .allow_dparams = true});

      if (errcode == err_none) {
        if (options.global) {
          global_registry.set_symbol_derivative_value(symbol, str, stack,
              options.unary, errcode);
        } else {
          this->registry.set_symbol_derivative_value(symbol, str, stack,
              options.unary, errcode);
        }
      }

      if (errcode != err_none) {
        err_handler->add_error(err_parser, errcode);
      }
    }

    //! Add a variable to the parser.
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to call the function
    //! @param[in] fn - Function to call when the expression is evaluated
    //! @param[inout] errcode - Error code from storing the variable
    //! @param[in] options - Optional parameters
    void add_variable(const std::string& name, parser_eval_fn<Num> fn,
        error& errcode, const parser_variable_options& options = {}) {
      if (options.global) {
        global_registry.add_variable(name, fn, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden});
      } else {
        this->registry.add_variable(name, fn, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden});
      }
    }

    //! Add a deferred variable to the parser. A non-deferred version
    //! must be added before any stacks using the variable are evaluated
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to call the function
    //! @param[inout] errcode - Error code from storing the variable
    //! @param[in] options - Optional parameters
    void add_variable(const std::string& name, error& errcode,
        const parser_variable_options& options = {}) {
      parser_variable_options deferred = options;

      deferred.defer = true;
      add_variable(name, dummy<Num>, errcode, deferred);
    }

    //! Add a variable supplied by C code to the parser
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to call the function
    //! @param[in] fn - Function to call when the expression is evaluated
    //! @param[inout] errcode - Error code from storing the variable
    //! @param[in] options - Optional parameters
    void add_variable(const std::string& name, c_parser_eval_fn<Num> fn,
        error& errcode, const parser_variable_options& options = {}) {
      if (options.global) {
        global_registry.add_variable(name, fn, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden});
      } else {
        this->registry.add_variable(name, fn, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden});
      }
    }

    //! Add a function supplied by C code to the parser
    //! @param[in] name - Name to register function with. Will be used in
    //! expressions to call the function
    //! @param[in] fn - Function to call when the expression is evaluated
    //! @param[inout] errcode - Error code from storing the function
    //! @param[in] options - Optional parameters
    void add_function(const std::string& name, c_parser_eval_fn<Num> fn,
        error& errcode, const parser_function_options& options = {}) {
      if (options.global) {
        global_registry.add_function(name, fn, options.expected_params,
            errcode, {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      } else {
        this->registry.add_function(name, fn, options.expected_params,
            errcode, {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden,
            .can_have_text_params = options.text_params,
            .text_param_mask = options.text_param_mask,
            .is_idempotent = options.is_idempotent});
      }
    }

    //! Add a variable to the parser using a pointer
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to call the function
    //! @param[in] ptr - Variable to be associated with the variable
    //! @param[inout] errcode - Error code from storing the variable
    //! @param[in] options - Optional parameters
    void add_pointer_variable(const std::string& name, std::int32_t* ptr,
        error& errcode, const parser_variable_options& options = {}) {
      if (options.global) {
        global_registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .i32data = ptr, .description = options.description});
      } else {
        this->registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .i32data = ptr, .description = options.description});
      }
    }

    //! Add a variable to the parser using a pointer
    void add_pointer_variable(const std::string& name, std::int64_t* ptr,
        error& errcode, const parser_variable_options& options = {}) {
      if (options.global) {
        global_registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .i64data = ptr, .description = options.description,
            .hidden = options.hidden});
      } else {
        this->registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .i64data = ptr, .description = options.description,
            .hidden = options.hidden});
      }
    }

    //! Add a variable to the parser using a pointer
    void add_pointer_variable(const std::string& name, float* ptr,
        error& errcode, const parser_variable_options& options = {}) {
      if (options.global) {
        global_registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .r32data = ptr, .description = options.description,
            .hidden = options.hidden});
      } else {
        this->registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .r32data = ptr, .description = options.description,
            .hidden = options.hidden});
      }
    }

    //! Add a variable to the parser using a pointer
    void add_pointer_variable(const std::string& name, double* ptr,
        error& errcode, const parser_variable_options& options = {}) {
      if (options.global) {
        global_registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .r64data = ptr, .description = options.description,
            .hidden = options.hidden});
      } else {
        this->registry.add_variable(name, dummy<Num>, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .r64data = ptr, .description = options.description,
            .hidden = options.hidden});
      }
    }

    //! Add a constant to the parser.
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to call the function
    //! @param[in] value - Value to associate with the name
    //! @param[inout] errcode - Error code from storing the variable
    //! @param[in] options - Optional parameters
    void add_constant(const std::string& name, Num value, error& errcode,
        const parser_constant_options& options = {}) {
      if (options.global) {
        global_registry.add_constant(name, value, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden});
      } else {
        this->registry.add_constant(name, value, errcode,
            {.can_simplify = options.can_simplify,
            .cap_bits = options.cap_bits, .defer = options.defer,
            .description = options.description, .hidden = options.hidden});
      }
    }

    //! Add a deferred constant to the parser. A non-deferred version
    //! must be added before any stacks using the variable are evaluated
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to call the function
    //! @param[inout] errcode - Error code from storing the variable
    //! @param[in] options - Optional parameters
    void add_constant(const std::string& name, error& errcode,
        const parser_constant_options& options = {}) {
      parser_constant_options deferred = options;

      deferred.defer = true;
      add_constant(name, Num(0), errcode, deferred);
    }

    //! Add a constant to the parser.
    void add_integer_constant(const std::string& name, i4 value,
        error& errcode, const parser_constant_options& options = {}) {
      add_constant(name, static_cast<Num>(value), errcode, options);
    }

    //! Add a constant to the parser.
    void add_integer_constant(const std::string& name, i8 value,
        error& errcode, const parser_constant_options& options = {}) {
      add_constant(name, static_cast<Num>(value), errcode, options);
    }

    //! Add a stack variable to the parser.
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to use the stack variable
    //! @param[in] stack - Stack to associate with the name
    //! @param[inout] errcode - Error code from storing the stack variable
    //! @param[in] options - Optional parameters
    void add_stack_variable(const std::string& name,
        const eis::stack<Num>& stack, error& errcode,
        const parser_stack_variable_options& options = {}) {
      if (options.global) {
        eis::stack<Num> shared = stack;
        shared.owner = nullptr;
        global_registry.add_stack_variable(name, shared, errcode,
            {.defer = options.defer, .description = options.description,
            .hidden = options.hidden});
      } else {
        this->registry.add_stack_variable(name, stack, errcode,
            {.defer = options.defer, .description = options.description,
            .hidden = options.hidden});
      }
    }

    //! Add a stack variable from a string expression to the parser.
    //! @param[in] name - Name to associated with the stack variable
    //! @param[in] string - Parseable string to make into stack variable
    //! @param[inout] errcode - Error code from the parse and the store of the
    //! variable
    //! @param[in] options - Optional parameters
    void add_stack_variable(const std::string& name, const std::string& string,
        error& errcode, const parser_stack_variable_options& options = {}) {
      eis::stack<Num> stack;

      initialise_stack(stack);
      tokenize(string, stack, errcode);
      if (options.global) stack.owner = nullptr;
      if (options.global) {
        global_registry.add_stack_variable(name, stack, errcode,
            {.description = options.description, .hidden = options.hidden});
      } else {
        this->registry.add_stack_variable(name, stack, errcode,
            {.description = options.description, .hidden = options.hidden});
      }
      deallocate_stack(stack);
    }

    //! Add a deferred stack variable to the parser. A non-deferred version
    //! must be added before any stacks using the variable are evaluated
    //! @param[in] name - Name to register variable with. Will be used in
    //! expressions to use the stack variable
    //! @param[inout] errcode - Error code from storing the variable
    //! @param[in] options - Optional parameters
    void add_stack_variable(const std::string& name, error& errcode,
        const parser_stack_variable_options& options = {}) {
      eis::stack<Num> stack;
      parser_stack_variable_options deferred = options;

      deferred.defer = true;
      add_stack_variable(name, stack, errcode, deferred);
    }

    //! Add an emplaced function to the parser. Emplaced functions
    //! cannot be added globally or deferred
    //! @param[in] name - Name to associated with the emplaced function
    //! @param[in] def_fn - Function to be called when emplacing the function
    //! @param[inout] errcode - Error code associated with the storing of the
    //! emplaced function
    //! @param[in] options - Optional parameters
    void add_emplaced_function(const std::string& name,
        parser_late_bind_fn<Num> def_fn, error& errcode,
        const parser_emplaced_options& options = {}) {
      errcode = err_none;
      this->registry.add_emplaced_function(name, errcode,
          {.expected_parameters = options.expected_params,
          .description = options.description, .hidden = options.hidden,
          .lb_fn = def_fn});
    }

    //! Add a stack function from a string expression to the parser.
    //! @param[in] name - Name to associated with the stack function
    //! @param[in] string - Parseable string to make into stack function
    //! @param[in] expected_params - Number of parameters that the function
    //! will expect
    //! @param[inout] errcode - Error code from the parse and the store of the
    //! function
    //! @param[in] options - Optional parameters
    void add_stack_function(const std::string& name, const std::string& string,
        offset_type expected_params, error& errcode,
        const parser_stack_function_options& options = {}) {
      eis::stack<Num> stack;

      errcode = err_none;

      if (expected_params <= 0) {
        errcode = err_bad_value;
        err_handler->add_error(err_parser, errcode, {.errstring = string});
        return;
      }

      initialise_stack(stack);
      tokenize_inner(string, stack, errcode, {.allow_params = true});
      if (!options.global) stack.owner = this;
      if (options.global) {
        global_registry.add_stack_function(name, stack, expected_params,
            errcode, {.description = options.description,
            .hidden = options.hidden});
      } else {
        this->registry.add_stack_function(name, stack, expected_params,
            errcode, {.description = options.description,
            .hidden = options.hidden});
      }
      deallocate_stack(stack);
    }

    //! Add a stack emplaced function to the parser. Emplaced functions
    //! cannot be added globally or deferred
    //! @param[in] name - Name to associated with the emplaced function
    //! @param[in] def_fn - Stack function to be called when emplacing the
    //! function
    //! @param[inout] errcode - Error code associated with the storing of the
    //! emplaced function
    //! @param[in] options - Optional parameters
    void add_emplaced_stack_function(const std::string& name,
        parser_late_bind_stack_fn<Num> def_fn, error& errcode,
        const parser_emplaced_options& options = {}) {
      errcode = err_none;
      this->registry.add_emplaced_function(name, errcode,
          {.expected_parameters = options.expected_params,
          .description = options.description, .hidden = options.hidden,
          .lbs_fn = def_fn});
    }

    //! Add an emplaced variable to the parser. Emplaced variables
    //! cannot be added globally or deferred
    //! @param[in] name - Name to associated with the emplaced variable
    //! @param[in] def_fn - Function to be called when emplacing the variable
    //! @param[inout] errcode - Error code associated with the storing of the
    //! emplaced variable
    //! @param[in] options - Optional parameters
    void add_emplaced_variable(const std::string& name,
        parser_late_bind_fn<Num> def_fn, error& errcode,
        const parser_emplaced_options& options = {}) {
      this->registry.add_emplaced_function(name, errcode,
          {.expected_parameters = 0, .description = options.description,
          .hidden = options.hidden, .lb_fn = def_fn});
    }

    //! Add an emplaced function that can hold state to the parser. Emplaced
    //! functions cannot be added globally or deferred
    //! @param[in] name - Name to associated with the emplaced function
    //! @param[in] def_fn - Function to be called when emplacing the function
    //! @param[inout] errcode - Error code associated with the storing of the
    //! emplaced function
    //! @param[in] options - Optional parameters
    void add_emplaced_functor(const std::string& name,
        parser_late_bind_functor<Num> def_fn, error& errcode,
        const parser_emplaced_options& options = {}) {
      errcode = err_none;
      this->registry.add_emplaced_function(name, errcode,
          {.expected_parameters = options.expected_params,
          .description = options.description, .hidden = options.hidden,
          .lb_functor = def_fn});
    }

    //! Function to tokenize a single text chunk into a element and a
    //! co-element and populate appropriate objects with that information
    //! This function tokenizes input in normal infix maths notation
    //! It uses Dijkstra's shunting yard algorithm to convert to RPN
    //! @param[in] current - Text to parse
    //! @param[inout] iblock - Element to fill with information
    //! @param[inout] icoblock - Co-element to fill with information
    //! @param[out] cap_bits - Cap bits from evaluating this element
    //! @param[in] charindex - Character index for "current" in the original
    //! string used for error reporting
    //! @param[in] trim_charindex - Character index for current in the trimmed
    //! string used for error reporting
    //! @param[inout] output - Output stack
    //! @param[inout] err - Error code from parsing this element
    //! @param[in] filename - Filename containing error
    //! @param[in] line_number - Line number in file containing error
    //! @param[in] full_line - Full version of the line being parsed
    //! @param[in] allow_text - Should the parser allow text blocks
    //! @param[in] allow_params - Should function parameters be allowed in
    //! parsing
    //! @param[in] allow_dparams - Should function derivative parameters be
    //! allowed in parsing
    //! @param[inout] can_accept_maths_domain_errors - Does the function flag
    //! the stack as allowing maths domain errors
    void tokenize_subexpression_infix(const std::string& current,
        stack_element<Num>& iblock, stack_co_element& icoblock,
        bitmask& cap_bits, offset_type charindex, offset_type trim_charindex,
        eis::stack<Num>& output, error& err, const std::string& filename,
        offset_type line_number, const std::string& full_line, bool allow_text,
        bool allow_params, bool allow_dparams,
        bool& can_accept_maths_domain_errors) {
      stack_element<Num> block2{};
      stack_co_element coblock2{};
      bool stack_empty, str_handled;
      std::string str;

      cap_bits = 0;
      if (current[0] == '\0') return;

      // Populate the block
      load_block(current, iblock, icoblock,
          {.allow_params = allow_params, .allow_dparams = allow_dparams,
          .can_accept_maths_domain_errors = &can_accept_maths_domain_errors});
      cap_bits = icoblock.cap_bits;
      icoblock.charindex = charindex;
      icoblock.full_line_pos = trim_charindex;
      if (iblock.ptype == pt_bad) {
        err = err_not_found;
        err_handler->add_error(err_parser, err,
            {.errstring = current, .charindex = charindex,
            .filename = filename, .line_number = line_number,
            .full_line = full_line, .full_line_pos = trim_charindex});
        return;
      }

      if (iblock.ptype == pt_null) return;

      if (iblock.ptype == pt_character) {
        if (allow_text) {
          //Turn the block into a numerical constant with a value
          //of the index of the string to look up
          iblock.ptype = pt_constant;
          iblock.can_simplify = false;
          iblock.value = pt_character;
          //First check for already existing string to avoid repetition
          //Consider switching to some kind of O(ln(n)) or O(1) store
          for (count_type istr = 0;
              istr < this->string_param_store.get_size(); ++istr) {
            str_handled = this->string_param_store.get(istr, str);
            if (!str_handled) continue;
            if (eis::strcmp(str, icoblock.text.substr(1,
                icoblock.text.size() - 2))) {
              iblock.data = static_cast<Num>(istr);
              str_handled = true;
              break;
            }
            str_handled = false;
          }
          str.clear();
          if (!str_handled) {
            iblock.data = static_cast<Num>(this->string_param_store.store(
                icoblock.text.substr(1, icoblock.text.size() - 2)));
          }
          if (this->stack.stack_point > 1) {
            this->stack.entries[this->stack.stack_point - 2]
                .has_string_params = true;
          }
        } else {
          err = err | err_bad_value;
          err_handler->add_error(err_parser, err,
              {.errstring = current, .charindex = charindex,
              .filename = filename, .line_number = line_number,
              .full_line = full_line, .full_line_pos = trim_charindex});
          return;
        }
      }

      //If the function that this token is a parameter of lists the parameter
      //positions that are text then check that this token is text at those
      //positions and only at those positions
      if (this->stack.stack_point > 1 && this->brackets.stack_point > 2) {
        const bitmask mask = this->stack.co_entries[
            this->stack.stack_point - 2].text_param_mask;
        const bool is_text = iblock.ptype == pt_constant
            && iblock.value == pt_character;
        const bool supplies_value = iblock.ptype == pt_constant
            || iblock.ptype == pt_variable
            || iblock.ptype == pt_pointer_variable
            || iblock.ptype == pt_stored_variable
            || iblock.ptype == pt_function
            || iblock.ptype == pt_emplaced_function;
        if (mask != 0 && supplies_value) {
          const offset_type position = this->brackets.entries[
              this->brackets.stack_point - 2].actual_params
              + this->brackets.entries[
              this->brackets.stack_point - 1].actual_params;
          const bool marked = position <= 64
              && ((mask >> (position - 1)) & 1) != 0;
          if (is_text != marked) {
            err = err | err_wrong_parameters;
            err_handler->add_error(err_parser, err,
                {.errstring = current, .charindex = charindex,
                .filename = filename, .line_number = line_number,
                .full_line = full_line, .full_line_pos = trim_charindex});
            return;
          }
        }
      }

      //If you have previously put in two items from a stack then you can't do
      //anything else
      if (this->last_block_item_count > 1) {
        err = err | err_stack_params;
        err_handler->add_error(err_parser, err,
            {.errstring = this->last_block_text,
            .charindex = this->last_charindex, .filename = filename,
            .line_number = line_number, .full_line = full_line,
            .full_line_pos = trim_charindex});
        return;
      }

      //Character variables must be followed by either a comma or a close
      //bracket
      if (this->last_block_type == pt_character
          && !(iblock.ptype == pt_separator
          || (iblock.ptype == pt_parenthesis
          && iblock.value == paren_right_bracket))) {
        err = err | err_bad_value;
        err_handler->add_error(err_parser, err,
            {.errstring = current, .charindex = charindex,
            .filename = filename, .line_number = line_number,
            .full_line = full_line, .full_line_pos = trim_charindex});
        return;
      }

      //Functions must be followed by a left bracket
      if ((this->last_block_type == pt_function
          || this->last_block_type == pt_emplaced_function)
          && !(iblock.ptype == pt_parenthesis
          && iblock.value == paren_left_bracket)) {
        if (this->stack.co_entries[this->stack.stack_point - 1].expected_params
            == 0) {
          //Functions that take no parameters
          this->stack.entries[this->stack.stack_point - 1].actual_params = 0;
          pop_to_stack(this->stack, output);
          pop_to_null(this->brackets);
        } else {
          err = err | err_malformed;
          err_handler->add_error(err_parser, err,
              {.errstring = this->last_block_text,
              .charindex = this->last_charindex, .filename = filename,
              .line_number = line_number, .full_line = full_line,
              .full_line_pos = trim_charindex});
          return;
        }
      }

      //Open brackets must not be preceeded by a constant or variable
      if (iblock.ptype == pt_parenthesis
          && iblock.value == paren_left_bracket) {
        if (this->last_block_type == pt_variable
            || this->last_block_type == pt_pointer_variable
            || this->last_block_type == pt_constant) {
          err = err | err_bracketed_constant;
          err_handler->add_error(err_parser, err,
              {.errstring = this->last_block_text,
              .charindex = this->last_charindex, .filename = filename,
              .line_number = line_number, .full_line = full_line,
              .full_line_pos = trim_charindex});
          return;
        }
      }

      //If previous block was an operator than almost anything else is valid
      //except a separator or a right bracket. Don't need to check for
      //characters because this will be checked as a parameter to the operator
      if (this->last_block_type == pt_operator
          && (iblock.ptype == pt_separator
          || (iblock.ptype == pt_parenthesis
          && iblock.value == paren_right_bracket))) {
        err = err | err_malformed;
        err_handler->add_error(err_parser, err,
            {.errstring = this->last_block_text,
            .charindex = this->last_charindex, .filename = filename,
            .line_number = line_number, .full_line = full_line,
            .full_line_pos = trim_charindex});
      }

      //If current block is a binary operator then previous block must not be
      //a separator or a left bracket. Unary operator must be preceded by
      //either a separator, an operator, an open bracket or null
      if (iblock.ptype == pt_operator) {
        if (icoblock.expected_params == 2) {
          if (this->last_block_type == pt_separator
              || (this->last_block_type == pt_parenthesis
              && this->last_block_value == paren_left_bracket)) {
            err = err | err_malformed;
            err_handler->add_error(err_parser, err,
                {.errstring = current, .charindex = charindex,
                .filename = filename, .line_number = line_number,
                .full_line = full_line, .full_line_pos = trim_charindex});
          }
        } else { //No ternary operators so must be unary
          if (!(this->last_block_type == pt_null
              || this->last_block_type == pt_separator
              || this->last_block_type == pt_operator
              || (this->last_block_type == pt_parenthesis
              && this->last_block_value == paren_left_bracket))) {
            err = err | err_malformed;
            err_handler->add_error(err_parser, err,
                {.errstring = current, .charindex = charindex,
                .filename = filename, .line_number = line_number,
                .full_line = full_line, .full_line_pos = trim_charindex});
          }
        }
      }

      //This will be replaced if needed
      this->last_block_item_count = 1;
      output.has_deferred = output.has_deferred || icoblock.defer;

      if (iblock.ptype == pt_variable
          || iblock.ptype == pt_pointer_variable
          || iblock.ptype == pt_constant) {
        push_to_stack(output, iblock, icoblock);

      } else if (iblock.ptype == pt_stored_variable) {
        if (!icoblock.defer) {
          this->registry.copy_in_stored(
              static_cast<count_type>(iblock.value), output, err,
              this->last_block_item_count);
          if (err != err_none || (this->last_block_item_count > 1
              && this->stack.stack_point > 0)) {
            if (this->last_block_item_count > 1
                && this->stack.stack_point > 0) {
              err = err | err_stack_params;
            }
            err_handler->add_error(err_parser, err,
                {.errstring = current, .charindex = charindex,
                .filename = filename, .line_number = line_number,
                .full_line = full_line, .full_line_pos = trim_charindex});
          }
        } else {
          push_to_stack(this->stack, iblock, icoblock);
        }
      } else if (iblock.ptype == pt_parenthesis) {
        if (iblock.value == paren_left_bracket) {
          iblock.actual_params = 0;
          push_to_stack(this->stack, iblock, icoblock);
          push_to_stack(this->brackets, iblock, icoblock);
        } else {
          while (true) {
            stack_snoop(this->stack, block2, 0);
            if (block2.ptype == pt_parenthesis
                && block2.value == paren_left_bracket) {
              pop_to_null(this->stack);
              if (this->brackets.stack_point > 1) {
                if (this->last_block_type == pt_parenthesis
                    && this->last_block_value == paren_left_bracket) {
                  //No actual parameters between open and close brackets
                  this->brackets.entries[this->brackets.stack_point - 1]
                      .actual_params = this->brackets.entries[
                      this->brackets.stack_point - 1].actual_params - 1;
                }
                this->brackets.entries[this->brackets.stack_point - 2]
                    .actual_params = this->brackets.entries[
                    this->brackets.stack_point - 2].actual_params
                    + this->brackets.entries[
                    this->brackets.stack_point - 1].actual_params;
                pop_to_null(this->brackets);
              }
              // If stack isn't empty then check for function
              if (this->stack.stack_point != 0) {
                stack_snoop(this->stack, block2, 0);
                if (block2.ptype == pt_function
                    || block2.ptype == pt_emplaced_function) {
                  const count_type top = this->stack.stack_point - 1;
                  this->stack.entries[top].actual_params =
                      this->brackets.entries[
                      this->brackets.stack_point - 1].actual_params;
                  if ((this->stack.entries[top].actual_params
                      != this->stack.co_entries[top].expected_params)
                      && this->stack.co_entries[top].expected_params >= 0) {
                    err = err | err_wrong_parameters;
                    if (!this->stack.co_entries.empty()) {
                      add_error_options err_options;
                      err_options.errstring = this->stack.co_entries[top].text;
                      err_options.charindex = this->stack.co_entries[top].charindex;
                      err_options.filename = filename;
                      err_options.line_number = line_number;
                      err_options.full_line = full_line;
                      err_options.full_line_pos = trim_charindex - charindex
                          + this->stack.co_entries[top].charindex;
                      err_handler->add_error(err_parser, err, err_options);
                    } else {
                      err_handler->add_error(err_parser, err,
                          {.errstring = "{unknown}", .charindex = charindex});
                    }
                  }

                  if ((this->stack.entries[top].has_string_params
                      && !this->stack.co_entries[top].can_have_string_params)) {
                    err = err | err_text;
                    if (!this->stack.co_entries.empty()) {
                      add_error_options err_options;
                      err_options.errstring = this->stack.co_entries[top].text;
                      err_options.charindex = this->stack.co_entries[top].charindex;
                      err_options.filename = filename;
                      err_options.line_number = line_number;
                      err_options.full_line = full_line;
                      err_options.full_line_pos = trim_charindex;
                      err_handler->add_error(err_parser, err, err_options);
                    } else {
                      err_handler->add_error(err_parser, err,
                          {.errstring = "{unknown}", .charindex = charindex});
                    }
                  }
                  pop_to_stack(this->stack, output);
                  pop_to_null(this->brackets);
                }
              }
              break;
            } else {
              stack_empty = pop_to_stack(this->stack, output);
              if (stack_empty) {
                err = err | err_extra_bracket;
                err_handler->add_error(err_parser, err,
                    {.errstring = ")", .charindex = charindex,
                    .filename = filename, .line_number = line_number,
                    .full_line = full_line, .full_line_pos = trim_charindex});
                return;
              }
            }
          }
        }

      } else if (iblock.ptype == pt_function
          || iblock.ptype == pt_emplaced_function) {
        if (iblock.ptype == pt_emplaced_function) {
          output.has_emplaced = true;
        }
        push_to_stack(this->stack, iblock, icoblock);
        iblock.actual_params = 1;
        push_to_stack(this->brackets, iblock, icoblock);

      } else if (iblock.ptype == pt_separator) {
        while (true) {
          if (this->stack.stack_point == 0) {
            //This is a separator in creating a vector, so nothing to do
            //except count parameters
            this->brackets.entries[0].actual_params =
                this->brackets.entries[0].actual_params + 1;
            break;
          } else {
            stack_snoop(this->stack, block2, 0);
            if (block2.ptype != pt_parenthesis) {
              pop_to_stack(this->stack, output);
            } else {
              if (block2.value != paren_left_bracket) {
                err = err | err_malformed;
                err_handler->add_error(err_parser, err,
                    {.errstring = current, .charindex = charindex,
                    .filename = filename, .line_number = line_number,
                    .full_line = full_line, .full_line_pos = trim_charindex});
                return;
              }
              if (this->brackets.stack_point > 0) {
                this->brackets.entries[this->brackets.stack_point - 1]
                    .actual_params = this->brackets.entries[
                    this->brackets.stack_point - 1].actual_params + 1;
              }
              break;
            }
          }
        }

      } else if (iblock.ptype == pt_operator) {
        while (true) {
          if (this->stack.stack_point == 0) {
            // stack is empty, so just push operator onto stack and
            // leave loop
            push_to_stack(this->stack, iblock, icoblock);
            break;
          }
          // stack is not empty so check precedence etc.
          stack_snoop(this->stack, block2, 0, coblock2);
          if (block2.ptype != pt_operator) {
            // Previous block is not an operator so push current operator
            // to stack and leave loop
            push_to_stack(this->stack, iblock, icoblock);
            break;
          } else {
            if (icoblock.associativity == assoc_la
                || icoblock.associativity == assoc_a) {
              // Operator is full associative or left associative
              if (icoblock.precedence <= coblock2.precedence) {
                pop_to_stack(this->stack, output);
                continue;
              } else {
                push_to_stack(this->stack, iblock, icoblock);
                break;
              }
            } else {
              if (icoblock.precedence < coblock2.precedence) {
                pop_to_stack(this->stack, output);
                continue;
              } else {
                push_to_stack(this->stack, iblock, icoblock);
                break;
              }
            }
          }
        }
      }

      if (iblock.ptype != pt_null) {
        this->last_block_type = iblock.ptype;
        if (iblock.ptype == pt_constant && iblock.value == pt_character) {
          this->last_block_type = pt_character;
        }
        this->last_block_value = iblock.value;
        this->last_charindex = charindex;
        this->last_block_text = current;
      }
    }

    //! Combine multiple stacks. Optionally combine as parameters to a function
    //! or operator
    //! @param[inout] stacks - Array of stacks to combine
    //! @param[inout] output - Stack to hold the combined stack
    //! @param[out] errcode - Error code
    //! @param[in] options - Should the new stacks be appended to the existing
    //! stack and an optional string for function to use to combine the stacks
    void combine_stacks(std::vector<eis::stack<Num>>& stacks,
        eis::stack<Num>& output, error& errcode,
        const combine_stacks_options& options = {}) {
      bool should_append;
      stack_element<Num> iblock{};
      stack_co_element icoblock{};
      eis::stack<Num> functor_stack;

      errcode = err_none;

      should_append = options.append;
      if (!should_append || !output.init) initialise_stack(output);
      output.owner = nullptr;
      if (stacks.empty()) return;

      if (!options.function_str.empty()) {
        load_block(options.function_str, iblock, icoblock);
        if (iblock.ptype != pt_function && iblock.ptype != pt_operator
            && iblock.ptype != pt_emplaced_function) return;
        if (icoblock.expected_params >= 0
            && icoblock.expected_params
            != static_cast<offset_type>(stacks.size())) {
          return;
        }
      }

      for (auto& element : stacks) {
        if (!element.init) {
          errcode = err_bad_stack;
          return;
        }
        //Stacks can only be combined if they belong to the same parser
        if (element.owner != nullptr) {
          if (output.owner != nullptr && output.owner != element.owner) {
            errcode = err_bad_stack;
            return;
          }
          output.owner = element.owner;
        }
        if (element.eval_fn != nullptr || element.c_eval_fn != nullptr) {
          this->registry.result_function_to_functor_stack(element, errcode,
              functor_stack);
          append_stack(output, functor_stack);
        } else {
          append_stack(output, element);
        }
      }

      if (!options.function_str.empty()) {
        iblock.actual_params = static_cast<offset_type>(stacks.size());
        push_to_stack(output, iblock, icoblock);
      }
    }

    //! Convert a stack to an unintialised stack
    void blank_stack(eis::stack<Num>& stack) {
      deallocate_stack(stack);
    }

    //! Print all of the errors to a stream
    //! @param[inout] out - Stream to print to. The default is stdout
    void print_errors(std::ostream& out = std::cout) {
      count_type ec;

      ec = err_handler->get_error_count();
      for (count_type ierr = 0; ierr < ec; ++ierr) {
        err_handler->print_error_string(ierr, out);
      }
      err_handler->flush_errors();
    }

    //! Get the number of errors reported on this parser
    count_type get_error_count() const {
      return err_handler->get_error_count();
    }

    //! Get the error report on a specified error
    //! @param[in] index - Index of error to get report on. Must be between 0
    //! and the result of get_error_count - 1
    std::string get_error_report(count_type index) const {
      return err_handler->get_error_report(index);
    }

    //! Get error information for a single error. This contains the
    //! same information as the error report but is not formatted for use
    //! @param[in] index - Index of error to get report on. Must be between 0
    //! and the result of get_error_count - 1
    error_info get_error_info(count_type index) const {
      error_info info;

      const error_cause cause = err_handler->get_error_cause(index);
      const error_strings strings = err_handler->get_error_string(index);

      info.error_cause = cause.err_text;
      info.error_cause_location = cause.charindex;
      info.error_phase = strings.err_source;
      info.error_type = strings.err_string;
      info.error_filename = cause.filename;
      info.error_line_number = cause.line_number;

      return info;
    }

    //! Flush all of the errors in the list of errors
    void flush_errors() {
      err_handler->flush_errors();
    }

    //! Get a string representation of a stack
    //! @param[in] stack_in - Stack input
    //! @return String equivalent of the stack (RPN notation)
    std::string get_tokens(const eis::stack<Num>& stack_in) const {
      std::string str_out;

      eis::get_tokens(stack_in, str_out);

      return str_out;
    }

    //! Get a string representation of the AST version of the stack
    //! @param[in] stack_in - Stack input
    //! @param[in] options - Optional character string to format numerical
    //! literals
    //! @return String to hold the output of the stack as a dot file
    std::string visualize_stack(const eis::stack<Num>& stack_in,
        const tree_output_options& options = {}) const {
      std::string str_out;

      eis::visualise_stack(stack_in, str_out, options);

      return str_out;
    }

    //! Get a string representation of the AST version of the stack
    std::string visualise_stack(const eis::stack<Num>& stack_in,
        const tree_output_options& options = {}) const {
      return visualize_stack(stack_in, options);
    }

    //! Get a string version of a stack in infix form
    //! @param[in] stack_in - Stack input
    //! @param[in] options - Optional character string to format numerical
    //! literals
    //! @return String to hold the infix version of the stack
    std::string get_infix(const eis::stack<Num>& stack_in,
        const tree_output_options& options = {}) const {
      std::string str_out;

      eis::infix_stack(stack_in, str_out, options);

      return str_out;
    }

    //! Get a string of the text stored in the parser
    //! @param[in] index - Index of string to retrieve
    //! @param[out] str - String to retieve
    //! @param[out] errcode - Error code
    void get_text(count_type index, std::string& str, error& errcode) {
      str.clear();
      errcode = err_none;
      if (!this->string_param_store.get(index, str)) {
        errcode = err_bad_value;
      }
    }

    //! Get the number of symbols in the global registry
    count_type get_global_symbol_count() {
      return get_registry_symbol_count(global_registry);
    }

    //! Get a symbol from the global registry
    //! @param[in] index - Index of the symbol to retrieve
    std::string get_global_symbol(count_type index) {
      return get_registry_symbol(global_registry, index);
    }

    //! Get the number of symbols in the parser local registry
    count_type get_local_symbol_count() {
      return get_registry_symbol_count(this->registry);
    }

    //! Get a symbol from the parser local registry
    //! @param[in] index - Index of the symbol to retrieve
    std::string get_local_symbol(count_type index) {
      return get_registry_symbol(this->registry, index);
    }

    //! Get the number of symbols in both local and global registries
    count_type get_symbol_count() {
      return get_global_symbol_count() + get_local_symbol_count();
    }

    //! Get a symbol from either the local or the global registry
    //! @param[in] index - Index of the symbol to retrieve
    std::string get_symbol(count_type index) {
      std::string symbol;
      count_type gsc;

      gsc = get_global_symbol_count();

      if (index < gsc) {
        symbol = get_global_symbol(index);
      } else {
        symbol = get_local_symbol(index - gsc);
      }

      return symbol;
    }

    //! Get a representation of the keys in the parser as markdown
    //! @param[in] options - Title of the document and how built in functions
    //! show their names. Built in functions are given descriptions that
    //! include the name so printing the name is confusing. Specify fsn_never to
    //! never show the function name, fsn_always to always show the name and
    //! fsn_auto to automatically determine if the first characters of the
    //! description are the name
    std::string get_structure_as_markdown(
        const structure_options& options = {}) {
      std::string markdown, sname, name2;
      typecode fsn;
      bool pname, any_printed;

      if (!options.title.empty()) {
        append_string(markdown, "# " + options.title);
      }

      fsn = options.function_show_name;

      append_string(markdown, "");
      append_string(markdown, "## Functions");
      append_string(markdown, "");
      any_printed = false;
      for (count_type ct = 0; ct < get_symbol_count(); ++ct) {
        sname = get_symbol(ct);
        const symbol_info info = get_symbol_info(sname);
        const std::string& desc = info.description;
        if (info.is_hidden) continue;
        if ((info.symbol_type == pt_function
            || info.symbol_type == pt_emplaced_function)
            && info.expected_params != 0) {
          any_printed = true;
          if (symbol_needs_namespace(sname)) {
            name2 = sname;
          } else {
            const std::size_t dotloc = sname.rfind('.');
            if (dotloc != std::string::npos && dotloc < sname.size() - 1) {
              name2 = sname.substr(dotloc + 1);
            } else {
              name2 = sname;
            }
          }
          if (fsn == fsn_never) {
            pname = false;
          } else if (fsn == fsn_always) {
            pname = true;
          } else {
            if (!desc.empty()) {
              pname = true;
              if (desc.size() >= name2.size()) {
                if (desc.compare(0, name2.size(), name2) == 0) pname = false;
              }
              if (desc.size() > name2.size()) {
                if (desc[0] == '`' && desc.compare(1, name2.size(), name2) == 0)
                    pname = false;
              }
            } else {
              pname = true;
            }
          }
          if (pname) {
            if (!desc.empty()) {
              append_string(markdown, "* `" + name2 + "` - " + desc);
            } else {
              append_string(markdown, "* `" + name2 + "` - No description");
            }
          } else {
            append_string(markdown, "* " + desc);
          }
          append_string(markdown, "");
        }
      }
      if (!any_printed) append_string(markdown, "None found");

      append_string(markdown, "");
      append_string(markdown, "## Operators");
      append_string(markdown, "");
      any_printed = false;
      for (count_type ct = 0; ct < get_symbol_count(); ++ct) {
        sname = get_symbol(ct);
        const symbol_info info = get_symbol_info(sname);
        const std::string& desc = info.description;
        if (info.is_hidden) continue;
        if (info.symbol_type == pt_operator) {
          any_printed = true;
          if (symbol_needs_namespace(sname)) {
            name2 = sname;
          } else {
            const std::size_t dotloc = sname.rfind('.');
            if (dotloc != std::string::npos && dotloc < sname.size() - 1) {
              name2 = sname.substr(dotloc + 1);
            } else {
              name2 = sname;
            }
          }
          if (!desc.empty()) {
            append_string(markdown, "* `" + name2 + "` - " + desc);
          } else {
            append_string(markdown, "* `" + name2 + "` - No description");
          }
          append_string(markdown, "");
        }
      }
      if (!any_printed) append_string(markdown, "None found");

      append_string(markdown, "");
      append_string(markdown, "## Constants");
      append_string(markdown, "");
      any_printed = false;
      for (count_type ct = 0; ct < get_symbol_count(); ++ct) {
        sname = get_symbol(ct);
        const symbol_info info = get_symbol_info(sname);
        const std::string& desc = info.description;
        if (info.is_hidden) continue;
        if (info.symbol_type == pt_constant) {
          any_printed = true;
          if (symbol_needs_namespace(sname)) {
            name2 = sname;
          } else {
            const std::size_t dotloc = sname.rfind('.');
            if (dotloc != std::string::npos && dotloc < sname.size() - 1) {
              name2 = sname.substr(dotloc + 1);
            } else {
              name2 = sname;
            }
          }
          if (!desc.empty()) {
            append_string(markdown, "* `" + name2 + "` - " + desc);
          } else {
            append_string(markdown, "* `" + name2 + "` - No description");
          }
          append_string(markdown, "");
        }
      }
      if (!any_printed) append_string(markdown, "None found");

      append_string(markdown, "");
      append_string(markdown, "## Variables");
      append_string(markdown, "");
      any_printed = false;
      for (count_type ct = 0; ct < get_symbol_count(); ++ct) {
        sname = get_symbol(ct);
        const symbol_info info = get_symbol_info(sname);
        const std::string& desc = info.description;
        if (info.is_hidden) continue;
        if (info.symbol_type == pt_variable
            || info.symbol_type == pt_stored_variable
            || info.symbol_type == pt_emplaced_variable
            || info.symbol_type == pt_pointer_variable
            || ((info.symbol_type == pt_function
            || info.symbol_type == pt_emplaced_function)
            && info.expected_params == 0)) {
          any_printed = true;
          if (symbol_needs_namespace(sname)) {
            name2 = sname;
          } else {
            const std::size_t dotloc = sname.rfind('.');
            if (dotloc != std::string::npos && dotloc < sname.size() - 1) {
              name2 = sname.substr(dotloc + 1);
            } else {
              name2 = sname;
            }
          }
          if (!desc.empty()) {
            append_string(markdown, "* `" + name2 + "` - " + desc);
          } else {
            append_string(markdown, "* `" + name2 + "` - No description");
          }
          append_string(markdown, "");
        }
      }
      if (!any_printed) append_string(markdown, "None found");

      return markdown;
    }

  };

}

#endif
