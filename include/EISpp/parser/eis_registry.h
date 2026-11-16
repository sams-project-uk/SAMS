// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_REGISTRY_H
#define EIS_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "../common/eis_constants.h"
#include "../common/eis_named_store.h"
#include "../common/eis_optional.h"
#include "../common/eis_ordered_store.h"
#include "eis_parser_constants.h"
#include "eis_stack.h"

namespace eis {

  inline constexpr typecode reg_index_none = 0;
  inline constexpr typecode reg_index_stack = 1;
  inline constexpr typecode reg_index_emplaced = 2;

  //! Functor that is used to encapsulate an evaluation function
  template <typename Num>
  struct parser_result_functor {
    parser_result_function<Num> result_fn = nullptr;

    //! Operation function for functor that is used to encapsulate an
    //! evaluation function
    Num operator()(count_type, const Num*, void* host_params,
        parser<Num>* calling_parser, status& status_code,
        error& errcode) const {
      Num results[1];
      count_type result_count = 1;

      result_fn(result_count, results, host_params, calling_parser,
          status_code, errcode);
      if (result_count != 1) {
        errcode = err_stack_params;
        return Num(0);
      }
      return results[0];
    }
  };

  //! Functor that is used to encapsulate a result function supplied by C code
  template <typename Num>
  struct c_result_functor {
    c_parser_result_function<Num> result_fn = nullptr;

    //! Operation function for functor that is used to encapsulate a result
    //! function supplied by C code
    Num operator()(count_type, const Num*, void* host_params,
        parser<Num>*, status& status_code, error& errcode) const {
      Num results[1];
      int result_count = 1;

      result_fn(&result_count, results, host_params,
          reinterpret_cast<std::int64_t*>(&status_code),
          reinterpret_cast<std::int64_t*>(&errcode));
      if (result_count != 1) {
        errcode = err_stack_params;
        return Num(0);
      }
      return results[0];
    }
  };

  //! Type holding pointer to the two kinds of functions used for emplaced
  //! variables and functions
  template <typename Num>
  struct late_bind_fn_holder {
    parser_late_bind_fn<Num> lb_contents = nullptr;
    parser_late_bind_stack_fn<Num> lbs_contents = nullptr;
    parser_late_bind_functor<Num> lb_functor;
  };

  //! Type holding the information needed to construct an instance of a
  //! function, constant or variable
  template <typename Num>
  struct function_entry {
    std::string description;
    bool is_hidden = false;
    bool can_simplify = true;
    typecode index_type = reg_index_none;
    offset_type value = 0;
    typecode ptype = pt_null;
    typecode rtype = pt_null;
    typecode associativity = assoc_null;
    i4 precedence = 0;
    offset_type expected_parameters = 0;
    bitmask cap_bits = 0;
    bool defer = false;
    bool owns_functor = true;
    bool per_stack_functor = false;
    bool string_params = false;
    bitmask text_param_mask = 0;
    bool can_accept_maths_domain_errors = false;
    bool is_idempotent = false;
    offset_type deriv_values = -1;
    element_data<Num> data = Num(0);
  };

  //! Tag type marking that the derivative of a symbol is forbidden
  struct deriv_forbidden {};

  //! A derivative held for a symbol. Either a stack or the marker that the
  //! derivative is forbidden
  template <typename Num>
  using deriv_item = std::variant<eis::stack<Num>, deriv_forbidden>;

  //! Optional parameters to adding a constant
  struct add_constant_options {
    bool can_simplify = true;
    bitmask cap_bits = 0;
    bool defer = false;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding a variable
  struct add_variable_options {
    bool can_simplify = true;
    bitmask cap_bits = 0;
    bool defer = false;
    std::int32_t* i32data = nullptr;
    std::int64_t* i64data = nullptr;
    float* r32data = nullptr;
    double* r64data = nullptr;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding a function
  struct add_function_options {
    bool can_simplify = true;
    bitmask cap_bits = 0;
    bool defer = false;
    std::string description;
    bool hidden = false;
    bool can_have_text_params = false;
    //! Parameter positions that must be text. Bit 0 is the first parameter. 0
    //! means that positions are not checked
    bitmask text_param_mask = 0;
    //! Does calling the function a second time with the same parameters in the
    //! same evaluation have no further effect and return the same result
    bool is_idempotent = true;
    bool can_accept_maths_domain_errors = false;
    typecode rtype = pt_function;
  };

  //! Optional parameters to adding a functor
  struct add_functor_options {
    bool can_simplify = true;
    bitmask cap_bits = 0;
    bool defer = false;
    bool per_stack = false;
    std::string description;
    bool hidden = false;
    bool can_have_text_params = false;
    //! Parameter positions that must be text. Bit 0 is the first parameter. 0
    //! means that positions are not checked
    bitmask text_param_mask = 0;
    //! Does calling the functor a second time with the same parameters in the
    //! same evaluation have no further effect and return the same result
    bool is_idempotent = true;
  };

  //! Optional parameters to adding a functor by pointer
  struct add_functor_pointer_options {
    bool can_simplify = true;
    bitmask cap_bits = 0;
    bool defer = false;
    std::string description;
    bool hidden = false;
    bool owns = false;
    bool can_have_text_params = false;
    //! Parameter positions that must be text. Bit 0 is the first parameter. 0
    //! means that positions are not checked
    bitmask text_param_mask = 0;
    //! Does calling the functor a second time with the same parameters in the
    //! same evaluation have no further effect and return the same result
    bool is_idempotent = true;
  };

  //! Optional parameters to adding an operator
  struct add_operator_options {
    bool can_simplify = true;
    bool unary = false;
    bitmask cap_bits = 0;
    std::string description;
    bool hidden = false;
    typecode operator_id = pt_operator;
    //! Does calling the operator a second time with the same parameters in the
    //! same evaluation have no further effect and return the same result
    bool is_idempotent = true;
  };

  //! Optional parameters to adding a stack variable
  struct add_stack_variable_options {
    bool defer = false;
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding a stack function
  struct add_stack_function_options {
    std::string description;
    bool hidden = false;
  };

  //! Optional parameters to adding an emplaced function or variable
  template <typename Num>
  struct add_emplaced_options {
    offset_type expected_parameters = -1;
    std::string description;
    bool hidden = false;
    parser_late_bind_fn<Num> lb_fn = nullptr;
    parser_late_bind_stack_fn<Num> lbs_fn = nullptr;
    parser_late_bind_functor<Num> lb_functor;
  };

  //! Optional parameters to copying in a stored stack
  struct copy_in_options {
    //! Insertion point. Item at the location is replaced with the content of
    //! the requested stack. Zero means insert at end of stack with no
    //! replacement
    offset_type insert_point = 0;
    //! Should an error be returned if you are filling with a stack
    //! that unpacks to multiple items
    bool allow_multiple = true;
  };

  //! Type representing a namespace. Holds the list of all namespaces within
  //! a given namespace and the list of all functions, constants and variables
  //! stored in a given namespace
  template <typename Num>
  class parser_namespace {

    named_store<parser_namespace> namespaces;
    named_store<function_entry<Num>> generic_store;
    ordered_store<std::string> included_namespaces;

  public:

    //! Add an item to a namespace
    //! If the name contains a dot then it separates the name on the dot
    //! recursively creating new levels of namespace as it goes until it can
    //! finally store the object
    void store(const std::string& name, const function_entry<Num>& item) {
      const std::size_t dotloc = name.find('.');
      if (dotloc == std::string::npos) {
        generic_store.store(name, item);
      } else if (dotloc == 0) {
        generic_store.store(name.substr(1), item);
      } else if (dotloc == name.size() - 1) {
        generic_store.store(name.substr(0, name.size() - 1), item);
      } else {
        parser_namespace* old = namespaces.get(name.substr(0, dotloc));
        if (old != nullptr) {
          old->store(name.substr(dotloc + 1), item);
        } else {
          parser_namespace* created = new parser_namespace;
          created->store(name.substr(dotloc + 1), item);
          namespaces.hold(name.substr(0, dotloc), created);
        }
      }
    }

    //! Include a namespace in the list of namespaces to search
    //! Specifies that a given namespace should be tested automatically
    //! when an item is looked up by name. This has the effect of putting the
    //! item in the global namespace. The lookup order is currently
    //! 1) the global namespace
    //! 2) included namespaces in the order they were included in
    //! this behaviour is not guaranteed to be preserved in future releases.
    //! Note that namspaces are not recursively included by default
    //! only_final is reserved for future use
    void include_namespace(const std::string& namespace_name,
        bool only_final = true) {
      if (only_final) {
        included_namespaces.store(namespace_name);
      } else {
        offset_type dotloc = static_cast<offset_type>(namespace_name.size()) + 1;
        while (dotloc > 0) {
          const std::string prefix = namespace_name.substr(0,
              static_cast<count_type>(dotloc - 1));
          included_namespaces.store(prefix);
          const std::size_t last = prefix.rfind('.');
          dotloc = (last == std::string::npos)
              ? 0 : static_cast<offset_type>(last) + 1;
        }
      }
    }

    //! Check if a given namespace is included
    bool is_included(const std::string& namespace_name) {
      bool result = false;

      for (count_type i = 0; i < included_namespaces.get_size(); ++i) {
        const std::string* text = included_namespaces.get(i);
        if (detail::trimmed(*text) == detail::trimmed(namespace_name)) {
          result = true;
          return result;
        }
      }

      return result;

      const std::size_t dotloc = namespace_name.find('.');

      parser_namespace* gptr = namespaces.get(namespace_name);
      if (gptr != nullptr) {
        result = true;
        return result;
      }

      if (dotloc == std::string::npos || dotloc == namespace_name.size() - 1) {
        gptr = namespaces.get(namespace_name);
        result = gptr != nullptr;
      } else if (dotloc == 0) {
        result = false;
      } else {
        gptr = namespaces.get(namespace_name.substr(0, dotloc));
        if (gptr == nullptr) {
          result = false;
          return result;
        }
        result = gptr->is_included(namespace_name.substr(dotloc + 1));
      }
      return result;
    }

    //! Get an item from a namespace
    //! Items are looked up using the specified name. If the name includes dots
    //! then the string is split on the dots and is used to specify a namespace
    //! to search in. If no dots are present then the global namespace and any
    //! included namespaces are considered
    //! The lookup order is currently
    //! 1) the global namespace
    //! 2) included namespaces in the order they were included in
    //! this behaviour is not guaranteed to be preserved in future releases.
    //! Note that namspaces are not recursively included by default
    function_entry<Num>* get(const std::string& name) {
      function_entry<Num>* item = nullptr;

      const std::size_t dotloc = name.find('.');
      if (dotloc == std::string::npos) {
        item = generic_store.get(name);
        const count_type ins_count = included_namespaces.get_size();
        count_type c_ins = 0;
        while (item == nullptr && c_ins < ins_count) {
          const std::string* txt = included_namespaces.get(c_ins);
          item = get(*txt + "." + name);
          c_ins = c_ins + 1;
        }
      } else if (dotloc == name.size() - 1 || dotloc == 0) {
        item = nullptr;
      } else {
        parser_namespace* old = namespaces.get(name.substr(0, dotloc));
        if (old == nullptr) return item;
        item = old->get(name.substr(dotloc + 1));
      }
      return item;
    }

    //! Get the number of names in this and all contained namespaces
    count_type get_name_count() const {
      count_type count = generic_store.get_name_count();

      for (count_type i = 0; i < namespaces.get_name_count(); ++i) {
        count = count + namespaces.get(i)->get_name_count();
      }
      return count;
    }

    //! Get the number of names in this and all contained namespaces
    //! only_me_count is the number of names in just this namespace
    count_type get_name_count(count_type& only_me_count) const {
      only_me_count = generic_store.get_name_count();
      return get_name_count();
    }

    //! Get a name based on an index from 0 to get_name_count - 1
    //! name_prefix is a prefix for the name. If it is set then it is used,
    //! even if it is empty
    std::string get_name(count_type index,
        const optional<std::string>& name_prefix = {}) const {
      const count_type generic_count = generic_store.get_name_count();
      const std::string& prefix_text = name_prefix;
      std::string name;

      if (index < generic_count) {
        const std::string n_name = generic_store.get_name(index);
        if (name_prefix.is_set()) {
          name = prefix_text + "." + n_name;
        } else {
          name = n_name;
        }
      } else {
        count_type iindex = index - generic_count;

        for (count_type i = 0; i < namespaces.get_name_count(); ++i) {
          parser_namespace* gptr = namespaces.get(i);
          const std::string rname = namespaces.get_name(i);
          std::string pname;
          if (name_prefix.is_set()) {
            pname = prefix_text + "." + rname;
          } else {
            pname = rname;
          }
          if (iindex < gptr->get_name_count()) {
            name = gptr->get_name(iindex, pname);
            break;
          } else {
            iindex = iindex - gptr->get_name_count();
          }
        }
      }
      return name;
    }
  };

  //! Registry for all valid functions, constants and variables for a given
  //! parser. Stores all variables except unary operators in a namespace object
  template <typename Num>
  class registry {

    parser_namespace<Num> stored_items;
    //! Contains named unary operators
    named_store<function_entry<Num>> uop_table;
    ordered_store<eis::stack<Num>> stack_variable_registry;
    //! Contains deferred stacks
    ordered_store<late_bind_fn_holder<Num>> emplaced_registry;
    //! Contains functors
    ordered_store<functor<Num>> functor_registry;
    //! Contains stack functions
    ordered_store<eis::stack<Num>> stack_function_registry;
    //! Contains derivatives
    ordered_store<named_store<deriv_item<Num>>> deriv_function_registry;

    //! Inner function to deal with emplaced functions
    void add_emplaced_func_holder(const std::string& name,
        const late_bind_fn_holder<Num>& holder, error&,
        const add_emplaced_options<Num>& options) {
      function_entry<Num> temp;
      function_entry<Num>* temp_ptr = stored_items.get(name);
      count_type index;

      if (temp_ptr != nullptr) {
        if (temp_ptr->index_type == reg_index_emplaced) {
          index = emplaced_registry.store(holder, temp_ptr->value);
        } else {
          temp_ptr = &temp;
          index = emplaced_registry.store(holder);
        }
      } else {
        temp_ptr = &temp;
        index = emplaced_registry.store(holder);
      }

      temp_ptr->expected_parameters = options.expected_parameters;
      if (!options.description.empty()) {
        temp_ptr->description = options.description;
      }
      temp_ptr->is_hidden = options.hidden;
      temp_ptr->ptype = pt_emplaced_function;
      temp_ptr->rtype = pt_emplaced_function;
      temp_ptr->index_type = reg_index_emplaced;
      temp_ptr->value = static_cast<offset_type>(index);
      temp_ptr->can_simplify = false;

      create_deriv_store(*temp_ptr);
      stored_items.store(name, *temp_ptr);
    }

    //! Create the derivative store for a function entry
    void create_deriv_store(function_entry<Num>& fn_entry) {
      fn_entry.deriv_values = static_cast<offset_type>(
          deriv_function_registry.store(named_store<deriv_item<Num>>()));
    }

    //! Get the functor that calls the result function of a stack, creating it
    //! if it does not exist
    eis::functor<Num>* get_result_functor(const eis::stack<Num>& stack_in) {
      for (count_type i = 0; i < functor_registry.get_size(); ++i) {
        eis::functor<Num>* gptr = functor_registry.get(i);
        if (stack_in.eval_fn != nullptr) {
          const auto* target =
              gptr->template target<parser_result_functor<Num>>();
          if (target != nullptr && target->result_fn == stack_in.eval_fn) {
            return gptr;
          }
        } else {
          const auto* target = gptr->template target<c_result_functor<Num>>();
          if (target != nullptr && target->result_fn == stack_in.c_eval_fn) {
            return gptr;
          }
        }
      }

      count_type index;
      if (stack_in.eval_fn != nullptr) {
        parser_result_functor<Num> created;
        created.result_fn = stack_in.eval_fn;
        index = functor_registry.store(eis::functor<Num>(created));
      } else {
        c_result_functor<Num> created;
        created.result_fn = stack_in.c_eval_fn;
        index = functor_registry.store(eis::functor<Num>(created));
      }
      return functor_registry.get(index);
    }

  public:

    //! Tell this registry object's namespace store to include a namespace
    void include_namespace(const std::string& namespace_name) {
      stored_items.include_namespace(namespace_name);
    }

    //! Check if the registry's namespace includes cover a given namespace
    bool is_included(const std::string& namespace_name) {
      if (namespace_name.find('.') == std::string::npos) return true;
      return stored_items.is_included(namespace_name);
    }

    //! Add a constant to the list of known names for this registry
    void add_constant(const std::string& name, Num value, error&,
        const add_constant_options& options = {}) {
      function_entry<Num> temp;

      temp.ptype = pt_constant;
      temp.rtype = pt_constant;
      temp.data = value;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      temp.defer = options.defer;
      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;

      stored_items.store(name, temp);
    }

    //! Add a variable to the list of known names for this registry
    //! fn is the eval function to be called when evaluating
    void add_variable(const std::string& name, parser_eval_fn<Num> fn, error&,
        const add_variable_options& options = {}) {
      function_entry<Num> temp;

      if (options.i32data != nullptr || options.i64data != nullptr
          || options.r32data != nullptr || options.r64data != nullptr) {
        temp.ptype = pt_variable;
        temp.rtype = pt_pointer_variable;
        temp.data = std::monostate();

        if (options.i32data != nullptr) temp.data = options.i32data;
        if (options.i64data != nullptr) temp.data = options.i64data;
        if (options.r32data != nullptr) temp.data = options.r32data;
        if (options.r64data != nullptr) temp.data = options.r64data;
      } else {
        temp.ptype = pt_variable;
        temp.rtype = pt_variable;
        temp.data = fn;
      }

      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      temp.defer = options.defer;

      stored_items.store(name, temp);
    }

    //! Add a function to the list of known names for this registry
    //! fn is the eval function to be called when evaluating
    //! expected_parameters is the expected number of parameters for this
    //! function. If <0 function will accept any number of parameters and fn
    //! will be called with the number actually provided (variadic function)
    void add_function(const std::string& name, parser_eval_fn<Num> fn,
        offset_type expected_parameters, error&,
        const add_function_options& options = {}) {
      function_entry<Num> temp;

      temp.data = fn;
      temp.ptype = pt_function;
      temp.rtype = options.rtype;
      temp.expected_parameters = expected_parameters;
      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      temp.defer = options.defer;
      temp.string_params = options.can_have_text_params;
      temp.text_param_mask = options.text_param_mask;
      temp.can_accept_maths_domain_errors =
          options.can_accept_maths_domain_errors;
      temp.is_idempotent = options.is_idempotent;
      create_deriv_store(temp);
      stored_items.store(name, temp);
    }

    //! Add a variable supplied by C code to the list of known names for this
    //! registry
    void add_variable(const std::string& name, c_parser_eval_fn<Num> fn,
        error&, const add_variable_options& options = {}) {
      function_entry<Num> temp;

      temp.ptype = pt_variable;
      temp.rtype = pt_variable;
      temp.data = fn;

      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      temp.defer = options.defer;

      stored_items.store(name, temp);
    }

    //! Add a function supplied by C code to the list of known names for this
    //! registry
    void add_function(const std::string& name, c_parser_eval_fn<Num> fn,
        offset_type expected_parameters, error&,
        const add_function_options& options = {}) {
      function_entry<Num> temp;

      temp.data = fn;
      temp.ptype = pt_function;
      temp.rtype = options.rtype;
      temp.expected_parameters = expected_parameters;
      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      temp.defer = options.defer;
      temp.string_params = options.can_have_text_params;
      temp.text_param_mask = options.text_param_mask;
      temp.can_accept_maths_domain_errors =
          options.can_accept_maths_domain_errors;
      temp.is_idempotent = options.is_idempotent;
      create_deriv_store(temp);
      stored_items.store(name, temp);
    }

    //! Add a functor to the list of known names for this registry
    //! The functor is copied
    void add_functor(const std::string& name, const eis::functor<Num>& functor,
        offset_type expected_parameters, error&,
        const add_functor_options& options = {}) {
      function_entry<Num> temp;

      const count_type index = functor_registry.store(functor);
      temp.data = functor_registry.get(index);
      temp.owns_functor = true;
      temp.per_stack_functor = options.per_stack;
      temp.ptype = pt_function;
      temp.rtype = pt_functor;
      temp.expected_parameters = expected_parameters;
      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      temp.defer = options.defer;
      temp.string_params = options.can_have_text_params;
      temp.text_param_mask = options.text_param_mask;
      temp.is_idempotent = options.is_idempotent;

      create_deriv_store(temp);
      stored_items.store(name, temp);
    }

    //! Add a functor to the list of known names for this registry using a
    //! pointer to the functor. options.owns is whether the registry owns the
    //! functor under this name
    void add_functor_pointer(const std::string& name,
        eis::functor<Num>* functor, offset_type expected_parameters, error&,
        const add_functor_pointer_options& options = {}) {
      function_entry<Num> temp;

      temp.data = functor;
      temp.owns_functor = options.owns;
      if (temp.owns_functor) functor_registry.hold(functor, -1, true);
      temp.ptype = pt_function;
      temp.rtype = pt_functor;
      temp.expected_parameters = expected_parameters;
      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      temp.defer = options.defer;
      temp.string_params = options.can_have_text_params;
      temp.text_param_mask = options.text_param_mask;
      temp.is_idempotent = options.is_idempotent;

      create_deriv_store(temp);
      stored_items.store(name, temp);
    }

    //! Add an operator to the list of known names for this registry
    //! Currently this is not exposed to the host code
    void add_operator(const std::string& name, parser_eval_fn<Num> fn,
        typecode associativity, i4 precedence, error&,
        const add_operator_options& options = {}) {
      function_entry<Num> temp;

      temp.data = fn;
      temp.ptype = pt_operator;
      temp.rtype = options.operator_id;
      temp.associativity = associativity;
      temp.precedence = precedence;
      temp.can_simplify = options.can_simplify;
      temp.cap_bits = options.cap_bits;
      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.is_idempotent = options.is_idempotent;

      if (!options.unary) {
        temp.expected_parameters = 2;
      } else {
        temp.expected_parameters = 1;
      }

      create_deriv_store(temp);

      if (!options.unary) {
        stored_items.store(name, temp);
      } else {
        uop_table.store(name, temp);
      }
    }

    //! Add a stack variable to the list of known names for this registry
    void add_stack_variable(const std::string& name,
        const eis::stack<Num>& stack, error&,
        const add_stack_variable_options& options = {}) {
      function_entry<Num> temp;
      function_entry<Num>* temp_ptr = stored_items.get(name);
      count_type index;

      if (temp_ptr != nullptr) {
        if (temp_ptr->index_type == reg_index_stack) {
          index = stack_variable_registry.store(stack, temp_ptr->value);
        } else {
          temp_ptr = &temp;
          index = stack_variable_registry.store(stack);
        }
      } else {
        temp_ptr = &temp;
        index = stack_variable_registry.store(stack);
      }

      temp_ptr->ptype = pt_stored_variable;
      temp_ptr->rtype = pt_stored_variable;
      temp_ptr->index_type = reg_index_stack;
      temp_ptr->value = static_cast<offset_type>(index);
      temp_ptr->cap_bits = stack.cap_bits;
      temp_ptr->defer = options.defer;
      if (!options.description.empty()) {
        temp_ptr->description = options.description;
      }
      temp_ptr->is_hidden = options.hidden;

      stored_items.store(name, *temp_ptr);
    }

    //! Add a stack function to the list of known names for this registry
    void add_stack_function(const std::string& name,
        const eis::stack<Num>& stack, offset_type expected_parameters, error&,
        const add_stack_function_options& options = {}) {
      function_entry<Num> temp;

      const count_type index = stack_function_registry.store(stack);
      temp.ptype = pt_function;
      temp.rtype = pt_stack_function;
      temp.expected_parameters = expected_parameters;
      if (!options.description.empty()) temp.description = options.description;
      temp.is_hidden = options.hidden;
      temp.can_simplify = false;
      temp.cap_bits = stack.cap_bits;
      temp.value = static_cast<offset_type>(index);

      stored_items.store(name, temp);
    }

    //! Get a stack function by its id
    void get_stack_function(count_type id, eis::stack<Num>& stack,
        error& errcode) {
      errcode = err_bad_value;
      eis::stack<Num>* sptr = stack_function_registry.get(id);
      if (sptr == nullptr) return;
      errcode = err_none;

      deallocate_stack(stack);
      initialise_stack(stack);
      stack = *sptr;
    }

    //! Add an emplaced function or variable to the list of known names for
    //! this registry
    //! emplaced variables are implemented as zero parameter functions
    void add_emplaced_function(const std::string& name, error& errcode,
        const add_emplaced_options<Num>& options = {}) {
      late_bind_fn_holder<Num> holder;

      holder.lb_contents = options.lb_fn;
      holder.lbs_contents = options.lbs_fn;
      holder.lb_functor = options.lb_functor;

      add_emplaced_func_holder(name, holder, errcode, options);
    }

    //! Fill a parser block used by the Shunting Yard Algorithm
    //! name is the name extracted from the parsed string. This will be looked
    //! up. unary_ops is whether it is possible that name could describe a
    //! unary operator
    void fill_block(const std::string& name, stack_element<Num>& block_in,
        stack_co_element& coblock_in, bool unary_ops) {
      function_entry<Num>* temp = nullptr;
      if (unary_ops) temp = uop_table.get(name);
      if (temp == nullptr) temp = stored_items.get(name);

      if (temp != nullptr) {
        block_in.ptype = temp->ptype;
        block_in.rtype = temp->rtype;
        coblock_in.cap_bits = temp->cap_bits;
        coblock_in.associativity = temp->associativity;
        coblock_in.precedence = temp->precedence;
        coblock_in.expected_params = temp->expected_parameters;
        coblock_in.defer = temp->defer;
        coblock_in.can_have_string_params = temp->string_params;
        coblock_in.can_accept_maths_domain_errors =
            temp->can_accept_maths_domain_errors;
        coblock_in.text_param_mask = temp->text_param_mask;
        coblock_in.deriv_values = temp->deriv_values;
        coblock_in.is_idempotent = temp->is_idempotent;
        block_in.actual_params = temp->expected_parameters;
        block_in.can_simplify = temp->can_simplify && !temp->defer;
        block_in.data = temp->data;
        block_in.value = temp->value;
      } else {
        block_in.ptype = pt_bad;
        block_in.rtype = pt_bad;
      }
    }

    //! Fill a parser block used by the Shunting Yard Algorithm
    //! This version also returns the description and whether the symbol is
    //! hidden
    void fill_block(const std::string& name, stack_element<Num>& block_in,
        stack_co_element& coblock_in, bool unary_ops,
        std::string& description, bool& hidden) {
      description.clear();
      hidden = false;

      function_entry<Num>* temp = nullptr;
      if (unary_ops) temp = uop_table.get(name);
      if (temp == nullptr) temp = stored_items.get(name);

      if (temp != nullptr) {
        block_in.ptype = temp->ptype;
        block_in.rtype = temp->rtype;
        coblock_in.cap_bits = temp->cap_bits;
        coblock_in.associativity = temp->associativity;
        coblock_in.precedence = temp->precedence;
        coblock_in.expected_params = temp->expected_parameters;
        coblock_in.defer = temp->defer;
        coblock_in.can_have_string_params = temp->string_params;
        coblock_in.can_accept_maths_domain_errors =
            temp->can_accept_maths_domain_errors;
        coblock_in.text_param_mask = temp->text_param_mask;
        coblock_in.deriv_values = temp->deriv_values;
        coblock_in.is_idempotent = temp->is_idempotent;
        block_in.actual_params = temp->expected_parameters;
        block_in.can_simplify = temp->can_simplify && !temp->defer;
        block_in.data = temp->data;
        block_in.value = temp->value;
        if (!temp->description.empty()) description = temp->description;
        hidden = temp->is_hidden;
      } else {
        block_in.ptype = pt_bad;
        block_in.rtype = pt_bad;
      }
    }

    //! Take a stack that has a result function bound to it and convert it to
    //! a functor that calls the same function
    //! The functor is built into output_stack. If append is true then the
    //! functor is added to the existing contents of output_stack
    void result_function_to_functor_stack(eis::stack<Num>& stack_in,
        error& errcode, eis::stack<Num>& output_stack, bool append = false) {
      output_stack = eis::stack<Num>();

      errcode = err_none;

      //Shouldn't happen but fall back gracefully
      if (stack_in.eval_fn == nullptr && stack_in.c_eval_fn == nullptr) {
        output_stack = stack_in;
        return;
      }

      eis::stack<Num>& res_stack = output_stack;

      eis::functor<Num>* eval_functor = get_result_functor(stack_in);
      stack_element<Num> iblock;
      stack_co_element icoblock;
      iblock.ptype = pt_function;
      iblock.rtype = pt_functor;
      iblock.data = eval_functor;
      iblock.actual_params = 0;
      iblock.can_simplify = false;
      icoblock.text = stack_in.full_line;

      if (!append) {
        initialise_stack(res_stack);
      } else {
        res_stack.eval_fn = nullptr;
        res_stack.c_eval_fn = nullptr;
      }
      push_to_stack(res_stack, iblock, icoblock);
    }

    //! Take a stack that has a result function bound to it and convert it to
    //! a functor that calls the same function
    //! The functor is built into stack_in. If append is true then the
    //! functor is added to the existing contents of stack_in
    void result_function_to_functor_stack(eis::stack<Num>& stack_in,
        error& errcode, bool append = false) {
      errcode = err_none;

      //Shouldn't happen but fall back gracefully
      if (stack_in.eval_fn == nullptr && stack_in.c_eval_fn == nullptr) return;

      eis::stack<Num>& res_stack = stack_in;

      eis::functor<Num>* eval_functor = get_result_functor(stack_in);
      stack_element<Num> iblock;
      stack_co_element icoblock;
      iblock.ptype = pt_function;
      iblock.rtype = pt_functor;
      iblock.data = eval_functor;
      iblock.actual_params = 0;
      iblock.can_simplify = false;
      icoblock.text = stack_in.full_line;

      if (!append) {
        initialise_stack(res_stack);
      } else {
        res_stack.eval_fn = nullptr;
        res_stack.c_eval_fn = nullptr;
      }
      push_to_stack(res_stack, iblock, icoblock);
    }

    //! Copy in a stack variable to a given point in an existing stack
    //! either append at the end or at the specified insertion point
    //! item_count is the number of items to be inserted. This is not the
    //! same as the number of entries in the stack but is the number of items
    //! that the stack evaluates to
    void copy_in_stored(count_type index, eis::stack<Num>& output,
        error& errcode, count_type& item_count,
        const copy_in_options& options = {}) {
      errcode = err_none;

      eis::stack<Num>* temp = stack_variable_registry.get(index);
      eis::stack<Num> filled;
      if (temp != nullptr) {
        if (temp->eval_fn != nullptr || temp->c_eval_fn != nullptr) {
          error err;
          result_function_to_functor_stack(*temp, err, filled);
          if (err != err_none) return;
          temp = &filled;
        }
        if (!options.allow_multiple && temp->params > 1) {
          errcode = err_stack_params;
          return;
        }
        item_count = temp->params;
        if (options.insert_point != 0) {
          replace_element(output, *temp, options.insert_point);
        } else {
          append_stack(output, *temp);
        }
        output.has_emplaced = output.has_emplaced || temp->has_emplaced;
        output.has_deferred = output.has_deferred || temp->has_deferred;
      }
    }

    //! Copy in a stack variable to a given point in an existing stack
    //! either append at the end or at the specified insertion point
    void copy_in_stored(count_type index, eis::stack<Num>& output,
        error& errcode, const copy_in_options& options = {}) {
      errcode = err_none;

      eis::stack<Num>* temp = stack_variable_registry.get(index);
      eis::stack<Num> filled;
      if (temp != nullptr) {
        if (temp->eval_fn != nullptr || temp->c_eval_fn != nullptr) {
          error err;
          result_function_to_functor_stack(*temp, err, filled);
          if (err != err_none) return;
          temp = &filled;
        }
        if (!options.allow_multiple && temp->params > 1) {
          errcode = err_stack_params;
          return;
        }
        if (options.insert_point != 0) {
          replace_element(output, *temp, options.insert_point);
        } else {
          append_stack(output, *temp);
        }
        output.has_emplaced = output.has_emplaced || temp->has_emplaced;
        output.has_deferred = output.has_deferred || temp->has_deferred;
      }
    }

    //! Get the stored emplacement functions by index
    late_bind_fn_holder<Num> get_stored_emplacement(count_type index) {
      late_bind_fn_holder<Num>* holder = emplaced_registry.get(index);
      if (holder == nullptr) return late_bind_fn_holder<Num>();
      return *holder;
    }

    //! Get the number of names in this registry
    count_type get_name_count() const {
      return stored_items.get_name_count();
    }

    //! Get a name specifying an index from 0 to get_name_count - 1
    std::string get_name(count_type index) const {
      return stored_items.get_name(index);
    }

    //! Set the derivative for a symbol
    void set_symbol_derivative_value(const std::string& symbol,
        const std::string& wrt, const eis::stack<Num>& stack, bool unary_ops,
        error& errcode) {
      errcode = err_none;

      function_entry<Num>* temp = nullptr;
      if (unary_ops) temp = uop_table.get(symbol);
      if (temp == nullptr) temp = stored_items.get(symbol);

      if (temp == nullptr) {
        errcode = err_not_found;
        return;
      }

      named_store<deriv_item<Num>>* store = nullptr;
      if (temp->deriv_values >= 0) {
        store = deriv_function_registry.get(
            static_cast<count_type>(temp->deriv_values));
      }

      if (store == nullptr) {
        temp->deriv_values = static_cast<offset_type>(
            deriv_function_registry.store(named_store<deriv_item<Num>>()));
        store = deriv_function_registry.get(
            static_cast<count_type>(temp->deriv_values));
      }

      store->store(wrt, deriv_item<Num>(stack));
    }

    //! Lock the derivative for a symbol
    void lock_symbol_derivative(const std::string& symbol,
        const std::string& wrt, bool unary_ops, error& errcode) {
      errcode = err_none;

      function_entry<Num>* temp = nullptr;
      if (unary_ops) temp = uop_table.get(symbol);
      if (temp == nullptr) temp = stored_items.get(symbol);

      if (temp == nullptr) {
        errcode = err_not_found;
        return;
      }

      named_store<deriv_item<Num>>* store = nullptr;
      if (temp->deriv_values >= 0) {
        store = deriv_function_registry.get(
            static_cast<count_type>(temp->deriv_values));
      }

      if (store == nullptr) {
        temp->deriv_values = static_cast<offset_type>(
            deriv_function_registry.store(named_store<deriv_item<Num>>()));
        store = deriv_function_registry.get(
            static_cast<count_type>(temp->deriv_values));
      }

      store->store(wrt, deriv_item<Num>(deriv_forbidden()));
    }

    //! Get the derivative for a symbol
    //! Returns a null pointer if there is no derivative stack. locked is
    //! whether the derivative is locked
    eis::stack<Num>* get_symbol_derivative_value(offset_type index, error,
        const std::string& wrt, bool& locked) {
      locked = false;

      named_store<deriv_item<Num>>* store = deriv_function_registry.get(
          static_cast<count_type>(index));
      if (store == nullptr) return nullptr;

      deriv_item<Num>* item = store->get(wrt);
      if (item == nullptr) item = store->get("[***]");
      if (item == nullptr) return nullptr;
      if (std::holds_alternative<deriv_forbidden>(*item)) {
        locked = true;
        return nullptr;
      }
      return std::get_if<eis::stack<Num>>(item);
    }

    //! Get the derivative for a symbol
    //! Returns a null pointer if there is no derivative stack. wrt empty means
    //! that the derivative is not with respect to a named variable
    eis::stack<Num>* get_symbol_derivative_value(offset_type index, error,
        const std::string& wrt = "") {
      named_store<deriv_item<Num>>* store = deriv_function_registry.get(
          static_cast<count_type>(index));
      if (store == nullptr) return nullptr;

      deriv_item<Num>* item = store->get(wrt);
      if (item == nullptr) item = store->get("[***]");
      if (item == nullptr) return nullptr;
      return std::get_if<eis::stack<Num>>(item);
    }
  };

}

#endif
