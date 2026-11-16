// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_TREE_H
#define EIS_TREE_H

#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../common/eis_constants.h"
#include "../common/eis_error.h"
#include "../common/eis_utils.h"
#include "eis_core_functions.h"
#include "eis_eval_stack.h"
#include "eis_parser_constants.h"
#include "eis_stack.h"

namespace eis {

  namespace detail {

    //! Copy of a string with the trailing blanks removed
    inline std::string trim_copy(std::string_view str) {
      return std::string(trimmed(str));
    }

    //! Write a number using a printf style format and remove the trailing
    //! blanks
    template <typename Num>
    std::string format_number(const std::string& format, Num value) {
      char buffer[100];

      std::snprintf(buffer, sizeof(buffer), format.c_str(),
          static_cast<double>(value));
      return trim_copy(buffer);
    }

    //! Write an integer as a node identifier of five digits with leading zeros
    inline std::string node_id(count_type id) {
      const std::string digits = std::to_string(id);
      if (digits.size() >= 5) return digits;
      return std::string(5 - digits.size(), '0') + digits;
    }

  }

  //! Type representing a parsed string stored as an AST rather than as a stack
  //! nodes_associated is whether nodes is in use, even if it holds no nodes
  template <typename Num>
  struct tree_item {
    stack_element<Num> value;
    stack_co_element co_value;
    std::vector<tree_item> nodes;
    bool nodes_associated = false;
  };

  //! Optional parameters to simplifying a stack
  template <typename Num>
  struct simplify_stack_options {
    //! Error handler object. Null means no error reporting
    error_handler* err_handler = nullptr;
    //! Stack to hold result of simplified stack. Null means store the
    //! simplified stack in the stack that is simplified
    eis::stack<Num>* stack_out = nullptr;
    //! Report maths domain errors during simplification. If false then an
    //! expression that fails with a maths domain error is replaced by a
    //! function that raises the error when it is evaluated
    bool maths_fail_quickly = false;
    //! The parser that the stack belongs to. Passed to the functions that are
    //! called when simplifying. Null means no parser
    parser<Num>* host_parser = nullptr;
  };

  //! Optional parameters to simplifying a tree
  template <typename Num>
  struct simplify_tree_options {
    //! Error handler object. Null means no error reporting
    error_handler* err_handler = nullptr;
    std::string filename;
    offset_type line_number = -1;
    std::string full_line;
    //! Report maths domain errors during simplification. If false then an
    //! expression that fails with a maths domain error is replaced by a
    //! function that raises the error when it is evaluated
    bool maths_fail_quickly = false;
    //! The parser that the tree belongs to. Passed to the functions that are
    //! called when simplifying. Null means no parser
    parser<Num>* host_parser = nullptr;
  };

  //! Optional parameters to producing a textual representation of a stack
  struct tree_output_options {
    //! Format string for literal numbers. Empty means use the text of the
    //! literal
    std::string nformat;
  };

  //! Convert a stack to an AST recursively
  //! @param[in] stack - stack that is being converted
  //! @param[inout] curindex - stack position that is being considered
  //! @param[inout] current - Current tree node that is being built
  template <typename Num>
  void build_node(const eis::stack<Num>& stack, count_type& curindex,
      tree_item<Num>& current) {
    curindex = curindex - 1;
    //Jumps are not nodes of the tree
    while (stack.entries[curindex].ptype == pt_jump_conditional
        || stack.entries[curindex].ptype == pt_jump_unconditional) {
      curindex = curindex - 1;
    }
    element_assign(current.value, stack.entries[curindex]);
    if (current.value.rtype == pt_control_if_join) {
      current.value.rtype = pt_control_if;
      current.value.data = parser_eval_fn<Num>(parser_if<Num>);
    }
    if (!stack.co_entries.empty()) {
      current.co_value = stack.co_entries[curindex];
    }

    if (current.value.ptype == pt_function
        || current.value.ptype == pt_operator
        || current.value.ptype == pt_emplaced_function) {
      const count_type pcount =
          static_cast<count_type>(current.value.actual_params);
      current.nodes.resize(pcount);
      current.nodes_associated = true;
      for (count_type iparam = 0; iparam < pcount; ++iparam) {
        build_node(stack, curindex, current.nodes[iparam]);
      }
    }
  }

  template <typename Num>
  void copy_tree(const tree_item<Num>& node_source,
      tree_item<Num>& node_dest) {
    node_dest.value = node_source.value;
    node_dest.co_value = node_source.co_value;
    node_dest.nodes.clear();
    node_dest.nodes_associated = false;

    if (node_source.nodes_associated) {
      node_dest.nodes.resize(node_source.nodes.size());
      node_dest.nodes_associated = true;
      for (count_type i = 0; i < node_source.nodes.size(); ++i) {
        copy_tree(node_source.nodes[i], node_dest.nodes[i]);
      }
    }
  }

  namespace detail {

    //! Is a node a literal number that can be moved and combined with other
    //! literal numbers
    template <typename Num>
    bool is_hoistable_constant(const tree_item<Num>& node) {
      return node.value.ptype == pt_constant && node.value.can_simplify;
    }

    //! Name that a binary operator is registered under. Empty if the type is
    //! not one of the binary + - * /
    inline const char* operator_name(typecode rtype) {
      if (rtype == pt_op_plus) return "+";
      if (rtype == pt_op_minus) return "-";
      if (rtype == pt_op_multiply) return "*";
      if (rtype == pt_op_divide) return "/";
      return "";
    }

    //! Get the element and co element of a binary operator. The operator is
    //! taken from the first of the nodes that holds it. If none of them do then
    //! it is taken from the global registry
    //! @return Whether the operator was found
    template <typename Num>
    bool find_operator(typecode rtype,
        std::initializer_list<const tree_item<Num>*> sources,
        stack_element<Num>& value, stack_co_element& co_value) {
      for (const tree_item<Num>* source : sources) {
        if (source->value.rtype == rtype) {
          value = source->value;
          co_value = source->co_value;
          return true;
        }
      }

      const std::string name = operator_name(rtype);
      if (name.empty()) return false;
      value = stack_element<Num>();
      co_value = stack_co_element();
      parser<Num>::global_registry.fill_block(name, value, co_value, false);
      if (value.ptype != pt_operator) return false;
      co_value.text = name;
      co_value.registry_type = registry_global;
      return true;
    }

    //! Can a node be merged with an identical node that appears elsewhere in
    //! an expression. Variables can because their value is fixed during an
    //! evaluation. Functions and operators can if they are marked as
    //! idempotent and all of their parameters can be merged. Nodes without
    //! text have lost the information needed to tell them apart
    template <typename Num>
    bool is_mergeable(const tree_item<Num>& node) {
      if (node.co_value.text.empty()) return false;

      switch (node.value.ptype) {
        case pt_variable:
          return true;
        case pt_constant:
          return node.value.can_simplify;
        case pt_function:
        case pt_operator:
          if (!node.co_value.is_idempotent) return false;
          for (const tree_item<Num>& child : node.nodes) {
            if (!is_mergeable(child)) return false;
          }
          return true;
        default:
          return false;
      }
    }

    //! Are two trees the same expression. Nodes are the same if they are the
    //! same kind of symbol from the same registry with the same name
    template <typename Num>
    bool trees_equal(const tree_item<Num>& a, const tree_item<Num>& b) {
      if (a.value.ptype != b.value.ptype || a.value.rtype != b.value.rtype
          || a.value.value != b.value.value
          || a.co_value.registry_type != b.co_value.registry_type
          || a.co_value.text != b.co_value.text
          || a.nodes.size() != b.nodes.size()) return false;

      for (count_type i = 0; i < a.nodes.size(); ++i) {
        if (!trees_equal(a.nodes[i], b.nodes[i])) return false;
      }
      return true;
    }

  }

  //! Reorder a node that is a binary +, -, * or / so that two constants that
  //! are in different parts of a chain of these operators end up in a single
  //! subtree. The simplifier then combines them. For example (x + 1) + 2
  //! becomes x + (1 + 2) and 2 * (3 / x) becomes (2 * 3) / x
  //! An operator that the reordering needs but that is not in the expression
  //! is taken from the global registry. All other nodes are moved unchanged.
  //! If the global registry does not hold a needed operator then the node is
  //! left alone
  //! @param[inout] tree - Node to reorder
  //! @return Whether the node was reordered
  template <typename Num>
  bool hoist_constants(tree_item<Num>& tree) {
    typecode oper, inver;

    //oper is the commutative operator of the pair, inver is its inverse
    if (tree.value.rtype == pt_op_plus || tree.value.rtype == pt_op_minus) {
      oper = pt_op_plus;
      inver = pt_op_minus;
    } else if (tree.value.rtype == pt_op_multiply
        || tree.value.rtype == pt_op_divide) {
      oper = pt_op_multiply;
      inver = pt_op_divide;
    } else {
      return false;
    }
    if (tree.value.ptype != pt_operator || !tree.value.can_simplify
        || tree.nodes.size() != 2) return false;

    //Node 1 is the left operand and node 0 is the right operand. The child
    //is an operator from the same pair on one side that has a constant on
    //one of its sides while the other operand of this node is a constant
    const typecode prt = tree.value.rtype;
    for (count_type side = 0; side < 2; ++side) {
      tree_item<Num>& child = tree.nodes[side];
      tree_item<Num>& other = tree.nodes[1 - side];
      if (child.value.ptype != pt_operator || !child.value.can_simplify
          || child.nodes.size() != 2) continue;
      const typecode crt = child.value.rtype;
      if (crt != oper && crt != inver) continue;
      if (!detail::is_hoistable_constant(other)) continue;
      const bool u_const = detail::is_hoistable_constant(child.nodes[1]);
      const bool v_const = detail::is_hoistable_constant(child.nodes[0]);
      if (u_const == v_const) continue;

      //Layouts, where w is the other operand, u is the left operand of the
      //child and v is the right operand of the child
      //0 : (u C v) P w -> u C (v J w)   v and w are constants
      //1 : (u O v) P w -> v O (u P w)   u and w are constants
      //2 : (u I v) P w -> (u P w) I v   u and w are constants
      //3 : w P (u C v) -> (w P u) T v   u and w are constants
      //4 : w O (u C v) -> u O (w C v)   v and w are constants
      //5 : w I (u C v) -> (w J v) I u   v and w are constants
      //Here O is the commutative operator, I is the inverse, P is the
      //operator of this node and C is the operator of the child. J is O if P
      //and C are the same and I otherwise. T is O if P and C are the same and
      //I otherwise
      const bool same = (prt == crt);
      const bool child_left = (side == 1);
      typecode top, inner;
      int layout;
      if (child_left && v_const) {
        layout = 0;
        top = crt;
        inner = same ? oper : inver;
      } else if (child_left) {
        layout = (crt == oper) ? 1 : 2;
        top = crt;
        inner = prt;
      } else if (u_const) {
        layout = 3;
        top = same ? oper : inver;
        inner = prt;
      } else if (prt == oper) {
        layout = 4;
        top = prt;
        inner = crt;
      } else {
        layout = 5;
        top = prt;
        inner = same ? oper : inver;
      }
      //Get the operators from the nodes that hold them, or from the global
      //registry if the expression does not contain them
      stack_element<Num> top_value, inner_value;
      stack_co_element top_co, inner_co;
      if (!detail::find_operator<Num>(top, {&tree, &child}, top_value, top_co)
          || !detail::find_operator<Num>(inner, {&tree, &child}, inner_value,
          inner_co)) {
        continue;
      }

      tree_item<Num> u = std::move(child.nodes[1]);
      tree_item<Num> v = std::move(child.nodes[0]);
      tree_item<Num> w = std::move(other);

      //Make an operator node from an operator element and two operands
      auto make_node = [](const stack_element<Num>& value,
          const stack_co_element& co_value, tree_item<Num>& left,
          tree_item<Num>& right) {
        tree_item<Num> node;
        node.value = value;
        node.co_value = co_value;
        node.nodes.resize(2);
        node.nodes[1] = std::move(left);
        node.nodes[0] = std::move(right);
        node.nodes_associated = true;
        return node;
      };

      tree_item<Num> inner_node, result;
      switch (layout) {
        case 0:
          inner_node = make_node(inner_value, inner_co, v, w);
          result = make_node(top_value, top_co, u, inner_node);
          break;
        case 1:
          inner_node = make_node(inner_value, inner_co, u, w);
          result = make_node(top_value, top_co, v, inner_node);
          break;
        case 2:
          inner_node = make_node(inner_value, inner_co, u, w);
          result = make_node(top_value, top_co, inner_node, v);
          break;
        case 3:
          inner_node = make_node(inner_value, inner_co, w, u);
          result = make_node(top_value, top_co, inner_node, v);
          break;
        case 4:
          inner_node = make_node(inner_value, inner_co, w, v);
          result = make_node(top_value, top_co, u, inner_node);
          break;
        default:
          inner_node = make_node(inner_value, inner_co, w, v);
          result = make_node(top_value, top_co, inner_node, u);
      }
      tree = std::move(result);
      return true;
    }

    return false;
  }

  //! Reorder a node that is a binary +, -, * or / when both of its operands
  //! are operators from the same pair that each have a constant on one side.
  //! The two constants are moved into a single subtree that the simplifier
  //! then combines. For example (x + 1) + (y + 2) becomes (x + y) + (1 + 2)
  //! and (x - 1) - (y - 2) becomes (x - y) - (1 - 2)
  //! An operator that the reordering needs but that is not in the expression
  //! is taken from the global registry. All other nodes are moved unchanged.
  //! If the global registry does not hold a needed operator then the node is
  //! left alone
  //! @param[inout] tree - Node to reorder
  //! @return Whether the node was reordered
  template <typename Num>
  bool hoist_constants_pair(tree_item<Num>& tree) {
    typecode oper, inver;

    //oper is the commutative operator of the pair, inver is its inverse
    if (tree.value.rtype == pt_op_plus || tree.value.rtype == pt_op_minus) {
      oper = pt_op_plus;
      inver = pt_op_minus;
    } else if (tree.value.rtype == pt_op_multiply
        || tree.value.rtype == pt_op_divide) {
      oper = pt_op_multiply;
      inver = pt_op_divide;
    } else {
      return false;
    }
    if (tree.value.ptype != pt_operator || !tree.value.can_simplify
        || tree.nodes.size() != 2) return false;

    //Each operand is split into the part that is not constant (x) and the
    //constant (k) along with the sign with which each of them contributes to
    //the operand. For + and - the sign says whether the part is added or
    //subtracted, for * and / whether it multiplies or divides
    const typecode prt = tree.value.rtype;
    tree_item<Num>* child[2] = {&tree.nodes[1], &tree.nodes[0]};
    tree_item<Num>* x[2];
    tree_item<Num>* k[2];
    int sx[2], sk[2];
    for (int i = 0; i < 2; ++i) {
      tree_item<Num>& c = *child[i];
      if (c.value.ptype != pt_operator || !c.value.can_simplify
          || c.nodes.size() != 2) return false;
      if (c.value.rtype != oper && c.value.rtype != inver) return false;
      const bool u_const = detail::is_hoistable_constant(c.nodes[1]);
      const bool v_const = detail::is_hoistable_constant(c.nodes[0]);
      if (u_const == v_const) return false;
      const int inverse = (c.value.rtype == inver) ? -1 : 1;
      if (v_const) {
        x[i] = &c.nodes[1];
        k[i] = &c.nodes[0];
        sx[i] = 1;
        sk[i] = inverse;
      } else {
        x[i] = &c.nodes[0];
        k[i] = &c.nodes[1];
        sk[i] = 1;
        sx[i] = inverse;
      }
    }
    //The right operand is subtracted or divided by when this node is
    if (prt == inver) {
      sx[1] = -sx[1];
      sk[1] = -sk[1];
    }

    //Both non constant parts being subtracted can only be written without a
    //unary minus if the constant is added
    const bool both_neg = sx[0] < 0 && sx[1] < 0;

    //The constants can be combined in two ways if they have different signs
    for (int option = 0; option < 2; ++option) {
      typecode kop, xop, fop;
      tree_item<Num>* ka;
      tree_item<Num>* kb;
      tree_item<Num>* xa = x[0];
      tree_item<Num>* xb = x[1];
      int s;

      if (sk[0] == sk[1]) {
        if (option == 1) break;
        kop = oper;
        ka = k[0];
        kb = k[1];
        s = sk[0];
      } else {
        const bool first_positive = (sk[0] > 0);
        kop = inver;
        if (option == 0) {
          s = 1;
          ka = first_positive ? k[0] : k[1];
          kb = first_positive ? k[1] : k[0];
        } else {
          s = -1;
          ka = first_positive ? k[1] : k[0];
          kb = first_positive ? k[0] : k[1];
        }
      }
      if (both_neg && s < 0) continue;

      if (both_neg) {
        xop = inver;
        fop = inver;
      } else {
        if (sx[0] > 0 && sx[1] > 0) {
          xop = oper;
        } else {
          xop = inver;
          if (sx[0] < 0) {
            xa = x[1];
            xb = x[0];
          }
        }
        fop = (s > 0) ? oper : inver;
      }
      //Get the operators from the nodes that hold them, or from the global
      //registry if the expression does not contain them
      const std::initializer_list<const tree_item<Num>*> sources =
          {&tree, child[0], child[1]};
      stack_element<Num> kop_value, xop_value, fop_value;
      stack_co_element kop_co, xop_co, fop_co;
      if (!detail::find_operator<Num>(kop, sources, kop_value, kop_co)
          || !detail::find_operator<Num>(xop, sources, xop_value, xop_co)
          || !detail::find_operator<Num>(fop, sources, fop_value, fop_co)) {
        continue;
      }

      //Make an operator node from an operator element and two operands
      auto make_node = [](const stack_element<Num>& value,
          const stack_co_element& co_value, tree_item<Num>& left,
          tree_item<Num>& right) {
        tree_item<Num> node;
        node.value = value;
        node.co_value = co_value;
        node.nodes.resize(2);
        node.nodes[1] = std::move(left);
        node.nodes[0] = std::move(right);
        node.nodes_associated = true;
        return node;
      };

      tree_item<Num> kc = make_node(kop_value, kop_co, *ka, *kb);
      tree_item<Num> result;
      if (both_neg) {
        tree_item<Num> t1 = make_node(fop_value, fop_co, kc, *x[0]);
        result = make_node(fop_value, fop_co, t1, *x[1]);
      } else {
        tree_item<Num> xp = make_node(xop_value, xop_co, *xa, *xb);
        result = make_node(fop_value, fop_co, xp, kc);
      }
      tree = std::move(result);
      return true;
    }

    return false;
  }

  //! Collect repeated terms in a chain of + and - into a single term that is
  //! multiplied by the number of repeats. For example x + y + x becomes
  //! x * 2 + y and x - x + y becomes y. Terms are only collected if they
  //! can be merged: variables, or idempotent functions and operators that only
  //! have parameters that can be merged. Constants are left for the other
  //! simplifications. The chain is rewritten only if it ends up with fewer
  //! terms and starts with a term that is added, so that no unary minus is
  //! needed
  //! @param[inout] tree - Node to rewrite
  //! @return Whether the node was rewritten
  template <typename Num>
  bool collect_like_terms(tree_item<Num>& tree) {
    struct term_info {
      tree_item<Num>* node;
      int sign;
    };
    struct item_info {
      tree_item<Num>* node;
      Num coefficient;
    };

    if (tree.value.ptype != pt_operator || !tree.value.can_simplify
        || tree.nodes.size() != 2) return false;
    if (tree.value.rtype != pt_op_plus && tree.value.rtype != pt_op_minus) {
      return false;
    }

    //Flatten the chain of + and - into a list of terms, left to right, each
    //with the sign that it is added with
    std::vector<term_info> terms;
    std::vector<term_info> work;
    work.push_back({&tree, 1});
    while (!work.empty()) {
      const term_info current = work.back();
      work.pop_back();
      tree_item<Num>& node = *current.node;
      if (node.value.ptype == pt_operator && node.value.can_simplify
          && node.nodes.size() == 2
          && (node.value.rtype == pt_op_plus
          || node.value.rtype == pt_op_minus)) {
        const int right_sign = (node.value.rtype == pt_op_minus)
            ? -current.sign : current.sign;
        work.push_back({&node.nodes[0], right_sign});
        work.push_back({&node.nodes[1], current.sign});
      } else {
        terms.push_back(current);
      }
    }

    //Split terms of the form f * constant into f and the constant, then
    //combine identical terms
    std::vector<item_info> items;
    for (const term_info& term : terms) {
      item_info info{term.node, static_cast<Num>(term.sign)};
      if (term.node->value.ptype == pt_operator
          && term.node->value.rtype == pt_op_multiply
          && term.node->value.can_simplify && term.node->nodes.size() == 2) {
        tree_item<Num>& left = term.node->nodes[1];
        tree_item<Num>& right = term.node->nodes[0];
        const bool left_const = detail::is_hoistable_constant(left);
        const bool right_const = detail::is_hoistable_constant(right);
        if (left_const != right_const) {
          tree_item<Num>& base = right_const ? left : right;
          if (base.value.ptype != pt_constant && detail::is_mergeable(base)) {
            info.node = &base;
            info.coefficient = static_cast<Num>(term.sign)
                * std::get<Num>((right_const ? right : left).value.data);
          }
        }
      }

      bool merged = false;
      if (info.node->value.ptype != pt_constant
          && detail::is_mergeable(*info.node)) {
        for (item_info& item : items) {
          if (item.node->value.ptype != pt_constant
              && detail::is_mergeable(*item.node)
              && detail::trees_equal(*item.node, *info.node)) {
            item.coefficient = item.coefficient + info.coefficient;
            merged = true;
            break;
          }
        }
      }
      if (!merged) items.push_back(info);
    }
    if (items.size() >= terms.size()) return false;

    //The first term must be added
    count_type head = items.size();
    count_type needed = 0;
    for (count_type i = 0; i < items.size(); ++i) {
      if (items[i].coefficient > 0 && head == items.size()) head = i;
      if (items[i].coefficient != 0) ++needed;
    }
    const bool all_cancel = (needed == 0);
    if (!all_cancel && head == items.size()) return false;

    //Get the operators that the new expression needs
    bool needs_times = false;
    for (const item_info& item : items) {
      if (item.coefficient != Num(0) && item.coefficient != Num(1)
          && item.coefficient != Num(-1)) needs_times = true;
    }
    stack_element<Num> plus_value, minus_value, times_value;
    stack_co_element plus_co, minus_co, times_co;
    if (!detail::find_operator<Num>(pt_op_plus, {&tree}, plus_value, plus_co)
        || !detail::find_operator<Num>(pt_op_minus, {&tree}, minus_value,
        minus_co)) return false;
    if (needs_times && !detail::find_operator<Num>(pt_op_multiply, {&tree},
        times_value, times_co)) return false;

    //Make a constant node
    auto make_constant = [](Num number, typecode rtype,
        const std::string& text) {
      tree_item<Num> node;
      node.value = stack_element<Num>();
      node.value.ptype = pt_constant;
      node.value.rtype = rtype;
      node.value.value = 0;
      node.value.data = number;
      node.value.can_simplify = true;
      node.co_value = stack_co_element();
      node.co_value.text = text;
      return node;
    };

    //Make an operator node from an operator element and two operands
    auto make_node = [](const stack_element<Num>& value,
        const stack_co_element& co_value, tree_item<Num>& left,
        tree_item<Num>& right) {
      tree_item<Num> node;
      node.value = value;
      node.co_value = co_value;
      node.nodes.resize(2);
      node.nodes[1] = std::move(left);
      node.nodes[0] = std::move(right);
      node.nodes_associated = true;
      return node;
    };

    //Make the term for an item, multiplied by the number of repeats
    auto make_term = [&](const item_info& item) {
      tree_item<Num> term;
      copy_tree(*item.node, term);
      const Num magnitude = (item.coefficient < Num(0)) ? -item.coefficient
          : item.coefficient;
      if (magnitude != Num(1)) {
        tree_item<Num> count = make_constant(magnitude, pt_constant,
            detail::format_number("%.17E", magnitude));
        term = make_node(times_value, times_co, term, count);
      }
      return term;
    };

    tree_item<Num> result;
    if (all_cancel) {
      result = make_constant(Num(0), pt_zero, "0");
    } else {
      result = make_term(items[head]);
      for (count_type i = 0; i < items.size(); ++i) {
        if (i == head || items[i].coefficient == 0) continue;
        tree_item<Num> term = make_term(items[i]);
        if (items[i].coefficient > 0) {
          result = make_node(plus_value, plus_co, result, term);
        } else {
          result = make_node(minus_value, minus_co, result, term);
        }
      }
    }
    tree = std::move(result);
    return true;
  }

  //! Cancel factors that appear both multiplied and divided by in a chain of *
  //! and /. For example x * y / x becomes y and x * x / x becomes x. Factors
  //! are only cancelled if they can be merged: variables, or idempotent
  //! functions and operators that only have parameters that can be merged.
  //! Factors that are repeated but do not cancel are left alone. If only
  //! divisors remain then the chain starts with 1
  //! Note that cancelling a factor that is zero removes the maths domain error
  //! that dividing by it would have caused
  //! @param[inout] tree - Node to rewrite
  //! @return Whether the node was rewritten
  template <typename Num>
  bool cancel_factors(tree_item<Num>& tree) {
    struct term_info {
      tree_item<Num>* node;
      int sign;
    };

    if (tree.value.ptype != pt_operator || !tree.value.can_simplify
        || tree.nodes.size() != 2) return false;
    if (tree.value.rtype != pt_op_multiply
        && tree.value.rtype != pt_op_divide) return false;

    //Flatten the chain of * and / into a list of factors, left to right, each
    //with whether it is multiplied by or divided by
    std::vector<term_info> terms;
    std::vector<term_info> work;
    work.push_back({&tree, 1});
    while (!work.empty()) {
      const term_info current = work.back();
      work.pop_back();
      tree_item<Num>& node = *current.node;
      if (node.value.ptype == pt_operator && node.value.can_simplify
          && node.nodes.size() == 2
          && (node.value.rtype == pt_op_multiply
          || node.value.rtype == pt_op_divide)) {
        const int right_sign = (node.value.rtype == pt_op_divide)
            ? -current.sign : current.sign;
        work.push_back({&node.nodes[0], right_sign});
        work.push_back({&node.nodes[1], current.sign});
      } else {
        terms.push_back(current);
      }
    }

    //Cancel each factor with a later identical factor that is the opposite
    //way round
    std::vector<bool> removed(terms.size(), false);
    bool cancelled = false;
    for (count_type i = 0; i < terms.size(); ++i) {
      if (removed[i]) continue;
      if (terms[i].node->value.ptype == pt_constant
          || !detail::is_mergeable(*terms[i].node)) continue;
      for (count_type j = i + 1; j < terms.size(); ++j) {
        if (removed[j] || terms[j].sign != -terms[i].sign) continue;
        if (!detail::is_mergeable(*terms[j].node)) continue;
        if (detail::trees_equal(*terms[i].node, *terms[j].node)) {
          removed[i] = true;
          removed[j] = true;
          cancelled = true;
          break;
        }
      }
    }
    if (!cancelled) return false;

    //Find the factors that remain and what the chain starts with
    count_type head = terms.size();
    count_type remaining = 0;
    for (count_type i = 0; i < terms.size(); ++i) {
      if (removed[i]) continue;
      ++remaining;
      if (terms[i].sign > 0 && head == terms.size()) head = i;
    }

    //Get the operators that the new expression needs
    stack_element<Num> times_value, divide_value;
    stack_co_element times_co, divide_co;
    if (!detail::find_operator<Num>(pt_op_multiply, {&tree}, times_value,
        times_co) || !detail::find_operator<Num>(pt_op_divide, {&tree},
        divide_value, divide_co)) return false;

    //Make a constant node
    auto make_constant = [](Num number, typecode rtype,
        const std::string& text) {
      tree_item<Num> node;
      node.value = stack_element<Num>();
      node.value.ptype = pt_constant;
      node.value.rtype = rtype;
      node.value.value = 0;
      node.value.data = number;
      node.value.can_simplify = true;
      node.co_value = stack_co_element();
      node.co_value.text = text;
      return node;
    };

    //Make an operator node from an operator element and two operands
    auto make_node = [](const stack_element<Num>& value,
        const stack_co_element& co_value, tree_item<Num>& left,
        tree_item<Num>& right) {
      tree_item<Num> node;
      node.value = value;
      node.co_value = co_value;
      node.nodes.resize(2);
      node.nodes[1] = std::move(left);
      node.nodes[0] = std::move(right);
      node.nodes_associated = true;
      return node;
    };

    tree_item<Num> result;
    if (remaining == 0) {
      result = make_constant(Num(1), pt_unity, "1");
    } else {
      if (head == terms.size()) {
        result = make_constant(Num(1), pt_unity, "1");
      } else {
        copy_tree(*terms[head].node, result);
      }
      for (count_type i = 0; i < terms.size(); ++i) {
        if (removed[i] || i == head) continue;
        tree_item<Num> factor;
        copy_tree(*terms[i].node, factor);
        if (terms[i].sign > 0) {
          result = make_node(times_value, times_co, result, factor);
        } else {
          result = make_node(divide_value, divide_co, result, factor);
        }
      }
    }
    tree = std::move(result);
    return true;
  }

  //! Simplify a tree given the head node. Does this by recursively
  //! coverting nodes to simple constants if all nodes beneath them are marked
  //! as simplifyable
  //! @param[inout] tree - Tree to simplify. Tree is always simplified "inline"
  //! and is changed to reflect the simplification
  //! @param[in] params - Host code specified custom parameters. These are
  //! passed to each function in the tree so that they can mark themselves
  //! if they are conditionally simplifyable based on parameters. May be null
  //! @param[inout] errcode - Error code returned by simplify operation
  //! @param[in] options - Optional error handler and context information
  //! @return can_simplify - Whether simplification is possible when this node
  //! is considered
  template <typename Num>
  bool simplify_tree(tree_item<Num>& tree, void* params, error& errcode,
      const simplify_tree_options<Num>& options = {}) {
    bool can_simplify, has_nodes, simp1;
    Num res = Num(0);
    eval_stack<Num> eval;
    eval.host_parser = options.host_parser;
    error err;
    status status_code;

    status_code = status_none;

    has_nodes = tree.nodes_associated;
    if (has_nodes) has_nodes = has_nodes && (tree.nodes.size() > 0);
    can_simplify = tree.value.can_simplify;

    if (has_nodes) {
      for (offset_type inode = static_cast<offset_type>(tree.nodes.size()) - 1;
          inode >= 0; --inode) {
        simp1 = simplify_tree(tree.nodes[static_cast<count_type>(inode)],
            params, errcode, options);
        can_simplify = can_simplify && simp1;
      }
    }

#ifndef EIS_NO_ADVANCED_SIMPLIFY
    //Move constants together so that they can be simplified. The reordered
    //tree is then simplified again
    if (tree.value.ptype == pt_operator && errcode == err_none
        && (hoist_constants(tree) || hoist_constants_pair(tree)
        || collect_like_terms(tree) || cancel_factors(tree))) {
      return simplify_tree(tree, params, errcode, options);
    }
#endif

    if (tree.value.ptype == pt_operator) {
      if (tree.value.rtype == pt_op_multiply) {
        //Simplify multiply by zero
        if (tree.nodes[0].value.rtype == pt_zero
            || tree.nodes[1].value.rtype == pt_zero) {
          tree.value.ptype = pt_constant;
          tree.value.rtype = pt_zero;
          tree.value.data = Num(0);
          tree.co_value.deriv_values = -1;
          if (!tree.co_value.text.empty()) tree.co_value.text = "0";
          tree.nodes.clear();
          tree.nodes_associated = false;
          can_simplify = true;
          return can_simplify;
        //Simplify multiply by 1 on right
        } else if (tree.nodes[0].value.rtype == pt_unity) {
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[1], tree);
          return can_simplify;
        //Simplify multiply by 1 of the left
        } else if (tree.nodes[1].value.rtype == pt_unity) {
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[0], tree);
          return can_simplify;
        }

      } else if (tree.value.rtype == pt_op_divide) {
        //Simplify divide by 1
        if (tree.nodes[0].value.rtype == pt_unity) {
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[1], tree);
          return can_simplify;
        //Simplify 0/f
        } else if (tree.nodes[1].value.rtype == pt_zero) {
          tree.value.ptype = pt_constant;
          tree.value.rtype = pt_zero;
          tree.value.data = std::get<Num>(tree.nodes[1].value.data);
          tree.co_value.deriv_values = -1;
          if (!tree.co_value.text.empty()) tree.co_value.text = "0";
          tree.nodes.clear();
          tree.nodes_associated = false;
          can_simplify = false;
          return can_simplify;
        //Error on divide by zero
        } else if (tree.nodes[0].value.rtype == pt_zero) {
          if (options.maths_fail_quickly) {
            errcode = err_maths_domain;
            return can_simplify;
          }
          tree.value.ptype = pt_function;
          tree.value.rtype = pt_function;
          tree.value.data = parser_eval_fn<Num>(maths_domain_error<Num>);
          tree.co_value.text = "MATHS DOMAIN ERROR";
          tree.value.actual_params = 0;
          tree.value.can_simplify = false;
          tree.nodes.clear();
          tree.nodes_associated = false;
          can_simplify = false;
          return can_simplify;
        }
      } else if (tree.value.rtype == pt_op_plus) {
        //Simplify add zero on right
        if (tree.nodes[0].value.rtype == pt_zero) {
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[1], tree);
          return can_simplify;
        //Simplify add zero on left
        } else if (tree.nodes[1].value.rtype == pt_zero) {
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[0], tree);
          return can_simplify;
        }
      } else if (tree.value.rtype == pt_op_minus) {
        //Simplify subtract zero on right
        if ((tree.nodes[0].value.rtype == pt_zero
            && tree.nodes[1].value.rtype == pt_zero)
            || (tree.nodes[0].value.rtype == pt_unity
            && tree.nodes[1].value.rtype == pt_unity)) {
          tree.value.ptype = pt_constant;
          tree.value.rtype = pt_zero;
          tree.value.data = Num(0);
          tree.co_value.deriv_values = -1;
          if (!tree.co_value.text.empty()) tree.co_value.text = "0";
          tree.nodes.clear();
          tree.nodes_associated = false;
          can_simplify = true;
          return can_simplify;
        } else if (tree.nodes[0].value.rtype == pt_zero) {
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[1], tree);
          return can_simplify;
        }
      } else if (tree.value.rtype == pt_op_unary_minus) {
        if (tree.nodes[0].value.rtype == pt_zero) {
          tree.value.ptype = pt_constant;
          tree.value.rtype = pt_zero;
          tree.value.data = Num(0);
          tree.co_value.deriv_values = -1;
          if (!tree.co_value.text.empty()) tree.co_value.text = "0";
          tree.nodes.clear();
          tree.nodes_associated = false;
          can_simplify = true;
          return can_simplify;
        } else if (tree.nodes[0].value.rtype == pt_unity) {
          //Leave -unity as -unity to allow other simplification
          can_simplify = false;
          return can_simplify;
        } else if (tree.nodes[0].value.ptype == pt_operator
            && tree.nodes[0].value.rtype == pt_op_unary_minus) {
          //Simplify -(-f)
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[0].nodes[0], tree);
          can_simplify = tree.value.ptype == pt_constant
              && tree.value.can_simplify;
          return can_simplify;
        }
      } else if (tree.value.rtype == pt_op_unary_plus) {
        //Simplify +f
        std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
        tree.nodes_associated = false;
        copy_tree(tdata[0], tree);
        can_simplify = tree.value.ptype == pt_constant
            && tree.value.can_simplify;
        return can_simplify;
      } else if (tree.value.rtype == pt_op_power) {
        if (tree.nodes[0].value.rtype == pt_zero
            || tree.nodes[1].value.rtype == pt_unity) {
          //Simplify f^0 and 1^f
          tree.value.ptype = pt_constant;
          tree.value.rtype = pt_unity;
          tree.value.data = Num(1);
          tree.co_value.deriv_values = -1;
          if (!tree.co_value.text.empty()) tree.co_value.text = "1";
          tree.nodes.clear();
          tree.nodes_associated = false;
          can_simplify = true;
          return can_simplify;
        } else if (tree.nodes[0].value.rtype == pt_unity) {
          //Simplify f^1
          std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
          tree.nodes_associated = false;
          copy_tree(tdata[1], tree);
          can_simplify = tree.value.ptype == pt_constant
              && tree.value.can_simplify;
          return can_simplify;
        }
      }
    }

    //Simplify an if with a constant condition. The nodes are held in the
    //reverse of the order of the parameters so the condition is last, the
    //first result is next and the second result is first
    if (tree.value.rtype == pt_control_if && has_nodes
        && tree.nodes.size() == 3
        && detail::is_hoistable_constant(tree.nodes[2])) {
      const bool condition =
          std::get<Num>(tree.nodes[2].value.data) != Num(0);
      std::vector<tree_item<Num>> tdata = std::move(tree.nodes);
      tree.nodes_associated = false;
      copy_tree(tdata[condition ? 1 : 0], tree);
      can_simplify = tree.value.ptype == pt_constant
          && tree.value.can_simplify;
      return can_simplify;
    }

    if (has_nodes) {
      if (can_simplify) {
        for (offset_type inode = static_cast<offset_type>(tree.nodes.size()) - 1;
            inode >= 0; --inode) {
          eval.push(std::get<Num>(
              tree.nodes[static_cast<count_type>(inode)].value.data), err);
        }
      }
    }

    if (can_simplify && tree.value.ptype != pt_constant) {
      err = err_none;
      eval.eval_element(tree.value, params, status_code, err, true);
      eval.pop_scalar(res, err);
      if (err == err_none || (err == err_maths_domain
          && tree.co_value.can_accept_maths_domain_errors)) {
        can_simplify = can_simplify
            && (status_code & status_no_simplify) == 0;
        if ((status_code & status_no_simplify) == 0) {
          tree.value.ptype = pt_constant;
          tree.value.rtype = pt_constant;
          //A result that is exactly 0 or 1 is flagged as such so that the
          //simplifications for 0 and 1 apply to it
          if (res == Num(0)) {
            tree.value.rtype = pt_zero;
          } else if (res == Num(1)) {
            tree.value.rtype = pt_unity;
          }
          tree.value.data = res;
          tree.co_value.deriv_values = -1;
          if (!tree.co_value.text.empty()) {
            if (res == Num(0)) {
              tree.co_value.text = "0";
            } else if (res == Num(1)) {
              tree.co_value.text = "1";
            } else {
              tree.co_value.text = detail::format_number("%.17E", res);
            }
          }
          if (has_nodes) {
            tree.nodes.clear();
            tree.nodes_associated = false;
          }
        } else {
          return can_simplify;
        }
      } else if (!options.maths_fail_quickly && err == err_maths_domain) {
        tree.value.ptype = pt_function;
        tree.value.rtype = pt_function;
        tree.value.data = parser_eval_fn<Num>(maths_domain_error<Num>);
        tree.co_value.text = "MATHS DOMAIN ERROR";
        tree.value.actual_params = 0;
        tree.value.can_simplify = false;
        tree.nodes.clear();
        tree.nodes_associated = false;
        can_simplify = false;
      } else {
        errcode = errcode | err;
        if (options.err_handler != nullptr) {
          if (!tree.co_value.text.empty()) {
            options.err_handler->add_error(err_simplifier, err,
                {.errstring = tree.co_value.text,
                .charindex = tree.co_value.charindex,
                .filename = tree.co_value.filename,
                .line_number = tree.co_value.line_number,
                .full_line = options.full_line,
                .full_line_pos = tree.co_value.full_line_pos,
                .context_line_number = options.line_number,
                .context_filename = options.filename});
          } else {
            options.err_handler->add_error(err_simplifier, err);
          }
        }
      }
    }

    return can_simplify;
  }

  //! Convert an AST back into a stack
  //! @param[in] tree - AST to covert to stack
  //! @param[inout] stack_in - stack to build the converted AST in
  template <typename Num>
  void tree_to_stack(const tree_item<Num>& tree, eis::stack<Num>& stack_in) {
    if (tree.value.rtype == pt_control_if && tree.nodes_associated
        && tree.nodes.size() == 3) {
      //Condition, jump over the first branch if false, first branch, jump over
      //the second branch, second branch, join. The parameters are held in
      //the tree in the reverse of the order that they are in the stack
      stack_element<Num> jump_cond{}, jump_uncond{};
      stack_element<Num> join = tree.value;
      stack_co_element jump_cond_co = tree.co_value;
      stack_co_element jump_uncond_co = tree.co_value;

      jump_cond.ptype = pt_jump_conditional;
      jump_cond.rtype = pt_jump_conditional;
      jump_cond.actual_params = 0;
      jump_cond.can_simplify = false;
      jump_cond_co.text = "jump_conditional";
      jump_cond_co.expected_params = 0;
      jump_cond_co.cap_bits = 0;
      jump_cond_co.defer = false;

      jump_uncond.ptype = pt_jump_unconditional;
      jump_uncond.rtype = pt_jump_unconditional;
      jump_uncond.actual_params = 0;
      jump_uncond.can_simplify = false;
      jump_uncond_co.text = "jump_unconditional";
      jump_uncond_co.expected_params = 0;
      jump_uncond_co.cap_bits = 0;
      jump_uncond_co.defer = false;

      join.rtype = pt_control_if_join;
      join.data = std::monostate();

      tree_to_stack(tree.nodes[2], stack_in);
      const count_type jc = stack_in.stack_point;
      push_to_stack(stack_in, jump_cond, jump_cond_co);
      tree_to_stack(tree.nodes[1], stack_in);
      const count_type ju = stack_in.stack_point;
      push_to_stack(stack_in, jump_uncond, jump_uncond_co);
      tree_to_stack(tree.nodes[0], stack_in);
      stack_in.entries[jc].value = static_cast<offset_type>(ju - jc);
      stack_in.entries[ju].value =
          static_cast<offset_type>(stack_in.stack_point - ju - 1);
      push_to_stack(stack_in, join, tree.co_value);
      return;
    }

    if (tree.nodes_associated) {
      for (offset_type inode = static_cast<offset_type>(tree.nodes.size()) - 1;
          inode >= 0; --inode) {
        tree_to_stack(tree.nodes[static_cast<count_type>(inode)], stack_in);
      }
    }

    push_to_stack(stack_in, tree.value, tree.co_value);
  }

  //! Simplify a stack. Does this by converting to an AST and then recursively
  //! coverting nodes to simple constants if all nodes beneath them are marked
  //! as simplifyable
  //! @param[inout] stack - Stack to simplify. If options.stack_out is not
  //! specified then the simplified stack is stored back into "stack". If it is
  //! specified then "stack" is not changed
  //! @param[in] params - Host code specified custom parameters. These are
  //! passed to each function in the stack so that they can mark themselves
  //! if they are conditionally simplifyable based on parameters. May be null
  //! @param[inout] errcode - Error code returned by simplify operation
  //! @param[in] options - Optional error handler and output stack
  template <typename Num>
  void simplify_stack(eis::stack<Num>& stack, void* params, error& errcode,
      const simplify_stack_options<Num>& options = {}) {
    eis::stack<Num> local;
    eis::stack<Num>* simplified;
    eis::stack<Num> intermediate;
    count_type sp;

    if (options.stack_out != nullptr) *options.stack_out = eis::stack<Num>();

    if (!stack.init) return;
    if (options.stack_out == nullptr) {
      simplified = &local;
    } else {
      simplified = options.stack_out;
    }

    if (!simplified->init) initialise_stack(*simplified);
    sp = stack.stack_point;
    while (sp > 0) {
      tree_item<Num> root;
      build_node(stack, sp, root);
      errcode = err_none;
      if (!stack.full_line.empty()) {
        simplify_tree(root, params, errcode,
            {.err_handler = options.err_handler, .filename = stack.filename,
            .line_number = stack.line_number, .full_line = stack.full_line,
            .maths_fail_quickly = options.maths_fail_quickly,
            .host_parser = options.host_parser});
      } else {
        simplify_tree(root, params, errcode,
            {.err_handler = options.err_handler,
            .maths_fail_quickly = options.maths_fail_quickly,
            .host_parser = options.host_parser});
      }
      if (errcode != err_none) return;
      initialise_stack(intermediate);
      tree_to_stack(root, intermediate);
      prepend_stack(*simplified, intermediate);
      deallocate_stack(intermediate);
    }

    simplified->cap_bits = stack.cap_bits;
    simplified->has_deferred = stack.has_deferred;
    simplified->has_emplaced = stack.has_emplaced;
    simplified->can_accept_maths_domain_errors =
        stack.can_accept_maths_domain_errors;
    simplified->full_line = stack.full_line;
    simplified->params = stack.params;
    simplified->filename = stack.filename;
    simplified->owner = stack.owner;

    simplified->line_number = stack.line_number;

    if (options.stack_out == nullptr) {
      deallocate_stack(stack);
      copy_stack(*simplified, stack);
      deallocate_stack(*simplified);
    }
  }

  template <typename Num>
  void dot_output(const tree_item<Num>& node, std::string& str,
      count_type& id_in, const tree_output_options& options) {
    count_type my_id;
    std::string ds, val1, val2;
    bool is_string;

    my_id = id_in;

    is_string = (node.value.ptype == pt_constant
        && node.value.value == pt_character);

    switch (node.value.ptype) {
      case pt_function:
        ds = "box";
        break;
      case pt_operator:
        ds = "diamond";
        break;
      default:
        if (is_string) {
          ds = "doublecircle";
        } else {
          ds = "circle";
        }
    }

    val1 = detail::node_id(id_in);
    if (node.value.ptype != pt_constant || options.nformat.empty()
        || is_string) {
      append_string(str, val1 + "[label=\"" + node.co_value.text
          + "\"] [shape=" + ds + "];");
    } else {
      append_string(str, val1 + "[label=\""
          + detail::format_number(options.nformat,
          std::get<Num>(node.value.data)) + "\"] [shape=" + ds + "];");
    }

    if (node.nodes_associated) {
      for (offset_type inode = static_cast<offset_type>(node.nodes.size()) - 1;
          inode >= 0; --inode) {
        id_in = id_in + 1;
        val1 = detail::node_id(my_id);
        val2 = detail::node_id(id_in);
        append_string(str, val1 + " -- " + val2 + ";");
        dot_output(node.nodes[static_cast<count_type>(inode)], str, id_in,
            options);
      }
    }
  }

  template <typename Num>
  void tree_to_dot(const tree_item<Num>& root, std::string& str,
      const tree_output_options& options) {
    count_type root_level;

    root_level = 1;

    append_string(str, "strict graph G {");
    dot_output(root, str, root_level, options);
    append_string(str, "}");
  }

  //! Convert a stack into a string containing a dot representation
  //! of the stack
  //! @param[in] stack - Stack to visualise
  //! @param[inout] str - String to hold the dot representation
  //! @param[in] options - Format string for literal numbers
  template <typename Num>
  void visualise_stack(const eis::stack<Num>& stack, std::string& str,
      const tree_output_options& options = {}) {
    count_type sp;

    if (!stack.init) return;

    sp = stack.stack_point;
    while (sp > 0) {
      tree_item<Num> root;
      build_node(stack, sp, root);
      tree_to_dot(root, str, options);
    }
  }

  template <typename Num>
  void infix_output(const tree_item<Num>& node, std::string& str,
      const tree_output_options& options) {
    std::string param1, param2;
    std::string bb1, ab1, bb2, ab2;

    switch (node.value.ptype) {
      case pt_function:
      case pt_emplaced_function:
        append_string(str, detail::trim_copy(node.co_value.text), false);
        if (node.nodes_associated) {
          append_string(str, "(", false);
          for (offset_type inode = static_cast<offset_type>(node.nodes.size())
              - 1; inode >= 0; --inode) {
            param1.clear();
            infix_output(node.nodes[static_cast<count_type>(inode)], param1,
                options);
            append_string(str, detail::trim_copy(param1), false);
            if (inode != 0) append_string(str, ",", false);
          }
          append_string(str, ")", false);
        }
        break;
      case pt_operator:
        if (node.nodes.size() == 1) {
          if (node.nodes[0].value.ptype == pt_operator) {
            if (node.nodes[0].co_value.precedence
                <= node.co_value.precedence) {
              bb1 = "(";
              ab1 = ")";
            }
          }
          infix_output(node.nodes[0], param1, options);
          append_string(str, " " + detail::trim_copy(node.co_value.text) + bb1
              + detail::trim_copy(param1) + ab1, false);
        } else {
          infix_output(node.nodes[1], param1, options);
          infix_output(node.nodes[0], param2, options);
          if (node.nodes[1].value.ptype == pt_operator) {
            if (node.nodes[1].co_value.precedence < node.co_value.precedence
                || (node.nodes[1].co_value.precedence
                == node.co_value.precedence
                && node.co_value.associativity == assoc_ra)) {
              bb1 = "(";
              ab1 = ")";
            }
          }
          if (node.nodes[0].value.ptype == pt_operator) {
            if (node.nodes[0].co_value.precedence < node.co_value.precedence
                || (node.nodes[0].co_value.precedence
                == node.co_value.precedence
                && node.co_value.associativity == assoc_la)) {
              bb2 = "(";
              ab2 = ")";
            }
          }
          append_string(str, bb1 + detail::trim_copy(param1) + ab1
              + detail::trim_copy(node.co_value.text) + bb2
              + detail::trim_copy(param2) + ab2, false);
        }
        break;
      case pt_constant:
        if (node.value.value == pt_character || options.nformat.empty()) {
          append_string(str, detail::trim_copy(node.co_value.text), false);
        } else {
          append_string(str, detail::format_number(options.nformat,
              std::get<Num>(node.value.data)), false);
        }
        break;
      default:
        append_string(str, detail::trim_copy(node.co_value.text));
    }
  }

  //! Convert a stack into a string the infix version of the stack
  //! @param[in] stack - Stack to visualise
  //! @param[inout] str - String to hold the infix representation
  //! @param[in] options - Format string for literal numbers
  template <typename Num>
  void infix_stack(const eis::stack<Num>& stack, std::string& str,
      const tree_output_options& options = {}) {
    std::string temp;
    count_type sp;

    if (!stack.init) return;

    sp = stack.stack_point;
    while (sp > 0) {
      tree_item<Num> root;
      build_node(stack, sp, root);
      temp.clear();
      infix_output(root, temp, options);
      prepend_string(str, temp, false);
      if (sp > 0) prepend_string(str, ", ", false);
    }
  }

}

#endif
