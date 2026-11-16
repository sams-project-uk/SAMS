// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_STACK_H
#define EIS_STACK_H

#include <cstddef>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../common/eis_utils.h"
#include "eis_parser_constants.h"

namespace eis {

  //! Routine to initialise a stack element
  //! @param[inout] element - Element to initialise
  template <typename Num>
  void initialise_stack_element(stack_element<Num>& element) {
    element.ptype = 0;
    element.value = 0;
    element.data = Num(0);
  }

  //! Routine to initialise a stack coelement
  //! @param[inout] element - Coelement to initialise
  inline void initialise_stack_co_element(stack_co_element&) {
  }

  //! Function to deallocate a stack element. Null operator at present
  //! @param[inout] element - Stack element to deallocate
  template <typename Num>
  void deallocate_stack_element(stack_element<Num>&) {
  }

  //! Function to deallocate a stack coelement.
  //! @param[inout] coelement - Stack coelement to deallocate
  inline void deallocate_stack_co_element(stack_co_element& coelement) {
    coelement.text.clear();
  }

  //! Deallocate a stack object
  //! @param[in] stack - Stack to deallocate
  template <typename Num>
  void deallocate_stack(eis::stack<Num>& stack) {
    stack.stack_point = 0;
    stack.stack_size = 0;
    stack.eval_fn = nullptr;
    stack.c_eval_fn = nullptr;
    if (stack.init) {
      for (auto& element : stack.entries) deallocate_stack_element(element);
      for (auto& coelement : stack.co_entries) {
        deallocate_stack_co_element(coelement);
      }
      stack.entries.clear();
      stack.co_entries.clear();
    }
    stack.init = false;
  }

  //! Initialise a stack object
  //! @param[in] stack - Stack to initialise
  template <typename Num>
  void initialise_stack(eis::stack<Num>& stack) {
    deallocate_stack(stack);
    stack.stack_size = 1;
    stack.cap_bits = 0;
    stack.entries.resize(stack.stack_size);
    stack.co_entries.resize(stack.stack_size);
    for (auto& element : stack.entries) initialise_stack_element(element);
    for (auto& coelement : stack.co_entries) {
      initialise_stack_co_element(coelement);
    }
    stack.init = true;
    stack.eval_fn = nullptr;
    stack.c_eval_fn = nullptr;
  }

  //! Function to copy a stack. If possible should just be
  //! simple equality so that the user can just do simple equality
  //! EIS core should always use this function just in case this
  //! isn't possible
  //! @param[in] stack - Source stack
  //! @param[in] copy - Destination stack
  template <typename Num>
  void copy_stack(const eis::stack<Num>& stack, eis::stack<Num>& copy) {
    copy = stack;
  }

  //! Reallocate a stack to have a new size. Cannot be shrunk to a size
  //! below the number of elements in the stack
  //! @param[inout] stack - Stack to grow or shrink
  //! @param[in] new_elements - Number of elements to set the stack size to
  template <typename Num>
  void grow_stack(eis::stack<Num>& stack, count_type new_elements) {
    if (!stack.init) return;
    if (new_elements < stack.stack_point) return;

    const count_type count = stack.stack_point;

    std::vector<stack_element<Num>> new_buffer(new_elements);
    element_array_assign<Num>({new_buffer.data(), count},
        {stack.entries.data(), count});
    stack.entries = std::move(new_buffer);
    stack.stack_size = new_elements;
    for (count_type i = stack.stack_point; i < stack.stack_size; ++i) {
      initialise_stack_element(stack.entries[i]);
    }
    if (!stack.co_entries.empty()) {
      std::vector<stack_co_element> new_cobuffer(new_elements);
      for (count_type i = 0; i < count; ++i) {
        new_cobuffer[i] = stack.co_entries[i];
      }
      stack.co_entries = std::move(new_cobuffer);
      for (count_type i = stack.stack_point; i < stack.stack_size; ++i) {
        initialise_stack_co_element(stack.co_entries[i]);
      }
    }
  }

  //! Reduce a stack to be as small as possible while still functioning
  //! @param[inout] stack - Stack to minify
  template <typename Num>
  void minify_stack(eis::stack<Num>& stack) {
    if (stack.has_deferred) return;

    //Manually deleting strings unnecessary per standard
    //but there are compilers in the wild that fail
    for (count_type istack = 0; istack < stack.stack_point; ++istack) {
      stack.co_entries[istack].text.clear();
    }

    stack.co_entries.clear();
    stack.full_line.clear();

    grow_stack(stack, stack.stack_point);
  }

  //! Push a single value onto the stack.
  //! @param[inout] stack - Stack to push the value on to
  //! @param[in] value - Value to push on
  template <typename Num>
  void push_to_stack(eis::stack<Num>& stack, const stack_element<Num>& value) {
    if (stack.stack_point + 1 > stack.stack_size) {
      grow_stack(stack, 2 * stack.stack_size);
    }

    stack.stack_point = stack.stack_point + 1;
    element_assign(stack.entries[stack.stack_point - 1], value);
  }

  //! Push a single value onto the stack and a covalue
  //! @param[inout] stack - Stack to push the value on to
  //! @param[in] value - Value to push on
  //! @param[in] co_value - Covalue to push on.
  //! If stack cannot store co_values this is ignored
  template <typename Num>
  void push_to_stack(eis::stack<Num>& stack, const stack_element<Num>& value,
      const stack_co_element& co_value) {
    if (stack.stack_point + 1 > stack.stack_size) {
      grow_stack(stack, 2 * stack.stack_size);
    }

    stack.stack_point = stack.stack_point + 1;
    element_assign(stack.entries[stack.stack_point - 1], value);
    if (!stack.co_entries.empty()) {
      stack.co_entries[stack.stack_point - 1] = co_value;
    }
  }

  //! Prepend one stack to the start of another stack.
  //! @param[inout] stack - Stack to prepend to
  //! @param[in] prepend - Stack to prepend to "stack"
  template <typename Num>
  void prepend_stack(eis::stack<Num>& stack, eis::stack<Num>& prepend) {
    const count_type old_stack_point = stack.stack_point;

    if (stack.stack_point + prepend.stack_point > stack.stack_size) {
      grow_stack(stack, 2 * (stack.stack_point + prepend.stack_point));
    }

    stack.stack_point = old_stack_point + prepend.stack_point;

    std::vector<stack_element<Num>> temp(stack.entries.begin(),
        stack.entries.begin() + static_cast<offset_type>(old_stack_point));
    element_array_assign<Num>({stack.entries.data() + prepend.stack_point,
        old_stack_point}, {temp.data(), old_stack_point});
    element_array_assign<Num>({stack.entries.data(), prepend.stack_point},
        {prepend.entries.data(), prepend.stack_point});

    if (!stack.co_entries.empty()) {
      std::vector<stack_co_element> cotemp(stack.co_entries.begin(),
          stack.co_entries.begin()
          + static_cast<offset_type>(old_stack_point));
      for (count_type i = 0; i < old_stack_point; ++i) {
        stack.co_entries[prepend.stack_point + i] = cotemp[i];
      }
      for (count_type i = 0; i < prepend.stack_point; ++i) {
        stack.co_entries[i] = prepend.co_entries[i];
      }
    }
    stack.cap_bits = stack.cap_bits | prepend.cap_bits;
  }

  //! Append one stack to the end of another stack.
  //! @param[inout] stack - Stack to append to
  //! @param[in] append - Stack to append to "stack"
  template <typename Num>
  void append_stack(eis::stack<Num>& stack, const eis::stack<Num>& append) {
    const count_type old_stack_point = stack.stack_point;

    if (stack.stack_point + append.stack_point > stack.stack_size) {
      grow_stack(stack, 2 * (stack.stack_point + append.stack_point));
    }

    stack.stack_point = old_stack_point + append.stack_point;
    for (count_type i = 0; i < append.stack_point; ++i) {
      element_assign(stack.entries[old_stack_point + i], append.entries[i]);
    }

    if (!stack.co_entries.empty()) {
      for (count_type i = 0; i < append.stack_point; ++i) {
        stack.co_entries[old_stack_point + i] = append.co_entries[i];
      }
    }
    stack.cap_bits = stack.cap_bits | append.cap_bits;
  }

  //! Replace a single element of a stack with all of the elements of another
  //! stack. The specified element is removed and replaced.
  //! @param[inout] dest - Stack to replace the item in
  //! @param[in] src - Stack to copy into "dest" at the specified point
  //! @param[in] insert_point - Point in "dest" to replace with "src"
  template <typename Num>
  void replace_element(eis::stack<Num>& dest, eis::stack<Num>& src,
      offset_type insert_point) {
    if (!dest.init || !src.init) return;

    if (insert_point > static_cast<offset_type>(dest.stack_point)) {
      append_stack(dest, src);
      return;
    }

    const count_type old_stack_point = dest.stack_point;

    if (dest.stack_point + src.stack_point > dest.stack_size) {
      grow_stack(dest, 2 * (dest.stack_point + src.stack_point));
    }

    const count_type insert_count = static_cast<count_type>(insert_point);
    const count_type tail_count = dest.stack_point - insert_count;

    std::vector<stack_element<Num>> temp(dest.entries.begin() + insert_point,
        dest.entries.begin() + static_cast<offset_type>(dest.stack_point));
    element_array_assign<Num>(
        {dest.entries.data() + insert_count + src.stack_point, tail_count},
        {temp.data(), tail_count});
    if (!dest.co_entries.empty() && !src.co_entries.empty()) {
      std::vector<stack_co_element> cotemp(
          dest.co_entries.begin() + insert_point,
          dest.co_entries.begin() + static_cast<offset_type>(dest.stack_point));
      for (count_type i = 0; i < tail_count; ++i) {
        dest.co_entries[insert_count + src.stack_point + i] = cotemp[i];
      }
    }

    dest.stack_point = old_stack_point + src.stack_point;
    for (count_type i = 0; i < src.stack_point; ++i) {
      element_assign(dest.entries[insert_count - 1 + i], src.entries[i]);
    }

    if (!dest.co_entries.empty() && !src.co_entries.empty()) {
      for (count_type i = 0; i < src.stack_point; ++i) {
        dest.co_entries[insert_count - 1 + i] = src.co_entries[i];
      }
    }
    dest.cap_bits = dest.cap_bits | src.cap_bits;
  }

  //! Pop a value off a stack and push it onto another stack.
  //! If covalues are enabled in stack1 then they are also copied
  //! @param[inout] stack1 - Stack to remove the value from
  //! @param[inout] stack2 - Stack to add value to
  //! @return stack_empty - Is the stack empty when trying to pop
  template <typename Num>
  bool pop_to_stack(eis::stack<Num>& stack1, eis::stack<Num>& stack2) {
    if (stack1.stack_point == 0) return true;

    const count_type top = stack1.stack_point - 1;
    if (!stack1.co_entries.empty()) {
      push_to_stack(stack2, stack1.entries[top], stack1.co_entries[top]);
    } else {
      push_to_stack(stack2, stack1.entries[top]);
    }
    stack1.stack_point = stack1.stack_point - 1;
    return false;
  }

  //! Pop a value off a stack and ignore it
  //! @param[inout] stack - Stack to remove the value from
  template <typename Num>
  void pop_to_null(eis::stack<Num>& stack) {
    stack.stack_point = stack.stack_point - 1;
  }

  //! Pop a value off a stack and return it
  //! @param[inout] stack - Stack to remove the value from
  //! @param[out] value - Value at the top of the stack
  template <typename Num>
  void pop_from_stack(eis::stack<Num>& stack, stack_element<Num>& value) {
    element_assign(value, stack.entries[stack.stack_point - 1]);
    stack.stack_point = stack.stack_point - 1;
  }

  //! Pop a value off a stack and return it with its covalue
  //! @param[inout] stack - Stack to remove the value from
  //! @param[out] value - Value at the top of the stack
  //! @param[out] covalue - Covalue at the top of the stack
  template <typename Num>
  void pop_from_stack(eis::stack<Num>& stack, stack_element<Num>& value,
      stack_co_element& covalue) {
    element_assign(value, stack.entries[stack.stack_point - 1]);
    if (!stack.co_entries.empty()) {
      covalue = stack.co_entries[stack.stack_point - 1];
    }
    stack.stack_point = stack.stack_point - 1;
  }

  //! Check the value at a distance "offset" from the top of the stack and
  //! return value
  //! @param[inout] stack - Stack to return the value from
  //! @param[out] value - Value at the top of the stack
  //! @param[in] offset - Offset from the top of the stack to snoop at.
  //! 0 returns the item at the top of the stack
  template <typename Num>
  void stack_snoop(eis::stack<Num>& stack, stack_element<Num>& value,
      count_type offset) {
    element_assign(value, stack.entries[stack.stack_point - offset - 1]);
  }

  //! Check the value at a distance "offset" from the top of the stack and
  //! return value and covalue
  //! @param[inout] stack - Stack to return the value from
  //! @param[out] value - Value at the top of the stack
  //! @param[in] offset - Offset from the top of the stack to snoop at.
  //! 0 returns the item at the top of the stack
  //! @param[out] covalue - Covalue at the top of the stack
  template <typename Num>
  void stack_snoop(eis::stack<Num>& stack, stack_element<Num>& value,
      count_type offset, stack_co_element& covalue) {
    element_assign(value, stack.entries[stack.stack_point - offset - 1]);
    if (!stack.co_entries.empty()) {
      covalue = stack.co_entries[stack.stack_point - offset - 1];
    }
  }

  //! Routine to return a textual version of the stack. Shows the stack in RPN
  //! form
  //! @param[in] token_list - Stack to visualise
  //! @param[inout] str - String. Will be exactly the right size to hold the
  //! token stream
  template <typename Num>
  void get_tokens(const eis::stack<Num>& token_list, std::string& str) {
    str.clear();
    if (token_list.co_entries.empty()) return;

    for (count_type i = 0; i < token_list.stack_point; ++i) {
      if (token_list.entries[i].ptype == pt_jump_conditional
          || token_list.entries[i].ptype == pt_jump_unconditional) continue;
      const std::string& text = token_list.co_entries[i].text;
      if (!text.empty()) {
        append_string(str, text + " ", false);
      } else {
        append_string(str, "{bad_token}", false);
      }
    }
  }

  //! Routine to automatically print the result of get_tokens to a stream
  //! @param[in] token_list - Stack to visualise
  //! @param[inout] out - Stream to print to. The default is stdout
  template <typename Num>
  void display_tokens_inline(const eis::stack<Num>& token_list,
      std::ostream& out = std::cout) {
    if (token_list.co_entries.empty()) return;

    std::string str;
    get_tokens(token_list, str);
    if (!str.empty()) {
      out << str << '\n';
    }
    out << '\n';
  }

}

#endif
