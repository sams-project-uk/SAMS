// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_INTEROP_HANDLES_H
#define EIS_INTEROP_HANDLES_H

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "eis_parser.h"

namespace eis_c_interop {

  //! Parser that is referenced from C
  struct parser_slot {
    eis::parser<double>* contents = nullptr;
    bool owns = false;
    int refcount = 0;
    bool in_use = false;
  };

  //! Stack that is referenced from C
  struct stack_slot {
    eis::stack<double>* contents = nullptr;
    eis::parser<double>* parser = nullptr;
    bool owns = false;
    int refcount = 0;
    bool in_use = false;
  };

  inline std::vector<parser_slot>& parser_slots() {
    static std::vector<parser_slot> slots;
    return slots;
  }

  inline std::vector<stack_slot>& stack_slots() {
    static std::vector<stack_slot> slots;
    return slots;
  }

  //! Get the ID of a stored parser. -1 if the parser is not stored
  inline int find_parser_id(const eis::parser<double>* parser) {
    const std::vector<parser_slot>& slots = parser_slots();

    for (std::size_t i = 0; i < slots.size(); ++i) {
      if (slots[i].in_use && slots[i].contents == parser) {
        return static_cast<int>(i) + 1;
      }
    }
    return -1;
  }

  //! Store a parser and return its ID. IDs start from 1. A parser that is
  //! already stored keeps its ID
  inline int add_parser(eis::parser<double>* parser, bool owns) {
    std::vector<parser_slot>& slots = parser_slots();

    const int existing = find_parser_id(parser);
    if (existing > 0) return existing;

    parser_slot added;
    added.contents = parser;
    added.owns = owns;
    added.refcount = 1;
    added.in_use = true;

    for (std::size_t i = 0; i < slots.size(); ++i) {
      if (!slots[i].in_use) {
        slots[i] = added;
        return static_cast<int>(i) + 1;
      }
    }
    slots.push_back(added);
    return static_cast<int>(slots.size());
  }

  //! Get a parser by ID. Null if the ID is not valid
  inline eis::parser<double>* get_parser(int id) {
    const std::vector<parser_slot>& slots = parser_slots();

    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return nullptr;
    const parser_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    return slot.in_use ? slot.contents : nullptr;
  }

  //! Release a parser, deleting it if it is owned
  inline void release_parser(int id) {
    std::vector<parser_slot>& slots = parser_slots();

    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return;
    parser_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    if (!slot.in_use) return;

    eis::parser<double>* owned = slot.owns ? slot.contents : nullptr;
    slot = parser_slot();
    delete owned;
  }

  //! Increment the reference count of a parser
  inline void inc_parser_ref(int id) {
    std::vector<parser_slot>& slots = parser_slots();

    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return;
    parser_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    if (slot.in_use) slot.refcount = slot.refcount + 1;
  }

  //! Decrement the reference count of a parser, releasing it at zero
  inline void dec_parser_ref(int id) {
    std::vector<parser_slot>& slots = parser_slots();

    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return;
    parser_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    if (!slot.in_use) return;

    slot.refcount = std::max(slot.refcount - 1, 0);
    if (slot.refcount == 0) release_parser(id);
  }

  //! Store a stack and return its ID. IDs start from 1. A stack that is
  //! already stored keeps its ID
  inline int add_stack(eis::stack<double>* stack, eis::parser<double>* parser,
      bool owns) {
    std::vector<stack_slot>& slots = stack_slots();

    for (std::size_t i = 0; i < slots.size(); ++i) {
      if (slots[i].in_use && slots[i].contents == stack) {
        return static_cast<int>(i) + 1;
      }
    }

    stack_slot added;
    added.contents = stack;
    added.parser = parser;
    added.owns = owns;
    added.refcount = 1;
    added.in_use = true;

    for (std::size_t i = 0; i < slots.size(); ++i) {
      if (!slots[i].in_use) {
        slots[i] = added;
        return static_cast<int>(i) + 1;
      }
    }
    slots.push_back(added);
    return static_cast<int>(slots.size());
  }

  //! Get a stack by ID and optionally the parser that it belongs to. Null if
  //! the ID is not valid
  inline eis::stack<double>* get_stack(int id,
      eis::parser<double>** parser = nullptr) {
    const std::vector<stack_slot>& slots = stack_slots();

    if (parser != nullptr) *parser = nullptr;
    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return nullptr;
    const stack_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    if (!slot.in_use) return nullptr;
    if (parser != nullptr) *parser = slot.parser;
    return slot.contents;
  }

  //! Release a stack, deleting it if it is owned
  inline void release_stack(int id) {
    std::vector<stack_slot>& slots = stack_slots();

    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return;
    stack_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    if (!slot.in_use) return;

    eis::stack<double>* owned = slot.owns ? slot.contents : nullptr;
    slot = stack_slot();
    delete owned;
  }

  //! Increment the reference count of a stack
  inline void inc_stack_ref(int id) {
    std::vector<stack_slot>& slots = stack_slots();

    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return;
    stack_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    if (slot.in_use) slot.refcount = slot.refcount + 1;
  }

  //! Decrement the reference count of a stack, releasing it at zero
  inline void dec_stack_ref(int id) {
    std::vector<stack_slot>& slots = stack_slots();

    if (id < 1 || static_cast<std::size_t>(id) > slots.size()) return;
    stack_slot& slot = slots[static_cast<std::size_t>(id) - 1];
    if (!slot.in_use) return;

    slot.refcount = std::max(slot.refcount - 1, 0);
    if (slot.refcount == 0) release_stack(id);
  }

  //! Copy a stack. The copy is owned. Returns -1 if the ID is not valid
  inline int copy_stack(int id) {
    eis::parser<double>* parser = nullptr;
    const eis::stack<double>* source = get_stack(id, &parser);

    if (source == nullptr) return -1;
    eis::stack<double>* copy = new eis::stack<double>;
    eis::copy_stack(*source, *copy);
    return add_stack(copy, parser, true);
  }

  //! Copy text to a buffer that is owned by C. Terminated unless buflen is not
  //! positive
  inline void copy_to_buffer(const std::string& text, int buflen, char* out) {
    if (out == nullptr || buflen <= 0) return;

    const std::size_t count = std::min(text.size(),
        static_cast<std::size_t>(buflen) - 1);
    std::memcpy(out, text.data(), count);
    out[count] = '\0';
  }

  //! Convert a C string to a string. Null gives an empty string
  inline std::string to_string(const char* text) {
    return text != nullptr ? std::string(text) : std::string();
  }

}

#endif
