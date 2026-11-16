// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_OPTIONAL_H
#define EIS_OPTIONAL_H

#include <utility>

namespace eis {

  //! Value that may or may not have been set. Is set by construction from a
  //! value or by assignment of a value
  template <typename T>
  class optional {

    T value{};
    bool holds = false;

  public:

    optional() = default;

    //! Has the value been set
    bool is_set() const {
      return holds;
    }

    optional(const T& value_in) : value(value_in), holds(true) {}

    optional(T&& value_in) : value(std::move(value_in)), holds(true) {}

    optional& operator=(const T& value_in) {
      value = value_in;
      holds = true;
      return *this;
    }

    optional& operator=(T&& value_in) {
      value = std::move(value_in);
      holds = true;
      return *this;
    }

    operator T&() {
      return value;
    }

    operator const T&() const {
      return value;
    }

    //! Get the value if it has been set and the default value if not
    T get_or_value(const T& default_value) const {
      return holds ? value : default_value;
    }

    //! Make this empty so that it does not hold a value
    void empty() {
      holds = false;
    }

  };

  //! Reference to a value that may or may not have been set. The first
  //! assignment captures the object that is assigned. Later assignments
  //! assign to the captured object
  template <typename T>
  class optional<T&> {

    T* value = nullptr;
    bool holds = false;

  public:

    optional() = default;

    //! Has the reference been set
    bool is_set() const {
      return holds;
    }

    optional(T& value_in) : value(&value_in), holds(true) {}

    optional& operator=(T& value_in) {
      if (holds) {
        *value = value_in;
      } else {
        value = &value_in;
        holds = true;
      }
      return *this;
    }

    operator T&() const {
      return *value;
    }

    //! Get the captured object if it has been set and the default value if not
    T& get_or_value(T& default_value) const {
      return holds ? *value : default_value;
    }

    //! Make this empty so that it does not hold a reference. The next
    //! assignment captures the object that is assigned
    void empty() {
      value = nullptr;
      holds = false;
    }

  };

}

#endif
