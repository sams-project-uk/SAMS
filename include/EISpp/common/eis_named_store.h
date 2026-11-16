// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_NAMED_STORE_H
#define EIS_NAMED_STORE_H

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "eis_constants.h"
#include "eis_utils.h"

namespace eis {

  //! Item stored in a named store. Holds a pointer to the item and whether
  //! the store owns the item
  template <typename T>
  struct named_store_item {
    T* item = nullptr;
    bool owns = true;

    named_store_item() = default;
    named_store_item(T* item_in, bool owns_in)
        : item(item_in), owns(owns_in) {}
    named_store_item(const named_store_item&) = delete;
    named_store_item& operator=(const named_store_item&) = delete;
    named_store_item(named_store_item&& other) noexcept
        : item(other.item), owns(other.owns) {
      other.item = nullptr;
    }
    named_store_item& operator=(named_store_item&& other) noexcept {
      if (this != &other) {
        cleanup();
        item = other.item;
        owns = other.owns;
        other.item = nullptr;
      }
      return *this;
    }
    ~named_store_item() {
      cleanup();
    }

    //! Deallocates the item held in the store if the store owns it
    void cleanup() {
      if (item != nullptr && owns) delete item;
      item = nullptr;
    }
  };

  //! Associative array of items of type T indexed by name. Items are held on
  //! the heap so that pointers returned by get remain valid when more items
  //! are stored. Trailing blanks on a name are not significant
  //! The names are also kept in the order that they were first stored
  //! Indices are zero based
  template <typename T>
  class named_store {

    std::unordered_map<std::string, named_store_item<T>> items;
    std::vector<std::string> names;

    //! Remove a name from the list of names
    void remove_name(std::string_view name) {
      for (auto found = names.begin(); found != names.end(); ++found) {
        if (detail::trimmed(*found) == detail::trimmed(name)) {
          names.erase(found);
          break;
        }
      }
    }

  public:

    named_store() = default;
    named_store(const named_store&) = delete;
    named_store(named_store&&) = default;
    named_store& operator=(named_store&&) = default;

    //! Assignment makes a copy of the store including copies of the items that
    //! it owns
    named_store& operator=(const named_store& source) {
      if (this != &source) source.copy(*this, false);
      return *this;
    }

    bool is_init = false;

    //! Add an item to the store taking a copy of it. An existing item with
    //! the same name is replaced
    void store(std::string_view name, T item) {
      hold(name, new T(std::move(item)), true);
    }

    //! Add an item to the store but hold the pointer rather than taking a
    //! copy. An existing item with the same name is replaced
    //! owns is whether the store owns the pointer to the item or not. The
    //! default is that it does
    void hold(std::string_view name, T* item, bool owns = true) {
      is_init = true;

      const std::string key(detail::trimmed(name));
      const auto found = items.find(key);
      if (found != items.end()) {
        found->second = named_store_item<T>(item, owns);
      } else {
        items.emplace(key, named_store_item<T>(item, owns));
        names.emplace_back(name);
      }
    }

    //! Get an item from the store by name
    //! Returns a null pointer if the item is not present
    T* get(std::string_view name) const {
      const auto found = items.find(std::string(detail::trimmed(name)));
      if (found == items.end()) return nullptr;
      return found->second.item;
    }

    //! Get an item from the store by the index of its name
    //! Returns a null pointer if the index is out of range
    T* get(count_type index) const {
      if (index >= get_name_count()) return nullptr;
      return get(names[index]);
    }

    //! Get the count of names
    count_type get_name_count() const {
      return names.size();
    }

    //! Get the name with a given index. Returns an empty string if the index
    //! is out of range
    std::string get_name(count_type index) const {
      if (index >= get_name_count()) return std::string();
      return names[index];
    }

    //! Alias for get_name_count
    count_type get_count() const {
      return get_name_count();
    }

    //! Delete an item specified by name
    void erase(std::string_view name) {
      const auto found = items.find(std::string(detail::trimmed(name)));
      if (found == items.end()) return;

      items.erase(found);
      remove_name(name);
    }

    //! Cleans up the store, deleting the items that it owns
    void cleanup() {
      if (!is_init) return;

      items.clear();
      is_init = false;
    }

    //! Copy this store into dest. If shallow is true then the items are not
    //! copied and dest does not own them
    void copy(named_store& dest, bool shallow = false) const {
      if (!is_init) return;
      if (dest.is_init) dest.cleanup();

      dest.is_init = true;
      dest.names = names;
      for (const auto& entry : items) {
        if (entry.second.owns && !shallow) {
          T* copied = nullptr;
          if (entry.second.item != nullptr) copied = new T(*entry.second.item);
          dest.items.emplace(entry.first, named_store_item<T>(copied, true));
        } else {
          dest.items.emplace(entry.first,
              named_store_item<T>(entry.second.item, false));
        }
      }
    }

  };

}

#endif
