// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_NUMBERED_STORE_H
#define EIS_NUMBERED_STORE_H

#include <unordered_map>
#include <utility>

#include "eis_constants.h"

namespace eis {

  //! Item stored in a numbered store. Holds a pointer to the item and whether
  //! the store owns the item
  template <typename T>
  struct numbered_store_item {
    T* item = nullptr;
    bool owns = true;

    numbered_store_item() = default;
    numbered_store_item(T* item_in, bool owns_in)
        : item(item_in), owns(owns_in) {}
    numbered_store_item(const numbered_store_item&) = delete;
    numbered_store_item& operator=(const numbered_store_item&) = delete;
    numbered_store_item(numbered_store_item&& other) noexcept
        : item(other.item), owns(other.owns) {
      other.item = nullptr;
    }
    numbered_store_item& operator=(numbered_store_item&& other) noexcept {
      if (this != &other) {
        cleanup();
        item = other.item;
        owns = other.owns;
        other.item = nullptr;
      }
      return *this;
    }
    ~numbered_store_item() {
      cleanup();
    }

    //! Deallocates the item held in the store if the store owns it
    void cleanup() {
      if (item != nullptr && owns) delete item;
      item = nullptr;
    }
  };

  //! Associative array of items of type T indexed by number. Items are held on
  //! the heap so that pointers returned by get remain valid when more items
  //! are stored
  template <typename T>
  class numbered_store {

    std::unordered_map<i8, numbered_store_item<T>> items;

  public:

    numbered_store() = default;
    numbered_store(const numbered_store&) = delete;
    numbered_store(numbered_store&&) = default;
    numbered_store& operator=(numbered_store&&) = default;

    bool is_init = false;

    //! Add an item to the store taking a copy of it. An existing item with
    //! the same number is replaced
    void store(i8 key, T item) {
      hold(key, new T(std::move(item)), true);
    }

    //! Add an item to the store but hold the pointer rather than taking a
    //! copy. An existing item with the same number is replaced
    //! owns is whether the store owns the pointer to the item or not. The
    //! default is that it does
    void hold(i8 key, T* item, bool owns = true) {
      is_init = true;

      const auto found = items.find(key);
      if (found != items.end()) {
        found->second = numbered_store_item<T>(item, owns);
      } else {
        items.emplace(key, numbered_store_item<T>(item, owns));
      }
    }

    //! Get an item from the store by number
    //! Returns a null pointer if the item is not present
    T* get(i8 id) const {
      const auto found = items.find(id);
      if (found == items.end()) return nullptr;
      return found->second.item;
    }

    //! Clear the contents of the store, deleting the items that it owns
    void clear() {
      items.clear();
    }

  };

}

#endif
