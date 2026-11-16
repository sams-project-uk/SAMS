// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_ORDERED_STORE_H
#define EIS_ORDERED_STORE_H

#include <cstddef>
#include <utility>
#include <vector>

#include "eis_constants.h"

namespace eis {

  //! Item stored in an ordered store. Holds a pointer to the item and whether
  //! the store owns the item
  template <typename T>
  struct ordered_store_item {
    T* item = nullptr;
    bool owns = true;

    ordered_store_item() = default;
    ordered_store_item(T* item_in, bool owns_in)
        : item(item_in), owns(owns_in) {}
    ordered_store_item(const ordered_store_item&) = delete;
    ordered_store_item& operator=(const ordered_store_item&) = delete;
    ordered_store_item(ordered_store_item&& other) noexcept
        : item(other.item), owns(other.owns) {
      other.item = nullptr;
    }
    ordered_store_item& operator=(ordered_store_item&& other) noexcept {
      if (this != &other) {
        cleanup();
        item = other.item;
        owns = other.owns;
        other.item = nullptr;
      }
      return *this;
    }
    ~ordered_store_item() {
      cleanup();
    }

    //! Deallocates the item held in the store if the store owns it
    void cleanup() {
      if (item != nullptr && owns) delete item;
      item = nullptr;
    }
  };

  //! Ordered store of items of type T. Items are held on the heap so that
  //! pointers returned by get remain valid when more items are stored
  //! Indices are zero based
  template <typename T>
  class ordered_store {

    std::vector<ordered_store_item<T>> items;
    offset_type disorder_level = 0;
    //! Has anything been stored in the store since it was last cleared
    bool allocated = false;

    //! Add an item pointer to the ordered store
    //! index is the index to store the item at. It is intended to overwrite an
    //! existing item, not to allow storage in an arbitrary location. If index
    //! is not an existing item index then the item is simply added to the end
    //! of the list
    //! owns is whether the store owns the pointer to the item or not
    //! Returns the index that the item is stored at
    count_type store_core(T* item, offset_type index, bool owns) {
      allocated = true;
      if (index >= 0 && static_cast<count_type>(index) < get_size()) {
        items[static_cast<count_type>(index)] =
            ordered_store_item<T>(item, owns);
        return static_cast<count_type>(index);
      }
      items.emplace_back(item, owns);
      return get_size() - 1;
    }

  public:

    ordered_store() = default;
    ordered_store(const ordered_store&) = delete;
    ordered_store(ordered_store&&) = default;
    ordered_store& operator=(ordered_store&&) = default;

    bool is_init = false;

    //! Get the number of items in this store. Returns 0 if empty
    count_type get_size() const {
      return items.size();
    }

    //! Get an item from the store by index
    //! Returns a null pointer if the item is not present
    T* get(count_type index) const {
      if (index >= get_size()) return nullptr;
      return items[index].item;
    }

    //! Add an item to the ordered store taking a copy of it
    //! index is the index to store the item at. It is intended to overwrite an
    //! existing item, not to allow storage in an arbitrary location. If index
    //! is not an existing item index then the item is simply added to the end
    //! of the list
    //! Returns the index that the item is stored at
    count_type store(T item, offset_type index = -1) {
      return store_core(new T(std::move(item)), index, true);
    }

    //! Add an item to the ordered store but hold the pointer rather than
    //! taking a copy
    //! index is as for store
    //! owns is whether the store owns the pointer to the item or not. The
    //! default is that it does
    //! Returns the index that the item is stored at
    count_type hold(T* item, offset_type index = -1, bool owns = true) {
      return store_core(item, index, owns);
    }

    //! If the store is configured to allow it delete an item
    //! Returns whether the item was deleted
    bool erase(count_type index) {
      if (!(disorder_level > 0 && allocated)) return false;
      items.erase(items.begin() + static_cast<offset_type>(index));
      return true;
    }

    //! Delete all items from the store and deallocate the memory
    bool clear() {
      items.clear();
      allocated = false;
      return true;
    }

    //! Set disorderable true
    void enable_disorder() {
      disorder_level = disorder_level + 1;
    }

    //! Set disorderable false
    void disable_disorder() {
      disorder_level = disorder_level - 1;
    }

    //! Insert an item before another item specified by index. A copy of the
    //! item is inserted. Returns -1 if insert is not possible (either nothing
    //! has been stored yet or the index is out of range)
    //! Can specify an index 1 greater than the last index to add at the end
    offset_type insert(T item, count_type index) {
      if (!allocated) return -1;
      if (index > get_size()) return -1;
      items.insert(items.begin() + static_cast<offset_type>(index),
          ordered_store_item<T>(new T(std::move(item)), true));
      return static_cast<offset_type>(index);
    }

  };

}

#endif
