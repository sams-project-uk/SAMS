// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_UTILS_H
#define EIS_UTILS_H

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <utility>

#include "eis_constants.h"

namespace eis {

  namespace detail {

    //! Remove the trailing blanks from a string
    inline std::string_view trimmed(std::string_view str) {
      const count_type last = str.find_last_not_of(' ');
      if (last == std::string_view::npos) return str.substr(0, 0);
      return str.substr(0, last + 1);
    }

  }

  //! Append a new string to the end of an existing string optionally with a
  //! newline between them. If str_old is empty then it becomes a copy of
  //! str_new
  inline void append_string(std::string& str_old, std::string_view str_new,
      bool newline = true) {
    std::string combined;
    combined.reserve(str_old.size() + 1 + str_new.size());
    combined += str_old;
    if (newline && !str_old.empty()) combined += '\n';
    combined += str_new;
    str_old = std::move(combined);
  }

  //! Prepend a new string to the start of an existing string optionally with a
  //! newline between them. If str_old is empty then it becomes a copy of
  //! str_new
  inline void prepend_string(std::string& str_old, std::string_view str_new,
      bool newline = true) {
    std::string combined(str_new);
    if (newline && !str_old.empty()) combined += '\n';
    combined += str_old;
    str_old = std::move(combined);
  }

  //! Opens a file and loads the entire contents into a string
  //! Returns false and leaves the string empty if the file cannot be opened
  inline bool load_file_to_string(const std::string& filename,
      std::string& string) {
    string.clear();
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamoff fsize = file.tellg();
    if (fsize < 0) return false;

    std::string contents(static_cast<count_type>(fsize), '\0');
    file.seekg(0);
    file.read(contents.data(), fsize);
    string = std::move(contents);
    return true;
  }

  //! Sets null defaults for error codes, status codes and or bitmasks. Null
  //! pointers are ignored
  inline void default_status(error* errcode = nullptr,
      status* status_code = nullptr, bitmask* host_bitmask = nullptr) {
    if (errcode != nullptr) *errcode = err_none;
    if (status_code != nullptr) *status_code = status_none;
    if (host_bitmask != nullptr) *host_bitmask = 0;
  }

  namespace detail {

    //! Produces the upper case equivalent of a single character
    inline char uppercase_character(char c) {
      if (c >= 'a' && c <= 'z') return static_cast<char>(c - 'a' + 'A');
      return c;
    }

  }

  //! Test if two strings are equal. The comparison is not case sensitive
  //! unless case_sensitive is true
  inline bool compare_string(std::string_view str1, std::string_view str2,
      bool case_sensitive = false) {
    if (str1.size() != str2.size()) return false;
    for (std::string_view::size_type istr = 0; istr < str1.size(); ++istr) {
      char c1 = str1[istr];
      char c2 = str2[istr];
      if (!case_sensitive) {
        c1 = detail::uppercase_character(c1);
        c2 = detail::uppercase_character(c2);
      }
      if (c1 != c2) return false;
    }
    return true;
  }

  //! Remove all characters between init and final inclusive. The indices are
  //! zero based. If init is less than 0 then init becomes 0. If final is
  //! greater than the last index then final becomes the last index
  inline void remove_string_section(std::string& str, offset_type init,
      offset_type final) {
    if (str.empty()) return;

    const offset_type first = std::max<offset_type>(0, init);
    const offset_type last = std::min(static_cast<offset_type>(str.size()) - 1,
        final);
    if (first > last) return;

    str.erase(static_cast<count_type>(first),
        static_cast<count_type>(last - first + 1));
  }

  //! Remove all of the sections of a string that start with the sequence
  //! init_s and end with the sequence final_s. Sections that are not ended are
  //! removed to the end of the string
  //! init_exclude and final_exclude specify that the removal does not include
  //! the sequences themselves
  //! first_only specifies that only the first section is removed
  inline void remove_string_section(std::string& str, std::string_view init_s,
      std::string_view final_s, bool init_exclude = false,
      bool final_exclude = false, bool first_only = false) {
    bool iter_on = true;
    while (iter_on) {
      //Positions here are one based as they are in the Fortran
      offset_type ipos = 0;
      offset_type fpos = 0;
      const std::string::size_type init_found = str.find(init_s);
      if (init_found != std::string::npos) {
        ipos = static_cast<offset_type>(init_found) + 1;
        const std::string::size_type search_start = init_found + init_s.size();
        if (init_exclude) ipos = ipos + static_cast<offset_type>(init_s.size());

        if (search_start <= str.size()) {
          const std::string::size_type final_found =
              str.find(final_s, search_start);
          if (final_found != std::string::npos) {
            fpos = static_cast<offset_type>(final_found - search_start) + 1;
          }
        }
        if (fpos > 0) {
          fpos = fpos + static_cast<offset_type>(search_start);
          if (final_exclude) {
            fpos = fpos - 1;
          } else {
            fpos = fpos + static_cast<offset_type>(final_s.size()) - 1;
          }
        }
        if (fpos <= 0) fpos = static_cast<offset_type>(str.size());
      }

      iter_on = ipos > 0 && fpos > 0;
      if (iter_on) {
        const std::string::size_type old_size = str.size();
        remove_string_section(str, ipos - 1, fpos - 1);
        //Stop if nothing was removed
        iter_on = str.size() != old_size;
      }
      iter_on = iter_on && !first_only;
    }
  }

  //! Generator of unique IDs
  struct uid_generator {
    uid_kind new_uid = 0;

    //! Gets the next UID from this generator
    uid_kind get_next_uid() {
      const uid_kind uid = new_uid;
      new_uid = new_uid + 1;
      return uid;
    }

    //! Resets the generator
    void reset(uid_kind value = 0) {
      new_uid = value;
    }
  };

}

#endif
