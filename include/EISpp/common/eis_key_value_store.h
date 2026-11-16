// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_KEY_VALUE_STORE_H
#define EIS_KEY_VALUE_STORE_H

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "eis_constants.h"
#include "eis_string_store.h"
#include "eis_utils.h"

namespace eis {

  //! Key value store. Maps a key string to a stored string
  //! Trailing blanks on a key are not significant
  class key_value_store {

    std::unordered_map<std::string, std::string> strings;

  public:

    //! Stores a string using a named key
    void store(std::string_view key, std::string_view text) {
      strings[std::string(detail::trimmed(key))] = std::string(text);
    }

    //! Retrieves a string from the store. Returns whether the key was found
    //! The string is not changed if the key is not found
    bool get(std::string_view key, std::string& text) const {
      const auto found = strings.find(std::string(detail::trimmed(key)));
      if (found == strings.end()) return false;
      text = found->second;
      return true;
    }

    //! Retrieves a string from the store and appends it to an existing string
    //! optionally with a newline between them. Returns whether the key was
    //! found
    bool append(std::string_view key, std::string& text,
        bool newline = true) const {
      std::string retrieved;
      if (!get(key, retrieved)) return false;
      append_string(text, retrieved, newline);
      return true;
    }

    //! Does formatted replacement of text based on keys stored in this store
    //! The text contains replacement keys of the form "{key}" which are
    //! replaced with the text associated with the key "key" stored in this
    //! store. Keys that are not found are replaced with nothing
    void format_fill(std::string& text) const {
      std::string output;
      std::string segment;
      bool recording_name = false;

      for (const char c : text) {
        if (!recording_name) {
          if (c != '{') {
            segment += c;
          } else {
            output += segment;
            recording_name = true;
            segment.clear();
          }
        } else {
          if (c != '}') {
            segment += c;
          } else {
            append(segment, output, false);
            recording_name = false;
            segment.clear();
          }
        }
      }
      output += segment;
      text = std::move(output);
    }

    //! Takes a single string of the form "key=value", splits it into key and
    //! value components and stores them. Lines that start with a hash "#" are
    //! comments. A hash after the equals sign starts a comment that ends the
    //! value. Malformed strings are ignored
    void parse_key_string(std::string_view string) {
      const std::string_view::size_type first =
          string.find_first_not_of(' ');
      if (first == std::string_view::npos) return;
      const std::string_view trstr = detail::trimmed(string.substr(first));

      if (trstr[0] == '#') return;

      const std::string_view::size_type eqpos = trstr.find('=');
      std::string_view::size_type hashpos = trstr.find('#');

      //No line terminal comment
      if (hashpos == std::string_view::npos) hashpos = trstr.size();
      //Malformed string without an equals. Not much we can do
      if (eqpos == std::string_view::npos) return;
      //Malformed string with a comment before the equals sign. Not much we can
      //do
      if (hashpos < eqpos) return;
      //Everything before the equals is a key, everything between the equals and
      //an optional #comment is the value
      store(trstr.substr(0, eqpos),
          trstr.substr(eqpos + 1, hashpos - eqpos - 1));
    }

    //! Adds keys to the store from an external file. Data should be stored in
    //! the form "key=value". Lines can be comments by having the first non
    //! whitespace character be a hash "#" character. Lines can also have a
    //! comment appended to the end of them by putting in a # character. Lines
    //! can be continued by adding a "\" character as the last character on the
    //! line. errcode is set to the error code from loading the file
    void load_from_ascii_file(const std::string& filename, error& errcode) {
      string_store raw_strings;
      raw_strings.load_from_ascii_file(filename, errcode);
      raw_strings.remove_blank_lines();
      raw_strings.combine_split_lines("\\");

      std::string str;
      for (count_type istr = 0; istr < raw_strings.get_size(); ++istr) {
        raw_strings.get(istr, str);
        parse_key_string(str);
      }
    }

  };

}

#endif
