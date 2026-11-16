// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_STRING_STORE_H
#define EIS_STRING_STORE_H

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "eis_constants.h"
#include "eis_utils.h"

namespace eis {

  //! Class holding the string within the string store
  struct string_holder {
    std::string text;
    std::string filename;
    offset_type minline = -1;
    offset_type maxline = -1;
    //! Number of white space characters removed from the left of the string
    //! when remove_whitespace is called
    count_type whitespace_len = 0;
  };

  //! Optional parameters to storing a string
  struct string_store_options {
    //! Index to store the string in. Intended to overwrite an existing item.
    //! If the index is not an existing item then the string is added to the
    //! end of the store. When overwriting an existing item the filename and
    //! line numbers of that item are kept unless new ones are given
    offset_type store_index = -1;
    //! Filename for a file that originally held the data
    //! Empty means that no filename is given
    std::string filename;
    //! Line number for the line. -1 means that no line number is given
    offset_type line_number = -1;
    //! Maximum line number for a line that is originally split over several
    //! lines. -1 means that no maximum line number is given
    offset_type line_number_max = -1;
    //! Number of white space characters removed from the left of the string
    //! when remove_whitespace is called. -1 means that no number is given
    offset_type trimmed_white_space_length = -1;
    //! Number of white space characters to add to the number of removed white
    //! space characters
    count_type extra_trimmed_white_space_length = 0;
    //! If true then even when overwriting an existing item the line number,
    //! filename and white space metadata of that item are not kept
    bool default_line_info = false;
  };

  //! Function that processes a filename before it is used to load a file
  using filename_processor_fn = std::function<void(std::string& filename,
      error& errcode)>;

  //! Function that processes the text of a file after it has been loaded
  using file_text_processor_fn = std::function<void(
      const std::string& filename, std::string& file_contents,
      error& errcode)>;

  //! Range of indices in a string store. Empty if first is greater than last
  struct string_range {
    offset_type first = std::numeric_limits<offset_type>::max();
    offset_type last = -1;
  };

  //! Optional parameters to adding lines from a string
  struct populate_options {
    //! Filename to associate with the lines. Empty means no filename
    std::string filename;
    //! Index to insert the lines at. -1 means that the lines are added to the
    //! end of the store
    offset_type index_start = -1;
  };

  //! Information returned from loading lines from a file
  struct file_load_info {
    //! Text of the file before it was split into lines
    std::string raw_text;
    //! Filename after processing
    std::string final_filename;
    //! Range of indices that the lines were stored at. Both -1 if the file
    //! could not be loaded
    string_range index_range;
  };

  //! Optional parameters to loading lines from a file
  struct load_file_options {
    //! Index to insert the lines at. -1 means that the lines are added to the
    //! end of the store
    offset_type index_start = -1;
    //! Function to process the filename before the file is loaded. Empty
    //! means no processing
    filename_processor_fn filename_processor;
    //! Function to process the text of the file after it is loaded. Empty
    //! means no processing
    file_text_processor_fn file_text_processor;
    //! Object that is filled with information about the load. A null pointer
    //! means that no information is returned
    file_load_info* info = nullptr;
  };

  namespace detail {

    //! Header of the serialised form of a string store
    inline constexpr std::string_view serial_header = "EISDSER001";

    //! Remove blanks from both ends of a string
    inline std::string_view strip_spaces(std::string_view text) {
      const std::string_view::size_type first = text.find_first_not_of(' ');
      if (first == std::string_view::npos) return std::string_view();
      return text.substr(first, text.find_last_not_of(' ') - first + 1);
    }

    //! Remove blanks from the end of a string
    inline std::string_view strip_trailing_spaces(std::string_view text) {
      const std::string_view::size_type last = text.find_last_not_of(' ');
      if (last == std::string_view::npos) return std::string_view();
      return text.substr(0, last + 1);
    }

    //! Write an integer right justified in a field of 20 characters
    inline std::string serial_integer(offset_type value) {
      std::string text = std::to_string(value);
      if (text.size() < 20) text.insert(0, 20 - text.size(), ' ');
      return text;
    }

    //! Read an integer from a serialised field. Returns whether it was read
    inline bool parse_serial_integer(std::string_view text,
        offset_type& value) {
      const std::string_view::size_type first = text.find_first_not_of(' ');
      if (first == std::string_view::npos) return false;
      text.remove_prefix(first);
      const char* end = text.data() + text.size();
      return std::from_chars(text.data(), end, value).ec == std::errc();
    }

  }

  //! String line store. Stores a list of strings by index number
  //! Indices are zero based
  class string_store {

    std::vector<string_holder> strings;
    //! Has anything been stored in the store
    bool allocated = false;

  public:

    //! Returns the number of stored strings
    count_type get_size() const {
      return strings.size();
    }

    //! Stores a string. Returns the index that the string is stored at
    count_type store(std::string_view text,
        const string_store_options& options = {});

    //! Inserts a string before the string at an index. The index can be the
    //! size of the store to add at the end. Only the filename and line number
    //! options are used. Returns the index that the string is stored at, or
    //! -1 if nothing has been stored yet or the index is out of range
    offset_type insert(count_type dest_index, std::string_view text,
        const string_store_options& options = {});

    //! Retrieves a string from the store. Returns whether the index was found
    bool get(count_type index, std::string& text) const;

    //! Retrieves a string from the store along with its filename and line
    //! numbers. Returns whether the index was found
    bool get(count_type index, string_holder& holder) const;

    //! Deletes an item from the store. Returns whether the deletion succeeded
    bool erase(count_type index);

    //! Combines all lines within a specified range into a single line
    void combine_lines(count_type line_start, count_type line_end);

    //! Retrieves a string from the store and appends it to an existing string
    //! optionally with a newline between them. Returns whether the index was
    //! found
    bool append(count_type index, std::string& text, bool newline = true) const;

    //! Deletes all items from the store
    bool clear();

    //! Add lines from a single string. Returns the range of indices that the
    //! lines were stored at. errcode is not used but is part of the interface
    string_range populate(std::string_view str, error& errcode,
        const populate_options& options = {});

    //! Get the stored strings as a single string with the strings separated
    //! by newlines
    std::string to_string() const;

    //! Add lines from an external file. errcode is set to the error code from
    //! the load
    void load_from_ascii_file(const std::string& filename, error& errcode,
        const load_file_options& options = {});

    //! Remove blank lines from the list of lines
    void remove_blank_lines();

    //! Combine lines when the last characters of a line is a specified
    //! character sequence
    void combine_split_lines(std::string_view eol_sequence);

    //! Removes text that appears on a line after a comment character
    //! Lines that have only a comment on them are removed. An empty sequence
    //! means that comments of that kind are not removed. Multi line comments
    //! are only removed if both the start and end sequences are given
    void remove_comments(std::string_view slc_start,
        std::string_view mlc_start, std::string_view mlc_end);

    //! Remove white space from the start and end of lines. Tabs are treated
    //! as spaces
    void remove_whitespace();

    //! Get the stored strings in raw line by line format
    std::string get_raw() const;

    //! Convert the stored strings to a portable representation that can
    //! recreate the strings along with their metadata
    std::string serialise() const;

    //! Convert strings in the portable representation back into strings
    //! stored in this store. store_index is the index to start storing at. -1
    //! means that the strings are added to the end of the store. Returns the
    //! range of indices that the strings were stored at
    string_range deserialise(std::string_view str, error& errcode,
        offset_type store_index = -1);

  };

  inline count_type string_store::store(std::string_view text,
      const string_store_options& options) {
    string_holder temp;
    temp.text = std::string(text);
    allocated = true;

    const bool replace = options.store_index >= 0
        && static_cast<count_type>(options.store_index) < get_size();
    if (replace && !options.default_line_info) {
      const string_holder& old = strings[
          static_cast<count_type>(options.store_index)];
      temp.filename = old.filename;
      temp.minline = old.minline;
      temp.maxline = old.maxline;
      temp.whitespace_len = old.whitespace_len;
    }

    if (!options.filename.empty()) temp.filename = options.filename;
    if (options.line_number != -1) {
      temp.minline = options.line_number;
      temp.maxline = options.line_number;
    }
    if (options.line_number_max != -1) temp.maxline = options.line_number_max;
    if (options.trimmed_white_space_length != -1) {
      temp.whitespace_len =
          static_cast<count_type>(options.trimmed_white_space_length);
    }
    temp.whitespace_len += options.extra_trimmed_white_space_length;

    if (replace) {
      strings[static_cast<count_type>(options.store_index)] = std::move(temp);
      return static_cast<count_type>(options.store_index);
    }
    strings.push_back(std::move(temp));
    return get_size() - 1;
  }

  inline bool string_store::get(count_type index, std::string& text) const {
    if (index >= get_size()) return false;
    text = strings[index].text;
    return true;
  }

  inline bool string_store::get(count_type index,
      string_holder& holder) const {
    if (index >= get_size()) return false;
    holder = strings[index];
    return true;
  }

  inline bool string_store::erase(count_type index) {
    strings.erase(strings.begin() + static_cast<offset_type>(index));
    return true;
  }

  inline void string_store::combine_lines(count_type line_start,
      count_type line_end) {
    if (line_end <= line_start) return;
    if (line_end >= get_size()) return;

    offset_type flmin = std::numeric_limits<offset_type>::max();
    offset_type flmax = -1;
    bool has_fname = false;
    std::string fname0;
    std::string combined;

    for (count_type istr = line_start; istr <= line_end; ++istr) {
      const string_holder& holder = strings[istr];
      if (!has_fname) {
        fname0 = holder.filename;
        has_fname = true;
      } else if (detail::trimmed(holder.filename)
          != detail::trimmed(fname0)) {
        fname0 = "{MULTIPLE SOURCES}";
      }
      flmin = std::min(flmin, holder.minline);
      flmax = std::max(flmax, holder.maxline);
      append_string(combined, holder.text, false);
    }

    strings[line_start] = string_holder{combined, fname0, flmin, flmax,
        strings[line_start].whitespace_len};
    for (count_type istr = line_end; istr > line_start; --istr) erase(istr);
  }

  inline offset_type string_store::insert(count_type dest_index,
      std::string_view text, const string_store_options& options) {
    if (!allocated) return -1;
    if (dest_index > get_size()) return -1;

    string_holder temp;
    temp.text = std::string(text);
    temp.filename = options.filename;
    if (options.line_number != -1) {
      temp.minline = options.line_number;
      temp.maxline = options.line_number;
    }
    if (options.line_number_max != -1) temp.maxline = options.line_number_max;

    strings.insert(strings.begin() + static_cast<offset_type>(dest_index),
        std::move(temp));
    return static_cast<offset_type>(dest_index);
  }

  inline bool string_store::append(count_type index, std::string& text,
      bool newline) const {
    std::string retrieved;
    if (!get(index, retrieved)) return false;

    append_string(text, retrieved, newline);
    return true;
  }

  inline bool string_store::clear() {
    strings.clear();
    allocated = false;
    return true;
  }

  inline string_range string_store::populate(std::string_view str,
      [[maybe_unused]] error& errcode, const populate_options& options) {
    string_range range;
    std::string_view::size_type last_pos = 0;
    offset_type line = 0;
    offset_type indx = options.index_start;

    auto add_line = [&](std::string_view text, offset_type number) {
      string_store_options line_options;
      line_options.filename = options.filename;
      line_options.line_number = number;
      offset_type stored;
      if (options.index_start >= 0) {
        stored = insert(static_cast<count_type>(indx), text, line_options);
        indx = stored + 1;
      } else {
        stored = static_cast<offset_type>(store(text, line_options));
        indx = stored;
      }
      range.first = std::min(range.first, stored);
      range.last = std::max(range.last, stored);
    };

    std::string_view::size_type newline_pos = str.find('\n');
    while (newline_pos != std::string_view::npos) {
      ++line;
      add_line(str.substr(last_pos, newline_pos - last_pos),
          (last_pos != newline_pos) ? line : -1);
      last_pos = newline_pos + 1;
      newline_pos = str.find('\n', last_pos);
    }

    if (last_pos < str.size()) add_line(str.substr(last_pos), line + 1);

    return range;
  }

  inline void string_store::load_from_ascii_file(const std::string& filename,
      error& errcode, const load_file_options& options) {
    file_load_info* info = options.info;

    errcode = err_none;

    std::string fname = filename;
    if (options.filename_processor) options.filename_processor(fname, errcode);

    if (info != nullptr) {
      info->final_filename = fname;
      info->index_range = string_range{-1, -1};
    }

    std::string str;
    if (!load_file_to_string(fname, str)) {
      errcode = errcode | err_no_file;
      return;
    }

    if (options.file_text_processor) {
      options.file_text_processor(fname, str, errcode);
    }

    if (info != nullptr) info->raw_text = str;

    const string_range range = populate(str, errcode,
        {.filename = fname, .index_start = options.index_start});
    if (info != nullptr) info->index_range = range;
  }

  inline std::string string_store::to_string() const {
    std::string str_out;
    for (count_type istr = 0; istr < get_size(); ++istr) {
      if (istr > 0) str_out += '\n';
      str_out += strings[istr].text;
    }
    return str_out;
  }

  inline void string_store::remove_blank_lines() {
    for (offset_type r_index = static_cast<offset_type>(get_size()) - 1;
        r_index >= 0; --r_index) {
      if (detail::trimmed(strings[static_cast<count_type>(r_index)].text)
          .empty()) erase(static_cast<count_type>(r_index));
    }
  }

  inline void string_store::combine_split_lines(std::string_view eol_sequence) {
    const std::string_view::size_type eol_length = eol_sequence.size();

    bool loop = true;
    while (loop) {
      loop = false;
      offset_type con_start = -1;
      for (count_type c_index = 0; c_index < get_size(); ++c_index) {
        const std::string_view text = strings[c_index].text;
        const std::string_view::size_type str_length = text.size();
        if (str_length > eol_length
            && text.substr(str_length - eol_length) == eol_sequence) {
          if (con_start < 0) con_start = static_cast<offset_type>(c_index);
          string_store_options options;
          options.store_index = static_cast<offset_type>(c_index);
          store(text.substr(0, str_length - eol_length), options);
          continue;
        }
        if (con_start >= 0) {
          combine_lines(static_cast<count_type>(con_start), c_index);
          loop = true;
          break;
        }
      }
    }
  }

  inline void string_store::remove_comments(std::string_view slc_start,
      std::string_view mlc_start, std::string_view mlc_end) {
    std::string str;

    //Do single line comments
    if (!slc_start.empty()) {
      offset_type istr = 0;
      while (istr < static_cast<offset_type>(get_size())) {
        get(static_cast<count_type>(istr), str);
        const std::string::size_type ipos = str.find(slc_start);
        if (ipos != std::string::npos) {
          if (detail::strip_spaces(str).find(slc_start) != 0) {
            store(std::string_view(str).substr(0, ipos),
                {.store_index = istr});
          } else {
            erase(static_cast<count_type>(istr));
            istr = istr - 1;
          }
        }
        istr = istr + 1;
      }
    }

    //Do multi line comments
    if (!mlc_start.empty() && !mlc_end.empty()) {
      offset_type istr = 0;
      bool in_comment = false;
      bool start_kept = false;
      std::string str_temp;
      while (istr < static_cast<offset_type>(get_size())) {
        get(static_cast<count_type>(istr), str);
        const std::string::size_type len = str.size();
        if (!in_comment) {
          //If not already inside an mlc comment check for the start sequence
          const std::string::size_type ipos = str.find(mlc_start);
          if (ipos != std::string::npos) {
            std::string::size_type end_pos = std::string::npos;
            if (ipos + mlc_start.size() + 1 < len) {
              end_pos = str.find(mlc_end, ipos + mlc_start.size());
            }
            //If we didn't find the end comment section on the same line
            //then we're in a comment field
            in_comment = (end_pos == std::string::npos);
            //The line holding the mlc start sequence is kept unless the
            //sequence is at the start of the line
            start_kept = detail::strip_spaces(str).find(mlc_start) != 0;
            const std::string::size_type end_one_based =
                in_comment ? ipos + mlc_start.size() : end_pos + 1;
            const std::string::size_type tail_start =
                end_one_based + mlc_end.size() - 1;
            const std::string_view tail = (tail_start < len)
                ? std::string_view(str).substr(tail_start) : std::string_view();
            if (detail::strip_spaces(str).find(mlc_start) != 0) {
              //The mlc start is not at the start of the line
              if (!in_comment && end_one_based < len) {
                //If the mlc section is hanging in the middle of a line then
                //chop it out.
                store(str.substr(0, ipos) + std::string(tail),
                    {.store_index = istr});
              } else {
                //If either the mlc end sequence isn't on this line or is
                //right at the end of this line then just delete everything
                //after the start of the mlc start sequence
                store(std::string_view(str).substr(0, ipos),
                    {.store_index = istr});
              }
            } else {
              //The mlc start sequence is at the start of a line
              if (!in_comment && end_one_based < len) {
                //If the mlc end sequence is on the same line and not at the
                //end of the line then trim out all of the text between the
                //start and end markers inclusively
                store(tail, {.store_index = istr});
              } else {
                //In this case the line is all mlc comment, so delete it
                erase(static_cast<count_type>(istr));
                istr = istr - 1;
              }
            }
          }
        } else {
          //If we are in an mlc section then check for the mlc end sequence
          const std::string::size_type ipos = str.find(mlc_end);
          if (ipos != std::string::npos) {
            //Found the mlc end sequence
            in_comment = false;
            if (start_kept) {
              //Append anything after the mlc end sequence to the line that
              //held the mlc start sequence
              get(static_cast<count_type>(istr - 1), str_temp);
              if (ipos + 1 + mlc_end.size() < len) {
                store(str_temp + str.substr(ipos + mlc_end.size()),
                    {.store_index = istr - 1});
              }
              erase(static_cast<count_type>(istr));
              istr = istr - 1;
            } else if (ipos + 1 + mlc_end.size() < len) {
              //The line holding the start was deleted so what is left of this
              //line replaces it
              store(str.substr(ipos + mlc_end.size()), {.store_index = istr});
            } else {
              erase(static_cast<count_type>(istr));
              istr = istr - 1;
            }
          } else {
            erase(static_cast<count_type>(istr));
            istr = istr - 1;
          }
        }
        istr = istr + 1;
      }
    }
  }

  inline void string_store::remove_whitespace() {
    for (count_type istr = 0; istr < get_size(); ++istr) {
      std::string text = strings[istr].text;
      std::replace(text.begin(), text.end(), '\t', ' ');
      const std::string_view stripped = detail::strip_spaces(text);
      const std::string_view without_trailing =
          detail::strip_trailing_spaces(text);
      store(stripped, {.store_index = static_cast<offset_type>(istr),
          .extra_trimmed_white_space_length =
          without_trailing.size() - stripped.size()});
    }
  }

  inline std::string string_store::get_raw() const {
    std::string str;
    for (count_type istr = 0; istr < get_size(); ++istr) {
      if (istr > 0) str += '\n';
      str += strings[istr].text;
    }
    return str;
  }

  inline std::string string_store::serialise() const {
    std::string str(detail::serial_header);
    for (const string_holder& holder : strings) {
      str += holder.text;
      str += '\0';
      str += holder.filename;
      str += '\0';
      str += detail::serial_integer(holder.minline);
      str += '\0';
      str += detail::serial_integer(holder.maxline);
      str += '\0';
      str += detail::serial_integer(
          static_cast<offset_type>(holder.whitespace_len));
      str += '\0';
    }
    return str;
  }

  inline string_range string_store::deserialise(std::string_view str,
      error& errcode, offset_type store_index) {
    string_range range;

    errcode = err_none;
    if (str.substr(0, detail::serial_header.size()) != detail::serial_header) {
      errcode = err_malformed_file;
      return range;
    }

    std::string_view::size_type last_pos = detail::serial_header.size();
    offset_type sindex = store_index;
    while (last_pos != str.size()) {
      std::string_view::size_type ends[5];
      std::string_view::size_type search_from = last_pos;
      for (std::string_view::size_type& end : ends) {
        end = str.find('\0', search_from);
        if (end == std::string_view::npos) {
          errcode = err_malformed_file;
          return range;
        }
        search_from = end + 1;
      }

      offset_type line_number, line_number_max, whitespace;
      if (!detail::parse_serial_integer(
          str.substr(ends[1] + 1, ends[2] - ends[1] - 1), line_number)
          || !detail::parse_serial_integer(
          str.substr(ends[2] + 1, ends[3] - ends[2] - 1), line_number_max)
          || !detail::parse_serial_integer(
          str.substr(ends[3] + 1, ends[4] - ends[3] - 1), whitespace)) {
        errcode = err_malformed_file;
        return range;
      }

      string_store_options options;
      options.store_index = sindex;
      options.line_number = line_number;
      options.line_number_max = line_number_max;
      options.filename = std::string(
          str.substr(ends[0] + 1, ends[1] - ends[0] - 1));
      options.trimmed_white_space_length = whitespace;
      options.default_line_info = true;
      const offset_type indx = static_cast<offset_type>(store(
          str.substr(last_pos, ends[0] - last_pos), options));
      range.first = std::min(range.first, indx);
      range.last = std::max(range.last, indx);
      if (store_index >= 0) sindex = sindex + 1;

      last_pos = ends[4] + 1;
    }

    return range;
  }

}

#endif
