// Copyright (c) 2019-2026, C. S. Brady and H. Ratcliffe
// SPDX-License-Identifier: BSD-3-Clause

#ifndef EIS_ERROR_H
#define EIS_ERROR_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "eis_constants.h"
#include "eis_key_value_store.h"
#include "eis_utils.h"

namespace eis {

  namespace detail {

    //! Is a string non empty after removing trailing blanks
    inline bool filled(std::string_view str) {
      return str.find_first_not_of(' ') != std::string_view::npos;
    }

    //! Write an integer right justified in a field of the given width. A
    //! width of zero gives the smallest field that holds the integer. A field
    //! that is too narrow gives the smallest field that holds the integer
    inline std::string format_integer(offset_type value, offset_type width) {
      std::string text = std::to_string(value);
      if (width <= 0) return text;
      const count_type field = static_cast<count_type>(width);
      if (text.size() >= field) return text;
      return std::string(field - text.size(), ' ') + text;
    }

  }

  //! Type representing an error
  struct error_item {
    //! String representation of the source of the error
    std::string errstring;
    //! Filename for the source of the error
    std::string filename;
    //! Full line that contains the source of the error
    std::string full_line;
    //! Numerical representation of the type of the error
    error errcode = err_none;
    //! Character offset from start of the string of the error
    offset_type charindex = -1;
    //! Line number for an error on a specific line of a file
    offset_type line_number = -1;
    //! Context line number. Used if the error is caused by a
    //! different line than defines the text
    offset_type context_line_number = -1;
    //! Context filename
    std::string context_filename;
    offset_type full_line_pos = -1;
  };

  //! Optional parameters to initialising an error handler
  struct error_init_options {
    //! Optional filename for a language pack. This will described the required
    //! errors in a "key=value" form. See the manual for a description of the
    //! keys needed. Empty means use the default language
    std::string language_pack;
    //! Function to be called whenever an error is added to the error handler
    //! A null pointer leaves any existing function in place
    on_error_callback on_error = nullptr;
  };

  //! Optional parameters to adding an error to an error handler
  struct add_error_options {
    //! String describing parser key that causes the error
    std::string errstring;
    //! Character location of key in error
    offset_type charindex = -1;
    //! String describing the file that contains the error
    std::string filename;
    //! Line number of the error in multiple lines
    offset_type line_number = -1;
    //! Full line containing the error
    std::string full_line;
    //! Position of the error in the full line
    offset_type full_line_pos = -1;
    //! Line number for context information
    offset_type context_line_number = -1;
    //! Filename for context information
    std::string context_filename;
  };

  //! Information about the part of the parsed string that caused an error
  struct error_cause {
    //! Error text for the returned error text
    std::string err_text;
    //! Character location for the returned error (in the original string)
    offset_type charindex = -1;
    //! Filename for the error source
    std::string filename;
    //! Line number of the cause of the error
    offset_type line_number = -1;
    //! Filename for the error context source
    std::string context_filename;
    //! Line number for the error context
    offset_type context_line_number = -1;
    //! Position of the error text in the full line that encompasses it
    offset_type full_line_pos = -1;
  };

  //! Strings describing the type and the source of an error
  struct error_strings {
    //! Error type string in specified language
    std::string err_string;
    //! Error source string in specified language
    std::string err_source;
  };

  //! Error handler class
  //! Indices of stored errors are zero based
  class error_handler {

    //! Held list of errors
    std::vector<error_item> errors;
    //! Object holding the error report strings in the specified language
    key_value_store strings;
    //! Is this error handler initialised
    bool is_init = false;
    std::string language_pack;
    on_error_callback on_error_fn = nullptr;

  public:

    //! Initialise the error handler with a specific language for error
    //! messages. errcode is set to the error code from loading the language
    //! pack
    void init(error& errcode, const error_init_options& options = {});

    //! Add an error
    //! err_source is the type code for the source of the error (tokenize,
    //! evaluate etc.) and errcode is the type code for the type of the error
    void add_error(error err_source, error errcode,
        const add_error_options& options = {});

    //! Delete all stored errors
    void flush_errors() {
      errors.clear();
    }

    //! Return number of errors
    count_type get_error_count() const {
      return errors.size();
    }

    //! Return strings describing the type and source of a stored error
    //! The strings are empty if the index is out of range
    error_strings get_error_string(count_type index);

    //! Return strings describing the type and source of an error code in the
    //! specified language
    error_strings get_error_string_from_code(error errcode);

    //! Return information about the key in the parsed string that is the cause
    //! of the errors. The returned values are the defaults if the index is out
    //! of range
    error_cause get_error_cause(count_type index);

    //! Return entire error report
    std::string get_error_report(count_type index);

    //! Print error report to a stream. The default is to the screen
    void print_error_string(count_type index, std::ostream& out = std::cout) {
      out << get_error_report(index) << '\n';
    }

  };

  inline void error_handler::init(error& errcode,
      const error_init_options& options) {
    errcode = err_none;
    const bool has_pack = !options.language_pack.empty();
    bool same_language;

    if (!language_pack.empty()) {
      if (has_pack) {
        same_language = detail::trimmed(options.language_pack)
            == detail::trimmed(language_pack);
      } else {
        same_language = false;
        language_pack.clear();
      }
    } else {
      if (has_pack) {
        same_language = false;
        language_pack = options.language_pack;
      } else {
        same_language = true;
      }
    }

    if (options.on_error != nullptr) on_error_fn = options.on_error;

    if (is_init && same_language) return;
    is_init = true;

    bool lang_default = false;
    if (has_pack) {
      errcode = err_none;
      strings.load_from_ascii_file(options.language_pack, errcode);
      if (errcode != err_none) {
        std::cerr << "Error loading language pack for error handler. Default "
            "to English\n";
        lang_default = true;
        language_pack.clear();
      }
    }

    if (!has_pack || lang_default) {
      strings.store("err_src_parse", "parsing text to stack");
      strings.store("err_src_simplify", "simplifying a stack");
      strings.store("err_src_emplace", "emplacing a function");
      strings.store("err_src_evaluate", "evaluating stack");
      strings.store("err_src_file", "handling text file");
      strings.store("err_src_deck", "loading input deck file");
      strings.store("err_src_deck_definition",
          "handling input deck definition");
      strings.store("err_src_deck_parser", "parsing input deck file");
      strings.store("err_src_undefined", "in unspecified part of EIS. (This "
          "is probably a bug in EIS)");

      strings.store("err_bad_value", "There was a bad value in the "
          "expression");
      strings.store("err_malformed", "The expression was of invalid form");
      strings.store("err_wrong_parameters", "The wrong number or kind of "
          "parameters was used in a function call");
      strings.store("err_maths_domain", "A mathematically invalid operation "
          "was requested");
      strings.store("err_not_found", "Unknown value or function");
      strings.store("err_has_emplaced", "The specified stack has unresolved "
          "emplaced elements");
      strings.store("err_has_deferred", "The specified stack has unresolved "
          "deferred elements");
      strings.store("err_bracketed_constant", "Attempting to subscript a "
          "constant");
      strings.store("err_extra_bracket", "Extraneous bracket");
      strings.store("err_bad_stack", "Invalid or unavailable stack");
      strings.store("err_bad_parser", "Invalid or unavailable parser");
      strings.store("err_extra_results", "Unexpected additional results from "
          "a stack");

      strings.store("err_no_luns", "No logical unit numbers were available "
          "for file access");
      strings.store("err_no_file", "The requested file was not found");
      strings.store("err_malformed_file", "Malformed string when reading a "
          "serialisation string.");
      strings.store("err_mismatched_begin_end", "Deck block \"end\"ed without "
          "matching \"start\" directive");
      strings.store("err_deck_too_deep", "Deck block encountered with too "
          "high a depth (i.e is a child of a block that should not have "
          "children). This is a malformed deck.");
      strings.store("err_deck_empty_block", "Deck block has no keys when this "
          "is not permitted. This is a malformed deck");
      strings.store("err_root_keys", "Deck key encountered outside a block. "
          "This is a malformed deck");
      strings.store("err_defn_invalid", "Deck definition object is in an "
          "invalid state");
      strings.store("err_unknown_block", "An unknown block was found in the "
          "deck");
      strings.store("err_unknown_key", "An unknown key was found in the "
          "deck");
      strings.store("err_bad_key", "A key in the deck could not be parsed");
      strings.store("err_host", "A non-parser error was reported");
      strings.store("err_text", "This function should not take text "
          "parameters");
      strings.store("err_interop", "You are using a Fortran/C "
          "interoperability function but the item you are using is not "
          "interoperable");
      strings.store("err_out_of_range", "The specified value is outwith the "
          "range of acceptable values for this function");
      strings.store("err_stack_params", "A stack variable that returns more "
          "than one parameter was used in a mathematical expression");
      strings.store("err_no_deriv", "A derivative was requested of a "
          "function for which no derivative is defined");
      strings.store("err_invalid_minify", "An operation was requested that "
          "is invalid after a stack has been minified");

      strings.store("err_is_error", "{error} at {errfile}, line {errline}");
      strings.store("err_is_error_file", "{error} at {errfile}");
      strings.store("err_is_error_only", "{error}");
      strings.store("err_report_place", "{charpos}:\"{errtext}\" - {error}");
      strings.store("err_proximate_source", "Error reported from "
          "{context_errfile}:{context_errline}");
      strings.store("err_report_fail", "Unable to report source of error");
    }

  }

  inline void error_handler::add_error(error err_source, error errcode,
      const add_error_options& options) {
    if (!is_init) {
      error init_err;
      init(init_err);
    }

    if (errcode == err_none) return;

    error_item item;
    item.errcode = err_source | errcode;
    item.errstring = options.errstring;
    if (detail::filled(options.filename)) item.filename = options.filename;
    item.charindex = options.charindex;
    item.line_number = options.line_number;
    if (detail::filled(options.full_line)) item.full_line = options.full_line;
    item.full_line_pos = options.full_line_pos;
    item.context_line_number = options.context_line_number;
    if (detail::filled(options.context_filename))
        item.context_filename = options.context_filename;

    errors.push_back(std::move(item));

    if (on_error_fn != nullptr) on_error_fn(errcode);
  }

  inline error_cause error_handler::get_error_cause(count_type index) {
    if (!is_init) {
      error init_err;
      init(init_err);
    }

    error_cause cause;
    if (index >= get_error_count()) return cause;

    const error_item& item = errors[index];
    cause.charindex = item.charindex;
    cause.line_number = item.line_number;
    cause.context_line_number = item.context_line_number;
    cause.err_text = detail::trimmed(item.errstring);
    cause.filename = detail::trimmed(item.filename);
    cause.context_filename = detail::trimmed(item.context_filename);
    cause.full_line_pos = item.full_line_pos;

    return cause;
  }

  inline error_strings error_handler::get_error_string_from_code(error errcode) {
    if (!is_init) {
      error init_err;
      init(init_err);
    }

    error_strings result;
    std::string& err_string = result.err_string;
    std::string& err_source = result.err_source;
    bool ok;

    if ((errcode & err_parser) != 0) {
      ok = strings.append("err_src_parse", err_source);
    } else if ((errcode & err_simplifier) != 0) {
      ok = strings.append("err_src_simplify", err_source);
    } else if ((errcode & err_emplacer) != 0) {
      ok = strings.append("err_src_emplace", err_source);
    } else if ((errcode & err_evaluator) != 0) {
      ok = strings.append("err_src_evaluate", err_source);
    } else if ((errcode & err_file) != 0) {
      ok = strings.append("err_src_file", err_source);
    } else if ((errcode & err_deck_file) != 0) {
      ok = strings.append("err_src_deck", err_source);
    } else if ((errcode & err_deck_definition) != 0) {
      ok = strings.append("err_src_deck_definition", err_source);
    } else if ((errcode & err_deck_parser) != 0) {
      ok = strings.append("err_src_deck_parser", err_source);
    } else {
      ok = strings.append("err_src_undefined", err_source);
    }

    if (!ok) {
      err_source = "SERIOUS ERROR: UNABLE TO FIND ERROR SOURCE STRING. THIS "
          "MIGHT BE A MALFORMED LANGUAGE PACK OR  MIGHT BE A SERIOUS ERROR IN "
          "THE EIS PARSER SYSTEM";
    }

    auto add_string = [&](error code, const char* key) {
      if ((errcode & code) != 0) strings.append(key, err_string);
    };

    add_string(err_bad_value, "err_bad_value");
    add_string(err_malformed, "err_malformed");
    add_string(err_wrong_parameters, "err_wrong_parameters");
    add_string(err_maths_domain, "err_maths_domain");
    add_string(err_not_found, "err_not_found");
    add_string(err_has_emplaced, "err_has_emplaced");
    add_string(err_has_deferred, "err_has_deferred");
    add_string(err_bracketed_constant, "err_bracketed_constant");
    add_string(err_extra_bracket, "err_extra_bracket");
    add_string(err_bad_stack, "err_bad_stack");
    add_string(err_bad_parser, "err_bad_parser");

    //Errors from deck parser
    add_string(err_no_luns, "err_no_luns");
    add_string(err_no_file, "err_no_file");
    add_string(err_malformed_file, "err_malformed_file");
    add_string(err_mismatched_begin_end, "err_mismatched_begin_end");
    add_string(err_deck_too_deep, "err_deck_too_deep");
    add_string(err_deck_empty_block, "err_deck_empty_block");
    add_string(err_root_keys, "err_root_keys");

    //Errors from deck evaluation
    add_string(err_unknown_block, "err_unknown_block");
    add_string(err_unknown_key, "err_unknown_key");
    add_string(err_bad_key, "err_bad_key");
    add_string(err_host, "err_host");
    add_string(err_text, "err_text");
    add_string(err_interop, "err_interop");
    add_string(err_out_of_range, "err_out_of_range");
    add_string(err_stack_params, "err_stack_params");
    add_string(err_no_deriv, "err_no_deriv");
    add_string(err_extra_results, "err_extra_results");
    add_string(err_bad_deck_definition, "err_defn_invalid");
    add_string(err_invalid_minify, "err_invalid_minify");

    if (err_string.empty()) err_string = "No error text found";

    return result;
  }

  inline error_strings error_handler::get_error_string(count_type index) {
    if (!is_init) {
      error init_err;
      init(init_err);
    }

    if (index >= get_error_count()) return error_strings();

    return get_error_string_from_code(errors[index].errcode);
  }

  inline std::string error_handler::get_error_report(count_type index) {
    if (!is_init) {
      error init_err;
      init(init_err);
    }

    std::string report;

    if (errors.empty()) {
      append_string(report, "No stored errors");
      return report;
    }

    if (index >= get_error_count()) {
      append_string(report, "Error out of range");
      return report;
    }

    const error_item& item = errors[index];
    const error_cause cause = get_error_cause(index);
    const error_strings described = get_error_string(index);
    const offset_type charpos = cause.charindex;
    const offset_type line_number = cause.line_number;
    const offset_type context_line_number = cause.context_line_number;
    const std::string& filename = cause.filename;
    const std::string& context_filename = cause.context_filename;
    const std::string& errname = cause.err_text;
    const std::string unknown = "{unknown}";

    //Set up the replacement strings
    key_value_store errstr_store;

    errstr_store.store("error", described.err_string.empty()
        ? unknown : described.err_string);
    errstr_store.store("errpart", described.err_source.empty()
        ? unknown : described.err_source);

    if (line_number > 0) {
      errstr_store.store("errline", std::to_string(line_number));
    } else {
      errstr_store.store("errline", unknown);
    }
    if (context_line_number > 0) {
      errstr_store.store("context_errline",
          std::to_string(context_line_number));
    } else {
      errstr_store.store("context_errline", unknown);
    }

    if (!filename.empty()) errstr_store.store("errfile", filename);

    if (!context_filename.empty()) {
      errstr_store.store("context_errfile", context_filename);
    } else {
      errstr_store.store("context_errfile", unknown);
    }

    if (charpos > 0) {
      errstr_store.store("charpos", std::to_string(charpos));
    } else {
      errstr_store.store("charpos", unknown);
    }

    if (!errname.empty()) {
      errstr_store.store("errtext", errname);
    } else {
      errstr_store.store("errtext", unknown);
    }

    //Now start building the actual error report

    append_string(report, std::string(80, '='));

    //Report that there is an error
    std::string temp;
    if (!filename.empty() && line_number > 0) {
      strings.get("err_is_error", temp);
    } else if (!filename.empty()) {
      if (filename == unknown) {
        strings.get("err_is_error_only", temp);
      } else {
        strings.get("err_is_error_file", temp);
      }
    } else {
      strings.get("err_is_error_only", temp);
    }
    errstr_store.format_fill(temp);
    append_string(report, temp);

    //Print the context and location information
    if (!item.full_line.empty()) {
      const offset_type nchar = static_cast<offset_type>(
          std::floor(std::log10(static_cast<double>(charpos)))) + 1;
      offset_type nline = 0;
      if (line_number > 0) nline = static_cast<offset_type>(
          std::floor(std::log10(static_cast<double>(line_number)))) + 1;
      const offset_type nspace = std::max(nchar, nline);
      const std::string linestr =
          detail::format_integer(std::max<offset_type>(line_number, 1), nspace);
      const std::string posstr = detail::format_integer(charpos, nspace);
      append_string(report, "");

      const offset_type full_length =
          static_cast<offset_type>(item.full_line.size());
      const offset_type spos = std::max<offset_type>(1,
          item.full_line_pos - 10);
      const offset_type epos = std::min(full_length, item.full_line_pos
          + std::max<offset_type>(10, static_cast<offset_type>(errname.size())));
      const std::string predots = (spos != 1) ? "..." : "";
      const std::string postdots = (epos != full_length) ? "..." : "";
      const std::string visible = (epos >= spos)
          ? item.full_line.substr(spos - 1, epos - spos + 1) : "";
      append_string(report, linestr + " : " + predots + visible + postdots);

      const offset_type offset = std::max<offset_type>(item.full_line_pos - spos
          + static_cast<offset_type>(predots.size()) + nspace + 3, 0);
      append_string(report, std::string(offset, ' ') + "^");
      append_string(report, std::string(offset, ' ') + posstr);
    } else {
      if (charpos > 0 || !errname.empty()) {
        //Report the error location
        strings.get("err_report_place", temp);
        errstr_store.format_fill(temp);
        append_string(report, temp);
      }
    }

    bool has_context;
    if (!context_filename.empty() || context_line_number > 0) {
      if (!filename.empty()) {
        has_context = filename != context_filename;
      } else {
        has_context = true;
      }
      has_context = has_context || line_number != context_line_number;
    } else {
      has_context = false;
    }

    if (has_context) {
      strings.get("err_proximate_source", temp);
      errstr_store.format_fill(temp);
      append_string(report, temp);
    }

    append_string(report, std::string(80, '='));
    append_string(report, "");

    return report;
  }

}

#endif
