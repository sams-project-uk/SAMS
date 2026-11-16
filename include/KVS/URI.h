/*
 *    Copyright 2025 SAMS Team
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef SAMS_KVS_URI_H
#define SAMS_KVS_URI_H

#include <string>
#include <stdexcept>
namespace SAMS {
    struct URI {
        std::string scheme; /** e.g., "http", "https", "ftp", "file" **/
        std::string authority; /** e.g., "user:password@www.example.com:80"; empty if there is no authority **/
        std::string userInfo; /** e.g., "user:password" **/
        std::string host; /** e.g., "www.example.com" **/
        int port = -1; /** e.g., 80, 443; -1 if not specified **/
        std::string path; /** e.g., "/path/to/resource" **/
        std::string query; /** e.g., "key1=value1&key2=value2" **/
        std::string fragment; /** e.g., "section1" **/

        /**
         * Parse a URI string into its components.
         * The forms of RFC 3986 are accepted: scheme://authority/path, scheme:/absolute/path and scheme:relative/path,
         * each optionally followed by ?query and #fragment. Without an authority the host is empty
         * NOTE THIS IS NOT AN RFC 3986 COMPLIANT PARSER - IT IS JUST GOOD ENOUGH
         * IN PARTICULAR IT DOES NOT HANDLE ANY ENCODINGS OR IPV6 ADDRESSES
         * IF BOOST IS EVER ADDED AS A DEPENDENCY REPLACE WITH BOOST
         * @param uriStr The URI string to parse.
         * @return A URI struct with the parsed components.
         * @throws std::runtime_error if the URI has no scheme.
         * @throws std::invalid_argument if the port is not a number.
         */
        static URI parse(const std::string& uriStr) {
            URI uri;
            size_t pos = 0, end;

            // Parse scheme
            end = uriStr.find(':', pos);
            if (end != std::string::npos) {
                uri.scheme = uriStr.substr(pos, end - pos);
                pos = end + 1; // Move past ':'
            } else {
                throw std::runtime_error("Error: Invalid URI, missing scheme");
            }

            //All other parts are optional

            // An authority follows "//". Without one the path starts straight after the scheme
            if (uriStr.compare(pos, 2, "//") == 0) {
                pos += 2; // Move past "//"

                // The authority runs to the start of the path, query or fragment
                end = uriStr.find_first_of("/?#", pos);
                size_t authorityEnd = (end == std::string::npos) ? uriStr.length() : end;
                uri.authority = uriStr.substr(pos, authorityEnd - pos);

                // Parse userInfo
                end = uriStr.find('@', pos);
                if (end != std::string::npos && end < authorityEnd) {
                    uri.userInfo = uriStr.substr(pos, end - pos);
                    pos = end + 1; // Move past '@'
                }

                // Parse host and port. An empty port is the same as no port
                size_t colonPos = uriStr.find(':', pos);
                if (colonPos != std::string::npos && colonPos < authorityEnd) {
                    uri.host = uriStr.substr(pos, colonPos - pos);
                    if (colonPos + 1 < authorityEnd) uri.port = std::stoi(uriStr.substr(colonPos + 1, authorityEnd - colonPos - 1));
                } else {
                    uri.host = uriStr.substr(pos, authorityEnd - pos);
                }
                pos = authorityEnd;
            }

            // Parse path
            end = uriStr.find_first_of("?#", pos);
            size_t pathEnd = (end == std::string::npos) ? uriStr.length() : end;
            uri.path = uriStr.substr(pos, pathEnd - pos);
            pos = pathEnd;

            // Parse query
            if (pos < uriStr.length() && uriStr[pos] == '?') {
                end = uriStr.find('#', pos);
                size_t queryEnd = (end == std::string::npos) ? uriStr.length() : end;
                uri.query = uriStr.substr(pos + 1, queryEnd - pos - 1);
                pos = queryEnd;
            }

            // Parse fragment
            if (pos < uriStr.length() && uriStr[pos] == '#') {
                uri.fragment = uriStr.substr(pos + 1);
            }
            return uri;
        }

        /**
         * Build the path of a file from the URI.
         * A local path takes the authority as the start of a relative path, so scheme://data/file is data/file.
         * Otherwise the authority is a host and the path is only the path component
         * @param local Whether the authority is the start of the path
         * @return The path
         */
        std::string buildPath(bool local = true) const {
            if (local) return authority + path;
            return path;
        }
    };
}

#endif //SAMS_KVS_URI_H