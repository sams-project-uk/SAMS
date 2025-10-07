#ifndef SAMS_KVS_URI_H
#define SAMS_KVS_URI_H

#include <string>
#include <stdexcept>
namespace SAMS {
    struct URI {
        std::string scheme; /** e.g., "http", "https", "ftp", "file" **/
        std::string userInfo; /** e.g., "user:password" **/
        std::string host; /** e.g., "www.example.com" **/
        int port = -1; /** e.g., 80, 443; -1 if not specified **/
        std::string path; /** e.g., "/path/to/resource" **/
        std::string query; /** e.g., "key1=value1&key2=value2" **/
        std::string fragment; /** e.g., "section1" **/

        /**
         * Parse a URI string into its components.
         * NOTE THIS IS NOT AN RFC 3986 COMPLIANT PARSER - IT IS JUST GOOD ENOUGH
         * IN PARTICULAR IT DOES NOT HANDLE ANY ENCODINGS OR IPV6 ADDRESSES
         * IF BOOST IS EVER ADDED AS A DEPENDENCY REPLACE WITH BOOST
         * @param uriStr The URI string to parse.
         * @return A URI struct with the parsed components.
         * @throws std::runtime_error if the URI is invalid.
         */
        static URI parse(const std::string& uriStr) {
            URI uri;
            size_t pos = 0, end;

            // Parse scheme
            end = uriStr.find("://", pos);
            if (end != std::string::npos) {
                uri.scheme = uriStr.substr(pos, end - pos);
                pos = end + 3; // Move past "://"
            } else {
                throw std::runtime_error("Error: Invalid URI, missing scheme");
            }

            //All other parts are optional

            // Parse userInfo and host
            end = uriStr.find('@', pos);
            if (end != std::string::npos) {
                uri.userInfo = uriStr.substr(pos, end - pos);
                pos = end + 1; // Move past '@'
            }

            // Parse host and port
            end = uriStr.find_first_of("/?:#", pos);
            size_t hostPortEnd = (end == std::string::npos) ? uriStr.length() : end;
            size_t colonPos = uriStr.find(':', pos);
            if (colonPos != std::string::npos && colonPos < hostPortEnd) {
                uri.host = uriStr.substr(pos, colonPos - pos);
                uri.port = std::stoi(uriStr.substr(colonPos + 1, hostPortEnd - colonPos - 1));
            } else {
                uri.host = uriStr.substr(pos, hostPortEnd - pos);
            }
            pos = hostPortEnd;

            // Parse path
            if (pos < uriStr.length() && uriStr[pos] == '/') {
                end = uriStr.find_first_of("?#", pos);
                size_t pathEnd = (end == std::string::npos) ? uriStr.length() : end;
                uri.path = uriStr.substr(pos, pathEnd - pos);
                pos = pathEnd;
            }

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
    };
}

#endif //SAMS_KVS_URI_H