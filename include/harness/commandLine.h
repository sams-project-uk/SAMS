#ifndef SAMS_COMMANDLINE_H
#define SAMS_COMMANDLINE_H

#ifndef _WIN32
#include <getopt.h>
#endif
#include <map>
#include <string>
#include <vector>

namespace SAMS
{
    /**
     * Command line options, parsed with getopt_long. Only long options, as --name or --name value
     */
    class commandLine
    {
    public:
        /**
         * Whether an option takes a value. An optional value is only given as --name=value
         */
        enum class argumentType { none, required, optional };

    private:
        std::map<std::string, std::string> descriptions;
        std::map<std::string, argumentType> takesValue;
        std::map<std::string, std::string> values;
        std::vector<std::string> positional;
        std::string errorMessage;

    public:
        /**
         * Register an option
         * @param name The name of the option, without the leading --
         * @param description A description of the option
         * @param argument Whether the option takes a value
         */
        void registerOption(const std::string &name, const std::string &description, argumentType argument)
        {
            descriptions[name] = description;
            takesValue[name] = argument;
        }

        /**
         * Parse the command line. Unknown options are errors unless ACCEPT_UNKNOWN_CLI is defined
         * @return Whether the command line is valid. If not, error() gives the reason
         */
        bool parse(int argc, char **argv)
        {
#ifdef _WIN32
            //No getopt_long on Windows
            for (int i = 1; i < argc; ++i){
                const std::string arg = argv[i];
                if (arg == "--"){
                    for (++i; i < argc; ++i) positional.push_back(argv[i]);
                    break;
                }
                if (arg.size() < 2 || arg[0] != '-'){
                    positional.push_back(arg);
                    continue;
                }
                const size_t equals = arg.find('=');
                const std::string name = arg.compare(0, 2, "--") == 0 ? arg.substr(2, equals == std::string::npos ? std::string::npos : equals - 2) : std::string();
                if (descriptions.count(name) == 0){
#ifndef ACCEPT_UNKNOWN_CLI
                    errorMessage = "Unknown option " + arg;
                    return false;
#else
                    continue;
#endif
                }
                if (takesValue[name] == argumentType::none){
                    values[name] = "";
                } else if (equals != std::string::npos){
                    values[name] = arg.substr(equals + 1);
                } else if (takesValue[name] == argumentType::optional){
                    values[name] = "";
                } else if (i + 1 < argc){
                    values[name] = argv[++i];
                } else {
                    errorMessage = "Missing value for " + arg;
                    return false;
                }
            }
            return true;
#else
            std::vector<option> longOptions;
            for (const auto &[name, description] : descriptions){
                const argumentType argument = takesValue[name];
                longOptions.push_back({name.c_str(), argument == argumentType::required ? required_argument : argument == argumentType::optional ? optional_argument : no_argument, nullptr, 0});
            }
            longOptions.push_back({nullptr, 0, nullptr, 0});
            //- returns positional arguments as 1, in order. : returns a missing value as :
            opterr = 0;
            int index = 0;
            int result;
            while ((result = getopt_long(argc, argv, "-:", longOptions.data(), &index)) != -1){
                if (result == 0){
                    values[longOptions[index].name] = optarg ? optarg : "";
                } else if (result == 1){
                    positional.push_back(optarg);
                } else if (result == ':'){
                    errorMessage = std::string("Missing value for ") + argv[optind - 1];
                    return false;
                } else {
#ifndef ACCEPT_UNKNOWN_CLI
                    errorMessage = std::string("Unknown option ") + argv[optind - 1];
                    return false;
#endif
                }
            }
            return true;
#endif
        }

        /**
         * Whether an option was given
         */
        bool has(const std::string &name) const { return values.count(name) > 0; }

        /**
         * Whether a flag was given
         */
        bool flagSet(const std::string &name) const { return values.count(name) > 0; }

        /**
         * The value of an option that was given
         */
        const std::string &get(const std::string &name) const { return values.at(name); }

        /**
         * The arguments that are not options, in order
         */
        const std::vector<std::string> &positionalArguments() const { return positional; }

        /**
         * The descriptions of the registered options, by name
         */
        const std::map<std::string, std::string> &getDescriptions() const { return descriptions; }

        /**
         * Why the command line is not valid
         */
        const std::string &error() const { return errorMessage; }
    };
}

#endif // SAMS_COMMANDLINE_H
