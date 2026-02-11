#ifndef SAMS_WELCOME_H
#define SAMS_WELCOME_H

#include "streams.h"

namespace SAMS {

    //Raw VT100 escape codes for colored terminal output
    //Bit of a hack, but better than introducing 
    //a dependency on curses or similar just for this
    inline const char VT100Red[] = "\033[31m";
    inline const char VT100Green[] = "\033[32m";
    inline const char VT100Yellow[] = "\033[33m";
    inline const char VT100Blue[] = "\033[34m";
    inline const char VT100Magenta[] = "\033[35m";
    inline const char VT100Cyan[] = "\033[36m";
    inline const char VT100Reset[] = "\033[0m";
    inline const char VT100Normal[] = "\033[22m";
    inline const char VT100Bold[] = "\033[1m";
    inline const char VT100Underline[] = "\033[4m";
    inline const char VT100Reversed[] = "\033[7m";

    //Perhaps move to harnessDef.h later?
    inline const int majorVersion = 0;
    inline const int minorVersion = 0;
    inline const int patchVersion = 3;

    #include <unistd.h>

    namespace {
        std::string centrePrint(const std::string& str, int totalWidth) {
            int padding = (totalWidth - static_cast<int>(str.length())) / 2
                            + (totalWidth - static_cast<int>(str.length())) % 2;
            return std::string(padding, ' ') + str;
        }

        //Use a concept for "Any number of char*s"
        template<typename T, typename... T_others>
        std::string colorSet(T colorCode, T_others... others) {
            if (isatty(fileno(stdout)) == 0) {
                //Not a terminal, do not print color codes
                return "";
            }
            if constexpr (sizeof...(others) > 0) {
                return std::string(colorCode) + colorSet(others...);
            } else{
                return std::string(colorCode);
            } 
        }
    }

    /**
     * Print the SAMS welcome message
     */
    inline void printWelcomeMessage() {

        std::string versionStr = "Version " + std::to_string(majorVersion) + "." + std::to_string(minorVersion) + "." + std::to_string(patchVersion);

        //Logo from https://www.asciiart.eu/text-to-ascii-art
        //REPLACE BEFORE FULL RELEASE
        SAMS::cout << colorSet(VT100Red,VT100Bold);
        SAMS::cout << R"(________  ________  _____ ______   ________        )" << std::endl;
        SAMS::cout << R"(|\   ____\|\   __  \|\   _ \  _   \|\   ____\      )" << std::endl;
        SAMS::cout << R"(\ \  \___|\ \  \|\  \ \  \\\__\ \  \ \  \___|_     )" << std::endl;
        SAMS::cout << R"( \ \_____  \ \   __  \ \  \\|__| \  \ \_____  \    )" << std::endl;
        SAMS::cout << colorSet(VT100Green);
        SAMS::cout << R"(  \|____|\  \ \  \ \  \ \  \    \ \  \|____|\  \   )" << std::endl;
        SAMS::cout << R"(    ____\_\  \ \__\ \__\ \__\    \ \__\____\_\  \  )" << std::endl;
        SAMS::cout << R"(   |\_________\|__|\|__|\|__|     \|__|\_________\ )" << std::endl;
        SAMS::cout << R"(   \|_________|                       \|_________| )" << std::endl;
        SAMS::cout << colorSet(VT100Red);
        SAMS::cout << centrePrint(versionStr, 52) << std::endl;
        SAMS::cout << colorSet(VT100Reset);
        SAMS::cout <<  "====================================================" << std::endl;
        SAMS::cout <<  colorSet(VT100Cyan);
        SAMS::cout << centrePrint("Solar Atmospheric Modeling Suite (SAMS)", 52) << std::endl;
        SAMS::cout << centrePrint("Released under the Apache 2.0 License", 52) << std::endl;
        //Construct the version string and centre it
        SAMS::cout << colorSet(VT100Reset);
        SAMS::cout << "=====================================================" << std::endl;
        SAMS::cout << "Run information:" << std::endl;
        SAMS::cout << "=====================================================" << std::endl;
        portableWrapper::printParallelizationInfo();
        SAMS::cout << "=====================================================" << std::endl;
        SAMS::debug1 <<"Debug level 1 output enabled\n";
        SAMS::debug2 <<"Debug level 2 output enabled\n";
        SAMS::debug3 <<"Debug level 3 output enabled\n";

    }

    inline void finishWelcomeMessage() {
        SAMS::cout << "=====================================================" << std::endl;
    }

} //namespace SAMS

#endif