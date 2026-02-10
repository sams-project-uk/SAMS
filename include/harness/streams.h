#ifndef SAMS_STREAMS_H
#define SAMS_STREAMS_H

#include <iostream>
#include <fstream>

namespace SAMS{

    /**
     * Class wrapping a stream to only output from rank 0 in an MPI environment
     */
    template<typename TStream>
    class streamWrapper
    {
        int rank=0;
        //Has to be a pointer to allow reassignment
        TStream* stream=nullptr;
        //Internal file stream if opened via filename
        std::ofstream fileStream;
        //Should the filename have rank appended
        bool appendRank=false;
        //Filename to open if needed
        std::string filename;

    public:

        /*
         * Default constructor - uses std::cout
         */
        streamWrapper() {stream = &std::cout;}

        /**
         * Constructor with reference to external stream
         */
        streamWrapper(TStream& s) : stream(&s) {}

        /**
         * Constructor opening a file
         */
        streamWrapper(const std::string &filename, bool appendRank=false) {
            this->filename = filename;
            if (!appendRank) {
                fileStream.open(filename);
                stream = &fileStream;
            }
            this->appendRank = appendRank;
        }

        /**
         * Assignment operator for setting the stream to an external stream
         * @param s Reference to the external stream
         */
        streamWrapper& operator=(TStream& s) {
            if (fileStream.is_open()) {
                fileStream.close();
            }
            stream = &s;
            return *this;
        }

        /**
         * Open a file for output
         * @param filename Name of the file to open
         */
        void openFile(const std::string &filename) {
            if (fileStream.is_open()) {
                fileStream.close();
            }
            fileStream.open(filename);
            stream = fileStream;
        }

        // Generic values (numbers, strings, user types with operator<<, etc.)
        template <typename T>
        streamWrapper &operator<<(T &&value)
        {
            TStream &s = *stream;
            if (rank==0) s << std::forward<T>(value);
            return *this;
        }

        // std::ostream manipulators (e.g. std::endl)
        streamWrapper &operator<<(std::ostream &(*manip)(std::ostream &))
        {
            TStream &s = *stream;
            if (rank==0) manip(s);
            return *this;
        }

        // std::ios_base manipulators (e.g. std::hex, std::dec)
        streamWrapper &operator<<(std::ios_base &(*manip)(std::ios_base &))
        {
            TStream &s = *stream;
            if (rank==0) manip(s);
            return *this;
        }

        void setRank(int r) {
            if (appendRank){
                //Have to reopen the file with the rank appended
                if (fileStream.is_open()) {
                    fileStream.close();
                }
                //Split any extension from filename
                size_t lastdot = filename.find_last_of(".");
                std::string baseName = filename;
                std::string extension = "";
                if (lastdot != std::string::npos) {
                    baseName = filename.substr(0, lastdot);
                    extension = filename.substr(lastdot);
                }
                std::string rankedFilename = baseName + "_" + std::to_string(r) + extension;
                fileStream.open(rankedFilename);
                stream = &fileStream;
            } else {
                rank = r;
            }
        }
    };

    /**
     * Dummy class that implements stream semantics but does nothing
     */
    class nullStream {
    public:
        template <typename T>
        nullStream &operator<<[[maybe_unused]](T &&) {
            return *this;;
        }

        // std::ostream manipulators (e.g. std::endl)
        nullStream &operator<<([[maybe_unused]] std::ostream &(*manip)(std::ostream &))
        {
            return *this;
        }

        // std::ios_base manipulators (e.g. std::hex, std::dec)
        nullStream &operator<<([[maybe_unused]] std::ios_base &(*manip)(std::ios_base &))
        {
            return *this;
        }


        void setRank([[maybe_unused]] int) {
            // Do nothing
        }
    };

    // Inline instance usable like: SAMS::mpiout << "hello" << std::endl;
    // cout - standard output
    // cerr - standard error
    // cerrAll - standard error output from all ranks
    // debug1, debug2, debug3 - debug output streams with increasing verbosity
    // debugAll1, debugAll2, debugAll3 - debug output streams that output from all ranks
    inline streamWrapper<std::ostream> cout{std::cout};
    inline streamWrapper<std::ostream> cerr{std::cerr};
    inline streamWrapper<std::ostream> cerrAll{std::cerr};
    #if !defined(SAMS_DEBUG) || SAMS_DEBUG == 0
    //No debug state set - disable all debug output
    inline nullStream debug1;
    inline nullStream debug2;
    inline nullStream debug3;
    inline nullStream debugAll1;
    inline nullStream debugAll2;
    inline nullStream debugAll3;
    #elif SAMS_DEBUG == 1
    //Debug level 1 - enable debug1 only
    #ifdef SAMS_DEBUG_TO_STDERR
    inline streamWrapper<std::ostream> debug1{std::cerr};
    #else
    inline streamWrapper<std::ostream> debug1{"debug1.log", false};
    #endif
    inline nullStream debug2;
    inline nullStream debug3;
    #ifdef SAMS_DEBUG_TO_STDERR
    inline streamWrapper<std::ostream> debugAll1{std::cerr};
    #else
    inline streamWrapper<std::ostream> debugAll1{"debug1.log", true};
    #endif //SAMS_DEBUG_TO_STDERR
    inline nullStream debugAll2;
    inline nullStream debugAll3;
    #elif SAMS_DEBUG == 2
    //Debug level 2 - enable debug1 and debug2
    #ifdef SAMS_DEBUG_TO_STDERR
    inline streamWrapper<std::ostream> debug1{std::cerr};
    inline streamWrapper<std::ostream> debug2{std::cerr};
    #else
    inline streamWrapper<std::ostream> debug1{"debug1.log", false};
    inline streamWrapper<std::ostream> debug2{"debug2.log", false};
    #endif
    inline nullStream debug3;
    #ifdef SAMS_DEBUG_TO_STDERR
    inline streamWrapper<std::ostream> debugAll1{std::cerr};
    inline streamWrapper<std::ostream> debugAll2{std::cerr};
    #else
    inline streamWrapper<std::ostream> debugAll1{"debug1.log", true};
    inline streamWrapper<std::ostream> debugAll2{"debug2.log", true};
    #endif //SAMS_DEBUG_TO_STDERR
    inline nullStream debugAll3;
    #elif SAMS_DEBUG >= 3
    //Debug level 3 or higher - enable all debug output
    #ifdef SAMS_DEBUG_TO_STDERR
    inline streamWrapper<std::ostream> debug1{std::cerr};
    inline streamWrapper<std::ostream> debug2{std::cerr};
    inline streamWrapper<std::ostream> debug3{std::cerr};
    inline streamWrapper<std::ostream> debugAll1{std::cerr};
    inline streamWrapper<std::ostream> debugAll2{std::cerr};
    inline streamWrapper<std::ostream> debugAll3{std::cerr};
    #else
    inline streamWrapper<std::ostream> debug1{"debug1.log", false};
    inline streamWrapper<std::ostream> debug2{"debug2.log", false};
    inline streamWrapper<std::ostream> debug3{"debug3.log", false};
    inline streamWrapper<std::ostream> debugAll1{"debug1.log", true};
    inline streamWrapper<std::ostream> debugAll2{"debug2.log", true};
    inline streamWrapper<std::ostream> debugAll3{"debug3.log", true};
    #endif // SAMS_DEBUG_TO_STDERR
    #endif // SAMS_DEBUG

} // namespace SAMS

#endif