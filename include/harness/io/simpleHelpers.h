#ifndef SAMS_IO_SIMPLEHELPERS_H
#define SAMS_IO_SIMPLEHELPERS_H

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "mpiDefaultTypes.h"

namespace SAMS::IO
{

#ifdef USE_MPI
    /**
     * Broadcast a string from rank 0
     * @param comm The communicator
     * @param value The string, which is replaced by rank 0's on the other ranks
     */
    inline void broadcastFromRoot(SAMSMPI::comm comm, std::string &value)
    {
        unsigned long long length = value.size();
        MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
        value.resize(length);
        MPI_Bcast(value.data(), static_cast<int>(length), MPI_CHAR, 0, comm);
    }

    /**
     * Read a file on rank 0 and broadcast its content. Throws on rank 0 if the file does not exist, so the caller must abort
     * @param comm The communicator
     * @param fileName The name of the file
     * @return The content of the file, on every rank
     */
    inline std::string readOnRoot(SAMSMPI::comm comm, const std::string &fileName)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);
        std::string content;
        if (rank == 0) {
            if (!std::filesystem::exists(fileName)) throw std::runtime_error("File " + fileName + " does not exist");
            std::ifstream file(fileName);
            std::stringstream buffer;
            buffer << file.rdbuf();
            content = buffer.str();
        }
        broadcastFromRoot(comm, content);
        return content;
    }

    /**
     * Whether a directory exists, checked on rank 0 and broadcast
     * @param comm The communicator
     * @param name The name of the directory
     */
    inline bool isDirectoryOnRoot(SAMSMPI::comm comm, const std::string &name)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);
        int isDirectory = 0;
        if (rank == 0) isDirectory = std::filesystem::is_directory(name) ? 1 : 0;
        MPI_Bcast(&isDirectory, 1, MPI_INT, 0, comm);
        return isDirectory != 0;
    }

    /**
     * Whether a file exists, checked on rank 0 and broadcast
     * @param comm The communicator
     * @param fileName The name of the file
     */
    inline bool existsOnRoot(SAMSMPI::comm comm, const std::string &fileName)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);
        int exists = 0;
        if (rank == 0) exists = std::filesystem::exists(fileName) ? 1 : 0;
        MPI_Bcast(&exists, 1, MPI_INT, 0, comm);
        return exists != 0;
    }
#else
    /**
     * Read a file. Throws if the file does not exist
     * @param fileName The name of the file
     * @return The content of the file
     */
    inline std::string readOnRoot(const std::string &fileName)
    {
        if (!std::filesystem::exists(fileName)) throw std::runtime_error("File " + fileName + " does not exist");
        std::ifstream file(fileName);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    /**
     * Whether a file exists
     * @param fileName The name of the file
     */
    inline bool existsOnRoot(const std::string &fileName)
    {
        return std::filesystem::exists(fileName);
    }

    /**
     * Whether a directory exists
     * @param name The name of the directory
     */
    inline bool isDirectoryOnRoot(const std::string &name)
    {
        return std::filesystem::is_directory(name);
    }
#endif

}

#endif //SAMS_IO_SIMPLEHELPERS_H
