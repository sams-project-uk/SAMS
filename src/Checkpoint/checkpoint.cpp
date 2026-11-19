// Restart (checkpoint) module: file format, writing, reading and the decision when to write.
// See checkpoint.h for the description and the environment variables.

#include "checkpoint.h"
#include "variableRegistry.h"

#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iostream>

namespace SAMS
{
    namespace
    {
        constexpr char magic[8] = {'S', 'A', 'M', 'S', 'C', 'K', 'P', 'T'};
        constexpr uint32_t version = 1;

        std::string rankFile(const std::string &dir, int rank)
        {
            std::ostringstream ss;
            ss << dir << "/ckpt_rank" << std::setw(5) << std::setfill('0') << rank << ".bin";
            return ss.str();
        }

        template<typename T> void put(std::ofstream &f, const T &v) { f.write(reinterpret_cast<const char *>(&v), sizeof(T)); }
        void putString(std::ofstream &f, const std::string &s)
        {
            put<uint32_t>(f, static_cast<uint32_t>(s.size()));
            f.write(s.data(), static_cast<std::streamsize>(s.size()));
        }
        template<typename T> T get(std::ifstream &f, const std::string &file)
        {
            T v{};
            f.read(reinterpret_cast<char *>(&v), sizeof(T));
            if (!f) throw std::runtime_error("restart file " + file + " is truncated or unreadable");
            return v;
        }
        std::string getString(std::ifstream &f, const std::string &file)
        {
            const uint32_t n = get<uint32_t>(f, file);
            if (n > (1u << 20)) throw std::runtime_error("restart file " + file + " is corrupt (name length)");
            std::string s(n, '\0');
            f.read(s.data(), n);
            if (!f) throw std::runtime_error("restart file " + file + " is truncated or unreadable");
            return s;
        }

        void barrier([[maybe_unused]] MPI_Comm comm)
        {
#ifdef USE_MPI
            MPI_Barrier(comm);
#endif
        }

        void writeInfo(const std::string &dir, const checkpointHeader &h, const std::string &status)
        {
            std::ofstream f(dir + "/checkpoint_info.txt", std::ios::trunc);
            f << std::setprecision(17)
              << "time " << h.time << "\n"
              << "step " << h.step << "\n"
              << "dt " << h.dt << "\n"
              << "snapshots_written " << h.outputCount << "\n"
              << "ranks " << h.nRanks << "\n"
              << "status " << status << "\n";
        }
    } // namespace

    void checkpointManager::configureFromEnvironment()
    {
        auto env = [](const char *k) -> const char * { const char *v = std::getenv(k); return (v && *v) ? v : nullptr; };
        if (const char *v = env("SAMS_CHECKPOINT_DIR")) dir = v;
        if (const char *v = env("SAMS_CHECKPOINT_DT")) dtSim = std::stod(v);
        if (const char *v = env("SAMS_CHECKPOINT_STEPS")) everySteps = std::stoll(v);
        if (const char *v = env("SAMS_CHECKPOINT_WALLTIME")) wallLimit = std::stod(v);
        if (const char *v = env("SAMS_RESTART")) {
            const std::string s = v;
            restartDir = (s == "1" || s == "yes" || s == "true") ? dir : s;
        }
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_KOKKOS)
        if (enabled() || restartRequested())
            throw std::runtime_error("Checkpoint/restart supports host-memory builds only (not CUDA/HIP/Kokkos)");
#endif
    }

    checkpointManager::action checkpointManager::afterStep(double time, long long step, double wallSeconds, [[maybe_unused]] MPI_Comm comm)
    {
        if (!enabled()) return action::none;
        bool due = false;
        if (dtSim > 0.0) {
            if (nextSim < 0.0) nextSim = dtSim;        // first call of a new run (read() sets it on restart)
            if (time >= nextSim) {
                due = true;
                while (nextSim <= time) nextSim += dtSim;
            }
        }
        if (everySteps > 0 && step % everySteps == 0) due = true;
        if (wallLimit > 0.0) {
            int stop = wallSeconds >= wallLimit ? 1 : 0;
#ifdef USE_MPI
            MPI_Bcast(&stop, 1, MPI_INT, 0, comm);
#endif
            if (stop) return action::writeAndStop;
        }
        return due ? action::write : action::none;
    }

    void checkpointManager::write(variableRegistry &varReg, const checkpointHeader &h, const checkpointState &state,
                                  MPI_Comm comm, const std::string &status)
    {
        namespace fs = std::filesystem;
        if (h.rank == 0) fs::create_directories(dir);
        barrier(comm);

        // Phase 1: every rank writes <file>.tmp
        const std::string file = rankFile(dir, h.rank);
        {
            std::ofstream f(file + ".tmp", std::ios::binary | std::ios::trunc);
            if (!f) throw std::runtime_error("cannot open checkpoint file " + file + ".tmp for writing");
            f.write(magic, 8);
            put<uint32_t>(f, version);
            put<int32_t>(f, h.nRanks);
            put<int32_t>(f, h.rank);
            put<double>(f, h.time);
            put<double>(f, h.dt);
            put<int64_t>(f, h.step);
            put<int64_t>(f, h.outputCount);
            put<uint64_t>(f, state.reals.size());
            for (const auto &[k, v] : state.reals) { putString(f, k); put<double>(f, v); }
            put<uint64_t>(f, state.ints.size());
            for (const auto &[k, v] : state.ints) { putString(f, k); put<int64_t>(f, v); }
            std::vector<std::string> names;
            for (const auto &n : varReg.getVariableNames())
                if (varReg.getVariable(n).data() && varReg.getVariable(n).getAllocatedBytes() > 0) names.push_back(n);
            put<uint64_t>(f, names.size());
            for (const auto &n : names) {
                const auto &v = varReg.getVariable(n);
                putString(f, n);
                put<uint64_t>(f, v.getAllocatedBytes());
                f.write(static_cast<const char *>(v.data()), static_cast<std::streamsize>(v.getAllocatedBytes()));
            }
            f.flush();
            if (!f) throw std::runtime_error("error writing checkpoint file " + file + ".tmp (disk full?)");
        }
        barrier(comm);   // every rank has a complete new file before any old one is replaced

        // Phase 2: keep the previous checkpoint as .prev, move the new one into place
        std::error_code ec;
        if (fs::exists(file)) fs::rename(file, file + ".prev", ec);
        fs::rename(file + ".tmp", file, ec);
        if (ec) throw std::runtime_error("cannot move checkpoint " + file + ".tmp into place: " + ec.message());
        barrier(comm);
        if (h.rank == 0) {
            writeInfo(dir, h, status);
            std::cout << "Checkpoint written to " << dir << " at step " << h.step << ", time " << h.time << std::endl;
        }
    }

    void checkpointManager::read(variableRegistry &varReg, int rank, int nRanks, checkpointHeader &h, checkpointState &state)
    {
        const std::string file = rankFile(restartDir, rank);
        std::ifstream f(file, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open restart file " + file);
        char m[8];
        f.read(m, 8);
        if (!f || std::memcmp(m, magic, 8) != 0) throw std::runtime_error(file + " is not a SAMS restart file");
        const uint32_t ver = get<uint32_t>(f, file);
        if (ver != version) throw std::runtime_error(file + ": unsupported restart file version " + std::to_string(ver));
        h.nRanks = get<int32_t>(f, file);
        h.rank = get<int32_t>(f, file);
        if (h.nRanks != nRanks)
            throw std::runtime_error("restart files were written by " + std::to_string(h.nRanks) + " ranks, but this run has " +
                                     std::to_string(nRanks) + ": restart with the same number of ranks and decomposition");
        if (h.rank != rank) throw std::runtime_error(file + " belongs to rank " + std::to_string(h.rank));
        h.time = get<double>(f, file);
        h.dt = get<double>(f, file);
        h.step = get<int64_t>(f, file);
        h.outputCount = get<int64_t>(f, file);
        state = checkpointState{};
        const uint64_t nr = get<uint64_t>(f, file);
        for (uint64_t i = 0; i < nr; i++) { std::string k = getString(f, file); state.reals[k] = get<double>(f, file); }
        const uint64_t ni = get<uint64_t>(f, file);
        for (uint64_t i = 0; i < ni; i++) { std::string k = getString(f, file); state.ints[k] = get<int64_t>(f, file); }

        const auto names = varReg.getVariableNames();
        std::map<std::string, bool> restored;
        const uint64_t nv = get<uint64_t>(f, file);
        for (uint64_t i = 0; i < nv; i++) {
            const std::string name = getString(f, file);
            const uint64_t bytes = get<uint64_t>(f, file);
            if (!std::binary_search(names.begin(), names.end(), name))
                throw std::runtime_error("restart file has variable '" + name + "', which this executable does not register: wrong executable?");
            auto &v = varReg.getVariable(name);
            if (!v.data() || v.getAllocatedBytes() != bytes)
                throw std::runtime_error("variable '" + name + "' has " + std::to_string(v.getAllocatedBytes()) + " bytes here but " +
                                         std::to_string(bytes) + " in the restart file: different grid, decomposition or executable");
            f.read(static_cast<char *>(v.data()), static_cast<std::streamsize>(bytes));
            if (!f) throw std::runtime_error("restart file " + file + " is truncated (in variable '" + name + "')");
            restored[name] = true;
        }
        for (const auto &n : names) {
            const auto &v = varReg.getVariable(n);
            if (v.data() && v.getAllocatedBytes() > 0 && !restored.count(n))
                throw std::runtime_error("variable '" + n + "' is not in the restart file: wrong executable?");
        }
        if (dtSim > 0.0) {   // next checkpoint time after the restored time, as the original run had it
            nextSim = dtSim;
            while (nextSim <= h.time) nextSim += dtSim;
        }
    }

    void checkpointManager::markComplete(int rank)
    {
        if (rank != 0 || !enabled()) return;
        const std::string info = dir + "/checkpoint_info.txt";
        if (!std::filesystem::exists(info)) return;
        std::ifstream in(info);
        std::ostringstream out;
        std::string line;
        while (std::getline(in, line))
            out << (line.rfind("status ", 0) == 0 ? std::string("status complete") : line) << "\n";
        in.close();
        std::ofstream(info, std::ios::trunc) << out.str();
    }
} // namespace SAMS
